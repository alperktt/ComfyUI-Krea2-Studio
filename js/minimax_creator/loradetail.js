// The LoRA detail sheet: double-click on a manager card.
//
// Two deliberately different shapes, because the two kinds of file know
// different things about themselves. A LoRA with a CiviMeta sidecar opens
// showcase-first — big media, a filmstrip, and under the selected image the
// generation recipe images.json recorded for it, which is the closest thing to
// an answer to "how do I prompt this". A LoRA without one opens as a spec
// sheet read from the safetensors header itself: trainer, rank, precision,
// training run, and (for kohya-trained files) the dataset's tag frequency,
// which is the closest thing such a file has to trigger words.

import { el, ICONS, svg, mountOverlay } from "./dom.js";
import { loraDetail, loraShowcaseUrl } from "./api.js";

/** Open the sheet for one listing row. Resolves when it closes. */
export function openLoraDetail(row) {
  return new Promise((resolve) => {
    new LoraDetailSheet(row, resolve).mount();
  });
}

// ---- formatting -------------------------------------------------------------

function fmtBytes(size) {
  if (!Number.isFinite(size) || size <= 0) return null;
  if (size >= 1024 ** 3) return `${(size / 1024 ** 3).toFixed(2)} GB`;
  return `${Math.round(size / 1024 ** 2)} MB`;
}

function fmtCount(value) {
  if (!Number.isFinite(value)) return null;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return String(value);
}

function fmtDate(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleDateString();
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// ---- description HTML -------------------------------------------------------

// The sidecar's description is HTML straight from Civitai. Only this structural
// subset survives; anything else is unwrapped to its text. No images and no
// iframes: a detail sheet must not phone remote hosts on open.
const SAFE_TAGS = new Set([
  "P", "BR", "UL", "OL", "LI", "STRONG", "B", "EM", "I", "U", "S",
  "CODE", "PRE", "BLOCKQUOTE",
]);

function sanitize(html) {
  const doc = new DOMParser().parseFromString(String(html), "text/html");
  const frag = document.createDocumentFragment();
  const walk = (source, out) => {
    for (const node of source.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) {
        out.appendChild(document.createTextNode(node.nodeValue));
        continue;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) continue;
      const tag = node.tagName;
      if (tag === "SCRIPT" || tag === "STYLE" || tag === "IFRAME") continue;
      if (/^H[1-6]$/.test(tag)) {
        // Civitai authors headline freely; in a 340px column they all become
        // one bold paragraph size or the type scale belongs to the description.
        const heading = el("p", { class: "mmc-sheet-h" });
        walk(node, heading);
        out.appendChild(heading);
        continue;
      }
      if (tag === "A") {
        const href = node.getAttribute("href") || "";
        if (/^https?:\/\//i.test(href)) {
          const anchor = el("a", { href, target: "_blank", rel: "noopener noreferrer" });
          walk(node, anchor);
          out.appendChild(anchor);
          continue;
        }
        walk(node, out);
        continue;
      }
      if (!SAFE_TAGS.has(tag)) {
        walk(node, out);   // unknown structure keeps its words
        continue;
      }
      const copy = document.createElement(tag.toLowerCase());
      walk(node, copy);
      out.appendChild(copy);
    }
  };
  walk(doc.body, frag);
  return frag;
}

// ---- the safetensors header, interpreted ------------------------------------

/**
 * The spec rows the header supports, as [label, value] pairs. Two trainer
 * dialects are understood: ai-toolkit writes JSON-in-string values (`software`,
 * `training_info`), kohya's sd-scripts writes flat `ss_*` keys.
 */
function headerFacts(header, size) {
  const md = header?.metadata || {};
  const software = parseJson(md.software);
  const training = parseJson(md.training_info);
  const rows = [];
  const push = (label, value) => { if (value) rows.push([label, String(value)]); };

  if (software?.name) push("Trainer", `${software.name} ${software.version || ""}`.trim());
  else if (Object.keys(md).some((key) => key.startsWith("ss_"))) push("Trainer", "kohya sd-scripts");
  push("Base model", md.ss_base_model_version || md.ss_sd_model_name);
  const rank = header?.ranks?.length ? header.ranks.join(" / ") : md.ss_network_dim;
  const alpha = md.ss_network_alpha;
  push("Rank", rank && (alpha ? `${rank} · α ${alpha}` : String(rank)));
  const dtypes = Object.keys(header?.dtypes || {});
  push("Precision", dtypes.join(" + "));
  push("Tensors", header?.tensors);
  const steps = training?.step ?? md.ss_steps ?? md.ss_max_train_steps;
  const epoch = training?.epoch ?? md.ss_epoch ?? md.ss_num_epochs;
  push("Trained", [steps && `${fmtCount(Number(steps)) ?? steps} steps`, epoch && `epoch ${epoch}`]
    .filter(Boolean).join(" · "));
  push("Resolution", md.ss_resolution);
  push("Dataset", md.ss_num_train_images && `${md.ss_num_train_images} images`);
  push("Learning rate", md.ss_learning_rate);
  const hash = md.sshs_model_hash || md.ss_new_sd_model_hash;
  push("Hash", hash && String(hash).slice(0, 12));
  push("File size", fmtBytes(size));
  return rows;
}

/**
 * kohya's ss_tag_frequency: {dataset: {tag: count}} — the words the training
 * captions actually used, which for a sidecar-less LoRA is the best available
 * stand-in for trigger words. Aggregated across datasets, most frequent first.
 */
function tagFrequency(metadata) {
  const sets = parseJson(metadata?.ss_tag_frequency);
  if (!sets || typeof sets !== "object") return [];
  const totals = new Map();
  for (const tags of Object.values(sets)) {
    if (!tags || typeof tags !== "object") continue;
    for (const [tag, count] of Object.entries(tags)) {
      if (Number.isFinite(count)) totals.set(tag, (totals.get(tag) || 0) + count);
    }
  }
  return [...totals.entries()].sort((a, b) => b[1] - a[1]);
}

// ---- the sheet --------------------------------------------------------------

class LoraDetailSheet {
  constructor(row, resolve) {
    this.row = row;
    this.resolve = resolve;
    this.current = 0;   // which showcase item the stage shows
  }

  mount() {
    this.sheet = el("div", { class: "mmc-sheet" }, [
      el("div", { class: "mmc-sheet-info" }, [el("div", { class: "mmc-empty", text: "Loading…" })]),
    ]);
    this.overlay = el("div", {
      class: "mmc-overlay",
      onpointerdown: (event) => { if (event.target === this.overlay) this.close(); },
    }, [this.sheet]);
    this.unmount = mountOverlay(this.overlay, () => this.close());
    this.load();
  }

  async load() {
    try {
      this.detail = await loraDetail(this.row.name);
    } catch (error) {
      this.detail = { error: error.message };
    }
    if (!this.overlay.isConnected) return;
    this.render();
  }

  close() {
    this.unmount();
    this.resolve();
  }

  render() {
    const detail = this.detail;
    if (detail.error) {
      this.sheet.replaceChildren(el("div", { class: "mmc-sheet-info" }, [
        this.closeButton(),
        el("div", { class: "mmc-empty", text: `Could not read this LoRA: ${detail.error}` }),
      ]));
      return;
    }
    const showcase = detail.showcase || [];
    this.sheet.classList.toggle("bare", !showcase.length);
    this.sheet.replaceChildren(
      ...(showcase.length ? [this.stage(showcase)] : []),
      detail.civitai ? this.civitaiInfo() : this.headerInfo(),
    );
  }

  closeButton() {
    return el("button", { class: "mmc-close mmc-sheet-close", text: "✕", onclick: () => this.close() });
  }

  // ---- left pane: showcase -------------------------------------------------

  stage(showcase) {
    this.stageMedia = el("div", { class: "mmc-sheet-media" });
    this.recipeBox = el("div", { class: "mmc-sheet-recipe" });
    const strip = showcase.length > 1
      ? el("div", { class: "mmc-sheet-strip" }, showcase.map((item, index) => this.stripCell(item, index)))
      : null;
    this.stripCells = strip ? [...strip.children] : [];
    this.showItem(this.current);
    return el("div", { class: "mmc-sheet-stage" }, [this.stageMedia, strip, this.recipeBox]);
  }

  stripCell(item, index) {
    const cell = el("button", {
      class: "mmc-sheet-thumb",
      "aria-selected": index === this.current,
      onclick: () => this.showItem(index),
    });
    if (item.kind === "video" && !item.thumb) {
      // A video showcase has no generated thumbnail; the media-fragment trick
      // the manager's cards use paints its first usable frame instead.
      const video = el("video", { muted: true, playsInline: true, preload: "metadata" });
      video.src = `${loraShowcaseUrl(this.row.name, item.index)}#t=0.12`;
      cell.appendChild(video);
    } else {
      cell.appendChild(el("img", {
        src: loraShowcaseUrl(this.row.name, item.index, { thumb: true }),
        loading: "lazy", alt: "",
      }));
    }
    return cell;
  }

  showItem(index) {
    const item = (this.detail.showcase || [])[index];
    if (!item) return;
    this.current = index;
    this.stripCells.forEach((cell, at) => cell.setAttribute("aria-selected", String(at === index)));
    const source = loraShowcaseUrl(this.row.name, item.index);
    this.stageMedia.replaceChildren(item.kind === "video"
      ? el("video", { src: source, controls: true, loop: true, muted: true, autoplay: true, playsInline: true })
      : el("img", { src: source, alt: "" }));
    this.renderRecipe(item.meta);
  }

  /** The generation settings recorded for the shown image — the sheet's whole
   *  reason to exist. Absent metadata leaves the strip empty rather than
   *  padding it with dashes. */
  renderRecipe(meta) {
    if (!meta) {
      this.recipeBox.replaceChildren();
      this.recipeBox.classList.remove("on");
      return;
    }
    this.recipeBox.classList.add("on");
    const facts = [
      meta.seed != null && ["seed", String(meta.seed)],
      meta.steps != null && ["steps", String(meta.steps)],
      meta.cfgScale != null && ["cfg", String(meta.cfgScale)],
      meta.sampler && ["sampler", String(meta.sampler)],
      meta.scheduler && ["sched", String(meta.scheduler)],
    ].filter(Boolean);
    const children = [];
    if (facts.length) {
      children.push(el("div", { class: "mmc-sheet-recipe-facts" }, facts.map(([label, value]) =>
        el("span", {}, [
          el("span", { class: "mmc-sheet-recipe-k", text: label }),
          " ",
          el("span", { class: "mmc-sheet-recipe-v", text: value }),
        ]))));
    }
    if (meta.prompt) {
      children.push(el("div", { class: "mmc-sheet-prompt" }, [
        el("div", { class: "mmc-sheet-prompt-text", text: meta.prompt, title: meta.prompt }),
        this.copyButton(meta.prompt),
      ]));
    }
    if (meta.negativePrompt) {
      children.push(el("div", {
        class: "mmc-sheet-negative",
        text: `negative: ${meta.negativePrompt}`,
        title: meta.negativePrompt,
      }));
    }
    this.recipeBox.replaceChildren(...children);
  }

  copyButton(text) {
    const button = el("button", {
      class: "mmc-sheet-copy",
      text: "Copy prompt",
      onclick: async () => {
        try {
          await navigator.clipboard.writeText(text);
          button.textContent = "Copied";
        } catch {
          button.textContent = "Copy failed";
        }
        setTimeout(() => { button.textContent = "Copy prompt"; }, 1500);
      },
    });
    return button;
  }

  // ---- right pane: the sidecar's story -------------------------------------

  section(label, children) {
    const body = [].concat(children).filter(Boolean);
    if (!body.length) return null;
    return el("div", { class: "mmc-sheet-section" }, [
      el("div", { class: "mmc-sheet-label", text: label }),
      ...body,
    ]);
  }

  chips(words, className = "mmc-sheet-chip") {
    if (!words?.length) return null;
    return el("div", { class: "mmc-sheet-chips" },
      words.map((word) => el("span", { class: className, text: word })));
  }

  civitaiInfo() {
    const meta = this.detail.civitai;
    const stats = meta.stats || {};
    const eyebrow = [meta.type, meta.baseModel].filter(Boolean).join(" · ");

    const statRow = el("div", { class: "mmc-sheet-stats" }, [
      Number.isFinite(stats.downloads) && this.stat(fmtCount(stats.downloads), "downloads"),
      stats.rating > 0 && this.stat(`★ ${stats.rating.toFixed(1)}`, "rating"),
      stats.favorites > 0 && this.stat(fmtCount(stats.favorites), "favorites"),
      stats.commentCount > 0 && this.stat(fmtCount(stats.commentCount), "comments"),
    ].filter(Boolean));

    const description = [meta.description, meta.versionDescription]
      .filter((html) => html && String(html).trim());
    const about = description.length
      ? el("div", { class: "mmc-sheet-desc" }, description.map((html) => {
        const block = el("div");
        block.appendChild(sanitize(html));
        return block;
      }))
      : null;

    const link = meta.modelId ? el("a", {
      class: "mmc-sheet-link",
      href: `https://civitai.com/models/${meta.modelId}${meta.versionId ? `?modelVersionId=${meta.versionId}` : ""}`,
      target: "_blank", rel: "noopener noreferrer",
      text: "Open on Civitai ↗",
    }) : null;

    return el("div", { class: "mmc-sheet-info" }, [
      this.closeButton(),
      el("div", { class: "mmc-sheet-eyebrow" }, [
        eyebrow,
        meta.nsfw ? el("span", { class: "mmc-sheet-nsfw", text: "NSFW" }) : null,
      ]),
      el("div", { class: "mmc-sheet-title", text: meta.name || this.row.base }),
      el("div", {
        class: "mmc-sheet-byline",
        text: [
          meta.versionName,
          meta.creator?.username && `by ${meta.creator.username}`,
          meta.fetchedAt && `fetched ${fmtDate(meta.fetchedAt)}`,
        ].filter(Boolean).join(" · "),
      }),
      statRow.childElementCount ? statRow : null,
      this.section("Trigger words", this.chips(meta.trainedWords, "mmc-sheet-chip accent")),
      this.section("About", about),
      this.section("Versions", this.versions(meta)),
      this.section("License", this.license(meta.license)),
      this.section("Tags", meta.tags?.length
        ? el("div", { class: "mmc-sheet-tags", text: meta.tags.join(" · ") })
        : null),
      this.section("File", this.fileFacts(meta)),
      link,
    ]);
  }

  stat(value, label) {
    return el("span", { class: "mmc-sheet-stat" }, [
      el("span", { class: "mmc-sheet-stat-v", text: value }),
      el("span", { class: "mmc-sheet-stat-k", text: label }),
    ]);
  }

  /** Sibling versions from the sidecar, the installed one marked. */
  versions(meta) {
    if (!meta.versions?.length) return null;
    return el("div", { class: "mmc-sheet-versions" }, meta.versions.map((version) =>
      el("div", { class: "mmc-sheet-version", "aria-current": version.id === meta.versionId }, [
        el("span", { class: "mmc-sheet-version-name", text: version.name || String(version.id) }),
        el("span", {
          class: "mmc-sheet-version-sub",
          text: [version.baseModel, fmtDate(version.createdAt)].filter(Boolean).join(" · "),
        }),
        version.id === meta.versionId ? el("span", { class: "mmc-sheet-installed", text: "installed" }) : null,
      ])));
  }

  license(license) {
    if (!license) return null;
    // allowCommercialUse arrives as a set literal in a string: "{Image,Rent,Sell}".
    const commercial = String(license.allowCommercialUse ?? "").replace(/["{}]/g, "").split(",")
      .map((part) => part.trim()).filter((part) => part && part.toLowerCase() !== "none");
    const lines = [
      commercial.length ? `Commercial use: ${commercial.join(", ")}` : "No commercial use",
      license.allowNoCredit ? "Credit not required" : "Credit required",
      license.allowDerivatives ? "Derivatives allowed" : "No derivatives",
    ];
    return el("div", { class: "mmc-sheet-license", text: lines.join(" · ") });
  }

  fileFacts(meta) {
    const header = this.detail.header || {};
    const facts = headerFacts(header, this.detail.size)
      .filter(([label]) => ["Rank", "Precision", "Tensors", "File size"].includes(label));
    return el("div", { class: "mmc-sheet-file" }, [
      el("div", { class: "mmc-sheet-path", text: this.row.name, title: this.row.name }),
      facts.length ? el("div", {
        class: "mmc-sheet-file-facts",
        text: facts.map(([label, value]) => `${label.toLowerCase()} ${value}`).join(" · "),
      }) : null,
      meta.hash ? el("div", { class: "mmc-sheet-hash", text: `sha256 ${String(meta.hash).slice(0, 12)}…`, title: meta.hash }) : null,
    ]);
  }

  // ---- the bare sheet: what the file says about itself ---------------------

  headerInfo() {
    const header = this.detail.header || {};
    const metadata = header.metadata || {};
    const facts = headerFacts(header, this.detail.size);
    const tags = tagFrequency(metadata);

    const spec = facts.length ? el("div", { class: "mmc-sheet-spec" }, facts.map(([label, value]) =>
      el("div", { class: "mmc-sheet-spec-row" }, [
        el("span", { class: "mmc-sheet-spec-k", text: label }),
        el("span", { class: "mmc-sheet-spec-v", text: value }),
      ]))) : null;

    const tagChips = tags.length ? el("div", { class: "mmc-sheet-chips" },
      tags.slice(0, 24).map(([tag, count]) => el("span", { class: "mmc-sheet-chip" }, [
        tag,
        el("span", { class: "mmc-sheet-chip-n", text: String(count) }),
      ]))) : null;

    const keys = Object.keys(metadata);
    const raw = keys.length ? el("details", { class: "mmc-sheet-raw" }, [
      el("summary", { text: `All header fields (${keys.length})` }),
      el("div", { class: "mmc-sheet-raw-rows" }, keys.sort().map((key) => {
        const value = String(metadata[key]);
        return el("div", { class: "mmc-sheet-raw-row" }, [
          el("span", { class: "mmc-sheet-raw-k", text: key }),
          el("span", {
            class: "mmc-sheet-raw-v",
            text: value.length > 200 ? `${value.slice(0, 200)}…` : value,
            title: value.length > 200 ? value : null,
          }),
        ]);
      })),
    ]) : null;

    return el("div", { class: "mmc-sheet-info" }, [
      this.closeButton(),
      el("div", { class: "mmc-sheet-eyebrow" }, [
        el("span", { class: "mmc-sheet-mono-mark" }, [svg(ICONS.effect, 13)]),
        header.error ? "safetensors" : "safetensors header",
      ]),
      el("div", { class: "mmc-sheet-title", text: metadata.name || metadata.ss_output_name || this.row.base }),
      el("div", {
        class: "mmc-sheet-byline",
        text: "No CiviMeta sidecar — everything below was read from the file itself.",
      }),
      header.error
        ? el("div", { class: "mmc-sheet-license", text: `The header could not be read: ${header.error}` })
        : null,
      this.section("Specification", spec),
      this.section("Dataset tags", tagChips && [
        el("div", {
          class: "mmc-sheet-hint",
          text: "The most frequent words in the training captions — the closest thing this file has to trigger words.",
        }),
        tagChips,
      ]),
      raw,
      this.section("File", el("div", { class: "mmc-sheet-file" }, [
        el("div", { class: "mmc-sheet-path", text: this.row.name, title: this.row.name }),
      ])),
    ]);
  }
}
