// The PreStage body: stills for the pipeline, in the Creator's own vocabulary.
//
// Same skeleton as CreatorEditor — rail, chips, panel, pills, sampler row —
// because it is driven the same way; what changed is what it generates. The
// prompt is a plain textarea rather than the @-mention PromptBox: an image
// prompt references nothing by handle (the Qwen-edit encoder labels the style
// references itself), so the mention machinery would be an empty menu.
//
// The model pill is the one control the video nodes do not have. Krea 2 and
// Ideogram 4 want different sampler rows — RAW runs 52 steps at cfg 3.5 where
// Ideogram runs its preset's steps at cfg 7 on its own schedule — so switching
// the arch rewrites the row through the same widgetIO the turbo pill uses, and
// the turbo pill itself only exists on Krea (Turbo *is* a checkpoint there;
// Ideogram's speed axis is its preset table, which gets a pill of its own).

import { el, icon, ICONS, svg, dismissable, placeNear } from "./dom.js";
import { openPicker } from "./picker.js";
import { openLoras } from "./loras.js";
import { openFrameGrab } from "./framegrab.js";
import { openChoicePopover, openOutputPopover, stepperPill, aspectGlyph, edgeSlider, PILL_GLYPH } from "./pills.js";
import { IMAGE_PREFIX, folderOf } from "./outputs.js";
import { samplingBar } from "./sampling.js";
import { Stage } from "./stage.js";
import { loadCatalog, catalogByFolder } from "./models.js";
import { viewUrl } from "./api.js";
import * as S from "./state.js";

const QUALITY_TITLE = {
  quality: "48 steps on the tight schedule — the hosted service's 'Quality' tier.",
  default: "20 steps — the hosted service's default tier.",
  turbo: "12 steps on the shifted schedule — the hosted service's 'Turbo' tier.",
};

const TURBO_TITLE = {
  draft: "4 steps — the fast look. Softer detail.",
  medium: "6 steps — quick and usable.",
  good: "8 steps — what the Turbo checkpoint was distilled for.",
};

export class PreStageEditor {
  /**
   * @param {object} options
   * @param {object} options.state    a parsePreStage state, mutated in place
   * @param {() => void} options.onCommit
   * @param {object} options.samplingWidgets  the node's hidden sampler widgets
   * @param {() => void} options.onWidgetChange
   * @param {() => string|number} options.nodeId
   * @param {() => {label: string, attach: (role, filename) => string|null}|null}
   *   [options.peer]  resolved late — the Creator/Timeline this node was
   *   spawned beside, for the result card's hand-off chips. Returns null when
   *   the peer is gone; `attach` returns a refusal message or null on success.
   */
  constructor({ state, onCommit, samplingWidgets, onWidgetChange, nodeId, peer = null }) {
    this.state = state;
    this.onCommit = onCommit;
    this.samplingWidgets = samplingWidgets;
    this.onWidgetChange = onWidgetChange;
    this.nodeId = nodeId;
    this.peer = peer;
    this.sizes = new Map();   // filename -> {width,height}, for the adaptive canvas readout

    this.promptBox = el("textarea", {
      class: "mmc-prestage-prompt",
      placeholder: "Describe the image. Both models were trained on long, detailed natural-language prompts.",
      oninput: () => {
        this.state.prompt = this.promptBox.value;
        this.onCommit?.();
      },
      onpointerdown: (event) => event.stopPropagation(),
      onkeyup: (event) => event.stopPropagation(),
    });

    this.railHost = el("div");
    this.assetsHost = el("div");
    this.loraHost = el("div");
    this.pillsHost = el("div");
    this.noticeHost = el("div");
    this.samplingHost = el("div");

    this.stage = new Stage({
      nodeId: this.nodeId,
      resultChips: (saved) => this.renderResultChips(saved),
    });

    this.root = el("div", { class: "mmc-root mmc-prestage" }, [
      this.railHost,
      this.assetsHost,
      this.loraHost,
      el("div", { class: "mmc-panel" }, [this.promptBox, this.pillsHost]),
      this.noticeHost,
      this.samplingHost,
    ]);

    loadCatalog(() => this.adoptWeights());
    this.promptBox.value = this.state.prompt ?? "";
    this.render();
    this.probeInit();
  }

  destroy() {
    this.stage?.destroy();
  }

  adoptWeights() {
    if (S.guessPreStageModels(this.state.models, catalogByFolder())) this.commit();
    else this.render();
  }

  widgetIO() {
    return {
      value: (name, fallback) => this.samplingWidgets?.[name]?.value ?? fallback,
      set: (name, value) => {
        const widget = this.samplingWidgets?.[name];
        if (!widget) return;
        widget.value = value;
        widget.callback?.(value);
        this.onWidgetChange?.();
      },
    };
  }

  commit() {
    this.onCommit?.();
    this.render();
  }

  setState(state) {
    this.state = state;
    this.sizes.clear();
    this.promptBox.value = this.state.prompt ?? "";
    this.render();
    this.probeInit();
  }

  // ---- the model pill --------------------------------------------------------

  /** Switch architectures. The other side keeps its files (per-arch sub-blocks)
   *  and gets its own sampler row written in — the two run at numbers that have
   *  nothing to do with each other, so carrying the row across would be wrong
   *  on arrival. */
  setArch(arch) {
    if (arch === this.state.arch) return;
    const io = this.widgetIO();
    // Leaving Krea with turbo thrown would strand `saved`; the row is about to
    // be rewritten anyway, so the switch just resets.
    this.state.turbo.on = false;
    this.state.turbo.saved = null;
    this.state.arch = arch;
    if (arch === "krea2") {
      const row = S.PRESTAGE_KREA_RAW;
      io.set("steps", row.steps);
      io.set("cfg", row.cfg);
      io.set("sampler_name", row.sampler_name);
      io.set("scheduler", row.scheduler);
    } else {
      io.set("steps", S.PRESTAGE_IDEOGRAM_STEPS[this.state.quality]);
      io.set("cfg", S.PRESTAGE_IDEOGRAM_ROW.cfg);
      io.set("sampler_name", S.PRESTAGE_IDEOGRAM_ROW.sampler_name);
    }
    this.commit();
  }

  // ---- init image and style references --------------------------------------

  /** Pick the init image — the still this render restyles rather than starts
   *  from nothing. From the picker, or grabbed off a video's playhead. */
  async setInit(fromVideo = false) {
    let path = null;
    if (fromVideo) {
      const clip = await openPicker({
        kinds: ["video", "renders"], kind: "video", single: true,
        capacity: () => ({ used: 0, max: 1, filesLeft: 1 }),
      });
      if (!clip) return;
      const grabbed = await openFrameGrab({ path: clip[0].path });
      if (!grabbed) return;
      path = grabbed.path;
    } else {
      const chosen = await openPicker({
        kinds: ["image", "renders"], kind: "image", single: true,
        capacity: () => ({ used: 0, max: 1, filesLeft: 1 }),
      });
      if (!chosen) return;
      path = chosen[0].path;
    }
    this.state.init = { filename: path, denoise: this.state.init?.denoise ?? S.PRESTAGE_DEFAULT_DENOISE };
    this.commit();
    this.probeInit();
  }

  async addRefs(fromVideo = false) {
    if (this.state.arch === "ideogram4") {
      return this.flash("Ideogram 4.0 has no local reference conditioning — switch the model "
                      + "pill to Krea 2 to use style references.");
    }
    const room = S.PRESTAGE_MAX_REFS - this.state.refs.length;
    if (room <= 0) {
      return this.flash(`At most ${S.PRESTAGE_MAX_REFS} style references — the Qwen edit encoder `
                      + `the model reads them through has exactly three image slots.`);
    }
    if (fromVideo) {
      const clip = await openPicker({
        kinds: ["video", "renders"], kind: "video", single: true,
        capacity: () => ({ used: 0, max: 1, filesLeft: 1 }),
      });
      if (!clip) return;
      const grabbed = await openFrameGrab({ path: clip[0].path });
      if (!grabbed) return;
      this.state.refs.push({ handle: S.nextPreStageHandle(this.state), filename: grabbed.path });
      return this.commit();
    }
    const chosen = await openPicker({
      kinds: ["image", "renders"], kind: "image",
      capacity: () => ({ used: this.state.refs.length, max: S.PRESTAGE_MAX_REFS, filesLeft: room }),
    });
    if (!chosen) return;
    for (const asset of chosen.slice(0, room)) {
      this.state.refs.push({ handle: S.nextPreStageHandle(this.state), filename: asset.path });
    }
    this.commit();
  }

  async manageLoras() {
    await openLoras({ state: this.state, checkpointModes: false, onChange: () => this.commit() });
    this.commit();
  }

  probeInit() {
    const init = this.state.init;
    if (!init || this.sizes.has(init.filename)) return;
    const probe = new Image();
    probe.onload = () => {
      this.sizes.set(init.filename, { width: probe.naturalWidth, height: probe.naturalHeight });
      this.render();
    };
    probe.src = viewUrl(init.filename);
  }

  flash(message) {
    this.notice = message;
    this.render();
    clearTimeout(this.noticeTimer);
    this.noticeTimer = setTimeout(() => { this.notice = null; this.render(); }, 6000);
  }

  // ---- the hand-off ----------------------------------------------------------

  /** The chips on the finished still: one click writes it into the peer's blob
   *  as a start frame, end frame or reference. The annotated `[output]` path is
   *  the same currency the gallery attach uses — one store, no copy. */
  renderResultChips(saved) {
    const target = this.peer?.();
    if (!target) return [];
    const filename = `${saved.subfolder ? `${saved.subfolder}/` : ""}${saved.filename} [output]`;
    const chip = (role, label, title) => el("button", {
      class: "mmc-stage-chip mmc-stage-send",
      text: label,
      title: `${title} on ${target.label}.`,
      onpointerdown: (event) => event.stopPropagation(),
      onclick: () => {
        const refused = target.attach(role, filename);
        if (refused) this.flash(refused);
      },
    });
    return [
      chip("first_frame", "→ start", "Use this still as the start frame"),
      chip("last_frame", "→ end", "Use this still as the end frame"),
      chip("reference", "→ ref", "Attach this still as a reference"),
    ];
  }

  // ---- render ----------------------------------------------------------------

  render() {
    const state = this.state;
    this.railHost.replaceChildren(this.renderRail());
    const chips = [
      ...(state.init ? [this.renderInitChip()] : []),
      ...state.refs.map((ref) => this.renderRefChip(ref)),
    ];
    this.assetsHost.replaceChildren(...(chips.length ? [el("div", { class: "mmc-assets" }, chips)] : []));
    this.loraHost.replaceChildren(...(state.loras.length ? [this.renderLoras()] : []));
    this.pillsHost.replaceChildren(this.renderPills());
    this.noticeHost.replaceChildren(
      ...(this.notice ? [el("div", { class: "mmc-warn", text: this.notice })] : []));
    this.samplingHost.replaceChildren(samplingBar({
      widgets: this.samplingWidgets,
      ...this.widgetIO(),
      set: (name, value) => { this.widgetIO().set(name, value); this.render(); },
      perSegment: false,
      turbo: state.arch === "krea2" ? this.renderTurbo() : [],
      trailing: [this.renderWeightsPill()],
    }));
  }

  renderRail() {
    const tool = (label, iconName, title, onclick) => el("button", {
      class: "mmc-tool", title, onclick,
    }, [el("span", { class: "mmc-tool-icon" }, [icon(iconName)]), el("span", { text: label })]);

    return el("div", { class: "mmc-rail" }, [
      tool("Init image", "frameIn",
           "Start from an image instead of noise — img2img. The strength pill says how much of it survives.",
           () => this.setInit(false)),
      tool("Style refs", "image",
           this.state.arch === "ideogram4"
             ? "Ideogram 4.0 has no local reference conditioning — style references are a Krea 2 feature."
             : "Up to three images whose look this render should carry. Encoded through the Qwen edit "
             + "path Krea 2 was post-trained against; the krea2_style_reference LoRA strengthens it.",
           () => this.addRefs(false)),
      tool("From video", "video",
           "Pull a single frame off a video's playhead — as the init image, saved as a PNG in the input folder.",
           () => this.setInit(true)),
      tool("Add LoRA", "effect",
           "Manage the LoRAs patched onto the image model. Krea LoRAs train on RAW and apply on Turbo too.",
           () => this.manageLoras()),
    ]);
  }

  renderInitChip() {
    const init = this.state.init;
    return el("div", { class: "mmc-asset mmc-tag-0", title: init.filename }, [
      el("img", { class: "mmc-asset-thumb", src: viewUrl(init.filename, { preview: true }), alt: init.filename }),
      el("span", { class: "mmc-asset-handle", text: "init" }),
      el("button", {
        class: "mmc-ghost",
        style: { fontSize: "11px" },
        title: "How much of the render is new. 1.00 ignores the init entirely; low values keep its "
             + "composition and only restyle. Click to step down, right-click to step up.",
        text: init.denoise.toFixed(2),
        onclick: () => {
          init.denoise = Math.max(S.PRESTAGE_MIN_DENOISE, Math.round((init.denoise - 0.05) * 100) / 100);
          this.commit();
        },
        oncontextmenu: (event) => {
          event.preventDefault();
          init.denoise = Math.min(1, Math.round((init.denoise + 0.05) * 100) / 100);
          this.commit();
        },
      }),
      el("button", {
        class: "mmc-asset-x", text: "✕", title: "Remove the init image",
        onclick: () => { this.state.init = null; this.commit(); },
      }),
    ]);
  }

  renderRefChip(ref) {
    return el("div", {
      class: `mmc-asset mmc-tag-${S.tagIndex(ref.handle)}`,
      title: ref.filename,
    }, [
      el("img", { class: "mmc-asset-thumb", src: viewUrl(ref.filename, { preview: true }), alt: ref.filename }),
      el("span", { class: "mmc-asset-handle", text: `@${ref.handle}` }),
      el("span", { class: "mmc-asset-role", text: "style" }),
      el("button", {
        class: "mmc-asset-x", text: "✕", title: `Remove @${ref.handle}`,
        onclick: () => {
          this.state.refs = this.state.refs.filter((r) => r.handle !== ref.handle);
          this.commit();
        },
      }),
    ]);
  }

  renderLoras() {
    const chip = (entry) => el("div", { class: "mmc-asset", title: entry.name }, [
      el("span", { class: "mmc-asset-thumb" }, [svg(ICONS.effect, 15)]),
      el("span", { class: "mmc-asset-handle", text: entry.name.split("/").pop().replace(/\.[^.]+$/, "") }),
      el("button", {
        class: "mmc-ghost",
        style: { fontSize: "11px" },
        title: "Strength — edit on the LoRA card",
        text: Number(entry.strength ?? 1).toFixed(2),
        onclick: () => this.manageLoras(),
      }),
      el("button", {
        class: "mmc-asset-x", text: "✕", title: `Remove ${entry.name}`,
        onclick: () => { S.removeLora(this.state, entry.name); this.commit(); },
      }),
    ]);

    const parts = [el("div", { class: "mmc-assets" }, this.state.loras.map(chip))];
    const triggers = S.promptTriggers(this.state);
    if (triggers.length) {
      parts.push(el("div", {
        class: "mmc-note",
        title: "Prefixed to the prompt when this queues. Edit the list on the LoRA cards.",
      }, [
        el("span", { class: "mmc-note-key", text: "triggers" }),
        el("span", { text: triggers.join(", ") }),
      ]));
    }
    return el("div", { class: "mmc-lora-block" }, parts);
  }

  renderPills() {
    const state = this.state;
    const geometry = S.resolvedPreStage(state, state.init ? this.sizes.get(state.init.filename) : null);

    const archPill = el("button", {
      class: "mmc-pill mmc-prestage-arch",
      title: state.arch === "krea2"
        ? "Krea 2 — 12.9B open-weights DiT. RAW samples at cfg 3.5; the turbo pill swaps in the "
          + "8-step Turbo checkpoint. Click to switch to Ideogram 4."
        : "Ideogram 4.0 — 9.3B open-weights DiT with its own resolution-shifted schedule and a "
          + "second checkpoint for the unconditional branch. Click to switch to Krea 2.",
      onclick: (event) => openChoicePopover(event.currentTarget, {
        title: "Image model",
        options: S.PRESTAGE_ARCHES.map((arch) => S.PRESTAGE_ARCH_LABEL[arch]),
        value: S.PRESTAGE_ARCH_LABEL[state.arch],
        onPick: (picked) => this.setArch(
          S.PRESTAGE_ARCHES.find((arch) => S.PRESTAGE_ARCH_LABEL[arch] === picked) ?? "krea2"),
      }),
    }, [icon("model", 16), el("span", { text: S.PRESTAGE_ARCH_LABEL[state.arch] })]);

    const aspectPill = el("button", {
      class: "mmc-pill",
      disabled: geometry.fromImage || undefined,
      title: geometry.fromImage
        ? "The aspect follows the init image — the resolution pill still sets the scale."
        : "Aspect Ratio",
      onclick: (event) => this.openAspect(event.currentTarget),
    }, geometry.fromImage
      ? [aspectGlyph(geometry.ratio, PILL_GLYPH), el("span", { class: "mmc-pill-sub", text: "from image" })]
      : [aspectGlyph(geometry.ratio, PILL_GLYPH), el("span", { text: state.aspect })]);

    const resPill = el("button", {
      class: "mmc-pill",
      title: "Short edge. Both models are comfortable up to a 2048×2048 area.",
      onclick: (event) => this.openResolution(event.currentTarget),
    }, [
      icon("res", 16),
      el("span", { text: `${state.short_edge}p` }),
      el("span", { class: "mmc-pill-sub", text: `${geometry.width} × ${geometry.height}` }),
    ]);

    // Where the still lands. Its own default folder, apart from the video
    // node's, which is what sorts the gallery into stills and finished renders
    // without the picker having to know the difference between them.
    const outFolder = folderOf(state.output_prefix || IMAGE_PREFIX);
    const outputPill = el("button", {
      class: "mmc-pill",
      title: `Stills land in output/${outFolder ? `${outFolder}/` : ""} — click to change it.`,
      onclick: (event) => openOutputPopover(
        event.currentTarget, state, () => this.commit(),
        { fallback: IMAGE_PREFIX, extension: "png" }),
    }, [icon("folder", 16), el("span", { text: outFolder ? outFolder.split("/").pop() : "output" })]);

    const pills = [archPill, aspectPill, resPill, outputPill];

    if (state.arch === "ideogram4") {
      // Ideogram's speed axis. The preset owns the schedule shape as well as
      // the step count, which is why this is a preset pill and not a slider.
      pills.push(el("button", {
        class: "mmc-pill",
        title: QUALITY_TITLE[state.quality],
        onclick: (event) => openChoicePopover(event.currentTarget, {
          title: "Ideogram preset",
          options: [...S.PRESTAGE_IDEOGRAM_QUALITIES],
          value: state.quality,
          onPick: (picked) => {
            state.quality = picked;
            this.widgetIO().set("steps", S.PRESTAGE_IDEOGRAM_STEPS[picked]);
            this.commit();
          },
        }),
      }, [icon("steps", 16), el("span", { text: `${state.quality} · ${S.PRESTAGE_IDEOGRAM_STEPS[state.quality]}` })]));
    }

    if (state.init) {
      pills.push(stepperPill({
        value: state.init.denoise, min: S.PRESTAGE_MIN_DENOISE, max: 1, step: 0.05, width: "52px",
        title: "img2img strength — how much of the render is new. 1.00 ignores the init entirely; "
             + "low values keep its composition and only restyle.",
        format: (n) => `img ${n.toFixed(2)}`,
        onChange: (next) => { state.init.denoise = next; this.commit(); },
      }));
    }

    return el("div", { class: "mmc-pills" }, pills);
  }

  // ---- turbo (Krea 2) --------------------------------------------------------

  /** The turbo pill, under the H3 contract: save the row once per throw, put it
   *  back exactly on release, own no second stack. What it throws here is a
   *  *checkpoint* — Krea 2 Turbo is a distillation of RAW, not a LoRA — so the
   *  stack is untouched either way (Krea LoRAs train on RAW, apply on Turbo). */
  renderTurbo() {
    const state = this.state;
    const turbo = state.turbo;
    const io = this.widgetIO();
    const pills = [];

    pills.push(el("div", { class: `mmc-pill mmc-pill-group${turbo.on ? " accel-on" : ""}` }, [
      el("button", {
        class: "mmc-turbo-main",
        title: turbo.on
          ? `Turbo — running the Turbo checkpoint at ${io.value("steps", "?")} steps, cfg 1. `
            + "Switching off loads RAW again and puts the sampler row back."
          : "Turbo off — running RAW. On, the Turbo checkpoint (an 8-step distillation) is loaded "
            + "instead and the row drops to the picked quality at cfg 1.",
        onclick: () => {
          if (turbo.on) {
            const saved = turbo.saved ?? S.PRESTAGE_KREA_RAW;
            io.set("steps", saved.steps);
            io.set("cfg", saved.cfg);
            io.set("sampler_name", saved.sampler_name);
            io.set("scheduler", saved.scheduler);
            turbo.on = false;
            turbo.saved = null;
          } else {
            turbo.saved = {
              steps: Number(io.value("steps", S.PRESTAGE_KREA_RAW.steps)),
              cfg: Number(io.value("cfg", S.PRESTAGE_KREA_RAW.cfg)),
              sampler_name: String(io.value("sampler_name", S.PRESTAGE_KREA_RAW.sampler_name)),
              scheduler: String(io.value("scheduler", S.PRESTAGE_KREA_RAW.scheduler)),
            };
            turbo.on = true;
            io.set("steps", S.PRESTAGE_TURBO_STEPS[turbo.quality]);
            io.set("cfg", S.PRESTAGE_KREA_TURBO.cfg);
            io.set("sampler_name", S.PRESTAGE_KREA_TURBO.sampler_name);
            io.set("scheduler", S.PRESTAGE_KREA_TURBO.scheduler);
          }
          this.commit();
        },
      }, [icon("bolt", 16), el("span", { text: turbo.on ? "turbo" : "turbo off" })]),
    ]));

    if (turbo.on) {
      const steps = Number(io.value("steps", 0));
      pills.push(el("div", { class: "mmc-pill mmc-turbo-seg" }, S.PRESTAGE_TURBO_QUALITIES.map((quality) =>
        el("button", {
          class: "mmc-turbo-opt",
          "aria-pressed": steps === S.PRESTAGE_TURBO_STEPS[quality],
          title: TURBO_TITLE[quality],
          onclick: () => {
            turbo.quality = quality;
            io.set("steps", S.PRESTAGE_TURBO_STEPS[quality]);
            this.commit();
          },
        }, [
          el("span", { text: quality === "medium" ? "med" : quality }),
          el("span", { class: "mmc-pill-sub", text: String(S.PRESTAGE_TURBO_STEPS[quality]) }),
        ]))));
    }

    return pills;
  }

  // ---- weights ---------------------------------------------------------------

  renderWeightsPill() {
    const missing = S.missingPreStageModels(this.state);
    const label = missing.length
      ? (missing.length === 1
          ? `no ${S.PRESTAGE_FIELD_LABEL[missing[0]].toLowerCase()}`
          : `${missing.length} weights missing`)
      : this.state.models.dtype === "default"
        ? "weights" : `weights · ${this.state.models.dtype.replace("fp8_", "fp8 ")}`;
    return el("button", {
      class: `mmc-pill mmc-weights${missing.length ? " missing" : ""}`,
      title: missing.length
        ? `Not picked yet: ${missing.map((f) => S.PRESTAGE_FIELD_LABEL[f]).join(", ")}. `
          + "The render is refused without them."
        : `Which files ${S.PRESTAGE_ARCH_LABEL[this.state.arch]} loads.`,
      onclick: (event) => this.openWeights(event.currentTarget),
    }, [icon("weights", 16), el("span", { text: label })]);
  }

  openWeights(anchor) {
    const NONE = "— none —";
    const state = this.state;
    const pop = el("div", { class: "mmc-pop mmc-weights-pop" });
    const body = el("div");

    const render = () => {
      const byFolder = catalogByFolder();
      const lists = {
        model: byFolder.diffusion_models ?? [], turbo_model: byFolder.diffusion_models ?? [],
        uncond_model: byFolder.diffusion_models ?? [],
        clip: byFolder.text_encoders ?? [], vae: byFolder.vae ?? [],
      };
      const side = state.models[state.arch];
      const missing = new Set(S.missingPreStageModels(state));

      const rows = S.PRESTAGE_FIELDS[state.arch].map((field) => el("div", {
        class: `mmc-weight-row${missing.has(field) ? " missing" : ""}`,
      }, [
        el("span", { class: "mmc-weight-name", text: S.PRESTAGE_FIELD_LABEL[field] }),
        el("button", {
          class: `mmc-weight-file${side[field] ? "" : " empty"}`,
          title: S.PRESTAGE_FIELD_HINT[state.arch][field],
          text: side[field] || "not set",
          onclick: (event) => openChoicePopover(event.currentTarget, {
            title: S.PRESTAGE_FIELD_LABEL[field],
            options: [NONE, ...lists[field]],
            value: side[field] || NONE,
            onPick: (picked) => {
              side[field] = picked === NONE ? "" : picked;
              this.commit();
              render();
            },
          }),
        }),
      ]));

      rows.push(el("div", { class: "mmc-weight-row" }, [
        el("span", { class: "mmc-weight-name", text: "Precision" }),
        el("button", {
          class: "mmc-weight-file",
          title: "How the checkpoints are loaded. fp8 halves the weights in VRAM at some cost "
               + "in fidelity; 'default' loads them as they were saved.",
          text: state.models.dtype,
          onclick: (event) => openChoicePopover(event.currentTarget, {
            title: "Precision",
            options: S.MODEL_DTYPES,
            value: state.models.dtype,
            onPick: (picked) => { state.models.dtype = picked; this.commit(); render(); },
          }),
        }),
      ]));

      body.replaceChildren(...rows);
    };

    pop.append(el("div", { class: "mmc-pop-title", text: `Weights — ${S.PRESTAGE_ARCH_LABEL[state.arch]}` }), body);
    render();
    document.body.appendChild(pop);
    placeNear(pop, anchor);
    dismissable(pop);
    loadCatalog(() => pop.isConnected && render());
  }

  // ---- popovers --------------------------------------------------------------

  openAspect(anchor) {
    const pop = el("div", { class: "mmc-pop" }, [el("div", { class: "mmc-pop-title", text: "Aspect Ratio" })]);
    for (const [label, ratio] of S.PRESTAGE_ASPECTS) {
      pop.appendChild(el("button", {
        class: "mmc-opt",
        "aria-checked": this.state.aspect === label,
        onclick: () => { this.state.aspect = label; close(); this.commit(); },
      }, [
        el("span", { class: "mmc-opt-label" }, [aspectGlyph(ratio), el("span", { text: label })]),
        el("span", { class: "mmc-radio" }),
      ]));
    }
    document.body.appendChild(pop);
    placeNear(pop, anchor);
    const close = dismissable(pop);
  }

  openResolution(anchor) {
    const body = edgeSlider({
      min: S.PRESTAGE_MIN_EDGE, max: S.PRESTAGE_MAX_EDGE, step: S.PRESTAGE_CANVAS_MULTIPLE,
      value: this.state.short_edge,
      mark: S.PRESTAGE_DEFAULT_EDGE, markLabel: "default",
      apply: (edge) => { this.state.short_edge = edge; },
      describe: () => {
        const geometry = S.resolvedPreStage(this.state,
          this.state.init ? this.sizes.get(this.state.init.filename) : null);
        return {
          size: `${geometry.width} × ${geometry.height}`,
          note: this.state.short_edge >= S.PRESTAGE_MAX_EDGE
            ? "The models' 2048 ceiling — wide ratios trade the short edge down to hold the area."
            : `${this.state.short_edge < S.PRESTAGE_DEFAULT_EDGE ? "Faster, softer." : "Sharper, slower."} `
              + `${S.PRESTAGE_DEFAULT_EDGE} is the comfortable default for both models.`,
        };
      },
      commit: () => this.commit(),
    });
    const pop = el("div", { class: "mmc-pop mmc-slider" }, [body]);
    document.body.appendChild(pop);
    placeNear(pop, anchor);
    dismissable(pop);
  }
}
