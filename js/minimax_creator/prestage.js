// The PreStage node: stills for the pipeline, made on the left.
//
// Two classes. `PreStageEditor` is the body for the two *image* architectures,
// Krea 2 and Ideogram 4 — same skeleton as CreatorEditor (rail, chips, panel,
// pills, sampler row) because it is driven the same way, with a plain textarea
// for a prompt because an image prompt references nothing by handle: the
// Qwen-edit encoder labels the style references itself, so the mention
// machinery would be an empty menu.
//
// `PreStageBody` below owns the node, and on the third architecture it mounts
// `CreatorEditor` instead — MiniMax H3's still is a video generation with one
// latent frame decoded, so the body that drives a shot drives it too, on a
// request in the same shape. Its docstring says what that buys.
//
// The model pill is the one control the video nodes do not have, and it belongs
// to the body rather than to either editor: it is the control that swaps them.
// Krea 2 and Ideogram 4 want different sampler rows — RAW runs 52 steps at cfg
// 3.5 where Ideogram runs its preset's steps at cfg 7 on its own schedule — so
// switching the arch rewrites the row, and the turbo pill exists only on Krea
// (Turbo *is* a checkpoint there; Ideogram's speed axis is its preset table).

import { el, icon, ICONS, svg, dismissable, placeNear } from "./dom.js";
import { openPicker } from "./picker.js";
import { openLoras } from "./loras.js";
import { openFrameGrab } from "./framegrab.js";
import { openChoicePopover, stepperPill, aspectGlyph, edgeSlider, PILL_GLYPH } from "./pills.js";
import { CreatorEditor } from "./editor.js";
import { samplingBar } from "./sampling.js";
import { loadLoraNames, loraNames } from "./turbo.js";
import { Stage } from "./stage.js";
import { loadCatalog, catalogByFolder } from "./models.js";
import { viewUrl, listMoodboards } from "./api.js";
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
   */
  constructor({ state, onCommit, samplingWidgets, onWidgetChange, nodeId,
                stage = null, archPill = null }) {
    this.state = state;
    this.onCommit = onCommit;
    this.samplingWidgets = samplingWidgets;
    this.onWidgetChange = onWidgetChange;
    this.nodeId = nodeId;
    // Both supplied by `PreStageBody`, which outlives this editor: it rebuilds
    // the body when the architecture changes, and the stage was floated beside
    // the node once. The arch pill is the control that does the rebuilding, so
    // it cannot belong to the thing being rebuilt.
    this.stage = stage;
    this.archPill = archPill;
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
    // The stage is the body's — see the constructor.
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

  /** Measure the images the canvas adapts to, so the pills can say what will
   *  actually be generated before anything is queued.
   *
   *  Both the init image and the edit source, because both make the canvas
   *  follow them — see `compile_image.compile_prestage`. */
  probeInit() {
    for (const filename of [this.state.init?.filename, this.state.edit?.source?.filename]) {
      if (!filename || this.sizes.has(filename)) continue;
      const probe = new Image();
      probe.onload = () => {
        this.sizes.set(filename, { width: probe.naturalWidth, height: probe.naturalHeight });
        this.render();
      };
      probe.src = viewUrl(filename);
    }
  }

  flash(message) {
    this.notice = message;
    this.render();
    clearTimeout(this.noticeTimer);
    this.noticeTimer = setTimeout(() => { this.notice = null; this.render(); }, 6000);
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
      ...(this.notice ? [el("div", { class: "mmc-warn", text: this.notice })] : []),
      ...this.standingNotes());
    this.samplingHost.replaceChildren(samplingBar({
      widgets: this.samplingWidgets,
      ...this.widgetIO(),
      set: (name, value) => { this.widgetIO().set(name, value); this.render(); },
      perSegment: false,
      turbo: state.arch === "krea2" ? [...this.renderLoader(), ...this.renderTurbo()] : [],
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

  /** How this LoRA reaches a 4-bit model: exactly, or fast.
   *
   *  A two-state toggle rather than a popover — there are two options and both
   *  fit in the label, so a menu would be a click to read what the button
   *  already says. It only exists while the SVDQuant loader is chosen. */
  renderAdapterButton(entry) {
    const mode = S.PRESTAGE_ADAPTER_MODES.includes(entry.adapters) ? entry.adapters : "bypass";
    return el("button", {
      class: "mmc-ghost",
      style: { fontSize: "11px" },
      title: S.PRESTAGE_ADAPTER_HINT[mode] + "\n\nClick to switch.",
      text: S.PRESTAGE_ADAPTER_LABEL[mode],
      onclick: () => {
        const modes = S.PRESTAGE_ADAPTER_MODES;
        entry.adapters = modes[(modes.indexOf(mode) + 1) % modes.length];
        this.commit();
      },
    });
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
      // Only on the quantized path, because it is the only place it means
      // anything: core's loader has no such choice, and offering one that does
      // nothing is worse than not offering it.
      ...(this.state.loader === "svdquant" ? [this.renderAdapterButton(entry)] : []),
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

  /** Standing observations about the state, as opposed to `this.notice`, which
   *  is something that just happened and fades. These are true until the setting
   *  changes, so they persist and they never block the render — every one of them
   *  is "this will work, and here is what it will cost".
   *
   *  Mirrors what `compile_image` computes, and deliberately not a clamp: the
   *  resolution pill and the boost dial are the user's. */
  standingNotes() {
    const state = this.state;
    const notes = [];
    if (!state.edit.on) return notes;

    const geometry = S.resolvedPreStage(state, state.edit.source
      ? this.sizes.get(state.edit.source.filename) : null);
    const ceiling = state.edit.source_b ? S.PRESTAGE_EDIT_TWO_REF_MAX : S.PRESTAGE_EDIT_SWEET_SPOT;
    if (geometry.width * geometry.height > ceiling) {
      const megapixels = (geometry.width * geometry.height / 1e6).toFixed(1);
      notes.push(el("div", { class: "mmc-note" }, [
        el("span", { class: "mmc-note-key", text: "edit size" }),
        el("span", {
          text: `${megapixels} MP. The edit weights work best around `
              + `${(ceiling / 1e6).toFixed(1)} MP — this renders, but slower and with a `
              + `looser hold on the source. Drop the resolution pill to tighten it.`,
        }),
      ]));
    }
    if (state.edit.ref_boost > S.PRESTAGE_REF_BOOST_OVERCOPY) {
      notes.push(el("div", { class: "mmc-note" }, [
        el("span", { class: "mmc-note-key", text: "boost" }),
        el("span", {
          text: `${state.edit.ref_boost.toFixed(2)} is past the point where the reference `
              + `starts being copied rather than referenced — removals and replacements `
              + `stop landing. 4 is the recommended setting.`,
        }),
      ]));
    }
    return notes;
  }

  // ---- krea2edit --------------------------------------------------------------

  /** The in-context edit path: a source image the render is an edit *of*.
   *
   *  Not img2img — the init pill is that, and the two are different things. An
   *  init is the latent the sampler starts from and gets partly overwritten; an
   *  edit source is conditioning, injected as frame=1 tokens and grounded into
   *  the text encode, and the sampler still starts from noise. */
  editPill() {
    const edit = this.state.edit;
    const blocked = this.state.refs.length > 0 && !edit.on;
    return el("button", {
      class: `mmc-pill${edit.on ? " accel-on" : ""}`,
      disabled: blocked || undefined,
      title: blocked
        ? "An edit and style references both build the positive conditioning, so only "
          + "one can run. Clear the style references to edit."
        : edit.on
          ? `Editing ${edit.source.filename}${edit.source_b ? " + a second reference" : ""}. `
            + "Click to change the source, the boost or the edit LoRA."
          : "Edit an image instead of describing one: the source goes in as in-context "
            + "tokens and grounds the instruction through the vision encoder. "
            + "Off — the prompt is read on its own.",
      onclick: (event) => this.openEdit(event.currentTarget),
    }, [
      icon("frameIn", 16),
      el("span", { text: edit.on ? "edit" : "edit off" }),
      ...(edit.on ? [el("span", {
        class: "mmc-pill-sub",
        text: edit.source.filename.split("/").pop().slice(0, 18),
      })] : []),
    ]);
  }

  async pickEditSource(slot) {
    const chosen = await openPicker({
      kinds: ["image", "renders"], kind: "image", single: true,
      capacity: () => ({ used: 0, max: 1, filesLeft: 1 }),
    });
    if (!chosen) return false;
    this.state.edit[slot] = { filename: chosen[0].path };
    if (slot === "source") {
      this.state.edit.on = true;
      // Both own the positive conditioning; the compile refuses the pair, so
      // the UI never builds it.
      this.state.refs = [];
    }
    this.commit();
    return true;
  }

  openEdit(anchor) {
    const state = this.state;
    const edit = state.edit;
    const pop = el("div", { class: "mmc-pop mmc-weights-pop" });
    const body = el("div");

    const fileRow = (label, slot, hint) => el("div", { class: "mmc-weight-row" }, [
      el("span", { class: "mmc-weight-name", text: label }),
      el("button", {
        class: `mmc-weight-file${edit[slot] ? "" : " empty"}`,
        title: hint,
        text: edit[slot] ? edit[slot].filename.split("/").pop() : "not set",
        onclick: async () => { if (await this.pickEditSource(slot)) render(); },
      }),
      ...(edit[slot] && slot === "source_b" ? [el("button", {
        class: "mmc-asset-x", text: "✕", title: "Drop the second reference",
        onclick: () => { edit.source_b = null; this.commit(); render(); },
      })] : []),
    ]);

    const numberRow = (label, key, hint, step = 0.05) => el("div", { class: "mmc-weight-row" }, [
      el("span", { class: "mmc-weight-name", text: label }),
      stepperPill({
        value: edit[key], min: 0, max: S.PRESTAGE_MAX_REF_BOOST, step, width: "56px",
        title: hint,
        format: (n) => n.toFixed(2),
        onChange: (value) => { edit[key] = value; this.commit(); render(); },
      }),
    ]);

    const render = () => {
      const rows = [fileRow("Source", "source",
                            "The image this render is an edit of. Its aspect becomes the canvas.")];

      if (edit.on) {
        rows.push(fileRow("Second reference", "source_b",
                          "Optional, for multi-reference edit LoRAs. Training order is scene "
                        + "first, subject second — so this slot is the subject."));

        // The edit LoRA. Suggested, never imposed: it may equally well be in the
        // main stack above, and someone who trained their own belongs here too.
        const loras = loraNames();
        const suggestion = loras.find((name) => S.PRESTAGE_EDIT_LORA_HINTS.some(
          (needle) => name.toLowerCase().includes(needle)));
        rows.push(el("div", { class: "mmc-weight-row" }, [
          el("span", { class: "mmc-weight-name", text: "Edit LoRA" }),
          el("button", {
            class: `mmc-weight-file${edit.lora ? "" : " empty"}`,
            title: "krea2edit's patch adds the in-context path; the Identity Edit LoRA is what "
                 + "was trained to use it. Patched on before the patch node. Leave this empty "
                 + "if the LoRA is already in the stack above — adding it twice doubles it.",
            text: edit.lora || (suggestion ? `not set — ${suggestion} found` : "not set"),
            onclick: (event) => openChoicePopover(event.currentTarget, {
              title: "Edit LoRA",
              options: ["— none —", ...loras],
              value: edit.lora || "— none —",
              onPick: (picked) => {
                edit.lora = picked === "— none —" ? "" : picked;
                this.commit();
                render();
              },
            }),
          }),
        ]));

        rows.push(numberRow("Reference boost", "ref_boost",
                            "Multiplies attention toward the last reference — the subject in a "
                          + "two-reference edit, the only one otherwise. 1.0 is off; higher pulls "
                          + "harder toward its appearance. The useful value is model-specific."));
        if (edit.source_b) {
          rows.push(numberRow("First-ref boost", "ref_boost_a",
                              "The same dial for the scene. No effect without a second reference."));
        }

        rows.push(el("div", { class: "mmc-weight-row" }, [
          el("span", { class: "mmc-weight-name", text: "Fit" }),
          el("div", { class: "mmc-pill mmc-turbo-seg" }, S.PRESTAGE_EDIT_FIT_MODES.map((mode) =>
            el("button", {
              class: "mmc-turbo-opt",
              "aria-pressed": edit.fit_mode === mode,
              title: S.PRESTAGE_EDIT_FIT_HINT[mode],
              onclick: () => { edit.fit_mode = mode; this.commit(); render(); },
            }, [el("span", { text: mode === "fit" ? "fit" : "crop" })]))),
        ]));

        rows.push(el("div", { class: "mmc-weight-row" }, [
          el("span", { class: "mmc-weight-name", text: "Grounding" }),
          stepperPill({
            value: edit.grounding_px, min: 0, max: S.PRESTAGE_MAX_GROUNDING_PX,
            step: 64, width: "64px",
            title: "Longest side of the source fed to the vision encoder. The edit LoRA trained "
                 + "with 384–768 px jitter, so 768 is in distribution; 0 means native.",
            onChange: (value) => { edit.grounding_px = Math.round(value); this.commit(); render(); },
          }),
        ]));

        rows.push(el("button", {
          class: "mmc-opt",
          onclick: () => {
            edit.on = false;
            edit.source = null;
            edit.source_b = null;
            this.commit();
            render();
          },
        }, [el("span", { class: "mmc-opt-label" }, [el("span", { text: "Switch the edit off" })])]));
      }

      body.replaceChildren(...rows);
    };

    pop.append(el("div", { class: "mmc-pop-title", text: "Edit" }), body);
    render();
    document.body.appendChild(pop);
    placeNear(pop, anchor);
    dismissable(pop);
    loadLoraNames(() => pop.isConnected && render());
  }

  // ---- style transfer ---------------------------------------------------------

  /** RF-inversion style transfer: a reference whose *look* this render takes.
   *
   *  Distinct from the style-reference rail tool, which goes through the
   *  Qwen-edit encoder and builds conditioning. This patches the attention path
   *  instead, and the two are mutually exclusive — the compile refuses the pair,
   *  so the pill that would create it is disabled rather than left to fail. */
  stylePill() {
    const style = this.state.style;
    const blocked = this.state.refs.length > 0 && !style.on;
    return el("button", {
      class: `mmc-pill${style.on ? " accel-on" : ""}`,
      disabled: blocked || undefined,
      title: blocked
        ? "Style transfer and style references are two different reference paths, so "
          + "only one can run. Clear the style references to use it."
        : style.on
          ? `Style transfer from ${style.refs.length} reference`
            + `${style.refs.length > 1 ? "s" : ""}. Click to change.`
          : "Take a reference image's look by RF-inversion — a patch on the attention "
            + "path rather than conditioning. Off.",
      onclick: (event) => this.openStyle(event.currentTarget),
    }, [
      icon("image", 16),
      el("span", { text: style.on ? "style" : "style off" }),
      ...(style.on && style.strength !== 1
        ? [el("span", { class: "mmc-pill-sub", text: style.strength.toFixed(2) })] : []),
    ]);
  }

  async addStyleRef() {
    const room = S.PRESTAGE_MAX_STYLE_TRANSFER_REFS - this.state.style.refs.length;
    if (room <= 0) {
      return this.flash(`At most ${S.PRESTAGE_MAX_STYLE_TRANSFER_REFS} style-transfer `
                      + `references — the pack has no route for a third.`);
    }
    const chosen = await openPicker({
      kinds: ["image", "renders"], kind: "image",
      capacity: () => ({ used: this.state.style.refs.length,
                         max: S.PRESTAGE_MAX_STYLE_TRANSFER_REFS, filesLeft: room }),
    });
    if (!chosen) return false;
    for (const asset of chosen.slice(0, room)) {
      this.state.style.refs.push({ filename: asset.path });
    }
    this.state.style.on = true;
    this.state.refs = [];
    this.commit();
    return true;
  }

  openStyle(anchor) {
    const style = this.state.style;
    const pop = el("div", { class: "mmc-pop mmc-weights-pop" });
    const body = el("div");

    const render = () => {
      const rows = style.refs.map((ref, index) => el("div", { class: "mmc-weight-row" }, [
        el("span", {
          class: "mmc-weight-name",
          text: style.refs.length > 1 ? `Reference ${index + 1}` : "Reference",
        }),
        el("button", {
          class: "mmc-weight-file",
          title: ref.filename,
          text: ref.filename.split("/").pop(),
          onclick: () => this.addStyleRef().then((added) => added && render()),
        }),
        el("button", {
          class: "mmc-asset-x", text: "✕", title: `Drop ${ref.filename}`,
          onclick: () => {
            style.refs.splice(index, 1);
            if (!style.refs.length) style.on = false;
            this.commit();
            render();
          },
        }),
      ]));

      if (style.refs.length < S.PRESTAGE_MAX_STYLE_TRANSFER_REFS) {
        rows.push(el("button", {
          class: "mmc-opt",
          onclick: () => this.addStyleRef().then((added) => added && render()),
        }, [el("span", { class: "mmc-opt-label" }, [
          el("span", { text: style.refs.length ? "Add a second reference" : "Choose a reference" }),
        ])]));
      }

      if (style.on) {
        rows.push(el("div", { class: "mmc-weight-row" }, [
          el("span", { class: "mmc-weight-name", text: "Fit" }),
          el("div", { class: "mmc-pill mmc-turbo-seg" }, S.PRESTAGE_STYLE_FITS.map((fit) =>
            el("button", {
              class: "mmc-turbo-opt",
              "aria-pressed": style.fit === fit,
              title: S.PRESTAGE_STYLE_FIT_HINT[fit],
              onclick: () => { style.fit = fit; this.commit(); render(); },
            }, [el("span", { text: fit })]))),
        ]));

        // One dial rather than fourteen. The transfer nodes ignore their advanced
        // widgets in "recommended" mode — so moving this is what switches them to
        // "custom", and the rest still come from the pack's own defaults.
        rows.push(el("div", { class: "mmc-weight-row" }, [
          el("span", { class: "mmc-weight-name", text: "Strength" }),
          stepperPill({
            value: style.strength, min: 0, max: S.PRESTAGE_MAX_STYLE_STRENGTH,
            step: 0.05, width: "56px",
            format: (n) => n.toFixed(2),
            title: "Overall style mix. At exactly 1.00 the pack's recommended preset runs "
                 + "as shipped; moving it switches the node to custom mode, where this "
                 + "dial is read and every other one keeps the recommended value.",
            onChange: (value) => { style.strength = value; this.commit(); render(); },
          }),
        ]));

        if (style.refs.length > 1) {
          rows.push(el("div", { class: "mmc-weight-row" }, [
            el("span", { class: "mmc-weight-name", text: "Leads" }),
            el("div", { class: "mmc-pill mmc-turbo-seg" }, [1, 2].map((n) =>
              el("button", {
                class: "mmc-turbo-opt",
                "aria-pressed": style.primary === n,
                title: `Reference ${n} carries the primary look; the other supports it.`,
                onclick: () => { style.primary = n; this.commit(); render(); },
              }, [el("span", { text: `ref ${n}` })]))),
          ]));
        }

        rows.push(el("button", {
          class: "mmc-opt",
          onclick: () => {
            style.on = false;
            style.refs = [];
            this.commit();
            render();
          },
        }, [el("span", { class: "mmc-opt-label" }, [
          el("span", { text: "Switch style transfer off" }),
        ])]));
      }

      body.replaceChildren(...rows);
    };

    pop.append(el("div", { class: "mmc-pop-title", text: "Style transfer" }), body);
    render();
    document.body.appendChild(pop);
    placeNear(pop, anchor);
    dismissable(pop);
  }

  // ---- moodboard --------------------------------------------------------------

  /** A Krea moodboard folded into the prompt. Off by default.
   *
   *  Not a rail tool, because it attaches nothing — no file, no handle, no chip.
   *  It is a property of the prompt, so it reads as a pill beside the aspect and
   *  resolution, and it says which board is on it. */
  moodboardPill() {
    const board = this.state.moodboard;
    return el("button", {
      class: `mmc-pill${board.on ? " accel-on" : ""}`,
      title: board.on
        ? `Moodboard — "${board.title || board.board}" at ${board.strength} strength, `
          + "appended to the prompt as style guidance. Click to change or clear."
        : "Fold one of 3,549 Krea moodboards into the prompt: a look chosen instead "
          + "of described. Off — the prompt goes as typed.",
      onclick: (event) => this.openMoodboard(event.currentTarget),
    }, [
      icon("effect", 16),
      el("span", { text: board.on ? (board.title || board.board).slice(0, 28) : "moodboard" }),
      ...(board.on ? [el("span", { class: "mmc-pill-sub", text: board.strength })] : []),
    ]);
  }

  openMoodboard(anchor) {
    const state = this.state;
    const board = state.moodboard;
    const pop = el("div", { class: "mmc-pop mmc-weights-pop" });
    const results = el("div");
    let query = "";
    let timer = null;

    const pick = (item) => {
      board.on = true;
      board.board = item.uuid || item.slug || item.title;
      board.title = item.title || "";
      board.collection = item.collection || board.collection;
      this.commit();
      render();
    };

    const load = () => {
      listMoodboards({ query, collection: board.collection }).then((payload) => {
        if (!pop.isConnected) return;
        if (!payload.available) {
          results.replaceChildren(el("div", {
            class: "mmc-warn",
            text: "The moodboard catalog is not installed — vendor/moodboards/data is missing.",
          }));
          return;
        }
        results.replaceChildren(...(payload.items.length
          ? payload.items.map((item) => el("button", {
              class: "mmc-opt",
              "aria-checked": board.on && board.board === item.uuid,
              title: item.source_summary || item.title,
              onclick: () => pick(item),
            }, [
              el("span", { class: "mmc-opt-label" }, [
                el("span", { text: item.title }),
                ...(item.keywords?.length
                  ? [el("span", { class: "mmc-pill-sub", text: item.keywords.slice(0, 3).join(" · ") })]
                  : []),
              ]),
              el("span", { class: "mmc-radio" }),
            ]))
          : [el("div", { class: "mmc-note", text: "No moodboard matched that." })]));
      });
    };

    const render = () => {
      const rows = [];

      rows.push(el("input", {
        class: "mmc-search",
        type: "text",
        placeholder: "Search 3,549 boards — “film noir”, “kodachrome”, “brutalist”",
        value: query,
        oninput: (event) => {
          query = event.target.value;
          // Every keystroke is a scan of the whole catalog on the server, so it
          // waits for the typing to stop rather than racing it.
          clearTimeout(timer);
          timer = setTimeout(load, 250);
        },
      }));
      rows.push(results);

      if (board.on) {
        rows.push(el("div", { class: "mmc-weight-row" }, [
          el("span", { class: "mmc-weight-name", text: "Strength" }),
          el("div", { class: "mmc-pill mmc-turbo-seg" }, S.PRESTAGE_MOODBOARD_STRENGTHS.map((level) =>
            el("button", {
              class: "mmc-turbo-opt",
              "aria-pressed": board.strength === level,
              title: S.PRESTAGE_MOODBOARD_HINT[level],
              onclick: () => { board.strength = level; this.commit(); render(); },
            }, [el("span", { text: level })]))),
        ]));
        // A board's negative guidance becomes the render's actual negative
        // conditioning. Worth a switch, and worth saying that Turbo skips it.
        rows.push(el("div", { class: "mmc-weight-row" }, [
          el("span", { class: "mmc-weight-name", text: "Negative guidance" }),
          el("button", {
            class: "mmc-weight-file",
            title: "The board's own list of what the look is not, encoded as the negative "
                 + "conditioning. At cfg 1 — which is where Turbo runs — the sampler skips "
                 + "the unconditional branch, so it costs nothing and does nothing there.",
            text: board.use_negative ? "used" : "ignored",
            onclick: () => { board.use_negative = !board.use_negative; this.commit(); render(); },
          }),
        ]));
        rows.push(el("button", {
          class: "mmc-opt",
          onclick: () => {
            board.on = false;
            board.board = "";
            board.title = "";
            this.commit();
            render();
          },
        }, [el("span", { class: "mmc-opt-label" }, [el("span", { text: "Clear the moodboard" })])]));
      }

      body.replaceChildren(...rows);
    };

    const body = el("div");
    pop.append(el("div", { class: "mmc-pop-title", text: "Moodboard" }), body);
    render();
    load();
    document.body.appendChild(pop);
    placeNear(pop, anchor);
    dismissable(pop);
  }

  renderPills() {
    const state = this.state;
    const geometry = S.resolvedPreStage(state, state.init ? this.sizes.get(state.init.filename) : null);

    const archPill = this.archPill?.() ?? el("span");

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

    const pills = [archPill, aspectPill, resPill, this.moodboardPill()];
    if (state.arch === "krea2") pills.push(this.editPill(), this.stylePill());

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

  // ---- loader (Krea 2) -------------------------------------------------------

  /** Which node loads the DiT: core's, or the vendored W4A4 one.
   *
   *  A segment rather than a toggle, on the turbo segment's own classes, because
   *  it is a choice between two named things and not a thing being switched on —
   *  and because "standard" has to read as a real, chosen position rather than
   *  as the absence of a setting.
   *
   *  It sits left of the turbo pill for the same reason it sits before it in the
   *  graph: this picks the file, turbo picks the schedule. */
  renderLoader() {
    const state = this.state;
    return [el("div", { class: "mmc-pill mmc-turbo-seg" }, S.PRESTAGE_LOADERS.map((loader) =>
      el("button", {
        class: "mmc-turbo-opt",
        "aria-pressed": state.loader === loader,
        title: S.PRESTAGE_LOADER_HINT[loader],
        onclick: () => {
          if (state.loader === loader) return;
          state.loader = loader;
          this.commit();
        },
      }, [el("span", { text: S.PRESTAGE_LOADER_LABEL[loader] })])))];
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
        svdq_model: byFolder.diffusion_models ?? [],
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

      // The SVDQuant loader takes no `weight_dtype` — a quantized checkpoint has
      // already decided its precision — so the row is left out rather than shown
      // as a control nothing reads.
      if (S.preStageUsesDtype(state)) {
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
      }

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

/**
 * The PreStage node's body: the blob, the stage, and whichever editor the
 * architecture calls for.
 *
 * Two of the three architectures are image models and are driven by the editor
 * above. The third is MiniMax H3, whose still is a *video generation* with one
 * latent frame decoded — so it is driven by `CreatorEditor`, the same body the
 * Creator node and every timeline segment use, on a request in the same shape.
 * That is not a saving of a few lines: the reference pipeline, the keyframe
 * pair, the slot arithmetic, the @-mention prompt, the routing badge and the
 * weights popover are one implementation, and a still gets all of them by
 * being what it is rather than by having them re-described.
 *
 * What this owns is what has to outlive a switch between the two: the blob, the
 * stage floated beside the node (the satellite bound it once), and the arch
 * pill itself — the control that does the switching cannot belong to the thing
 * being switched.
 */
export class PreStageBody {
  constructor({ state, onCommit, samplingWidgets, onWidgetChange, nodeId, peer = null }) {
    this.state = state;
    this.onCommit = onCommit;
    this.samplingWidgets = samplingWidgets;
    this.onWidgetChange = onWidgetChange;
    this.nodeId = nodeId;
    this.peer = peer;

    this.stage = new Stage({
      nodeId: this.nodeId,
      resultChips: (saved) => this.renderResultChips(saved),
    });

    this.host = el("div", { class: "mmc-prestage-host" });
    this.root = this.host;
    this.mount();
  }

  destroy() {
    this.editor?.destroy();
    this.stage?.destroy();
  }

  commit() {
    this.onCommit?.();
    this.editor?.render();
  }

  /** A saved workflow, or a stash restored onto a freshly spawned node. The
   *  architecture may differ from what is mounted, so this remounts. */
  setState(state) {
    this.state = state;
    this.mount();
  }

  mount() {
    this.editor?.destroy();
    this.editor = S.isStill(this.state) ? this.mountStill() : this.mountImage();
    this.host.replaceChildren(this.editor.root);
  }

  mountImage() {
    return new PreStageEditor({
      state: this.state,
      onCommit: () => this.onCommit?.(),
      samplingWidgets: this.samplingWidgets,
      onWidgetChange: this.onWidgetChange,
      nodeId: this.nodeId,
      stage: this.stage,
      archPill: () => this.renderArchPill(),
    });
  }

  /** The H3 branch: a Creator body on the still's own request.
   *
   *  Everything it is handed is what a Creator node hands it, minus three
   *  things a still has no use for — the seconds pill (how much video gets
   *  sampled to obtain the one frame is its own pill), the settings tool (it
   *  holds the video rate control), and the pre-stage pill, because this *is*
   *  the pre-stage.
   */
  mountStill() {
    const still = this.state.minimax;
    const editor = new CreatorEditor({
      state: still.request,
      onCommit: () => this.onCommit?.(),
      samplingWidgets: this.samplingWidgets,
      onWidgetChange: this.onWidgetChange,
      nodeId: this.nodeId,
      stage: this.stage,
      durationPill: false,
      // The settings page holds the video rate control and this node writes
      // PNGs, so it would be a button over nothing.
      settingsTool: false,
      extraPills: () => [this.renderArchPill(), ...this.renderStillPills()],
      extraTools: () => [this.renderFrameGrabTool()],
      setRoute: (route) => {
        still.request.models.route = route;
        this.commit();
      },
    });
    return editor;
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

  // ---- the model pill --------------------------------------------------------

  /** Switch architectures. Each side keeps its own state — its files, its
   *  canvas, its attachments — because the two have nothing in common but the
   *  node they are on; only the prompt is carried across, since that is the
   *  thing you were in the middle of writing. The sampler row is rewritten,
   *  because these models run at numbers that have nothing to do with each
   *  other and carrying the row across would be wrong on arrival. */
  setArch(arch) {
    if (arch === this.state.arch) return;
    const io = this.widgetIO();
    const from = this.promptOf();

    this.state.turbo.on = false;
    this.state.turbo.saved = null;
    this.state.arch = arch;

    if (arch === "krea2") {
      const row = S.PRESTAGE_KREA_RAW;
      io.set("steps", row.steps);
      io.set("cfg", row.cfg);
      io.set("sampler_name", row.sampler_name);
      io.set("scheduler", row.scheduler);
    } else if (arch === "ideogram4") {
      io.set("steps", S.PRESTAGE_IDEOGRAM_STEPS[this.state.quality]);
      io.set("cfg", S.PRESTAGE_IDEOGRAM_ROW.cfg);
      io.set("sampler_name", S.PRESTAGE_IDEOGRAM_ROW.sampler_name);
    } else {
      const row = S.PRESTAGE_STILL_ROW;
      io.set("steps", row.steps);
      io.set("cfg", row.cfg);
      io.set("sampler_name", row.sampler_name);
      io.set("scheduler", row.scheduler);
    }

    if (from && !this.promptOf()) this.setPrompt(from);
    this.onCommit?.();
    this.mount();
  }

  promptOf() {
    return (S.isStill(this.state) ? this.state.minimax.request.prompt : this.state.prompt) ?? "";
  }

  setPrompt(text) {
    if (S.isStill(this.state)) this.state.minimax.request.prompt = text;
    else this.state.prompt = text;
  }

  renderArchPill() {
    const state = this.state;
    const ARCH_TITLE = {
      krea2: "Krea 2 — 12.9B open-weights DiT. RAW samples at cfg 3.5; the turbo pill swaps in "
           + "the 8-step Turbo checkpoint.",
      ideogram4: "Ideogram 4.0 — 9.3B open-weights DiT with its own resolution-shifted schedule "
               + "and a second checkpoint for the unconditional branch.",
      minimax: "MiniMax H3 — experimental. The still is a video generation whose first latent "
             + "frame is decoded by the single-image H3 VAE, on the weights and the canvas your "
             + "render already uses. No second model family is loaded.",
    };
    return el("button", {
      class: `mmc-pill mmc-prestage-arch${S.isStill(state) ? " mmc-experimental" : ""}`,
      title: `${ARCH_TITLE[state.arch]} Click to switch.`,
      onclick: (event) => openChoicePopover(event.currentTarget, {
        title: "Image model",
        options: S.PRESTAGE_ARCHES.map((arch) => S.PRESTAGE_ARCH_LABEL[arch]),
        value: S.PRESTAGE_ARCH_LABEL[state.arch],
        onPick: (picked) => this.setArch(
          S.PRESTAGE_ARCHES.find((arch) => S.PRESTAGE_ARCH_LABEL[arch] === picked) ?? "krea2"),
      }),
    }, [icon("model", 16), el("span", { text: S.PRESTAGE_ARCH_LABEL[state.arch] })]);
  }

  // ---- the H3 branch's own pills ---------------------------------------------

  /** What a still costs and which frame of it is kept — the two things H3 has
   *  that a video render does not, because a video render keeps all of them. */
  renderStillPills() {
    const still = this.state.minimax;
    const latents = S.stillLatentFrames(still.frames);

    const length = el("button", {
      class: "mmc-pill",
      title: `${still.frames} frames sampled — ${latents} latent frames, of which one is `
           + "decoded. The shortest clip is the cheapest still; H3's trained range starts at "
           + "124 frames, so longer is more in-distribution and proportionally slower.",
      onclick: (event) => openChoicePopover(event.currentTarget, {
        title: "Sampled length",
        options: S.PRESTAGE_STILL_LENGTHS.map((n) => `${n} frames · ${S.stillLatentFrames(n)} latent`),
        value: `${still.frames} frames · ${latents} latent`,
        onPick: (picked) => {
          still.frames = Number(picked.split(" ")[0]);
          // A shorter clip can leave the kept frame past the end of it.
          const total = S.stillLatentFrames(still.frames);
          if (still.latent_index >= total) still.latent_index = total - 1;
          if (still.latent_index < -total) still.latent_index = 0;
          this.commit();
        },
      }),
    }, [icon("clock", 16), el("span", { text: `${still.frames}f` }),
        el("span", { class: "mmc-pill-sub", text: `${latents} latent` })]);

    const index = stepperPill({
      value: still.latent_index, min: -latents, max: latents - 1, step: 1, width: "56px",
      title: "Which latent frame becomes the picture. 0 is the causal first frame — the one "
           + "slice the single-image VAE was trained on, and the only one that is a function "
           + "of a single video frame. Negative counts from the end.",
      format: (n) => `latent ${n}`,
      onChange: (next) => { still.latent_index = Math.round(next); this.commit(); },
    });

    return [length, index];
  }

  // ---- the rest of the rail --------------------------------------------------

  /** Not on the Creator's rail, because the Creator has this node. Here it is
   *  the only way to turn a moment of a clip into a keyframe. */
  renderFrameGrabTool() {
    return el("button", {
      class: "mmc-tool",
      title: "Pull a single frame off a video's playhead and open on it — saved as a PNG in "
           + "the input folder.",
      onclick: () => this.grabFrame(),
    }, [el("span", { class: "mmc-tool-icon" }, [icon("video")]), el("span", { text: "From video" })]);
  }

  async grabFrame() {
    const request = this.state.minimax.request;
    const blocked = S.blockedReason(request, "first_frame");
    if (blocked) return;
    const clip = await openPicker({
      kinds: ["video", "renders"], kind: "video", single: true,
      capacity: () => ({ used: 0, max: 1, filesLeft: 1 }),
    });
    if (!clip) return;
    const grabbed = await openFrameGrab({ path: clip[0].path });
    if (!grabbed) return;
    const existing = S.frameAsset(request, "first_frame");
    if (existing) request.assets = request.assets.filter((a) => a.handle !== existing.handle);
    request.assets.push({
      handle: S.nextHandle(request, "image"),
      kind: "image",
      role: "first_frame",
      filename: grabbed.path,
    });
    this.commit();
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
      onclick: () => target.attach(role, filename),
    });
    return [
      chip("first_frame", "→ start", "Use this still as the start frame"),
      chip("last_frame", "→ end", "Use this still as the end frame"),
      chip("reference", "→ ref", "Attach this still as a reference"),
    ];
  }
}
