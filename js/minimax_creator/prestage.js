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
    const previous = S.prestageEdges(this.state);
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
    // The two sides do not share a canvas: H3 generates video frames on a /32
    // grid at a 768 short edge, the image models generate on /16 up to 2048.
    // Carried across rather than reset, so switching back and forth does not
    // lose the shape you set — only what the new side cannot express.
    if (!S.prestageAspects(this.state).some(([label]) => label === this.state.aspect)) {
      this.state.aspect = "16:9";
    }
    // An edge left at the old arch's default lands on the new one's rather than
    // being clamped into range: 1024 and 768 are each their side's "normal",
    // and arriving at 896 would be neither.
    this.state.short_edge = this.state.short_edge === previous.default
      ? S.prestageEdges(this.state).default
      : S.clampPreStageEdge(this.state, this.state.short_edge);
    this.commit();
  }

  // ---- init image and style references --------------------------------------

  /** Pick the init image — the still this render restyles rather than starts
   *  from nothing. From the picker, or grabbed off a video's playhead. */
  async setInit(fromVideo = false) {
    if (S.isStill(this.state) && this.state.refs.length) {
      return this.flash("H3 runs start frames on FL2VA and references on Ref2VA, and one "
                      + "generation uses one checkpoint — remove the references first.");
    }
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
    // FL2VA and Ref2VA are two checkpoints and one generation runs on one of
    // them, so H3 cannot take a keyframe and a reference together. Said here
    // rather than left to the queue, which is where the video nodes say it too.
    if (S.isStill(this.state) && (this.state.init || this.state.end)) {
      return this.flash("H3 runs start/end frames on FL2VA and references on Ref2VA, and one "
                      + "generation uses one checkpoint — remove the keyframe first.");
    }
    // On H3 a reference is anything a shot's is: nine images, three clips,
    // three sounds, twelve files. On the image models it is three image slots
    // and nothing else, which is the Qwen edit encoder's shape.
    if (S.isStill(this.state) && !fromVideo) return this.addStillRefs();
    const max = S.prestageMaxRefs(this.state);
    const room = max - this.state.refs.length;
    if (room <= 0) {
      return this.flash(`At most ${max} style references — the Qwen edit encoder the model `
                      + `reads them through has exactly three image slots.`);
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

  /** H3 references: the video node's whole reference pipeline, on a still.
   *
   *  Nine images, three clips, three sounds, twelve files — the same picker,
   *  the same buckets, the same slot counters, because the payload this builds
   *  is the one a shot builds. A clip brings its own trim and track through the
   *  picker's segment editor; both ride into the request untouched.
   */
  async addStillRefs(kind = "image") {
    const blocked = S.stillBlockedReason(this.state, "reference");
    if (blocked) return this.flash(blocked);
    const chosen = await openPicker({
      // Every tab is offered whichever button opened the picker — the button
      // decides which one it lands on, the same way the Creator's do.
      kinds: kind === "renders"
        ? ["renders", "image", "video", "audio"]
        : ["image", "video", "audio", "renders"],
      kind,
      capacity: (bucket) => S.stillCapacity(this.state, bucket),
    });
    if (!chosen) return;
    for (const asset of chosen) {
      const entry = {
        handle: S.nextPreStageHandle(this.state, asset.kind),
        kind: asset.kind,
        filename: asset.path,
        ref_size: "max",
      };
      if (asset.kind === "video") entry.track = asset.track || S.DEFAULT_TRACK;
      if (asset.trim) entry.trim = asset.trim;
      this.state.refs.push(entry);
    }
    // The picker counts slots as they are picked, but a track changed in its
    // segment editor moves a file between buckets — so the finished list is
    // checked once against the same limits compile.py holds it to.
    const over = S.stillOverflow(this.state);
    if (over) {
      this.state.refs = this.state.refs.slice(0, this.state.refs.length - chosen.length);
      return this.flash(over);
    }
    this.commit();
  }

  /** The end frame: the image the sampled clip closes on. H3 only — the image
   *  models have no such thing, and on a still it is the other half of the
   *  keyframe pair the video nodes already take. */
  async setEnd() {
    if (this.state.refs.length) {
      return this.flash("H3 runs start/end frames on FL2VA and references on Ref2VA, and one "
                      + "generation uses one checkpoint — remove the references first.");
    }
    const chosen = await openPicker({
      kinds: ["image", "renders"], kind: "image", single: true,
      capacity: () => ({ used: 0, max: 1, filesLeft: 1 }),
    });
    if (!chosen) return;
    this.state.end = { filename: chosen[0].path };
    this.commit();
  }

  async manageLoras() {
    // On H3 a LoRA belongs to a checkpoint — the same per-checkpoint modes the
    // video nodes offer, because these are the video nodes' weights. The image
    // models have one DiT per arch and so have no such split.
    await openLoras({ state: this.state, checkpointModes: S.isStill(this.state),
                      onChange: () => this.commit() });
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
    this.promptBox.placeholder = S.isStill(state)
      ? "Describe the still. H3 reads the same long, cinematic prompt a shot does — it is "
        + "generating a moment of video and keeping one frame of it."
      : "Describe the image. Both models were trained on long, detailed natural-language prompts.";
    this.railHost.replaceChildren(this.renderRail());
    const chips = [
      ...(state.init ? [this.renderInitChip()] : []),
      ...(state.end && S.isStill(state) ? [this.renderEndChip()] : []),
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

    // On H3 the rail is the Creator's rail, because the references are the
    // Creator's references: one button per kind, gated together when a keyframe
    // is attached, plus the gallery. The keyframes themselves move to pills at
    // the bottom, exactly as they are on a Creator — a still is set up the way
    // a shot is or it is a second thing to learn.
    if (S.isStill(this.state)) return this.renderStillRail(tool);

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

  renderStillRail(tool) {
    const blocked = S.stillBlockedReason(this.state, "reference");
    const kindTool = (kind, label, iconName) => el("button", {
      class: "mmc-tool",
      disabled: blocked || undefined,
      title: blocked || `Attach a reference ${kind} — cited in the prompt as @${
        { image: "img", video: "vid", audio: "aud" }[kind]}-1, exactly as in a shot`,
      onclick: () => this.addStillRefs(kind),
    }, [el("span", { class: "mmc-tool-icon" }, [icon(iconName)]), el("span", { text: label })]);

    return el("div", { class: "mmc-rail" }, [
      kindTool("image", "Add image", "image"),
      kindTool("video", "Add video", "video"),
      kindTool("audio", "Add audio", "audio"),
      // Ungated, like the Creator's: a LoRA sits on the checkpoint rather than
      // in a reference slot, so it is the one thing frames and references share.
      tool("Add LoRA", "effect",
           "Manage the LoRAs patched onto the routed H3 checkpoint.",
           () => this.manageLoras()),
      tool("Gallery", "gallery",
           "Browse, organize and attach finished renders and pre-stage stills.",
           () => this.addStillRefs("renders")),
      // Not on the Creator's rail, because the Creator has this node. Here it
      // is the only way to turn a moment of a clip into a keyframe.
      tool("From video", "video",
           "Pull a single frame off a video's playhead and open on it — saved as a PNG in the input folder.",
           () => this.setInit(true)),
    ]);
  }

  renderInitChip() {
    const init = this.state.init;
    // On H3 the image is a keyframe, which has no strength — the stepper would
    // be a control that changes nothing.
    const strength = S.isStill(this.state) ? [] : [
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
    ];

    return el("div", { class: "mmc-asset mmc-tag-0", title: init.filename }, [
      el("img", { class: "mmc-asset-thumb", src: viewUrl(init.filename, { preview: true }), alt: init.filename }),
      el("span", { class: "mmc-asset-handle", text: S.isStill(this.state) ? "start" : "init" }),
      ...strength,
      el("button", {
        class: "mmc-asset-x", text: "✕",
        title: S.isStill(this.state) ? "Remove the start frame" : "Remove the init image",
        onclick: () => { this.state.init = null; this.commit(); },
      }),
    ]);
  }

  /** The end frame's chip. Its own rather than a parameter on the init chip's,
   *  because the two carry different controls: an init has a strength on the
   *  image models, an end frame has none anywhere. */
  renderEndChip() {
    const end = this.state.end;
    return el("div", { class: "mmc-asset mmc-tag-1", title: end.filename }, [
      el("img", { class: "mmc-asset-thumb", src: viewUrl(end.filename, { preview: true }), alt: end.filename }),
      el("span", { class: "mmc-asset-handle", text: "end" }),
      el("button", {
        class: "mmc-asset-x", text: "✕", title: "Remove the end frame",
        onclick: () => { this.state.end = null; this.commit(); },
      }),
    ]);
  }

  renderRefChip(ref) {
    const kind = ref.kind || "image";
    // A clip's soundtrack is a reference of its own — it takes an <Audio N>
    // label and its own slot — so which streams are cited has to be visible and
    // changeable on the chip. Cycled rather than a popover: three states.
    const track = kind === "video" ? [el("button", {
      class: "mmc-ghost",
      style: { fontSize: "11px" },
      title: "Which streams of this clip are referenced: picture, picture+sound, or the "
           + "soundtrack alone. Sound needs the audio VAE and takes an audio slot.",
      text: (ref.track ?? S.DEFAULT_TRACK).replace("picture+sound", "pic+snd").replace("picture", "pic"),
      onclick: () => {
        const next = S.TRACKS[(S.TRACKS.indexOf(ref.track ?? S.DEFAULT_TRACK) + 1) % S.TRACKS.length];
        const was = ref.track;
        ref.track = next;
        // Switching to sound-only moves the file between buckets and the one it
        // lands in may be full — the same rule the picker holds a selection to.
        const over = S.stillOverflow(this.state);
        if (over) {
          ref.track = was;
          return this.flash(over);
        }
        this.commit();
      },
    })] : [];

    return el("div", {
      class: `mmc-asset mmc-tag-${S.tagIndex(ref.handle)}`,
      title: ref.filename,
    }, [
      kind === "image"
        ? el("img", { class: "mmc-asset-thumb", src: viewUrl(ref.filename, { preview: true }), alt: ref.filename })
        : el("span", { class: "mmc-asset-thumb" }, [svg(ICONS[kind], 15)]),
      el("span", { class: "mmc-asset-handle", text: `@${ref.handle}` }),
      el("span", { class: "mmc-asset-role", text: S.isStill(this.state) ? "ref" : "style" }),
      ...track,
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

    const ARCH_TITLE = {
      krea2: "Krea 2 — 12.9B open-weights DiT. RAW samples at cfg 3.5; the turbo pill swaps in "
           + "the 8-step Turbo checkpoint.",
      ideogram4: "Ideogram 4.0 — 9.3B open-weights DiT with its own resolution-shifted schedule "
               + "and a second checkpoint for the unconditional branch.",
      minimax: "MiniMax H3 — experimental. The still is a video generation whose first latent "
             + "frame is decoded by the single-image H3 VAE, on the weights and the canvas your "
             + "render already uses. No second model family is loaded.",
    };
    const archPill = el("button", {
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
      title: S.isStill(state)
        ? "Short edge. H3's own: the weights are trained at 768 with a 768×1344 area cap, so a "
          + "still lands on exactly the canvas the video render will use."
        : "Short edge. Both models are comfortable up to a 2048×2048 area.",
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

    // The Creator's order — what is attached, then how big, then how long, then
    // where it lands — because a still is set up the way a shot is.
    const pills = S.isStill(state)
      ? [archPill, ...this.renderStillFramePills(), aspectPill, resPill, outputPill]
      : [archPill, aspectPill, resPill, outputPill];

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

    if (S.isStill(state)) pills.push(...this.renderStillPills());

    if (state.init && !S.isStill(state)) {
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

  // ---- the H3 branch ---------------------------------------------------------

  /** What a still costs and which frame of it is kept.
   *
   * Both are H3's alone. The length is how much video gets sampled to obtain
   * the one latent frame that becomes the picture — the model's trained range
   * starts at 124 frames, so the cheap end of this pill is deliberately off
   * distribution and the dev sweep exists to find out how far down it holds up.
   */
  /** The keyframe pair, in the Creator's own idiom: a pill each, disabled with
   *  the reason when references have taken the generation to Ref2VA. The chip
   *  above the prompt is what removes one, exactly as on a Creator. */
  renderStillFramePills() {
    const pill = (role, field, label, iconName, setter) => {
      const blocked = S.stillBlockedReason(this.state, role);
      const attached = this.state[field];
      return el("button", {
        class: "mmc-pill",
        disabled: blocked ? true : undefined,
        title: blocked || (attached
          ? `${label}: ${attached.filename}. Click to replace it; the chip above removes it.`
          : `Choose the ${label.toLowerCase()}`),
        onclick: blocked ? undefined : setter,
      }, [icon(iconName, 16), el("span", { text: attached ? label.toLowerCase() : label })]);
    };

    return [
      pill("first_frame", "init", "Start frame", "frameIn", () => this.setInit(false)),
      pill("last_frame", "end", "End frame", "frameOut", () => this.setEnd()),
    ];
  }

  renderStillPills() {
    const still = this.state.minimax;
    const latents = S.stillLatentFrames(still.frames);
    const pills = [];

    pills.push(el("button", {
      class: "mmc-pill",
      title: `${still.frames} frames sampled — ${latents} latent frames, of which one is `
           + `decoded. The shortest clip is the cheapest still; H3's trained range starts at `
           + `${S.PRESTAGE_STILL_LENGTHS[S.PRESTAGE_STILL_LENGTHS.length - 1]} frames, so longer `
           + `is more in-distribution and proportionally slower.`,
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
        el("span", { class: "mmc-pill-sub", text: `${latents} latent` })]));

    pills.push(stepperPill({
      value: still.latent_index, min: -latents, max: latents - 1, step: 1, width: "56px",
      title: "Which latent frame becomes the picture. 0 is the causal first frame — the one "
           + "slice the single-image VAE was trained on, and the only one that is a function of "
           + "a single video frame. Negative counts from the end.",
      format: (n) => `latent ${n}`,
      onChange: (next) => { still.latent_index = Math.round(next); this.commit(); },
    }));

    pills.push(this.renderDevPill());   // DEV

    // The Creator's routing badge, read-only: what this compiles to and which
    // checkpoint it will load. Same derivation, same words, same place on the
    // row — the routing is not different here, so it should not look different.
    const badge = el("span", {
      class: "mmc-mode",
      title: "What this still compiles to, and the checkpoint it loads. References are encoded "
           + "for Ref2VA; keyframes and bare prompts run on FL2VA. The same routing a video "
           + "render does, because this is one.",
    });
    badge.appendChild(el("b", { text: S.stillMode(this.state) }));
    badge.appendChild(document.createTextNode(` → ${S.CHECKPOINT_LABEL[S.stillCheckpoint(this.state)]}`));
    pills.push(badge);
    return pills;
  }

  // DEV: the sweep pill, and everything it opens. One queue renders every
  // combination of sampled length, kept latent frame and decoder, each saved
  // under a name carrying its coordinate, so the three open questions can be
  // answered by looking at a folder. Comes out with `compile_still`'s DEV block.
  renderDevPill() {
    const still = this.state.minimax;
    const sweep = S.stillSweep(still);
    return el("button", {
      class: `mmc-pill${sweep.on ? " accel-on" : ""}`,
      title: sweep.on
        ? `Dev sweep on: ${sweep.passes} sampler pass(es), ${sweep.images} pictures, one per `
          + `combination. Each file is named with its length, latent frame and decoder.`
        : "Dev sweep — render several lengths, latent frames and decoders in one queue to "
        + "compare them. Off, this renders the settings above once.",
      onclick: (event) => this.openDevSweep(event.currentTarget),
    }, [icon("effect", 16),
        el("span", { text: sweep.on ? `sweep · ${sweep.images}` : "sweep" })]);
  }

  // DEV
  openDevSweep(anchor) {
    const still = this.state.minimax;
    const pop = el("div", { class: "mmc-pop mmc-weights-pop" });
    const body = el("div");

    const toggle = (list, value) => {
      const at = list.indexOf(value);
      if (at === -1) list.push(value);
      else list.splice(at, 1);
      list.sort((a, b) => (typeof a === "string" ? String(a).localeCompare(String(b)) : a - b));
      this.commit();
      render();
    };

    const row = (label, options, list, format) => el("div", { class: "mmc-weight-row" }, [
      el("span", { class: "mmc-weight-name", text: label }),
      el("div", { class: "mmc-pills" }, options.map((option) => el("button", {
        class: "mmc-pill",
        "aria-pressed": list.includes(option),
        onclick: () => toggle(list, option),
      }, [el("span", { text: format ? format(option) : String(option) })]))),
    ]);

    const render = () => {
      const vaes = (catalogByFolder().vae ?? []).filter((name) => /h3|minimax/i.test(name)
        && !/audio/i.test(name));
      // Every latent frame the longest swept length has, so the choices do not
      // change under you as the length list does. Out-of-range combinations are
      // refused at queue time by compile_still.
      const longest = Math.max(still.frames, ...(still.dev.lengths.length ? still.dev.lengths : [0]));
      const indices = [...Array(S.stillLatentFrames(longest)).keys()];
      body.replaceChildren(
        row("Lengths", S.PRESTAGE_STILL_LENGTHS, still.dev.lengths, (n) => `${n}f`),
        row("Latent frames", indices, still.dev.indices, (n) => `i${n}`),
        row("Decoders", vaes, still.dev.vaes, (name) => name.split("/").pop().replace(/\.[^.]+$/, "").slice(-18)),
        el("div", { class: "mmc-note" }, [
          el("span", { class: "mmc-note-key", text: "writes" }),
          el("span", { text: `${S.stillSweep(still).images} picture(s) per queue — an empty row `
                           + `means "just the pill's own setting".` }),
        ]),
      );
    };

    pop.append(el("div", { class: "mmc-pop-title", text: "Dev sweep" }), body);
    render();
    document.body.appendChild(pop);
    placeNear(pop, anchor);
    dismissable(pop);
    loadCatalog(() => pop.isConnected && render());
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
      const lists = S.prestageFileLists(catalogByFolder());
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
    for (const [label, ratio] of S.prestageAspects(this.state)) {
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
    const bounds = S.prestageEdges(this.state);
    const still = S.isStill(this.state);
    const body = edgeSlider({
      min: bounds.min, max: bounds.max, step: bounds.step,
      value: this.state.short_edge,
      mark: bounds.default, markLabel: still ? "native" : "default",
      apply: (edge) => { this.state.short_edge = edge; },
      describe: () => {
        const geometry = S.resolvedPreStage(this.state,
          this.state.init ? this.sizes.get(this.state.init.filename) : null);
        const note = still
          ? (this.state.short_edge > bounds.default
              ? "Past what H3 was trained on — the still gets bigger, not better."
              : `${bounds.default} is what the weights were trained at, and the canvas your `
                + `video render will use.`)
          : (this.state.short_edge >= bounds.max
              ? "The models' 2048 ceiling — wide ratios trade the short edge down to hold the area."
              : `${this.state.short_edge < bounds.default ? "Faster, softer." : "Sharper, slower."} `
                + `${bounds.default} is the comfortable default for both models.`);
        return { size: `${geometry.width} × ${geometry.height}`, note };
      },
      commit: () => this.commit(),
    });
    const pop = el("div", { class: "mmc-pop mmc-slider" }, [body]);
    document.body.appendChild(pop);
    placeNear(pop, anchor);
    dismissable(pop);
  }
}
