// The timeline modal: a global prompt, one canvas, and a strip of segments.
//
// The strip is the whole idea — cards laid out left to right, each as wide as
// its own duration, with the join between two of them saying whether the second
// cuts or continues. Everything inside a card is a whole generation, so opening
// one hands it to CreatorEditor unchanged: same rail, same @ prompt, same
// LoRAs, same routing badge. There is no reduced "segment UI" to keep in step
// with the node's, because there is only one editor.

import { el, icon, mountOverlay } from "./dom.js";
import { CreatorEditor } from "./editor.js";
import { openLoras } from "./loras.js";
import { openPicker } from "./picker.js";
import { openAspectPopover, openResolutionPopover, openChoicePopover, stepperPill, aspectGlyph, PILL_GLYPH } from "./pills.js";
import { refine, refineButton, chosenModel as refineModel } from "./refine.js";
import { samplingBar } from "./sampling.js";
import { Stage } from "./stage.js";
import { weightsPill, loadCatalog, catalogFiles } from "./models.js";
import * as S from "./state.js";
import * as Turbo from "./turbo.js";
import {
  framesForSeconds, secondsForFrames, resolveCanvas, ASPECT_PRESETS, describeRatio, isTrainedLength,
} from "./canvas.js";

/**
 * @param {object} options
 * @param {object} options.timeline    mutated in place
 * @param {() => void} options.onCommit
 * @returns {Promise<void>} resolves when the modal closes
 */
export function openTimeline(options) {
  return new Promise((resolve) => new Timeline(options, resolve).mount());
}

/**
 * Card width from duration. Compressed rather than linear: durations run 1 s to
 * 60 s, and at any scale that keeps a 1-second card wide enough for its own
 * buttons a 60-second one would be most of a metre. Square root keeps the
 * ordering legible — a longer shot is visibly a wider card — without that.
 * The lane in the node body stays strictly proportional; that is its whole job.
 */
const cardWidth = (seconds) => 132 + Math.round(Math.sqrt(seconds) * 26);

class Timeline {
  constructor({ timeline, onCommit }, resolve) {
    this.timeline = timeline;
    this.onCommit = onCommit;
    this.resolve = resolve;
  }

  commit() {
    S.syncTimeline(this.timeline);
    this.onCommit?.();
    this.render();
  }

  /**
   * One of the timeline's global text fields.
   *
   * Built once and never re-rendered: a full render would rebuild the element
   * under the caret and lose the selection mid-sentence, which is why `commit`
   * redraws the bar and the strip and leaves these alone.
   */
  textBox(key, { className = "mmc-tl-prompt", placeholder, rows }) {
    const box = el("textarea", {
      class: className,
      placeholder,
      ...(rows ? { rows: String(rows) } : {}),
      oninput: (event) => {
        this.timeline[key] = event.target.value;
        this.onCommit?.();
        this.renderBar();
      },
    });
    box.value = this.timeline[key] ?? "";
    // The canvas is drag-to-pan territory in the graph; a textarea needs its
    // own pointer events.
    box.addEventListener("pointerdown", (event) => event.stopPropagation());
    return box;
  }

  mount() {
    this.promptBox = this.textBox("prompt", {
      placeholder: "The whole piece: setting, look, who is in it. Added in front of every segment's own prompt.",
    });

    // The two audio fields H3's own prompt format has, kept side by side and
    // shorter than the prompt: they are a few sentences each, and putting them
    // under the picture description is the order the model reads them in.
    // Held rather than built inline: the refiner writes into both, and it has to
    // put the text where the user can see it rather than only into the state
    // behind them.
    this.soundscapeBox = this.textBox("soundscape", {
      className: "mmc-tl-prompt mmc-tl-small", rows: 3,
      placeholder: "Ambience, action sounds, breathing — everything heard in the room. "
                 + "Empty leaves it to the model; write N/A for silence.",
    });
    this.musicBox = this.textBox("music", {
      className: "mmc-tl-prompt mmc-tl-small", rows: 3,
      placeholder: "The score only the audience hears: instruments, tempo, how it moves. "
                 + "Empty leaves it to the model; write N/A for none.",
    });

    this.audioHost = el("div", { class: "mmc-tl-audio" }, [
      el("label", { class: "mmc-tl-field" }, [
        el("span", { class: "mmc-tl-field-name", text: "overall_soundscape" }),
        this.soundscapeBox,
      ]),
      el("label", { class: "mmc-tl-field" }, [
        el("span", { class: "mmc-tl-field-name", text: "non_diegetic_music" }),
        this.musicBox,
      ]),
    ]);

    this.barHost = el("div", { class: "mmc-tl-bar" });
    this.stripHost = el("div", { class: "mmc-tl-strip" });

    this.modal = el("div", { class: "mmc-modal mmc-tl-modal" }, [
      el("div", { class: "mmc-modal-head" }, [
        el("span", { class: "mmc-tab", "aria-selected": "true", text: "Timeline" }),
        el("button", { class: "mmc-close", text: "✕", title: "Close", onclick: () => this.close() }),
      ]),
      el("div", { class: "mmc-tl-body" }, [
        this.promptBox, this.audioHost, this.barHost, this.stripHost,
      ]),
    ]);

    this.overlay = el("div", {
      class: "mmc-overlay",
      onpointerdown: (event) => { if (event.target === this.overlay) this.close(); },
    }, [this.modal]);

    this.unmount = mountOverlay(this.overlay, () => this.close());
    this.render();
  }

  close() {
    this.unmount();
    this.resolve();
  }

  // ---- render ---------------------------------------------------------------

  render() {
    // The global prompt is the same field either way but lands in a different
    // place: in front of every segment when chained, at the head of Shot 1's
    // description when not — which is where the guide puts the style and the
    // opening composition, and is worth saying because it changes how to write it.
    this.promptBox.placeholder = S.isSingle(this.timeline)
      ? "The whole piece: setting, look, who is in it. Opens Shot 1's description, so write it as the start of one."
      : "The whole piece: setting, look, who is in it. Added in front of every segment's own prompt.";
    this.renderBar();
    this.renderStrip();
  }

  geometry() {
    const ratio = ASPECT_PRESETS.find(([label]) => label === this.timeline.aspect)?.[1] ?? 16 / 9;
    const [width, height] = resolveCanvas(ratio, this.timeline.short_edge);
    return { width, height, ratio };
  }

  /**
   * Chained or one pass — the same strip, read two ways.
   *
   * A toggle rather than two nodes because the timeline *is* the same document
   * in both: the same shots, the same durations, the same prompts. What changes
   * is whether they are rendered one after another and joined, or compiled into
   * one description with cut times in it and generated at once.
   */
  renderMode() {
    const single = S.isSingle(this.timeline);
    const option = (mode, label, title) => el("button", {
      class: `mmc-tl-render-opt${(mode === "single") === single ? " on" : ""}`,
      text: label,
      title,
      onclick: () => {
        if (this.timeline.render === mode) return;
        this.timeline.render = mode;
        this.commit();
      },
    });
    return el("div", { class: "mmc-tl-render" }, [
      option("chained", "Chained",
        "One generation per segment, joined end to end. No limit on the finished length, "
        + "and a segment can start from the previous one's last frame — but every join is a real seam."),
      option("single", "One pass",
        "One generation. The segments become the shots of a single description, cut times and all, "
        + "so nothing is decoded and re-encoded mid-clip and there is no seam to cross. "
        + "Everything a single pass can only have one of — mode, checkpoint, LoRAs, seed — "
        + "becomes the timeline's."),
    ]);
  }

  /**
   * The mode, the canvas, and the running total.
   *
   * The canvas sits here rather than on a segment because chained segments are
   * concatenated frame by frame at the end, which is only defined if they all
   * came out the same size. compile.py enforces it by compiling every segment
   * against the geometry the first one resolved; this is where the user sets it.
   */
  renderBar() {
    const single = S.isSingle(this.timeline);
    const { width, height, ratio } = this.geometry();
    const seconds = S.timelineSeconds(this.timeline);
    const frames = S.timelineFrames(this.timeline);
    const count = this.timeline.segments.length;
    const active = S.activeGlobalLoras(this.timeline).length;
    const idle = (this.timeline.loras?.length ?? 0) - active;
    const problem = single ? S.singleProblem(this.timeline) : null;
    const refined = this.timeline.segments.some((segment) => segment.refined?.body);

    this.barHost.replaceChildren(
      this.renderMode(),
      el("button", {
        class: "mmc-pill",
        title: "Aspect ratio, shared by every segment — they are joined end to end and have to match.",
        onclick: (event) => openAspectPopover(event.currentTarget, this.timeline, () => this.commit()),
      }, [aspectGlyph(ratio, PILL_GLYPH),
          el("span", { text: this.timeline.aspect }),
          el("span", { class: "mmc-pill-sub", text: describeRatio(ratio) })]),
      el("button", {
        class: "mmc-pill",
        title: S.twoPass(this.timeline)
          ? `Sampled at a ${S.sampleEdge(this.timeline)} px short edge, refined up to `
            + `${width} × ${height} by a second pass — every segment alike.`
          : "Short edge. Lower is faster; 768 is what the open weights were trained at.",
        onclick: (event) => openResolutionPopover(
          event.currentTarget, this.timeline, () => this.geometry(), () => this.commit()),
      }, [
        icon("res", 16),
        el("span", { text: `${this.timeline.short_edge}p` }),
        el("span", { class: "mmc-pill-sub", text: S.twoPass(this.timeline)
          ? `${S.sampleEdge(this.timeline)} → ${width} × ${height}`
          : `${width} × ${height}` }),
      ]),
      // Global LoRAs sit on the bar with the canvas rather than inside a
      // segment, because that is what they are: patched onto every segment,
      // which is the whole reason to have them separately from a segment's own.
      el("button", {
        class: `mmc-pill${active ? " on" : ""}`,
        title: "LoRAs patched onto every segment, in front of whatever that segment adds. "
             + "Where a turbo LoRA belongs.",
        onclick: () => this.openLoras(),
      }, [
        icon("effect", 16),
        el("span", { text: active ? `${active} LoRA${active === 1 ? "" : "s"}` : "LoRAs" }),
        ...(idle ? [el("span", { class: "mmc-pill-sub", text: `${idle} idle` })] : []),
      ]),
      // Only once a seam actually carries sound: until then it is a setting for
      // a feature that is not in use — which includes all of one-pass mode,
      // where there are no seams to carry anything.
      ...(!single && this.timeline.segments.some(S.continuesAudio) ? [stepperPill({
        value: Number(this.timeline.audio_tail_s), min: 0.1, max: S.MAX_AUDIO_TAIL_S,
        step: 0.1, width: "52px", iconName: "audio",
        title: "How much of the previous segment's sound a continuing seam inherits. "
             + "Longer costs sampling time and pulls the inherited start frame off the clip's opening.",
        format: (n) => `${n.toFixed(1)}s tail`,
        onChange: (next) => { this.timeline.audio_tail_s = next; this.commit(); },
      })] : []),
      // The mode belongs to a generation, and in one pass there is one for the
      // whole timeline rather than one per card.
      ...(single ? [el("span", {
        class: "mmc-pill mmc-pill-static",
        title: "What the merged request compiles to — every shot's references and "
             + "keyframes are one pool, so this is asked of the whole timeline at once.",
      }, [el("span", { text: S.singleMode(this.timeline) })])] : []),
      // One call for the whole strip, not one per card: continuity across a cut
      // is only kept by a rewrite that wrote both sides of it.
      refineButton({
        run: () => this.refineAll(),
        label: refined ? "Refine again" : "Refine all",
        className: "mmc-pill mmc-tl-refine",
        title: `Rewrite ${count === 1 ? "the shot" : `all ${count} shots`} into the expanded `
             + `description H3 was trained to read, in one pass so the later shots keep what the `
             + `first establishes. Everything you wrote is kept and expanded. A rewrite is queued `
             + `in place of the card's own prompt, not alongside it.`,
      }),
      // The way back from that one press. Without it, undoing a whole-strip
      // refine means opening every card in turn.
      ...(refined ? [el("button", {
        class: "mmc-pill mmc-tl-unrefine",
        title: "Throw every rewrite away and go back to the prompts you typed. The soundscape "
             + "and score the refiner wrote go with them.",
        onclick: () => this.revertAll(),
      }, [el("span", { text: "Revert all" })])] : []),
      el("div", { class: "mmc-tl-total" }, [
        el("b", { text: `${seconds.toFixed(1)} s` }),
        el("span", { text: single ? `${count} shot${count === 1 ? "" : "s"} · ${frames} frames`
                                  : `${count} segment${count === 1 ? "" : "s"}` }),
      ]),
      // How many sampler passes this queue costs, which is not obvious either
      // way: a strip of cards looks like several small edits, and in one pass it
      // is the whole clip riding on a single denoise.
      el("div", {
        class: "mmc-note",
        title: single
          ? "The whole timeline is generated at once, so the shots cost no more than one clip of the same length."
          : "Each segment is generated separately and they run one after another.",
      }, [
        el("span", { class: "mmc-note-key", text: "cost" }),
        el("span", { text: single ? "1 generation per queue"
                                  : `${count} generation${count === 1 ? "" : "s"} per queue` }),
      ]),
      // The refusals compile.py would raise, said here while the shots are still
      // in front of you. Only in one pass: they are all about things a chained
      // timeline is allowed to have and a single generation is not.
      ...(problem ? [el("div", { class: "mmc-tl-problem" }, [
        el("span", { class: "mmc-note-key", text: "one pass" }),
        el("span", { text: problem }),
      ])] : []),
      // Whatever the last refine had to say — no text encoder is chosen, or it
      // wrote a label nothing backs. Shown on the bar rather than in a card,
      // because the call was about all of them.
      ...(this.refineError ? [el("div", { class: "mmc-warn", text: this.refineError })] : []),
    );
  }

  renderStrip() {
    const parts = [];
    const single = S.isSingle(this.timeline);
    this.timeline.segments.forEach((segment, index) => {
      if (index > 0) parts.push(single ? this.renderCut(index) : this.renderJoin(index));
      parts.push(this.renderCard(segment, index));
    });
    const what = single ? "Shot" : "Segment";
    parts.push(el("button", {
      class: "mmc-tl-add",
      title: this.timeline.segments.length >= S.MAX_SEGMENTS
        ? `A timeline holds at most ${S.MAX_SEGMENTS}.`
        : `Add a ${what.toLowerCase()} to the end`,
      disabled: this.timeline.segments.length >= S.MAX_SEGMENTS || undefined,
      onclick: () => this.add(),
    }, [el("span", { text: "+" }), el("span", { text: what })]));
    this.stripHost.replaceChildren(...parts);
  }

  /**
   * The seam between two segments, and the only control on it.
   *
   * Continuing means segment N starts on segment N-1's last frame, which makes
   * it a keyframe generation — so it cannot also carry references, and the
   * switch is refused rather than silently dropping them.
   */
  /**
   * The same place in the strip, in one-pass mode: a cut, and when it happens.
   *
   * Nothing to switch. A cut inside one generation is a line of the description
   * — `[Shot 3] At 00:09.000, ...` — so what there is to show is the timestamp
   * the compiler will write, which is the one number the shot durations decide
   * that is not visible anywhere else.
   */
  renderCut(index) {
    const { at } = S.cutTimes(this.timeline);
    return el("div", { class: "mmc-tl-seam" }, [
      el("div", {
        class: "mmc-tl-cut",
        title: `Shot ${index + 1} cuts in ${S.shotTime(at[index])} into the clip. `
             + `Write its prompt as the cut — "the camera cuts to…", "the shot transitions to…" — `
             + `and the timestamp is added for you.`,
      }, [el("span", { text: "✂" }), el("span", { text: S.shotTime(at[index]) })]),
    ]);
  }

  renderJoin(index) {
    const segment = this.timeline.segments[index];
    const on = S.continues(segment);
    const blocked = on ? null : S.blockedReason(segment, "continue");

    const sound = S.continuesAudio(segment);
    const soundBlocked = sound ? null : S.blockedReason(segment, "continue_audio");

    // Which earlier segment a live seam inherits from — the previous one unless
    // the seam names another, which is what makes a circular narrative possible:
    // segment 3 can return to segment 1's hallway after an unrelated segment 2.
    const from = S.continueSource(segment, index);

    // Two switches, not one control with three states. The picture and the sound
    // cross a seam independently: a hard cut whose score keeps playing is as
    // ordinary as a match cut that resets the room tone.
    return el("div", { class: "mmc-tl-seam" }, [
      el("button", {
        class: `mmc-tl-join${on ? " on" : ""}`,
        disabled: blocked ? true : undefined,
        title: blocked || (on
          ? `Segment ${index + 1} starts on segment ${from}'s last frame. Click for a hard cut.`
          : `Hard cut into segment ${index + 1}. Click to start it on segment ${index}'s last frame.`),
        onclick: blocked ? undefined : () => { segment.continue = !on; this.commit(); },
      }, [el("span", { text: on ? "↝" : "✂" }), el("span", { text: on ? "continues" : "cut" })]),
      el("button", {
        class: `mmc-tl-join mmc-tl-join-sound${sound ? " on" : ""}`,
        disabled: soundBlocked ? true : undefined,
        title: soundBlocked || (sound
          ? `Segment ${index + 1}'s sound carries on from segment ${from}'s. `
            + `Click to let it start its own.`
          : `Segment ${index + 1} generates its own sound from scratch. `
            + `Click to carry the last ${this.timeline.audio_tail_s}s of segment ${from}'s into it.`),
        onclick: soundBlocked ? undefined : () => { segment.continue_audio = !sound; this.commit(); },
      }, [icon("audio", 13), el("span", { text: sound ? "sound" : "silent seam" })]),
      // Where the seam inherits from. Only on a live seam, and only once there
      // is a choice to make: seam 2 can only continue from segment 1, and a
      // one-option picker would only raise the question it answers.
      ...((on || sound) && index >= 2 ? [el("button", {
        class: `mmc-tl-join mmc-tl-join-from${from !== index ? " on" : ""}`,
        title: `What continues across this seam is segment ${from}'s last `
             + `${on && sound ? "frame and sound" : on ? "frame" : "sound"}. `
             + `Click to inherit from a different earlier segment — a story returning to `
             + `segment 1 after an unrelated shot continues from segment 1.`,
        onclick: (event) => this.pickContinueFrom(event.currentTarget, segment, index),
      }, [el("span", { text: `from #${from}` })])] : []),
    ]);
  }

  /** The seam's source, chosen from every segment before this one. */
  pickContinueFrom(anchor, segment, index) {
    const options = [];
    for (let n = 1; n <= index; n += 1) {
      options.push(n === index ? `segment ${n} — previous` : `segment ${n}`);
    }
    openChoicePopover(anchor, {
      title: `Segment ${index + 1} continues from`,
      options,
      value: options[S.continueSource(segment, index) - 1],
      onPick: (choice) => {
        const n = Number(/\d+/.exec(choice)[0]);
        // The previous segment is the default, so choosing it is choosing to
        // store nothing — an absent key survives reordering with no bookkeeping.
        if (n === index) delete segment.continue_from;
        else segment.continue_from = n;
        this.commit();
      },
    });
  }

  renderCard(segment, index) {
    const single = S.isSingle(this.timeline);
    // In one pass the shot does not snap to the grid on its own — the total
    // does — so the card shows what the user set and the bar shows the truth.
    const frames = framesForSeconds(segment.duration_s);
    const seconds = single ? Number(segment.duration_s) || 0 : secondsForFrames(frames);
    const refs = S.references(segment).length;
    const loras = S.activeLoras(segment).length;
    const typed = (segment.prompt || "").trim();
    const rewrite = segment.refined?.body?.trim();
    const using = rewrite && segment.refined.enabled !== false;
    // The typed sentence is what the user recognises the card by, so it stays
    // the caption and the rewrite is only marked — a paragraph of generated
    // prose on a 160 px card says less about which shot this is, not more. A
    // card refined from nothing falls back to the rewrite, which is then the
    // only description it has.
    const prompt = typed || rewrite || "";

    const meta = [];
    if (refs) meta.push(`${refs} ref${refs === 1 ? "" : "s"}`);
    if (loras) meta.push(`${loras} LoRA${loras === 1 ? "" : "s"}`);
    if (rewrite) meta.push(using ? "refined" : "refined (off)");

    return el("div", {
      class: "mmc-tl-card",
      style: { width: `${cardWidth(seconds)}px` },
      // Double-click anywhere on the card, because "Edit" is a small target and
      // opening a segment is the thing you do most in here.
      ondblclick: () => this.edit(index),
    }, [
      el("div", { class: "mmc-tl-card-head" }, [
        el("span", { class: "mmc-tl-index", text: String(index + 1) }),
        // The off-distribution mark belongs to whatever is actually generated in
        // one go. Chained, that is this card; in one pass it is the whole
        // timeline, and marking every card would say it about the wrong thing.
        el("span", {
          class: `mmc-tl-dur${single || isTrainedLength(frames) ? "" : " off-distribution"}`,
          text: `${segment.duration_s} s`,
          title: single
            ? `${segment.duration_s} s of the one generation — the frame count is the timeline's.`
            : isTrainedLength(frames)
              ? `${frames} frames at 24 fps`
              : `${frames} frames — outside the ~5–15 s the weights were trained on.`,
        }),
        // The mode is a property of the generation, and in one pass there is one
        // of those for the whole timeline — so it moves to the bar.
        ...(single ? [] : [el("span", { class: "mmc-tl-mode", text: S.mode(segment) })]),
      ]),
      // Dimmed while a rewrite stands in for it, the same way the editor dims the
      // box this caption is showing: the card would otherwise read as if the
      // sentence under it were what this shot queues.
      el("div", {
        class: `mmc-tl-card-prompt${prompt ? "" : " empty"}${using && typed ? " superseded" : ""}`,
        text: prompt || "No prompt yet",
        title: using && typed ? "Not queued — this card's rewrite is. Open it to read or revert." : "",
      }),
      ...(meta.length ? [el("div", { class: "mmc-tl-card-meta", text: meta.join(" · ") })] : []),
      el("div", { class: "mmc-tl-card-foot" }, [
        el("button", { class: "mmc-tl-edit", text: "Edit", onclick: () => this.edit(index) }),
        el("button", {
          class: "mmc-ghost", text: "◀", title: "Move earlier",
          disabled: index === 0 || undefined,
          onclick: () => this.move(index, -1),
        }),
        el("button", {
          class: "mmc-ghost", text: "▶", title: "Move later",
          disabled: index === this.timeline.segments.length - 1 || undefined,
          onclick: () => this.move(index, 1),
        }),
        el("button", {
          class: "mmc-ghost", text: "⧉", title: "Duplicate",
          disabled: this.timeline.segments.length >= S.MAX_SEGMENTS || undefined,
          onclick: () => this.duplicate(index),
        }),
        el("button", {
          class: "mmc-asset-x", text: "✕", title: "Remove this segment",
          disabled: this.timeline.segments.length <= 1 || undefined,
          onclick: () => this.remove(index),
        }),
      ]),
    ]);
  }

  // ---- actions ---------------------------------------------------------------

  add() {
    if (this.timeline.segments.length >= S.MAX_SEGMENTS) return;
    this.timeline.segments.push(S.emptySegment());
    this.commit();
  }

  duplicate(index) {
    if (this.timeline.segments.length >= S.MAX_SEGMENTS) return;
    this.timeline.segments.splice(index + 1, 0, S.cloneSegment(this.timeline.segments[index]));
    // Every segment after the insertion moved down one card; a seam naming one
    // of them follows it. Nothing pointed at the clone a moment ago, and a seam
    // naming the original still does.
    S.remapContinueFrom(this.timeline, (n) => (n > index + 1 ? n + 1 : n));
    this.commit();
  }

  remove(index) {
    if (this.timeline.segments.length <= 1) return;
    this.timeline.segments.splice(index, 1);
    // A seam that named the removed segment falls back to the previous one;
    // one naming a later segment follows it up a card.
    S.remapContinueFrom(this.timeline,
      (n) => (n === index + 1 ? null : n > index + 1 ? n - 1 : n));
    this.commit();
  }

  /**
   * Reorder. A segment carries its continuation flag with it, and `syncTimeline`
   * clears it off whatever ends up first — a segment moved to the front has
   * nothing left to continue from. A named seam source follows the card it
   * points at, and `syncTimeline` likewise drops any source the swap carried
   * to or past its own seam.
   */
  move(index, delta) {
    const target = index + delta;
    const segments = this.timeline.segments;
    if (target < 0 || target >= segments.length) return;
    [segments[index], segments[target]] = [segments[target], segments[index]];
    S.remapContinueFrom(this.timeline,
      (n) => (n === index + 1 ? target + 1 : n === target + 1 ? index + 1 : n));
    this.commit();
  }

  /**
   * The LoRA manager, editing the timeline's own list.
   *
   * Handed the checkpoints the segments actually route to rather than one: a
   * global LoRA is patched onto every segment and the segments need not agree,
   * so "idle" here means it lands on none of them, not on the wrong one.
   */
  async openLoras() {
    await openLoras({
      state: this.timeline,
      targets: S.timelineCheckpoints(this.timeline),
      onChange: () => this.commit(),
    });
    this.render();
  }

  /**
   * The two fields a rewrite writes that belong to the piece rather than a shot.
   *
   * Straight into the timeline's own textareas, which are the ones the user is
   * looking at — a refined soundscape hidden inside a card would be invisible
   * and would then disagree with the box above it. An empty `music` is left
   * alone rather than written: the refiner returns one only when the request
   * asked for music, and clearing a score the user typed is not what "the model
   * had nothing to add" means.
   */
  takeAudio(result) {
    // What was in them first, so `revertAll` can put them back. Taken once and
    // then left alone: refining again must not record the last rewrite's prose
    // as the thing the user typed.
    const replaced = this.timeline.refined?.replaced
      ?? { soundscape: this.timeline.soundscape ?? "", music: this.timeline.music ?? "" };

    if (result.soundscape) this.timeline.soundscape = result.soundscape;
    if (result.music) this.timeline.music = result.music;
    this.timeline.refined = {
      ...(this.timeline.refined || {}),
      replaced,
      // Only the reference form has these, and in one pass they describe the one
      // merged generation, so they are the timeline's.
      ...(result.sections && S.isSingle(this.timeline) ? { sections: result.sections } : {}),
    };
    this.soundscapeBox.value = this.timeline.soundscape ?? "";
    this.musicBox.value = this.timeline.music ?? "";
    this.onCommit?.();
  }

  /**
   * Throw away every rewrite in the strip and everything written alongside them.
   *
   * The counterpart of `refineAll`, and the only way back from it: a strip
   * refined in one press was queueing prose in place of every card's own
   * sentence, and undoing that card by card means opening every one of them.
   * The timeline's soundscape, score and reference sections go too — they were
   * written by the same call and describe rewrites that no longer exist.
   */
  revertAll() {
    for (const segment of this.timeline.segments) segment.refined = null;
    this.dropTimelineRewrite();
    this.commit();
  }

  /**
   * Drop what the refiner left on the timeline itself, once no card uses it.
   *
   * Reached from `revertAll` and from a single card's own Revert: the audio
   * fields and the reference analysis belong to the rewrite as a whole, so the
   * moment the last one is gone they are prose describing nothing. A card
   * reverted while others stay refined leaves them exactly as they are.
   */
  dropTimelineRewrite() {
    if (this.timeline.segments.some((segment) => segment.refined?.body)) return;
    const replaced = this.timeline.refined?.replaced;
    if (replaced) {
      this.timeline.soundscape = replaced.soundscape ?? "";
      this.timeline.music = replaced.music ?? "";
      if (this.soundscapeBox) this.soundscapeBox.value = this.timeline.soundscape;
      if (this.musicBox) this.musicBox.value = this.timeline.music;
    }
    this.timeline.refined = null;
  }

  /**
   * Rewrite every card in one call.
   *
   * The point of doing the whole strip at once rather than card by card is that
   * a rewrite of shot 4 can only keep what shot 1 established if the same call
   * wrote both — the look, the people, the light and the speakers carry because
   * the model saw them, not because anything here copied them forward.
   */
  async refineAll() {
    this.refineError = null;
    try {
      const result = await refine({
        kind: "timeline",
        data: JSON.parse(S.serializeTimeline(this.timeline)),
      });
      for (const shot of result.shots ?? []) {
        const segment = this.timeline.segments[shot.index];
        if (!segment || !shot.body) continue;
        segment.refined = {
          body: shot.body,
          // Chained, a segment is its own generation over its own references, so
          // the reference form's analysis sections are the segment's. In one
          // pass there is one merged pool and they go on the timeline instead —
          // which is why the server refuses a chained strip of several reference
          // segments rather than copying one analysis across all of them.
          ...(result.sections && !S.isSingle(this.timeline) ? { sections: result.sections } : {}),
          source: segment.prompt ?? "",
          model: refineModel(),
          enabled: true,
        };
      }
      this.takeAudio(result);
      this.refineError = (result.problems ?? []).join(" · ") || null;
    } catch (error) {
      this.refineError = String(error.message || error);
    }
    this.commit();
  }

  /** The segment editor: the node's own body, over the strip. */
  edit(index) {
    const segment = this.timeline.segments[index];
    const editor = new CreatorEditor({
      state: segment,
      onCommit: () => { this.onCommit?.(); this.renderStrip(); },
      // Both belong to the timeline rather than to one shot: the canvas because
      // the segments are joined, the continuation because it describes the seam
      // in front of this segment and so does not exist for the first one.
      canvasPills: false,
      // The route belongs to the timeline, like the canvas: a clip whose shots
      // ran on different checkpoints per the same setting would not be one
      // setting. Read here, set from the node body's weights control.
      routeOf: () => this.timeline.models?.route ?? "auto",
      // No seam in one pass, so nothing to switch: the shots are cuts inside a
      // single generation and continuity is the model's to keep, not a wiring
      // decision.
      continuePill: index > 0 && !S.isSingle(this.timeline),
      // One card, refined against the whole timeline: the server compiles the
      // strip to build this segment's payload, so the rewrite is written knowing
      // the global prompt, the canvas and whether this shot continues the last.
      refineTarget: () => ({
        kind: "segment",
        index,
        data: JSON.parse(S.serializeTimeline(this.timeline)),
      }),
      // The soundscape and the score describe the piece rather than the shot, so
      // they land on the timeline's own fields where they are visible and
      // editable — not inside the card that happened to be refined.
      onRefined: (result) => this.takeAudio(result),
      // …and go with the last rewrite that was using them. The commit is this
      // callback's own: the editor's fired before it, so what it wrote out still
      // had the timeline's audio fields in it.
      onReverted: () => { this.dropTimelineRewrite(); this.onCommit?.(); },
    });

    const what = S.isSingle(this.timeline) ? "Shot" : "Segment";
    const modal = el("div", { class: "mmc-modal mmc-tl-editor" }, [
      el("div", { class: "mmc-modal-head" }, [
        el("span", { class: "mmc-tab", "aria-selected": "true", text: `${what} ${index + 1}` }),
        el("span", { class: "mmc-tl-editor-sub", text: `of ${this.timeline.segments.length}` }),
        el("button", { class: "mmc-close", text: "✕", title: "Back to the timeline", onclick: () => done() }),
      ]),
      el("div", { class: "mmc-tl-editor-body" }, [editor.root]),
    ]);

    const overlay = el("div", {
      class: "mmc-overlay",
      onpointerdown: (event) => { if (event.target === overlay) done(); },
    }, [modal]);

    const unmount = mountOverlay(overlay, () => done());
    const done = () => { unmount(); this.render(); };
  }
}

/** After-generate modes, in the order ComfyUI lists them. */

/**
 * The Timeline node's body in the graph: the piece at a glance, the way in, and
 * the sampler settings.
 *
 * The strip itself lives in the modal — it needs room the node does not have,
 * and drawing an editable one here would be a second implementation to keep in
 * step. What the node shows is the global prompt, the segments at their real
 * relative lengths, and the numbers.
 *
 * The sampler widgets are ComfyUI's own, hidden and re-drawn as pills. This node
 * owns the sampler because it writes the KSampler into the graph, but that is no
 * reason for half the node to be stock widgets and half of it to be this. The
 * widgets still hold the values — they are what `graphToPrompt` reads — so the
 * pills only read and write `widget.value`, exactly as the JSON blob does.
 */
export class TimelineBody {
  /** `preStage` is the pre-stage pill's wiring — see minimax_creator.js. */
  constructor({ read, write, widgets = {}, onWidgetChange, nodeId, preStage = null }) {
    this.read = read;
    this.write = write;
    this.widgets = widgets;
    this.onWidgetChange = onWidgetChange;
    this.nodeId = nodeId;
    this.preStage = preStage;
    this.timeline = S.parseTimeline(read());

    // The same stage the Creator has, showing the same thing: a timeline is one
    // clip, and what it is making is one picture whatever the strip looks like.
    // attach() floats it beside the node in a Satellite; it never mounts here.
    this.root = el("div", { class: "mmc-root" });
    this.stage = new Stage({
      nodeId,
      // View-only: a timeline's references live on its segments, so a pick from
      // here would have no card to land on. The Creator attaches; this browses.
      onGallery: () => openPicker({
        kinds: ["renders"],
        kind: "renders",
        viewOnly: true,
        capacity: () => ({ used: 0, max: 0, filesLeft: 0 }),
      }),
    });
    loadCatalog(() => this.adoptWeights());
    this.render();
  }

  destroy() {
    this.stage?.destroy();
  }

  /** See `CreatorEditor.adoptWeights` — same rescue, same reason. */
  adoptWeights() {
    if (S.guessModels(this.timeline.models, catalogFiles())) this.commit();
    else this.render();
  }

  /** Re-read the widget. Loading a saved workflow assigns widget values after
   *  the node is created, so the body built in `nodeCreated` saw the default. */
  reload() {
    this.timeline = S.parseTimeline(this.read());
    this.render();
  }

  commit() {
    S.syncTimeline(this.timeline);
    // Removing or disabling the turbo LoRA anywhere — the global stack's
    // manager included — is switching turbo off, and the sampler row has to
    // come back before the blob is written with `on` still in it.
    Turbo.sync(this.timeline, this.widgetIO());
    this.write(S.serializeTimeline(this.timeline));
    this.render();
  }

  open() {
    openTimeline({ timeline: this.timeline, onCommit: () => this.commit() });
  }

  value(name, fallback) {
    const widget = this.widgets[name];
    return widget ? widget.value : fallback;
  }

  /** The sampler widgets as turbo.js wants them: write-through without the
   *  re-render, because everything that uses this commits — and renders — once
   *  at the end rather than three times along the way. */
  widgetIO() {
    return {
      value: (name, fallback) => this.value(name, fallback),
      set: (name, value) => {
        const widget = this.widgets[name];
        if (!widget) return;
        widget.value = value;
        widget.callback?.(value);
        this.onWidgetChange?.();
      },
    };
  }

  /** Write through to the real widget, callback included — some of them (the
   *  seed's after-generate control) hang behaviour off it. */
  set(name, value) {
    const widget = this.widgets[name];
    if (!widget) return;
    widget.value = value;
    widget.callback?.(value);
    this.onWidgetChange?.();
    this.render();
  }

  render() {
    // Same order as the Creator: what you are asking for, then how it is run.
    // The picture is beside the node, in the satellite.
    this.root.replaceChildren(this.renderPanel(), this.renderSampling());
  }

  renderPanel() {
    const segments = this.timeline.segments;
    const single = S.isSingle(this.timeline);
    const seconds = S.timelineSeconds(this.timeline);
    const [width, height] = resolveCanvas(
      ASPECT_PRESETS.find(([label]) => label === this.timeline.aspect)?.[1] ?? 16 / 9,
      this.timeline.short_edge);
    const prompt = (this.timeline.prompt || "").trim();
    const globalLoras = S.activeGlobalLoras(this.timeline).length;
    const audio = [
      ...(this.timeline.soundscape?.trim() ? ["soundscape"] : []),
      ...(this.timeline.music?.trim() ? ["music"] : []),
    ];

    return el("div", { class: "mmc-panel mmc-tl-summary" }, [
      el("div", {
        class: `mmc-tl-summary-prompt${prompt ? "" : " empty"}`,
        text: prompt || (single
          ? "No global prompt yet — the standing description that opens Shot 1."
          : "No global prompt yet — the standing description every segment inherits."),
        onclick: () => this.open(),
      }),
      // The one picture of the timeline the node has room for: blocks at their
      // real relative lengths, so a 10-second shot is visibly twice a 5.
      el("div", { class: "mmc-tl-lane", onclick: () => this.open() }, segments.map((segment, index) => {
        // In one pass there is no seam to continue across — the lane is the shot
        // list of one generation, so every block reads the same way.
        const continues = !single && S.continues(segment);
        const { at } = S.cutTimes(this.timeline);
        return el("div", {
          class: `mmc-tl-tick${continues ? " on" : ""}`,
          style: { flexGrow: String(Math.max(1, segment.duration_s)) },
          title: single
            ? `Shot ${index + 1} · ${segment.duration_s} s`
              + (index ? ` · cuts in at ${S.shotTime(at[index])}` : " · opens the clip")
            : `Segment ${index + 1} · ${segment.duration_s} s · ${S.mode(segment)}`
              + (continues ? ` · continues from segment ${S.continueSource(segment, index)}` : " · hard cut"),
        }, [
          ...(continues ? [icon("link", 13)] : []),
          el("span", { class: "mmc-tl-tick-n", text: String(index + 1) }),
          el("span", { class: "mmc-tl-tick-s", text: `${segment.duration_s}s` }),
        ]);
      })),
      el("div", { class: "mmc-pills" }, [
        // The render mode leads, because it is the one thing about this node
        // that changes what all the other numbers mean.
        el("span", {
          class: "mmc-pill mmc-pill-static",
          title: single
            ? "One generation: the segments are the shots of a single description, cut times and all."
            : "One generation per segment, joined end to end.",
        }, [
          icon("timeline", 16),
          el("span", { text: single ? "one pass" : "chained" }),
          el("span", {
            class: "mmc-pill-sub",
            text: single ? `${segments.length} shot${segments.length === 1 ? "" : "s"}`
                         : `${segments.length} segment${segments.length === 1 ? "" : "s"}`,
          }),
        ]),
        el("span", { class: "mmc-pill mmc-pill-static", title: "The finished clip's length at 24 fps" }, [
          icon("clock", 16),
          el("span", { text: `${seconds.toFixed(1)} s` }),
        ]),
        el("span", {
          class: "mmc-pill mmc-pill-static",
          title: single
            ? "The canvas the one generation runs at."
            : "Shared by every segment — they are joined end to end and have to match.",
        }, [
          el("span", { text: this.timeline.aspect }),
          el("span", { class: "mmc-pill-sub", text: `${width} × ${height}` }),
        ]),
        // Only when there are any: an empty pill would say the timeline has a
        // LoRA feature, which is the modal's job to say, not the node's.
        ...(globalLoras ? [el("span", {
          class: "mmc-pill mmc-pill-static",
          title: single
            ? "Patched onto the one generation, in front of whatever the shots add."
            : "Patched onto every segment, in front of whatever that segment adds.",
        }, [
          icon("effect", 16),
          el("span", { text: `${globalLoras} LoRA${globalLoras === 1 ? "" : "s"}` }),
        ])] : []),
        ...(audio.length ? [el("span", {
          class: "mmc-pill mmc-pill-static",
          title: "The Context-IR audio fields this timeline sets for every segment.",
        }, [icon("audio", 16), el("span", { text: audio.join(" · ") })])] : []),
        el("button", {
          class: "mmc-tl-open",
          title: "Open the timeline: the global prompt, the segments, and what happens between them",
          onclick: () => this.open(),
        }, [icon("sliders", 16), el("span", { text: "Edit timeline" })]),
        ...(this.preStage ? [this.renderPreStagePill()] : []),
      ]),
    ]);
  }

  /** Same pill as the Creator's — see `CreatorEditor.renderPreStagePill`. */
  renderPreStagePill() {
    const on = this.preStage.active();
    return el("button", {
      class: `mmc-pill mmc-prestage-toggle${on ? " on" : ""}`,
      title: on
        ? "The pre-stage node on the left generates stills for this timeline — the opening "
          + "frame, the closing frame, references. Click to remove it."
        : "Add a pre-stage: an image node (Krea 2 / Ideogram 4) at this node's left edge whose "
          + "stills land on the timeline's shots with one click.",
      onclick: () => { this.preStage.toggle(); this.render(); },
    }, [icon("image", 16), el("span", { text: "pre-stage" })]);
  }

  /**
   * A finished pre-stage still, pushed into the timeline by the neighbour's
   * result chips. The roles land where one pass would put them — a start frame
   * opens shot 1, an end frame closes the last shot, a reference joins shot 1 —
   * under each shot's own capacity and exclusivity rules. Returns a refusal
   * message, or null on success.
   */
  attachFromPreStage({ role, filename }) {
    const shots = this.timeline.segments;
    const index = role === "last_frame" ? shots.length - 1 : 0;
    const segment = shots[index];
    const where = `segment ${index + 1}`;
    if (role === "reference") {
      const blocked = S.blockedReason(segment, "reference");
      if (blocked) return `${where}: ${blocked}`;
      const { used, max, filesLeft } = S.capacity(segment, "image");
      if (used >= max || filesLeft <= 0) return `${where}: no image slots left (${used}/${max} used).`;
      segment.assets.push({
        handle: S.nextHandle(segment, "image"),
        kind: "image", role: "reference", filename, ref_size: "max",
      });
      this.commit();
      return null;
    }
    const blocked = S.blockedReason(segment, role);
    if (blocked) return `${where}: ${blocked}`;
    const existing = S.frameAsset(segment, role);
    if (existing) segment.assets = segment.assets.filter((a) => a.handle !== existing.handle);
    segment.assets.push({
      handle: S.nextHandle(segment, "image"),
      kind: "image", role, filename,
    });
    this.commit();
    return null;
  }

  /**
   * The sampler row, shared with the Creator node — see `sampling.js`. Both
   * nodes own their sampler and declare the same widgets, so neither draws its
   * own version of this.
   */
  renderSampling() {
    return samplingBar({
      widgets: this.widgets,
      value: (name, fallback) => this.value(name, fallback),
      set: (name, value) => this.set(name, value),
      // Chained is many generations; one pass is one, and the seed and step
      // counts mean different things in each.
      perSegment: !S.isSingle(this.timeline),
      // The turbo switch, on the timeline's global stack: a speed-up belongs to
      // the run, which is the whole reason the global stack exists.
      turbo: Turbo.turboPills({
        container: this.timeline,
        ...this.widgetIO(),
        onCommit: () => this.commit(),
      }),
      // A chained timeline legitimately runs some shots on one checkpoint and
      // some on the other, so the pill is asked about the set rather than
      // about one — a Ref2VA it never reaches for is not missing.
      trailing: [weightsPill({
        models: this.timeline.models,
        checkpoints: S.timelineCheckpoints(this.timeline),
        onChange: () => this.commit(),
        turbo: { container: this.timeline, widgetIO: this.widgetIO() },
      })],
    });
  }
}
