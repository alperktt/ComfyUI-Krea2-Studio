// The two canvas popovers — aspect ratio and short edge.
//
// They live here rather than on CreatorEditor because the canvas is a property
// of a *generation* in the Creator node and a property of the *timeline* in the
// Timeline node, and both need the same two controls over the same two fields.

import { el, icon, dismissable, placeNear } from "./dom.js";
import { ASPECT_PRESETS, MIN_SHORT_EDGE, MAX_SHORT_EDGE, NATIVE_SHORT_EDGE, CANVAS_MULTIPLE } from "./canvas.js";

/**
 * A −/value/+ pill. The same shape as the duration control, because a number
 * you nudge should look the same everywhere in the node.
 *
 * @param {object} spec
 * @param {number} spec.value
 * @param {(value:number) => void} spec.onChange
 * @param {string} [spec.iconName]   drawn between the two steppers
 * @param {(value:number) => string} [spec.format]
 */
export function stepperPill({ value, onChange, min = -Infinity, max = Infinity, step = 1,
                              iconName, format = String, title, width = "34px" }) {
  const clamp = (next) => Math.min(max, Math.max(min, Math.round(next * 1e6) / 1e6));
  const arrow = (label, delta) => el("button", {
    class: "mmc-step", text: label,
    disabled: clamp(value + delta) === value || undefined,
    onclick: () => onChange(clamp(value + delta)),
  });
  return el("div", { class: "mmc-pill mmc-pill-group", title }, [
    arrow("−", -step),
    ...(iconName ? [icon(iconName, 16)] : []),
    el("span", { text: format(value), style: { minWidth: width, textAlign: "center" } }),
    arrow("+", step),
  ]);
}

/**
 * A pill that opens a list of choices. Used for anything whose options come
 * from the backend — samplers, schedulers — where there is nothing to draw but
 * the name.
 */
export function openChoicePopover(anchor, { title, options, value, onPick }) {
  const pop = el("div", { class: "mmc-pop mmc-pop-scroll" },
    title ? [el("div", { class: "mmc-pop-title", text: title })] : []);
  for (const option of options) {
    pop.appendChild(el("button", {
      class: "mmc-opt",
      "aria-checked": option === value,
      onclick: () => { close(); onPick(option); },
    }, [
      el("span", { class: "mmc-opt-label" }, [el("span", { text: option })]),
      el("span", { class: "mmc-radio" }),
    ]));
  }
  document.body.appendChild(pop);
  placeNear(pop, anchor);
  const close = dismissable(pop);
  pop.querySelector('[aria-checked="true"]')?.scrollIntoView({ block: "center" });
}

/** A frame drawn at the ratio itself, so portrait and landscape are legible
 *  without reading the numbers. Sized to fit `long` on its long edge; the box
 *  is square, which keeps every glyph on the same baseline and left edge.
 *
 *  The pill wears the same glyph one size down, matching the 16px icon on the
 *  resolution pill beside it — the chip is what you look at while the list is
 *  closed, so telling 9:16 from 16:9 there is worth more than in the list. */
export function aspectGlyph(ratio, long = 18) {
  const width = ratio >= 1 ? long : long * ratio;
  const height = ratio >= 1 ? long / ratio : long;
  return el("span", { class: "mmc-aspect-glyph", style: { width: `${long}px`, height: `${long}px` } }, [
    el("span", { style: { width: `${width}px`, height: `${height}px` } }),
  ]);
}

/** The glyph as a pill wears it. */
export const PILL_GLYPH = 16;

/**
 * @param {HTMLElement} anchor  the pill to hang the popover off
 * @param {object} target       anything with an `aspect` field — a state or a timeline
 * @param {() => void} commit   called once, after a choice
 */
export function openAspectPopover(anchor, target, commit) {
  const pop = el("div", { class: "mmc-pop" }, [el("div", { class: "mmc-pop-title", text: "Aspect Ratio" })]);
  for (const [label, ratio] of ASPECT_PRESETS) {
    pop.appendChild(el("button", {
      class: "mmc-opt",
      "aria-checked": target.aspect === label,
      onclick: () => { target.aspect = label; close(); commit(); },
    }, [
      el("span", { class: "mmc-opt-label" }, [aspectGlyph(ratio), el("span", { text: label })]),
      el("span", { class: "mmc-radio" }),
    ]));
  }
  document.body.appendChild(pop);
  placeNear(pop, anchor);
  const close = dismissable(pop);
}

/**
 * The short-edge control, shared by every node that has one.
 *
 * The one rule that matters here: *nothing inside this may change size while
 * the thumb is down*. A range input maps the pointer's x onto the width of its
 * own track, so a readout that grows by a digit, or a note that wraps onto a
 * second line and widens the popover, moves the track out from under the
 * pointer mid-drag and the value jumps. That is why the readout is tabular, the
 * note holds two lines whatever it says, and the popover is a fixed width
 * rather than one fitted to its text.
 *
 * The steppers are the other half of the answer: on the pre-stage's 512–2048
 * range a step is two pixels of track, which no hand can hit. Arrow keys do the
 * same thing once the slider has focus; the buttons say so out loud.
 *
 * @param {object} spec
 * @param {number} spec.value
 * @param {number} [spec.mark]        a value worth marking on the track
 * @param {string} [spec.markLabel]   what it is — "native", "default"
 * @param {(edge:number) => void} spec.apply    write the value onto the target
 * @param {() => {size:string, note:string, warn?:boolean}} spec.describe
 * @param {() => void} spec.commit    called on release, not on every pixel
 */
export function edgeSlider({ min, max, step, value, mark, markLabel, apply, describe, commit }) {
  const edge = el("span", { class: "mmc-edge" });
  const size = el("span");
  const note = el("div", { class: "mmc-native" });
  const read = el("div", { class: "mmc-slider-read" }, [
    el("span", {}, [edge, el("span", { class: "mmc-edge-unit", text: "px" })]),
    size,
  ]);
  const slider = el("input", {
    type: "range", min, max, step, value,
    "aria-label": "Short edge in pixels",
    // The graph canvas reads a pointerdown anywhere on the node as the start of
    // a node drag, and would carry the whole node off under the thumb.
    onpointerdown: (event) => event.stopPropagation(),
  });

  const snap = (n) => Math.min(max, Math.max(min, Math.round((n - min) / step) * step + min));

  const paint = () => {
    const current = Number(slider.value);
    edge.textContent = String(current);
    const shown = describe();
    size.textContent = shown.size;
    note.textContent = shown.note;
    note.classList.toggle("over", Boolean(shown.warn));
    down.disabled = current <= min;
    up.disabled = current >= max;
    marker?.classList.toggle("on", current === mark);
  };

  /** Set from a button — the slider itself feeds `input` instead. */
  const set = (next) => {
    slider.value = String(snap(next));
    apply(Number(slider.value));
    paint();
    commit();
  };

  const stepper = (label, delta) => el("button", {
    class: "mmc-step", text: label,
    title: `${delta < 0 ? "Down" : "Up"} ${step} px`,
    "aria-label": `${delta < 0 ? "Smaller" : "Larger"} by ${step} pixels`,
    onclick: () => set(Number(slider.value) + delta * step),
  });
  const down = stepper("−", -1);
  const up = stepper("+", 1);

  const marker = mark > min && mark < max
    ? el("button", {
        class: "mmc-slider-mark",
        title: `${markLabel} — ${mark} px`,
        onclick: () => set(mark),
      }, [el("span", { text: markLabel })])
    : null;
  // A custom property has to go through setProperty; Object.assign drops it.
  marker?.style.setProperty("--p", String((mark - min) / (max - min)));

  slider.addEventListener("input", () => { apply(Number(slider.value)); paint(); });
  slider.addEventListener("change", () => commit());

  // A hand-edited creator_data can hold an edge off the step grid; the input
  // silently snaps it, and the readout would otherwise disagree with the size
  // beside it. Written back without committing — the next change carries it.
  if (Number(slider.value) !== value) apply(Number(slider.value));

  const body = el("div", { class: "mmc-slider-body" }, [
    read,
    el("div", { class: "mmc-slider-row" }, [
      down,
      el("div", { class: "mmc-slider-track" }, [slider, marker]),
      up,
    ]),
    note,
  ]);
  paint();
  return body;
}

/**
 * @param {HTMLElement} anchor
 * @param {object} target             anything with a `short_edge` field
 * @param {() => {width:number, height:number}} geometry  recomputed as the slider moves
 * @param {() => void} commit         called on release, not on every pixel
 */
export function openResolutionPopover(anchor, target, geometry, commit) {
  const body = edgeSlider({
    min: MIN_SHORT_EDGE, max: MAX_SHORT_EDGE, step: CANVAS_MULTIPLE,
    value: target.short_edge, mark: NATIVE_SHORT_EDGE, markLabel: "native",
    apply: (edge) => { target.short_edge = edge; },
    describe: () => {
      const { width, height } = geometry();
      const over = target.short_edge > NATIVE_SHORT_EDGE;
      return {
        size: `${width} × ${height}`,
        warn: over,
        note: over
          ? `Above the trained ${NATIVE_SHORT_EDGE} px short edge — off-distribution, not just slower.`
          : target.short_edge === NATIVE_SHORT_EDGE
            ? "Native. What the open weights were trained at."
            : `${(NATIVE_SHORT_EDGE / target.short_edge).toFixed(1)}× smaller short edge than native — faster, softer.`,
      };
    },
    commit,
  });
  const pop = el("div", { class: "mmc-pop mmc-slider" }, [body]);
  document.body.appendChild(pop);
  placeNear(pop, anchor);
  dismissable(pop);
}
