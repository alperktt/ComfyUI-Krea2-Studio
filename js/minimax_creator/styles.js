// Injected once. Everything is scoped under .mmc-root (or .mmc-overlay for the
// modal, which portals to document.body) so nothing leaks into the graph canvas.

const CSS = `
/* Tokens live on :root, not .mmc-root: popovers and the picker portal to
   document.body, so anything scoped to the node body would leave them
   resolving to nothing. */
:root {
  --mmc-bg: #0e0e0e;
  --mmc-surface: #1c1c1c;
  --mmc-surface-2: #262626;
  --mmc-surface-3: #2f2f2f;
  --mmc-line: rgba(255,255,255,.09);
  --mmc-text: #ededed;
  --mmc-dim: #8b8b8b;
  --mmc-off: #565656;
  --mmc-accent: #f0a63c;
  --mmc-blue: #2f7bf6;
  /* Reference identity hues: one per attached asset, worn by its thumbnail
     ring, its handle in the bar, and its chip in the prompt, so a chip in the
     sentence can be matched to a picture without reading. Equal perceived
     lightness on the dark surfaces; the amber zone is skipped so an asset
     never masquerades as the accent. Index comes from state.tagIndex(). */
  --mmc-tag-0: #5cb8f0;
  --mmc-tag-1: #63c98e;
  --mmc-tag-2: #9d95f5;
  --mmc-tag-3: #f07da0;
  --mmc-tag-4: #45c4c0;
  --mmc-tag-5: #f0906b;
  --mmc-tag-6: #d57de8;
  --mmc-tag-7: #a8c858;
}

/* Setting --tag is all these do; components read it with an accent fallback,
   so an untagged element (a LoRA row, a dangling handle) keeps today's look. */
.mmc-tag-0 { --tag: var(--mmc-tag-0); }
.mmc-tag-1 { --tag: var(--mmc-tag-1); }
.mmc-tag-2 { --tag: var(--mmc-tag-2); }
.mmc-tag-3 { --tag: var(--mmc-tag-3); }
.mmc-tag-4 { --tag: var(--mmc-tag-4); }
.mmc-tag-5 { --tag: var(--mmc-tag-5); }
.mmc-tag-6 { --tag: var(--mmc-tag-6); }
.mmc-tag-7 { --tag: var(--mmc-tag-7); }

.mmc-root {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
  color: var(--mmc-text);
  display: flex; flex-direction: column; gap: 10px;
  padding: 12px; box-sizing: border-box; height: 100%; overflow: hidden;
}

/* The pre-stage's outer body. It holds whichever editor the architecture calls
   for and is swapped when that changes, so it has to be the full height the DOM
   widget gave it — the editor inside is the `.mmc-root` doing the layout. */
.mmc-prestage-host { display: flex; flex-direction: column; height: 100%; min-height: 0; }
.mmc-prestage-host > * { flex: 1 1 auto; min-height: 0; }

/* --- the satellite and its stage ------------------------------------------ */
/*
 * The picture: the preview while it samples, the finished video after, the error
 * if there was one — and nothing whatsoever before any of that. It floats in a
 * satellite card beside the node (satellite.js), which sets translate+scale from
 * the node's graph position every frame — so inside here one CSS px is one graph
 * unit, and the card's height is the node's. Width comes from the picture: the
 * media keeps its own aspect at full card height, so a portrait render makes a
 * portrait card.
 */
.mmc-satellite {
  position: fixed; left: 0; top: 0; z-index: 100;
  transform-origin: 0 0; display: none;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
  color: var(--mmc-text);
}
.mmc-satellite.showing { display: block; }

.mmc-stage {
  /* The card's own floor matters when there is no media to size it — a failed
     render is a chip of text in an otherwise empty box. */
  position: relative; height: 100%; min-width: 240px;
  flex-direction: column; min-height: 0;
  border-radius: 16px; overflow: hidden;
  background: #000; border: 1px solid var(--mmc-line);
  box-shadow: 0 8px 30px rgba(0,0,0,.45);
}
/* The same running halo Comfy paints around the executing node: litegraph's
   "running" stroke style is a 3px line centered on a path 6px outside the node,
   covering 4.5–7.5px out. outline-offset 4.5px puts our 3px outline over the
   same band, so node and card keep the same silhouette while it runs — without
   it the card read as shorter than the node for the whole render. Black, not
   the node's status green: the card is not reporting progress, only holding
   its edge. */
.mmc-stage[data-state="sampling"] { outline: 3px solid #000; outline-offset: 4.5px; }
.mmc-stage-media { flex: 1; min-height: 0; display: flex; }
.mmc-stage-img, .mmc-stage-video {
  height: 100%; width: auto; min-height: 0; object-fit: contain;
  /* Until the media reports its size the card would be a sliver; until it is
     absurdly wide it may be as wide as it likes. Both bounds in graph units. */
  min-width: 240px; max-width: 1200px;
  display: block; background: #000;
}

/* Progress, as a rule along the bottom edge of the picture rather than a bar of
   its own. The step count is already overlaid; a second reading of the same
   number in a different shape would be decoration. */
.mmc-stage-rule {
  position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
  background: var(--mmc-accent);
  transform-origin: left center; transform: scaleX(0);
  opacity: 0; transition: transform .3s linear, opacity .2s ease;
  pointer-events: none;
}
/* Over the picture, not under it: a caption row would be height the picture
   could have had. pointer-events off so it never eats a click meant for the
   video's own transport. */
.mmc-stage-readout {
  position: absolute; left: 0; right: 0; bottom: 0;
  display: flex; justify-content: space-between; gap: 10px;
  padding: 24px 10px 8px; pointer-events: none;
  font-size: 11px; font-variant-numeric: tabular-nums;
  background: linear-gradient(transparent, rgba(0,0,0,.72));
}
.mmc-stage-readout:empty { display: none; }
.mmc-stage-chip { color: #ededed; text-shadow: 0 1px 3px rgba(0,0,0,.8); }
.mmc-stage-chip.warn { color: #e0743c; }
.mmc-stage[data-state="sampling"] .mmc-stage-chip:first-child { color: var(--mmc-accent); }
/* The readout swallows the pointer so the finished video's controls stay
   reachable under it; its one real button opts back in. */
.mmc-stage-gallery {
  pointer-events: auto; cursor: pointer; font: inherit;
  background: rgba(0,0,0,.55); border: 1px solid var(--mmc-line);
  border-radius: 999px; padding: 3px 12px;
}
.mmc-stage-gallery:hover { border-color: #7a7a7a; }

/* --- the weights control -------------------------------------------------- */
/* A required file nobody has picked. The same warm orange the resolution slider
   uses past 768 and for the same reason: it is a fact about the render, said
   before you queue instead of after. */
.mmc-weights.missing { border-color: rgba(224,116,60,.45); color: #e0743c; }
.mmc-weights.missing:hover:not(:disabled) { border-color: rgba(224,116,60,.75); }
/* Wider with a device column: "cuda:0" beside a folder-qualified filename needs
   the room, and a popover that ellipsises both tells you neither. */
.mmc-weights-pop { width: 380px; padding: 8px; }
.mmc-weight-row {
  display: flex; align-items: center; gap: 8px; width: 100%;
  padding: 3px 4px; box-sizing: border-box;
}
.mmc-weight-name { flex: none; color: var(--mmc-dim); font-size: 12px; width: 112px; }
.mmc-weight-file, .mmc-weight-device {
  background: none; border: 0; border-radius: 8px; padding: 5px 8px;
  color: var(--mmc-text); font-family: inherit; font-size: 13px;
  text-align: left; cursor: pointer;
}
.mmc-weight-file:hover, .mmc-weight-device:hover { background: var(--mmc-surface-2); }
/* Names run to a folder-qualified minimax/h3_ref2va_fp8.safetensors, and the end
   of one is the part that identifies it — so this ellipsises from the *left*
   and keeps the filename rather than the folder. */
.mmc-weight-file {
  flex: 1; min-width: 0; overflow: hidden; white-space: nowrap;
  text-overflow: ellipsis; direction: rtl;
}
.mmc-weight-file.empty { color: var(--mmc-off); direction: ltr; }
.mmc-weight-row.missing .mmc-weight-file { color: #e0743c; }
/* The device this field's weights load on. Quiet at "auto", which is the answer
   on every single-GPU machine and most multi-GPU ones; lit once it is a decision
   somebody made, because which card a thing is on is worth seeing at a glance. */
.mmc-weight-device {
  flex: none; width: 72px; text-align: center; font-size: 11px;
  color: var(--mmc-off); border: 1px solid transparent;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.mmc-weight-device.pinned {
  color: var(--mmc-blue); border-color: rgba(47,123,246,.35);
}
/* A standing route, which is a decision rather than a default — and the reason
   the checkpoint below it may be greyed out. */
.mmc-weight-file.forced { color: var(--mmc-accent); }
/* A checkpoint the route has taken out of play. Still listed, so the setting is
   not thrown away, but visibly out of the run — the same treatment an idle LoRA
   gets on the asset row. */
.mmc-weight-row.idle { opacity: .45; }
/* A route that will be refused: forcing FL2VA on a generation with references.
   Said on the badge rather than at queue time. */
.mmc-mode.bad { color: #e0743c; border-color: rgba(224,116,60,.45); }
.mmc-mode.bad b, .mmc-mode.bad .mmc-pin { color: inherit; }

/* --- tool rail ------------------------------------------------------------ */
/* Every icon comes from ICONS, and every path in there is drawn rather than
   filled. Set once, before any component rule, because forgetting it renders a
   stroke-only path as a solid black blob — which is what a missing per-component
   rule looks like, not a missing icon. Components still override the size and
   weight; equal specificity, so the later rule wins. */
.mmc-root svg, .mmc-overlay svg, .mmc-pop svg {
  fill: none; stroke: currentColor; stroke-width: 1.6;
  stroke-linecap: round; stroke-linejoin: round;
}

.mmc-rail { display: flex; gap: 10px; flex-wrap: wrap; }
.mmc-tool {
  display: flex; flex-direction: column; align-items: center; gap: 6px;
  background: none; border: 0; padding: 0; cursor: pointer;
  color: var(--mmc-dim); font-size: 12px; font-family: inherit;
}
.mmc-tool-icon {
  width: 56px; height: 56px; border-radius: 14px;
  background: var(--mmc-surface-2); border: 1px solid var(--mmc-line);
  display: flex; align-items: center; justify-content: center;
  transition: background .12s ease;
}
.mmc-tool:hover:not(:disabled) .mmc-tool-icon { background: var(--mmc-surface-3); }
.mmc-tool:hover:not(:disabled) { color: var(--mmc-text); }
.mmc-tool:disabled { cursor: not-allowed; color: var(--mmc-off); }
.mmc-tool:disabled .mmc-tool-icon { opacity: .45; }
.mmc-tool svg { width: 22px; height: 22px; stroke: currentColor; fill: none;
  stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }

/* --- attached assets ------------------------------------------------------ */
.mmc-assets { display: flex; gap: 8px; flex-wrap: wrap; }
.mmc-asset {
  display: flex; align-items: center; gap: 8px; padding: 4px 8px 4px 4px;
  background: var(--mmc-surface-2); border: 1px solid var(--mmc-line);
  border-radius: 10px; font-size: 12px;
}
.mmc-asset-thumb {
  width: 30px; height: 30px; border-radius: 7px; object-fit: cover;
  background: var(--mmc-surface-3); display: flex; align-items: center; justify-content: center;
  color: var(--mmc-dim); flex: none;
  /* The identity ring: paints the asset's hue onto the actual picture, which is
     what the same-hued chip in the prompt points back to. Transparent when the
     row carries no tag (LoRA chips share this class). */
  box-shadow: 0 0 0 2px var(--tag, transparent);
}
.mmc-asset-handle { color: var(--tag, var(--mmc-accent)); font-weight: 500; }
.mmc-asset-role { color: var(--mmc-dim); }
.mmc-asset-x {
  background: none; border: 0; color: var(--mmc-off); cursor: pointer;
  font-size: 15px; line-height: 1; padding: 2px 3px; font-family: inherit;
}
.mmc-asset-x:hover { color: var(--mmc-text); }
/* A LoRA set to the checkpoint this graph does not route to. Still listed —
   removing it on a mode change would throw the setting away — but visibly
   out of the run. */
.mmc-asset.idle { opacity: .5; }
.mmc-asset.idle .mmc-asset-handle { color: var(--mmc-dim); }
.mmc-lora-block { display: flex; flex-direction: column; gap: 6px; }
/* What the LoRAs add to the front of the prompt. Not a warning — it is working
   as intended — but it has to be readable, because the prompt box does not
   show it. */
.mmc-note {
  display: flex; gap: 8px; font-size: 11px; color: var(--mmc-dim); line-height: 1.4;
}
.mmc-note-key {
  color: var(--mmc-off); letter-spacing: .06em; text-transform: uppercase;
  font-size: 10px; padding-top: 1px; flex: none;
}

/* --- prompt + pills ------------------------------------------------------- */
.mmc-panel {
  background: var(--mmc-surface); border: 1px solid var(--mmc-line);
  border-radius: 20px; padding: 14px; display: flex; flex-direction: column;
  gap: 12px; flex: 1; min-height: 0;
}
/* contenteditable, not a textarea: @references are atomic chips, and a textarea
   can only hold flat text. white-space: pre-wrap so the literal "\n" the box
   inserts on Enter renders as a line break. */
.mmc-prompt {
  flex: 1; min-height: 56px; background: none; border: 0; outline: none;
  color: var(--mmc-text); font-family: inherit; font-size: 15px; line-height: 1.6;
  white-space: pre-wrap; word-break: break-word; overflow-y: auto;
}
.mmc-prompt:empty::before {
  content: attr(data-placeholder); color: #6a6a6a; pointer-events: none;
}
/* A rewrite replaces this text rather than joining it, so while one is on the
   box is holding a draft, not the prompt. Dimmed rather than disabled: it is
   still where the next rewrite comes from. */
.mmc-prompt.superseded { opacity: .42; }
.mmc-prompt.superseded:focus { opacity: .72; }
/* .mmc-ref, not .mmc-chip: the refiner's language chips own that name, and the
   two rules fighting over it is what once turned these gray. */
.mmc-ref {
  display: inline-block; padding: 1px 7px; margin: 0 1px; border-radius: 7px;
  background: color-mix(in srgb, var(--tag, var(--mmc-accent)) 14%, transparent);
  color: var(--tag, var(--mmc-accent));
  font-size: .92em; white-space: nowrap; user-select: all;
}

/* --- @ mention menu ------------------------------------------------------- */
.mmc-mention {
  position: fixed; z-index: 1350; width: 330px; max-height: 300px; overflow-y: auto;
  background: #212121; border: 1px solid var(--mmc-line); border-radius: 14px;
  padding: 6px; box-shadow: 0 20px 50px rgba(0,0,0,.65);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
}
.mmc-mention-head {
  color: #7d7d7d; font-size: 10px; letter-spacing: .09em; text-transform: uppercase;
  padding: 10px 10px 6px;
}
/* min-width:0 all the way down: a flex item defaults to min-content width, so
   without it a 90-character generated filename forces the row wider than the
   menu instead of ellipsizing. */
.mmc-mention-row {
  display: flex; align-items: center; gap: 10px; width: 100%; min-width: 0;
  padding: 7px 8px; background: none; border: 1px solid transparent;
  border-radius: 10px; font-family: inherit; text-align: left; cursor: pointer;
  color: #ededed; overflow: hidden;
}
.mmc-mention-row[aria-selected="true"] { background: #2e2e2e; border-color: rgba(255,255,255,.13); }
.mmc-mention-thumb {
  width: 30px; height: 30px; border-radius: 7px; object-fit: cover; flex: none;
  background: #333; display: flex; align-items: center; justify-content: center;
  color: #8b8b8b; font-size: 13px;
}
.mmc-mention-text { display: flex; flex-direction: column; min-width: 0; flex: 1; }
.mmc-mention-handle {
  color: var(--tag, var(--mmc-accent)); font-size: 14px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mmc-mention-sub {
  color: #7d7d7d; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mmc-mention-empty { color: #7d7d7d; font-size: 13px; padding: 14px 10px; }

.mmc-pills { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.mmc-pill {
  display: flex; align-items: center; gap: 7px; height: 38px; padding: 0 14px;
  border-radius: 19px; background: var(--mmc-surface-2); border: 1px solid var(--mmc-line);
  color: var(--mmc-text); font-size: 13px; font-family: inherit; cursor: pointer;
  white-space: nowrap; transition: background .12s ease;
}
.mmc-pill:hover:not(:disabled) { background: var(--mmc-surface-3); }
.mmc-pill:disabled { cursor: not-allowed; color: var(--mmc-off); }
.mmc-pill svg { width: 16px; height: 16px; stroke: currentColor; fill: none;
  stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }
.mmc-pill-sub { color: var(--mmc-dim); font-size: 11px; }
.mmc-pill-group { gap: 0; padding: 0 6px; }
.mmc-step {
  background: none; border: 0; color: var(--mmc-text); cursor: pointer;
  font-size: 16px; width: 26px; height: 36px; font-family: inherit;
}
.mmc-step:disabled { color: var(--mmc-off); cursor: not-allowed; }
/* No text-transform: the socket name has to read exactly as it does on the
   input, and 'model_fl2va' uppercased is not the name of anything. */
.mmc-mode {
  margin-left: auto; font-size: 11px; letter-spacing: .04em; color: var(--mmc-dim);
  display: flex; align-items: center; gap: 6px;
  background: none; border: 1px solid transparent; border-radius: 13px;
  padding: 5px 10px; font-family: inherit;
}
/* Only the clickable form gets affordances — as a span it is a plain readout. */
button.mmc-mode { cursor: pointer; }
button.mmc-mode:hover { background: var(--mmc-surface-2); border-color: var(--mmc-line); }
.mmc-mode.pinned { border-color: var(--mmc-line); background: var(--mmc-surface-2); }
.mmc-mode b { color: var(--mmc-accent); font-weight: 600; }
.mmc-pin {
  font-size: 10px; letter-spacing: .06em; text-transform: uppercase;
  color: var(--mmc-accent); border: 1px solid currentColor; border-radius: 8px;
  padding: 0 5px; opacity: .8;
}
.mmc-warn { color: #e0743c; font-size: 12px; }

/* --- popovers ------------------------------------------------------------- */
.mmc-pop {
  position: fixed; z-index: 1300; background: #141414; border: 1px solid var(--mmc-line);
  border-radius: 16px; padding: 8px; min-width: 190px;
  box-shadow: 0 18px 48px rgba(0,0,0,.6);
}
.mmc-pop-title { color: var(--mmc-dim); font-size: 12px; padding: 6px 10px 8px; }

/* The output-folder popover. Fixed width for the same reason the slider's is:
   the example line changes length on every keystroke, and a popover that
   resized under the caret would be unusable to type in. */
.mmc-out-pop { width: 320px; }
.mmc-out-field {
  width: 100%; box-sizing: border-box; padding: 8px 10px;
  background: var(--mmc-surface); border: 1px solid var(--mmc-line);
  border-radius: 10px; color: var(--mmc-text);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px;
}
.mmc-out-field:focus { outline: none; border-color: var(--mmc-blue); }
.mmc-out-field.bad { border-color: #e0743c; }
.mmc-out-problem { color: #e0743c; font-size: 11.5px; line-height: 1.45; padding: 6px 2px 0; }
.mmc-out-example {
  padding: 8px 2px 2px; font-size: 11.5px; line-height: 1.6;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--mmc-text);
  /* A long dated folder must not widen the popover — see above. */
  overflow-wrap: anywhere;
}
.mmc-out-line { display: flex; gap: 8px; }
.mmc-out-key {
  color: var(--mmc-off); flex: none; width: 62px; text-align: right;
  font-family: inherit;
}
.mmc-out-tokens { display: flex; flex-wrap: wrap; gap: 4px; padding: 8px 2px 2px; }
.mmc-out-token {
  padding: 3px 7px; background: var(--mmc-surface-2); border: 0; border-radius: 7px;
  color: var(--mmc-dim); font-size: 11px; font-family: ui-monospace, Menlo, monospace;
  cursor: pointer;
}
.mmc-out-token:hover { background: var(--mmc-surface-3); color: var(--mmc-text); }
.mmc-out-note {
  color: var(--mmc-off); font-size: 11px; line-height: 1.5; padding: 8px 2px 2px;
  border-top: 1px solid var(--mmc-line); margin-top: 8px;
}
.mmc-out-note code {
  font-family: ui-monospace, Menlo, monospace; font-size: 10.5px; color: var(--mmc-dim);
}
.mmc-opt {
  display: flex; align-items: center; justify-content: space-between; width: 100%;
  padding: 9px 10px; background: none; border: 0; border-radius: 10px;
  color: var(--mmc-text); font-size: 14px; font-family: inherit; cursor: pointer;
  /* A button centres its text by default, which nothing notices while every
     option is one short word and looks broken the moment one wraps. */
  text-align: left;
}
.mmc-opt:hover { background: var(--mmc-surface-2); }
.mmc-opt-label { display: flex; align-items: center; gap: 10px; }
.mmc-aspect-glyph {
  width: 18px; height: 18px; flex: none;
  display: flex; align-items: center; justify-content: center;
}
.mmc-aspect-glyph > span { box-sizing: border-box; border: 1.5px solid #6a6a6a; border-radius: 2px; }
.mmc-opt[aria-checked="true"] .mmc-aspect-glyph > span { border-color: var(--mmc-blue); }
/* On a pill it is a glyph beside a label rather than a swatch in a list, so it
   takes the pill's own colour — and greys out with it when the ratio is coming
   from a keyframe and the pill is disabled. */
.mmc-pill .mmc-aspect-glyph > span { border-color: currentColor; border-width: 1.25px; }
.mmc-radio {
  width: 18px; height: 18px; border-radius: 50%; border: 1.5px solid #4a4a4a; flex: none;
}
.mmc-opt[aria-checked="true"] .mmc-radio {
  border-color: var(--mmc-blue); background: var(--mmc-blue);
  display: flex; align-items: center; justify-content: center;
}
.mmc-opt[aria-checked="true"] .mmc-radio::after {
  content: ""; width: 5px; height: 9px; border: solid #fff;
  border-width: 0 2px 2px 0; transform: rotate(45deg) translate(-1px,-1px);
}
/* The short-edge popover. Every measurement here is fixed on purpose: a range
   input reads the pointer against the width of its own track, so a popover that
   grows by a digit — or reflows a note onto a second line — slides the track out
   from under the thumb and the value jumps. Fixed width, tabular digits, and a
   note that always occupies two lines. Nothing in it may size to its text. */
.mmc-slider { width: 300px; padding: 12px; box-sizing: border-box; }
.mmc-slider-body { display: flex; flex-direction: column; gap: 8px; }
.mmc-slider-read {
  display: flex; align-items: baseline; justify-content: space-between;
  font-size: 13px; font-variant-numeric: tabular-nums; line-height: 20px;
}
.mmc-slider-read .mmc-edge { font-size: 16px; }
.mmc-slider-read .mmc-edge-unit { color: var(--mmc-dim); font-size: 11px; margin-left: 3px; }
.mmc-slider-read > span:last-child { color: var(--mmc-dim); }

.mmc-slider-row { display: flex; align-items: center; gap: 2px; }
/* The tick sits below the rail rather than over it, so it can be a real click
   target without eating the drag it is standing next to. */
.mmc-slider-track { position: relative; flex: 1; padding-bottom: 14px; min-width: 0; }
.mmc-slider input[type="range"] {
  display: block; width: 100%; margin: 0; height: 20px; accent-color: var(--mmc-blue);
}
.mmc-slider-mark {
  position: absolute; bottom: 0; height: 14px; width: 34px; padding: 0; border: 0;
  background: none; cursor: pointer; font-family: inherit; font-size: 9px;
  letter-spacing: .04em; color: var(--mmc-off);
  display: flex; flex-direction: column; align-items: center; gap: 2px;
  /* A range thumb is 16px, so the track it travels is inset 8px each side —
     the tick has to land on the value, not on the box. */
  left: calc(8px + var(--p) * (100% - 16px)); margin-left: -17px;
}
.mmc-slider-mark::before { content: ""; width: 2px; height: 4px; border-radius: 1px; background: currentColor; }
.mmc-slider-mark:hover { color: var(--mmc-text); }
.mmc-slider-mark.on { color: var(--mmc-blue); }

.mmc-native { color: var(--mmc-dim); font-size: 11px; line-height: 1.45; min-height: 32px; }
.mmc-native.over { color: #e0743c; }

/* --- picker modal --------------------------------------------------------- */
.mmc-overlay {
  position: fixed; inset: 0; z-index: 1400; background: rgba(0,0,0,.62);
  display: flex; align-items: center; justify-content: center; padding: 40px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
}
.mmc-modal {
  background: #161616; border: 1px solid var(--mmc-line); border-radius: 22px;
  width: min(1100px, 100%); height: min(760px, 100%);
  display: flex; flex-direction: column; color: #ededed; overflow: hidden;
  box-shadow: 0 30px 80px rgba(0,0,0,.65);
}
.mmc-modal-head {
  display: flex; align-items: center; gap: 22px; padding: 20px 24px 14px;
  border-bottom: 1px solid var(--mmc-line);
}
.mmc-tab {
  background: none; border: 0; padding: 4px 0; color: var(--mmc-dim);
  font-size: 17px; font-family: inherit; cursor: pointer;
}
.mmc-tab[aria-selected="true"] { color: #fff; font-weight: 500; }
.mmc-close {
  margin-left: auto; width: 34px; height: 34px; border-radius: 50%;
  background: #2a2a2a; border: 0; color: #ededed; cursor: pointer; font-size: 16px;
}
.mmc-modal-bar { display: flex; gap: 12px; padding: 16px 24px; align-items: center; }
.mmc-search {
  flex: 1; height: 40px; border-radius: 12px; background: #202020;
  border: 1px solid var(--mmc-line); color: #ededed; padding: 0 14px;
  font-size: 14px; font-family: inherit; outline: none;
}
.mmc-upload {
  height: 40px; padding: 0 18px; border-radius: 20px; background: #fff; border: 0;
  color: #111; font-size: 14px; font-weight: 500; font-family: inherit; cursor: pointer;
}
/* padding-bottom clears the floating Add/Cancel bar, which is positioned over
   the grid — without it the last row sits underneath and cannot be clicked. */
.mmc-grid {
  flex: 1; overflow-y: auto; padding: 4px 24px 96px;
  display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 14px;
  align-content: start;
  /* The grid is a flex child with a definite height, and implicit auto rows in
     one get fitted into that height rather than sized to their contents: with
     more rows than fit, every row is squeezed and the cards clip. max-content
     pins each row to the cards in it and lets the grid scroll, which is what
     overflow-y is here for. */
  grid-auto-rows: max-content;
}
/* The square is height:0 + padding-bottom:100%, not aspect-ratio, and the media
   is positioned out of flow.
   With aspect-ratio and in-flow media, thumbnails rendered at their natural
   height and spilled over the rows above and below. Whatever the host page does
   to img/video sizing, an absolutely positioned child cannot push its
   container taller, so the cell stays square and clips. */
.mmc-cell {
  position: relative; display: block; box-sizing: border-box;
  height: 0; padding: 0 0 100%;
  border-radius: 12px; overflow: hidden;
  background: #202020; border: 2px solid transparent; cursor: pointer;
}
.mmc-cell[aria-selected="true"] { border-color: #fff; }
.mmc-cell:focus-visible { outline: none; border-color: #7a7a7a; }
.mmc-cell img, .mmc-cell video, .mmc-cell-fallback {
  position: absolute; inset: 0; width: 100%; height: 100%; box-sizing: border-box;
}
.mmc-cell img, .mmc-cell video { object-fit: cover; display: block; }
.mmc-cell-fallback {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 8px; color: var(--mmc-dim);
  font-size: 11px; padding: 8px; text-align: center; word-break: break-all;
}
.mmc-cell-fallback svg { width: 26px; height: 26px; stroke: currentColor; fill: none; stroke-width: 1.5; }
.mmc-check {
  position: absolute; top: 8px; right: 8px; width: 22px; height: 22px;
  border-radius: 50%; background: #2f7bf6; display: none;
  align-items: center; justify-content: center;
}
.mmc-cell[aria-selected="true"] .mmc-check { display: flex; }
.mmc-check::after {
  content: ""; width: 5px; height: 10px; border: solid #fff;
  border-width: 0 2px 2px 0; transform: rotate(45deg) translate(-1px,-1px);
}
/* The segment badge. Invisible until the cell is hovered, focused or selected,
   unless it carries a setting — an untouched grid should look untouched. */
.mmc-cell-trim {
  position: absolute; top: 8px; left: 8px; display: none;
  align-items: center; gap: 5px; max-width: calc(100% - 44px);
  padding: 3px 8px 3px 6px; border-radius: 9px; border: 1px solid var(--mmc-line);
  background: rgba(0,0,0,.72); color: #ededed;
  font-size: 10px; font-family: inherit; cursor: pointer;
  white-space: nowrap; overflow: hidden;
}
.mmc-cell:hover .mmc-cell-trim,
.mmc-cell:focus-visible .mmc-cell-trim,
.mmc-cell[aria-selected="true"] .mmc-cell-trim,
.mmc-cell-trim.set { display: flex; }
.mmc-cell-trim.set { background: rgba(47,123,246,.9); border-color: transparent; }
.mmc-cell-trim:hover { background: rgba(0,0,0,.9); }
.mmc-cell-trim.set:hover { background: var(--mmc-blue); }
.mmc-cell-trim svg { width: 12px; height: 12px; flex: none; stroke: currentColor; fill: none;
  stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }

/* --- shelves -------------------------------------------------------------- */
/* One row of places between the search bar and the grid: All, favorites, every
   input subfolder, and the "+" that makes a new one. The same chip family as
   the refiner's language row, so the picker keeps the node's vocabulary. */
.mmc-shelves { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; padding: 0 24px 12px; }
.mmc-shelf {
  display: flex; align-items: center; gap: 6px; height: 30px; padding: 0 12px;
  border-radius: 15px; background: var(--mmc-surface-2); border: 1px solid var(--mmc-line);
  color: var(--mmc-dim); font-size: 12px; font-family: inherit; cursor: pointer;
  transition: background .12s ease, color .12s ease, transform .12s ease;
}
.mmc-shelf:hover { color: var(--mmc-text); background: var(--mmc-surface-3); }
.mmc-shelf[aria-selected="true"] { color: var(--mmc-bg); background: var(--mmc-accent); border-color: var(--mmc-accent); }
.mmc-shelf svg { width: 13px; height: 13px; flex: none; }
.mmc-shelf-n { font-size: 10px; opacity: .7; }
/* A chip with cargo hovering over it: swell and light up. The one place the
   picker spends motion. */
.mmc-shelf.drop {
  transform: scale(1.08); color: var(--mmc-text);
  background: var(--mmc-surface-3); border-color: var(--mmc-accent);
}
/* While a cell is riding, the chips announce they are drop targets. */
.mmc-modal.dragging .mmc-shelf { border-style: dashed; }
.mmc-shelf-new { font-size: 15px; line-height: 1; }
.mmc-shelf-input {
  height: 30px; width: 140px; border-radius: 15px; background: #202020;
  border: 1px solid var(--mmc-accent); color: #ededed; padding: 0 12px;
  font-size: 12px; font-family: inherit; outline: none;
}

/* The star. Same quiet-until-hover rule as the segment badge — an untouched
   grid stays untouched — but a set star stays lit. Steps left when the
   selection check needs the corner. */
.mmc-cell-star {
  position: absolute; top: 8px; right: 8px; width: 24px; height: 24px;
  display: none; align-items: center; justify-content: center;
  border: 0; border-radius: 50%; background: rgba(0,0,0,.55);
  color: #ededed; cursor: pointer; padding: 0;
}
.mmc-cell:hover .mmc-cell-star,
.mmc-cell:focus-visible .mmc-cell-star,
.mmc-cell-star.on { display: flex; }
.mmc-cell-star.on { color: var(--mmc-accent); }
.mmc-cell-star.on svg { fill: currentColor; }
.mmc-cell-star svg { width: 13px; height: 13px; }
.mmc-cell[aria-selected="true"] .mmc-cell-star { right: 36px; }
/* Where a file lives — worth a caption only on the All shelf, where everything
   is mixed together. Sits above the name gradient. */
.mmc-cell-home {
  position: absolute; left: 6px; bottom: 22px; max-width: calc(100% - 12px);
  padding: 2px 8px; border-radius: 8px; background: rgba(0,0,0,.6);
  color: #bdbdbd; font-size: 10px; pointer-events: none;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* --- organize mode --------------------------------------------------------- */
/* The bar toggle. Outlined next to the solid Upload button — a mode you enter,
   not an action you fire — and lit like a selected shelf while it is on. */
.mmc-organize {
  display: flex; align-items: center; gap: 7px; height: 40px; padding: 0 16px;
  border-radius: 20px; background: none; border: 1px solid var(--mmc-line);
  color: var(--mmc-dim); font-size: 14px; font-family: inherit; cursor: pointer;
  white-space: nowrap;
}
.mmc-organize:hover { color: var(--mmc-text); background: var(--mmc-surface-2); }
.mmc-organize[aria-pressed="true"] {
  color: var(--mmc-bg); background: var(--mmc-accent); border-color: var(--mmc-accent);
}
.mmc-organize svg { width: 14px; height: 14px; flex: none; }
/* Delete reads as danger from the start, and arming it turns it solid: the
   second press is the one that removes files. */
.mmc-del {
  background: none; border: 0; color: #e0743c; font-size: 14px;
  font-family: inherit; cursor: pointer; white-space: nowrap;
}
.mmc-del:hover { color: #f08a55; }
.mmc-del:disabled { color: var(--mmc-off); cursor: not-allowed; }
.mmc-del.armed {
  height: 36px; padding: 0 14px; border-radius: 10px;
  background: #b03a2a; color: #fff;
}
.mmc-del.armed:hover { color: #fff; background: #c74433; }
/* The Move to… popover, pinned above the footer it opened from. */
.mmc-move-menu {
  position: absolute; right: 44px; bottom: 100px; z-index: 5;
  display: flex; flex-direction: column; gap: 2px;
  min-width: 210px; max-height: 320px; overflow-y: auto; padding: 8px;
  background: #202020; border: 1px solid var(--mmc-line); border-radius: 12px;
  box-shadow: 0 12px 32px rgba(0,0,0,.5);
}
.mmc-move-opt {
  display: flex; align-items: center; gap: 8px; padding: 8px 10px;
  border-radius: 8px; background: none; border: 0; color: #ededed;
  font-size: 13px; font-family: inherit; cursor: pointer; text-align: left;
}
.mmc-move-opt:hover { background: var(--mmc-surface-3); }
.mmc-move-opt svg { width: 13px; height: 13px; flex: none; }
.mmc-move-menu .mmc-shelf-input { margin-top: 6px; width: auto; }

/* --- lora manager --------------------------------------------------------- */
/* Wider cells than the asset grid: a card carries a name, a base model, trigger
   words and, once active, three controls. */
.mmc-lora-grid { grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); }
/* Which folder under models/loras is being browsed. A native select: the list
   is as deep as the user's collection and the browser's own scrolling popup
   handles a hundred entries better than anything built here would. */
.mmc-folder {
  flex: none; max-width: 260px; height: 40px; border-radius: 12px;
  background: #202020; border: 1px solid var(--mmc-line); color: #ededed;
  padding: 0 10px; font-size: 13px; font-family: inherit; cursor: pointer; outline: none;
}
/* Sits after the last card and spans the grid: how many are still unrendered
   while scrolling, and what the server left out when it stops. */
.mmc-grid-note {
  grid-column: 1/-1; color: var(--mmc-dim); font-size: 12px;
  padding: 18px 0 4px; text-align: center; line-height: 1.5;
}
.mmc-lora {
  display: flex; flex-direction: column; border-radius: 12px; overflow: hidden;
  background: #202020; border: 2px solid transparent;
}
.mmc-lora[aria-selected="true"] { border-color: #fff; }
/* 4:3, and aspect-ratio rather than .mmc-cell's height:0 + padding-bottom.
   The cell IS a grid item, so its percentage padding sizes its own row. The art
   is a child of one, and percentage padding contributes nothing to a grid item's
   intrinsic height — the row would be sized for the body alone and the card,
   which clips, would swallow the picture whole. aspect-ratio gives a real height
   that counts. Every child is still positioned out of flow, so a showcase clip
   cannot push the box taller than its ratio. */
.mmc-lora-art {
  position: relative; aspect-ratio: 4 / 3; cursor: pointer; background: #191919;
}
.mmc-lora-art img, .mmc-lora-art canvas, .mmc-lora-art video, .mmc-lora-art .mmc-cell-fallback {
  position: absolute; inset: 0; width: 100%; height: 100%; box-sizing: border-box;
}
.mmc-lora-art img, .mmc-lora-art canvas, .mmc-lora-art video { object-fit: cover; display: block; }
.mmc-lora-art:focus-visible { outline: 2px solid #7a7a7a; outline-offset: -2px; }
.mmc-lora[aria-selected="true"] .mmc-check { display: flex; }
.mmc-lora-body { display: flex; flex-direction: column; gap: 4px; padding: 10px 11px 11px; }
.mmc-lora-name {
  font-size: 13px; color: #ededed;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mmc-lora-sub {
  font-size: 11px; color: var(--mmc-dim);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mmc-lora-words {
  font-size: 11px; color: var(--mmc-accent); margin-top: 2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mmc-lora-ctl {
  display: flex; flex-direction: column; gap: 7px;
  margin-top: 8px; padding-top: 9px; border-top: 1px solid var(--mmc-line);
}
.mmc-lora-row { display: flex; justify-content: space-between; align-items: center; font-size: 11px; }
.mmc-lora-label { color: var(--mmc-dim); }
.mmc-lora-strength { color: #ededed; font-variant-numeric: tabular-nums; }
.mmc-lora-ctl input[type="range"] { width: 100%; accent-color: var(--mmc-blue); margin: 0; }
.mmc-lora-idle { font-size: 10px; color: #e0743c; }
.mmc-seg {
  display: flex; border: 1px solid var(--mmc-line); border-radius: 9px; overflow: hidden;
}
.mmc-seg-btn {
  flex: 1; padding: 5px 0; background: none; border: 0; border-right: 1px solid var(--mmc-line);
  color: var(--mmc-dim); font-size: 10px; font-family: inherit; cursor: pointer;
}
.mmc-seg-btn:last-child { border-right: 0; }
.mmc-seg-btn:hover { color: #ededed; }
.mmc-seg-btn[aria-pressed="true"] { background: rgba(47,123,246,.22); color: #ededed; }

.mmc-trig-box { display: flex; flex-direction: column; gap: 6px; }
.mmc-trigs { display: flex; flex-wrap: wrap; gap: 4px; }
.mmc-trigs:empty { display: none; }
/* Off is an outline, on is filled: a sidecar word you have not taken should not
   look like one you have. */
.mmc-trig {
  padding: 2px 8px; border-radius: 8px; cursor: pointer; font-family: inherit;
  font-size: 10px; max-width: 100%; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; background: none; border: 1px solid var(--mmc-line);
  color: var(--mmc-off);
}
.mmc-trig:hover { color: #ededed; }
.mmc-trig[aria-pressed="true"] {
  background: rgba(240,166,60,.14); border-color: transparent; color: var(--mmc-accent);
}
/* Yours rather than the sidecar's — same weight in the prompt, but you can tell
   which list a word came from. */
.mmc-trig.own { border: 1px dashed rgba(240,166,60,.5); background: none; }
.mmc-trig-add {
  height: 24px; border-radius: 8px; background: #191919; border: 1px solid var(--mmc-line);
  color: #ededed; padding: 0 8px; font-size: 11px; font-family: inherit; outline: none;
  width: 100%; box-sizing: border-box;
}
.mmc-trig-add:focus { border-color: var(--mmc-blue); }

/* --- lightbox ------------------------------------------------------------- */
/* Double-click on any picker cell. Above the picker, like the segment editor. */
.mmc-light {
  display: flex; flex-direction: column; gap: 10px; align-items: center;
  max-width: 100%; max-height: 100%; min-height: 0;
}
.mmc-light-media {
  max-width: min(1400px, 100%); max-height: calc(100vh - 140px);
  min-height: 0; border-radius: 12px; background: #000; object-fit: contain;
  box-shadow: 0 30px 80px rgba(0,0,0,.65);
}
.mmc-light-audio { width: min(520px, 90vw); }
.mmc-light-name { font-size: 12px; color: var(--mmc-dim); }

/* --- LoRA detail sheet ---------------------------------------------------- */
/* Double-click on a manager card. Two shapes on purpose: with a sidecar the
   sheet is showcase-first (media pane + info column); without one there is
   nothing to show, only things to say, so it collapses to a single spec
   column. The layout itself tells you which kind of file you opened. */
.mmc-sheet {
  position: relative; display: flex; overflow: hidden;
  background: #161616; border: 1px solid var(--mmc-line); border-radius: 22px;
  width: min(1040px, 100%); height: min(680px, 100%);
  box-shadow: 0 30px 80px rgba(0,0,0,.65); color: var(--mmc-text);
}
.mmc-sheet.bare { width: min(560px, 100%); height: auto; max-height: min(680px, 100%); }
.mmc-sheet-close { position: absolute; top: 14px; right: 14px; z-index: 2; }

/* left: the showcase */
.mmc-sheet-stage {
  flex: 1; min-width: 0; display: flex; flex-direction: column;
  background: #0b0b0b; border-right: 1px solid var(--mmc-line);
}
.mmc-sheet-media { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; }
.mmc-sheet-media img, .mmc-sheet-media video {
  max-width: 100%; max-height: 100%; object-fit: contain; display: block;
}
.mmc-sheet-strip {
  display: flex; gap: 8px; padding: 10px 14px 0; overflow-x: auto; flex: none;
}
.mmc-sheet-thumb {
  flex: none; width: 56px; height: 56px; padding: 0; border-radius: 8px; overflow: hidden;
  background: #202020; border: 2px solid transparent; cursor: pointer;
}
.mmc-sheet-thumb[aria-selected="true"] { border-color: #fff; }
.mmc-sheet-thumb img, .mmc-sheet-thumb video {
  width: 100%; height: 100%; object-fit: cover; display: block;
}
/* the recipe strip: how the shown image was actually generated */
.mmc-sheet-recipe { flex: none; padding: 0 14px; }
.mmc-sheet-recipe.on { padding: 10px 14px 12px; display: flex; flex-direction: column; gap: 7px; }
.mmc-sheet-recipe-facts {
  display: flex; gap: 14px; flex-wrap: wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px;
}
.mmc-sheet-recipe-k { color: var(--mmc-off); }
.mmc-sheet-recipe-v { color: var(--mmc-text); }
.mmc-sheet-prompt { display: flex; gap: 10px; align-items: flex-start; }
.mmc-sheet-prompt-text {
  flex: 1; min-width: 0; font-size: 11.5px; line-height: 1.45; color: var(--mmc-dim);
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.mmc-sheet-copy {
  flex: none; height: 24px; padding: 0 10px; border-radius: 12px;
  background: var(--mmc-surface-2); border: 1px solid var(--mmc-line);
  color: var(--mmc-dim); font-size: 11px; font-family: inherit; cursor: pointer;
}
.mmc-sheet-copy:hover { color: var(--mmc-text); background: var(--mmc-surface-3); }
.mmc-sheet-negative {
  font-size: 10.5px; color: var(--mmc-off);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* right: the info column */
.mmc-sheet-info {
  flex: none; width: 340px; min-width: 0; overflow-y: auto;
  padding: 22px 24px 24px; display: flex; flex-direction: column; gap: 14px;
}
.mmc-sheet.bare .mmc-sheet-info { flex: 1; width: auto; }
.mmc-sheet-eyebrow {
  display: flex; align-items: center; gap: 8px;
  font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--mmc-dim); padding-right: 40px;
}
.mmc-sheet-nsfw { color: #e0743c; }
.mmc-sheet-mono-mark { display: inline-flex; color: var(--mmc-accent); }
.mmc-sheet-title { font-size: 20px; font-weight: 600; line-height: 1.25; padding-right: 30px; }
.mmc-sheet-byline { font-size: 12px; color: var(--mmc-dim); }
.mmc-sheet-stats { display: flex; gap: 18px; padding: 2px 0; }
.mmc-sheet-stat { display: flex; flex-direction: column; gap: 1px; }
.mmc-sheet-stat-v { font-size: 14px; font-weight: 600; font-variant-numeric: tabular-nums; }
.mmc-sheet-stat-k { font-size: 10px; color: var(--mmc-off); }
.mmc-sheet-section { display: flex; flex-direction: column; gap: 6px; }
.mmc-sheet-label {
  font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--mmc-off);
}
.mmc-sheet-chips { display: flex; gap: 6px; flex-wrap: wrap; }
.mmc-sheet-chip {
  display: inline-flex; align-items: baseline; gap: 5px;
  padding: 3px 9px; border-radius: 11px; font-size: 11.5px;
  background: var(--mmc-surface-2); border: 1px solid var(--mmc-line); color: var(--mmc-text);
}
.mmc-sheet-chip.accent { border-color: rgba(240,166,60,.4); color: var(--mmc-accent); }
.mmc-sheet-chip-n { font-size: 9.5px; color: var(--mmc-off); font-variant-numeric: tabular-nums; }
.mmc-sheet-desc { font-size: 12.5px; line-height: 1.5; color: var(--mmc-dim); }
.mmc-sheet-desc p { margin: 0 0 8px; }
.mmc-sheet-desc ul, .mmc-sheet-desc ol { margin: 0 0 8px; padding-left: 18px; }
.mmc-sheet-desc code, .mmc-sheet-desc pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px;
  background: var(--mmc-surface-2); border-radius: 4px; padding: 1px 4px;
}
.mmc-sheet-desc pre { padding: 8px 10px; overflow-x: auto; }
.mmc-sheet-desc blockquote {
  margin: 0 0 8px; padding-left: 10px; border-left: 2px solid var(--mmc-line);
}
.mmc-sheet-desc a { color: var(--mmc-blue); text-decoration: none; }
.mmc-sheet-desc a:hover { text-decoration: underline; }
.mmc-sheet-h { margin: 0 0 6px; font-weight: 600; color: var(--mmc-text); }
.mmc-sheet-versions { display: flex; flex-direction: column; }
.mmc-sheet-version {
  display: flex; align-items: baseline; gap: 8px; padding: 5px 0;
  border-bottom: 1px solid var(--mmc-line); font-size: 12px;
}
.mmc-sheet-version:last-child { border-bottom: 0; }
.mmc-sheet-version-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mmc-sheet-version[aria-current="true"] .mmc-sheet-version-name { color: var(--mmc-accent); }
.mmc-sheet-version-sub { margin-left: auto; flex: none; font-size: 10.5px; color: var(--mmc-off); }
.mmc-sheet-installed { flex: none; font-size: 10px; color: var(--mmc-accent); }
.mmc-sheet-license, .mmc-sheet-tags, .mmc-sheet-hint { font-size: 11.5px; line-height: 1.5; color: var(--mmc-dim); }
.mmc-sheet-file { display: flex; flex-direction: column; gap: 3px; font-size: 11px; color: var(--mmc-dim); }
.mmc-sheet-path {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; direction: rtl; text-align: left;
}
.mmc-sheet-hash, .mmc-sheet-file-facts { font-variant-numeric: tabular-nums; }
.mmc-sheet-link { font-size: 12px; color: var(--mmc-blue); text-decoration: none; }
.mmc-sheet-link:hover { text-decoration: underline; }

/* the bare sheet's spec grid and raw-header disclosure */
.mmc-sheet-spec { display: flex; flex-direction: column; }
.mmc-sheet-spec-row {
  display: flex; gap: 14px; padding: 5px 0; font-size: 12px;
  border-bottom: 1px solid var(--mmc-line);
}
.mmc-sheet-spec-row:last-child { border-bottom: 0; }
.mmc-sheet-spec-k { flex: none; width: 110px; color: var(--mmc-dim); }
.mmc-sheet-spec-v {
  min-width: 0; overflow-wrap: anywhere;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px;
}
.mmc-sheet-raw { font-size: 11px; }
.mmc-sheet-raw summary { cursor: pointer; color: var(--mmc-off); }
.mmc-sheet-raw summary:hover { color: var(--mmc-dim); }
.mmc-sheet-raw-rows { display: flex; flex-direction: column; gap: 4px; padding-top: 8px; }
.mmc-sheet-raw-row {
  display: flex; gap: 10px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10.5px;
}
.mmc-sheet-raw-k { flex: none; width: 170px; color: var(--mmc-off); overflow-wrap: anywhere; }
.mmc-sheet-raw-v { min-width: 0; color: var(--mmc-dim); overflow-wrap: anywhere; }

/* --- segment editor ------------------------------------------------------- */
/* Above the picker: it opens on top of it. */
.mmc-trim {
  background: #161616; border: 1px solid var(--mmc-line); border-radius: 20px;
  width: min(640px, 100%); padding: 16px 18px 14px;
  display: flex; flex-direction: column; gap: 12px;
  box-shadow: 0 30px 80px rgba(0,0,0,.65);
}
.mmc-trim-head-row { display: flex; align-items: center; gap: 12px; }
.mmc-trim-name {
  font-size: 13px; color: var(--mmc-dim); min-width: 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mmc-trim-media {
  width: 100%; height: auto; max-height: 46vh; border-radius: 12px; background: #000; display: block;
  /* The stage is a canvas sized to the clip. Height follows the aspect on its
     own until max-height clamps it; past that only object-fit keeps the picture
     from being squashed into the box. */
  object-fit: contain;
}
.mmc-trim-bar { display: flex; align-items: center; gap: 12px; }
.mmc-trim-play {
  width: 34px; height: 34px; flex: none; border-radius: 50%;
  background: var(--mmc-surface-2); border: 1px solid var(--mmc-line);
  color: var(--mmc-text); cursor: pointer; display: flex; align-items: center; justify-content: center;
}
.mmc-trim-play svg { width: 16px; height: 16px; stroke: currentColor; fill: none;
  stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }
.mmc-trim-track {
  position: relative; flex: 1; height: 30px; border-radius: 8px;
  background: var(--mmc-surface-2); cursor: pointer; touch-action: none;
}
/* Audio files: no picture, so the waveform is the preview. */
.mmc-trim-track-tall { height: 120px; border-radius: 12px; }
.mmc-trim-wave {
  position: absolute; inset: 0; width: 100%; height: 100%;
  display: block; pointer-events: none;
}
.mmc-trim-sel { position: absolute; top: 0; bottom: 0; background: rgba(47,123,246,.28);
  border-top: 1px solid var(--mmc-blue); border-bottom: 1px solid var(--mmc-blue);
  cursor: grab; touch-action: none; }
.mmc-trim-sel:active { cursor: grabbing; background: rgba(47,123,246,.36); }
.mmc-trim-sel:focus-visible { outline: 2px solid #fff; outline-offset: -2px; }
.mmc-trim-head { position: absolute; top: 0; bottom: 0; width: 2px; margin-left: -1px;
  background: #fff; opacity: .8; pointer-events: none; }
.mmc-trim-handle {
  position: absolute; top: -3px; bottom: -3px; width: 12px; margin-left: -6px;
  border-radius: 5px; background: var(--mmc-blue); cursor: ew-resize; touch-action: none;
}
.mmc-trim-handle:focus-visible { outline: 2px solid #fff; outline-offset: 1px; }
.mmc-trim-read { display: flex; flex-wrap: wrap; justify-content: space-between; font-size: 12px; color: var(--mmc-text); }
.mmc-trim-len { color: var(--mmc-dim); }
/* Its own line under the times, so the two of them keep their ends of the row. */
.mmc-trim-note { flex: 1 0 100%; margin-top: 4px; color: var(--mmc-dim); font-size: 11px; }
/* Wraps rather than squeezing: the track switch is three words wide and the
   modal is only 640 px. */
.mmc-trim-foot { display: flex; align-items: center; flex-wrap: wrap; gap: 10px 14px; }
.mmc-trim-spacer { flex: 1; }
/* The track switch: three mutually exclusive choices, so one bordered group
   rather than three loose pills that would read as independent toggles. */
.mmc-seg {
  display: flex; height: 30px; border-radius: 15px; overflow: hidden;
  background: var(--mmc-surface-2); border: 1px solid var(--mmc-line);
}
.mmc-seg-opt {
  padding: 0 12px; cursor: pointer; background: none; border: 0;
  border-left: 1px solid var(--mmc-line);
  color: var(--mmc-dim); font-size: 12px; font-family: inherit; white-space: nowrap;
}
.mmc-seg-opt:first-child { border-left: 0; }
.mmc-seg-opt:hover:not(:disabled) { color: #ededed; }
.mmc-seg-opt[aria-pressed="true"] { background: rgba(47,123,246,.22); color: #ededed; }
.mmc-seg-opt:disabled { color: var(--mmc-off); cursor: not-allowed; }
.mmc-ghost:disabled { color: var(--mmc-off); cursor: not-allowed; }

.mmc-cell-name {
  position: absolute; left: 0; right: 0; bottom: 0; padding: 14px 8px 6px;
  font-size: 10px; color: #d0d0d0; text-align: left;
  background: linear-gradient(transparent, rgba(0,0,0,.8));
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mmc-empty { grid-column: 1/-1; color: var(--mmc-dim); font-size: 14px; padding: 40px 0; text-align: center; }
.mmc-modal-foot {
  position: absolute; bottom: 34px; right: 44px;
  display: flex; align-items: center; gap: 16px; padding: 10px 12px 10px 20px;
  background: #202020; border: 1px solid var(--mmc-line); border-radius: 14px;
  box-shadow: 0 12px 32px rgba(0,0,0,.5); font-size: 14px;
}
.mmc-slots { color: #ededed; }
.mmc-slots.full { color: #e0743c; }
.mmc-ghost { background: none; border: 0; color: var(--mmc-dim); font-size: 14px;
  font-family: inherit; cursor: pointer; }
.mmc-ghost:hover { color: #ededed; }
.mmc-add {
  height: 36px; padding: 0 20px; border-radius: 10px; background: var(--mmc-blue);
  border: 0; color: #fff; font-size: 14px; font-weight: 500; font-family: inherit; cursor: pointer;
}
.mmc-add:disabled { opacity: .4; cursor: not-allowed; }

/* --- timeline ------------------------------------------------------------- */

/* The continuation switch, on a segment's pill row. Lit when on, because a
   segment that inherits its first frame is not the default and should not
   look like it. */
.mmc-continue.on { border-color: var(--mmc-accent); color: var(--mmc-accent); }

/* Past the ~5-15 s the weights were trained on. The same warm orange the
   resolution slider uses above 768: a statement about distribution, not a
   refusal, so it marks rather than disables. */
.mmc-pill-group.off-distribution { border-color: rgba(224,116,60,.4); }
.mmc-pill-group.off-distribution > span { color: #e0743c; }
.mmc-tl-dur.off-distribution { color: #e0743c; }

.mmc-tl-modal { height: min(680px, 100%); }
.mmc-tl-body {
  display: flex; flex-direction: column; gap: 16px;
  padding: 18px 24px 24px; overflow: auto; flex: 1; min-height: 0;
}
.mmc-tl-prompt {
  width: 100%; box-sizing: border-box; min-height: 84px; resize: vertical;
  background: var(--mmc-surface); border: 1px solid var(--mmc-line); border-radius: 14px;
  color: var(--mmc-text); font-family: inherit; font-size: 14px; line-height: 1.5;
  padding: 14px 16px; outline: none;
}
.mmc-tl-prompt:focus { border-color: rgba(255,255,255,.2); }
.mmc-tl-prompt::placeholder { color: var(--mmc-off); }

/* The two Context-IR audio fields, side by side under the prompt. They wrap to
   one column when the modal is too narrow to give each a readable measure. */
.mmc-tl-audio {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px;
}
.mmc-tl-field { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
/* Named for the field they become, not prettified: the value goes into the
   prompt under exactly this key, and someone comparing against MiniMax's guide
   should be able to find it by the same word. */
.mmc-tl-field-name {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px; color: var(--mmc-dim); letter-spacing: .02em;
}
.mmc-tl-small { min-height: 64px; font-size: 13px; padding: 10px 12px; }

.mmc-tl-bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.mmc-pill.on { border-color: rgba(255,255,255,.22); }
/* An accelerator that is doing something. Lit rather than merely outlined,
   because a render with one on is not a native render and that is worth seeing
   without reading the pill. */
.mmc-pill.accel-on { border-color: rgba(110,190,255,.45); color: #6ebeff; }
.mmc-pill.accel-on:hover:not(:disabled) { border-color: rgba(110,190,255,.7); }
/* An architecture that is not settled yet. Dashed rather than coloured: this
   says "the output may not be good", which is a different statement from the
   accelerator blue's "this render is not native". */
.mmc-pill.mmc-experimental { border-style: dashed; border-color: rgba(255,196,110,.5); }
.mmc-pill.mmc-experimental:hover:not(:disabled) { border-color: rgba(255,196,110,.8); }
/* A sweep choice inside the dev popover: the same on/off reading as the turbo
   stops, on ordinary pills because the lists are of no fixed length. */
.mmc-pill[aria-pressed="true"] { border-color: rgba(110,190,255,.45); color: #6ebeff; }
/* The turbo switch: the seed pill's shape — one pill, a big half that throws
   it and a small half that picks what it throws. Both inherit the group's
   colour so the accelerator blue lights the whole pill, chevron included. */
.mmc-turbo-main {
  display: flex; align-items: center; gap: 7px; height: 100%; padding: 0 2px 0 8px;
  background: none; border: 0; color: inherit; font-size: 13px;
  font-family: inherit; cursor: pointer; white-space: nowrap;
}
.mmc-turbo-pick {
  display: flex; align-items: center; justify-content: center; width: 22px; color: inherit;
}
/* The turbo quality stops. One pill holding three mutually exclusive answers,
   like the trim editor's track switch — three loose pills would read as
   independent toggles, and draft/med/good are one dial. Lit in the accelerator
   blue, because that is the family it belongs to. */
.mmc-pill.mmc-turbo-seg { gap: 0; padding: 0; overflow: hidden; }
.mmc-turbo-opt {
  display: flex; align-items: center; gap: 5px; height: 100%; padding: 0 12px;
  background: none; border: 0; border-left: 1px solid var(--mmc-line);
  color: var(--mmc-dim); font-size: 13px; font-family: inherit; cursor: pointer;
}
.mmc-turbo-opt:first-child { border-left: 0; }
.mmc-turbo-opt:hover { color: #ededed; }
.mmc-turbo-opt[aria-pressed="true"] { background: rgba(110,190,255,.14); color: #6ebeff; }
.mmc-turbo-opt[aria-pressed="true"] .mmc-pill-sub { color: rgba(110,190,255,.75); }
.mmc-tl-total { display: flex; gap: 8px; align-items: baseline; margin-left: auto; font-size: 13px; }
.mmc-tl-total span { color: var(--mmc-dim); }

/* Chained / one pass. A segmented control rather than two pills, because they
   are one choice with two answers and every other pill on this bar is a value
   you set independently. It leads the bar for the same reason: it is the one
   control that changes what all the others mean. */
.mmc-tl-render {
  display: flex; gap: 2px; padding: 2px; border-radius: 10px;
  background: var(--mmc-surface-3); border: 1px solid var(--mmc-line);
}
.mmc-tl-render-opt {
  height: 24px; padding: 0 10px; border: 0; border-radius: 8px; background: none;
  color: var(--mmc-dim); font-family: inherit; font-size: 12px; cursor: pointer;
}
.mmc-tl-render-opt:hover { color: var(--mmc-text); }
.mmc-tl-render-opt.on { background: var(--mmc-surface); color: var(--mmc-text); }

/* A refusal compile.py would raise, said while the shots are still editable.
   Reads as a note rather than an alarm — the timeline is still saveable, and
   switching back to chained makes it correct again. */
.mmc-tl-problem {
  display: flex; gap: 8px; align-items: baseline; flex-basis: 100%;
  font-size: 11px; line-height: 1.4; color: #e0743c;
}
.mmc-tl-problem .mmc-note-key { color: inherit; opacity: .8; }

/* Laid out left to right and scrolled, not wrapped: a timeline that wraps onto
   a second line stops reading as time. */
.mmc-tl-strip {
  display: flex; align-items: stretch; gap: 0;
  overflow-x: auto; padding-bottom: 10px; min-height: 190px;
}
.mmc-tl-card {
  flex: 0 0 auto; box-sizing: border-box;
  display: flex; flex-direction: column; gap: 8px;
  background: var(--mmc-surface); border: 1px solid var(--mmc-line);
  border-radius: 14px; padding: 12px; font-size: 12px; cursor: default;
}
.mmc-tl-card:hover { border-color: rgba(255,255,255,.18); }
.mmc-tl-card-head { display: flex; align-items: center; gap: 8px; }
.mmc-tl-index {
  width: 20px; height: 20px; border-radius: 50%; background: var(--mmc-surface-3);
  display: flex; align-items: center; justify-content: center; font-size: 11px; flex: 0 0 auto;
}
.mmc-tl-dur { color: var(--mmc-text); font-weight: 500; }
.mmc-tl-mode { color: var(--mmc-accent); font-size: 11px; margin-left: auto; }
.mmc-tl-card-prompt {
  flex: 1; color: var(--mmc-text); line-height: 1.45; overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical;
}
.mmc-tl-card-prompt.empty { color: var(--mmc-off); font-style: italic; }
/* The card keeps showing the typed sentence because that is what the shot is
   recognised by — dimmed, because it is not what the shot queues. */
.mmc-tl-card-prompt.superseded { opacity: .42; }
.mmc-tl-card-meta { color: var(--mmc-dim); font-size: 11px; }
.mmc-tl-card-foot { display: flex; align-items: center; gap: 4px; }
.mmc-tl-edit {
  height: 26px; padding: 0 12px; border-radius: 8px; background: var(--mmc-surface-3);
  border: 0; color: var(--mmc-text); font-size: 12px; font-family: inherit; cursor: pointer;
  margin-right: auto;
}
.mmc-tl-edit:hover { background: #3a3a3a; }
.mmc-tl-card-foot .mmc-ghost { padding: 0 4px; font-size: 12px; }
.mmc-tl-card-foot button:disabled { opacity: .3; cursor: not-allowed; }

/* The seam between two cards. It is a control, so it is wide enough to hit. */
.mmc-tl-join {
  flex: 0 0 auto; align-self: center; width: 62px;
  display: flex; flex-direction: column; align-items: center; gap: 2px;
  background: none; border: 0; color: var(--mmc-off); cursor: pointer;
  font-family: inherit; font-size: 10px; padding: 4px 0;
}
.mmc-tl-join span:first-child { font-size: 15px; }
.mmc-tl-join:hover:not(:disabled) { color: var(--mmc-text); }
.mmc-tl-join.on { color: var(--mmc-accent); }
.mmc-tl-join:disabled { cursor: not-allowed; opacity: .5; }

/* Picture above, sound below — the two switches on one seam. Stacked rather
   than side by side so the seam stays as narrow as it was. */
.mmc-tl-seam {
  flex: 0 0 auto; align-self: center;
  display: flex; flex-direction: column; align-items: center; gap: 2px;
}
/* The same place, in one pass: not a control, because a cut inside a single
   generation is a line of the description rather than a wiring decision. What
   it shows is the timestamp that line will carry. */
.mmc-tl-cut {
  display: flex; flex-direction: column; align-items: center; gap: 2px;
  width: 62px; padding: 4px 0; color: var(--mmc-off); font-size: 10px;
  font-variant-numeric: tabular-nums; cursor: default;
}
.mmc-tl-cut span:first-child { font-size: 15px; }

.mmc-tl-join-sound { padding-top: 0; }
.mmc-tl-join-sound svg { width: 13px; height: 13px; stroke: currentColor; fill: none;
  stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }

.mmc-tl-add {
  flex: 0 0 auto; align-self: stretch; width: 108px; margin-left: 12px;
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px;
  background: none; border: 1px dashed var(--mmc-line); border-radius: 14px;
  color: var(--mmc-dim); font-family: inherit; font-size: 12px; cursor: pointer;
}
.mmc-tl-add span:first-child { font-size: 20px; }
.mmc-tl-add:hover:not(:disabled) { color: var(--mmc-text); border-color: rgba(255,255,255,.2); }
.mmc-tl-add:disabled { cursor: not-allowed; opacity: .4; }

/* The segment editor, over the strip. Its body is the Creator node's, unchanged. */
.mmc-tl-editor { width: min(880px, 100%); height: min(720px, 100%); }
.mmc-tl-editor-sub { color: var(--mmc-dim); font-size: 13px; }
.mmc-tl-editor-body { overflow: auto; flex: 1; min-height: 0; }
.mmc-tl-editor-body .mmc-root { height: auto; overflow: visible; padding: 18px 24px 24px; }

/* --- timeline node body --------------------------------------------------- */

.mmc-tl-summary { gap: 14px; flex: 1; min-height: 0; }
.mmc-tl-summary-prompt {
  font-size: 13px; line-height: 1.5; color: var(--mmc-text); cursor: text;
  flex: 1; min-height: 40px; overflow: hidden;
}
.mmc-tl-summary-prompt.empty { color: var(--mmc-off); }
/* Segments at their real relative lengths — the node's one honest picture of
   the timeline without room for the strip itself. */
.mmc-tl-lane { display: flex; gap: 4px; height: 40px; cursor: pointer; flex: 0 0 auto; }
.mmc-tl-tick {
  display: flex; align-items: center; justify-content: center; gap: 5px;
  background: var(--mmc-surface-2); border: 1px solid var(--mmc-line);
  border-radius: 8px; min-width: 18px; overflow: hidden; padding: 0 4px;
}
.mmc-tl-tick svg { width: 13px; height: 13px; stroke: currentColor; fill: none;
  stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; flex: 0 0 auto; }
.mmc-tl-tick-n { color: var(--mmc-text); font-size: 12px; font-weight: 500; }
.mmc-tl-tick-s { color: var(--mmc-dim); font-size: 11px; }
.mmc-tl-lane:hover .mmc-tl-tick { border-color: rgba(255,255,255,.18); }
.mmc-tl-tick.on {
  background: rgba(240,166,60,.13); border-color: rgba(240,166,60,.32); color: var(--mmc-accent);
}
.mmc-tl-tick.on .mmc-tl-tick-n { color: var(--mmc-accent); }

/* Where the Creator puts its mode badge: the right end of the pill row. */
.mmc-tl-open {
  margin-left: auto; height: 32px; padding: 0 14px; display: flex; align-items: center; gap: 8px;
  border-radius: 999px; background: var(--mmc-surface-3); border: 1px solid var(--mmc-line);
  color: var(--mmc-text); font-family: inherit; font-size: 13px; cursor: pointer;
}
.mmc-tl-open:hover { background: #3a3a3a; border-color: rgba(255,255,255,.18); }
.mmc-tl-open svg { width: 16px; height: 16px; stroke: currentColor; fill: none;
  stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }

/* --- sampler pills -------------------------------------------------------- */

/* A pill that only reports. Same shape, no hover lift — it is not a control. */
.mmc-pill-static { cursor: default; }
.mmc-pill-static:hover { background: var(--mmc-surface-2); }
.mmc-pill-static svg { color: var(--mmc-dim); }
.mmc-seed-dice { display: flex; align-items: center; padding: 0 4px; }
.mmc-seed-dice svg { width: 15px; height: 15px; stroke: currentColor; fill: none;
  stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }
.mmc-seed-input {
  width: 92px; background: none; border: 0; outline: none; color: var(--mmc-text);
  font-family: inherit; font-size: 13px; text-align: center; padding: 0;
}
.mmc-seed-mode { font-size: 11px; padding: 0 8px 0 4px; }
/* Sampler lists are long; the popover scrolls rather than running off screen. */
.mmc-pop-scroll { max-height: 320px; overflow-y: auto; min-width: 190px; }

/* --- the refiner ---------------------------------------------------------- */
/* One press refines; the corner opens the settings. The corner sits inside the
   tile's own rounded box rather than floating over it, so the rail keeps its
   alignment and the control cannot be mistaken for a badge on its neighbour. */
.mmc-refine-split { position: relative; display: flex; }
/* Anchored off the tile's centre rather than the wrapper's left edge, so a
   longer label cannot slide it off the corner it belongs in. */
.mmc-refine-more {
  position: absolute; top: 36px; left: 50%; margin-left: 8px; width: 18px; height: 18px;
  display: flex; align-items: center; justify-content: center;
  background: none; border: 0; border-radius: 6px; padding: 0;
  color: var(--mmc-off); cursor: pointer; transition: color .12s ease, background .12s ease;
}
.mmc-refine-more svg { width: 12px; height: 12px; }
.mmc-refine-split:hover .mmc-refine-more { color: var(--mmc-dim); }
.mmc-refine-more:hover { color: var(--mmc-text); background: var(--mmc-surface-3); }
/* Only the keyboard gets a ring. Clicking left one sitting on the tile, which
   is exactly what made the old control read as a notification dot. */
.mmc-refine-more:focus:not(:focus-visible) { outline: none; }

/* A refine is a round trip to a local model and can take a minute. The pulse is
   the only thing saying the click landed. */
.mmc-tool.busy, .mmc-pill.busy { color: var(--mmc-accent); cursor: progress; }
.mmc-tool.busy .mmc-tool-icon { animation: mmc-pulse 1.4s ease-in-out infinite; }
.mmc-pill.busy { animation: mmc-pulse 1.4s ease-in-out infinite; }
@keyframes mmc-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .45; } }

/* On the timeline bar the same control is a pill, so the corner becomes an
   ordinary chevron at the end of the label. */
.mmc-tl-refine .mmc-tool-icon {
  width: auto; height: auto; background: none; border: 0; border-radius: 0;
}
.mmc-tl-refine svg { width: 15px; height: 15px; }
/* Beside it, and quieter than it: undoing is the rarer press of the two. */
.mmc-tl-unrefine { color: var(--mmc-dim); }
.mmc-tl-unrefine:hover { color: var(--mmc-text); }
.mmc-refine-split.pill { align-items: stretch; }
.mmc-refine-split.pill .mmc-refine-more {
  position: static; width: 24px; height: 38px; border-radius: 0 19px 19px 0;
  margin-left: -10px; background: var(--mmc-surface-2);
  border: 1px solid var(--mmc-line); border-left: 0; color: var(--mmc-dim);
}
.mmc-refine-split.pill .mmc-refine-more:hover { background: var(--mmc-surface-3); }
.mmc-refine-split.pill .mmc-pill { padding-right: 16px; }

.mmc-refine-pop { width: 264px; padding: 8px; }
.mmc-refine-models { max-height: 190px; overflow-y: auto; }
/* Names run to a folder-qualified qwen3vl/qwen3vl_4b_instruct_fp8.safetensors.
   The row ellipsises and the title carries the whole of it. */
.mmc-refine-name {
  display: block; min-width: 0; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;
}
/* What kind of prompting a row is — "prompt" or "skill" — said as a tiny tag
   between the name and the radio, because the rows otherwise look like two of
   the same thing and behave like two different ones. */
.mmc-opt-kind {
  flex: none; margin-left: auto; margin-right: 8px;
  font-size: 10px; color: var(--mmc-dim);
  border: 1px solid var(--mmc-line); border-radius: 999px; padding: 1px 7px;
}
.mmc-refine-hint { font-size: 11px; color: var(--mmc-dim); line-height: 1.4; }
.mmc-refine-note { padding: 2px 10px 8px; }
.mmc-refine-empty {
  display: flex; flex-direction: column; align-items: flex-start; gap: 8px;
  padding: 4px 10px 10px; font-size: 12px; color: var(--mmc-dim);
}
.mmc-refine-empty code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px;
  color: var(--mmc-text); background: var(--mmc-surface-2);
  border: 1px solid var(--mmc-line); border-radius: 7px; padding: 5px 8px;
}
.mmc-refine-empty .mmc-ghost { font-size: 12px; }

/* Everything but the model, folded away: it is set once and the model is not. */
.mmc-refine-fold { font-size: 12px; }
.mmc-refine-fold > summary {
  cursor: pointer; color: var(--mmc-dim); padding: 8px 10px;
  border-top: 1px solid var(--mmc-line); list-style-position: inside;
}
.mmc-refine-fold > summary:hover { color: var(--mmc-text); }
.mmc-refine-more-body {
  display: flex; flex-direction: column; gap: 12px; padding: 4px 10px 8px;
}
.mmc-refine-group { display: flex; flex-direction: column; gap: 7px; }
.mmc-refine-row { display: flex; gap: 8px; flex-wrap: wrap; }
.mmc-refine-seed { font-size: 12px; padding: 0 10px 0 2px; }

/* Eleven languages: chips wrap into three lines and stay scannable, where a
   list would scroll and a <select> would be the only browser chrome in the node. */
.mmc-chips { display: flex; flex-wrap: wrap; gap: 5px; }
.mmc-chip {
  padding: 4px 10px; border-radius: 12px; background: var(--mmc-surface-2);
  border: 1px solid var(--mmc-line); color: var(--mmc-dim);
  font-family: inherit; font-size: 11px; cursor: pointer; transition: all .12s ease;
}
.mmc-chip:hover { color: var(--mmc-text); background: var(--mmc-surface-3); }
.mmc-chip[aria-checked="true"] {
  color: var(--mmc-bg); background: var(--mmc-accent); border-color: var(--mmc-accent);
}

/* --- the rewrite ---------------------------------------------------------- */
/* An editor, not a readout: the rewrite is a draft, and correcting one word of
   it should not mean running the model again. */
.mmc-refined { display: flex; flex-direction: column; gap: 8px; }
.mmc-refined:empty { display: none; }
.mmc-refined-head { display: flex; align-items: center; gap: 8px; font-size: 12px; }
.mmc-refined-toggle {
  display: flex; align-items: center; gap: 6px; background: none; border: 0;
  padding: 0; cursor: pointer; color: var(--mmc-off); font: inherit; font-size: 12px;
}
.mmc-refined-toggle.on { color: var(--mmc-accent); }
.mmc-dot {
  width: 7px; height: 7px; border-radius: 50%; background: currentColor; opacity: .5;
}
.mmc-refined-toggle.on .mmc-dot { opacity: 1; }
.mmc-refined-model { color: var(--mmc-dim); font-size: 11px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.mmc-refined-stale { color: var(--mmc-accent); font-size: 11px; opacity: .8; }
/* Said once, next to the dimmed prompt it is talking about. */
.mmc-refined-lede { color: var(--mmc-dim); font-size: 11px; margin-top: -4px; }
.mmc-refined-box {
  width: 100%; box-sizing: border-box; resize: vertical;
  background: var(--mmc-surface); border: 1px solid var(--mmc-line); border-radius: 12px;
  color: var(--mmc-text); font-family: inherit; font-size: 13px; line-height: 1.5;
  padding: 10px 12px; outline: none;
}
.mmc-refined-box:focus { border-color: rgba(255,255,255,.2); }
.mmc-refined-fold { font-size: 12px; color: var(--mmc-dim); }
.mmc-refined-fold summary { cursor: pointer; padding: 2px 0; }
.mmc-refined-sections { display: flex; flex-direction: column; gap: 8px; padding-top: 8px; }
.mmc-refined-section { display: flex; flex-direction: column; gap: 4px; }
/* A readout, not a field: it is the model's own account of the pictures and
   editing it would change nothing that gets queued. */
.mmc-refined-seen {
  padding: 8px 12px; border-left: 2px solid var(--mmc-line);
  white-space: pre-wrap; user-select: text;
}

/* ---- pre-stage --------------------------------------------------------------
   The image node wears the Creator's clothes: same tokens, same pills, same
   chip vocabulary. Only what it does not share is styled here. */

/* A plain textarea, not the contenteditable PromptBox: an image prompt has no
   @-handles to chip. Dressed exactly like the timeline's prompt box. */
.mmc-prestage-prompt {
  width: 100%; box-sizing: border-box; min-height: 96px; resize: vertical;
  background: var(--mmc-surface); border: 1px solid var(--mmc-line); border-radius: 14px;
  color: var(--mmc-text); font-family: inherit; font-size: 14px; line-height: 1.5;
  padding: 14px 16px; outline: none;
}
.mmc-prestage-prompt:focus { border-color: rgba(255,255,255,.2); }
.mmc-prestage-prompt::placeholder { color: var(--mmc-off); }

/* The spawn pill. On, it wears the accent the continue pill wears — the
   pre-stage is part of this shot now, which is a stronger statement than the
   accelerators' blue "not native". */
.mmc-prestage-toggle.on { border-color: rgba(240,166,60,.5); color: var(--mmc-accent); }
.mmc-prestage-toggle.on:hover:not(:disabled) { border-color: rgba(240,166,60,.8); }

/* The left-hand satellite anchors on its right edge (satellite.js sets the
   transform); nothing else about the card changes side. */
.mmc-satellite-left { transform-origin: 100% 0; }

/* The hand-off chips on a finished still — real buttons in the readout row,
   dressed like the gallery chip so the overlay stays one vocabulary. The
   readout swallows the pointer (see above), so like the gallery chip these
   have to opt back in or they are pictures of buttons. */
.mmc-stage-send {
  pointer-events: auto;
  background: rgba(0,0,0,.55); border: 1px solid #4a4a4a; border-radius: 999px;
  padding: 3px 10px; cursor: pointer; font-family: inherit; font-size: 12px;
  color: #ededed;
}
.mmc-stage-send:hover { border-color: var(--mmc-accent); color: var(--mmc-accent); }

/* ---- the frame grab ---------------------------------------------------------
   The trim editor's scrubbing with a different ending; dressed like it too. */
.mmc-grab-card {
  display: flex; flex-direction: column; gap: 14px;
  width: min(720px, 92vw); padding: 20px 24px;
  background: var(--mmc-bg); border: 1px solid var(--mmc-line); border-radius: 18px;
  box-shadow: 0 24px 64px rgba(0,0,0,.55);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
  color: var(--mmc-text); font-size: 13px;
}
.mmc-grab-title { display: flex; align-items: center; gap: 8px; font-size: 14px; }
.mmc-grab-title svg { stroke: currentColor; fill: none; stroke-width: 1.6; }
.mmc-grab-stage {
  width: 100%; max-height: 46vh; object-fit: contain;
  background: #000; border-radius: 12px;
}
.mmc-grab-row { display: flex; align-items: center; gap: 10px; }
.mmc-grab-scrub { flex: 1; }
.mmc-grab-time { min-width: 64px; text-align: right; color: var(--mmc-dim); font-variant-numeric: tabular-nums; }
.mmc-grab-actions { display: flex; justify-content: flex-end; gap: 12px; }
.mmc-grab-actions .mmc-btn {
  padding: 8px 18px; border-radius: 999px; cursor: pointer; font-family: inherit;
  font-size: 13px; background: var(--mmc-surface-2); color: var(--mmc-text);
  border: 1px solid var(--mmc-line);
}
.mmc-grab-actions .mmc-btn:hover:not(:disabled) { border-color: rgba(255,255,255,.25); }
.mmc-grab-actions .mmc-btn-primary { background: var(--mmc-accent); color: #141414; border-color: transparent; }
.mmc-grab-actions .mmc-btn:disabled { opacity: .5; cursor: progress; }
`;

export function installStyles() {
  if (document.getElementById("mmc-styles")) return;
  const tag = document.createElement("style");
  tag.id = "mmc-styles";
  tag.textContent = CSS;
  document.head.appendChild(tag);
}
