// Pre-stage and the frame grab.
// No backticks or ${} anywhere in the CSS: each chunk is one template literal.
export const css = `
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

/* ---- the moodboard picker ----------------------------------------------------
   A grid of thumbnails rather than a list of titles: "Abyssal Gothic Surrealism"
   is not a description you can act on, and the catalog ships a 256px image for
   every board. Built from the same tokens as everything else — surface-2 tiles,
   the line border, the accent for the chosen one — so it reads as this panel
   rather than beside it. */
.mmc-board-pop { width: 520px; }
.mmc-board-facets { margin: 6px 0 2px; }
/* The facet chips are pills at a smaller size: there are nine of them and they
   are a filter, not a control the eye should land on first. */
.mmc-board-facets .mmc-pill { height: 26px; padding: 0 10px; font-size: 12px; border-radius: 13px; }
.mmc-board-results { min-height: 120px; }
.mmc-board-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
  max-height: 300px; overflow-y: auto; padding: 2px;
}
.mmc-board-card {
  display: flex; flex-direction: column; gap: 5px; padding: 0; cursor: pointer;
  background: none; border: 0; font-family: inherit; color: var(--mmc-dim);
  text-align: left;
}
.mmc-board-thumb {
  width: 100%; aspect-ratio: 1; object-fit: cover; display: block;
  border-radius: 10px; background: var(--mmc-surface-2);
  border: 1px solid var(--mmc-line);
}
/* Two lines, clamped: the titles run long and a card that sizes to its text
   would make the rows of the grid disagree about their height. */
.mmc-board-title {
  font-size: 11px; line-height: 1.3; overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.mmc-board-card:hover { color: var(--mmc-text); }
.mmc-board-card:hover .mmc-board-thumb { border-color: rgba(255,255,255,.28); }
.mmc-board-card.picked { color: var(--mmc-accent); }
.mmc-board-card.picked .mmc-board-thumb { border-color: var(--mmc-accent); }
.mmc-board-foot {
  display: flex; align-items: center; justify-content: space-between;
  gap: 8px; margin-top: 8px;
}

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
