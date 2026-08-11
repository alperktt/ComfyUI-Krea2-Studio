# ComfyUI-Krea-Moodboards

- **Source:** https://github.com/Andro-Meta/ComfyUI-Krea-Moodboards
- **Commit:** `7d11ec31cb700d881fdf2d73731ecde0093b9540` (`master`)
- **License:** MIT — see `LICENSE`. Catalog provenance: `DATA_NOTICE.md`.
- **Copied:** 2026-08-11

## Changes made here

None to `moodboard_catalog.py`; it is byte-identical.

`nodes.py`, `web/` and `tools/` are not carried, and neither is
`data/thumb_cache/`. This package registers **no moodboard node at all** — see
below — so the node file would be dead code, and the thumbnails are a cache the
catalog rebuilds on demand.

## Why this one registers nothing

Every `KreaMoodboard*` node returns `STRING` and nothing else: a positive
fragment, a negative fragment, a title, some metadata. There is no model to
patch and no conditioning to build, so putting one in the graph would mean a
node whose entire job is to hand a piece of text to the node beside it.

`moodboard_catalog.py` is pure — `json`, `random`, `re`, `pathlib`, and no
torch, no ComfyUI, no folder_paths — so the catalog is read directly and the
chosen board's guidance is merged into the prompt in `compile_image.py`, where
trigger words are already merged. The blob names a board; the render carries
text. Nothing new reaches the graph.

That is also why `VENDORED` in `nodes_vendor.py` has no entry for this pack.

## The API this package uses

| Function | Used for |
|---|---|
| `load_catalog(path)` | the 3,549-board slim catalog, read once and cached |
| `find_board(catalog, value)` | resolve a title / slug / uuid / url / search phrase |
| `search_boards(...)` | the picker's search field |
| `style_from_board(board, strength=...)` | `{positive, negative, title, metadata_json}` |
| `mashup_boards(...)` | blending two or more boards into one style |
| `random_board(...)` | the dice button |

`strength` is one of `concise` / `normal` / `strong` and decides how much of the
board's prose is folded in — `STRENGTHS` in the source is the list.

## Notes for the next update

- The catalog ships as `data/krea_moodboards_slim.json` (9.3 MB, 3,549 boards)
  plus `data/andrometa_moodboards.json` (a smaller curated set). Both are read
  through `load_catalog`, which takes a path — do not hardcode either.
- `style_from_board` prefixes a guardrail sentence (`STYLE_GUARDRAIL`) unless
  `include_guardrail=False`. It is what keeps a board's prose from being read as
  a subject instead of a style, so leave it on.
