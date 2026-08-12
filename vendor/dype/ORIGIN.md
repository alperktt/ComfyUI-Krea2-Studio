# ComfyUI-DyPE

- **Source:** https://github.com/wildminder/ComfyUI-DyPE
- **Commit:** `874f59066f4d2c6f7fbbd4149fe5bc6f07099aa3` (`main` tip)
- **License:** Apache-2.0 — see `LICENSE`
- **Copied:** 2026-08-12, `git clone --depth 1` (a real git repo, so its HEAD is
  trustworthy)

## Changes made here

None. `__init__.py` and the whole of `src/` are byte-identical.

`conftest.py`, `tests/`, `example_workflows/` and the icon are not carried.

## What this package uses

| Their class | Our id | Role |
|---|---|---|
| `DyPE_FLUX` | `K2S_DyPE` | Dynamic Position Extrapolation, for renders past the trained resolution |
| `SEGA` | `K2S_SEGA` | Spectral-Energy Guided Attention, sharpening driven by the latent's spectrum |

**These are V3 nodes** — `io.ComfyNode` subclasses with a `Schema` — unlike the
other four packs, which are dict-based V1 classes. That is why `nodes_vendor`
has two renaming paths: a V1 node's id and category are class attributes, a V3
node's live inside the schema its `define_schema` returns, and the loader reads
the schema rather than the registry key. Ours rebuilds the schema with our id on
it, hidden through `is_dev_only` — the same field the pack's own internal save
node uses.

## Krea 2 is not one of its listed architectures, and works anyway

`model_type` offers `auto | flux | nunchaku | qwen | zimage | anima`. There is no
`krea2`, and at this commit the string "krea" does not appear anywhere in the
source. The repository *description* names Krea2; the code does not.

It still works, and the reason is worth writing down because it is load-bearing:

- **`SingleStreamDiT` carries `pe_embedder`, and it is literally Flux's.**
  `comfy/ldm/krea2/model.py` does `from comfy.ldm.flux.layers import EmbedND` and
  builds `self.pe_embedder = EmbedND(...)`. DyPE's flux branch patches
  `diffusion_model.pe_embedder`, which is exactly that object.
- **Auto-detection lands on flux by elimination.** The class name is not
  `QwenImage` or `Anima`; there is no `rope_embedder`, no `model.pos_embed`, no
  `pos_embedder`. So the `else` branch runs, and the `else` branch is flux.
- **The noise-schedule patch applies**, because Krea 2 is `ModelType.FLUX` and
  therefore gets a `ModelSamplingFlux`, which is what DyPE checks for before
  recomputing the shift.

One rough edge: DyPE reads `diffusion_model.patch_size`, and Krea 2 calls the
same thing `self.patch`. The read raises, DyPE catches it, logs "Could not read
patch_size from model (defaulting to 2)", and carries on — and 2 is Krea 2's
actual patch size, so the default is right by coincidence. **If Krea 2 ever ships
a variant with a different patch size, DyPE will silently use 2.** Worth checking
here first if a high-resolution render starts coming out geometrically wrong.

## Notes for the next update

- Both nodes take `width` and `height` and their tooltips say they must match the
  empty latent. This package passes the resolved canvas, so they cannot drift.
- `enable_dype` exists as a boolean *and* the pack removes its schedule patch
  when it is false. This package does not emit the node at all when the pill is
  off, which is stronger — nothing to undo.
- DyPE and SEGA are independent patches over the same embedder and compose in
  that order; `patch_utils` has separate `apply_dype_to_model` and
  `apply_sega_to_model` entry points.
- `base_resolution` defaults to 1024 and means the native training resolution.
  Krea 2's is 1024, so the default is correct here rather than coincidental.
