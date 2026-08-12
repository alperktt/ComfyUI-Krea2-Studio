# Krea-2-Two-Stage-Sampler

- **Source:** https://github.com/Auryg/Krea-2-Two-Stage-Sampler
- **Commit:** `8194c366a9dea530e57ba7cc8ae77f0dff262baa` (`main` tip)
- **License:** MIT — see `LICENSE`
- **Copied:** 2026-08-12, `git clone --depth 1` (this one *is* a git repo, so its
  own HEAD is trustworthy — unlike the moodboard and style packs, see their
  ORIGIN files)

## Changes made here

None. `__init__.py` is byte-identical; the whole pack is that one file.

`example_workflows/` and `images/` are not carried.

## What this package uses

| Their class | Our id | Role |
|---|---|---|
| `KreaTwoStageSampler` | `K2S_KreaTwoStageSampler` | replaces `KSampler` for a two-stage run |
| `KreaThreeStageSampler` | `K2S_KreaThreeStageSampler` | three stages; a subclass of the above |

`KreaDualResolutionSelector` and `Krea2ModelSampling` are not carried into the
graph. The first picks a canvas, which `compile_image.resolve_canvas` already
owns — a second opinion about the size is a second place for it to be wrong. The
second is a `ModelSamplingFlux` preset, and the reference branch already sets
that shift where it belongs.

## What it is for

Running several steps of the undistilled base model before finishing on the
distilled one. The base gives real variation between seeds, which Turbo — being
a distillation — largely does not; Turbo then finishes cheaply. `handoff_percent`
is where the schedule crosses over.

Three stages is base → Turbo → base, and stage 3 reuses **all** of stage 1's
settings and its model. `stage3_handoff_percent` must be at or above
`handoff_percent`.

## How this package maps its own controls onto it

The node carries a full sampler row per stage. This package has one row of
widgets, so:

- **Stage 1 takes the widget row** — steps, cfg, sampler, scheduler. It is the
  row the user can see, and stage 1 is the pass that decides the image.
- **Stage 2 takes the Turbo preset** — `compile_image.KREA_TURBO` for cfg,
  sampler and scheduler, and `TURBO_STEPS[quality]` for its step count. Which is
  what the turbo pill has always meant: how fast the distilled pass runs.

So with a multi-stage run the turbo pill stops choosing a checkpoint — stage 2 is
*always* the Turbo file — and keeps choosing that stage's step budget. The
frontend switches the pill off and restores the base row when a stage count above
one is chosen, because a widget row already rewritten to Turbo's 8 steps at cfg 1
is the wrong row for stage 1.

## Notes for the next update

- `stage1_model` and `stage2_model` are two separate MODEL inputs, so a
  multi-stage render needs **both** the base and the Turbo checkpoint picked.
  This is why `compile_image` refuses a multi-stage run on the SVDQuant loader:
  that loader names one file, and mixing precisions across the handoff is not
  something the pack describes.
- `final_width` / `final_height` default to `0`, meaning "do not resize". Set
  together they let stage 1 sample small and stage 2 finish at the target — the
  dual-resolution route, wired here through the `stage1_scale` control.
- There is **no `denoise` input**. Every stage starts from the given latent at
  full strength, which is why img2img and a multi-stage run are mutually
  exclusive rather than merely awkward.
- `upscale_method` defaults to `bislerp` and only matters when the two
  resolutions differ.
