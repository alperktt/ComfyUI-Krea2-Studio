# ComfyUI-Krea2-StyleTransfer

- **Source:** https://github.com/nkxx188/ComfyUI-Krea2-StyleTransfer
- **Commit:** `b30d495ab7e5626a2effc72a071430297643b718` (`master` tip, 2026-07-21)
- **License:** MIT — see `LICENSE`
- **Copied:** 2026-08-11, from the local install, then verified against upstream

**How the commit was established.** Like the moodboard pack, this one ships as a
plain directory rather than a git checkout, so `git -C <dir> rev-parse HEAD`
answers with the enclosing repository's HEAD — ComfyUI's own — and not with
anything about this pack. The SHA above is upstream's tip at copy time, confirmed
by fetching `nodes.py` at that commit and diffing it against the vendored file:
2,579 lines each, zero hunks. Do that on the next update rather than trusting
`git -C`.

## Changes made here

None. `nodes.py` is line-for-line identical to that commit. The local install
carries a UTF-8 BOM and CRLF endings where upstream has neither, so a byte hash
differs while the content does not — compare decoded lines, not bytes.

`web/`, `docs/` and `workflows/` are not carried: the workflows wire this pack's
own node ids, which do not survive the `K2S_` rename, and the web extension
belongs to nodes this package does not register.

## What this package uses

| Their class | Our id | Role |
|---|---|---|
| `Krea2StyleReference` | `K2S_Krea2StyleReference` | reference image → reference latent, fitted to the target grid |
| `Krea2StyleTransfer` | `K2S_Krea2StyleTransfer` | one reference: patches the model |
| `Krea2TwoStyleReferences` | `K2S_Krea2TwoStyleReferences` | two reference latents → a `STYLE_REFS` bundle |
| `Krea2TwoStyleTransfer` | `K2S_Krea2TwoStyleTransfer` | two references: patches the model |

`Krea2SizePreset` is not carried into the graph — this package resolves its own
canvas in `compile_image.resolve_canvas`, and a second opinion about the size
would be a second place for it to be wrong.

## The wiring, taken from the shipped workflow

```
Krea2StyleReference(vae, target_latent, reference_image, fit, upscale_method)
    -> reference_latent
Krea2StyleTransfer(model, reference_latent, ref_conditioning, mode, ...)
    -> (model, rf_reference, debug)
KSampler(model=<patched>, positive=<the same encode>, latent_image=<target_latent>)
```

Two things worth knowing from that trace:

- **`ref_conditioning` is the render's own positive conditioning**, not a second
  prompt. The workflow feeds the same `CLIPTextEncode` to both the transfer node
  and the sampler.
- **`target_latent` is the sampler's latent**, and the sampler still starts from
  it — the `rf_reference` output is not what it samples. So this is a model
  patch, and the latent path is unchanged.

## `mode` and the fourteen dials

Both transfer nodes take `mode: recommended | custom`. In `recommended` they
apply their own `_RECOMMENDED` table and ignore the advanced widgets — including
`style_strength`, whose tooltip says so outright ("Recommended mode fixes this at
1.0").

So the emitter sends `recommended` unless the user has moved the strength, and
`custom` when they have, filling the rest from `accel.node_defaults` — which
reads the installed class's own declared defaults and therefore follows a retune
instead of freezing it. Fourteen dials in the UI would be noise; one that works
is better than fourteen that are mostly the table.

## Notes for the next update

- `Krea2StyleTransfer` returns `(MODEL, LATENT, STRING)` and
  `Krea2TwoStyleTransfer` returns `(MODEL, STRING)` — different arities. The
  emitter takes output 0 from each; do not assume they match.
- `fit` is `crop | contain | stretch`, defaulting to `crop`. This package
  already scales the reference to the target canvas, so `crop` is close to a
  no-op — it is exposed because a reference whose aspect differs sharply from the
  render's is a real case and cropping is not always the right answer.
- The blocks string (`"7-27"`) is part of the recommended table and is not
  exposed. It names which DiT blocks the reference path touches.
