# comfyui-krea2edit

- **Source:** https://github.com/lbouaraba/comfyui-krea2edit
- **Commit:** `86f886dac23013d88996e3a2e99093ba44d322fb` (`main`, v1.2.5)
- **License:** Apache-2.0 — see `LICENSE`
- **Copied:** 2026-08-11

## Changes made here

None. `__init__.py` is byte-identical; the whole pack is that one file.

`tests/` and `workflows/` are not carried — the tests import the pack under its
own name and the workflows wire its own node ids, neither of which survives the
`K2S_` rename.

## What this package uses

| Their class | Our id | Where |
|---|---|---|
| `Krea2EditModelPatch` | `K2S_Krea2EditModelPatch` | after the LoRAs in `_emit_model`'s chain |
| `Krea2EditGroundedEncode` | `K2S_Krea2EditGroundedEncode` | replaces `CLIPTextEncode` on both branches |

## The two halves, and why both are wired

krea2edit is not one node with a helper. It is two halves of one training
recipe, and running either alone is running half of it:

- **`Krea2EditModelPatch`** is the *appearance* path. The source image's VAE
  latent is injected as frame=1 tokens, so the model sees the pixels it is
  editing. `ref_boost` multiplies target→reference attention on the last
  reference; `ref_boost_a` does the same for the first in a two-reference setup.
- **`Krea2EditGroundedEncode`** is the *semantic* path. Training always encoded
  the instruction together with the source through Qwen3-VL — user turn is
  `<vision tokens><instruction>`, twelve layers tapped. Stock `CLIPTextEncode`
  is text-only, so an edit run through it is missing the grounding half and
  cannot resolve "the man on the left" or "the sign in the back".

**The negative is grounded too.** The pack's own docstring says it: for CFG,
encode the negative through a second grounded node with an empty prompt and the
same image, which is what training's unconditional looked like. So the emitter
builds two of these rather than zeroing the positive.

## Notes for the next update

- **It needs the Krea 2 Identity Edit LoRA applied first** —
  `krea2_identity_edit_v1_2.safetensors` upstream. The patch node adds the
  in-context path; the LoRA is what was trained to use it. Hence the edit panel
  carries its own LoRA picker and the patch sits *after* the LoRA chain.
- **`source_latent` is a required socket this emitter deliberately does not
  feed.** `patch` takes its pixel path whenever `vae` and `source_image` are both
  connected — `if vae is not None and source_image is not None` — and that path
  re-encodes the source from pixels, discarding whatever `source_latent`
  carried. This package always connects both, so a `VAEEncode` wired there would
  run on every render and be thrown away; the emitter passes the target latent,
  which the graph already has. **If a future version stops overriding
  unconditionally, `_edit_sources` in `render_image.py` becomes wrong rather than
  merely wasteful — check this line first.**
- `vae` + `source_image` + `target_latent` together are the "blur-proof" path
  the README recommends: pixel-space fitting, and the source encoded before
  sampling starts rather than on the first step. All three are wired.
- `fit_mode` defaults to `fit`, which matches how the current weights were
  trained. `crop (legacy)` is for v1/v1.1 only.
- `grounding_px` caps the longest side fed to the VLM. The 2026-07-02 LoRA
  trained with 384–768 px jitter, so 768 (the node's default) is in
  distribution; 0 means native resolution.
- The node takes `**_future` and warns rather than failing when a workflow
  carries inputs it does not know — so a re-vendor that gains an input will not
  break graphs emitted by the older code.
