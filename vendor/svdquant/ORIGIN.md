# Krea-2-SVDQuant-ComfyUI

- **Source:** https://github.com/alperktt/Krea-2-SVDQuant-ComfyUI
- **Commit:** `87641f181f0a70cc47abd1b1dba01852370d74c0` (`master`, 2026-08-01)
- **License:** Krea 2 Community License Agreement — see `LICENSE.md`
- **Copied:** 2026-08-11, `git archive HEAD` (tracked files only)

## Changes made here

No code changes: every `.py` is byte-identical to that commit.

`examples/` was dropped. It is ~100 MB of rank-sweep and fidelity-sheet
comparison renders — the argument for the format, not the format — and carrying
it would make this package's clone twenty times the size of its code. The
gallery is a click away at the source URL above.

## What this package uses

Two classes, both re-registered by `nodes_vendor.py` under `K2S_` ids:

| Their class | Our id | Where it is emitted |
|---|---|---|
| `svdquant_w4a4.Krea2SVDQuantW4A4Loader` | `K2S_SVDQuantW4A4Loader` | `render_image._emit_krea2`, in place of `UNETLoader` |
| `svdquant_lora.Krea2SVDQuantLoraLoader` | `K2S_SVDQuantLoraLoader` | the LoRA chain, in place of `LoraLoaderModelOnly` |

The rest of the tree — the offline `quantize_krea2.py` CLI, activation capture,
diagnostics, the sage-attention mask guard — is carried but not emitted. It is
here so the vendored tree stays a faithful copy that an upstream patch applies
to cleanly, and because the loaders import `quantize_krea2` for the layer-name
helpers (`is_target`, `leaf_name`, `detect_prefix`).

## Notes for the next update

- `Krea2SVDQuantW4A4Loader` takes **only** `model_name` — no `weight_dtype`.
  The dtype pill has nothing to say on this loader and the UI hides it.
- It returns `(MODEL, STRING)`; the emitter takes output 0 and drops the report.
- It loads **only** `--format svdq` checkpoints (the ones carrying `*.svdq_l1` /
  `*.svdq_l2` tensors). `w4a4` / `int8` / `fp8` checkpoints have no low-rank
  branch and belong on the stock `UNETLoader`.
- `Krea2SVDQuantLoraLoader`'s `adapters` options are prose, not identifiers
  (`"bypass (exact, slower)"`). `render_image` matches them by prefix rather
  than by literal, the same trick `accel._block_cache_kwargs` uses, so a
  reworded label does not break the emitter.
