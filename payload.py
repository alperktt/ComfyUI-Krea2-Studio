"""A keyframe and a reference in the same generation.

The open weights turn out to accept reference audio on the FL2VA checkpoint —
its documented input conditions are text and frames only, but the packed
sequence has `ref_audio` rows and the model reads them, the same way LTX takes
an audio conditioning track. That is what makes a timeline's sound able to
continue across a seam instead of restarting: the previous segment's audio tail
goes in as an audio reference while its last frame goes in as the keyframe.

Core cannot currently express both at once, because of one line:

    keyframes = kwargs.get("minimax_keyframes", None)
    if keyframes is not None:
        payload["cond_video_latents"] = [kf["latent"] for kf in keyframes]
    refs = kwargs.get("minimax_refs", None)
    if refs is not None:
        payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]

    -- comfy/model_base.py, MiniMaxH3.extra_conds

The reference branch *overwrites* the keyframe branch rather than extending it.
Set both and an audio-only reference list — which has no `latent` key in it —
replaces the keyframe latents with an empty list, so the layout still lays out
`cond` rows for the keyframe and the DiT gets nothing to put in them.

`PackedLayout` itself is fine with the combination: it emits `cond` rows for the
keyframes and then `ref_img` / `ref_audio` rows for the references, and the
forward pass walks `cond`, `ref_img` and `video` segments off one running
`video_embed` offset. So the list only has to be in that same order —
keyframes first, then whatever reference images follow — and everything lines up.

Rather than patch core, this installs a diffusion-model wrapper that rebuilds the
list from the payload's own `keyframes` and `refs` just before the forward. It is
a repair, not a behaviour: with no keyframes or no refs it reproduces exactly
what `extra_conds` already computed, so it is safe to leave on.
"""

import comfy.patcher_extension

WRAPPER_KEY = "minimax_creator_cond_video_latents"


def _rebuild(payload):
    """`cond_video_latents` in layout order: keyframes, then reference images."""
    latents = [kf["latent"] for kf in payload.get("keyframes") or []]
    latents += [ref["latent"] for ref in payload.get("refs") or [] if "latent" in ref]
    return latents


def _wrapper(executor, *args, **kwargs):
    payload = kwargs.get("minimax_payload")
    # Only when both are present. Either one alone is already correct, and
    # rewriting it would be this module claiming a behaviour it does not have.
    if payload and payload.get("keyframes") and payload.get("refs"):
        payload = dict(payload)
        payload["cond_video_latents"] = _rebuild(payload)
        kwargs = {**kwargs, "minimax_payload": payload}
    return executor(*args, **kwargs)


def repair(model):
    """A clone of `model` whose keyframe latents survive a reference list.

    Keyed, so applying it twice — the Timeline node patches per segment — leaves
    one wrapper rather than a stack of identical ones.
    """
    patched = model.clone()
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, _wrapper)
    return patched
