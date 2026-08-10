"""One still from the video model, as a graph. The PreStage's H3 branch.

`render.py` emits a video render and `render_image.py` emits an image-model
render; this emits a video render that stops at the first latent frame:

    loaders -> segment -> [preview] -> KSampler -> still slice -> VAEDecode -> save

Every node in that line except the slice is one the video path already uses, and
the segment node is *the* video segment node — same conditioning, same reference
ordering, same LoRA patching, same FL2VA/Ref2VA routing. That reuse is the whole
argument for the branch: a still made here is made by the weights that will
render the shot it is a keyframe for, at the canvas that shot will run at, with
no second model family loaded to get there.

Two things it does not take from `render.py`. There is no audio *decode*: the
sampled latent's audio half is generated and dropped, and the audio VAE is only
loaded when something attached has to be *encoded* into the conditioning — a
reference clip's soundtrack, which a still can cite exactly as a shot can. And
there is no chaining, because there is nothing to chain: each pass here is
independent, and more than one of them only happens under the dev sweep.

**The dev sweep.** `compile_still` may hand over several passes and several
decodes per pass; the loop below is written for that shape so the sweep needs no
second emitter. In normal use it is one pass with one decode and the loop runs
once. See `compile_still`'s DEV block for what comes out later.
"""

import json

from . import models, outputs, render

SEGMENT_NODE = "MiniMaxH3TimelineSegment"
STILL_NODE = "MiniMaxH3StillLatent"
SAVE_NODE = "MiniMaxH3SaveImage"

FILENAME_PREFIX = outputs.IMAGE_PREFIX


def weights_from_blob(data):
    """`models.Weights` for the pre-stage's MiniMax sub-block.

    The pre-stage keeps one sub-block per architecture so switching the model
    pill never forgets the other side's files, and the H3 side's fields are the
    video node's fields under the video node's names — so this lifts the
    sub-block into the shape `models.Weights.from_blob` already reads rather
    than teaching it a second one.
    """
    block = (data or {}).get("models")
    if not isinstance(block, dict):
        block = {}
    side = block.get("minimax")
    if not isinstance(side, dict):
        side = {}
    lifted = dict(side)
    for shared in ("dtype", "devices"):
        if block.get(shared) is not None and lifted.get(shared) is None:
            lifted[shared] = block[shared]
    return models.Weights.from_blob({"models": lifted})


def _label(index, total):
    """What one pass is called in any error raised about it."""
    return "This still" if total == 1 else f"Still {index + 1}"


def emit(plan, weights, sampling, unique_id, filename_prefix=FILENAME_PREFIX):
    """-> the graph, which the caller finalizes with `render.expanded`.

    `sampling` is a `render.Sampling`, under the same widget names the two video
    nodes use. Every pass samples on the *same* seed rather than seed+index, the
    opposite of `render.emit`'s rule and for the opposite reason: consecutive
    segments of a video want different noise, where the passes of a sweep are
    only comparable if they share it.
    """
    from comfy_execution.graph_utils import GraphBuilder

    labels = [_label(i, len(plan.passes)) for i in range(len(plan.passes))]
    payloads = [weights.routed(one.payload) for one in plan.passes]
    # `render`'s own two helpers: the same early compile a video render does, so
    # a request that cannot compile fails before a loader is built, and only the
    # checkpoint it actually routes to gets one.
    compiled = render.compile_all(payloads, labels)
    where = render.routed(compiled, labels)
    # A still decodes no sound, but it can *cite* some: a reference audio clip,
    # or a reference video taken with its soundtrack, is encoded into the
    # conditioning exactly as it is for a video render. Read off the compiled
    # requests rather than the blob, so what decides is what the encoder will
    # actually reach for. Nothing attached, nothing loaded.
    audio = any(one.ref_audios or any(v.track == "picture+sound" for v in one.ref_videos)
                for one in compiled)
    models.check(weights, set(where), where, audio=audio)

    graph = GraphBuilder()
    links = models.emit_links(graph, weights, set(where), audio=audio)
    # One loader per distinct override, shared across the decodes that ask for
    # it — a sweep comparing two decoders should load each of them once.
    extra_vaes = {}

    for index, one in enumerate(plan.passes):
        inputs = {
            "clip": links.clip, "vae": links.vae,
            # sort_keys so an unchanged payload serialises identically every
            # time — this string is the segment node's cache key.
            "segment_data": json.dumps(payloads[index], sort_keys=True),
        }
        if links.audio_vae is not None:
            inputs["audio_vae"] = links.audio_vae
        if links.model_fl2va is not None:
            inputs["model_fl2va"] = links.model_fl2va
        if links.model_ref2va is not None:
            inputs["model_ref2va"] = links.model_ref2va
        segment = graph.node(SEGMENT_NODE, **inputs)

        # The distilled H3 checkpoints run at cfg 1.0, where the negative is
        # skipped outright — the same zeroed conditioning the video path uses.
        against = graph.node("ConditioningZeroOut", conditioning=segment.out(1)).out(0)
        # taeh3 in the node body, exactly as on a video render. The preview is a
        # clip of the whole sampled latent, not of the frame that will be kept:
        # watching the motion is how you see the still is going somewhere.
        model = models.graph_preview(graph, segment.out(0), weights)

        sampled = graph.node(
            "KSampler", model=model, positive=segment.out(1), negative=against,
            latent_image=segment.out(2), seed=sampling.seed, steps=sampling.steps,
            cfg=sampling.cfg, sampler_name=sampling.sampler_name,
            scheduler=sampling.scheduler, denoise=1.0,
        )

        for decode in one.decodes:
            still = graph.node(STILL_NODE, samples=sampled.out(0),
                               index=decode.index).out(0)
            vae = links.vae
            if decode.vae:                                          # DEV
                if decode.vae not in extra_vaes:
                    extra_vaes[decode.vae] = graph.node(
                        "VAELoader", vae_name=decode.vae).out(0)
                vae = extra_vaes[decode.vae]
            image = graph.node("VAEDecode", samples=still, vae=vae).out(0)
            prefix = f"{filename_prefix}_{decode.label}" if decode.label else filename_prefix
            save = graph.node(SAVE_NODE, images=image, filename_prefix=prefix)
            # The save node lives in an expanded graph on nobody's canvas; the
            # stamp files its result under the PreStage the user is looking at,
            # which is what lets the stage card show the still it just made.
            save.set_override_display_id(unique_id)

    return graph
