"""Conditioning + AV latent for a compiled request.

This is a re-dispatch of core's `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo`
against `Compiled` instead of against node sockets. The sizing and payload
helpers are imported from core rather than copied, so upstream fixes to the
canvas math or the reference presentation reach us without a re-port.

The reference path does not decide its own ordering. It executes
`compiled.plan`, the same walk `compile.py` numbered `<Picture N>` / `<Video N>`
/ `<Audio N>` from, one step at a time. That is deliberate: a mis-binding
between the labels in the prompt and the tensors in the payload produces a
subtly wrong video rather than an exception, so the two sides are built from one
list instead of two loops that have to be kept in agreement by hand.
"""

import math

import node_helpers
from comfy_extras.nodes_minimax_h3 import (
    CANVAS_MULTIPLE,
    FPS,
    REF_IMAGE_SHORT_EDGE,
    MiniMaxH3ReferenceToVideo,
    _empty_av_latent,
    _resize,
    adapt_canvas,
)

_encode_ref_audio = MiniMaxH3ReferenceToVideo._encode_ref_audio

# Where a timeline segment's inherited start frame arrives in `loaded`. It is the
# previous segment's decoded last frame, so unlike every other entry it has no
# Asset and no filename — a reserved key rather than a handle, because handles
# are the user's namespace and this frame is not something they attached.
PREV_FRAME = "__prev__"

# Where a timeline segment's inherited audio tail arrives. Same reasoning as
# PREV_FRAME: it is the previous segment's *generated* sound, so there is no file
# and no handle behind it.
PREV_AUDIO = "__prev_audio__"


def encode(clip, vae, audio_vae, compiled, loaded):
    """-> (conditioning, latent). `loaded` maps asset handle -> decoded media."""
    if compiled.mode == "REF2VA":
        return _encode_references(clip, vae, audio_vae, compiled, loaded)
    return _encode_frames(clip, vae, audio_vae, compiled, loaded)


def _encode_frames(clip, vae, audio_vae, compiled, loaded):
    """T2VA / I2VA / L2VA / FL2VA, optionally continuing the previous sound.

    The sound continuation is the one thing here core has no node for: the
    previous segment's audio tail rides in as a `ref_audio` block, which the
    FL2VA weights read even though their documented inputs are text and frames.
    See `payload.py` for the one core line that has to be worked around to send
    it alongside a keyframe.
    """
    latent, frame_count = _empty_av_latent(compiled.width, compiled.height, compiled.frames)

    images = []
    keyframes = []

    if compiled.continues:
        # The previous segment's last frame. It was generated on this same canvas
        # — the timeline pins one geometry across every segment — so the resize
        # is a no-op that exists only so a hand-built request cannot skip it.
        image = _resize(loaded[PREV_FRAME]["image"], compiled.width, compiled.height, "center")
        images.append(image)
        keyframes.append({"resolved_frame_index": 0, "image": image})
    elif compiled.first_frame is not None:
        # Geometry anchor: plain stretch, because the canvas was derived from
        # this image's own aspect ratio and already matches it.
        image = _resize(loaded[compiled.first_frame.handle]["image"], compiled.width, compiled.height, "disabled")
        images.append(image)
        keyframes.append({"resolved_frame_index": 0, "image": image})

    if compiled.last_frame is not None:
        # Follower: cover-crop onto whatever canvas the first frame established.
        # Follower whenever something already set the canvas — a first frame, or
        # in a timeline the frame inherited from the previous segment.
        crop = "center" if (compiled.first_frame is not None or compiled.continues) else "disabled"
        image = _resize(loaded[compiled.last_frame.handle]["image"], compiled.width, compiled.height, crop)
        images.append(image)
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": image})

    if compiled.continues_audio:
        # The tokenizer's `images=` branch is an `else` on `minimax_ref_items`:
        # pass both and the keyframes vanish from the presentation. So when there
        # is an audio reference to send, the keyframes are presented as reference
        # items instead. The two branches emit the same "<Picture N>: " + vision
        # tokens, so this is the same presentation by a different road — and the
        # keyframe *latents* still go in through `minimax_keyframes`, which is
        # what makes them pinned frames rather than loose references.
        items = [{"type": "image", "data": image} for image in images]
        items.append({"type": "audio"})
        tokens = clip.tokenize(compiled.prompt, minimax_ref_items=items)
    else:
        tokens = clip.tokenize(compiled.prompt, images=images)
    cond = clip.encode_from_tokens_scheduled(tokens)

    if keyframes:
        for keyframe in keyframes:
            keyframe["latent"] = vae.encode(keyframe.pop("image"))
        cond = node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        })

    if compiled.continues_audio:
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, loaded[PREV_AUDIO]["audio"])
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": [
            {"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent},
        ]})
    return cond, latent


def _snap(value):
    return max(CANVAS_MULTIPLE, round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def video_canvas(source_w, source_h, gen_w, gen_h, ref_size):
    """What a reference video is encoded at. -> (width, height).

    'max' is core's own reference canvas: a 768 short edge under a 768*1344 area
    cap, or the clip's native size when that is already smaller. It is the
    ceiling — unlike a reference image, whose 'max' reaches for 2048, a video
    never gets more than this, so the setting only ever buys speed.

    'match' takes the generation's pixel area instead, scaled down from whatever
    'max' would have used and keeping the clip's own aspect. Down-only and
    measured against the 'max' canvas rather than the source, which is what makes
    it impossible for 'match' to come out the more expensive of the two.

    Worth the knob because of how a video block is shaped: it is `latent_t`
    copies of this grid, not one, so at full length a single reference clip is
    about as long as the target video itself and every row of it rides through
    every sampling step.
    """
    width, height = adapt_canvas(source_w, source_h)
    if source_w * source_h < width * height:
        width, height = _snap(source_w), _snap(source_h)
    if ref_size == "match":
        scale = min(1.0, math.sqrt((gen_w * gen_h) / (width * height)))
        width, height = _snap(width * scale), _snap(height * scale)
    return width, height


def _encode_references(clip, vae, audio_vae, compiled, loaded):
    """REF2VA."""
    latent, frame_count = _empty_av_latent(compiled.width, compiled.height, compiled.frames)

    items = []   # tokenizer presentation, in request order
    blocks = []  # DiT payload, same order
    pending_soundtrack = None  # set by a 'soundtrack' step, consumed by the 'video' step after it

    for step in compiled.plan:
        asset = step["asset"]
        entry = loaded[asset.handle]

        if step["op"] == "image":
            image = entry["image"]
            height, width = image.shape[1], image.shape[2]
            if asset.ref_size == "match":
                # Down-only, to the generation's pixel area.
                scale = min(1.0, math.sqrt((compiled.width * compiled.height) / (width * height)))
            else:
                # 'max': the reference pipeline's own 2048 short edge. Best identity
                # retention, and several times slower — reference tokens ride through
                # every sampling step.
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(width, height))
            target_w, target_h = _snap(width * scale), _snap(height * scale)
            resized = _resize(image, target_w, target_h, "disabled")
            items.append({"type": "image", "data": resized})
            blocks.append({
                "kind": "image",
                "latent_h": target_h // 16,
                "latent_w": target_w // 16,
                "latent": vae.encode(resized),
            })

        elif step["op"] == "soundtrack":
            pending_soundtrack = _encode_ref_audio(audio_vae, entry["audio"])
            items.append({"type": "audio"})

        elif step["op"] == "video":
            frames = entry["frames"]
            source_h, source_w = frames.shape[1], frames.shape[2]
            canvas_w, canvas_h = video_canvas(
                source_w, source_h, compiled.width, compiled.height, asset.ref_size)
            frames = _resize(frames, canvas_w, canvas_h, "disabled")

            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            count = frames.shape[0]
            if count < 5:
                raise ValueError(
                    f"@{asset.handle}: reference videos need at least 5 frames "
                    f"(~0.2 s at 24 fps), got {count}"
                )
            while count % 17 != 5:
                count -= 1
            frames = frames[:count]

            audio_latent, ref_audio_t = pending_soundtrack or (None, 0)
            pending_soundtrack = None

            # Qwen sees the clip at 2 fps with timestamps, not every frame.
            sampled = list(range(0, frames.shape[0], FPS // 2))
            items.append({
                "type": "video",
                "data": frames[sampled],
                "timestamps": [i / 2.0 for i in range(len(sampled))],
            })
            encoded = vae.encode(frames)
            blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": encoded.shape[2],
                "latent_h": canvas_h // 16,
                "latent_w": canvas_w // 16,
                "ref_audio_t": ref_audio_t,
                "latent": encoded,
                "audio_latent": audio_latent,
            })

        elif step["op"] == "audio":
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, entry["audio"])
            items.append({"type": "audio"})
            blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        else:
            raise ValueError(f"unknown reference plan step {step['op']!r}")

    tokens = clip.tokenize(compiled.prompt, minimax_ref_items=items)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if blocks:
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": blocks})
    return cond, latent
