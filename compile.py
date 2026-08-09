"""`creator_data` (the UI's JSON blob) -> a validated, ordered generation request.

This is the load-bearing module. The `@chip` a user types in the prompt box is
not decoration: it is how they bind a slot to a role ("use @img-2 for her
face"), and H3 only understands that binding through its own ordinal labels,
`<Picture N>` / `<Video N>` / `<Audio N>`.

So the ordinals assigned here MUST match the order the encoder presents
references to the tokenizer, or `<Picture 3>` in the prompt points at the wrong
tensor and the failure is silent — a slightly-wrong video, not an exception.
That order is: images, then videos (each video's soundtrack emitting its
`<Audio j>` label *before* its own `<Video k>`), then standalone audio, with
ordinals counted 1-based per type across that sequence. `encode.py` walks the
lists this module produces in exactly that order; the two must be read together.
"""

import re
from dataclasses import dataclass, field, replace

from . import canvas, contextir

MODES = ("T2VA", "I2VA", "L2VA", "FL2VA", "REF2VA")

MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
MAX_REF_FILES = 12

# A timeline segment is a whole generation, so this is a cap on how many sampler
# passes one queue can expand into. High enough to never be the reason a real
# timeline is refused, low enough that a malformed blob does not run for a day.
MAX_SEGMENTS = 24

# How much of the previous segment's sound is handed to the next one.
#
# Not the whole thing, for two reasons that point the same way. A reference audio
# block costs `40 * seconds * 2` rows in the packed sequence, which ride through
# every sampling step. And an audio reference advances the layout's RoPE cursor
# by its own length, pushing the target's time origin away from the keyframe cond
# rows — which stay pinned at the text — so a long tail turns the inherited start
# frame from "this is frame 0" into "this is from some seconds earlier".
#
# A short tail is also all a seam needs: what carries across a cut is the room
# tone, the key and the tempo, not the phrase.
DEFAULT_AUDIO_TAIL_S = 1.0
MAX_AUDIO_TAIL_S = 4.0

HANDLE_RE = re.compile(r"@([A-Za-z]+-\d+)")

CHECKPOINTS = ("fl2va", "ref2va")

# Which of a reference video's streams are actually referenced. "sound" drops the
# picture entirely: the file becomes an audio reference like any other, which is
# how you cite a clip's soundtrack without also citing how it looks.
TRACKS = ("picture", "picture+sound", "sound")

# What a reference is encoded at when the blob does not say, per kind.
#
# Both entries are the behaviour that shipped before the setting reached video,
# so a blob written without one is read exactly as it used to be. They differ
# because `max` means something different for each: an image's is the reference
# pipeline's 2048 short edge, a video's is core's 768-short-edge reference
# canvas, which is already the ceiling — for video the setting only ever buys
# speed, never more detail than it had.
DEFAULT_REF_SIZE = {"image": "match", "video": "max"}

# What of a reference image is actually the reference. "full" — the default and
# the only behaviour that existed before the setting — is the whole picture.
# The others narrow it: a "person" reference contributes the person's likeness
# and nothing else, so the picture's background, palette and pose stop bleeding
# into the target video the moment the user says "her from @img-1". The DiT is
# handed the same tensor either way — the narrowing lives in the prose, which
# is where H3's reference form expresses it (`retention_analysis`) — so the
# field is read by the refiner's glossary and by nothing on the encode path.
TAKES = ("full", "person", "object", "scene", "style")


class CompileError(ValueError):
    """A `creator_data` blob that cannot become a valid H3 request."""


@dataclass
class Asset:
    handle: str          # "img-1", "vid-1", "aud-1" — what the user types after @
    kind: str            # image | video | audio
    role: str            # reference | first_frame | last_frame
    filename: str        # relative to ComfyUI/input
    track: str | None = None   # video only: one of TRACKS; None for images and audio
    ref_size: str = "match"    # reference image/video: match | max; see DEFAULT_REF_SIZE
    trim: tuple[float, float] | None = None   # video/audio only: (start, end) seconds; None = whole file
    takes: str = "full"        # reference image only: one of TAKES; what of it is the reference


@dataclass
class CanvasSpec:
    """A resolved canvas, so a timeline can hold every segment to one geometry.

    Segments are concatenated frame-by-frame at the end, which is only defined if
    they all came out the same size. So the first segment resolves the canvas
    the way a lone generation would — from its own keyframe, if it has one — and
    every segment after it is compiled against that answer rather than its own.
    """
    width: int
    height: int
    ratio: float
    label: str
    from_image: bool
    clamped: bool


@dataclass
class Compiled:
    mode: str
    checkpoint: str                 # fl2va | ref2va — which MODEL input is routed out
    prompt: str                     # the composed Context-IR: instruction + sections
    body: str                       # just the description, @handles and triggers resolved
    frames: int
    seconds: float                  # the real duration, not the pill's whole number
    width: int
    height: int
    ratio: float
    ratio_label: str
    ratio_from_image: bool
    ratio_clamped: bool
    soundscape: str = ""            # overall_soundscape, as written
    music: str = ""                 # non_diegetic_music, as written
    first_frame: Asset | None = None
    last_frame: Asset | None = None
    ref_images: list[Asset] = field(default_factory=list)
    ref_videos: list[Asset] = field(default_factory=list)
    ref_audios: list[Asset] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)   # handle -> "<Picture 1>"
    plan: list[dict] = field(default_factory=list)         # REF2VA only; see plan_references
    triggers: list[str] = field(default_factory=list)      # already prefixed onto `prompt`
    checkpoint_pinned: bool = False                        # the user chose it; not derived
    # Timeline only: the first frame is the previous segment's last frame, which
    # is a tensor produced mid-graph and so has no Asset and no filename.
    continues: bool = False
    # Timeline only: the previous segment's audio tail rides in as a reference so
    # the sound carries across the seam. Independent of `continues` — a hard cut
    # whose music keeps playing is an ordinary thing to want.
    continues_audio: bool = False
    audio_tail_s: float = 0.0


def lora_modes(entry):
    """The checkpoints a LoRA entry claims. Missing or unrecognised means both."""
    claimed = tuple(m for m in (entry.get("modes") or ()) if m in CHECKPOINTS)
    return claimed or CHECKPOINTS


def active_loras(entries, checkpoint):
    """The entries that will actually be patched onto `checkpoint`, in order.

    Lives here rather than in `lora.py` so that the trigger words and the weights
    can never disagree about which LoRAs are in the run — `lora.py` imports this.
    This module stays free of torch and ComfyUI, which is also what keeps it
    testable.
    """
    active = []
    for entry in entries or []:
        if not entry.get("name") or entry.get("enabled") is False:
            continue
        if checkpoint not in lora_modes(entry):
            continue
        try:
            if float(entry.get("strength", 1.0)) == 0.0:
                continue
        except (TypeError, ValueError):
            raise CompileError(f"LoRA {entry['name']}: strength must be a number")
        active.append(entry)
    return active


def collect_triggers(entries):
    """The trigger words of the LoRAs in the run, in order, deduped.

    Deduped case-insensitively but kept in the casing they were written with: two
    LoRAs from the same family routinely share a token, and repeating it in the
    prompt would weight it twice for no reason the user asked for.
    """
    triggers = []
    seen = set()
    for entry in entries:
        for word in entry.get("triggers") or ():
            word = str(word).strip()
            if not word or word.lower() in seen:
                continue
            seen.add(word.lower())
            triggers.append(word)
    return triggers


def _parse_trim(handle, kind, raw):
    """`{"start": s, "end": s}` -> (start, end) seconds on the source timeline.

    Absent means the whole file, which is the default everywhere. Only time-based
    media can be cut: a still has no timeline to cut on, and silently accepting a
    trim on one would hide a mistake in the blob.
    """
    if raw is None:
        return None
    if kind not in ("video", "audio"):
        raise CompileError(f"@{handle}: only video and audio can be trimmed")
    try:
        start = float(raw["start"])
        end = float(raw["end"])
    except (TypeError, KeyError, ValueError) as exc:
        raise CompileError(f"@{handle}: trim needs numeric 'start' and 'end' seconds") from exc
    if start < 0 or end <= start:
        raise CompileError(f"@{handle}: trim must satisfy 0 <= start < end (got {start} .. {end})")
    return (start, end)


def _parse_assets(raw):
    assets = []
    seen = set()
    for index, item in enumerate(raw or []):
        handle = str(item.get("handle") or "").strip()
        if not handle:
            raise CompileError(f"asset #{index + 1} has no handle")
        if handle in seen:
            raise CompileError(f"duplicate asset handle @{handle}")
        seen.add(handle)

        kind = item.get("kind")
        if kind not in ("image", "video", "audio"):
            raise CompileError(f"@{handle}: unknown kind {kind!r}")

        role = item.get("role", "reference")
        if role not in ("reference", "first_frame", "last_frame"):
            raise CompileError(f"@{handle}: unknown role {role!r}")
        if role != "reference" and kind != "image":
            raise CompileError(f"@{handle}: only images can be a {role}")

        filename = str(item.get("filename") or "").strip()
        if not filename:
            raise CompileError(f"@{handle}: no filename")

        # Defaulted per kind rather than globally — see DEFAULT_REF_SIZE. Audio
        # has no size to speak of and is left on the dataclass default, which
        # nothing downstream reads.
        ref_size = item.get("ref_size") or DEFAULT_REF_SIZE.get(kind, "match")
        if ref_size not in ("match", "max"):
            raise CompileError(f"@{handle}: ref_size must be 'match' or 'max'")

        # Only a reference image has anything to narrow: a keyframe is bound
        # whole by the alignment line, and video/audio narrowing is `track`'s
        # job. Refused rather than ignored, so a blob claiming a person-only
        # end frame does not queue quietly meaning something else.
        takes = item.get("takes") or "full"
        if takes not in TAKES:
            raise CompileError(
                f"@{handle}: takes must be one of {', '.join(TAKES)} (got {takes!r})")
        if takes != "full" and (kind != "image" or role != "reference"):
            raise CompileError(
                f"@{handle}: 'takes' narrows a reference image; a {role.replace('_', ' ')} "
                f"{kind} is always used whole"
            )

        assets.append(Asset(
            handle=handle,
            kind=kind,
            role=role,
            filename=filename,
            track=_parse_track(handle, kind, item),
            ref_size=ref_size,
            trim=_parse_trim(handle, kind, item.get("trim")),
            takes=takes,
        ))
    return assets


def _parse_track(handle, kind, item):
    """Which streams of a reference video are referenced. Video only.

    Blobs written before the picture/sound split carry the `with_audio` boolean
    instead; it says the same thing about the two states it could express, so it
    is read as one rather than being migrated on disk.
    """
    if kind != "video":
        if item.get("track"):
            raise CompileError(f"@{handle}: only video has a track selection")
        return None
    track = item.get("track") or ("picture+sound" if item.get("with_audio") else "picture")
    if track not in TRACKS:
        raise CompileError(f"@{handle}: unknown track {track!r}")
    return track


def _derive_mode(first_frame, last_frame, ref_images, ref_videos, ref_audios, continues=False):
    has_refs = bool(ref_images or ref_videos or ref_audios)
    has_frames = first_frame is not None or last_frame is not None or continues

    # Continuing *is* having a start frame — it is the previous segment's last
    # one — so a segment cannot also name a file for the slot.
    if continues and first_frame is not None:
        raise CompileError(
            "this segment continues from the previous one, so its start frame is "
            "already the previous segment's last frame — remove the start frame "
            "or turn continuation off"
        )

    # FL2VA and Ref2VA are two different DiT checkpoints. The hosted API can mix
    # a start frame with references in one call; the open weights cannot, so the
    # UI greys one out against the other. Refuse loudly rather than silently
    # dropping whichever the user cared about less.
    if has_refs and has_frames:
        raise CompileError(
            ("continuing from the previous segment's last frame is a keyframe "
             "generation (FL2VA) and references need the Ref2VA checkpoint, so "
             "one segment cannot do both. Turn continuation off, or remove the "
             "references."
             ) if continues else
            ("Start/end frames and references need different checkpoints "
             "(FL2VA vs Ref2VA) and cannot be combined in one generation. "
             "Remove the frames or remove the references.")
        )

    if has_refs:
        if len(ref_images) > MAX_REF_IMAGES:
            raise CompileError(f"at most {MAX_REF_IMAGES} reference images ({len(ref_images)} given)")
        if len(ref_videos) > MAX_REF_VIDEOS:
            raise CompileError(f"at most {MAX_REF_VIDEOS} reference videos ({len(ref_videos)} given)")
        total_audio = len(ref_audios) + sum(1 for v in ref_videos if v.track == "picture+sound")
        if total_audio > MAX_REF_AUDIOS:
            raise CompileError(
                f"at most {MAX_REF_AUDIOS} reference audio clips, counting video "
                f"soundtracks ({total_audio} given)"
            )
        total = len(ref_images) + len(ref_videos) + total_audio
        if total > MAX_REF_FILES:
            raise CompileError(f"at most {MAX_REF_FILES} reference files total ({total} given)")
        if not ref_images and not ref_videos:
            # Per the model card: audio is never a standalone reference.
            raise CompileError("reference audio needs at least one reference image or video alongside it")
        return "REF2VA"

    if continues:
        # The inherited frame fills the first slot, so this is I2VA on its own
        # and FL2VA once the segment also names an end frame.
        return "FL2VA" if last_frame is not None else "I2VA"
    if first_frame is not None and last_frame is not None:
        return "FL2VA"
    if first_frame is not None:
        return "I2VA"
    if last_frame is not None:
        return "L2VA"
    return "T2VA"


def _resolve_checkpoint(mode, raw):
    """Which weights the generation runs on, given the mode and the user's pin.

    The mode says how the request is *encoded*; the checkpoint says which weights
    it is encoded for. Those normally follow each other, but not always: FL2VA
    and Ref2VA are two trainings of one architecture, so keyframe conditioning is
    a payload Ref2VA can also take, and running start/end frames through it is a
    legitimate thing to want.

    The reverse is not a preference. Reference blocks have nothing to attend to
    in FL2VA, and the result would be a quietly wrong video rather than an error,
    so pinning against a REF2VA request is refused here.
    """
    choice = raw or "auto"
    if choice not in ("auto",) + CHECKPOINTS:
        raise CompileError(f"unknown checkpoint {choice!r}")
    derived = "ref2va" if mode == "REF2VA" else "fl2va"
    if choice == "auto":
        return derived, False
    if mode == "REF2VA" and choice == "fl2va":
        raise CompileError(
            "references are encoded for the Ref2VA checkpoint and cannot be run "
            "through FL2VA — set the route back to auto or Ref2VA, or remove the "
            "references"
        )
    return choice, choice != derived


def plan_references(ref_images, ref_videos, ref_audios):
    """The one ordered walk that both the labels and the DiT payload come from.

    `encode.py` executes this plan step by step rather than re-deriving the order
    from the three lists, so the ordinals in the prompt and the tensors in the
    payload cannot drift apart — there is only one order and both sides read it
    from here.

    A reference video with a soundtrack produces two steps: its `soundtrack`
    comes first and takes an `<Audio j>`, then the video itself takes its
    `<Video k>`. That is the presentation order the tokenizer expects.

    A video referenced for its sound alone never reaches `ref_videos` at all —
    it arrives in `ref_audios` and is walked as a plain audio reference.
    """
    plan = []
    picture = video = audio = 0

    for asset in ref_images:
        picture += 1
        plan.append({"op": "image", "asset": asset, "label": f"<Picture {picture}>"})
    for asset in ref_videos:
        if asset.track == "picture+sound":
            audio += 1
            plan.append({"op": "soundtrack", "asset": asset, "label": f"<Audio {audio}>"})
        video += 1
        plan.append({"op": "video", "asset": asset, "label": f"<Video {video}>"})
    for asset in ref_audios:
        audio += 1
        plan.append({"op": "audio", "asset": asset, "label": f"<Audio {audio}>"})
    return plan


def _labels_from_plan(plan):
    """handle -> label. A video with a soundtrack owns two, so the soundtrack's
    is keyed `"<handle>:audio"`."""
    labels = {}
    for step in plan:
        key = step["asset"].handle
        if step["op"] == "soundtrack":
            key += ":audio"
        labels[key] = step["label"]
    return labels


def _keyframe_labels(first_frame, last_frame, continues=False):
    """handle -> `<Picture N>` for the keyframe modes.

    A continuing segment's start frame is a tensor from the previous segment, so
    it has no handle to map — but it is still presented to the tokenizer first
    and still consumes `<Picture 1>`. Counting it without keying it is what keeps
    an end frame in the same segment correctly labelled `<Picture 2>`.
    """
    labels = {}
    ordinal = 0
    if continues:
        ordinal += 1
    elif first_frame is not None:
        ordinal += 1
        labels[first_frame.handle] = f"<Picture {ordinal}>"
    if last_frame is not None:
        ordinal += 1
        labels[last_frame.handle] = f"<Picture {ordinal}>"
    return labels


def _substitute(prompt, labels, assets, where="prompt"):
    """Replace every `@handle` with its H3 label.

    Only handles that name a real asset are touched, so ordinary prose ("meet me
    @ 5") survives. A handle-shaped token with no asset behind it is an error:
    it means an asset was deleted and the prompt now refers to something that
    will not be in the payload.

    `where` names the field in the error, because this runs over more than the
    prompt: the refiner writes `@handles` into the reference sections and the
    two audio fields too, and they are substituted with the same labels.
    """
    known = {a.handle for a in assets}
    dangling = sorted({h for h in HANDLE_RE.findall(prompt) if h not in known})
    if dangling:
        raise CompileError(
            f"{where} references " + ", ".join("@" + h for h in dangling)
            + " but no such asset is attached"
        )
    return HANDLE_RE.sub(lambda m: labels.get(m.group(1), m.group(0)), prompt)


def refined_body(data):
    """The refiner's prose for this request, or None if it is not to be used.

    A refined body is stored with its `@handles` intact rather than with H3's
    ordinals in it, which is what lets it be treated as an ordinary prompt from
    here on: `_substitute` assigns the labels at queue time exactly as it does
    for typed text, so adding or removing an asset re-labels a refined prompt
    correctly instead of leaving it pointing at the tensor that used to be there.

    `enabled: false` is the toggle in the panel — the rewrite is kept so it can
    be switched back on, and the user's own sentence is used meanwhile.
    """
    refined = data.get("refined")
    if not isinstance(refined, dict) or refined.get("enabled") is False:
        return None
    body = str(refined.get("body") or "").strip()
    return body or None


def refined_sections(data):
    """The reference form's three extra sections, when a refiner wrote them."""
    refined = data.get("refined")
    if not isinstance(refined, dict) or refined.get("enabled") is False:
        return None
    sections = refined.get("sections")
    if not isinstance(sections, dict):
        return None
    kept = {name: str(sections.get(name) or "").strip() for name in contextir.REF_SECTIONS}
    return kept if any(kept.values()) else None


def audio_tail_seconds(raw):
    """The requested tail length in seconds, clamped to something sendable."""
    if raw is None:
        return DEFAULT_AUDIO_TAIL_S
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        raise CompileError(f"audio_tail_s must be a number (got {raw!r})")
    if seconds <= 0:
        raise CompileError("audio_tail_s must be greater than 0")
    return min(seconds, MAX_AUDIO_TAIL_S)


def compile_request(data, image_size_lookup=None, continues=False, canvas_spec=None,
                    continues_audio=False, shots=1):
    """`creator_data` dict -> `Compiled`.

    `image_size_lookup(filename) -> (width, height)` supplies the keyframe
    dimensions for the adaptive canvas in the image modes. It is injected so
    this module stays free of disk access and stays unit-testable.

    `continues` and `canvas_spec` are the timeline's two additions and are both
    off in the single-generation path: the first says the start frame arrives as
    a tensor from the previous segment, the second pins the geometry the first
    segment resolved onto every segment after it.
    """
    if not isinstance(data, dict):
        raise CompileError("creator_data must be a JSON object")

    assets = _parse_assets(data.get("assets"))

    frame_assets = [a for a in assets if a.role in ("first_frame", "last_frame")]
    for role in ("first_frame", "last_frame"):
        if sum(1 for a in frame_assets if a.role == role) > 1:
            raise CompileError(f"only one {role} is allowed")
    first_frame = next((a for a in frame_assets if a.role == "first_frame"), None)
    last_frame = next((a for a in frame_assets if a.role == "last_frame"), None)

    refs = [a for a in assets if a.role == "reference"]
    ref_images = [a for a in refs if a.kind == "image"]
    # A video referenced for its soundtrack alone is an audio reference and
    # nothing else: it takes an <Audio> label, no <Video> one, and its picture is
    # never encoded. Which bucket a file lands in is settled here, once, so the
    # limits, the plan and the loader all count it the same way.
    ref_videos = [a for a in refs if a.kind == "video" and a.track != "sound"]
    ref_audios = [a for a in refs if a.kind == "audio" or (a.kind == "video" and a.track == "sound")]

    mode = _derive_mode(first_frame, last_frame, ref_images, ref_videos, ref_audios, continues)

    # A REF2VA segment already fills `minimax_refs` with its own ordered plan, and
    # slotting the inherited sound into it means giving it an <Audio N> in that
    # numbering rather than a line of its own. Worth doing; not done here, and
    # refused rather than silently dropped so the seam never claims to carry
    # sound it did not send.
    if continues_audio and mode == "REF2VA":
        raise CompileError(
            "continuing the sound is not yet supported on a segment with @ references — "
            "the reference list owns the audio slots. Turn the sound seam off, or "
            "remove the references."
        )
    audio_tail_s = audio_tail_seconds(data.get("audio_tail_s")) if continues_audio else 0.0

    checkpoint, pinned = _resolve_checkpoint(mode, data.get("checkpoint"))
    if mode == "REF2VA":
        plan = plan_references(ref_images, ref_videos, ref_audios)
        labels = _labels_from_plan(plan)
    else:
        plan = []
        labels = _keyframe_labels(first_frame, last_frame, continues)

    # The refiner's prose stands in for the user's sentence and is substituted the
    # same way — it holds the same `@handles`, which is the whole reason it is
    # stored in that form. Switching the panel's toggle off falls back here
    # rather than anywhere downstream, so nothing else has to know it exists.
    body = _substitute(refined_body(data) or str(data.get("prompt") or ""), labels, assets)

    # Trigger words come from the LoRAs that are actually in this run — an entry
    # set to the other checkpoint contributes neither weights nor words. Keyed on
    # the routed checkpoint rather than the mode, so pinning moves the words and
    # the weights together. They go in front, which is the convention every LoRA
    # is documented against, and after substitution because they are literal
    # words with no @handles in them.
    #
    # In front of the *body*, not of the finished prompt: the keyframe-alignment
    # instruction has to be the prompt's first line, so words prefixed above it
    # would push it out of position.
    triggers = collect_triggers(active_loras(data.get("loras"), checkpoint))
    if triggers:
        prefix = ", ".join(triggers)
        body = f"{prefix}, {body}" if body.strip() else prefix

    seconds_shown = data.get("duration_s", 6)
    frames = canvas.frames_for_seconds(seconds_shown)
    short_edge = data.get("short_edge", canvas.NATIVE_SHORT_EDGE)

    # The instruction line carries the real duration to two decimals, so this has
    # to come after the frame count and never off `duration_s`.
    #
    # Substituted like the body: the reference form cites `<Audio N>` in the
    # soundscape, and the refiner stores that citation as `@aud-1` exactly as it
    # does in a shot body.
    soundscape = _substitute(str(data.get("soundscape") or ""), labels, assets,
                             where="overall_soundscape")
    music = _substitute(str(data.get("music") or ""), labels, assets,
                        where="non_diegetic_music")
    sections = refined_sections(data)
    if sections:
        sections = {name: _substitute(text, labels, assets, where=name)
                    for name, text in sections.items()}
    prompt = contextir.compose(
        mode, body, soundscape, music, canvas.seconds_for_frames(frames),
        # The inherited tail is presented to the tokenizer as <Audio 1>, so the
        # prompt has to say what it is or the label points at nothing. Phrased
        # the way the reference guide defines its own labels.
        preamble=contextir.AUDIO_SEAM_LINE if continues_audio else "",
        # Which shot the end frame is reached by. A one-pass render says so
        # outright — it assembled the description and counted the cards' shots —
        # and any body that numbers its own shots says so by carrying them, which
        # is a refined Creator prompt with the model's cuts in it. The larger
        # wins: a card may write several shots inside the one the timeline
        # allotted it, and the last of those is the one holding the end frame.
        shots=max(int(shots or 1), contextir.count_shots(body)),
        # Only ever a refiner's: the reference form's other three sections cannot
        # be derived from a sentence, so without them a REF2VA body is left
        # exactly as it has always been.
        sections=sections)

    # In the image modes the aspect comes from the keyframe (the hosted API calls
    # this "adaptive"); the slider still owns the scale. The first frame wins
    # when both are set, because the encoder treats it as the geometry anchor
    # and cover-crops the last frame onto the canvas it defines.
    #
    # A pinned canvas overrides the lot: within a timeline the geometry was
    # settled by segment 1 and every later segment has to land on it, or the
    # frames cannot be concatenated at the end. `continues` also removes the only
    # anchor a later segment could have had — the inherited frame is already at
    # the canvas size, so there is nothing to adapt to.
    anchor = first_frame or last_frame
    if canvas_spec is not None:
        width, height = canvas_spec.width, canvas_spec.height
        ratio, clamped, ratio_from_image = canvas_spec.ratio, canvas_spec.clamped, canvas_spec.from_image
    elif anchor is not None and image_size_lookup is not None:
        source_w, source_h = image_size_lookup(anchor.filename)
        width, height, ratio, clamped = canvas.canvas_from_image(source_w, source_h, short_edge)
        ratio_from_image = True
    else:
        label = data.get("aspect", "16:9")
        if label not in canvas.ASPECT_PRESETS:
            raise CompileError(f"unknown aspect ratio {label!r}")
        ratio = canvas.ASPECT_PRESETS[label]
        width, height = canvas.resolve_canvas(ratio, short_edge)
        clamped = False
        ratio_from_image = False

    return Compiled(
        mode=mode,
        checkpoint=checkpoint,
        checkpoint_pinned=pinned,
        prompt=prompt,
        body=body,
        soundscape=soundscape,
        music=music,
        frames=frames,
        seconds=canvas.seconds_for_frames(frames),
        width=width,
        height=height,
        ratio=ratio,
        ratio_label=canvas.describe_ratio(ratio),
        ratio_from_image=ratio_from_image,
        ratio_clamped=clamped,
        first_frame=first_frame,
        last_frame=last_frame,
        ref_images=ref_images,
        ref_videos=ref_videos,
        ref_audios=ref_audios,
        labels=labels,
        plan=plan,
        triggers=triggers,
        continues=continues,
        continues_audio=continues_audio,
        audio_tail_s=audio_tail_s,
    )


# ---- timeline ---------------------------------------------------------------

# How a timeline becomes video.
#
# "chained" is the original: every segment is its own generation and they are
# concatenated, so the clip can run to any length and a segment can start from
# the previous one's decoded last frame.
#
# "single" is one generation. The segments stop being separate renders and become
# the shots of one Context-IR description — `[Shot 2] At 00:05.000, ...` — which
# is the format the model documents and was trained on. Nothing is decoded and
# re-encoded in the middle, so there is no seam to carry a frame or a tail across
# and no roundtrip drift; the whole clip's picture and sound are generated at
# once. The price is that everything the pass can only have one of — mode,
# checkpoint, LoRA stack, seed — is now the timeline's rather than the segment's.
RENDER_MODES = ("chained", "single")

_HANDLE_PREFIX = {"image": "img", "video": "vid", "audio": "aud"}


def render_mode(data):
    """Which of `RENDER_MODES` a timeline blob asks for. Absent means chained."""
    mode = data.get("render") or "chained"
    if mode not in RENDER_MODES:
        raise CompileError(f"unknown render mode {mode!r}")
    return mode


def _join_prompt(global_prompt, segment_prompt):
    """The global prompt in front of the segment's own.

    Two lines rather than a comma splice: the global prompt is a standing
    description of the piece and the segment's is what happens in this shot, and
    running them together as one sentence reads as one clause qualifying the
    other. No `@handle` can appear in the global prompt — assets are per-segment
    in this pass — so `_substitute` sees the join as ordinary prose.
    """
    parts = [p for p in (global_prompt.strip(), str(segment_prompt or "").strip()) if p]
    return "\n".join(parts)


def merge_loras(global_entries, segment_entries):
    """The timeline's LoRAs plus a segment's own, as one stack.

    Global first, then the segment's, so a turbo LoRA meant for the whole piece
    is patched before anything a single shot adds. A segment naming a LoRA the
    timeline already carries replaces it rather than stacking a second copy of
    the same weights at two strengths — the more specific entry is the one the
    user was editing when they set it.

    Trigger words follow the entries, because `collect_triggers` walks whatever
    `active_loras` returns and this is what it will be handed.
    """
    segment_entries = list(segment_entries or [])
    named = {e.get("name") for e in segment_entries if isinstance(e, dict)}
    kept = [e for e in (global_entries or [])
            if isinstance(e, dict) and e.get("name") not in named]
    return kept + segment_entries


def timeline_segments(data):
    """The segment list off a timeline blob, validated. Shared by both render modes.

    `MAX_SEGMENTS` means two different things depending on the mode — sampler
    passes per queue when chained, shots in one description when single — but the
    number is a sanity bound on a hand-edited blob either way, so it is one cap.
    """
    if not isinstance(data, dict):
        raise CompileError("timeline_data must be a JSON object")

    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise CompileError("a timeline needs at least one segment")
    if len(segments) > MAX_SEGMENTS:
        raise CompileError(f"at most {MAX_SEGMENTS} segments ({len(segments)} given)")
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise CompileError(f"segment {index + 1} is not a JSON object")
    return segments


def timeline_payloads(data, image_size_lookup=None):
    """`timeline_data` dict -> one self-contained payload per segment, in play order.

    A payload is everything one segment needs and nothing the others do: a
    single-generation request with the global prompt already folded in, whether
    it starts from the previous segment's last frame, and the canvas the whole
    timeline is held to.

    Splitting before compiling is what makes the cache useful. The segments run
    as separate nodes and a node's cache key is its inputs, so handing each one
    the whole timeline would mean editing the last shot re-generated all of them.
    A payload only changes when its own segment does.
    """
    segments = timeline_segments(data)
    global_prompt = str(data.get("prompt") or "")
    payloads = []

    for index, segment in enumerate(segments):
        request = dict(segment)
        # Lifted out of the request and onto the payload: they are facts about
        # the seam in front of this segment, not about the generation.
        request.pop("continue", None)
        request.pop("continue_audio", None)
        request["prompt"] = _join_prompt(global_prompt, segment.get("prompt"))
        request["aspect"] = data.get("aspect", "16:9")
        request["short_edge"] = data.get("short_edge", canvas.NATIVE_SHORT_EDGE)
        request["loras"] = merge_loras(data.get("loras"), segment.get("loras"))
        # The soundscape and the score are properties of the piece, not of one
        # shot — a cut is not where the room tone changes. A segment may still
        # say its own; an empty one inherits rather than clearing.
        for key in ("soundscape", "music"):
            request[key] = str(segment.get(key) or data.get(key) or "")
        # The tail length is the timeline's — one seam sounding different from
        # the next is not a thing anyone tunes per cut.
        request["audio_tail_s"] = data.get("audio_tail_s", DEFAULT_AUDIO_TAIL_S)
        payloads.append({
            "request": request,
            # Segment 1 has nothing in front of it, so the flags are ignored there
            # rather than rejected: they are leftovers from reordering, not a
            # mistake worth refusing a whole timeline over.
            "continue": index > 0 and bool(segment.get("continue")),
            # Independent of the picture: a hard cut whose music keeps playing is
            # an ordinary thing to want, and so is a match cut that resets the
            # sound. Two switches on the seam rather than one with three states.
            "continue_audio": index > 0 and bool(segment.get("continue_audio")),
        })

    # Resolved the way a lone generation would resolve it — segment 1 may take
    # its aspect from its own keyframe — then imposed on the rest.
    try:
        first = compile_request(payloads[0]["request"], image_size_lookup)
    except CompileError as exc:
        raise CompileError(f"segment 1: {exc}") from exc
    spec = {
        "width": first.width, "height": first.height, "ratio": first.ratio,
        "label": first.ratio_label, "from_image": first.ratio_from_image,
        "clamped": first.ratio_clamped,
    }
    for payload in payloads:
        payload["canvas"] = dict(spec)
    return payloads


# ---- one pass ---------------------------------------------------------------


def _asset_dict(asset):
    """`Asset` -> the blob shape `_parse_assets` reads. The inverse of parsing.

    Only what differs from the default is written, so the merged request looks
    like something a user could have typed and diffs cleanly against a segment's
    own list.
    """
    out = {"handle": asset.handle, "kind": asset.kind, "role": asset.role,
           "filename": asset.filename}
    if asset.track:
        out["track"] = asset.track
    if asset.ref_size != "match":
        out["ref_size"] = asset.ref_size
    if asset.trim:
        out["trim"] = {"start": asset.trim[0], "end": asset.trim[1]}
    if asset.takes != "full":
        out["takes"] = asset.takes
    return out


def _agree(values, what, blank=""):
    """The one value the shots agree on, or an error naming the disagreement.

    One pass has one of each of these. A shot setting its own is a deliberate
    override in a chained timeline and simply has nowhere to go here, so it is
    refused rather than quietly resolved in favour of whichever shot came first.
    """
    distinct = [v for v in dict.fromkeys(values) if v and v != blank]
    if len(distinct) > 1:
        raise CompileError(
            f"the shots disagree about {what} ({', '.join(map(str, distinct))}) — "
            f"one pass has only one, so it has to be the same across the timeline"
        )
    return distinct[0] if distinct else blank


def single_payload(data):
    """`timeline_data` -> one payload, the whole timeline as a single generation.

    The segments stop being renders and become the shots of one Context-IR
    description. What that costs is everything a single pass can only have one
    of, and each of those is resolved here rather than deferred:

    - **One reference pool.** Handles are allocated per segment, so `img-1`
      means a different file in each of them. Every segment's attachments are
      merged — same file, role and trim is the same reference, which is the point:
      a face cited in shot 1 and again in shot 4 is one `<Picture N>` — and each
      shot's prompt is rewritten onto the merged handles before the labels are
      assigned. There is no second labelling scheme; `compile_request` does it.
    - **One keyframe pair.** A start frame opens the clip and an end frame closes
      it, so they belong to the first and last shot and nowhere else.
    - **One LoRA stack, one checkpoint, one soundscape.** Folded and checked for
      agreement rather than per shot.

    The seam flags are ignored, not refused: there are no seams in one pass, and
    a timeline switched over from chained mode carries them harmlessly.
    """
    segments = timeline_segments(data)
    last_index = len(segments)
    global_prompt = str(data.get("prompt") or "").strip()

    assets = []          # the merged reference list, in first-appearance order
    position_of = {}     # dedup key -> index into `assets`
    counters = {}        # kind -> how many merged handles of it exist
    shots = []           # (cut time in seconds, text) per shot
    at = 0.0
    stack = data.get("loras")
    with_refs, with_frames = [], []

    for number, segment in enumerate(segments, start=1):
        try:
            parsed = _parse_assets(segment.get("assets"))
        except CompileError as exc:
            raise CompileError(f"shot {number}: {exc}") from exc

        rename = {}
        for asset in parsed:
            if asset.role == "first_frame" and number != 1:
                raise CompileError(
                    f"shot {number} has a start frame, but one pass opens on shot 1 — "
                    f"a start frame is the video's first frame, so it can only be shot 1's"
                )
            if asset.role == "last_frame" and number != last_index:
                raise CompileError(
                    f"shot {number} has an end frame, but one pass ends on shot "
                    f"{last_index} — an end frame is the video's final frame, so it can "
                    f"only be the last shot's"
                )
            if asset.role == "reference":
                with_refs.append(number)
            else:
                with_frames.append(number)

            key = (asset.kind, asset.role, asset.filename, asset.track,
                   asset.ref_size, asset.trim, asset.takes)
            position = position_of.get(key)
            if position is None:
                counters[asset.kind] = counters.get(asset.kind, 0) + 1
                position = position_of[key] = len(assets)
                assets.append(replace(
                    asset, handle=f"{_HANDLE_PREFIX[asset.kind]}-{counters[asset.kind]}"))
            rename[asset.handle] = assets[position].handle

        # A refined shot replaces the typed one here rather than downstream,
        # because the merged request is a single generation and `compile_request`
        # would otherwise see one `refined` blob standing for the whole strip.
        # The refiner was shown the global prompt as the piece's standing
        # description, so a refined shot 1 has already absorbed it and is not
        # given it a second time.
        written = refined_body(segment)

        # One pass, single-pass substitution: a rename map applied in two passes
        # could turn this shot's img-1 into img-2 and then that into img-3.
        text = HANDLE_RE.sub(
            lambda m: "@" + rename.get(m.group(1), m.group(1)),
            written or str(segment.get("prompt") or ""),
        ).strip()
        if number == 1 and global_prompt and not written:
            # The standing description of the piece opens the description, which
            # is where the guide puts the style and the initial composition. A
            # terminator is added when the user left none, because without one the
            # two clauses run together into a sentence neither of them is.
            joiner = "" if global_prompt[-1] in ".!?,;:—" else "."
            text = f"{global_prompt}{joiner} {text}".strip()
        shots.append((at, text))
        at += float(segment.get("duration_s", 6) or 0)

        stack = merge_loras(stack, segment.get("loras"))

    # Named here rather than left to `_derive_mode`'s merged view, which can only
    # say that the timeline has both and not which shots to go and look at.
    if with_refs and with_frames:
        raise CompileError(
            f"shot {with_frames[0]} has a start/end frame and shot {with_refs[0]} has "
            f"references. Those are two different checkpoints (FL2VA vs Ref2VA) and one "
            f"pass runs on one of them, so a one-pass timeline cannot hold both."
        )

    try:
        body = contextir.shot_body(shots)
    except ValueError as exc:
        raise CompileError(str(exc)) from exc

    request = {
        "prompt": body,
        "assets": [_asset_dict(a) for a in assets],
        "loras": stack,
        # The whole clip is one generation, so there is one duration and it snaps
        # to the 17n+5 grid once, at the end — not per shot.
        "duration_s": at,
        "aspect": data.get("aspect", "16:9"),
        "short_edge": data.get("short_edge", canvas.NATIVE_SHORT_EDGE),
    }
    for key in ("soundscape", "music"):
        request[key] = _agree(
            [str(data.get(key) or "")] + [str(s.get(key) or "") for s in segments], key)
    # One pass is one reference pool, so the reference form's analysis sections
    # describe the whole clip and are the timeline's rather than a shot's. Only
    # the sections: the body is the assembled shot list above.
    sections = refined_sections(data)
    if sections:
        request["refined"] = {"sections": sections}
    pin = _agree([s.get("checkpoint") for s in segments], "the checkpoint", blank="auto")
    if pin != "auto":
        request["checkpoint"] = pin

    # `shots` is counted off the finished description rather than taken as the
    # number of cards, because a card may number several shots of its own. The
    # end frame is reached by the last of them, whichever card wrote it.
    #
    # No canvas: with nothing to concatenate there is nothing to hold to one
    # geometry, so a start frame sets the aspect adaptively exactly as it does in
    # a lone generation.
    return {"request": request, "shots": contextir.count_shots(body),
            "continue": False, "continue_audio": False}


def compile_segment(payload, image_size_lookup=None):
    """One payload from `timeline_payloads` or `single_payload` -> `Compiled`."""
    spec = payload.get("canvas")
    return compile_request(
        payload["request"], image_size_lookup,
        continues=bool(payload.get("continue")),
        continues_audio=bool(payload.get("continue_audio")),
        shots=int(payload.get("shots", 1)),
        canvas_spec=CanvasSpec(**spec) if spec else None)


def compile_single(data, image_size_lookup=None):
    """`timeline_data` -> the one `Compiled` a one-pass render generates."""
    return compile_segment(single_payload(data), image_size_lookup)


def compile_timeline(data, image_size_lookup=None):
    """`timeline_data` dict -> one `Compiled` per segment, in play order.

    Each segment is a whole generation, as capable as a lone Creator node —
    same references, same LoRAs, same checkpoint routing. Only three things are
    the timeline's rather than the segment's: the prompt every segment inherits,
    the canvas they must all share to be concatenable at the end, and whether a
    segment starts from the previous one's last frame.
    """
    compiled = []
    for index, payload in enumerate(timeline_payloads(data, image_size_lookup)):
        try:
            compiled.append(compile_segment(payload, image_size_lookup))
        except CompileError as exc:
            raise CompileError(f"segment {index + 1}: {exc}") from exc
    return compiled
