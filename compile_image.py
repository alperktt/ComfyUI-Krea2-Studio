"""Blob -> payload for the PreStage image node.

The image counterpart of `compile.py`, and held to the same rule: everything in
here is pure — no torch, no ComfyUI, no disk — so the frontend can mirror these
constants (`state.js`) and the tests can run without a GPU. The blob is the
frontend's serialised state exactly as `creator_data` is; this validates it and
reduces it to the payload `render_image.emit` builds a graph from.

Two architectures, both open weights, both native in core:

- **Krea 2** (12.9B DiT, Qwen3-VL-4B text encoder, Qwen Image VAE). RAW is the
  base checkpoint and samples like an ordinary CFG model; Turbo is an 8-step
  distillation that runs at cfg 1. Style references go through core's
  Qwen-edit encoder, which has exactly three image slots — that is where
  `MAX_STYLE_REFS` comes from, not taste.
- **Ideogram 4.0** (9.3B single-stream DiT, Qwen3-VL-8B). Sampled on its own
  resolution-shifted schedule with a *pair* of checkpoints — the unconditional
  branch is a separate model — so its knobs are the official preset table
  rather than a steps widget alone.

The prompt is plain natural language for both. Ideogram was trained on
structured JSON captions and its hosted magic-prompt expands text into that
schema, but for the art-directed work this pipeline feeds, a clean sentence
prompt reads better than the schema — so the schema is deliberately not
modelled here at all.
"""

from dataclasses import dataclass, field

from .compile import CompileError, collect_triggers

ARCHES = ("krea2", "ideogram4")
DEFAULT_ARCH = "krea2"

# The image DiTs take /16 canvases (their latent is a /8 downsample and both
# empty-latent nodes step by 16) — half as coarse as the video's /32.
CANVAS_MULTIPLE = 16

# Both models are comfortable up to a 2048x2048 area; past it Krea 2's own card
# stops and Ideogram's presets do too. The floor is where either stops making
# usable keyframes for the video pipeline.
MIN_SHORT_EDGE = 512
MAX_SHORT_EDGE = 2048
DEFAULT_SHORT_EDGE = 1024
MAX_PIXELS = 2048 * 2048

# Wider than the video envelope on purpose: a style sheet or a poster is a
# legitimate still even though no H3 render could take its shape.
MIN_RATIO = 1 / 3
MAX_RATIO = 3.0

ASPECT_PRESETS = {
    "21:9": 21 / 9,
    "16:9": 16 / 9,
    "3:2": 3 / 2,
    "4:3": 4 / 3,
    "1:1": 1.0,
    "3:4": 3 / 4,
    "2:3": 2 / 3,
    "9:16": 9 / 16,
}
DEFAULT_ASPECT = "16:9"

# Core's TextEncodeQwenImageEditPlus has three image inputs. The cap is that
# node's shape, mirrored here so the UI refuses a fourth instead of the graph.
MAX_STYLE_REFS = 3

# What each Krea 2 checkpoint wants from the sampler row. RAW is undistilled and
# runs real CFG; Turbo is distilled and runs at 1. These are what the turbo pill
# writes into the widgets and what a fresh node defaults to.
KREA_RAW = {"steps": 52, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple"}
KREA_TURBO = {"cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"}
TURBO_STEPS = {"draft": 4, "medium": 6, "good": 8}
DEFAULT_TURBO_QUALITY = "good"

# Which loader builds the DiT. "standard" is core's `UNETLoader` and is what
# this has always done; "svdquant" is the W4A4 loader vendored under
# `vendor/svdquant`, which reads a checkpoint carrying an SVDQuant low-rank
# branch (`*.svdq_l1` / `*.svdq_l2`) and runs the blocks at 4 bits.
#
# It is a separate field rather than a third entry in the turbo pill because
# the two say different things. The turbo pill picks a *distillation* — how many
# steps the schedule takes and at what cfg — and an SVDQuant checkpoint is
# quantized from RAW or from Turbo, so it still needs that answer. This picks a
# *precision*, and the file it names is its own.
LOADERS = ("standard", "svdquant")
DEFAULT_LOADER = "standard"

# What `Krea2SVDQuantLoraLoader` does with a LoRA that cannot fold into the
# low-rank branch — LoKr, LoHa, OFT. A plain LoRA is free either way, which is
# why this is per-entry and why the default is the exact one.
#
# These are our names, not the vendored node's: its options are prose
# ("bypass (exact, slower)"), matched by prefix at emit time so a reworded label
# does not break the graph.
ADAPTER_MODES = ("bypass", "bake")
DEFAULT_ADAPTER = "bypass"

# How much of a moodboard's prose is folded into the prompt. The vendored
# catalog's own three levels, named here so `state.js` can mirror them.
MOODBOARD_STRENGTHS = ("concise", "normal", "strong")
DEFAULT_MOODBOARD_STRENGTH = "normal"
MOODBOARD_COLLECTIONS = ("krea", "andrometa")
DEFAULT_MOODBOARD_COLLECTION = "krea"

# krea2edit: the in-context edit path. `fit` resamples the source onto the target
# grid the way the current weights were trained; `crop (legacy)` is v1/v1.1
# geometry and only belongs with those older weights. The strings are the
# vendored node's own option values, so they are passed through verbatim.
EDIT_FIT_MODES = ("fit", "crop (legacy)")
DEFAULT_EDIT_FIT = "fit"

# How much of the source the VLM sees, as a cap on its longest side. The edit
# LoRA trained with 384-768 px jitter, so the node's own 768 is in distribution;
# 0 means native resolution, which the jitter makes tolerable too.
DEFAULT_GROUNDING_PX = 768
MAX_GROUNDING_PX = 4096

# `ref_boost` multiplies target->reference attention.
#
# **4.0, not the node's own 1.0.** The node defaults to 1.0 because that is what
# v1.1 did and the input was added without changing old graphs; the v1.2 release
# workflow ships it at 4 and its note says why — "4.0 = recommended (pre-set):
# much stronger face + body likeness, more reliable edits", against "1.0 =
# classic v1.1 behavior". Taking the node's default would quietly hand everyone
# the old behaviour, so this follows the release rather than the input.
#
# Above ~10 the note reports over-copying: removals and replacements start
# failing because the reference is pulled in too hard. Below 1 suppresses the
# reference for creative freedom. The ceiling is the node's own.
DEFAULT_REF_BOOST = 4.0
REF_BOOST_OVERCOPY = 10.0
MAX_REF_BOOST = 1000.0

# Where the edit weights work best, in pixels of output. The release workflow's
# notes: "1MP is the sweet spot; go higher only for single-person edits" and
# "Inputs around 1MP work best. Two-person images: stay at/below ~1.5MP."
#
# Not a clamp. The resolution pill is the user's, and an edit at 2 MP renders —
# it is just slower and looser than the same edit at 1 MP. So the payload
# carries the observation and the UI says it.
EDIT_SWEET_SPOT_PIXELS = 1024 * 1024
EDIT_TWO_REF_MAX_PIXELS = 1536 * 1024

# Krea2-StyleTransfer: RF-inversion style transfer. One reference patches the
# model through `Krea2StyleTransfer`; two go through a `STYLE_REFS` bundle and
# `Krea2TwoStyleTransfer`. Past two the pack has no route, so the cap is its
# shape rather than a preference.
MAX_STYLE_TRANSFER_REFS = 2

# How a reference whose aspect differs from the render's is made to fit. The
# pack's own options and default.
STYLE_FITS = ("crop", "contain", "stretch")
DEFAULT_STYLE_FIT = "crop"

# Overall style mix. The transfer nodes ignore this in `recommended` mode — their
# tooltip says so — so moving it is what switches them to `custom`, with every
# other dial still coming from the installed class's own defaults.
DEFAULT_STYLE_STRENGTH = 1.0
MAX_STYLE_STRENGTH = 2.0

# Krea-2-Two-Stage-Sampler: run some steps on the undistilled base for real
# variation between seeds, then finish on the distillation. 1 is off — the stock
# `KSampler`, unchanged.
STAGE_COUNTS = (1, 2, 3)
DEFAULT_STAGES = 1
# The pack's own handoff defaults. Three stages is base -> Turbo -> base, and the
# second crossover has to be at or after the first.
DEFAULT_HANDOFF = 16.67
DEFAULT_HANDOFF3 = 83.33

# How much smaller stage 1 samples than the final canvas. 1.0 is off — both
# stages at the target size. Below it, stage 1 runs on a smaller latent and
# stage 2 finishes at the target, which is the pack's dual-resolution route.
DEFAULT_STAGE1_SCALE = 1.0
MIN_STAGE1_SCALE = 0.25

# Ideogram's official preset table, verbatim from the shipped ComfyUI template
# (V4_QUALITY_48 / V4_DEFAULT_20 / V4_TURBO_12). mu and std shape the
# resolution-shifted schedule, so they belong to the preset, not to the user.
IDEOGRAM_QUALITIES = {
    "quality": {"steps": 48, "mu": 0.0, "std": 1.5},
    "default": {"steps": 20, "mu": 0.0, "std": 1.75},
    "turbo": {"steps": 12, "mu": 0.5, "std": 1.75},
}
DEFAULT_IDEOGRAM_QUALITY = "default"
# The template's guidance: cfg 7 for most of the trajectory, dropped to 3 over
# the last 30% so the fine steps stop over-sharpening. The 7 is the node's cfg
# widget; the late drop is constant wiring.
IDEOGRAM_CFG = 7.0
IDEOGRAM_CFG_LATE = {"cfg": 3.0, "start_percent": 0.7, "end_percent": 1.0}

# How much of the init image survives by default when one is attached. The same
# number the img2img tradition has always landed on: enough to keep the
# composition, enough noise to actually restyle it.
DEFAULT_DENOISE = 0.65
MIN_DENOISE = 0.05


@dataclass(frozen=True)
class ImagePayload:
    """What `render_image.emit` needs, and nothing the widgets already carry."""

    arch: str
    prompt: str
    width: int
    height: int
    # Which weights field the DiT loads from — "model", "turbo_model" or
    # "svdq_model". Resolved here rather than in the emitter so the payload
    # states which file runs.
    checkpoint_field: str
    # Which node loads it. "standard" is core's; "svdquant" is the vendored W4A4
    # loader, which also swaps every LoRA onto its own loader.
    loader: str = DEFAULT_LOADER
    loras: list = field(default_factory=list)        # [{"name", "strength", "adapters"}]
    refs: list = field(default_factory=list)         # filenames, krea2 only
    init: dict = None                                # {"filename", "denoise"} or None
    # The moodboard's negative fragment, or None. The positive half is already
    # in `prompt`; this one needs a conditioning of its own, so it travels
    # separately — see `render_image._emit_krea2`.
    negative_prompt: str = None
    # krea2edit, or None. `{source, source_b, lora, ref_boost, ref_boost_a,
    # fit_mode, grounding_px}` — see `_parse_edit`.
    edit: dict = None
    # RF-inversion style transfer, or None. `{refs, fit, strength, primary}` —
    # see `_parse_style`.
    style: dict = None
    # A multi-stage sampler run, or None when a single `KSampler` is enough.
    # `{count, handoff, handoff3, width, height}` — the two sizes are stage 1's
    # canvas; `ImagePayload.width/height` stays the final one.
    stages: dict = None
    # Set when an edit's canvas is past the size its weights work best at. Not a
    # refusal and not a clamp — the render is fine, just slower and looser than
    # the same edit at 1 MP, and the UI says so.
    edit_oversize: bool = False
    # Ideogram's schedule shape, None on krea2.
    mu: float = None
    std: float = None
    ratio_clamped: bool = False


def active_image_loras(entries):
    """The entries that will be patched on, in order, as the original dicts.

    The video pipeline's `active_loras` filters by checkpoint mode as well;
    image LoRAs have no such split — one DiT per arch — so this keeps only the
    enabled/strength rules and drops the modes machinery. Original dicts rather
    than a reduced copy so `collect_triggers` can still read them.
    """
    active = []
    for entry in entries or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        if entry.get("enabled") is False:
            continue
        try:
            strength = float(entry.get("strength", 1.0))
        except (TypeError, ValueError):
            raise CompileError(f"LoRA {entry['name']}: strength must be a number")
        if strength == 0.0:
            continue
        active.append(entry)
    return active


def _parse_adapters(raw):
    """A LoRA entry's adapter mode, defaulted and validated.

    Carried on every entry rather than only on the SVDQuant path so that
    flipping the loader pill back and forth does not lose what was chosen —
    the same reason the weights block keeps both architectures' files.
    """
    if raw is None:
        return DEFAULT_ADAPTER
    if raw not in ADAPTER_MODES:
        raise CompileError(
            f"unknown adapter mode {raw!r} — it is one of {', '.join(ADAPTER_MODES)}")
    return raw


def clamp_ratio(ratio):
    if ratio < MIN_RATIO:
        return MIN_RATIO, True
    if ratio > MAX_RATIO:
        return MAX_RATIO, True
    return ratio, False


def _snap(value):
    return max(CANVAS_MULTIPLE, int(value / CANVAS_MULTIPLE + 0.5) * CANVAS_MULTIPLE)


def resolve_canvas(ratio, short_edge):
    """(aspect ratio, slider short edge) -> the (width, height) generated.

    Same construction as `canvas.resolve_canvas` at the image models' /16 grid:
    the short edge is what the slider says, the long edge follows the ratio, and
    the area cap steps the long axis back down if snapping pushed past it.
    """
    ratio, _ = clamp_ratio(float(ratio))
    short_edge = max(MIN_SHORT_EDGE, min(MAX_SHORT_EDGE, int(short_edge)))

    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio

    if width * height > MAX_PIXELS:
        scale = (MAX_PIXELS / (width * height)) ** 0.5
        width, height = width * scale, height * scale
    # The long side is capped too: 2048 is the models' ceiling per axis, not
    # only as an area, and a 3:1 sheet at a big short edge would sail past it.
    if max(width, height) > MAX_SHORT_EDGE:
        scale = MAX_SHORT_EDGE / max(width, height)
        width, height = width * scale, height * scale

    width, height = _snap(width), _snap(height)
    while width * height > MAX_PIXELS and max(width, height) > CANVAS_MULTIPLE:
        if width >= height:
            width -= CANVAS_MULTIPLE
        else:
            height -= CANVAS_MULTIPLE
    return width, height


def _parse_init(raw):
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw.get("filename"):
        raise CompileError("the init image entry must carry a filename")
    try:
        denoise = float(raw.get("denoise", DEFAULT_DENOISE))
    except (TypeError, ValueError):
        raise CompileError("the init image strength must be a number")
    # A denoise of 1.0 with an init attached is a t2i render that quietly
    # ignored the image; below the floor it is the image with noise on it.
    # Clamped rather than refused: both ends are slider overshoot, not intent.
    denoise = max(MIN_DENOISE, min(1.0, denoise))
    return {"filename": raw["filename"], "denoise": denoise}


def _parse_refs(raw):
    refs = []
    for item in raw or []:
        filename = item.get("filename") if isinstance(item, dict) else item
        if not filename or not isinstance(filename, str):
            raise CompileError("every style reference must carry a filename")
        refs.append(filename)
    if len(refs) > MAX_STYLE_REFS:
        raise CompileError(
            f"at most {MAX_STYLE_REFS} style references — the Qwen edit encoder "
            f"the model reads them through has exactly three image slots"
        )
    return refs


def _parse_moodboard(raw):
    """The moodboard block, validated. `None` when the pill is off.

    Off is the default and off means the block never reaches the lookup, so a
    catalog that is missing or trimmed costs nothing to anyone who has not asked
    for a board.
    """
    if not isinstance(raw, dict) or not raw.get("on"):
        return None
    board = str(raw.get("board") or "").strip()
    if not board:
        raise CompileError("the moodboard pill is on but no board is chosen")
    strength = raw.get("strength", DEFAULT_MOODBOARD_STRENGTH)
    if strength not in MOODBOARD_STRENGTHS:
        raise CompileError(
            f"unknown moodboard strength {strength!r} — it is one of "
            f"{', '.join(MOODBOARD_STRENGTHS)}")
    collection = raw.get("collection", DEFAULT_MOODBOARD_COLLECTION)
    if collection not in MOODBOARD_COLLECTIONS:
        raise CompileError(f"unknown moodboard collection {collection!r}")
    return {"board": board, "strength": strength, "collection": collection,
            # A board's negative guidance is its own opinion about what the look
            # is not. Kept separable because at cfg 1 nothing reads it.
            "use_negative": raw.get("use_negative", True) is not False}


def _number(value, label, minimum, maximum, default):
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise CompileError(f"{label} must be a number")
    if not minimum <= number <= maximum:
        raise CompileError(f"{label} must be between {minimum} and {maximum}")
    return number


def _parse_edit(raw):
    """The krea2edit block, validated. `None` when the pill is off.

    Two references are the node's limit and its order is load-bearing: the
    training order is scene first, subject second, which is why the second slot
    is `source_b` rather than a list.
    """
    if not isinstance(raw, dict) or not raw.get("on"):
        return None

    def source(key, required):
        entry = raw.get(key)
        filename = entry.get("filename") if isinstance(entry, dict) else entry
        if not filename:
            if required:
                raise CompileError("the edit pill is on but no source image is chosen")
            return None
        if not isinstance(filename, str):
            raise CompileError(f"the edit {key} must be a filename")
        return filename

    fit_mode = raw.get("fit_mode", DEFAULT_EDIT_FIT)
    if fit_mode not in EDIT_FIT_MODES:
        raise CompileError(f"unknown edit fit mode {fit_mode!r}")

    grounding = raw.get("grounding_px", DEFAULT_GROUNDING_PX)
    try:
        grounding = int(grounding)
    except (TypeError, ValueError):
        raise CompileError("the grounding resolution must be a whole number")
    if not 0 <= grounding <= MAX_GROUNDING_PX:
        raise CompileError(f"the grounding resolution must be between 0 and {MAX_GROUNDING_PX}")

    lora = raw.get("lora")
    return {
        "source": source("source", True),
        "source_b": source("source_b", False),
        # The Identity Edit LoRA, patched before the model patch. Optional here
        # because it may equally well be in the main LoRA stack — see
        # `render_image._emit_edit_patch`.
        "lora": lora.strip() if isinstance(lora, str) and lora.strip() else None,
        "lora_strength": _number(raw.get("lora_strength"), "the edit LoRA strength",
                                 -10.0, 10.0, 1.0),
        "ref_boost": _number(raw.get("ref_boost"), "ref_boost", 0.0, MAX_REF_BOOST,
                             DEFAULT_REF_BOOST),
        # The scene dial, and it stays at 1: the release notes say to leave it
        # there unless you are exploring. Only the subject dial is pre-boosted.
        "ref_boost_a": _number(raw.get("ref_boost_a"), "ref_boost_a", 0.0, MAX_REF_BOOST, 1.0),
        "fit_mode": fit_mode,
        "grounding_px": grounding,
    }


def _parse_style(raw):
    """The style-transfer block, validated. `None` when the pill is off."""
    if not isinstance(raw, dict) or not raw.get("on"):
        return None
    refs = []
    for item in raw.get("refs") or []:
        filename = item.get("filename") if isinstance(item, dict) else item
        if not filename or not isinstance(filename, str):
            raise CompileError("every style-transfer reference must carry a filename")
        refs.append(filename)
    if not refs:
        raise CompileError("the style-transfer pill is on but no reference is chosen")
    if len(refs) > MAX_STYLE_TRANSFER_REFS:
        raise CompileError(
            f"at most {MAX_STYLE_TRANSFER_REFS} style-transfer references — the pack "
            f"has no route for a third")

    fit = raw.get("fit", DEFAULT_STYLE_FIT)
    if fit not in STYLE_FITS:
        raise CompileError(f"unknown style fit {fit!r}")

    strength = _number(raw.get("strength"), "the style strength", 0.0, MAX_STYLE_STRENGTH,
                       DEFAULT_STYLE_STRENGTH)

    # Which of the two references leads. Only meaningful with two, and the pack
    # takes it as the string "1" or "2".
    primary = raw.get("primary", 1)
    try:
        primary = int(primary)
    except (TypeError, ValueError):
        raise CompileError("the primary style reference must be 1 or 2")
    if primary not in (1, 2):
        raise CompileError("the primary style reference must be 1 or 2")

    return {"refs": refs, "fit": fit, "strength": strength, "primary": primary}


def _parse_stages(raw):
    """The multi-stage block, validated. `None` when one stage is enough.

    Returns the *count and crossovers* only; stage 1's canvas is resolved later,
    once the final one is known.
    """
    if not isinstance(raw, dict):
        return None
    count = raw.get("count", DEFAULT_STAGES)
    try:
        count = int(count)
    except (TypeError, ValueError):
        raise CompileError("the stage count must be a whole number")
    if count not in STAGE_COUNTS:
        raise CompileError(
            f"unknown stage count {count} — it is one of {', '.join(map(str, STAGE_COUNTS))}")
    if count == 1:
        return None

    handoff = _number(raw.get("handoff"), "the handoff percentage", 0.0, 100.0,
                      DEFAULT_HANDOFF)
    handoff3 = _number(raw.get("handoff3"), "the third-stage handoff percentage",
                       0.0, 100.0, DEFAULT_HANDOFF3)
    if count == 3 and handoff3 < handoff:
        # The pack says so in its own tooltip, and a schedule that crosses back
        # before it crossed forward is not a thing it can run.
        raise CompileError(
            "the third stage has to start at or after the second — "
            f"{handoff3:g}% is before {handoff:g}%")

    scale = _number(raw.get("stage1_scale"), "the first-stage scale",
                    MIN_STAGE1_SCALE, 1.0, DEFAULT_STAGE1_SCALE)
    return {"count": count, "handoff": handoff, "handoff3": handoff3, "scale": scale}


def compile_prestage(data, image_size_lookup=None, moodboard_lookup=None):
    """`prestage_data` dict -> `ImagePayload`.

    `image_size_lookup(filename) -> (width, height)` supplies the init image's
    dimensions so an img2img render keeps its source's aspect — the same
    adaptive behaviour a keyframe has on the video canvas. Injected for the same
    reason it is in `compile.py`: this module never touches disk.
    """
    if not isinstance(data, dict):
        raise CompileError("prestage_data must be a JSON object")

    arch = data.get("arch", DEFAULT_ARCH)
    if arch not in ARCHES:
        raise CompileError(f"unknown model architecture {arch!r}")

    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise CompileError("describe the image first — the prompt is empty")

    loader = data.get("loader", DEFAULT_LOADER)
    if loader not in LOADERS:
        raise CompileError(f"unknown loader {loader!r}")
    if loader == "svdquant" and arch != "krea2":
        # Refused rather than ignored: the W4A4 loader reads Krea 2's block
        # layout by name, and pointing it at another architecture's checkpoint
        # fails deep inside the loader with a message about tensor names.
        raise CompileError(
            "the SVDQuant loader is Krea 2's — switch the model pill to Krea 2, "
            "or the loader pill back to standard"
        )

    active = active_image_loras(data.get("loras"))
    loras = [{"name": e["name"],
              "strength": float(e.get("strength", 1.0)),
              "adapters": _parse_adapters(e.get("adapters"))} for e in active]
    # Trigger words in front of the prompt, same construction and same dedup as
    # the video compile — a word only counts if its LoRA is actually in the run.
    triggers = collect_triggers(active)
    if triggers:
        prompt = f"{', '.join(triggers)}, {prompt}"

    # The moodboard's guidance goes *after* the prompt: the subject is what the
    # user typed, and the board describes how it should look. Resolved through
    # an injected lookup for the same reason the init image's size is — this
    # module reads no disk, and the catalog is 9 MB of it.
    negative_prompt = None
    board = _parse_moodboard(data.get("moodboard"))
    if board is not None:
        if moodboard_lookup is None:
            raise CompileError(
                "the moodboard catalog is not available — reinstall the package "
                "with vendor/moodboards/data intact, or switch the moodboard pill off")
        try:
            look = moodboard_lookup(board["board"], board["strength"], board["collection"])
        except LookupError as exc:
            raise CompileError(f"moodboard {board['board']!r}: {exc}") from exc
        if look.get("positive"):
            prompt = f"{prompt}\n\n{look['positive']}"
        if board["use_negative"] and look.get("negative"):
            negative_prompt = look["negative"]

    refs = _parse_refs(data.get("refs"))
    if refs and arch == "ideogram4":
        # Refused rather than dropped: Ideogram 4's model reads no reference
        # conditioning at all, and a render that silently ignored the attached
        # images is the failure this package exists to avoid.
        raise CompileError(
            "Ideogram 4.0 has no local reference conditioning — switch the "
            "model pill to Krea 2, or clear the style references"
        )

    init = _parse_init(data.get("init"))

    edit = _parse_edit(data.get("edit"))
    if edit is not None:
        if arch != "krea2":
            raise CompileError(
                "krea2edit is Krea 2's in-context edit path — switch the model "
                "pill to Krea 2, or the edit pill off")
        if refs:
            # Both build the positive conditioning, and from different encoders:
            # the Qwen-edit node for references, the grounded node for edits.
            # Whichever ran second would silently be the only one that counted.
            raise CompileError(
                "an edit and style references cannot run together — both build the "
                "positive conditioning. Clear the style references, or switch the "
                "edit pill off")

    style = _parse_style(data.get("style"))
    if style is not None:
        if arch != "krea2":
            raise CompileError(
                "style transfer is Krea 2's — switch the model pill to Krea 2, "
                "or the style pill off")
        if refs:
            # Two different reference mechanisms, both claiming to be how this
            # render carries a look: the Qwen-edit encoder builds conditioning
            # from its images, RF-inversion patches the model from its own. Run
            # together, neither is doing what its own docs describe.
            raise CompileError(
                "style transfer and style references cannot run together — they are "
                "two different reference paths. Clear the style references, or switch "
                "the style-transfer pill off")

    stages = _parse_stages(data.get("stages"))
    if stages is not None:
        if arch != "krea2":
            raise CompileError(
                "the multi-stage sampler is Krea 2's — it runs the base checkpoint "
                "into the Turbo one. Switch the model pill to Krea 2, or the stages "
                "pill back to one")
        if loader == "svdquant":
            # Two MODEL inputs, and the quantized loader names one file. Mixing a
            # 4-bit stage into an unquantized one across the handoff is not
            # something the pack describes, so it is refused rather than guessed.
            raise CompileError(
                "a multi-stage run needs both the base and the Turbo checkpoint, and "
                "the SVDQuant loader loads one quantized file. Set the loader pill "
                "back to standard, or the stages pill back to one")
        if init is not None:
            # The node has no `denoise` input: every stage starts from the latent
            # it is given, at full strength. So there is no way for it to honour
            # an init image's strength, and pretending otherwise would silently
            # throw the init away.
            raise CompileError(
                "a multi-stage run always starts from noise — it has no denoise "
                "control — so it cannot restyle an init image. Remove the init "
                "image, or set the stages pill back to one")

    short_edge = data.get("short_edge", DEFAULT_SHORT_EDGE)
    ratio_clamped = False
    # An edit's canvas follows its source for the same reason an img2img render's
    # follows its init: the answer to "what shape should this be" is already on
    # screen. The init wins if both are set, because it is the latent the sampler
    # actually starts from.
    adaptive = init["filename"] if init is not None else (edit["source"] if edit else None)
    if adaptive is not None and image_size_lookup is not None:
        source_w, source_h = image_size_lookup(adaptive)
        ratio, ratio_clamped = clamp_ratio(source_w / source_h)
    else:
        aspect = data.get("aspect", DEFAULT_ASPECT)
        try:
            ratio = ASPECT_PRESETS.get(aspect) or float(aspect)
        except (TypeError, ValueError):
            raise CompileError(f"unknown aspect {aspect!r}")
    width, height = resolve_canvas(ratio, short_edge)

    # A two-reference edit has a lower ceiling than a single-reference one: the
    # release notes give ~1 MP for one image and ~1.5 MP for two people.
    edit_oversize = False
    if edit is not None:
        ceiling = (EDIT_TWO_REF_MAX_PIXELS if edit["source_b"]
                   else EDIT_SWEET_SPOT_PIXELS)
        edit_oversize = width * height > ceiling

    # Stage 1's canvas, once the final one is known. Same /16 grid, because it is
    # a latent the DiT samples like any other.
    if stages is not None:
        if stages["scale"] < 1.0:
            stages["width"] = _snap(width * stages["scale"])
            stages["height"] = _snap(height * stages["scale"])
        else:
            # 0/0 is the node's own "do not resize", and passing the target twice
            # would make it upscale from and to the same size.
            stages["width"] = stages["height"] = 0

    checkpoint_field = "model"
    mu = std = None
    if arch == "krea2":
        turbo = data.get("turbo") or {}
        if stages is not None:
            # Stage 2 *is* the Turbo file, so the turbo pill stops choosing a
            # checkpoint here and keeps choosing that stage's step budget — which
            # travels on the block, because the emitter has no other way to know
            # which quality was picked.
            checkpoint_field = "model"
            quality = turbo.get("quality", DEFAULT_TURBO_QUALITY)
            if quality not in TURBO_STEPS:
                raise CompileError(f"unknown turbo quality {quality!r}")
            stages["quality"] = quality
        elif loader == "svdquant":
            # The quantized file is its own, so the turbo pill stops choosing a
            # checkpoint here — but it keeps choosing the schedule, because a
            # checkpoint quantized from Turbo still wants Turbo's steps and cfg
            # and the loader has no way to say which one it came from.
            checkpoint_field = "svdq_model"
        elif turbo.get("on"):
            checkpoint_field = "turbo_model"
    else:
        quality = data.get("quality", DEFAULT_IDEOGRAM_QUALITY)
        if quality not in IDEOGRAM_QUALITIES:
            raise CompileError(f"unknown Ideogram quality preset {quality!r}")
        preset = IDEOGRAM_QUALITIES[quality]
        mu, std = preset["mu"], preset["std"]

    return ImagePayload(
        arch=arch, prompt=prompt, width=width, height=height,
        checkpoint_field=checkpoint_field, loader=loader,
        loras=loras, refs=refs, init=init, negative_prompt=negative_prompt,
        edit=edit, edit_oversize=edit_oversize, style=style, stages=stages,
        mu=mu, std=std, ratio_clamped=ratio_clamped,
    )
