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


def compile_prestage(data, image_size_lookup=None):
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

    short_edge = data.get("short_edge", DEFAULT_SHORT_EDGE)
    ratio_clamped = False
    if init is not None and image_size_lookup is not None:
        source_w, source_h = image_size_lookup(init["filename"])
        ratio, ratio_clamped = clamp_ratio(source_w / source_h)
    else:
        aspect = data.get("aspect", DEFAULT_ASPECT)
        try:
            ratio = ASPECT_PRESETS.get(aspect) or float(aspect)
        except (TypeError, ValueError):
            raise CompileError(f"unknown aspect {aspect!r}")
    width, height = resolve_canvas(ratio, short_edge)

    checkpoint_field = "model"
    mu = std = None
    if arch == "krea2":
        turbo = data.get("turbo") or {}
        if loader == "svdquant":
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
        loras=loras, refs=refs, init=init,
        mu=mu, std=std, ratio_clamped=ratio_clamped,
    )
