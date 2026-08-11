"""One image generation, as a graph. The PreStage's half of `render.py`.

The PreStage owns its sampler for the reason the Creator does: a node that
samples has to *be* the sampler, and ComfyUI has no way to say that except by
returning a subgraph. Unlike the video render there is no segment worker node —
an image render is loaders, an encode, a sampler and a decode, all of which are
core's, so the graph is emitted from core nodes directly and core's caching
falls out per node (an unchanged prompt re-decodes nothing).

Two architectures, two sampler shapes, both taken verbatim from the official
ComfyUI templates rather than invented:

**Krea 2** samples like an ordinary model — `KSampler`, zeroed-out negative
(cfg 3.5 on RAW, 1.0 on Turbo). Style references go through core's
`TextEncodeQwenImageEditPlus` + `FluxKontextMultiReferenceLatentMethod`
("index_timestep_zero"), which is exactly what `model_base.Krea2.extra_conds`
reads (`reference_latents` / `reference_latents_method`); the reference
template also swaps the shift onto `ModelSamplingFlux(1.15, 0.5)`, so this does
too, only on that branch.

**Ideogram 4** carries its unconditional branch as a *separate checkpoint*, so
the guider is `DualModelGuider(model, model_negative)` at cfg 7 with a
`CFGOverride` dropping the conditional branch to cfg 3 over the last 30% — and
its schedule is `Ideogram4Scheduler`'s resolution-shifted sigmas, which is why
this branch samples through `SamplerCustomAdvanced` rather than `KSampler`.
Without the unconditional file picked, `DualModelGuider` degrades to ordinary
CFG on the one model, which the node itself documents — so that file is
optional here too.

The weights are named in the blob per architecture, so switching the model
pill never forgets the other side's files — the same reason `models.Weights`
keeps both video checkpoints.
"""

from dataclasses import dataclass, field

from . import outputs
from .compile import CompileError
from .compile_image import IDEOGRAM_CFG_LATE

SAVE_NODE = "MiniMaxH3SaveImage"
# Where a still lands when the blob does not say — see `outputs`, which owns
# both defaults and what a typed prefix is allowed to be.
FILENAME_PREFIX = outputs.IMAGE_PREFIX

# Which directory each pickable field browses — ComfyUI's own folder keys, and
# the listing route hands the same map to the frontend.
FOLDERS = {
    "model": "diffusion_models",
    "turbo_model": "diffusion_models",
    "svdq_model": "diffusion_models",
    "uncond_model": "diffusion_models",
    "clip": "text_encoders",
    "vae": "vae",
}

# Which fields each architecture actually has. Ideogram has no distilled
# checkpoint (its speed axis is the preset table); Krea has no second branch.
ARCH_FIELDS = {
    "krea2": ("model", "turbo_model", "svdq_model", "clip", "vae"),
    "ideogram4": ("model", "uncond_model", "clip", "vae"),
}

CLIP_TYPE = {"krea2": "krea2", "ideogram4": "ideogram4"}

LABEL = {
    "model": "the checkpoint",
    "turbo_model": "the Turbo checkpoint",
    "svdq_model": "the SVDQuant checkpoint",
    "uncond_model": "the unconditional checkpoint",
    "clip": "the text encoder",
    "vae": "the VAE",
}

# The vendored SVDQuant pair, under the ids `nodes_vendor` registers them with.
SVDQUANT_LOADER = "K2S_SVDQuantW4A4Loader"
SVDQUANT_LORA = "K2S_SVDQuantLoraLoader"

# The reference template's shift, applied only on the style-reference branch —
# plain t2i leaves the shift the checkpoint detection already set (1.15).
KREA_REF_SHIFT = {"max_shift": 1.15, "base_shift": 0.5}
KREA_REF_METHOD = "index_timestep_zero"


@dataclass(frozen=True)
class ImageWeights:
    """The files the node was pointed at for one architecture.

    Unset is the normal state of a freshly spawned node, so this validates at
    emit time and names the empty field — the same contract `models.Weights`
    holds for the video side.
    """

    arch: str
    files: dict = field(default_factory=dict)
    dtype: str = "default"

    @classmethod
    def from_blob(cls, data):
        """The `models` block of a prestage_data blob, for the blob's arch.

        The block is `{krea2: {...}, ideogram4: {...}, dtype}` — per-arch
        sub-blocks so flipping the model pill never forgets the other side.
        """
        arch = (data or {}).get("arch", "krea2")
        block = (data or {}).get("models")
        if not isinstance(block, dict):
            block = {}
        side = block.get(arch)
        if not isinstance(side, dict):
            side = {}
        files = {}
        for name in ARCH_FIELDS.get(arch, ()):
            value = side.get(name)
            if isinstance(value, str) and value.strip():
                files[name] = value.strip()
        dtype = block.get("dtype")
        return cls(arch=arch,
                   files=files,
                   dtype=dtype if isinstance(dtype, str) and dtype else "default")

    def get(self, name):
        return self.files.get(name)


def check(weights, payload):
    """Refuse now if a file this render needs was never picked.

    The DiT field is whichever one the payload resolved (`model` or
    `turbo_model`); the unconditional checkpoint is never required, because the
    guider degrades to ordinary CFG without it.
    """
    for name in ("clip", "vae", payload.checkpoint_field):
        if weights.get(name):
            continue
        # Not .capitalize(), which would lowercase "Turbo" mid-label.
        label = LABEL[name][0].upper() + LABEL[name][1:]
        raise ValueError(
            f"{label} has not been picked. Open the "
            f"pre-stage node's 'weights' control and choose a file from "
            f"models/{FOLDERS[name]}."
        )


def _require_arch(arch):
    """Refuse an architecture the installed core does not know.

    Both models are native in current ComfyUI; a stale install fails inside the
    loader with a shape mismatch nobody can read, so this says it up front.
    Keyed off what is actually registered rather than a version number.
    """
    import nodes

    if arch == "ideogram4" and "Ideogram4Scheduler" not in nodes.NODE_CLASS_MAPPINGS:
        raise ValueError(
            "This ComfyUI does not know Ideogram 4 yet (no Ideogram4Scheduler "
            "node). Update ComfyUI and restart."
        )
    if arch == "krea2":
        declared = nodes.NODE_CLASS_MAPPINGS["CLIPLoader"].INPUT_TYPES()
        types = declared.get("required", {}).get("type", [[]])[0]
        if "krea2" not in types:
            raise ValueError(
                "This ComfyUI does not know Krea 2 yet (CLIPLoader has no "
                "'krea2' type). Update ComfyUI and restart."
            )


def emit(payload, weights, sampling, unique_id, filename_prefix=FILENAME_PREFIX):
    """-> the graph, which the caller finalizes with `render.expanded`.

    `sampling` is a `render.Sampling` — the same widget names as the video
    nodes, meaning the same thing. On the Ideogram branch `scheduler` is
    unused (the model owns its schedule) and `cfg` feeds the dual-model guider.
    """
    from comfy_execution.graph_utils import GraphBuilder

    if payload.arch != weights.arch:
        raise CompileError("the payload and the weights disagree about the architecture")
    _require_arch(payload.arch)
    check(weights, payload)

    graph = GraphBuilder()

    clip = graph.node("CLIPLoader", clip_name=weights.get("clip"),
                      type=CLIP_TYPE[payload.arch]).out(0)
    vae = graph.node("VAELoader", vae_name=weights.get("vae")).out(0)
    model = _emit_model(graph, payload, weights)

    if payload.arch == "krea2":
        _emit_krea2(graph, payload, sampling, clip, vae, model, unique_id, filename_prefix)
    else:
        _emit_ideogram4(graph, payload, sampling, weights, clip, vae, model, unique_id,
                        filename_prefix)
    return graph


def _adapter_option(node, wanted):
    """Our `bypass`/`bake` as the vendored loader's own option string.

    Its options are prose with the measured trade-off in them — "bypass (exact,
    slower)" — so matching by prefix rather than by literal means a reworded
    label costs nothing here. The same trick `accel._block_cache_kwargs` plays
    on the block-cache modes, and for the same reason: this module must not
    carry a stale copy of somebody else's wording.
    """
    declared = node.INPUT_TYPES().get("optional", {}).get("adapters")
    options = declared[0] if isinstance(declared, (tuple, list)) and declared else []
    match = next((o for o in options if str(o).lower().startswith(wanted)), None)
    if match is None:
        raise ValueError(
            f"The SVDQuant LoRA loader has no {wanted!r} adapter mode — it offers "
            f"{list(options)}. The vendored pack has renamed its modes; "
            f"re-vendor it or set the LoRA back to the other mode."
        )
    return match


def _emit_model(graph, payload, weights):
    """The DiT, loaded and patched with the LoRAs. Returns the MODEL link.

    Two loaders, and the choice reaches the LoRAs as well as the checkpoint. A
    plain `LoraLoaderModelOnly` on a 4-bit model would make ComfyUI rewrite the
    quantized weight — dequantize, add, requantize — putting the LoRA's delta
    through 4 bits along with everything else. The vendored loader folds it into
    the low-rank branch instead, which is the whole point of the format, so the
    two nodes move together and are never mixed.
    """
    if payload.loader != "svdquant":
        model = graph.node("UNETLoader", unet_name=weights.get(payload.checkpoint_field),
                           weight_dtype=weights.dtype).out(0)
        # Model-only, exactly as the official workflows patch these DiTs — there
        # is no text-encoder half to a Krea or Ideogram LoRA.
        for entry in payload.loras:
            model = graph.node("LoraLoaderModelOnly", model=model,
                               lora_name=entry["name"],
                               strength_model=entry["strength"]).out(0)
        return model

    from . import nodes_vendor

    if nodes_vendor.missing(SVDQUANT_LOADER):
        raise ValueError(
            "The SVDQuant loader did not load — see the ComfyUI log for why "
            "vendor/svdquant could not be imported. Set the loader pill back to "
            "standard to render on the stock loader."
        )
    import nodes

    # No `weight_dtype`: the checkpoint carries its own precision, and the dtype
    # pill has nothing to say about a file that is already quantized.
    model = graph.node(SVDQUANT_LOADER,
                       model_name=weights.get(payload.checkpoint_field)).out(0)
    lora_node = nodes.NODE_CLASS_MAPPINGS.get(SVDQUANT_LORA)
    for entry in payload.loras:
        if lora_node is None:
            raise ValueError(
                "The SVDQuant LoRA loader did not load, so this LoRA would be "
                "baked into the 4-bit weight instead of folded into the low-rank "
                "branch. Remove the LoRA, or set the loader pill back to standard."
            )
        model = graph.node(SVDQUANT_LORA, model=model,
                           lora_name=entry["name"],
                           strength=entry["strength"],
                           adapters=_adapter_option(lora_node, entry["adapters"])).out(0)
    return model


def _latent(graph, payload, vae, empty_node):
    """The starting latent: empty for t2i, the encoded init image for img2img.

    The init is scaled to the resolved canvas rather than the canvas following
    the init exactly — `compile_image` already derived the aspect from the
    image, so this only absorbs the /16 snap. Returns (latent, denoise).
    """
    if payload.init is None:
        empty = graph.node(empty_node, width=payload.width, height=payload.height,
                           batch_size=1)
        return empty.out(0), 1.0
    image = graph.node("LoadImage", image=payload.init["filename"]).out(0)
    scaled = graph.node("ImageScale", image=image, upscale_method="lanczos",
                        width=payload.width, height=payload.height,
                        crop="center").out(0)
    encoded = graph.node("VAEEncode", pixels=scaled, vae=vae).out(0)
    return encoded, payload.init["denoise"]


def _emit_krea2(graph, payload, sampling, clip, vae, model, unique_id, filename_prefix):
    if payload.refs:
        # The Qwen-edit encoder reads up to three references: it feeds them to
        # the text encoder as vision tokens *and* VAE-encodes them into the
        # conditioning's reference latents, which is the pair Krea 2 was
        # post-trained against. The method node picks the variant the official
        # workflow uses.
        images = {f"image{i + 1}": graph.node("LoadImage", image=name).out(0)
                  for i, name in enumerate(payload.refs)}
        positive = graph.node("TextEncodeQwenImageEditPlus", clip=clip,
                              prompt=payload.prompt, vae=vae, **images).out(0)
        positive = graph.node("FluxKontextMultiReferenceLatentMethod",
                              conditioning=positive,
                              reference_latents_method=KREA_REF_METHOD).out(0)
        model = graph.node("ModelSamplingFlux", model=model,
                           width=payload.width, height=payload.height,
                           **KREA_REF_SHIFT).out(0)
    else:
        positive = graph.node("CLIPTextEncode", clip=clip, text=payload.prompt).out(0)

    # Zeroed-out conditioning as the negative on both checkpoints: at Turbo's
    # cfg 1.0 it is skipped outright, and RAW's cfg 3.5 wants an unconditional,
    # not a second prompt.
    negative = graph.node("ConditioningZeroOut", conditioning=positive).out(0)

    latent, denoise = _latent(graph, payload, vae, "EmptySD3LatentImage")
    sampled = graph.node(
        "KSampler", model=model, positive=positive, negative=negative,
        latent_image=latent, seed=sampling.seed, steps=sampling.steps,
        cfg=sampling.cfg, sampler_name=sampling.sampler_name,
        scheduler=sampling.scheduler, denoise=denoise,
    )
    _emit_tail(graph, sampled.out(0), vae, unique_id, filename_prefix)


def _emit_ideogram4(graph, payload, sampling, weights, clip, vae, model, unique_id,
                    filename_prefix):
    positive = graph.node("CLIPTextEncode", clip=clip, text=payload.prompt).out(0)
    negative = graph.node("ConditioningZeroOut", conditioning=positive).out(0)

    # The late-cfg drop wraps the conditional model, after the LoRAs — it is a
    # sampling wrapper, not a weight patch.
    model = graph.node("CFGOverride", model=model, **IDEOGRAM_CFG_LATE).out(0)

    guider_inputs = {"model": model, "positive": positive, "negative": negative,
                     "cfg": sampling.cfg}
    if weights.get("uncond_model"):
        guider_inputs["model_negative"] = graph.node(
            "UNETLoader", unet_name=weights.get("uncond_model"),
            weight_dtype=weights.dtype).out(0)
    guider = graph.node("DualModelGuider", **guider_inputs).out(0)

    sigmas = graph.node("Ideogram4Scheduler", steps=sampling.steps,
                        width=payload.width, height=payload.height,
                        mu=payload.mu, std=payload.std).out(0)
    latent, denoise = _latent(graph, payload, vae, "EmptyFlux2LatentImage")
    if denoise < 1.0:
        # img2img on a custom schedule: keep the tail of the sigmas and let the
        # noise node start the latent at the truncated schedule's first sigma —
        # the same statement KSampler's denoise makes, said in sigmas.
        sigmas = graph.node("SplitSigmasDenoise", sigmas=sigmas,
                            denoise=denoise).out(1)

    sampled = graph.node(
        "SamplerCustomAdvanced",
        noise=graph.node("RandomNoise", noise_seed=sampling.seed).out(0),
        guider=guider,
        sampler=graph.node("KSamplerSelect", sampler_name=sampling.sampler_name).out(0),
        sigmas=sigmas, latent_image=latent,
    )
    _emit_tail(graph, sampled.out(0), vae, unique_id, filename_prefix)


def _emit_tail(graph, samples, vae, unique_id, filename_prefix):
    """Decode and save, reported against the node the user is looking at.

    The display-id stamp is the same mechanism `render.emit_tail` uses and
    exists for the same reason: the save node lives in an expanded graph on
    nobody's canvas, and the stamp files its `executed` message under the
    PreStage node so the stage card can show what it just made.
    """
    image = graph.node("VAEDecode", samples=samples, vae=vae).out(0)
    save = graph.node(SAVE_NODE, images=image, filename_prefix=filename_prefix)
    save.set_override_display_id(unique_id)
    return save
