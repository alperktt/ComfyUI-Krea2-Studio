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

# The vendored nodes, under the ids `nodes_vendor` registers them with.
SVDQUANT_LOADER = "K2S_SVDQuantW4A4Loader"
SVDQUANT_LORA = "K2S_SVDQuantLoraLoader"
EDIT_PATCH = "K2S_Krea2EditModelPatch"
EDIT_ENCODE = "K2S_Krea2EditGroundedEncode"
STYLE_REFERENCE = "K2S_Krea2StyleReference"
STYLE_TRANSFER = "K2S_Krea2StyleTransfer"
STYLE_TWO_REFERENCES = "K2S_Krea2TwoStyleReferences"
STYLE_TWO_TRANSFER = "K2S_Krea2TwoStyleTransfer"
TWO_STAGE = "K2S_KreaTwoStageSampler"
THREE_STAGE = "K2S_KreaThreeStageSampler"

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

    The DiT field is whichever one the payload resolved (`model`, `turbo_model`
    or `svdq_model`); the unconditional checkpoint is never required, because the
    guider degrades to ordinary CFG without it.

    A multi-stage run needs a second checkpoint on top of that — its stage 2 is
    the Turbo file — so it is asked for here rather than left to fail inside the
    loader with an empty filename.
    """
    needed = ["clip", "vae", payload.checkpoint_field]
    if payload.stages:
        needed.append("turbo_model")
    for name in needed:
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
        _emit_krea2(graph, payload, sampling, weights, clip, vae, model, unique_id,
                    filename_prefix)
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
    else:
        _require_vendored(
            SVDQUANT_LOADER, "The SVDQuant loader",
            "Set the loader pill back to standard to render on the stock loader.")
        # No `weight_dtype`: the checkpoint carries its own precision, and the
        # dtype pill has nothing to say about a file that is already quantized.
        model = graph.node(SVDQUANT_LOADER,
                           model_name=weights.get(payload.checkpoint_field)).out(0)

    for entry in payload.loras:
        model = emit_lora(graph, payload, model, entry["name"], entry["strength"],
                          entry.get("adapters", "bypass"))
    return model


def emit_lora(graph, payload, model, name, strength, adapters="bypass"):
    """One LoRA onto a MODEL link, through whichever loader this render is using.

    Model-only, exactly as the official workflows patch these DiTs — there is no
    text-encoder half to a Krea or Ideogram LoRA.

    Public because the edit panel carries a LoRA of its own and has to patch it
    on the same way: two code paths for "add a LoRA" is how one of them ends up
    on the wrong loader.
    """
    if payload.loader != "svdquant":
        return graph.node("LoraLoaderModelOnly", model=model,
                          lora_name=name, strength_model=strength).out(0)

    import nodes

    node = nodes.NODE_CLASS_MAPPINGS.get(SVDQUANT_LORA)
    if node is None:
        raise ValueError(
            f"The SVDQuant LoRA loader did not load, so '{name}' would be baked "
            f"into the 4-bit weight instead of folded into the low-rank branch. "
            f"Remove the LoRA, or set the loader pill back to standard."
        )
    return graph.node(SVDQUANT_LORA, model=model, lora_name=name, strength=strength,
                      adapters=_adapter_option(node, adapters)).out(0)


def _require_vendored(node_id, feature, fallback):
    """Refuse a feature whose vendored node did not load, by name.

    Emitters address these by string into a `GraphBuilder`, which does not check
    ids — an unregistered one fails much later, inside execution, with an error
    naming neither the pack nor the feature.
    """
    from . import nodes_vendor

    if nodes_vendor.missing(node_id):
        raise ValueError(
            f"{feature} needs the '{node_id}' node, which did not load — see the "
            f"ComfyUI log for why vendor/ could not be imported. {fallback}")


def _edit_sources(graph, payload, placeholder):
    """The edit's source image(s), loaded and scaled to the render's canvas.

    Scaled here rather than left to the node because `compile_image` has already
    made the canvas follow the source's aspect, so this only absorbs the /16
    snap — and a source that arrives at the target grid is the case `fit_mode`
    was trained on rather than the one it has to rescue.

    **`source_latent` gets `placeholder` rather than an encode of the source.**
    It is a required socket that this emitter's wiring guarantees is never read:
    the node takes its pixel path whenever `vae` and `source_image` are both
    connected (`if vae is not None and source_image is not None` in the vendored
    patch), and that path encodes the source itself from pixels. This emitter
    always connects both — it is the blur-proof path the pack recommends — so a
    `VAEEncode` here would run on every render and have its result thrown away.
    Passing a latent the graph already has costs nothing and adds no node.

    Flagged in `vendor/krea2edit/ORIGIN.md`: if a future version stops
    overriding, this becomes wrong rather than merely wasteful.
    """
    images, latents = {}, {}
    for slot, key in (("", "source"), ("_b", "source_b")):
        filename = payload.edit[key]
        if not filename:
            continue
        loaded = graph.node("LoadImage", image=filename).out(0)
        images[f"source_image{slot}"] = graph.node(
            "ImageScale", image=loaded, upscale_method="lanczos",
            width=payload.width, height=payload.height, crop="center").out(0)
        latents[f"source_latent{slot}"] = placeholder
    return images, latents


def _emit_edit_patch(graph, payload, model, vae, images, latents, target_latent):
    """krea2edit's appearance half: the source's tokens, in context.

    Wired the way the pack's README recommends and not one input less — `vae`
    plus `source_image` is the pixel-space path that does not blur on a
    resolution mismatch, and `target_latent` gets the source encoded before
    sampling starts rather than on the first step, which on a tight card is the
    difference between the VAE evicting part of the DiT and not.
    """
    _require_vendored(EDIT_PATCH, "The edit", "Switch the edit pill off to render without it.")
    return graph.node(EDIT_PATCH, model=model, vae=vae,
                      target_latent=target_latent,
                      ref_boost=payload.edit["ref_boost"],
                      ref_boost_a=payload.edit["ref_boost_a"],
                      fit_mode=payload.edit["fit_mode"],
                      **images, **latents).out(0)


def _emit_edit_encode(graph, payload, clip, images, text):
    """krea2edit's semantic half: the instruction, grounded on the source.

    Training encoded the instruction *with* the image through Qwen3-VL, twelve
    layers tapped. Stock `CLIPTextEncode` is text-only, so an edit run through it
    is missing the half that resolves "the man on the left".

    `text=""` is the grounded unconditional, which is what training's negative
    looked like — the pack's own docstring asks for it, so the negative branch
    calls this too rather than zeroing the positive.
    """
    _require_vendored(EDIT_ENCODE, "The edit", "Switch the edit pill off to render without it.")
    grounded = {"image": images["source_image"]}
    if "source_image_b" in images:
        grounded["image_b"] = images["source_image_b"]
    return graph.node(EDIT_ENCODE, clip=clip, prompt=text,
                      grounding_px=payload.edit["grounding_px"], **grounded).out(0)


def _emit_style(graph, payload, model, vae, latent, positive):
    """RF-inversion style transfer, patched onto the model. Returns the new link.

    Wired from the pack's own shipped workflow, which settles two things that are
    not obvious from the signatures: `ref_conditioning` is the render's *own*
    positive conditioning rather than a second prompt, and `target_latent` is the
    sampler's latent — the sampler still starts from it, so this is a model patch
    and the latent path is untouched.

    **`mode` decides whether the fourteen advanced dials are read at all.** In
    `recommended` the nodes apply their own table and ignore the widgets,
    `style_strength` included — its tooltip says so. So the strength being moved
    is what switches them to `custom`, and everything else then comes from
    `accel.node_defaults`, which reads the installed class's declared defaults
    and therefore follows a retune instead of freezing a copy of it.
    """
    from . import accel, nodes_vendor

    two = len(payload.style["refs"]) > 1
    transfer_node = STYLE_TWO_TRANSFER if two else STYLE_TRANSFER
    for node_id in (STYLE_REFERENCE, transfer_node,
                    *([STYLE_TWO_REFERENCES] if two else ())):
        _require_vendored(node_id, "Style transfer",
                          "Switch the style pill off to render without it.")

    references = [
        graph.node(STYLE_REFERENCE, vae=vae, target_latent=latent,
                   reference_image=graph.node("LoadImage", image=name).out(0),
                   fit=payload.style["fit"], upscale_method="lanczos").out(0)
        for name in payload.style["refs"]
    ]

    import nodes

    kwargs = accel.node_defaults(nodes.NODE_CLASS_MAPPINGS[transfer_node],
                                 skip=("model", "reference_latent", "ref_conditioning",
                                       "style_refs"))
    custom = payload.style["strength"] != 1.0
    kwargs["mode"] = "custom" if custom else "recommended"
    if custom:
        kwargs["style_strength"] = payload.style["strength"]

    if two:
        bundle = graph.node(STYLE_TWO_REFERENCES,
                            reference_latent_1=references[0],
                            reference_latent_2=references[1]).out(0)
        kwargs["primary_reference"] = str(payload.style["primary"])
        return graph.node(transfer_node, model=model, style_refs=bundle,
                          ref_conditioning=positive, **kwargs).out(0)
    return graph.node(transfer_node, model=model, reference_latent=references[0],
                      ref_conditioning=positive, **kwargs).out(0)


def _emit_sampler(graph, payload, sampling, weights, model, positive, negative,
                  latent, denoise):
    """The sampling pass: one `KSampler`, or the multi-stage node in its place.

    Returns the sampled LATENT link. With `stages` unset this is exactly the
    `KSampler` the PreStage has always emitted, down to the argument order — the
    no-regression claim depends on that, and `test_prestage_graph` checks it.

    The multi-stage route runs the undistilled base first, for the variation
    between seeds a distillation largely does not have, then finishes on Turbo.
    Its own row-per-stage maps onto this package's single row as: **stage 1 takes
    the widgets** (it is the pass that decides the image, and the row is the one
    the user can see), **stage 2 takes the Turbo preset** — which is what the
    turbo pill has always meant. Three stages is base -> Turbo -> base, and the
    pack reuses stage 1's settings for stage 3 itself.
    """
    if not payload.stages:
        return graph.node(
            "KSampler", model=model, positive=positive, negative=negative,
            latent_image=latent, seed=sampling.seed, steps=sampling.steps,
            cfg=sampling.cfg, sampler_name=sampling.sampler_name,
            scheduler=sampling.scheduler, denoise=denoise,
        ).out(0)

    from .compile_image import KREA_TURBO, TURBO_STEPS

    node_id = THREE_STAGE if payload.stages["count"] == 3 else TWO_STAGE
    _require_vendored(node_id, "The multi-stage sampler",
                      "Set the stages pill back to one to sample normally.")

    # Stage 2 is the Turbo checkpoint, always — `compile_image` says so by
    # leaving `checkpoint_field` on the base file. `check` has already refused a
    # render where either is unpicked.
    stage2 = graph.node("UNETLoader", unet_name=weights.get("turbo_model"),
                        weight_dtype=weights.dtype).out(0)

    quality = payload.stages.get("quality", "good")
    extra = {}
    if payload.stages["count"] == 3:
        extra["stage3_handoff_percent"] = payload.stages["handoff3"]

    return graph.node(
        node_id,
        stage1_model=model, stage2_model=stage2,
        positive=positive, negative=negative, latent_image=latent,
        seed=sampling.seed,
        handoff_percent=payload.stages["handoff"],
        stage1_steps=sampling.steps, stage1_cfg=sampling.cfg,
        stage1_sampler_name=sampling.sampler_name, stage1_scheduler=sampling.scheduler,
        stage2_steps=TURBO_STEPS[quality], stage2_cfg=KREA_TURBO["cfg"],
        stage2_sampler_name=KREA_TURBO["sampler_name"],
        stage2_scheduler=KREA_TURBO["scheduler"],
        # Where stage 2 *finishes*, which is the render's own canvas. Stage 1's
        # smaller canvas is on the empty latent (`_latent`), not here. 0/0 is the
        # node's "do not resize" and is what a full-size first stage compiles to.
        final_width=payload.width if payload.stages["width"] else 0,
        final_height=payload.height if payload.stages["height"] else 0,
        upscale_method="bislerp",
        **extra,
    ).out(0)


def _negative(graph, payload, clip, positive):
    """The unconditional branch: zeroed conditioning, or a real negative prompt.

    Zeroed is what this has always emitted and is still the default, because
    RAW at cfg 3.5 wants an unconditional rather than a second prompt and Turbo
    at cfg 1.0 skips the branch outright.

    A moodboard can supply the exception. A board that says what its look is
    *not* — no bokeh, no lens flare — is stating conditioning, and dropping it
    on the floor because this pipeline happens to have no negative box would be
    ignoring half of what the board says. So when one is carried, it is encoded
    and used. At cfg 1 the sampler still skips it, which is correct rather than
    broken: a negative that nothing reads costs nothing.
    """
    if payload.negative_prompt:
        return graph.node("CLIPTextEncode", clip=clip, text=payload.negative_prompt).out(0)
    return graph.node("ConditioningZeroOut", conditioning=positive).out(0)


def _latent(graph, payload, vae, empty_node):
    """The starting latent: empty for t2i, the encoded init image for img2img.

    The init is scaled to the resolved canvas rather than the canvas following
    the init exactly — `compile_image` already derived the aspect from the
    image, so this only absorbs the /16 snap. Returns (latent, denoise).
    """
    if payload.init is None:
        # A multi-stage run may start smaller than it finishes: stage 1 samples
        # on this latent and the node upscales at the handoff. `compile_image`
        # leaves the two sizes at 0 when they are the same, so this reads the
        # final canvas unless it was told otherwise.
        width, height = payload.width, payload.height
        if payload.stages and payload.stages["width"]:
            width, height = payload.stages["width"], payload.stages["height"]
        empty = graph.node(empty_node, width=width, height=height, batch_size=1)
        return empty.out(0), 1.0
    image = graph.node("LoadImage", image=payload.init["filename"]).out(0)
    scaled = graph.node("ImageScale", image=image, upscale_method="lanczos",
                        width=payload.width, height=payload.height,
                        crop="center").out(0)
    encoded = graph.node("VAEEncode", pixels=scaled, vae=vae).out(0)
    return encoded, payload.init["denoise"]


def _emit_krea2(graph, payload, sampling, weights, clip, vae, model, unique_id,
                filename_prefix):
    # Built first on the edit path: the patch node wants the sampler's own
    # latent, so that it can encode the source before sampling starts rather
    # than pulling the VAE onto the card on the first step.
    latent, denoise = _latent(graph, payload, vae, "EmptySD3LatentImage")

    if payload.edit:
        # The Identity Edit LoRA goes on before the patch: the patch adds the
        # in-context path, the LoRA is what was trained to use it. Optional
        # because it may equally well be sitting in the main stack above — the
        # panel says so rather than adding it twice.
        if payload.edit["lora"]:
            model = emit_lora(graph, payload, model, payload.edit["lora"],
                              payload.edit["lora_strength"])
        images, latents = _edit_sources(graph, payload, latent)
        model = _emit_edit_patch(graph, payload, model, vae, images, latents, latent)
        positive = _emit_edit_encode(graph, payload, clip, images, payload.prompt)
        if sampling.cfg > 1.0:
            # Grounded on the same source with an empty instruction — training's
            # unconditional, and what the pack asks for when CFG is in play.
            negative = _emit_edit_encode(graph, payload, clip, images,
                                         payload.negative_prompt or "")
        else:
            # At cfg 1 the sampler never evaluates the unconditional branch — but
            # a graph is executed by dependency, so a grounded encode wired into
            # `negative` still runs, and running it means a second full pass of
            # Qwen3-VL's vision tower over the source for a result nothing reads.
            # On the first edit rendered here that was most of the render.
            # `ConditioningZeroOut` satisfies the socket for free.
            negative = graph.node("ConditioningZeroOut", conditioning=positive).out(0)
        # An edit and a style transfer compose: one is conditioning built from the
        # source, the other a patch on the attention path. The style patch goes
        # last so it wraps the edit's, matching the plain branch's order.
        if payload.style:
            model = _emit_style(graph, payload, model, vae, latent, positive)
        sampled = _emit_sampler(graph, payload, sampling, weights, model,
                                positive, negative, latent, denoise)
        return _emit_tail(graph, sampled, vae, unique_id, filename_prefix)

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

    negative = _negative(graph, payload, clip, positive)

    # Exclusive with the reference branch above — the compile refuses the pair —
    # so this only ever wraps a plain t2i or img2img model.
    if payload.style:
        model = _emit_style(graph, payload, model, vae, latent, positive)

    sampled = _emit_sampler(graph, payload, sampling, weights, model,
                            positive, negative, latent, denoise)
    _emit_tail(graph, sampled, vae, unique_id, filename_prefix)


def _emit_ideogram4(graph, payload, sampling, weights, clip, vae, model, unique_id,
                    filename_prefix):
    positive = graph.node("CLIPTextEncode", clip=clip, text=payload.prompt).out(0)
    negative = _negative(graph, payload, clip, positive)

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
