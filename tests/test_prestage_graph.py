"""What `MiniMaxH3PreStage.execute` wires up, for both architectures.

Same harness as `test_creator_graph.py`: nothing is sampled, the expansion is
inspected as a dict. The load-bearing cases are the two sampler shapes — Krea 2
through `KSampler`, Ideogram 4 through its own scheduler and the dual-model
guider — and that the graphs are taken from the official templates' wiring
rather than drifting toward each other.

    COMFYUI_PATH=~/ComfyUI <comfy-venv>/bin/python3 tests/test_prestage_graph.py

Skips itself with a message if ComfyUI cannot be imported.
"""

import asyncio
import importlib
import json
import os
import sys

# The checkout this file lives in *is* the package under test, so the import
# name is read off the directory rather than guessed — `__init__.py` imports
# relatively, which means it has to come in as a package under whatever name
# the clone was given.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.basename(ROOT)

# A stock install is one tree and `--base-directory` defaults to it. Point
# COMFYUI_PATH at the ComfyUI that actually runs, and set COMFYUI_BASE as well
# if the base directory is somewhere else (a Desktop install: it usually is).
COMFY = os.environ.get("COMFYUI_PATH", os.path.expanduser("~/ComfyUI"))
BASE = os.environ.get("COMFYUI_BASE", COMFY)


def _boot():
    sys.path.insert(0, COMFY)
    sys.argv = ["main.py", "--base-directory", BASE]
    import nodes
    import server

    loop = asyncio.new_event_loop()
    server.PromptServer(loop)
    asyncio.set_event_loop(loop)
    loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=False))

    sys.path.insert(0, os.path.dirname(ROOT))
    return importlib.import_module(PACKAGE), nodes


try:
    package, comfy_nodes = _boot()
except Exception as exc:  # noqa: BLE001
    print(f"skipped: ComfyUI not importable ({type(exc).__name__}: {exc})")
    sys.exit(0)

ps = importlib.import_module(f"{PACKAGE}.prestage")
ci = importlib.import_module(f"{PACKAGE}.compile_image")
cs = importlib.import_module(f"{PACKAGE}.compile_still")
outputs = importlib.import_module(f"{PACKAGE}.outputs")

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def expect_error(label, fn, fragment):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        if fragment not in str(exc):
            FAILURES.append(f"{label}: error {str(exc)!r} does not mention {fragment!r}")
    else:
        FAILURES.append(f"{label}: expected an error mentioning {fragment!r}, got none")


MODELS = {
    "krea2": {
        "model": "krea2_raw_bf16.safetensors",
        "turbo_model": "krea2_turbo_fp8_scaled.safetensors",
        "clip": "qwen3vl_4b_fp8_scaled.safetensors",
        "vae": "qwen_image_vae.safetensors",
    },
    "ideogram4": {
        "model": "ideogram4_fp8_scaled.safetensors",
        "uncond_model": "ideogram4_unconditional_fp8_scaled.safetensors",
        "clip": "qwen3vl_8b_fp8_scaled.safetensors",
        "vae": "flux2-vae.safetensors",
    },
}

NODE_ID = "9"


def blob(**overrides):
    data = {
        "version": 1,
        "arch": "krea2",
        "prompt": "a red room",
        "aspect": "16:9",
        "short_edge": 1024,
        "init": None,
        "refs": [],
        "loras": [],
        "turbo": {"on": False, "quality": "good", "saved": None},
        "quality": "default",
        "models": dict(MODELS),
    }
    data.update(overrides)
    return json.dumps(data)


def with_id(node_class, unique_id, run):
    from comfy_api.latest import io as comfy_io

    previous = node_class.hidden
    node_class.hidden = comfy_io.HiddenHolder(
        unique_id=unique_id, prompt=None, extra_pnginfo=None, dynprompt=None,
        auth_token_comfy_org=None, api_key_comfy_org=None)
    try:
        return run()
    finally:
        node_class.hidden = previous


def build(data=None, **overrides):
    kwargs = dict(prestage_data=data if data is not None else blob(),
                  seed=100, steps=52, cfg=3.5, sampler_name="euler", scheduler="simple")
    kwargs.update(overrides)
    return with_id(ps.MiniMaxH3PreStage, NODE_ID,
                   lambda: ps.MiniMaxH3PreStage.execute(**kwargs))


def by_class(graph):
    out = {}
    for node_id, node in graph.items():
        out.setdefault(node["class_type"], []).append((node_id, node["inputs"]))
    return out


# ---- Krea 2 t2i --------------------------------------------------------------
#
# The template shape: loaders, one text encode, a zeroed negative, an empty
# 16-channel latent, one KSampler, one decode, one save. Nothing else.

out = build()
graph = out.expand
kinds = by_class(graph)

check("expansion exports no links", out.result, ())
check("one sampler", len(kinds["KSampler"]), 1)
check("one text encode", len(kinds["CLIPTextEncode"]), 1)
check("the negative is the prompt zeroed out", len(kinds["ConditioningZeroOut"]), 1)
check("an empty 16-channel latent", len(kinds["EmptySD3LatentImage"]), 1)
check("one decode", len(kinds["VAEDecode"]), 1)
for absent in ("LoadImage", "VAEEncode", "LoraLoaderModelOnly", "SamplerCustomAdvanced",
               "Ideogram4Scheduler", "TextEncodeQwenImageEditPlus", "ModelSamplingFlux"):
    check(f"no {absent} in a bare t2i render", absent in kinds, False)

check("the RAW checkpoint is loaded",
      [i["unet_name"] for _, i in kinds["UNETLoader"]], [MODELS["krea2"]["model"]])
check("the text encoder is loaded as Krea 2's",
      (kinds["CLIPLoader"][0][1]["clip_name"], kinds["CLIPLoader"][0][1]["type"]),
      (MODELS["krea2"]["clip"], "krea2"))
check("the VAE is the Qwen image VAE",
      [i["vae_name"] for _, i in kinds["VAELoader"]], [MODELS["krea2"]["vae"]])

sampler = kinds["KSampler"][0][1]
check("the sampler settings arrive verbatim",
      (sampler["seed"], sampler["steps"], sampler["cfg"], sampler["sampler_name"],
       sampler["scheduler"], sampler["denoise"]),
      (100, 52, 3.5, "euler", "simple", 1.0))

# 16:9 at a 1024 short edge on the /16 grid.
latent = kinds["EmptySD3LatentImage"][0][1]
check("the canvas follows the aspect pill on the /16 grid",
      (latent["width"] % 16, latent["height"], latent["width"] > latent["height"]),
      (0, 1024, True))

check("one save node", len(kinds["MiniMaxH3SaveImage"]), 1)
save_id, save_inputs = kinds["MiniMaxH3SaveImage"][0]
check("it is reported against the node that built it",
      graph[save_id].get("override_display_id"), NODE_ID)
check("it saves the decoded picture",
      graph[save_inputs["images"][0]]["class_type"], "VAEDecode")
check("it lands in the pre-stage folder, which is its own",
      save_inputs["filename_prefix"], outputs.IMAGE_PREFIX)


def save_prefix(**overrides):
    """Where a blob's still would land."""
    return by_class(build(blob(**overrides)).expand)["MiniMaxH3SaveImage"][0][1]["filename_prefix"]


# The blob decides where the file goes, and a blob that says nothing gets the
# default above. This is the whole output-structure control: before it, the
# prefix was a module constant and every install on earth wrote its stills to
# the same folder with no way to say otherwise.
check("a blob's own prefix is used instead",
      save_prefix(output_prefix="my-project/stills/take"), "my-project/stills/take")
check("a trailing slash means a folder, and keeps the default's stem",
      save_prefix(output_prefix="my-project/"), "my-project/prestage")
check("an empty prefix falls back to the default rather than the output root",
      save_prefix(output_prefix="   "), outputs.IMAGE_PREFIX)
# Refused while the graph is being built, *not* by get_save_image_path after the
# still has been sampled — which is the whole reason `outputs` exists rather
# than the save node just taking whatever it is handed.
expect_error("a prefix that climbs out of the output folder",
             lambda: save_prefix(output_prefix="../../H3"),
             "'.' and '..' are not allowed")
expect_error("an absolute prefix, pointed at the flag that does work",
             lambda: save_prefix(output_prefix="/mnt/big/stills"),
             "--output-directory")

# ---- turbo -------------------------------------------------------------------
#
# The pill swaps the checkpoint file; the sampler row it wrote arrives through
# the ordinary widgets. Nothing else about the graph may move.

turbo = by_class(build(blob(turbo={"on": True, "quality": "good", "saved": None}),
                       steps=8, cfg=1.0).expand)
check("turbo loads the Turbo checkpoint",
      [i["unet_name"] for _, i in turbo["UNETLoader"]], [MODELS["krea2"]["turbo_model"]])
check("turbo changes nothing structural",
      sorted(turbo), sorted(kinds))
check("the pill's sampler row arrives verbatim",
      (turbo["KSampler"][0][1]["steps"], turbo["KSampler"][0][1]["cfg"]), (8, 1.0))

# ---- LoRAs and triggers ------------------------------------------------------

with_lora = by_class(build(blob(loras=[
    {"name": "krea2_darkbrush.safetensors", "strength": 0.8,
     "triggers": ["monochrome ink wash style"]},
    {"name": "off.safetensors", "strength": 1.0, "enabled": False},
])).expand)
loras = with_lora["LoraLoaderModelOnly"]
check("one LoRA patch — the disabled one is skipped", len(loras), 1)
check("model-only, at the entry's strength",
      (loras[0][1]["lora_name"], loras[0][1]["strength_model"]),
      ("krea2_darkbrush.safetensors", 0.8))
check("the sampler reads the patch",
      with_lora["KSampler"][0][1]["model"][0], loras[0][0])
check("the trigger word rides in front of the prompt",
      with_lora["CLIPTextEncode"][0][1]["text"],
      "monochrome ink wash style, a red room")

# ---- img2img -----------------------------------------------------------------
#
# An init image replaces the empty latent with an encode of the scaled source,
# and the KSampler's denoise becomes the entry's strength. (The adaptive-aspect
# half of img2img reads the file's size and is compile-time tested elsewhere —
# here the file does not exist, so the blob is compiled with the aspect pill.)

init_payload = ci.compile_prestage(
    {"arch": "krea2", "prompt": "p", "init": {"filename": "seed.png", "denoise": 0.6},
     "aspect": "1:1", "short_edge": 1024})
check("the payload carries the init", init_payload.init,
      {"filename": "seed.png", "denoise": 0.6})

render_mod = importlib.import_module(f"{PACKAGE}.render")
ri = importlib.import_module(f"{PACKAGE}.render_image")

i2i_graph = ri.emit(init_payload, ri.ImageWeights(arch="krea2", files=MODELS["krea2"]),
                    render_mod.Sampling(seed=1, steps=52, cfg=3.5,
                                        sampler_name="euler", scheduler="simple"), NODE_ID)
i2i = by_class(i2i_graph.finalize())
check("the init is loaded and encoded",
      ("LoadImage" in i2i, "VAEEncode" in i2i, "EmptySD3LatentImage" in i2i),
      (True, True, False))
check("scaled to the resolved canvas first",
      i2i["ImageScale"][0][1]["upscale_method"], "lanczos")
check("the sampler starts from the encode at the entry's strength",
      i2i["KSampler"][0][1]["denoise"], 0.6)

# ---- style references (Krea 2) -----------------------------------------------
#
# The official reference workflow's wiring: the Qwen-edit encoder with the
# references in its image slots, the method node on its conditioning, and the
# shift moved onto ModelSamplingFlux — none of which appears without refs.

refs_payload = ci.compile_prestage(
    {"arch": "krea2", "prompt": "p", "refs": [{"filename": "a.png"}, {"filename": "b.png"}],
     "aspect": "1:1", "short_edge": 1024})
refs = by_class(ri.emit(refs_payload, ri.ImageWeights(arch="krea2", files=MODELS["krea2"]),
                        render_mod.Sampling(), NODE_ID).finalize())
encode = refs["TextEncodeQwenImageEditPlus"][0][1]
check("both references sit in the encoder's image slots",
      ("image1" in encode, "image2" in encode, "image3" in encode),
      (True, True, False))
check("the encoder also gets the VAE for the reference latents",
      encode["vae"][0] in {nid for nid, _ in refs["VAELoader"]}, True)
check("the reference method is the official one",
      refs["FluxKontextMultiReferenceLatentMethod"][0][1]["reference_latents_method"],
      "index_timestep_zero")
check("the shift moves onto ModelSamplingFlux on this branch",
      (refs["ModelSamplingFlux"][0][1]["max_shift"], refs["ModelSamplingFlux"][0][1]["base_shift"]),
      (1.15, 0.5))
check("no plain text encode on the reference branch", "CLIPTextEncode" in refs, False)

expect_error("a fourth reference is refused",
             lambda: ci.compile_prestage(
                 {"arch": "krea2", "prompt": "p",
                  "refs": ["a.png", "b.png", "c.png", "d.png"]}),
             "three image slots")

# ---- Ideogram 4 --------------------------------------------------------------
#
# The other sampler shape: its own scheduler's sigmas, SamplerCustomAdvanced,
# and the dual-model guider with the late-cfg drop on the conditional branch.

ideo = by_class(build(blob(arch="ideogram4"), steps=20, cfg=7.0).expand)
check("Ideogram samples through the custom path",
      ("SamplerCustomAdvanced" in ideo, "KSampler" in ideo), (True, False))
check("on its own schedule",
      (ideo["Ideogram4Scheduler"][0][1]["steps"], ideo["Ideogram4Scheduler"][0][1]["mu"],
       ideo["Ideogram4Scheduler"][0][1]["std"]),
      (20, 0.0, 1.75))
check("the latent is Flux2's", "EmptyFlux2LatentImage" in ideo, True)
check("both checkpoints load — the unconditional branch is a separate model",
      sorted(i["unet_name"] for _, i in ideo["UNETLoader"]),
      sorted([MODELS["ideogram4"]["model"], MODELS["ideogram4"]["uncond_model"]]))
guider = ideo["DualModelGuider"][0][1]
check("the guider runs at the widget's cfg", guider["cfg"], 7.0)
override = ideo["CFGOverride"][0]
check("the conditional branch carries the late-cfg drop",
      (override[1]["cfg"], override[1]["start_percent"], override[1]["end_percent"]),
      (3.0, 0.7, 1.0))
check("...and the guider reads it as its conditional model",
      guider["model"][0], override[0])
check("the text encoder is loaded as Ideogram's",
      ideo["CLIPLoader"][0][1]["type"], "ideogram4")

# The quality preset owns mu/std, not the user.
quality = by_class(build(blob(arch="ideogram4", quality="quality"), steps=48).expand)
check("the quality preset reshapes the schedule",
      (quality["Ideogram4Scheduler"][0][1]["mu"], quality["Ideogram4Scheduler"][0][1]["std"]),
      (0.0, 1.5))

# Without the unconditional file the guider degrades to ordinary CFG — the
# node's own documented behaviour — rather than refusing.
one_model = dict(MODELS["ideogram4"])
del one_model["uncond_model"]
single = by_class(build(blob(arch="ideogram4",
                             models={**MODELS, "ideogram4": one_model})).expand)
check("one checkpoint is ordinary CFG, not an error",
      ("model_negative" in single["DualModelGuider"][0][1],
       len(single["UNETLoader"])),
      (False, 1))

# Ideogram i2i truncates the schedule instead of using a denoise widget.
ideo_i2i_payload = ci.compile_prestage(
    {"arch": "ideogram4", "prompt": "p", "init": {"filename": "seed.png", "denoise": 0.5},
     "aspect": "1:1", "short_edge": 1024})
ideo_i2i = by_class(ri.emit(ideo_i2i_payload,
                            ri.ImageWeights(arch="ideogram4", files=MODELS["ideogram4"]),
                            render_mod.Sampling(steps=20, cfg=7.0, sampler_name="euler"),
                            NODE_ID).finalize())
check("i2i keeps the tail of the sigmas",
      ideo_i2i["SplitSigmasDenoise"][0][1]["denoise"], 0.5)
check("the sampler reads the truncated schedule",
      ideo_i2i["SamplerCustomAdvanced"][0][1]["sigmas"][0],
      ideo_i2i["SplitSigmasDenoise"][0][0])

# ---- refusals ----------------------------------------------------------------

expect_error("Ideogram with references is refused with directions",
             lambda: build(blob(arch="ideogram4", refs=["a.png"])),
             "switch the model pill to Krea 2")
expect_error("an empty prompt is refused",
             lambda: build(blob(prompt="  ")),
             "prompt is empty")


def without(arch, field):
    trimmed = {k: dict(v) for k, v in MODELS.items()}
    del trimmed[arch][field]
    return blob(arch=arch, models=trimmed)


expect_error("a missing checkpoint is refused up front, naming the folder",
             lambda: build(without("krea2", "model")),
             "models/diffusion_models")
expect_error("a missing text encoder is refused too",
             lambda: build(without("krea2", "clip")),
             "models/text_encoders")
expect_error("turbo demands the Turbo file, not the RAW one",
             lambda: build(blob(turbo={"on": True, "quality": "good", "saved": None},
                                models={**MODELS, "krea2": {k: v for k, v in MODELS["krea2"].items()
                                                            if k != "turbo_model"}})),
             "Turbo checkpoint")
# The file turbo skips is not demanded — the mirror of the video nodes'
# unrouted-checkpoint rule.
check("turbo does not require the RAW file",
      "UNETLoader" in by_class(build(
          blob(turbo={"on": True, "quality": "good", "saved": None},
               models={**MODELS, "krea2": {k: v for k, v in MODELS["krea2"].items()
                                           if k != "model"}}),
          steps=8, cfg=1.0).expand), True)

# ---- MiniMax H3: a still from the video model ---------------------------------
#
# The branch that is not an image model. What matters here is that it is the
# *video* path: the video segment node, the video checkpoints, the video canvas
# — with one latent frame taken out of the sampled clip and decoded.

H3_MODELS = {
    "fl2va": "minimax_h3_fl2va_fp8.safetensors",
    "ref2va": "minimax_h3_ref2va_fp8.safetensors",
    "clip": "minimax_qwen3vl_32b.safetensors",
    "vae": "minimax_h3_t1_image_vae_step1597.safetensors",
    "audio_vae": "minimax_h3_audio_vae.safetensors",
}


def still_blob(request=None, **overrides):
    """A pre-stage blob on the H3 branch.

    The generation lives in `minimax.request` in exactly the Creator's shape —
    the branch is driven by the Creator's own editor — so the weights, the
    assets and the canvas are all in there, under the video nodes' own keys.
    """
    inner = {"prompt": "a red room", "assets": [], "loras": [],
             "aspect": "16:9", "short_edge": 768, "models": dict(H3_MODELS)}
    inner.update(request or {})
    block = {"frames": 5, "latent_index": 0, "request": inner}
    block.update(overrides.pop("minimax", {}))
    data = {"version": 1, "arch": "minimax", "minimax": block}
    data.update(overrides)
    return json.dumps(data)


def still(data=None, **overrides):
    kwargs = dict(seed=7, steps=20, cfg=1.0, sampler_name="res_multistep", scheduler="simple")
    kwargs.update(overrides)
    return build(data if data is not None else still_blob(), **kwargs)


h3_graph = still().expand
h3 = by_class(h3_graph)

check("it samples once", len(h3["KSampler"]), 1)
check("through the video segment node", len(h3["MiniMaxH3TimelineSegment"]), 1)
check("and keeps one latent frame", len(h3["MiniMaxH3StillLatent"]), 1)
check("decoded once, saved once",
      (len(h3["VAEDecode"]), len(h3["MiniMaxH3SaveImage"])), (1, 1))
check("the first latent frame by default", h3["MiniMaxH3StillLatent"][0][1]["index"], 0)
check("the still is what gets decoded",
      h3_graph[h3["VAEDecode"][0][1]["samples"][0]]["class_type"], "MiniMaxH3StillLatent")
check("no audio is decoded", "VAEDecodeAudio" in h3, False)
check("and no video is written", "MiniMaxH3Save" in h3, False)

# Only the routed checkpoint is loaded, exactly as on a video render — and the
# audio VAE is not loaded at all, because nothing here cites sound.
check("a bare prompt loads FL2VA and nothing else",
      [i["unet_name"] for _, i in h3["UNETLoader"]], [H3_MODELS["fl2va"]])
check("one VAE, the single-image one",
      [i["vae_name"] for _, i in h3["VAELoader"]], [H3_MODELS["vae"]])
check("the text encoder is loaded as H3's",
      (h3["CLIPLoader"][0][1]["clip_name"], h3["CLIPLoader"][0][1]["type"]),
      (H3_MODELS["clip"], "minimax"))
check("the segment node gets no audio VAE",
      "audio_vae" in h3["MiniMaxH3TimelineSegment"][0][1], False)

sampler = h3["KSampler"][0][1]
check("the sampler settings arrive verbatim",
      (sampler["seed"], sampler["steps"], sampler["cfg"], sampler["sampler_name"],
       sampler["denoise"]),
      (7, 20, 1.0, "res_multistep", 1.0))

# The shortest legal clip, and the canvas a video render would use.
payload = json.loads(h3["MiniMaxH3TimelineSegment"][0][1]["segment_data"])
check("it samples the shortest legal clip",
      cs.latent_frames(round(payload["request"]["duration_s"] * 24)), 2)
check("on H3's own canvas",
      (payload["request"]["short_edge"], payload["request"]["aspect"]), (768, "16:9"))
check("it lands in the pre-stage folder",
      h3["MiniMaxH3SaveImage"][0][1]["filename_prefix"], outputs.IMAGE_PREFIX)

# The standing route, the video nodes' own control: Ref2VA takes the text-only
# payload FL2VA was trained for, so a t2i still can be made by the reference
# weights — and then FL2VA is neither loaded nor required.
routed = by_class(still(still_blob(request={
    "models": {**{k: v for k, v in H3_MODELS.items() if k != "fl2va"}, "route": "ref2va"}})).expand)
check("a forced route loads that checkpoint and no other",
      [i["unet_name"] for _, i in routed["UNETLoader"]], [H3_MODELS["ref2va"]])
check("and it reaches the segment node as the request's own pin",
      json.loads(routed["MiniMaxH3TimelineSegment"][0][1]["segment_data"])["request"]["checkpoint"],
      "ref2va")
expect_error("forcing FL2VA on a still with references is refused",
             lambda: still(still_blob(request={
                 "assets": [{"handle": "img-1", "kind": "image", "role": "reference",
                             "filename": "face.png"}],
                 "models": {**H3_MODELS, "route": "fl2va"}})),
             "cannot be run through FL2VA")

# References route to Ref2VA and are the video node's own — including a clip
# taken with its soundtrack, which is the one thing that loads the audio VAE.
refs = by_class(still(still_blob(request={"assets": [
    {"handle": "img-1", "kind": "image", "role": "reference", "filename": "face.png"},
    {"handle": "vid-1", "kind": "video", "role": "reference", "filename": "clip.mp4",
     "track": "picture+sound"},
]})).expand)
check("references route to Ref2VA",
      [i["unet_name"] for _, i in refs["UNETLoader"]], [H3_MODELS["ref2va"]])
check("a cited soundtrack loads the audio VAE",
      sorted(i["vae_name"] for _, i in refs["VAELoader"]),
      sorted([H3_MODELS["vae"], H3_MODELS["audio_vae"]]))
check("and hands it to the segment node",
      "audio_vae" in refs["MiniMaxH3TimelineSegment"][0][1], True)

silent = by_class(still(still_blob(request={"assets": [
    {"handle": "vid-1", "kind": "video", "role": "reference", "filename": "clip.mp4"},
]})).expand)
check("a clip cited for its picture alone loads no audio VAE",
      [i["vae_name"] for _, i in silent["VAELoader"]], [H3_MODELS["vae"]])

# Keyframes are the video node's too: a start frame and an end frame, with the
# canvas adapting to the start frame exactly as a shot's does. The size lookup
# is stubbed because these two files are not on this machine — it is the only
# thing on this path that touches the disk.
media = importlib.import_module(f"{PACKAGE}.media")
real_image_size = media.image_size
media.image_size = lambda filename: (1920, 1080)
try:
    frames = by_class(still(still_blob(request={"assets": [
        {"handle": "img-1", "kind": "image", "role": "first_frame", "filename": "open.png"},
        {"handle": "img-2", "kind": "image", "role": "last_frame", "filename": "close.png"},
    ]})).expand)
finally:
    media.image_size = real_image_size
frames_payload = json.loads(frames["MiniMaxH3TimelineSegment"][0][1]["segment_data"])
check("both keyframes reach the request",
      sorted((a["role"], a["filename"]) for a in frames_payload["request"]["assets"]),
      [("first_frame", "open.png"), ("last_frame", "close.png")])

# The preview is the video node's, patched on the model the segment hands out.
preview = by_class(still(still_blob(
    request={"models": {**H3_MODELS, "preview": "taeh3.safetensors"}})).expand)
check("taeh3 previews the still",
      "ModelPreviewOverrideKJ" in preview or comfy_nodes.NODE_CLASS_MAPPINGS.get(
          "ModelPreviewOverrideKJ") is None, True)

# ---- refusals ----------------------------------------------------------------

expect_error("a latent frame the clip does not have is refused",
             lambda: still(still_blob(minimax={"latent_index": 4})),
             "2 latent frames")
expect_error("a still with no VAE is refused, naming the folder",
             lambda: still(still_blob(request={
                 "models": {k: v for k, v in H3_MODELS.items() if k != "vae"}})),
             "models/vae")
expect_error("keyframes and references together are refused",
             lambda: still(still_blob(request={"assets": [
                 {"handle": "img-1", "kind": "image", "role": "first_frame", "filename": "open.png"},
                 {"handle": "img-2", "kind": "image", "role": "reference", "filename": "face.png"},
             ]})),
             "cannot be combined")

# ---- the SVDQuant loader ------------------------------------------------------
#
# The two loaders are a pair, not a checkpoint swap: a plain LoraLoaderModelOnly
# on a 4-bit model makes ComfyUI rewrite the quantized weight, which puts the
# LoRA's delta through 4 bits with it. So what these check is that choosing one
# loader moves the LoRAs too, and that choosing neither leaves the graph exactly
# as it was.

SVDQ_MODELS = {**MODELS, "krea2": {**MODELS["krea2"],
                                   "svdq_model": "krea2_turbo_svdq_w4a4_rank256.safetensors"}}


def svdq_blob(**overrides):
    data = json.loads(blob())
    data["loader"] = "svdquant"
    data["models"] = dict(SVDQ_MODELS)
    data.update(overrides)
    return json.dumps(data)


nodes_by_class = by_class(build(svdq_blob()).expand)
check("the SVDQuant loader replaces UNETLoader",
      "UNETLoader" in nodes_by_class, False)
check("and loads the quantized checkpoint by name",
      nodes_by_class[ri.SVDQUANT_LOADER][0][1]["model_name"],
      SVDQ_MODELS["krea2"]["svdq_model"])
check("with no dtype, which a quantized file has already decided",
      "weight_dtype" in nodes_by_class[ri.SVDQUANT_LOADER][0][1], False)

lora_graph = by_class(build(svdq_blob(loras=[
    {"name": "grain.safetensors", "strength": 0.8},
    {"name": "face.safetensors", "strength": 1.0, "adapters": "bake"},
])).expand)
check("core's LoRA loader never appears on the quantized path",
      "LoraLoaderModelOnly" in lora_graph, False)
check("both LoRAs go through the SVDQuant loader",
      len(lora_graph[ri.SVDQUANT_LORA]), 2)
modes = sorted(inputs["adapters"] for _, inputs in lora_graph[ri.SVDQUANT_LORA])
check("each entry keeps its own adapter mode, matched to the pack's wording",
      [str(m).split(" ")[0] for m in modes], ["bake", "bypass"])

# The pill is off by default, and off has to mean *unchanged* — this is the
# whole no-regression claim, checked rather than asserted in a comment.
check("the default blob still loads through core's UNETLoader",
      ri.SVDQUANT_LOADER in by_class(build().expand), False)
check("and still patches LoRAs with core's loader",
      "LoraLoaderModelOnly" in by_class(build(blob(loras=[
          {"name": "grain.safetensors", "strength": 0.8}])).expand), True)

expect_error("an unpicked quantized checkpoint is named, with its folder",
             lambda: build(svdq_blob(models={**SVDQ_MODELS, "krea2": {
                 k: v for k, v in SVDQ_MODELS["krea2"].items() if k != "svdq_model"}})),
             "models/diffusion_models")
expect_error("the SVDQuant loader is refused on Ideogram",
             lambda: build(svdq_blob(arch="ideogram4")),
             "Krea 2")

# ---- the moodboard's negative --------------------------------------------------
#
# A moodboard never reaches the graph except here. Its positive half is merged
# into the prompt in `compile_image` and arrives as text, but a board that says
# what its look is *not* has to become real conditioning, and this pipeline's
# negative has always been `ConditioningZeroOut`. Emitted directly rather than
# through a blob so the check does not depend on which boards ship in the 9 MB
# catalog.


def emitted(payload, weights=None, cfg=3.5):
    graph = ri.emit(payload, weights or ri.ImageWeights(arch="krea2", files=MODELS["krea2"]),
                    render_mod.Sampling(seed=1, steps=52, cfg=cfg,
                                        sampler_name="euler", scheduler="simple"), NODE_ID)
    return by_class(graph.finalize())


plain = ci.compile_prestage({"arch": "krea2", "prompt": "p", "aspect": "1:1", "short_edge": 1024})
zeroed = emitted(plain)
check("with no board the negative is still the zeroed positive",
      ("ConditioningZeroOut" in zeroed, len(zeroed["CLIPTextEncode"])), (True, 1))

with_negative = ci.compile_prestage(
    {"arch": "krea2", "prompt": "p", "aspect": "1:1", "short_edge": 1024,
     "moodboard": {"on": True, "board": "b"}},
    moodboard_lookup=lambda board, strength, collection: {
        "positive": "hard chiaroscuro", "negative": "flat lighting", "title": "t"})
negged = emitted(with_negative)
check("a board's negative guidance is encoded instead of zeroed",
      ("ConditioningZeroOut" in negged, len(negged["CLIPTextEncode"])), (False, 2))
check("and it is the board's own words",
      sorted(i["text"] for _, i in negged["CLIPTextEncode"])[0], "flat lighting")
check("the positive still carries the prompt and the board's prose",
      [t.startswith("p") and "hard chiaroscuro" in t
       for t in sorted((i["text"] for _, i in negged["CLIPTextEncode"]), reverse=True)][0],
      True)

# ---- krea2edit ------------------------------------------------------------------
#
# Two halves of one recipe, and the check that matters is that both are wired:
# the patch node carries the source's appearance, the grounded encoder carries
# its semantics, and running either alone runs half the training.

edit_payload = ci.compile_prestage(
    {"arch": "krea2", "prompt": "put him in a red coat", "aspect": "1:1", "short_edge": 1024,
     "edit": {"on": True, "source": {"filename": "man.png"}, "ref_boost": 1.3}})
edited = emitted(edit_payload)

check("the appearance half is patched onto the model", len(edited[ri.EDIT_PATCH]), 1)
patch = edited[ri.EDIT_PATCH][0][1]
check("wired the way the pack recommends: pixels, VAE and the sampler's own latent",
      all(key in patch for key in ("source_image", "vae", "target_latent", "source_latent")),
      True)
check("the boost is passed through", patch["ref_boost"], 1.3)
check("and the target latent is the one KSampler starts from",
      patch["target_latent"], edited["KSampler"][0][1]["latent_image"])

# The node takes its pixel path whenever vae + source_image are both wired, and
# that path encodes the source itself. This emitter always wires both, so a
# VAEEncode for `source_latent` would run every render and be discarded.
check("no VAE encode is emitted for a socket the pixel path overrides",
      "VAEEncode" in edited, False)
check("the required socket is fed a latent the graph already has",
      patch["source_latent"], patch["target_latent"])

check("the semantic half replaces the text-only encode", "CLIPTextEncode" in edited, False)
check("and at cfg 3.5 it is built twice — the negative is grounded too, as training's was",
      len(edited[ri.EDIT_ENCODE]), 2)
prompts = sorted(i["prompt"] for _, i in edited[ri.EDIT_ENCODE])
check("one carries the instruction, one is empty",
      (prompts[0], prompts[1]), ("", "put him in a red coat"))
check("both are grounded on the same source",
      len({i["image"][0] for _, i in edited[ri.EDIT_ENCODE]}), 1)

# The one that cost real time. A graph is executed by dependency, so a grounded
# encode wired into `negative` runs whether or not the sampler reads it — and at
# cfg 1, which is where Turbo lives, it never does. Building it there is a second
# full pass of Qwen3-VL's vision tower over the source for a result nothing looks
# at, and on the first edit rendered here it was most of the render.
cheap = emitted(edit_payload, cfg=1.0)
check("at cfg 1 the grounded negative is not built at all",
      len(cheap[ri.EDIT_ENCODE]), 1)
check("and the socket is satisfied by the zeroed positive instead",
      "ConditioningZeroOut" in cheap, True)
check("the instruction is still grounded",
      cheap[ri.EDIT_ENCODE][0][1]["prompt"], "put him in a red coat")
check("the source is scaled to the render's canvas before either sees it",
      edited["ImageScale"][0][1]["width"], edit_payload.width)

two_ref = emitted(ci.compile_prestage(
    {"arch": "krea2", "prompt": "p", "aspect": "1:1", "short_edge": 1024,
     "edit": {"on": True, "source": {"filename": "scene.png"},
              "source_b": {"filename": "face.png"}}}))
check("a second reference reaches both halves",
      ("source_image_b" in two_ref[ri.EDIT_PATCH][0][1],
       all("image_b" in i for _, i in two_ref[ri.EDIT_ENCODE])),
      (True, True))

with_lora = emitted(ci.compile_prestage(
    {"arch": "krea2", "prompt": "p", "aspect": "1:1", "short_edge": 1024,
     "edit": {"on": True, "source": {"filename": "man.png"},
              "lora": "krea2_identity_edit_v1_2.safetensors"}}))
check("the edit LoRA is patched on, and before the patch node",
      list(with_lora[ri.EDIT_PATCH][0][1]["model"]),
      [with_lora["LoraLoaderModelOnly"][0][0], 0])

# Off is off: nothing above reaches a render that did not ask for an edit.
for absent in (ri.EDIT_PATCH, ri.EDIT_ENCODE):
    check(f"no {absent} without the edit pill", absent in by_class(build().expand), False)

# ---- style transfer -------------------------------------------------------------
#
# A model patch, not a latent path: the pack's own workflow feeds the sampler the
# same empty latent it feeds the reference builder, and takes the model out of the
# transfer node. `ref_conditioning` is the render's own positive, not a second
# prompt — both settled by reading that workflow rather than the signatures.

styled_one = emitted(ci.compile_prestage(
    {"arch": "krea2", "prompt": "a red room", "aspect": "1:1", "short_edge": 1024,
     "style": {"on": True, "refs": ["look.png"], "fit": "contain"}}))

check("one reference builds one reference latent", len(styled_one[ri.STYLE_REFERENCE]), 1)
check("fitted to the sampler's own latent",
      styled_one[ri.STYLE_REFERENCE][0][1]["target_latent"],
      styled_one["KSampler"][0][1]["latent_image"])
check("with the chosen fit", styled_one[ri.STYLE_REFERENCE][0][1]["fit"], "contain")
check("the single-reference transfer patches the model",
      len(styled_one[ri.STYLE_TRANSFER]), 1)
check("and the sampler runs the patched model",
      list(styled_one["KSampler"][0][1]["model"]),
      [styled_one[ri.STYLE_TRANSFER][0][0], 0])
check("ref_conditioning is the render's own positive, not a second prompt",
      styled_one[ri.STYLE_TRANSFER][0][1]["ref_conditioning"],
      styled_one["KSampler"][0][1]["positive"])
# At strength 1 the pack's recommended table runs as shipped; the advanced dials
# are ignored in that mode, so sending them would be noise.
check("strength 1 leaves the node in recommended mode",
      styled_one[ri.STYLE_TRANSFER][0][1]["mode"], "recommended")
check("no two-reference machinery for one reference",
      ri.STYLE_TWO_REFERENCES in styled_one, False)

styled_two = emitted(ci.compile_prestage(
    {"arch": "krea2", "prompt": "a red room", "aspect": "1:1", "short_edge": 1024,
     "style": {"on": True, "refs": ["a.png", "b.png"], "primary": 2, "strength": 1.25}}))
check("two references build two latents and a bundle",
      (len(styled_two[ri.STYLE_REFERENCE]), len(styled_two[ri.STYLE_TWO_REFERENCES])), (2, 1))
check("through the two-reference transfer node",
      (ri.STYLE_TWO_TRANSFER in styled_two, ri.STYLE_TRANSFER in styled_two), (True, False))
check("which one leads, as the pack's string",
      styled_two[ri.STYLE_TWO_TRANSFER][0][1]["primary_reference"], "2")
# Moving the strength is what switches the node to custom, because recommended
# mode ignores it — its own tooltip says so.
check("a moved strength switches to custom mode and is passed through",
      (styled_two[ri.STYLE_TWO_TRANSFER][0][1]["mode"],
       styled_two[ri.STYLE_TWO_TRANSFER][0][1]["style_strength"]),
      ("custom", 1.25))
# The rest of the fourteen dials come from the installed class rather than from a
# copy here, so a retune upstream is followed instead of frozen.
check("and the other dials arrive from the pack's own defaults",
      styled_two[ri.STYLE_TWO_TRANSFER][0][1]["rf_mode"], "flowturbo_pc")

for absent in (ri.STYLE_REFERENCE, ri.STYLE_TRANSFER, ri.STYLE_TWO_TRANSFER):
    check(f"no {absent} without the style pill", absent in by_class(build().expand), False)

# ---- the multi-stage sampler ----------------------------------------------------
#
# It replaces `KSampler` outright and takes two models. The mapping onto this
# package's single sampler row is the load-bearing decision: stage 1 takes the
# widgets, stage 2 takes the Turbo preset — which is what the turbo pill has
# always meant.

STAGE_MODELS = {**MODELS, "krea2": {**MODELS["krea2"]}}


def staged(count=2, **overrides):
    payload = ci.compile_prestage(
        {"arch": "krea2", "prompt": "p", "aspect": "1:1", "short_edge": 1024,
         "turbo": {"on": False, "quality": "good"},
         "stages": {"count": count, **overrides}})
    return by_class(ri.emit(payload, ri.ImageWeights(arch="krea2", files=STAGE_MODELS["krea2"]),
                            render_mod.Sampling(seed=7, steps=52, cfg=4.0,
                                                sampler_name="euler", scheduler="simple"),
                            NODE_ID).finalize())


two = staged(2)
check("the two-stage node replaces KSampler",
      ("KSampler" in two, len(two[ri.TWO_STAGE])), (False, 1))
node = two[ri.TWO_STAGE][0][1]
check("stage 1 takes the widget row verbatim",
      (node["stage1_steps"], node["stage1_cfg"], node["stage1_sampler_name"],
       node["stage1_scheduler"]),
      (52, 4.0, "euler", "simple"))
check("stage 2 takes the Turbo preset, at the pill's quality",
      (node["stage2_steps"], node["stage2_cfg"], node["stage2_sampler_name"]),
      (ci.TURBO_STEPS["good"], ci.KREA_TURBO["cfg"], ci.KREA_TURBO["sampler_name"]))
check("the handoff arrives from the payload", node["handoff_percent"], ci.DEFAULT_HANDOFF)
check("stage 1 runs the base checkpoint",
      [i["unet_name"] for _, i in two["UNETLoader"]][0], MODELS["krea2"]["model"])
check("and a second loader is built for the Turbo file",
      sorted(i["unet_name"] for _, i in two["UNETLoader"]),
      sorted([MODELS["krea2"]["model"], MODELS["krea2"]["turbo_model"]]))
check("no resize at full first-stage scale",
      (node["final_width"], node["final_height"]), (0, 0))

three = staged(3)
check("three stages use the three-stage node and carry the second crossover",
      (ri.THREE_STAGE in three, three[ri.THREE_STAGE][0][1]["stage3_handoff_percent"]),
      (True, ci.DEFAULT_HANDOFF3))

scaled = staged(2, stage1_scale=0.5)
check("a scaled first stage samples on a smaller latent",
      scaled["EmptySD3LatentImage"][0][1]["width"], 512)
check("and the node is told where to finish",
      (scaled[ri.TWO_STAGE][0][1]["final_width"],
       scaled[ri.TWO_STAGE][0][1]["final_height"]), (1024, 1024))

# Off is off, and off is the argument-for-argument same KSampler as before.
plain_graph = by_class(build().expand)
check("one stage still emits a single KSampler and nothing else",
      (len(plain_graph["KSampler"]), ri.TWO_STAGE in plain_graph,
       ri.THREE_STAGE in plain_graph), (1, False, False))

# ---- DyPE and SEGA --------------------------------------------------------------
#
# One patch over the embedder, emitted last because everything above it assumes
# the position encoding it rewrites. One, not two: they are alternatives, and
# this block used to assert they chained.

positioned = emitted(ci.compile_prestage(
    {"arch": "krea2", "prompt": "p", "aspect": "1:1", "short_edge": 2048,
     "dype": {"on": True, "scale": 3.0}}))

check("DyPE alone is emitted", (len(positioned[ri.DYPE]), ri.SEGA in positioned), (1, False))
check("and the sampler runs the patched model",
      list(positioned["KSampler"][0][1]["model"]), [positioned[ri.DYPE][0][0], 0])
# Both nodes' tooltips say width/height must match the empty latent, so they are
# the resolved canvas rather than anything the user can set separately.
check("DyPE is told the canvas it is extrapolating for",
      (positioned[ri.DYPE][0][1]["width"], positioned[ri.DYPE][0][1]["height"]),
      (positioned["EmptySD3LatentImage"][0][1]["width"],
       positioned["EmptySD3LatentImage"][0][1]["height"]))
check("the scale arrives from the payload",
      positioned[ri.DYPE][0][1]["dype_scale"], 3.0)
# `auto` rather than `flux`: Krea 2 is not in the pack's list, and auto reaches
# the flux branch by elimination — which is right, because SingleStreamDiT's
# pe_embedder *is* Flux's EmbedND.
check("the architecture is left to the pack to detect",
      positioned[ri.DYPE][0][1]["model_type"], "auto")
check("and everything not exposed comes from the pack's own defaults",
      positioned[ri.DYPE][0][1]["method"], ci.DEFAULT_DYPE_METHOD)

only_sega = emitted(ci.compile_prestage(
    {"arch": "krea2", "prompt": "p", "aspect": "1:1", "short_edge": 1024,
     "sega": {"on": True, "alpha": 0.2}}))
check("SEGA alone patches the loaded model directly",
      (ri.DYPE in only_sega, len(only_sega[ri.SEGA])), (False, 1))
check("and its amplitude arrives from the payload",
      only_sega[ri.SEGA][0][1]["mscale_alpha"], 0.2)
check("the sampler runs SEGA's model when SEGA is the choice",
      list(only_sega["KSampler"][0][1]["model"]), [only_sega[ri.SEGA][0][0], 0])

for absent in (ri.DYPE, ri.SEGA):
    check(f"no {absent} without its pill", absent in by_class(build().expand), False)

expect_error("a multi-stage run with no Turbo checkpoint picked says which file",
             lambda: ri.emit(
                 ci.compile_prestage({"arch": "krea2", "prompt": "p", "aspect": "1:1",
                                      "short_edge": 1024, "stages": {"count": 2}}),
                 ri.ImageWeights(arch="krea2", files={
                     k: v for k, v in MODELS["krea2"].items() if k != "turbo_model"}),
                 render_mod.Sampling(seed=1, steps=52, cfg=4.0,
                                     sampler_name="euler", scheduler="simple"), NODE_ID),
             "Turbo checkpoint")
# ---- GGUF checkpoints -------------------------------------------------------
#
# The image branch reuses `models.loader_for`, so the claims are the creator
# graph's: a `.gguf` file swaps the loader class, drops `weight_dtype` (a core
# widget the GGUF nodes lack), and refuses up front without the pack.

GGUF_MODEL = "krea2_raw_Q4_K_M.gguf"

expect_error("a GGUF checkpoint without the pack is refused up front",
             lambda: build(blob(models={**MODELS, "krea2": {
                 **MODELS["krea2"], "model": GGUF_MODEL}})),
             "ComfyUI-GGUF")


class _FakeGGUF:
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


_restore_gguf = dict(comfy_nodes.NODE_CLASS_MAPPINGS)
comfy_nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"] = _FakeGGUF
try:
    quant = by_class(build(blob(models={**MODELS, "krea2": {
        **MODELS["krea2"], "model": GGUF_MODEL}})).expand)
    check("a .gguf checkpoint loads through the pack's loader",
          quant["UnetLoaderGGUF"][0][1], {"unet_name": GGUF_MODEL})
    check("...and no core UNETLoader beside it", "UNETLoader" in quant, False)
    check("the text encoder stays on the core loader", len(quant["CLIPLoader"]), 1)
finally:
    comfy_nodes.NODE_CLASS_MAPPINGS.clear()
    comfy_nodes.NODE_CLASS_MAPPINGS.update(_restore_gguf)

if FAILURES:
    print(f"{len(FAILURES)} failure(s):")
    for failure in FAILURES:
        print("  -", failure)
    sys.exit(1)
print("all prestage graph tests passed")
