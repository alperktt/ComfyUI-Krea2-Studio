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
check("it lands in the pre-stage folder",
      save_inputs["filename_prefix"], "minimax/prestage")

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

if FAILURES:
    print(f"{len(FAILURES)} failure(s):")
    for failure in FAILURES:
        print("  -", failure)
    sys.exit(1)
print("all prestage graph tests passed")
