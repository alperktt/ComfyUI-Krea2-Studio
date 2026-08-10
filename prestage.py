"""The MiniMax H3 PreStage node: stills for the pipeline, made on the left.

The Creator consumes images — a start frame, an end frame, references, style
sheets — and until this node existed it had no way to make one. The PreStage
generates them locally with either Krea 2 or Ideogram 4.0 (both open weights,
both native in core) and saves them where the picker already looks, so a
finished still is one chip away from being the next render's keyframe.

It is built exactly like the Creator, because it is driven exactly like the
Creator: zero sockets, one JSON blob the UI owns, weights named by filename,
and an expanded subgraph that loads, samples, decodes and saves — see
`creator_node.py`'s docstring for why a node that samples cannot be an ordinary
node. The one difference is social rather than structural: a PreStage is a
property of the shot being set up, not a node the user hunts the menu for, so
the frontend spawns and removes it from a pill on the Creator/Timeline body
(`js/minimax_creator/prestage.js`) rather than expecting it to be placed by
hand. It still *is* an ordinary node underneath — placeable, copyable,
saveable — because anything else would fight LiteGraph for no benefit.

Queueing both nodes at once is deliberately not an ordering: the hand-off is by
file, so there is no execution edge to get wrong, and ComfyUI's input-hash
caching makes an untouched PreStage a cache hit on the queue that renders the
video.
"""

import json

from comfy_api.latest import io

from . import compile_image, media, outputs, render, render_image

DEFAULT_DATA = json.dumps({
    "version": 1,
    "arch": compile_image.DEFAULT_ARCH,
    "prompt": "",
    "aspect": compile_image.DEFAULT_ASPECT,
    "short_edge": compile_image.DEFAULT_SHORT_EDGE,
    "init": None,
    "refs": [],
    "loras": [],
    "turbo": {"on": False, "quality": compile_image.DEFAULT_TURBO_QUALITY, "saved": None},
    "quality": compile_image.DEFAULT_IDEOGRAM_QUALITY,
    # Where the still lands under output/. Its own default, so the gallery
    # sorts stills apart from finished renders. See `outputs`.
    "output_prefix": outputs.IMAGE_PREFIX,
    # Per-arch sub-blocks, so switching the model pill never forgets the other
    # side's files. Empty rather than guessed — the UI fills it from the
    # listing route, exactly as the Creator's block is filled.
    "models": {"krea2": {}, "ideogram4": {}},
    # A hint for the frontend's peer discovery, never authoritative: node ids
    # renumber on paste, so the pill re-derives the relationship by scan.
    "peer": None,
}, indent=2)


class MiniMaxH3PreStage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import comfy.samplers

        return io.Schema(
            node_id="MiniMaxH3PreStage",
            display_name="MiniMax H3 PreStage",
            category="MiniMax",
            description=(
                "Generate a still with Krea 2 or Ideogram 4.0 for the video "
                "pipeline — a start or end frame, a reference, a style sheet. "
                "Spawned from the pre-stage pill on a Creator or Timeline."
            ),
            enable_expand=True,
            is_output_node=True,
            # The same sampler row, under the same names, as the two video
            # nodes — a control that means the same thing is not called
            # something else here. Defaults are Krea 2 RAW's; the arch and
            # turbo pills rewrite them.
            inputs=[
                io.String.Input("prestage_data", multiline=True, default=DEFAULT_DATA),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=compile_image.KREA_RAW["steps"], min=1, max=10000),
                io.Float.Input("cfg", default=compile_image.KREA_RAW["cfg"], min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS,
                               default=compile_image.KREA_RAW["sampler_name"]),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS,
                               default=compile_image.KREA_RAW["scheduler"],
                               tooltip="Krea 2 samples on this schedule. Ideogram 4 owns its own resolution-shifted schedule and ignores it."),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, prestage_data, **kwargs):
        """Re-run when a referenced file changes on disk — same contract as the
        Creator: media is addressed by filename, so mtimes are all ComfyUI has
        to notice a replaced file by."""
        import os

        from . import lora

        stamps = []
        try:
            data = json.loads(prestage_data)
            names = [ref.get("filename") if isinstance(ref, dict) else ref
                     for ref in data.get("refs") or []]
            init = data.get("init")
            if isinstance(init, dict):
                names.append(init.get("filename"))
            for name in names:
                try:
                    stamps.append(os.path.getmtime(media.resolve(name or "")))
                except Exception:
                    stamps.append(None)
            for entry in data.get("loras", []):
                try:
                    stamps.append(os.path.getmtime(lora.resolve(entry.get("name", ""))))
                except Exception:
                    stamps.append(None)
        except Exception:
            pass
        return (prestage_data, tuple(stamps))

    @classmethod
    def execute(cls, prestage_data, seed, steps, cfg, sampler_name, scheduler) -> io.NodeOutput:
        try:
            data = json.loads(prestage_data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"prestage_data is not valid JSON: {exc}") from exc

        try:
            payload = compile_image.compile_prestage(data, media.image_size)
        except compile_image.CompileError as exc:
            raise ValueError(str(exc)) from exc

        graph = render_image.emit(
            payload,
            render_image.ImageWeights.from_blob(data),
            render.Sampling(seed=seed, steps=steps, cfg=cfg,
                            sampler_name=sampler_name, scheduler=scheduler),
            cls.hidden.unique_id,
            # Refused before anything is sampled — see MiniMaxH3Creator.execute.
            filename_prefix=outputs.image(data))
        return render.expanded(graph)


class MiniMaxH3SaveImage(io.ComfyNode):
    """The last node of an image render: the still, written under output/.

    Core's `SaveImage` would write the same file, but it reports under
    "images", the key the stock frontend preview widget keys on — and with the
    PreStage's id stamped on this node, that widget would land on the canvas
    right under the stage card already showing the same picture. A key core
    does not know keeps the report and loses the widget; stage.js reads it by
    name, exactly as it reads `mmc_video`.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SaveImage",
            display_name="MiniMax H3 Save Image",
            category="MiniMax/internal",
            description="Writes a pre-stage render under output/ and reports it to the stage card.",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default=render_image.FILENAME_PREFIX),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, images, filename_prefix) -> io.NodeOutput:
        import os

        import numpy as np
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        import folder_paths
        from comfy.cli_args import args

        height, width = int(images.shape[1]), int(images.shape[2])
        directory, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)

        # The workflow, so a still dropped back onto the canvas rebuilds the
        # node that made it — the same two hidden fields core's savers write.
        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if cls.hidden.prompt is not None:
                metadata.add_text("prompt", json.dumps(cls.hidden.prompt))
            for key, value in (cls.hidden.extra_pnginfo or {}).items():
                metadata.add_text(key, json.dumps(value))

        results = []
        for image in images:
            array = (image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            filename = f"{name}_{counter:05}_.png"
            Image.fromarray(array).save(os.path.join(directory, filename),
                                        pnginfo=metadata, compress_level=4)
            results.append({"filename": filename, "subfolder": subfolder, "type": "output"})
            counter += 1

        return io.NodeOutput(ui={"mmc_image": results})


NODES = [MiniMaxH3PreStage, MiniMaxH3SaveImage]
