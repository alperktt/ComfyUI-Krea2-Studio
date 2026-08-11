"""The MiniMax H3 Creator node.

One node, one prompt box, one video — and no sockets at all. Media is chosen in
the UI and loaded from ComfyUI/input by filename; the weights are chosen the same
way and loaded by `models.emit_links` inside the subgraph; and the finished clip
is muxed, saved and played in the node body rather than handed to whatever the
user wired downstream. What the user attaches decides the mode, and the mode
decides which of the two checkpoints is loaded — FL2VA and Ref2VA are separate
weights, so routing the right one is the node's job rather than the user's, and
only the routed one is built. The routing can be pinned from the UI (`checkpoint`
in the blob) when you want the other weights on the same payload;
`compile._resolve_checkpoint` owns which pins are allowed.

**This node owns the sampler.** It used to hand out conditioning and let the
graph do the sampling, which meant every workflow re-assembled the same six
nodes behind it and got to choose wrong: the H3 templates sample with
`res_multistep` and decode sound with `VAEDecodeAudio`, and a hand-wired graph
that picked the defaults instead was quietly worse. Owning it also puts the two
optional accelerators somewhere they can be switched on, which they cannot be
from outside a node that ends at conditioning.

The cost is the one the Timeline node's docstring already argued: a node that
samples cannot be an ordinary node, because ComfyUI has no way to express "and
then sample" except by returning a subgraph. So `execute` compiles the blob to a
single payload, hands it to `render.emit`, and returns that subgraph through the
`expand` mechanism. A Creator render is exactly a one-segment timeline — same
payload shape, same emitted graph — which is why neither of them owns a copy of
it.

The node is also an *output* node, which is the other half of having no sockets:
`render.emit_tail` writes the file and stamps this node's id on the save node, so
the result is reported back against the node the user is looking at.

`creator_data` is the UI's serialised state and is managed entirely by `js/`. It
is a normal widget only so it round-trips through saved workflows; hand-editing
it is supported (that is how phase 1 was tested) but the frontend will overwrite
it.
"""

import json

from comfy_api.latest import ComfyExtension, io

from . import (accel, canvas, hires, lora, media, models, outputs, prestage,
               render, settings, timeline)

DEFAULT_DATA = json.dumps({
    "version": 1,
    "prompt": "",
    "assets": [],
    "loras": [],
    "duration_s": 6,
    "aspect": "16:9",
    "short_edge": canvas.NATIVE_SHORT_EDGE,
    "checkpoint": "auto",
    # Where the finished clip lands under output/. See `outputs`.
    "output_prefix": outputs.VIDEO_PREFIX,
    # Which files to load. Empty here rather than guessed: a fresh node has no
    # idea what is on this machine, and the UI fills it from the listing route.
    "models": {},
}, indent=2)


class MiniMaxH3Creator(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import comfy.samplers

        return io.Schema(
            node_id="MiniMaxH3Creator",
            display_name="MiniMax H3 Creator",
            category="MiniMax",
            description=(
                "Describe a video and reference attached media with @. Routes to the "
                "FL2VA or Ref2VA checkpoint depending on what you attach, samples it, "
                "and returns the finished frames and sound."
            ),
            # This node returns a subgraph rather than tensors, because it owns
            # the sampler — see the module docstring. It is also an output node:
            # it saves the finished clip itself, which is what lets it have no
            # output sockets either.
            enable_expand=True,
            is_output_node=True,
            # Deliberately the same inputs, in the same order, under the same
            # names as MiniMaxH3Timeline. The two nodes differ in what they
            # generate, not in how they are driven, and a control that means the
            # same thing should not be called something else here.
            inputs=[
                # No model sockets. The weights are named in `creator_data` and
                # `models.emit_links` builds the loaders inside the subgraph.
                io.String.Input("creator_data", multiline=True, default=DEFAULT_DATA),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000),
                # The released H3 checkpoints are CFG-distilled, so guidance is
                # already in the weights and 1.0 is the value they were trained
                # to run at. Left as an ordinary widget: it is a default, not a
                # constraint, and anyone who wants to push it can.
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS,
                               default="res_multistep"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS,
                               default="simple",
                               tooltip="The templates use 'simple'; for reference-heavy prompts they suggest 'beta' or 'normal' instead."),
                io.Combo.Input("block_cache", options=accel.BLOCK_CACHE_MODES, default="off",
                    tooltip="FirstBlockCache: skip the rest of the DiT on steps where the first block barely moved. 'fast' is the pack's recommended preset. Needs ComfyUI-MiniMaxH3-FirstBlockCache."),
                io.Boolean.Input("spectrum", default=False,
                    tooltip="Spectrum: forecast features across steps instead of evaluating every one. Needs ComfyUI-Spectrum-MiniMax-H3. Combines with block_cache; cannot be combined with EasyCache."),
                io.Float.Input("spectrum_blend", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="Spectrum's video spectral share. Higher is faster and further from a native render. Ignored unless 'spectrum' is on."),
            ],
            # Nothing comes out either: the render is saved and shown in the node
            # body, so there is no socket for a graph to hang off.
            outputs=[],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, creator_data, **kwargs):
        """Re-run when a referenced file changes on disk.

        Media is addressed by filename, not by a wired tensor, so ComfyUI has
        nothing else to notice a replaced file by.
        """
        import os

        stamps = []
        try:
            data = json.loads(creator_data)
            for asset in data.get("assets", []):
                try:
                    stamps.append(os.path.getmtime(media.resolve(asset.get("filename", ""))))
                except Exception:
                    stamps.append(None)
            for entry in data.get("loras", []):
                try:
                    stamps.append(os.path.getmtime(lora.resolve(entry.get("name", ""))))
                except Exception:
                    stamps.append(None)
        except Exception:
            pass
        return (creator_data, tuple(stamps))

    @classmethod
    def execute(cls, creator_data, seed, steps, cfg, sampler_name, scheduler,
                block_cache="off", spectrum=False, spectrum_blend=0.5) -> io.NodeOutput:
        try:
            data = json.loads(creator_data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"creator_data is not valid JSON: {exc}") from exc

        # A whole creator request is one segment payload with nothing in front of
        # it. Built here rather than by a compile-time helper because there is
        # nothing to work out: no seam to continue from, and no shared canvas to
        # be held to, so a start frame sets the aspect adaptively exactly as it
        # always has.
        # A hand-written blob may still carry `prompt_override`, which replaces
        # the composed prompt verbatim — see `MiniMaxH3TimelineSegment.execute`.
        # There is no socket for it any more: the node has none, and the
        # refiner's editable rewrite is the same escape hatch with a UI on it.
        payload = {"request": data, "continue": False, "continue_audio": False}
        if data.get("prompt_override"):
            payload["prompt_override"] = data["prompt_override"]

        graph = render.emit(
            [payload], ["This generation"],
            models.Weights.from_blob(data),
            render.Sampling(seed=seed, steps=steps, cfg=cfg,
                            sampler_name=sampler_name, scheduler=scheduler),
            accel.Settings(block_cache=block_cache, spectrum=spectrum,
                           spectrum_blend=spectrum_blend),
            cls.hidden.unique_id,
            # Resolved here rather than inside the save node: a prefix that
            # cannot be used should stop the queue before anything is sampled,
            # not after — `get_save_image_path` raising at the end of a render
            # costs the user the render.
            filename_prefix=outputs.video(data, settings.video_prefix()))
        return render.expanded(graph)


class MiniMaxCreatorExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3Creator, *timeline.NODES, *prestage.NODES, *hires.NODES]


async def comfy_entrypoint() -> MiniMaxCreatorExtension:
    return MiniMaxCreatorExtension()
