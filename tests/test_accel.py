"""Contract tests for `accel.py`, the accelerator wiring.

Runs standalone — `python tests/test_accel.py` — with no torch and no ComfyUI.
Both packs import torch, so the classes here are stand-ins carrying their *real*
`INPUT_TYPES`, copied from the installed sources. That is the point of the test:
`accel.py` reads defaults and preset labels back off whatever is installed, so
what has to be pinned is that the reading works against the shape those packs
actually declare, and that it fails loudly rather than quietly when it does not.
"""

import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# `accel` imports `nodes` (ComfyUI's registry) inside its functions, so a stub
# module under that name is the whole of the harness.
NODES = types.ModuleType("nodes")
NODES.NODE_CLASS_MAPPINGS = {}
sys.modules["nodes"] = NODES

package = types.ModuleType("mmc")
package.__path__ = [ROOT]
sys.modules["mmc"] = package
spec = importlib.util.spec_from_file_location("mmc.accel", os.path.join(ROOT, "accel.py"))
accel = importlib.util.module_from_spec(spec)
sys.modules["mmc.accel"] = accel
spec.loader.exec_module(accel)

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def expect_error(label, fn, fragment):
    try:
        fn()
    except Exception as exc:
        if fragment not in str(exc):
            FAILURES.append(f"{label}: error {str(exc)!r} does not mention {fragment!r}")
    else:
        FAILURES.append(f"{label}: expected an error mentioning {fragment!r}, got none")


# ---- stand-ins for the two installed packs ---------------------------------

class FakeBlockCache:
    """`ApplyMiniMaxH3FirstBlockCache.INPUT_TYPES`, verbatim."""

    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": ([
                    "H3 Safe — 0.08 / max 2",
                    "H3 Fast — 0.10 / max 2",
                    "H3 Aggressive — 0.12 / max 2",
                    "Custom — manual values",
                ], {"default": "H3 Fast — 0.10 / max 2"}),
                "threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.005}),
                "start_percent": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_consecutive_hits": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "temporal_guard": ("BOOLEAN", {"default": False}),
            },
        }

    def apply(self, model, **kwargs):
        return (("block_cache", model, tuple(sorted(kwargs.items()))),)


class FakeSpectrum:
    """`SpectrumApplyMiniMaxH3.INPUT_TYPES`, required half verbatim."""

    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "blend_weight": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "degree": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "ridge_lambda": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 10.0, "step": 0.01}),
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.05}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 8.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 1, "min": 0, "max": 64, "step": 1}),
                "tail_actual_steps": ("INT", {"default": 1, "min": 0, "max": 64, "step": 1}),
                "max_history": ("INT", {"default": 8, "min": 2, "max": 64, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            # Optional inputs are the pack's own business and must not be sent.
            "optional": {
                "history_storage": (["system_ram", "vram"], {"default": "system_ram"}),
                "audio_blend_weight": ("FLOAT", {"default": 0.0}),
            },
        }

    def apply(self, model, **kwargs):
        return (("spectrum", model, tuple(sorted(kwargs.items()))),)


def install(*, block_cache=True, spectrum=True):
    NODES.NODE_CLASS_MAPPINGS = {}
    if block_cache:
        NODES.NODE_CLASS_MAPPINGS[accel.BLOCK_CACHE_NODE] = FakeBlockCache
    if spectrum:
        NODES.NODE_CLASS_MAPPINGS[accel.SPECTRUM_NODE] = FakeSpectrum


class FakeGraph:
    """Enough of `GraphBuilder` to record what was built, in order."""

    def __init__(self):
        self.built = []

    def node(self, node_id, **kwargs):
        self.built.append((node_id, kwargs))
        outer = self

        class Node:
            def out(self, index):
                return f"{node_id}:{index}"

        return Node()


# ---- off is genuinely off ---------------------------------------------------

install()
off = accel.Settings()
check("default settings are off", off.any, False)
check("nothing planned when off", accel.plan(off), [])

graph = FakeGraph()
check("model link passes through untouched", accel.graph_apply(graph, "MODEL_LINK", off), "MODEL_LINK")
check("no nodes built when off", graph.built, [])

# An accelerator that is off must not be built even when its pack is missing —
# nothing should depend on a pack it was not asked to use.
install(block_cache=False, spectrum=False)
check("off needs no pack installed", accel.plan(accel.Settings()), [])

# ---- presets resolve against the pack's own labels --------------------------

install()
for mode, want in [("safe", "H3 Safe — 0.08 / max 2"),
                   ("fast", "H3 Fast — 0.10 / max 2"),
                   ("aggressive", "H3 Aggressive — 0.12 / max 2")]:
    steps = accel.plan(accel.Settings(block_cache=mode))
    check(f"{mode} resolves to the pack's label", steps[0][1]["mode"], want)

# Every required input the pack declares is supplied, and `model` never is —
# a missing required input is a hard executor error at queue time.
kwargs = accel.plan(accel.Settings(block_cache="fast"))[0][1]
check("block cache sends every required input",
      sorted(kwargs),
      ["end_percent", "max_consecutive_hits", "mode", "start_percent", "temporal_guard", "threshold"])
check("block cache keeps the pack's threshold", kwargs["threshold"], 0.10)
check("block cache keeps the pack's window", (kwargs["start_percent"], kwargs["end_percent"]), (0.10, 0.95))

# `off` is ours, not the pack's, and must never be sent as a mode.
check("off is not a pack mode", "off" in [m for m in accel.BLOCK_CACHE_MODES[1:]], False)

# ---- spectrum ---------------------------------------------------------------

kwargs = accel.plan(accel.Settings(spectrum=True))[0][1]
check("spectrum is enabled when asked for", kwargs["enabled"], True)
check("spectrum takes our blend", kwargs["blend_weight"], 0.5)
check("spectrum blend is overridable", accel.plan(accel.Settings(spectrum=True, spectrum_blend=0.8))[0][1]["blend_weight"], 0.8)
check("spectrum keeps the pack's tuning", (kwargs["degree"], kwargs["warmup_steps"], kwargs["max_history"]), (1, 1, 8))
check("spectrum sends no optional inputs", "history_storage" in kwargs, False)

# ---- ordering ---------------------------------------------------------------

both = accel.Settings(block_cache="fast", spectrum=True)
check("block cache is applied before spectrum",
      [node_id for node_id, _ in accel.plan(both)],
      [accel.BLOCK_CACHE_NODE, accel.SPECTRUM_NODE])

graph = FakeGraph()
out = accel.graph_apply(graph, "MODEL_LINK", both)
check("both nodes are built", [node_id for node_id, _ in graph.built],
      [accel.BLOCK_CACHE_NODE, accel.SPECTRUM_NODE])
check("block cache takes the incoming link", graph.built[0][1]["model"], "MODEL_LINK")
check("spectrum chains off the block cache", graph.built[1][1]["model"], f"{accel.BLOCK_CACHE_NODE}:0")
check("the sampler gets spectrum's output", out, f"{accel.SPECTRUM_NODE}:0")

# ---- a missing pack says which, and where to get it -------------------------

install(block_cache=False)
expect_error("missing block cache pack names the node",
             lambda: accel.plan(accel.Settings(block_cache="fast")),
             accel.BLOCK_CACHE_NODE)
expect_error("missing block cache pack names the repo",
             lambda: accel.plan(accel.Settings(block_cache="fast")),
             "duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache")

install(spectrum=False)
expect_error("missing spectrum pack names the repo",
             lambda: accel.plan(accel.Settings(spectrum=True)),
             "xmarre/ComfyUI-Spectrum-MiniMax-H3")

# A pack that renames its presets is refused rather than silently run on a
# preset we picked for the user.
class RenamedBlockCache(FakeBlockCache):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "mode": (["Balanced", "Turbo"], {"default": "Balanced"})}}


install()
NODES.NODE_CLASS_MAPPINGS[accel.BLOCK_CACHE_NODE] = RenamedBlockCache
expect_error("renamed presets are refused",
             lambda: accel.plan(accel.Settings(block_cache="fast")),
             "renamed its modes")

# ---- direct_apply runs the same plan ---------------------------------------

install()
result = accel.direct_apply("MODEL", both)
check("direct_apply chains both packs in order",
      (result[0], result[1][0]), ("spectrum", "block_cache"))
check("direct_apply is a no-op when off", accel.direct_apply("MODEL", accel.Settings()), "MODEL")

if FAILURES:
    print("\n".join(FAILURES))
    raise SystemExit(f"{len(FAILURES)} failure(s)")
print("all accelerator tests passed")
