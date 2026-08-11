"""`compile_image.compile_prestage` — the pure half of the image pipeline.

`test_prestage_graph.py` checks what gets built; this checks what gets decided,
and it runs with no ComfyUI, no torch and no GPU because `compile_image.py`
imports none of them and must keep not importing them — `state.js` mirrors this
module, and `test_prestage_mirror.py` can only compare the two by running one of
them in `node`.

    python3 tests/test_compile_image.py
"""

import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# compile_image.py does `from .compile import ...`, so it needs to live in a
# package — a synthetic one, so nothing reaches for server_routes or ComfyUI.
package = types.ModuleType("k2spkg")
package.__path__ = [ROOT]
sys.modules["k2spkg"] = package
for name in ("compile", "compile_image"):
    spec = importlib.util.spec_from_file_location(f"k2spkg.{name}", os.path.join(ROOT, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"k2spkg.{name}"] = module
    spec.loader.exec_module(module)
ci = sys.modules["k2spkg.compile_image"]

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def expect_error(label, fn, fragment):
    try:
        fn()
    except ci.CompileError as exc:
        if fragment not in str(exc):
            FAILURES.append(f"{label}: error {str(exc)!r} does not mention {fragment!r}")
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"{label}: raised {type(exc).__name__} rather than CompileError: {exc}")
    else:
        FAILURES.append(f"{label}: expected an error mentioning {fragment!r}, got none")


def blob(**overrides):
    data = {"version": 1, "arch": "krea2", "prompt": "a red room",
            "aspect": "16:9", "short_edge": 1024, "init": None, "refs": [],
            "loras": [], "turbo": {"on": False, "quality": "good", "saved": None},
            "quality": "default"}
    data.update(overrides)
    return data


def compile(**overrides):
    return ci.compile_prestage(blob(**overrides))


# ---- off by default ----------------------------------------------------------
#
# The claim the whole package rests on: a blob that says nothing about any of
# the new controls compiles to what it always compiled to. Checked here rather
# than trusted, because every later phase adds another field that could quietly
# default itself on.

base = compile()
check("the loader defaults to core's", base.loader, "standard")
check("and a plain blob still resolves the RAW checkpoint",
      base.checkpoint_field, "model")
check("the turbo pill still picks the Turbo checkpoint",
      compile(turbo={"on": True, "quality": "good"}).checkpoint_field, "turbo_model")


# ---- the loader pill ---------------------------------------------------------

svdq = compile(loader="svdquant")
check("the SVDQuant loader reads its own checkpoint field",
      svdq.checkpoint_field, "svdq_model")
check("and says so on the payload", svdq.loader, "svdquant")
# The pills say different things: one picks a precision, the other a schedule.
# A checkpoint quantized from Turbo still needs Turbo's step count, and the
# loader cannot report which one it came from — so turbo stays meaningful and
# stops choosing a file.
check("the turbo pill no longer chooses the file under SVDQuant",
      compile(loader="svdquant", turbo={"on": True, "quality": "good"}).checkpoint_field,
      "svdq_model")

expect_error("an unknown loader is refused",
             lambda: compile(loader="nunchaku"), "unknown loader")
expect_error("the SVDQuant loader is refused on Ideogram, naming the way out",
             lambda: compile(loader="svdquant", arch="ideogram4", refs=[]),
             "Krea 2")


# ---- adapter modes -----------------------------------------------------------

one = compile(loras=[{"name": "grain.safetensors", "strength": 0.8}])
check("a LoRA with no mode chosen gets the exact one",
      one.loras[0]["adapters"], ci.DEFAULT_ADAPTER)
check("which is bypass — the mode that never rewrites the 4-bit weight",
      ci.DEFAULT_ADAPTER, "bypass")

pair = compile(loras=[
    {"name": "grain.safetensors", "strength": 0.8, "adapters": "bake"},
    {"name": "face.safetensors", "strength": 1.0},
])
check("each entry keeps its own mode",
      [entry["adapters"] for entry in pair.loras], ["bake", "bypass"])

# Carried on every entry, not only under SVDQuant, so flipping the loader pill
# and flipping it back does not lose what was chosen.
check("the mode survives on the standard loader too",
      compile(loras=[{"name": "grain.safetensors", "strength": 1.0,
                      "adapters": "bake"}]).loras[0]["adapters"], "bake")

expect_error("an unknown adapter mode is refused, listing the real ones",
             lambda: compile(loras=[{"name": "g.safetensors", "adapters": "fold"}]),
             "bypass")

# A disabled or zero-strength LoRA never reaches the payload, so its mode
# cannot reach the graph either — the existing filter, still doing its job.
check("a zero-strength LoRA is dropped before its mode matters",
      compile(loras=[{"name": "g.safetensors", "strength": 0.0, "adapters": "bake"}]).loras, [])


if FAILURES:
    print(f"{len(FAILURES)} failure(s):")
    for failure in FAILURES:
        print("  -", failure)
    sys.exit(1)
print("all compile_image tests passed")
