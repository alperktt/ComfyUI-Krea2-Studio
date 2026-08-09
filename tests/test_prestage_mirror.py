"""`state.js`'s pre-stage section still agrees with `compile_image.py`.

Same contract as `test_canvas_mirror.py`: the duplication is deliberate — the
pills resolve the image canvas live and the pills' numbers are the presets —
but it is only safe while the two agree. `compile_image.py` is authoritative.

    python3 tests/test_prestage_mirror.py

Skips itself if node is not installed.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR = os.path.join(ROOT, "js", "minimax_creator", "state.js")

if shutil.which("node") is None:
    print("skipped: node is not installed")
    sys.exit(0)

# compile_image.py does `from .compile import ...`, so it needs to live in a
# package — a synthetic one, so nothing imports server_routes or ComfyUI.
package = types.ModuleType("mmcpkg")
package.__path__ = [ROOT]
sys.modules["mmcpkg"] = package
for name in ("compile", "compile_image"):
    spec = importlib.util.spec_from_file_location(f"mmcpkg.{name}", os.path.join(ROOT, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"mmcpkg.{name}"] = module
    spec.loader.exec_module(module)
ci = sys.modules["mmcpkg.compile_image"]

SCRIPT = """
const s = await import(process.argv[1]);
const out = { constants: {}, canvases: {}, ideogram: {}, turbo: s.PRESTAGE_TURBO_STEPS,
              krea_raw: s.PRESTAGE_KREA_RAW };
for (const name of ["PRESTAGE_CANVAS_MULTIPLE", "PRESTAGE_MIN_EDGE", "PRESTAGE_MAX_EDGE",
                    "PRESTAGE_DEFAULT_EDGE", "PRESTAGE_MAX_PIXELS", "PRESTAGE_MAX_REFS",
                    "PRESTAGE_DEFAULT_DENOISE", "PRESTAGE_MIN_DENOISE"]) {
  out.constants[name] = s[name];
}
out.arches = [...s.PRESTAGE_ARCHES];
out.presets = s.PRESTAGE_ASPECTS.map(([label]) => label).sort();
for (const [label] of s.PRESTAGE_ASPECTS) {
  for (const edge of [512, 768, 1024, 1536, 2048]) {
    const g = s.resolvedPreStage({ aspect: label, short_edge: edge, init: null, refs: [] });
    out.canvases[label + "@" + edge] = [g.width, g.height];
  }
}
for (const quality of s.PRESTAGE_IDEOGRAM_QUALITIES) {
  out.ideogram[quality] = s.PRESTAGE_IDEOGRAM_STEPS[quality];
}
console.log(JSON.stringify(out));
"""

result = subprocess.run(
    ["node", "--input-type=module", "--eval", SCRIPT, MIRROR],
    capture_output=True, text=True)
if result.returncode != 0:
    print("failed to read state.js:\n" + result.stderr.strip())
    sys.exit(1)
mirror = json.loads(result.stdout)

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: state.js says {got!r}, compile_image.py says {want!r}")


PY_CONSTANTS = {
    "PRESTAGE_CANVAS_MULTIPLE": ci.CANVAS_MULTIPLE,
    "PRESTAGE_MIN_EDGE": ci.MIN_SHORT_EDGE,
    "PRESTAGE_MAX_EDGE": ci.MAX_SHORT_EDGE,
    "PRESTAGE_DEFAULT_EDGE": ci.DEFAULT_SHORT_EDGE,
    "PRESTAGE_MAX_PIXELS": ci.MAX_PIXELS,
    "PRESTAGE_MAX_REFS": ci.MAX_STYLE_REFS,
    "PRESTAGE_DEFAULT_DENOISE": ci.DEFAULT_DENOISE,
    "PRESTAGE_MIN_DENOISE": ci.MIN_DENOISE,
}
for name, value in mirror["constants"].items():
    check(name, value, PY_CONSTANTS[name])

check("arches", mirror["arches"], list(ci.ARCHES))
check("aspect presets", mirror["presets"], sorted(ci.ASPECT_PRESETS))

for key, size in mirror["canvases"].items():
    label, edge = key.split("@")
    check(key, size, list(ci.resolve_canvas(ci.ASPECT_PRESETS[label], int(edge))))

for quality, steps in mirror["ideogram"].items():
    check(f"ideogram {quality} steps", steps, ci.IDEOGRAM_QUALITIES[quality]["steps"])

check("turbo steps", mirror["turbo"], ci.TURBO_STEPS)
check("krea RAW row", mirror["krea_raw"], ci.KREA_RAW)

if FAILURES:
    print(f"{len(FAILURES)} disagreement(s):")
    for failure in FAILURES:
        print("  -", failure)
    sys.exit(1)
print(f"state.js mirrors compile_image.py across {len(mirror['canvases'])} canvases")
