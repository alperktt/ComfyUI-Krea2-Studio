"""What the UI holds survives into the blob, and into the payload.

This test exists because it did not, and the bug it would have caught shipped:
`serializePreStage` is an explicit whitelist of keys, and seven Krea 2 controls
were added to the state, the parser, the pills and the compiler — but not to it.
Every symptom followed from that one omission. The pills lit up and changed
nothing; the resolution slider reached 4096 and the render came out at 2048;
opening another tab reset everything to standard, because the blob being read
back had never carried any of it.

Neither of the other tests could see it. `test_prestage_mirror` compares
constants, and `test_prestage_graph` builds payloads from dicts it writes
itself — so both agreed with each other while the one path the user actually
travels was broken.

So this closes the loop: state -> `serializePreStage` -> `parsePreStage` (in
node), and the same blob -> `compile_prestage` (in Python). A control that does
not survive both is a control that does nothing.

    python3 tests/test_prestage_roundtrip.py

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

package = types.ModuleType("rtpkg")
package.__path__ = [ROOT]
sys.modules["rtpkg"] = package
for name in ("canvas", "contextir", "compile", "compile_image", "compile_still"):
    spec = importlib.util.spec_from_file_location(f"rtpkg.{name}", os.path.join(ROOT, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"rtpkg.{name}"] = module
    spec.loader.exec_module(module)
ci = sys.modules["rtpkg.compile_image"]

# Every Krea 2 control, set to something that is not its default, in the shape
# the editors actually leave on the state.
SCRIPT = """
const { pathToFileURL } = await import("node:url");
const s = await import(pathToFileURL(process.argv[1]).href);

const state = s.parsePreStage(JSON.stringify({ version: 1 }));
state.prompt = "a red room";
state.aspect = "21:9";
state.short_edge = 4096;
state.models = { krea2: { model: "raw.safetensors", turbo_model: "turbo.safetensors",
                         svdq_model: "svdq.safetensors", clip: "te.safetensors",
                         vae: "vae.safetensors" }, ideogram4: {}, dtype: "default" };

state.loader = "svdquant";
state.loras = [{ name: "grain.safetensors", strength: 0.8, adapters: "bake", enabled: true }];
state.moodboard = { on: true, board: "noir-1", title: "Noir", strength: "strong",
                    collection: "krea", use_negative: false };
state.edit = { on: true, source: { filename: "man.png" }, source_b: { filename: "face.png" },
               lora: "identity.safetensors", lora_strength: 0.9,
               ref_boost: 5.5, ref_boost_a: 1.25, fit_mode: "crop (legacy)",
               grounding_px: 512 };
state.style = { on: true, refs: [{ filename: "look.png" }], fit: "contain",
                strength: 1.35, primary: 1 };
state.dype = { on: true, method: "yarn", scale: 3.5, yarn_alt: true };
// SEGA is the alternative to DyPE, not an addition, so it is carried on its own
// state below rather than alongside.
state.sega = { on: false, method: "ntk", alpha: 0.42 };

// An edit and style references cannot coexist, and the parser enforces it, so
// the reference list stays empty here rather than fighting that rule.
state.refs = [];

const blob = s.serializePreStage(state);
const reparsed = s.parsePreStage(blob);

// Stages is checked on its own state: it is mutually exclusive with an init
// image and with the quantized loader, so it cannot ride along with the above.
const staged = s.parsePreStage(JSON.stringify({ version: 1 }));
staged.prompt = "p";
staged.stages = { count: 3, handoff: 22.5, handoff3: 77.5, stage1_scale: 0.5 };
const stagedBlob = s.serializePreStage(staged);

// SEGA likewise: it is the alternative to DyPE, so it gets its own state rather
// than riding along with the one that has DyPE on.
const segaState = s.parsePreStage(JSON.stringify({ version: 1 }));
segaState.prompt = "p";
segaState.aspect = "21:9";
segaState.short_edge = 4096;
segaState.sega = { on: true, method: "ntk", alpha: 0.42 };
const segaBlob = s.serializePreStage(segaState);

// And a blob that claims both is settled rather than carried, so a hand-edited
// workflow cannot arrive with a pair the compile will refuse.
const bothOn = s.parsePreStage(JSON.stringify({
  version: 1, prompt: "p", dype: { on: true }, sega: { on: true } }));

console.log(JSON.stringify({
  blob, reparsed, stagedBlob, segaBlob,
  stagedReparsed: s.parsePreStage(stagedBlob),
  segaReparsed: s.parsePreStage(segaBlob),
  bothOn: { dype: bothOn.dype.on, sega: bothOn.sega.on },
  // What a fresh node writes, so "off" can be checked as being genuinely absent.
  emptyBlob: s.serializePreStage(s.parsePreStage(JSON.stringify({ version: 1 }))),
}));
"""

result = subprocess.run(["node", "--input-type=module", "--eval", SCRIPT, MIRROR],
                        capture_output=True, text=True)
if result.returncode != 0:
    print("failed to run state.js:\n" + result.stderr.strip())
    sys.exit(1)
out = json.loads(result.stdout)
blob = json.loads(out["blob"])
back = out["reparsed"]

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


# ---- 1. the blob carries every control ----------------------------------------
#
# The bug was exactly this: present in the state, absent from the blob. Checked
# key by key rather than by comparing whole objects, so a failure names the field.

for key in ("loader", "moodboard", "edit", "style", "dype"):
    check(f"the blob carries {key}", key in blob, True)
check("the blob carries stages", "stages" in json.loads(out["stagedBlob"]), True)
sega_blob = json.loads(out["segaBlob"])
check("the blob carries sega", "sega" in sega_blob, True)

check("the loader", blob.get("loader"), "svdquant")
check("the LoRA's adapter mode", [e.get("adapters") for e in blob.get("loras", [])], ["bake"])
check("the moodboard board", blob.get("moodboard", {}).get("board"), "noir-1")
check("its strength", blob.get("moodboard", {}).get("strength"), "strong")
check("its negative switch", blob.get("moodboard", {}).get("use_negative"), False)
check("the edit source", blob.get("edit", {}).get("source", {}).get("filename"), "man.png")
check("its second reference", blob.get("edit", {}).get("source_b", {}).get("filename"), "face.png")
check("its LoRA", blob.get("edit", {}).get("lora"), "identity.safetensors")
check("its boost", blob.get("edit", {}).get("ref_boost"), 5.5)
check("its fit", blob.get("edit", {}).get("fit_mode"), "crop (legacy)")
check("its grounding", blob.get("edit", {}).get("grounding_px"), 512)
check("the style references", [r["filename"] for r in blob.get("style", {}).get("refs", [])],
      ["look.png"])
check("the style fit and strength",
      (blob.get("style", {}).get("fit"), blob.get("style", {}).get("strength")),
      ("contain", 1.35))
check("the DyPE method and scale",
      (blob.get("dype", {}).get("method"), blob.get("dype", {}).get("scale")), ("yarn", 3.5))
check("its yarn scaling", blob.get("dype", {}).get("yarn_alt"), True)
check("the SEGA method and amplitude",
      (sega_blob["sega"].get("method"), sega_blob["sega"].get("alpha")), ("ntk", 0.42))

staged_blob = json.loads(out["stagedBlob"])
check("the stage count", staged_blob.get("stages", {}).get("count"), 3)
check("both handoffs",
      (staged_blob["stages"].get("handoff"), staged_blob["stages"].get("handoff3")),
      (22.5, 77.5))
check("the first-stage scale", staged_blob["stages"].get("stage1_scale"), 0.5)


# ---- 2. reading it back gives the same state -----------------------------------
#
# The other half of the reported bug: switching tabs re-parsed the blob, and
# anything the blob had dropped came back as a default.

check("the loader survives a reload", back["loader"], "svdquant")
check("the adapter mode survives", [e["adapters"] for e in back["loras"]], ["bake"])
check("the moodboard survives",
      (back["moodboard"]["on"], back["moodboard"]["board"], back["moodboard"]["strength"]),
      (True, "noir-1", "strong"))
check("the edit survives",
      (back["edit"]["on"], back["edit"]["source"]["filename"], back["edit"]["ref_boost"],
       back["edit"]["grounding_px"]),
      (True, "man.png", 5.5, 512))
check("the style survives",
      (back["style"]["on"], back["style"]["fit"], back["style"]["strength"]),
      (True, "contain", 1.35))
check("DyPE survives",
      (back["dype"]["on"], back["dype"]["method"], back["dype"]["scale"], back["dype"]["yarn_alt"]),
      (True, "yarn", 3.5, True))
sega_back = out["segaReparsed"]
check("SEGA survives",
      (sega_back["sega"]["on"], sega_back["sega"]["method"], sega_back["sega"]["alpha"]),
      (True, "ntk", 0.42))
check("and it did not drag DyPE on with it", sega_back["dype"]["on"], False)
# The pair is settled on the way in, so the UI can never hand the compile a
# combination it would refuse.
check("a blob claiming both keeps only DyPE",
      (out["bothOn"]["dype"], out["bothOn"]["sega"]), (True, False))
staged_back = out["stagedReparsed"]
check("the stages survive",
      (staged_back["stages"]["count"], staged_back["stages"]["handoff"],
       staged_back["stages"]["stage1_scale"]),
      (3, 22.5, 0.5))

# And off stays off: a fresh node writes none of these keys, so an untouched
# PreStage compiles to exactly what it compiled to before any of them existed.
empty = json.loads(out["emptyBlob"])
for key in ("loader", "moodboard", "edit", "style", "stages", "dype", "sega"):
    check(f"a fresh blob does not carry {key}", key in empty, False)


# ---- 3. the compiler sees it ---------------------------------------------------
#
# The blob is what the node executes, so the last link is what `compile_prestage`
# makes of it. This is where the 4096-slider-2048-render symptom lived.

payload = ci.compile_prestage(
    blob,
    image_size_lookup=lambda name: (1024, 1024),
    moodboard_lookup=lambda board, strength, collection: {
        "positive": "hard chiaroscuro", "negative": "flat lighting", "title": "Noir"})

check("the compiler sees the quantized loader", payload.loader, "svdquant")
check("and reads its checkpoint field", payload.checkpoint_field, "svdq_model")
check("the compiler sees the edit", payload.edit["source"], "man.png")
check("the compiler sees the style transfer", payload.style["refs"], ["look.png"])
check("the compiler sees DyPE", payload.dype["method"], "yarn")
check("the compiler sees SEGA, on the state that carries it",
      ci.compile_prestage(sega_blob).sega["alpha"], 0.42)
check("and the SEGA render reaches the raised ceiling too",
      max(ci.compile_prestage(sega_blob).width,
          ci.compile_prestage(sega_blob).height), ci.POSITION_MAX_SHORT_EDGE)
check("the moodboard reached the prompt", "hard chiaroscuro" in payload.prompt, True)
check("and its negative was declined, as the blob asked", payload.negative_prompt, None)

# The one the user saw first: the slider reached 4096 and the render did not.
# The edit source makes the canvas adaptive, so this is checked without it.
tall = dict(blob)
tall.pop("edit")
tall["short_edge"] = 4096
tall["aspect"] = "21:9"
raised = ci.compile_prestage(tall, moodboard_lookup=lambda *a: {
    "positive": "x", "negative": "", "title": "t"})
check("a 4096 request with DyPE on is not clamped to the model's ceiling",
      max(raised.width, raised.height) > ci.MAX_SHORT_EDGE, True)
check("and it lands on DyPE's ceiling instead",
      max(raised.width, raised.height), ci.POSITION_MAX_SHORT_EDGE)

no_dype = dict(tall)
no_dype.pop("dype")
check("with DyPE off the same request is clamped, as it should be",
      max(ci.compile_prestage(no_dype, moodboard_lookup=lambda *a: {
          "positive": "x", "negative": "", "title": "t"}).width,
          ci.compile_prestage(no_dype, moodboard_lookup=lambda *a: {
              "positive": "x", "negative": "", "title": "t"}).height),
      ci.MAX_SHORT_EDGE)


if FAILURES:
    print(f"{len(FAILURES)} failure(s):")
    for failure in FAILURES:
        print("  -", failure)
    sys.exit(1)
print(f"state -> blob -> payload holds across {len(blob)} blob keys")
