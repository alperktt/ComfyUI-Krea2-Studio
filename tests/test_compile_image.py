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


# A stand-in for `moodboard.style`, so this stays off disk: the real catalog is
# 9 MB of JSON and the thing under test is what the compile does with what comes
# back, not the catalog.
BOARDS = {
    "noir-1": {"positive": "Style-only guidance: hard chiaroscuro, deep blacks.",
               "negative": "flat lighting, contrastless gray midtones",
               "title": "Cinematic Chiaroscuro Noir"},
    "quiet-1": {"positive": "Style-only guidance: soft daylight, low saturation.",
                "negative": "", "title": "Quiet Daylight"},
}


def fake_lookup(board, strength, collection):
    if board not in BOARDS:
        raise LookupError(f"no moodboard matches {board!r}")
    found = dict(BOARDS[board])
    # The real catalog folds more prose in at higher strengths; enough of that
    # shows here for the tests to tell the levels apart.
    if strength == "strong":
        found["positive"] += " Style keywords: noir, graphic, high contrast."
    return found


def compile(lookup=fake_lookup, **overrides):
    return ci.compile_prestage(blob(**overrides), moodboard_lookup=lookup)


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


# ---- moodboards ---------------------------------------------------------------
#
# A moodboard never reaches the graph: the catalog is read through an injected
# lookup and its guidance is merged into the prompt here, next to the LoRA
# trigger words. So everything worth checking is a string.

check("a default blob has no moodboard and no negative",
      (base.prompt, base.negative_prompt), (blob()["prompt"], None))
# Off must mean the catalog is never opened at all — passing no lookup and
# leaving the pill off has to compile, or an install with the 9 MB catalog
# trimmed away would break for people who never asked for a board.
check("and compiles with no lookup supplied at all",
      ci.compile_prestage(blob()).prompt, blob()["prompt"])

styled = compile(moodboard={"on": True, "board": "noir-1"})
check("the board's guidance is appended, not prepended — the subject stays first",
      styled.prompt.startswith(blob()["prompt"]), True)
check("and it is the board's own text",
      BOARDS["noir-1"]["positive"] in styled.prompt, True)
check("the board's negative becomes real negative conditioning",
      styled.negative_prompt, BOARDS["noir-1"]["negative"])

check("strength reaches the lookup",
      "Style keywords" in compile(moodboard={"on": True, "board": "noir-1",
                                             "strength": "strong"}).prompt, True)
check("a board with no negative guidance leaves the zeroed one alone",
      compile(moodboard={"on": True, "board": "quiet-1"}).negative_prompt, None)
check("and the negative can be declined without losing the look",
      [compile(moodboard={"on": True, "board": "noir-1", "use_negative": False}).negative_prompt,
       BOARDS["noir-1"]["positive"] in compile(
           moodboard={"on": True, "board": "noir-1", "use_negative": False}).prompt],
      [None, True])

# Trigger words and a moodboard are both prompt edits and must compose: the
# LoRA's words go in front of the subject, the board's prose after it.
both = compile(loras=[{"name": "g.safetensors", "strength": 1.0, "triggers": ["gxace"]}],
               moodboard={"on": True, "board": "noir-1"})
check("triggers stay in front and the board stays behind",
      (both.prompt.startswith("gxace, "), both.prompt.rstrip().endswith(BOARDS["noir-1"]["positive"])),
      (True, True))

expect_error("a moodboard pill with no board chosen is refused",
             lambda: compile(moodboard={"on": True, "board": ""}), "no board is chosen")
expect_error("an unknown strength is refused, listing the real ones",
             lambda: compile(moodboard={"on": True, "board": "noir-1", "strength": "loud"}),
             "concise")
expect_error("an unknown collection is refused",
             lambda: compile(moodboard={"on": True, "board": "noir-1", "collection": "mine"}),
             "unknown moodboard collection")
expect_error("a board that is not in the catalog names itself",
             lambda: compile(moodboard={"on": True, "board": "does-not-exist"}),
             "does-not-exist")
expect_error("and a chosen board with no catalog installed says so",
             lambda: compile(lookup=None, moodboard={"on": True, "board": "noir-1"}),
             "catalog is not available")


# ---- krea2edit -----------------------------------------------------------------

check("a default blob carries no edit", base.edit, None)

edited = compile(edit={"on": True, "source": {"filename": "portrait.png"}})
check("the source reaches the payload", edited.edit["source"], "portrait.png")
check("with the release's defaults for everything not set",
      (edited.edit["fit_mode"], edited.edit["grounding_px"],
       edited.edit["ref_boost"], edited.edit["ref_boost_a"], edited.edit["lora"]),
      (ci.DEFAULT_EDIT_FIT, ci.DEFAULT_GROUNDING_PX, ci.DEFAULT_REF_BOOST, 1.0, None))
# The fidelity dial follows the v1.2 release workflow, not the node's own input
# default. The node still says 1.0 because the input was added without changing
# old graphs; the release ships 4 and calls 1.0 "classic v1.1 behavior". Taking
# the node's default would hand everyone the old behaviour without saying so.
check("the fidelity dial is pre-boosted the way the release ships it",
      ci.DEFAULT_REF_BOOST, 4.0)
check("but the scene dial is not — the notes say leave it at 1",
      edited.edit["ref_boost_a"], 1.0)

two = compile(edit={"on": True, "source": {"filename": "scene.png"},
                    "source_b": {"filename": "face.png"}, "ref_boost": 1.4})
check("the second reference keeps its slot — training order is scene, then subject",
      (two.edit["source"], two.edit["source_b"]), ("scene.png", "face.png"))
check("and the boost is carried as given", two.edit["ref_boost"], 1.4)

# The canvas follows the source, the same way it follows an init image: the
# answer to "what shape should this be" is already on screen.
square = compile(aspect="16:9", edit={"on": True, "source": {"filename": "tall.png"}})
check("the canvas follows the edit source's aspect",
      ci.compile_prestage(blob(aspect="16:9",
                               edit={"on": True, "source": {"filename": "tall.png"}}),
                          image_size_lookup=lambda name: (768, 1024),
                          moodboard_lookup=fake_lookup).height >
      ci.compile_prestage(blob(aspect="16:9",
                               edit={"on": True, "source": {"filename": "tall.png"}}),
                          image_size_lookup=lambda name: (768, 1024),
                          moodboard_lookup=fake_lookup).width,
      True)
check("but an init image still wins — it is the latent the sampler starts from",
      ci.compile_prestage(blob(init={"filename": "wide.png"},
                               edit={"on": True, "source": {"filename": "tall.png"}}),
                          image_size_lookup=lambda name: ((1024, 768) if name == "wide.png"
                                                          else (768, 1024)),
                          moodboard_lookup=fake_lookup).width >
      ci.compile_prestage(blob(init={"filename": "wide.png"},
                               edit={"on": True, "source": {"filename": "tall.png"}}),
                          image_size_lookup=lambda name: ((1024, 768) if name == "wide.png"
                                                          else (768, 1024)),
                          moodboard_lookup=fake_lookup).height,
      True)

# An edit composes with everything that is not conditioning.
check("an edit and a moodboard compose",
      bool(compile(edit={"on": True, "source": {"filename": "p.png"}},
                   moodboard={"on": True, "board": "noir-1"}).edit), True)
check("and an edit runs on the quantized loader",
      compile(loader="svdquant", edit={"on": True, "source": {"filename": "p.png"}}
              ).checkpoint_field, "svdq_model")

# The size advisory, which is an observation and never a clamp: the resolution
# pill stays the user's and an oversize edit still renders.
check("a 1 MP edit is not flagged",
      ci.compile_prestage(blob(aspect="1:1", short_edge=1024,
                               edit={"on": True, "source": {"filename": "p.png"}}),
                          moodboard_lookup=fake_lookup).edit_oversize, False)
check("a 16:9 edit at a 1024 short edge is — 1.9 MP is past where the weights hold",
      ci.compile_prestage(blob(aspect="16:9", short_edge=1024,
                               edit={"on": True, "source": {"filename": "p.png"}}),
                          moodboard_lookup=fake_lookup).edit_oversize, True)
check("and a second reference raises the ceiling rather than lowering it",
      ci.compile_prestage(blob(aspect="3:2", short_edge=1024,
                               edit={"on": True, "source": {"filename": "p.png"},
                                     "source_b": {"filename": "q.png"}}),
                          moodboard_lookup=fake_lookup).edit_oversize, False)
check("nothing is flagged when there is no edit", base.edit_oversize, False)

expect_error("the edit pill with no source is refused",
             lambda: compile(edit={"on": True}), "no source image is chosen")
expect_error("an edit and style references together are refused, naming both ways out",
             lambda: compile(edit={"on": True, "source": {"filename": "p.png"}},
                             refs=[{"filename": "style.png"}]),
             "positive conditioning")
expect_error("an edit is refused on Ideogram",
             lambda: compile(arch="ideogram4",
                             edit={"on": True, "source": {"filename": "p.png"}}),
             "Krea 2")
expect_error("an unknown fit mode is refused",
             lambda: compile(edit={"on": True, "source": {"filename": "p.png"},
                                   "fit_mode": "stretch"}),
             "unknown edit fit mode")
expect_error("a grounding resolution past the node's ceiling is refused",
             lambda: compile(edit={"on": True, "source": {"filename": "p.png"},
                                   "grounding_px": 9000}),
             "grounding resolution")
expect_error("and a boost past the node's ceiling is refused",
             lambda: compile(edit={"on": True, "source": {"filename": "p.png"},
                                   "ref_boost": 5000}),
             "ref_boost")


if FAILURES:
    print(f"{len(FAILURES)} failure(s):")
    for failure in FAILURES:
        print("  -", failure)
    sys.exit(1)
print("all compile_image tests passed")
