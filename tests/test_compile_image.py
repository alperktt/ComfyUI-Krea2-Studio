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


# ---- style transfer ------------------------------------------------------------

check("a default blob carries no style transfer", base.style, None)

one_ref = compile(style={"on": True, "refs": [{"filename": "look.png"}]})
check("the reference reaches the payload", one_ref.style["refs"], ["look.png"])
check("with the pack's own default fit and strength",
      (one_ref.style["fit"], one_ref.style["strength"]),
      (ci.DEFAULT_STYLE_FIT, ci.DEFAULT_STYLE_STRENGTH))

pair_refs = compile(style={"on": True, "refs": ["a.png", "b.png"], "primary": 2,
                           "fit": "contain", "strength": 1.25})
check("two references and which one leads",
      (pair_refs.style["refs"], pair_refs.style["primary"]), (["a.png", "b.png"], 2))
check("and the fit and strength are carried",
      (pair_refs.style["fit"], pair_refs.style["strength"]), ("contain", 1.25))

# Composes with everything that is not the other reference path.
check("style transfer and an edit compose",
      bool(compile(style={"on": True, "refs": ["a.png"]},
                   edit={"on": True, "source": {"filename": "p.png"}}).style), True)
check("and it runs on the quantized loader",
      compile(loader="svdquant", style={"on": True, "refs": ["a.png"]}
              ).checkpoint_field, "svdq_model")
check("and alongside a moodboard",
      bool(compile(style={"on": True, "refs": ["a.png"]},
                   moodboard={"on": True, "board": "noir-1"}).style), True)

expect_error("the style pill with no reference is refused",
             lambda: compile(style={"on": True, "refs": []}), "no reference is chosen")
expect_error("a third reference is refused, naming the pack's limit",
             lambda: compile(style={"on": True, "refs": ["a.png", "b.png", "c.png"]}),
             "no route for a third")
expect_error("style transfer and style references together are refused",
             lambda: compile(style={"on": True, "refs": ["a.png"]},
                             refs=[{"filename": "style.png"}]),
             "two different reference paths")
expect_error("style transfer is refused on Ideogram",
             lambda: compile(arch="ideogram4", style={"on": True, "refs": ["a.png"]}),
             "Krea 2")
expect_error("an unknown fit is refused",
             lambda: compile(style={"on": True, "refs": ["a.png"], "fit": "tile"}),
             "unknown style fit")
expect_error("a primary reference that is not 1 or 2 is refused",
             lambda: compile(style={"on": True, "refs": ["a.png"], "primary": 3}),
             "must be 1 or 2")


# ---- the multi-stage sampler ----------------------------------------------------

check("a default blob samples in one pass", base.stages, None)
check("and an explicit one stage is the same as none",
      compile(stages={"count": 1}).stages, None)

two_stage = compile(stages={"count": 2})
check("two stages carry the pack's handoff default",
      (two_stage.stages["count"], two_stage.stages["handoff"]), (2, ci.DEFAULT_HANDOFF))
check("stage 1 loads the base checkpoint, not Turbo",
      two_stage.checkpoint_field, "model")
# The turbo pill stops choosing a file and keeps choosing stage 2's step budget,
# so the quality has to travel with the block.
check("and the turbo quality rides along for stage 2",
      two_stage.stages["quality"], ci.DEFAULT_TURBO_QUALITY)
check("even with the turbo pill on, stage 1 is still the base",
      compile(stages={"count": 2}, turbo={"on": True, "quality": "draft"}).checkpoint_field,
      "model")
check("and the pill's quality is what stage 2 uses",
      compile(stages={"count": 2}, turbo={"on": True, "quality": "draft"}).stages["quality"],
      "draft")

three = compile(stages={"count": 3, "handoff": 20, "handoff3": 80})
check("three stages carry both crossovers",
      (three.stages["handoff"], three.stages["handoff3"]), (20.0, 80.0))

# Dual resolution: stage 1 small, stage 2 finishing at the target. 0/0 is the
# node's own "do not resize", which is what a scale of 1 has to compile to —
# passing the target twice would make it upscale from and to the same size.
check("no resize when stage 1 runs at full size",
      (two_stage.stages["width"], two_stage.stages["height"]), (0, 0))
half = compile(aspect="1:1", short_edge=1024, stages={"count": 2, "stage1_scale": 0.5})
check("a scaled first stage gets its own canvas, on the same /16 grid",
      (half.stages["width"], half.stages["height"], half.width), (512, 512, 1024))

expect_error("an unknown stage count is refused",
             lambda: compile(stages={"count": 4}), "unknown stage count")
expect_error("a third crossover before the second is refused",
             lambda: compile(stages={"count": 3, "handoff": 60, "handoff3": 40}),
             "at or after")
expect_error("a multi-stage run on the quantized loader is refused, naming both ways out",
             lambda: compile(loader="svdquant", stages={"count": 2}),
             "one quantized file")
expect_error("a multi-stage run with an init image is refused — it has no denoise",
             lambda: compile(init={"filename": "seed.png"}, stages={"count": 2}),
             "no denoise")
expect_error("and it is refused on Ideogram",
             lambda: compile(arch="ideogram4", stages={"count": 2}), "Krea 2")


# ---- an explicit width x height --------------------------------------------------
#
# The case this route exists for, verbatim: 1000x4000 on the ratio route came out
# as 1:3, because 0.25 is below MIN_RATIO and `clamp_ratio` silently rounded it.

check("a size past the ratio clamp is honoured, not rounded to 1:3",
      (lambda p: (p.width, p.height))(compile(size={"width": 1000, "height": 4000},
                                              sega={"on": True})),
      (1008, 4000))
clamped = compile(aspect="0.25", short_edge=1008, sega={"on": True})
check("...where the ratio route clamps the same shape to 1:3",
      round(clamped.height / clamped.width, 2), 3.0)
check("and says it clamped, which is the only sign the ratio route gives",
      clamped.ratio_clamped, True)
# 1000 is not on the /16 grid the DiT patchifies on, so it lands on 1008. Snapped
# rather than refused: the nearest cell is what every other route produces too.
check("both axes snap to the grid", compile(size={"width": 1000, "height": 4000},
                                           sega={"on": True}).width % ci.CANVAS_MULTIPLE, 0)
check("a size wins over the aspect pill beside it",
      (lambda p: (p.width, p.height))(compile(aspect="16:9", size={"width": 1024,
                                                                  "height": 1024})),
      (1024, 1024))
check("and over an init image's shape, because typing is the more specific answer",
      (lambda p: (p.width, p.height))(
          ci.compile_prestage(blob(init={"filename": "wide.png"},
                                   size={"width": 1024, "height": 1536}),
                              image_size_lookup=lambda name: (1920, 1080),
                              moodboard_lookup=fake_lookup)),
      (1024, 1536))
check("while the same blob without a size does follow the image",
      (lambda p: round(p.width / p.height, 2))(
          ci.compile_prestage(blob(init={"filename": "wide.png"}),
                              image_size_lookup=lambda name: (1920, 1080),
                              moodboard_lookup=fake_lookup)),
      1.78)
check("no size key still means the ratio route, exactly as before",
      (lambda p: (p.width, p.height))(compile(aspect="1:1", short_edge=1024)),
      (1024, 1024))

expect_error("a size needs both numbers",
             lambda: compile(size={"width": 1024}), "height is missing")
expect_error("an axis past the ceiling is refused rather than quietly scaled",
             lambda: compile(size={"width": 4096, "height": 1024}), "between")
# The per-axis cap is the binding one on this route: with both axes inside
# `max_edge`, the area cannot exceed `max_edge ** 2`, which is what `max_pixels`
# is. So the area check in `_parse_size` is unreachable by construction — kept
# because the two caps are separate arguments and a future pair need not agree.
check("the area cap is implied by the per-axis one, not a second gate",
      (ci.MAX_PIXELS, ci.POSITION_MAX_PIXELS),
      (ci.MAX_SHORT_EDGE ** 2, ci.POSITION_MAX_SHORT_EDGE ** 2))
check("the largest square the default route allows is exactly the cap",
      compile(size={"width": 2048, "height": 2048}).width, ci.MAX_SHORT_EDGE)
# And the ceiling the position pill raises applies to a typed size too, so the
# refusal above turns into an accepted render once a patch is on.
expect_error("4096 wide is refused with no position patch",
             lambda: compile(size={"width": 4096, "height": 1024}), "between")
check("and allowed with one",
      compile(size={"width": 4096, "height": 1024}, dype={"on": True}).width, 4096)


# ---- DyPE and SEGA --------------------------------------------------------------

check("a default blob has neither", (base.dype, base.sega), (None, None))

dyped = compile(dype={"on": True})
check("DyPE carries the pack's method and scale",
      (dyped.dype["method"], dyped.dype["scale"]),
      (ci.DEFAULT_DYPE_METHOD, ci.DEFAULT_DYPE_SCALE))
segad = compile(sega={"on": True, "method": "ntk", "alpha": 0.3})
check("SEGA carries its own", (segad.sega["method"], segad.sega["alpha"]), ("ntk", 0.3))
# They are alternatives, and this test used to claim the opposite — that they were
# independent and composed. The pack's README says "use as an alternative to
# DyPE", and its reference Krea 2 workflow ships the DyPE node bypassed with only
# SEGA live. Both rewrite the same encoding, so the second would overwrite the
# first's decision.
expect_error("the two together are refused, because they are two ways to do one thing",
             lambda: compile(dype={"on": True}, sega={"on": True}), "cannot both be on")

# The reason either is in this package at all: they are what lift the ceiling.
# With both off the model's own 2048 still holds, so the same blob resolves
# differently.
tall = {"aspect": "1:1", "short_edge": 4096}
check("with no position patch a 4096 request is clamped to the model's ceiling",
      compile(**tall).width, ci.MAX_SHORT_EDGE)
check("with DyPE it is not",
      compile(**tall, dype={"on": True}).width, ci.POSITION_MAX_SHORT_EDGE)
# The bug this pair of checks exists for: the cap moved for DyPE only, so
# choosing SEGA left the render clamped to 2048 with no sign of why.
check("and with SEGA it is not either",
      compile(**tall, sega={"on": True}).width, ci.POSITION_MAX_SHORT_EDGE)
# And the area cap moves with it, not just the per-axis one.
check("a 16:9 4K render keeps its aspect under the raised area cap",
      compile(aspect="16:9", short_edge=2160, dype={"on": True}).width > ci.MAX_SHORT_EDGE,
      True)

expect_error("an unknown DyPE method is refused, listing the real ones",
             lambda: compile(dype={"on": True, "method": "rope2"}), "vision_yarn")
expect_error("an unknown SEGA method is refused",
             lambda: compile(sega={"on": True, "method": "spectral"}), "unknown SEGA method")
expect_error("a DyPE scale past the node's ceiling is refused",
             lambda: compile(dype={"on": True, "scale": 20}), "DyPE scale")
expect_error("DyPE is refused on Ideogram, which the pack has no path for",
             lambda: compile(arch="ideogram4", dype={"on": True}), "no Ideogram 4 path")
expect_error("and so is SEGA",
             lambda: compile(arch="ideogram4", sega={"on": True}), "no Ideogram 4 path")


if FAILURES:
    print(f"{len(FAILURES)} failure(s):")
    for failure in FAILURES:
        print("  -", failure)
    sys.exit(1)
print("all compile_image tests passed")
