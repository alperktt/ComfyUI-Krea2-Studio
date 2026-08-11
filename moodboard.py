"""The Krea moodboard catalog, read as data.

3,549 boards scraped from Krea's own moodboard gallery, each carrying a title,
keywords, a taste profile and prompt guidance written for this model family.
Choosing one is choosing a look without having to describe it — which is the
half of prompting that a text box is worst at.

**Nothing here becomes a node.** The vendored pack's seven nodes all return
`STRING` and only `STRING`: there is no model to patch and no conditioning to
build, so a node in the graph would exist to hand text to the node beside it.
`vendor/moodboards/moodboard_catalog.py` is pure — json, re, pathlib, and not
one import of torch or ComfyUI — so the guidance is merged into the prompt in
`compile_image.py`, next to where LoRA trigger words are already merged, and the
graph never learns that moodboards exist.

This module is the seam between the two: it owns the disk (the catalog is 9 MB
of JSON and is read once) and hands `compile_image` a plain function, the same
way `media.image_size` is handed to it for init images. That is what keeps
`compile_image` importable with no ComfyUI, which `test_prestage_mirror.py`
depends on to compare it against `state.js` in node.

**The negative fragment is real conditioning, not decoration.** A board that
says what it is not — "no bokeh, no lens flare" — has nowhere to go in a
pipeline whose negative is `ConditioningZeroOut`, so when a board supplies one
and the user leaves it on, the render encodes it as the actual negative. At
Turbo's cfg 1.0 the sampler skips the unconditional branch entirely and it costs
nothing and does nothing; on RAW at cfg 3.5 it does what a negative prompt does.
Said in the UI rather than left to be discovered.
"""

import os
import threading

STRENGTHS = ("concise", "normal", "strong")
DEFAULT_STRENGTH = "normal"

# Which catalog file a board id is looked up in. "krea" is the scraped gallery;
# "andrometa" is the pack author's smaller curated set.
COLLECTIONS = ("krea", "andrometa")
DEFAULT_COLLECTION = "krea"

_FILES = {
    "krea": "krea_moodboards_slim.json",
    "andrometa": "andrometa_moodboards.json",
}

_lock = threading.Lock()
_cache = {}


def _catalog_path(collection):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "vendor", "moodboards", "data", _FILES[collection])


def catalog(collection=DEFAULT_COLLECTION):
    """The parsed catalog, read once per collection.

    9 MB of JSON, so this is cached behind a lock rather than re-read: the
    listing route and the compile path both reach for it, and the route runs on
    the event loop.
    """
    if collection not in _FILES:
        raise ValueError(f"unknown moodboard collection {collection!r}")
    with _lock:
        if collection not in _cache:
            from .vendor.moodboards import moodboard_catalog as mc

            _cache[collection] = mc.load_catalog(_catalog_path(collection))
        return _cache[collection]


def available():
    """Whether the catalog is on disk at all, for the UI to key off.

    The JSON is the bulk of this package's clone; someone who trimmed it should
    get a pill that says so rather than a render that fails at compile time.
    """
    return all(os.path.isfile(_catalog_path(name)) for name in _FILES)


def style(board, strength=DEFAULT_STRENGTH, collection=DEFAULT_COLLECTION):
    """`{positive, negative, title}` for one board reference.

    `board` is whatever the picker stored — a uuid, a slug, a title, a Krea URL,
    or a search phrase; `find_board` resolves all five. Raises `LookupError`
    when nothing matches, which the compile turns into a `CompileError` naming
    the board rather than rendering without the look that was asked for.
    """
    from .vendor.moodboards import moodboard_catalog as mc

    if strength not in STRENGTHS:
        raise ValueError(f"unknown moodboard strength {strength!r}")
    try:
        found = mc.find_board(catalog(collection), str(board))
    except Exception as exc:  # noqa: BLE001 — the pack raises bare ValueError/KeyError
        raise LookupError(str(exc) or f"no moodboard matches {board!r}") from exc
    resolved = mc.style_from_board(found, strength=strength)
    return {"positive": resolved["positive"],
            "negative": resolved["negative"],
            "title": resolved["title"]}


def lookup():
    """`style` as the injected callable `compile_prestage` takes.

    Injected for the same reason `media.image_size` is: `compile_image.py` must
    keep importing nothing, so the disk stays on this side of the call.
    """
    return style


def search(query="", page=1, page_size=30, collection=DEFAULT_COLLECTION):
    """One page of the picker: `{items, total, page, page_size}`.

    `catalog_listing` rather than `search_boards` for both the searching and the
    browsing case, because it is the one that already does both — an empty query
    is the whole catalog in title order, and a non-empty one is the scored
    matches. Its summaries carry the thumbnail URL the picker draws.
    """
    from .vendor.moodboards import moodboard_catalog as mc

    listing = mc.catalog_listing(catalog(collection), query=str(query or ""),
                                 page=int(page), page_size=int(page_size))
    import json

    payload = json.loads(listing["catalog_json"])
    return {"items": payload["items"], "total": payload["total"],
            "page": payload["page"], "page_size": payload["page_size"]}
