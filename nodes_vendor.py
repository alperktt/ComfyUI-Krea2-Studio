"""The vendored packs, registered under our ids and nobody else's.

`vendor/` holds six Krea 2 packs copied in verbatim (see `vendor/__init__.py`
for why they are copies rather than dependencies). This module is the one place
their classes become ComfyUI nodes, and it does two things to them:

**It renames them.** Every vendored node is registered as `K2S_<something>`,
never under the id its author gave it. Four of these packs are commonly
installed separately, and ComfyUI keeps exactly one class per id — so without
the rename, installing this package would silently shadow whichever copy loaded
second, and a user debugging their own graph would have no way to tell which
one ran. Under our own ids both can be installed and neither notices the other.

**It hides them.** These are plumbing for the PreStage's compiled subgraph, not
nodes anybody should be dragging out of the search menu — the pack's own
`MiniMaxH3SaveImage` is hidden for the same reason. `DEPRECATED = True` is what
does it: `/object_info` reports it (`server.py`), the frontend drops the node
from search, and a graph that already names the id still runs. That last part
is the requirement — the emitters address these by string.

Both are done by *subclassing*, never by mutating the vendored class. The files
under `vendor/` stay byte-identical to their upstream so a patch from upstream
applies cleanly; the deltas live here.

**Why this writes into `nodes.NODE_CLASS_MAPPINGS` itself instead of exporting
one.** ComfyUI's loader takes a package one way or the other, not both:
`load_custom_node` reads `NODE_CLASS_MAPPINGS` and *returns* if it finds one,
and only reaches `comfy_entrypoint` in the `elif` (`nodes.py`). This package is
a V3 extension — its own nodes are `io.ComfyNode` subclasses handed over by
`MiniMaxCreatorExtension.get_node_list`. So exporting a mapping for the
vendored classes would not add them alongside; it would make the loader take
the V1 branch and never call the entrypoint at all, and every Creator, Timeline
and PreStage node would quietly vanish. Registering directly is the only way to
have both, and it is what the vendored packs' own `__init__` files do anyway —
one dict, written once, at import.

The table is filled one phase at a time, and stays empty until a pack is
actually wired into an emitter — a registered node nothing emits is a node that
can go stale without a test noticing.
"""

import importlib
import logging

# Every vendored node id carries this. `tests/test_vendor_isolation.py` refuses
# any registration that does not.
PREFIX = "K2S_"

CATEGORY = "Krea2/internal"

# (our id, module under `vendor`, the class's name in that module).
#
# Filled per phase:
#   1  svdquant     — the W4A4 loader and its LoRA loader
#   2  moodboards   — nothing: the catalog is read as data, no node needed
#   3  krea2edit    — the model patch and the grounded encoder
#   4  styletransfer
#   5  twostage
#   6  dype         — DyPE and SEGA
VENDORED: list[tuple[str, str, str]] = [
    ("K2S_SVDQuantW4A4Loader", "svdquant.svdquant_w4a4", "Krea2SVDQuantW4A4Loader"),
    ("K2S_SVDQuantLoraLoader", "svdquant.svdquant_lora", "Krea2SVDQuantLoraLoader"),
    ("K2S_Krea2EditModelPatch", "krea2edit", "Krea2EditModelPatch"),
    ("K2S_Krea2EditGroundedEncode", "krea2edit", "Krea2EditGroundedEncode"),
]


def _wrap(node_id, source):
    """`source` as a hidden node under our own id, without touching `source`.

    A subclass rather than a copy: `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION` and
    everything else the vendored author declared are inherited, so a pack that
    gains an input gains it here too the next time it is re-vendored. Only the
    three attributes this module exists to change are overridden.
    """
    return type(node_id, (source,), {
        "CATEGORY": CATEGORY,
        "DEPRECATED": True,
        "__module__": __name__,
        "__doc__": (source.__doc__ or "").strip() or None,
    })


def register():
    """Put the table into ComfyUI's registry. Returns the ids that made it.

    A pack that fails to import is logged and skipped rather than taking the
    whole package down with it: the vendored trees carry their authors' imports,
    and one of them wanting a module this install does not have should cost that
    one feature, not the Creator. `missing` is how an emitter finds out.
    """
    import nodes

    registered = []
    for node_id, module_name, class_name in VENDORED:
        if not node_id.startswith(PREFIX):
            raise ValueError(
                f"{node_id!r} does not start with {PREFIX!r}. Every vendored node "
                f"is registered under our own id — see this module's docstring."
            )
        # Never clobber. An id already taken means either this ran twice or
        # somebody else picked our prefix; both are worth saying out loud
        # rather than silently winning, which is the failure this whole module
        # exists to avoid.
        if node_id in nodes.NODE_CLASS_MAPPINGS:
            logging.warning("[krea2-studio] %s is already registered; leaving it alone", node_id)
            continue
        try:
            module = importlib.import_module(f".vendor.{module_name}", __package__)
            source = getattr(module, class_name)
        except Exception:  # noqa: BLE001 — one pack, not the package
            logging.exception(
                "[krea2-studio] could not load vendored %s.%s; the features that "
                "emit %s will refuse rather than run", module_name, class_name, node_id)
            continue
        nodes.NODE_CLASS_MAPPINGS[node_id] = _wrap(node_id, source)
        nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] = node_id
        registered.append(node_id)
    return registered


def missing(node_id):
    """Whether an id in `VENDORED` failed to load, for an emitter to check.

    Emitters address these by string into a `GraphBuilder`, which does not
    validate ids — an unregistered node fails much later, inside execution, with
    an error that names neither the pack nor the feature. Checking here lets the
    feature refuse with its own name on it.
    """
    import nodes

    return node_id not in nodes.NODE_CLASS_MAPPINGS
