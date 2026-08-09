"""Optional sampling accelerators, wired in rather than reimplemented.

Two community packs make H3 substantially faster and neither is ours:

- **FirstBlockCache** (`ComfyUI-MiniMaxH3-FirstBlockCache`) skips the rest of the
  DiT when the first block's residual barely moved between steps.
- **Spectrum** (`ComfyUI-Spectrum-MiniMax-H3`) forecasts features across steps
  instead of evaluating every one of them.

Both are MODEL patchers: model in, patched model out, everything else unchanged.
That is the whole reason this module can be twenty lines of wiring — there is no
sampling logic here and there must never be any. Copying their maths in would
mean owning their bugs and freezing their tuning at whatever it was the day it
was copied, so this only ever *calls* them, and says so plainly when they are
not installed.

**Why the parameters are read rather than written.** Every required input of a
node has to be supplied explicitly when it is built into a graph, and both packs
have a dozen. Hardcoding that many defaults here means they go stale silently the
first time either pack retunes one — the node would keep running, just no longer
at the settings its author recommends. So `node_defaults` reads them back off the
installed class's own `INPUT_TYPES`, and this module only names the handful it
actually overrides. A pack that gains a knob gets its own default for it.

**Order is `block cache -> spectrum -> sampler`**, which is both packs' own
advice: FirstBlockCache refuses to sit downstream of another DiT block
replacement, and Spectrum documents itself as the last patch before the guider.
They compose — the caches are wrappers and block patches respectively, and
neither trips the other's conflict check.

Nothing here is Timeline-specific. `graph_apply` is for the nodes that build a
subgraph and `direct_apply` for the ones holding a real MODEL, so the Creator
node can take the same settings later without this module changing.
"""

from dataclasses import dataclass

BLOCK_CACHE_NODE = "ApplyMiniMaxH3FirstBlockCache"
SPECTRUM_NODE = "SpectrumApplyMiniMaxH3"

# Where to get each pack, named in the error rather than in a README nobody is
# reading at the moment the node fails.
SOURCES = {
    BLOCK_CACHE_NODE: "https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache",
    SPECTRUM_NODE: "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3",
}

# What the node's `block_cache` widget offers. The values are matched against the
# *installed* pack's mode list by prefix, because its labels carry the threshold
# in them ("H3 Fast — 0.10 / max 2") and would break this the first time one is
# retuned. "off" is not a mode: it means the node is never built.
BLOCK_CACHE_MODES = ["off", "safe", "fast", "aggressive"]


@dataclass(frozen=True)
class Settings:
    """What the user asked for. Both accelerators off is the default everywhere."""

    block_cache: str = "off"
    spectrum: bool = False
    spectrum_blend: float = 0.5

    @property
    def any(self):
        return self.block_cache != "off" or self.spectrum


def _node_class(node_id):
    """The installed class for `node_id`, or None. Looked up per call.

    Not cached and not imported at module load: a pack installed while ComfyUI is
    running should not need this one to be reloaded too, and importing either of
    them here would turn an optional accelerator into a hard dependency.
    """
    import nodes

    return nodes.NODE_CLASS_MAPPINGS.get(node_id)


def _require(node_id):
    node = _node_class(node_id)
    if node is None:
        raise ValueError(
            f"This needs the '{node_id}' node, which is not installed. "
            f"Get it from {SOURCES[node_id]}, restart ComfyUI, or switch the "
            f"accelerator off."
        )
    return node


def node_defaults(node, skip=("model",)):
    """`{input: default}` for every required input the class declares but `skip`.

    Required inputs have to be passed explicitly into a built graph, and reading
    them back off the class is what keeps this module from carrying a stale copy
    of somebody else's tuning. An input with no declared default is left out
    rather than guessed at — ComfyUI will say which one is missing, which is a
    better error than a number this module invented.

    Public because `models.py` wires up KJNodes' preview override on exactly the
    same terms, and two copies of this would be two copies of the argument for it.
    """
    spec = node.INPUT_TYPES().get("required", {})
    out = {}
    for name, declared in spec.items():
        if name in skip:
            continue
        if isinstance(declared, (tuple, list)) and len(declared) > 1 and isinstance(declared[1], dict):
            if "default" in declared[1]:
                out[name] = declared[1]["default"]
    return out


def _block_cache_kwargs(node, mode):
    """The pack's own arguments for one of our three preset names."""
    kwargs = node_defaults(node)
    options = node.INPUT_TYPES()["required"]["mode"][0]
    wanted = f"h3 {mode}"
    match = next((o for o in options if str(o).lower().startswith(wanted)), None)
    if match is None:
        raise ValueError(
            f"'{node.__name__}' has no '{mode}' preset — it offers {list(options)}. "
            f"The pack has renamed its modes; use its own node directly."
        )
    kwargs["mode"] = match
    return kwargs


def _spectrum_kwargs(node, blend):
    kwargs = node_defaults(node)
    kwargs["enabled"] = True
    kwargs["blend_weight"] = float(blend)
    return kwargs


def plan(settings):
    """`[(node_id, kwargs), ...]` in the order they must be applied.

    Shared by both entry points so the graph path and the direct path cannot
    drift apart on ordering or arguments — the difference between them is only
    how a node gets run, never which nodes or with what.
    """
    steps = []
    if settings.block_cache != "off":
        node = _require(BLOCK_CACHE_NODE)
        steps.append((BLOCK_CACHE_NODE, _block_cache_kwargs(node, settings.block_cache)))
    if settings.spectrum:
        node = _require(SPECTRUM_NODE)
        steps.append((SPECTRUM_NODE, _spectrum_kwargs(node, settings.spectrum_blend)))
    return steps


def graph_apply(graph, model, settings):
    """Patch a MODEL *link* inside a `GraphBuilder` subgraph. Returns the new link.

    For the nodes that return an expanded graph rather than tensors. With both
    accelerators off this returns `model` untouched and adds nothing to the
    graph — an unused node is still a node ComfyUI has to cache and schedule.
    """
    for node_id, kwargs in plan(settings):
        model = graph.node(node_id, model=model, **kwargs).out(0)
    return model


def direct_apply(model, settings):
    """Patch a real MODEL object. Returns the patched model.

    The Creator node's half of the same contract: it holds a loaded model rather
    than a link, so it calls the packs the way ComfyUI would. Unused today and
    kept beside `graph_apply` deliberately — the two are one decision, and
    splitting them across a later commit is how they stop agreeing.
    """
    for node_id, kwargs in plan(settings):
        node = _require(node_id)
        model = getattr(node(), node.FUNCTION)(model=model, **kwargs)[0]
    return model
