from . import server_routes  # noqa: F401  (registers /minimax_creator/assets on import)
from . import refine_routes  # noqa: F401  (registers /minimax_creator/refine)
from . import nodes_vendor
from .creator_node import comfy_entrypoint  # noqa: F401

# The vendored Krea 2 packs, under our own ids. Deliberately *not* exported as
# `NODE_CLASS_MAPPINGS`: ComfyUI's loader takes that branch and returns, and
# `comfy_entrypoint` below would never be called — see `nodes_vendor`.
nodes_vendor.register()

WEB_DIRECTORY = "./js"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
