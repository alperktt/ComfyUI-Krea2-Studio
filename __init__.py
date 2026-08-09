from . import server_routes  # noqa: F401  (registers /minimax_creator/assets on import)
from . import refine_routes  # noqa: F401  (registers /minimax_creator/refine)
from .creator_node import comfy_entrypoint  # noqa: F401

WEB_DIRECTORY = "./js"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
