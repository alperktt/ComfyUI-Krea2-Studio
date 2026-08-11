"""Third-party Krea 2 packs, copied in verbatim.

Nothing here registers a ComfyUI node, and nothing here ever should. Each
subdirectory is somebody else's pack at a named commit, kept as close to
untouched as possible so an upstream fix can be carried across by reading its
`ORIGIN.md` and applying the same diff.

The packs are used as *libraries*: `nodes_vendor.py` imports their classes and
re-registers them under our own `K2S_` ids. That indirection is not tidiness —
four of these packs are commonly installed separately, and two copies of the
same node id means ComfyUI silently keeps one of them. Registering under our
own ids means a user can have both this package and the original installed and
neither shadows the other.

So: no `NODE_CLASS_MAPPINGS` in this file, no `WEB_DIRECTORY`, no import side
effects. `tests/test_vendor_isolation.py` enforces it.
"""
