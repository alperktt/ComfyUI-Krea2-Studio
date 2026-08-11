"""The vendored packs stay vendored: our ids only, and the V3 entrypoint intact.

Two rules, both of which fail silently rather than loudly if they are broken,
which is why they are worth a test:

**A vendored node must not keep its author's id.** Four of the six packs under
`vendor/` are commonly installed on their own. ComfyUI keeps one class per id
(`nodes.load_custom_node`), so a collision does not raise — one copy simply
wins, and which one depends on directory order. Every id we register carries
`nodes_vendor.PREFIX`.

**`__init__.py` must not define `NODE_CLASS_MAPPINGS`.** The loader reads it and
`return`s; `comfy_entrypoint` is in the `elif` and never runs. Adding a mapping
would not add nodes alongside the Creator's — it would delete every one of
them, and the only symptom is an empty menu.

Static analysis rather than an import, so this runs with no ComfyUI, no torch
and no GPU:

    python tests/test_vendor_isolation.py
"""

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor")

FAILURES = []


def fail(message):
    FAILURES.append(message)


def parse(path):
    with open(path, encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def assigned_names(tree):
    """Every name assigned at module level, as `{name: node}`."""
    found = {}
    for statement in tree.body:
        targets = []
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        for target in targets:
            if isinstance(target, ast.Name):
                found[target.id] = statement
    return found


def dict_keys(statement):
    """The literal string keys of `X = {...}`, or None if it is not a literal."""
    value = getattr(statement, "value", None)
    if not isinstance(value, ast.Dict):
        return None
    keys = []
    for key in value.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.append(key.value)
    return keys


# --- 1. the package's own __init__ leaves the V3 branch reachable -------------

package_init = parse(os.path.join(ROOT, "__init__.py"))
names = assigned_names(package_init)
if "NODE_CLASS_MAPPINGS" in names:
    fail("__init__.py defines NODE_CLASS_MAPPINGS — ComfyUI would take the V1 "
         "branch and never call comfy_entrypoint, losing every Creator node")
if "comfy_entrypoint" not in {alias.asname or alias.name
                              for statement in package_init.body
                              if isinstance(statement, ast.ImportFrom)
                              for alias in statement.names}:
    fail("__init__.py no longer imports comfy_entrypoint")


# --- 2. vendor/__init__.py registers nothing ---------------------------------

vendor_init = parse(os.path.join(VENDOR, "__init__.py"))
for forbidden in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"):
    if forbidden in assigned_names(vendor_init):
        fail(f"vendor/__init__.py defines {forbidden} — the vendored packs are "
             f"libraries, and nodes_vendor.py is the only place they become nodes")


# --- 3. every id we register carries the prefix ------------------------------

nodes_vendor = parse(os.path.join(ROOT, "nodes_vendor.py"))
prefix_statement = assigned_names(nodes_vendor).get("PREFIX")
PREFIX = prefix_statement.value.value if prefix_statement else None
if not PREFIX:
    fail("nodes_vendor.py has no literal PREFIX")

table = assigned_names(nodes_vendor).get("VENDORED")
entries = []
if table is None:
    fail("nodes_vendor.py has no VENDORED table")
elif isinstance(table.value, ast.List):
    for element in table.value.elts:
        if not isinstance(element, ast.Tuple) or len(element.elts) != 3:
            fail(f"VENDORED entry is not a 3-tuple: {ast.dump(element)[:60]}")
            continue
        parts = [e.value if isinstance(e, ast.Constant) else None for e in element.elts]
        if any(part is None for part in parts):
            fail("VENDORED entries must be literal strings")
            continue
        entries.append(tuple(parts))

for node_id, module_name, class_name in entries:
    if PREFIX and not node_id.startswith(PREFIX):
        fail(f"{node_id!r} does not start with {PREFIX!r}")
    # `module_name` is dotted, the way `importlib` takes it: "svdquant.foo" is
    # vendor/svdquant/foo.py.
    path = os.path.join(VENDOR, *module_name.split("."))
    if not (os.path.isdir(path) or os.path.isfile(path + ".py")):
        fail(f"{node_id!r} names vendor/{module_name.replace('.', '/')}, which does not exist")


# --- 4. no vendored pack's own ids end up registered -------------------------

ours = {node_id for node_id, _, _ in entries}
if not os.path.isdir(VENDOR):
    fail("vendor/ does not exist")
else:
    for entry in sorted(os.listdir(VENDOR)):
        init = os.path.join(VENDOR, entry, "__init__.py")
        if not os.path.isfile(init):
            continue
        try:
            theirs = assigned_names(parse(init)).get("NODE_CLASS_MAPPINGS")
        except SyntaxError as exc:
            fail(f"vendor/{entry}/__init__.py does not parse: {exc}")
            continue
        for node_id in dict_keys(theirs) or []:
            if node_id in ours:
                fail(f"we register {node_id!r}, which is also vendor/{entry}'s own "
                     f"id — rename ours so both packs can be installed")


if FAILURES:
    print(f"{len(FAILURES)} problem(s):")
    for failure in FAILURES:
        print("  -", failure)
    sys.exit(1)
print(f"vendor isolation holds across {len(entries)} vendored node(s)")
