"""Listing routes for the asset picker and the LoRA manager.

The picker browses ComfyUI/input, so the only thing the frontend cannot work out
for itself is what is in there. Image thumbnails reuse core's `/api/view`, and
uploads reuse core's `/api/upload/image` (which despite the name is what
LoadVideo and LoadAudio post to as well), so neither needs a route here.

Video does need routes of its own: `/view` serves the whole clip, which is the
wrong thing to hand a 140 px grid cell or a waveform canvas. See preview.py.

LoRAs need both routes of their own. `/view` only serves input, output and temp,
so it cannot reach models/loras, and the CiviMeta sidecar next to each file has
to be read server-side.

The settings pair at the bottom is the one thing here that is not a listing. It
has to be a route rather than the frontend's userdata API for the reason
`settings.py` opens with: the save node reads the same file while a prompt runs,
and only the server can hand both ends the same path.
"""

import asyncio
import json
import os
import struct

from aiohttp import web

import folder_paths
from server import PromptServer

from . import models, preview, settings

# A large input folder should not turn the picker into a stall. Newest first,
# so the cap drops the least interesting files.
MAX_ASSETS = 4000

# How many LoRAs get the full sidecar treatment in one listing. A collection of
# a few thousand is normal, and reading a JSON file plus listing two directories
# for every one of them is seconds of work — so only the newest MAX_LORAS are
# described, and the manager says so and offers the folder picker instead.
MAX_LORAS = 600

# CiviMeta writes `{model}.civitai/` beside every file it has identified:
# meta.json (the Civitai model version), images.json, media/NNN.ext (the
# creator's showcase, downloaded) and thumbnails/NNN.webp (generated, images
# only — a video showcase has no thumbnail, so media/ is the fallback).
SIDECAR_SUFFIX = ".civitai"
PREVIEW_DIRS = ("thumbnails", "media")
PREVIEW_KINDS = {
    ".webp": "image", ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image",
    ".mp4": "video", ".webm": "video",
}
# CiviMeta leaves a stub behind when a media download fails part way.
MIN_PREVIEW_BYTES = 100


def _classify(filename):
    for kind in ("image", "video", "audio"):
        if folder_paths.filter_files_content_types([filename], [kind]):
            return kind
    return None


def _scan(root, annotation=""):
    """Walk one media folder. `annotation` is ComfyUI's ` [output]` suffix.

    Carried inside `path` rather than as a separate field because the path is
    the one thing that survives into creator_data: every consumer downstream —
    the thumb and probe routes here, `media.resolve` at execute time — already
    goes through `get_annotated_filepath`, so an annotated path is a file the
    whole pipeline can reach with no second load path.
    """
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            kind = _classify(filename)
            if kind is None:
                continue
            path = os.path.join(directory, filename)
            # A symlink pointing outside the root is a file this pack cannot
            # open: `get_annotated_filepath` resolves the link and then refuses
            # it for leaving the folder, so listing it would offer a thumbnail
            # that fails at execute time with "not in the input folder any
            # more" — about a file that is plainly sitting right there.
            #
            # Not worked around: the containment check is core's and is the
            # thing standing between a crafted filename and the rest of the
            # disk. Symlinking media into input/ does not work; the flag that
            # does is `--input-directory`, and the README says so.
            if os.path.islink(path) and not folder_paths.is_within_directory(root, path):
                continue
            subfolder = os.path.relpath(directory, root)
            subfolder = "" if subfolder == "." else subfolder.replace(os.sep, "/")
            try:
                mtime = os.path.getmtime(path)
                size = os.path.getsize(path)
            except OSError:
                continue
            relative = f"{subfolder}/{filename}" if subfolder else filename
            yield {
                "path": relative + annotation,
                "name": filename,
                "subfolder": subfolder,
                "kind": kind,
                "size": size,
                "mtime": mtime,
            }


def _input_path(request):
    """The absolute path behind a `?filename=` query, or None if it is not ours."""
    filename = request.query.get("filename", "")
    if not filename or not folder_paths.exists_annotated_filepath(filename):
        return None
    return folder_paths.get_annotated_filepath(filename)


def _read_header(path):
    import av  # ComfyUI's own decoder stack; imported here so the listing route never needs it.

    with av.open(path) as container:
        return {
            "has_audio": bool(container.streams.audio),
            "duration": float(container.duration / av.time_base) if container.duration else None,
        }


@PromptServer.instance.routes.get("/minimax_creator/probe")
async def probe_asset(request):
    """Does this clip carry a soundtrack?

    A reference video is attached with its sound on by default, which is only the
    right default when there is sound to bind — otherwise the generation would
    fail at queue time on a file the user never claimed was noisy. No browser
    reports the presence of an audio track portably, so the answer comes from
    here. It reads the container header, not the media.

    `has_audio: null` means the question could not be answered; the caller keeps
    its own default rather than guessing silence.
    """
    path = _input_path(request)
    if path is None:
        return web.json_response({"has_audio": None, "error": "not in the input folder"}, status=404)
    try:
        # Opening a container reads and seeks; on a network share that is long
        # enough to be felt, and anything blocking here blocks the whole server —
        # the prompt queue and the websocket included.
        loop = asyncio.get_running_loop()
        return web.json_response(await loop.run_in_executor(None, _read_header, path))
    except Exception as exc:  # noqa: BLE001 — an unreadable file is the caller's problem, later
        return web.json_response({"has_audio": None, "error": str(exc)})


@PromptServer.instance.routes.get("/minimax_creator/thumb")
async def asset_thumb(request):
    """A JPEG still of one clip, for a picker cell.

    404 rather than a placeholder: the cell falls back to an icon, and inventing
    an image here would make an undecodable file look like a fine one.
    """
    path = _input_path(request)
    if path is None:
        return web.Response(status=404)
    thumb = await preview.thumbnail(path)
    if thumb is None:
        return web.Response(status=404)
    return web.FileResponse(thumb, headers={
        "Content-Type": "image/jpeg",
        # The caller stamps the source mtime into the URL, so a given URL really
        # does name one immutable frame — replacing the file changes the URL.
        "Cache-Control": "public, max-age=31536000, immutable",
    })


@PromptServer.instance.routes.get("/minimax_creator/peaks")
async def asset_peaks(request):
    """Waveform peaks for the segment editor's timeline, normalised to 0..1.

    `peaks: null` means there is nothing to draw — no audio track, or a track
    that decoded to silence — and the timeline stays plain, which is exactly what
    it does when this is unavailable altogether.
    """
    path = _input_path(request)
    if path is None:
        return web.json_response({"peaks": None}, status=404)
    result = await preview.waveform(path)
    if result is None:
        return web.json_response({"peaks": None})
    # Not cached by the browser: the answer is keyed by mtime server-side, and
    # this is one small request per editor opening rather than one per cell.
    return web.json_response(result, headers={"Cache-Control": "no-cache"})


def _first_preview(sidecar):
    """The showcase image or clip CiviMeta cached, as (relative path, kind).

    Thumbnails first — they are generated WebP and a fraction of the bytes — then
    the raw media, which is the only thing a video-preview LoRA has. Both are
    numbered (`001.webp`), so alphabetical order is the creator's order.
    """
    for sub in PREVIEW_DIRS:
        directory = os.path.join(sidecar, sub)
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for name in names:
            kind = PREVIEW_KINDS.get(os.path.splitext(name)[1].lower())
            if kind is None:
                continue
            path = os.path.join(directory, name)
            try:
                if os.path.getsize(path) < MIN_PREVIEW_BYTES:
                    continue
            except OSError:
                continue
            return f"{sub}/{name}", kind
    return None, None


def _read_meta(sidecar):
    try:
        with open(os.path.join(sidecar, "meta.json"), encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


# name -> (sidecar mtime, description). Reading a JSON file and listing two
# directories per LoRA is cheap, but not cheap enough to redo on every keystroke
# in the manager's search box.
_LORA_CACHE = {}


def _describe_lora(name, path):
    sidecar = path + SIDECAR_SUFFIX
    try:
        stamp = os.path.getmtime(sidecar)
    except OSError:
        stamp = None

    cached = _LORA_CACHE.get(name)
    if cached and cached[0] == stamp:
        return cached[1]

    try:
        size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
    except OSError:
        size, mtime = 0, 0

    row = {
        "name": name,
        "base": os.path.splitext(os.path.basename(name))[0],
        "folder": os.path.dirname(name),
        "size": size,
        "mtime": mtime,
        "preview": None,
    }

    if stamp is not None:
        _, kind = _first_preview(sidecar)
        row["preview"] = kind
        meta = _read_meta(sidecar)
        if meta:
            stats = meta.get("stats") or {}
            row.update({
                "title": meta.get("name"),
                "version": meta.get("versionName"),
                "base_model": meta.get("baseModel"),
                "type": meta.get("type"),
                "tags": meta.get("tags") or [],
                "trained_words": meta.get("trainedWords") or [],
                "nsfw": bool(meta.get("nsfw")),
                "model_id": meta.get("modelId"),
                "version_id": meta.get("versionId"),
                "downloads": stats.get("downloads"),
            })

    _LORA_CACHE[name] = (stamp, row)
    return row


def _lora_names():
    """Every registered LoRA, as a forward-slash relative name.

    `get_filename_list` yields native separators; the manager stores these names
    in creator_data and posts them back, so they are normalised once here and
    stay one shape everywhere. `get_full_path` accepts either on both platforms.
    """
    return [name.replace(os.sep, "/") for name in folder_paths.get_filename_list("loras")]


def _folder_counts(names):
    """Every folder that holds LoRAs, with how many are under it.

    Counts are inclusive of nested folders — picking `Wan` and finding nothing
    because the files sit in `Wan/character` would make the picker useless. The
    root entry is the empty string, which is how the manager asks for all of them.
    """
    counts = {"": len(names)}
    for name in names:
        parts = name.split("/")[:-1]
        for depth in range(len(parts)):
            counts["/".join(parts[:depth + 1])] = counts.get("/".join(parts[:depth + 1]), 0) + 1
    return [{"path": path, "count": counts[path]} for path in sorted(counts)]


def _in_folder(name, folder):
    return not folder or name.startswith(folder + "/")


def _collect_loras(folder):
    """The rows for one folder, newest first, capped at MAX_LORAS.

    Two passes on purpose. Stat-ing every candidate is cheap and is the only way
    to know which ones are the newest; reading sidecars is not, so it happens
    only for the ones that survive the cap.
    """
    names = _lora_names()
    found = []
    for name in names:
        if not _in_folder(name, folder):
            continue
        path = folder_paths.get_full_path("loras", name)
        if path is None:
            continue
        try:
            found.append((os.path.getmtime(path), name, path))
        except OSError:
            continue
    found.sort(reverse=True)
    rows = [_describe_lora(name, path) for _, name, path in found[:MAX_LORAS]]
    return {
        "loras": rows,
        "folders": _folder_counts(names),
        "folder": folder,
        "matched": len(found),
        "truncated": len(found) > MAX_LORAS,
    }


@PromptServer.instance.routes.get("/minimax_creator/loras")
async def list_loras(request):
    # Thousands of files means thousands of stat calls and hundreds of sidecar
    # reads. On the event loop that is the prompt queue and the websocket held
    # up for as long as it takes.
    folder = request.query.get("folder", "").strip("/")
    loop = asyncio.get_running_loop()
    return web.json_response(await loop.run_in_executor(None, _collect_loras, folder))


@PromptServer.instance.routes.get("/minimax_creator/lora_preview")
async def lora_preview(request):
    """Serve the CiviMeta showcase image or clip for one LoRA.

    Core's `/view` is limited to input/output/temp, so models/loras is out of its
    reach. `get_full_path` normalises the name against the registered lora
    folders, which is also what keeps a crafted name inside them.
    """
    path = folder_paths.get_full_path("loras", request.query.get("name", ""))
    if path is None:
        return web.Response(status=404)
    sidecar = path + SIDECAR_SUFFIX
    relative, _ = _first_preview(sidecar)
    if relative is None:
        return web.Response(status=404)
    return web.FileResponse(os.path.join(sidecar, *relative.split("/")))


def _list_showcase(sidecar):
    """Every showcase file CiviMeta cached, in the creator's order.

    Only media/ is walked — thumbnails are looked up per entry, because a video
    showcase never has one and the detail sheet needs to know which is which.
    """
    entries = []
    directory = os.path.join(sidecar, "media")
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return entries
    for name in names:
        kind = PREVIEW_KINDS.get(os.path.splitext(name)[1].lower())
        if kind is None:
            continue
        try:
            if os.path.getsize(os.path.join(directory, name)) < MIN_PREVIEW_BYTES:
                continue
        except OSError:
            continue
        stem = os.path.splitext(name)[0]
        thumb = os.path.join(sidecar, "thumbnails", stem + ".webp")
        try:
            has_thumb = os.path.getsize(thumb) >= MIN_PREVIEW_BYTES
        except OSError:
            has_thumb = False
        entries.append({"file": name, "kind": kind, "stem": stem, "thumb": has_thumb})
    return entries


# The per-image generation settings worth showing on a detail sheet. images.json
# carries more (the whole Comfy graph, resource lists); the recipe is what a
# person can act on.
RECIPE_KEYS = ("prompt", "negativePrompt", "seed", "steps", "cfgScale", "sampler", "scheduler")


def _showcase_rows(sidecar):
    """The showcase as the detail sheet wants it: one row per cached file, each
    carrying the generation recipe images.json holds for it.

    Media files are numbered from the images.json order (`001` is items[0]), so
    a numeric stem is the authoritative link back to its metadata — positional
    matching would drift the moment one download in the middle failed.
    """
    try:
        with open(os.path.join(sidecar, "images.json"), encoding="utf-8") as handle:
            items = json.load(handle).get("items") or []
    except (OSError, ValueError, AttributeError):
        items = []

    rows = []
    for index, entry in enumerate(_list_showcase(sidecar)):
        row = {"index": index, "kind": entry["kind"], "thumb": entry["thumb"], "meta": None}
        try:
            item = items[int(entry["stem"]) - 1]
        except (ValueError, IndexError):
            item = None
        if isinstance(item, dict):
            meta = item.get("meta")
            if isinstance(meta, dict):
                recipe = {key: meta.get(key) for key in RECIPE_KEYS if meta.get(key) is not None}
                row["meta"] = recipe or None
            row["nsfw"] = bool(item.get("nsfw"))
        rows.append(row)
    return rows


# A safetensors header is one JSON blob at the front of the file. Anything
# claiming to be bigger than this is not a header, it is a corrupt length field
# about to become a memory allocation.
MAX_ST_HEADER = 64 * 1024 * 1024


def _read_safetensors_header(path):
    """What the file itself can say: training metadata, tensor census, rank.

    `metadata` is the trainer's `__metadata__` block verbatim (kohya's ss_*
    keys, ai-toolkit's json-in-string values — the frontend knows the dialects).
    `ranks` comes from the lora_A/lora_down shapes, which is the ground truth
    the metadata's ss_network_dim merely repeats.
    """
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                return {"error": "not a safetensors file"}
            (length,) = struct.unpack("<Q", prefix)
            if not 0 < length <= MAX_ST_HEADER:
                return {"error": "no readable header"}
            header = json.loads(handle.read(length))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return {"error": str(exc)}
    if not isinstance(header, dict):
        return {"error": "no readable header"}

    metadata = header.pop("__metadata__", None)
    dtypes = {}
    ranks = set()
    for key, tensor in header.items():
        if not isinstance(tensor, dict):
            continue
        dtype = tensor.get("dtype")
        if dtype:
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
        if key.endswith(("lora_A.weight", "lora_down.weight")):
            shape = tensor.get("shape") or []
            if shape:
                ranks.add(shape[0])
    return {
        "metadata": metadata if isinstance(metadata, dict) else {},
        "tensors": len(header),
        "dtypes": dtypes,
        "ranks": sorted(ranks),
    }


# What the detail sheet shows from meta.json, verbatim. The listing's
# _describe_lora flattens a chosen few of these; the sheet gets the lot,
# including the two description HTML blobs — which the frontend sanitizes
# before they touch the DOM.
DETAIL_META_KEYS = (
    "hash", "modelId", "versionId", "name", "versionName", "type", "baseModel",
    "creator", "description", "versionDescription", "tags", "trainedWords",
    "stats", "license", "nsfw", "files", "versions", "mediaCount", "fetchedAt",
)


def _lora_detail(name):
    path = folder_paths.get_full_path("loras", name)
    if path is None:
        return None
    try:
        size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
    except OSError:
        size, mtime = 0, 0
    detail = {
        "name": name,
        "size": size,
        "mtime": mtime,
        "header": _read_safetensors_header(path),
        "civitai": None,
        "showcase": [],
    }
    sidecar = path + SIDECAR_SUFFIX
    meta = _read_meta(sidecar)
    if meta:
        detail["civitai"] = {key: meta.get(key) for key in DETAIL_META_KEYS}
        detail["showcase"] = _showcase_rows(sidecar)
    return detail


@PromptServer.instance.routes.get("/minimax_creator/lora_detail")
async def lora_detail(request):
    """Everything one LoRA's detail sheet needs, in one request: the full
    CiviMeta sidecar (when there is one), the showcase with its generation
    recipes, and what the safetensors header itself says either way.
    """
    name = request.query.get("name", "")
    # Reading a header on a network share, plus a sidecar's JSON files, is I/O
    # the event loop must not sit on.
    loop = asyncio.get_running_loop()
    detail = await loop.run_in_executor(None, _lora_detail, name)
    if detail is None:
        return web.json_response({"error": "no such LoRA"}, status=404)
    return web.json_response(detail)


@PromptServer.instance.routes.get("/minimax_creator/lora_showcase")
async def lora_showcase(request):
    """Serve one showcase file by its index in the detail's showcase list.

    `?thumb=1` asks for the generated WebP thumbnail instead — the filmstrip's
    request — and falls back to the media file when there is none, which is the
    normal state of a video showcase.
    """
    path = folder_paths.get_full_path("loras", request.query.get("name", ""))
    if path is None:
        return web.Response(status=404)
    sidecar = path + SIDECAR_SUFFIX
    try:
        index = int(request.query.get("item", "0"))
    except ValueError:
        return web.Response(status=404)
    entries = _list_showcase(sidecar)
    if not 0 <= index < len(entries):
        return web.Response(status=404)
    entry = entries[index]
    if request.query.get("thumb") == "1" and entry["thumb"]:
        return web.FileResponse(os.path.join(sidecar, "thumbnails", entry["stem"] + ".webp"))
    return web.FileResponse(os.path.join(sidecar, "media", entry["file"]))


@PromptServer.instance.routes.get("/minimax_creator/models")
async def list_models(request):
    """What the weights control can offer: one file list per field.

    The node has no model sockets any more, so this is the only way the UI knows
    what is installed. It also reports whether KJNodes' preview override is
    present, because the taeh3 preview is the one control here that depends on
    somebody else's pack being loaded.
    """
    # Four `get_filename_list` calls, each of which may walk a model directory
    # that has never been scanned. On the event loop that is the prompt queue and
    # the websocket held up behind it.
    loop = asyncio.get_running_loop()
    return web.json_response(await loop.run_in_executor(None, models.available))


@PromptServer.instance.routes.get("/minimax_creator/assets")
async def list_assets(request):
    """The picker's grid: `?root=input` (the default) or `?root=output`.

    The output listing is the gallery — finished renders, browsed with the same
    machinery as the input folder. Its paths come back annotated (` [output]`),
    which is what lets one of them be attached as a reference: see `_scan`.
    """
    if request.query.get("root") == "output":
        root, annotation = folder_paths.get_output_directory(), " [output]"
    else:
        root, annotation = folder_paths.get_input_directory(), ""
    if not os.path.isdir(root):
        return web.json_response({"assets": [], "truncated": False})

    # A walk with two stat calls per file is nothing on a local disk and minutes
    # on a network share, and the event loop is also the prompt queue.
    loop = asyncio.get_running_loop()
    assets = await loop.run_in_executor(
        None, lambda: sorted(_scan(root, annotation), key=lambda a: a["mtime"], reverse=True))
    truncated = len(assets) > MAX_ASSETS
    return web.json_response({"assets": assets[:MAX_ASSETS], "truncated": truncated})


def _clean_subfolder(raw):
    """A user-typed shelf name as a safe root-relative directory, or None.

    Rejects rather than sanitizes: a name that needs rewriting to be safe is a
    name the user should see refused, not silently changed.
    """
    raw = str(raw).strip().strip("/")
    if not raw:
        return ""
    parts = raw.replace("\\", "/").split("/")
    if any(not p or p.startswith(".") for p in parts):
        return None
    return "/".join(parts)


def _rooted(filename):
    """A picker path -> `(root, relative, annotation)`, or None if it is not ours.

    The ` [output]` suffix a gallery path carries is what says which folder it
    came out of, so the two organize routes take their root from the file rather
    than from a separate parameter that could disagree with it. An unannotated
    path is an input path, which is the shape every caller used before the
    gallery could be organized at all.
    """
    name, base = folder_paths.annotated_filepath(str(filename))
    if base is None:
        base, annotation = folder_paths.get_input_directory(), ""
    else:
        # Only the two roots the picker browses. `[temp]` is a real annotation
        # core would resolve, and nothing in this pack should be rearranging it.
        if os.path.realpath(base) != os.path.realpath(folder_paths.get_output_directory()):
            return None
        annotation = " [output]"
    return os.path.realpath(base), name, annotation


@PromptServer.instance.routes.post("/minimax_creator/move")
async def move_asset(request):
    """Move one file into another subfolder of the root it already lives in —
    the picker's drag-a-thumbnail-onto-a-shelf.

    Renders organize the same way input files do. They *arrive* sorted, because
    the output prefix decides where a render lands (see `outputs.py`), but where
    a file was written is not where it has to stay: a keeper gets dragged out of
    the dated folder it landed in and onto a shelf of its own.
    """
    body = await request.json()
    subfolder = _clean_subfolder(body.get("subfolder", ""))
    if subfolder is None:
        return web.json_response({"error": "bad folder name"}, status=400)
    rooted = _rooted(body.get("filename", ""))
    if rooted is None:
        return web.json_response({"error": "that file is not in a folder the picker browses"},
                                 status=400)
    root, filename, annotation = rooted

    source = os.path.realpath(os.path.join(root, filename))
    if not folder_paths.is_within_directory(root, source) or not os.path.isfile(source):
        return web.json_response({"error": "no such file"}, status=404)

    target_dir = os.path.realpath(os.path.join(root, subfolder)) if subfolder else root
    if target_dir != root and not folder_paths.is_within_directory(root, target_dir):
        return web.json_response({"error": "bad folder name"}, status=400)
    target = os.path.join(target_dir, os.path.basename(source))
    if os.path.realpath(target) == source:
        return web.json_response({"path": filename + annotation})  # already there
    if os.path.exists(target):
        return web.json_response({"error": "a file with that name is already there"}, status=409)

    os.makedirs(target_dir, exist_ok=True)
    os.rename(source, target)
    relative = os.path.relpath(target, root).replace(os.sep, "/")
    # Annotated on the way back out, so the moved file is still addressable as
    # the same kind of thing it was: an attached render has to keep saying
    # `[output]` or `media.resolve` would look for it under input/.
    return web.json_response({"path": relative + annotation})


@PromptServer.instance.routes.post("/minimax_creator/delete")
async def delete_asset(request):
    """Delete one file — organize mode's other action. Files only, never
    directories: a shelf whose last file goes simply drops out of the listing.

    A workflow that still references the file will fail at execute time with
    media.resolve's "not in the input folder any more", which is the honest
    answer — the picker cannot know what every saved workflow points at.

    Deleting a *render* is the case worth pausing on, and it is deliberate: a
    gallery you cannot throw anything out of stops being a gallery after a
    week's rendering. The picker asks first, and there is no undo, which is the
    same deal the input folder has always had.
    """
    body = await request.json()
    rooted = _rooted(body.get("filename", ""))
    if rooted is None:
        return web.json_response({"error": "that file is not in a folder the picker browses"},
                                 status=400)
    root, filename, _ = rooted
    path = os.path.realpath(os.path.join(root, filename))
    if not folder_paths.is_within_directory(root, path) or not os.path.isfile(path):
        return web.json_response({"error": "no such file"}, status=404)
    os.remove(path)
    return web.json_response({"ok": True})


@PromptServer.instance.routes.get("/minimax_creator/settings")
async def read_settings(request):
    """What the settings page shows: every key, filled in. See `settings.py`."""
    return web.json_response({"settings": settings.load()})


@PromptServer.instance.routes.post("/minimax_creator/settings")
async def write_settings(request):
    """Store what the settings page changed and hand back what was stored.

    The reply is the whole settings object rather than an acknowledgement,
    because it is what the page then shows: a value the server would not write
    has to be visibly not written, not left on screen looking chosen.
    """
    try:
        stored = settings.save(await request.json())
    except ValueError as problem:
        return web.json_response({"error": str(problem)}, status=400)
    except OSError as problem:
        return web.json_response({"error": f"could not write the settings file: {problem}"},
                                 status=500)
    return web.json_response({"settings": stored})
