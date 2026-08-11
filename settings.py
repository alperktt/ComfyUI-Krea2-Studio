"""Preferences that belong to this ComfyUI rather than to a workflow.

The line this file draws: a workflow says what the piece *is* — the prompt, the
references, the duration, where the files land. This says how this machine
writes them. Encoding quality is the first thing on the right side of that line:
two people opening the same `.json` should get the same render, and should not
also be made to agree about how many megabytes it is allowed to take.

So it is not in `creator_data` and it is not a widget. It is one small JSON file
that the settings page writes and the save node reads.

**Where it lives, and why not under `user/default/`.** The picker's favorites go
through the frontend's userdata API, which files them per ComfyUI user. This
cannot: it is read while a queued prompt executes, and an execution has no
request behind it and therefore no user. A file that the settings page wrote to
one place and the node read from another would be a setting that silently does
nothing. One file beside `user/`, read the same way by both.

This file decides what a setting is *allowed* to be; `js/minimax_creator/
settings.js` decides what the page *offers*. They are not mirrors and there is
no mirror test: the encoder's whole scale is legal here, so a value typed into
the file by hand is honoured and shown, while the page offers the four points on
it worth choosing between.

Nothing here imports torch or ComfyUI at module scope — `tests/test_settings.py`
runs it standalone, the same way `outputs.py` is tested.
"""

import json
import os

FILE = "minimax_creator.settings.json"

# libx264's own scale, verbatim: 0 is (near) lossless and 51 is unwatchable.
# Refusing anything outside it here means the number reaching the encoder is
# always a number the encoder has an answer for.
MIN_CRF = 0
MAX_CRF = 51

# What libx264 picks when nothing tells it otherwise, which is exactly what this
# pack wrote before the setting existed. Passing it explicitly changes no file.
DEFAULT_CRF = 23

DEFAULTS = {"video_crf": DEFAULT_CRF}


def clean(raw):
    """A settings blob -> the settings this pack will use. Unknown keys dropped.

    Raises ValueError on a key that is present and unusable, so the route can
    refuse it rather than store a value the node would then ignore. A missing
    key is not an error: it is the default, and a file written by an older
    version is missing every key added since.
    """
    if not isinstance(raw, dict):
        raise ValueError("settings must be an object")
    clean_settings = dict(DEFAULTS)
    if "video_crf" in raw and raw["video_crf"] is not None:
        crf = raw["video_crf"]
        # `True` is an int in Python and would sail through as crf 1.
        if isinstance(crf, bool) or not isinstance(crf, (int, float)) or crf != int(crf):
            raise ValueError("video_crf must be a whole number")
        crf = int(crf)
        if not MIN_CRF <= crf <= MAX_CRF:
            raise ValueError(f"video_crf must be between {MIN_CRF} and {MAX_CRF}")
        clean_settings["video_crf"] = crf
    return clean_settings


def path():
    """The settings file. Imported lazily so this module stays standalone."""
    import folder_paths

    return os.path.join(folder_paths.get_user_directory(), FILE)


def load():
    """The stored settings, with every key filled in.

    A file that cannot be read or cannot be understood reads as the defaults —
    which is what this pack did before anyone opened the settings page, and what
    the page will show, so a value that did not survive is visibly gone rather
    than quietly in force.
    """
    try:
        with open(path(), "r", encoding="utf-8") as handle:
            return clean(json.load(handle))
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save(raw):
    """Store a settings blob and hand back what was stored. Raises ValueError on
    a value this pack will not write."""
    stored = clean(raw)
    target = path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    # Written whole and moved into place: the save node reads this file while
    # renders are queued, and a half-written one would read as the defaults.
    temporary = f"{target}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(stored, handle, indent=2)
    os.replace(temporary, target)
    return stored


def video_crf():
    """The quality target for every video this pack writes."""
    return load()["video_crf"]
