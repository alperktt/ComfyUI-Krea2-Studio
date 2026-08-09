# The blobs

Three nodes, three JSON widgets: `creator_data`, `timeline_data`,
`prestage_data`. The frontend owns them, but the backend accepts a hand-written
one unchanged — which is how the whole compiler was tested before there was a
UI. "Copy creator_data JSON" in the node's right-click menu puts the current one
on the clipboard.

## `creator_data`

```json
{
  "version": 1,
  "prompt": "the woman from @img-1 walks, camera move from @vid-1",
  "assets": [
    {"handle": "img-1", "kind": "image", "role": "reference",   "filename": "a.png", "ref_size": "match", "takes": "person"},
    {"handle": "img-2", "kind": "image", "role": "first_frame", "filename": "b.png"},
    {"handle": "vid-1", "kind": "video", "role": "reference",   "filename": "c.mp4",
     "track": "picture+sound", "trim": {"start": 2.0, "end": 7.5}},
    {"handle": "vid-2", "kind": "video", "role": "reference",   "filename": "e.mp4", "track": "sound"},
    {"handle": "aud-1", "kind": "audio", "role": "reference",   "filename": "d.wav"}
  ],
  "loras": [
    {"name": "h3/turbo_v4.safetensors", "strength": 1.0},
    {"name": "h3/face.safetensors", "strength": 0.7, "modes": ["ref2va"],
     "triggers": ["ohwx woman"], "enabled": false}
  ],
  "refined": {
    "body": "The woman from @img-1 steps into frame, shot on 16mm...",
    "source": "the woman from @img-1 walks, camera move from @vid-1",
    "model": "qwen3_vl_4b_instruct_bf16.safetensors",
    "template": "ref2va"
  },
  "soundscape": "Rain on the awning, one car passing.",
  "music": "",
  "duration_s": 6,
  "aspect": "16:9",
  "short_edge": 768,
  "checkpoint": "auto",
  "models": {
    "fl2va": "minimax_h3_fl2va_fp8_scaled.safetensors",
    "ref2va": "minimax_h3_ref2va_fp8_scaled.safetensors",
    "clip": "minimax_h3_text_encoder_fp8.safetensors",
    "vae": "minimax_h3_video_vae.safetensors",
    "audio_vae": "minimax_h3_audio_vae.safetensors",
    "preview": "taeh3.safetensors",
    "route": "auto",
    "devices": {}
  }
}
```

### Assets

- `handle` — what the user types after `@`. Any `@handle` in the prompt with no
  matching asset is an error, not a silent passthrough: it means an asset was
  deleted and the prompt now lies about the payload. Ordinary prose (`meet me
  @ 5`) is untouched.
- `role` — `reference`, `first_frame`, or `last_frame`. Only images can be a frame.
- `takes` — what a reference image is cited *for*: `full` (default, stored only
  when it is something else), `person`, `object`, `scene`, `style`. It changes
  the prose the refiner writes, not the tensor that is encoded, because that is
  where H3 expresses it. Refused on keyframes and videos, where it would quietly
  mean nothing.
- `track` — video only: which of a clip's streams are referenced.
  - `picture+sound` binds the clip's own soundtrack as a reference audio
    alongside the picture. This is what the UI defaults a newly attached video
    to — but only when the file actually has an audio track, which it asks
    `/minimax_creator/probe`. So a video reference normally costs *two* slots,
    one video and one of the three audio, and needs `audio_vae` set.
  - `picture` references the picture silently, for one video slot.
  - `sound` references the soundtrack and nothing else: the clip becomes an
    audio reference like a bare `.wav`, takes an `<Audio>` label in place of its
    `<Video>` one, costs one audio slot, and its picture is never decoded. It
    does not satisfy the "audio needs a picture beside it" rule.
  - Absent falls back to the older `with_audio` boolean (`true` ->
    `picture+sound`, otherwise `picture`), so workflows saved before the split
    still compile.
- `trim` — `{"start": s, "end": s}` in seconds on the source timeline, video and
  audio only. Absent means the whole file. The cut is applied after the 24 fps
  resample, and to a video's soundtrack over the same window.
- `ref_size` — `match` (scale to the generation's pixel area) or `max` (2048
  short edge; better identity retention, several times slower, because reference
  tokens ride through every sampling step).

Limits are the model's: 9 reference images, 3 videos, 3 audio (soundtracks
count), 12 files in total, and audio is never a standalone reference.

### LoRAs

- `name` — a file under `models/loras`, patched onto whichever checkpoint the
  mode routes to, in list order.
- `modes` — which checkpoints the LoRA claims: `fl2va`, `ref2va`, or absent for
  both. FL2VA and Ref2VA are different weights, so a LoRA is trained against one
  of them; entries the current mode does not route to are skipped. One that *is*
  applied and matches no weights is an error — it would otherwise generate an
  unchanged video and say nothing.
- `strength` — the usual model strength. `0` skips. `enabled: false` keeps the
  entry and its settings without applying it.
- `triggers` — words prefixed to the composed prompt, comma-joined, in list
  order and deduplicated case-insensitively. Seeded from the sidecar's
  `trainedWords` when the LoRA is added and editable from there. The entry holds
  the literal words, so the blob still means something on a machine that does not
  have that LoRA. Only LoRAs actually in the run contribute: the same
  `compile.active_loras()` decides the weights and the words, so they cannot
  disagree.

### The rest

- `checkpoint` — `auto` (follow the mode), `fl2va`, or `ref2va`. Absent means
  `auto`. Pinning changes only which MODEL input is routed out and which LoRAs
  are patched, never how the request is encoded.
- `refined` — the Refine button's rewrite, or absent. `body` stands in for
  `prompt` and is substituted the same way, so it holds `@handles` and never
  ordinals. `enabled: false` keeps it while queueing your own prompt instead.
  `source` is the prompt it was written from, so the panel can say when the
  prompt has moved on. A reference-mode rewrite also carries `sections` — the
  three analysis sections of the six-section form, which nothing can derive from
  a sentence.
- `soundscape` / `music` — `overall_soundscape` and `non_diegetic_music`. Absent
  or empty emits nothing at all, which is not the same as `N/A` ("deliberately
  none"). On a timeline these live on the timeline and a segment leaving them
  empty inherits rather than clearing.
- `models` — the weights, by filename, plus `route` and any `devices` pins. A
  missing block is every field unset rather than an error: a node nobody has
  configured yet is a node someone is still setting up.
- `prompt_override` — replaces the composed prompt verbatim, skeleton and all.
  There is no socket for it any more; the refiner's editable rewrite is the same
  escape hatch with a UI on it.

## `timeline_data`

The same shape with a `segments` array of them, plus `render` (`"chained"` or
`"single"`), the shared `aspect` / `short_edge` / `soundscape` / `music`, and a
global `loras` stack merged in front of each segment's own. Blobs saved before
the toggle existed read as chained. A segment may also carry `continue` and
`continue_audio`, which are the two seam switches and are ignored in one-pass
mode.

## `prestage_data`

`arch` (`krea2` or `ideogram4`), `prompt`, `aspect`, `short_edge`, an optional
`init` image, up to three `refs`, `loras`, `quality`, a `turbo` block, per-arch
`models` sub-blocks, and `peer` — a hint about which Creator or Timeline this
pre-stage belongs to, re-derived by scan and never trusted, because node ids
renumber on paste.
