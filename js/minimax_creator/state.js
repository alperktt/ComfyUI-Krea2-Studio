// The UI's model of creator_data, and the rules the backend will enforce anyway.
// Mirrors compile.py: it validates here so the user sees the problem while
// editing rather than at queue time, but compile.py stays authoritative.

import { ASPECT_PRESETS, FPS, MIN_SHORT_EDGE, MAX_SHORT_EDGE, NATIVE_SHORT_EDGE,
         clampRatio, framesForSeconds, secondsForFrames, resolveCanvas } from "./canvas.js";
import { VIDEO_PREFIX, IMAGE_PREFIX, cleanPrefix } from "./outputs.js";

export const MAX_REF_IMAGES = 9;
export const MAX_REF_VIDEOS = 3;
export const MAX_REF_AUDIOS = 3;
export const MAX_REF_FILES = 12;

const PREFIX = { image: "img", video: "vid", audio: "aud" };

/** Which of a reference video's streams are referenced. Mirrors compile.TRACKS.
 *  "sound" drops the picture, so the clip counts as an audio reference and
 *  nothing else. */
export const TRACKS = ["picture", "picture+sound", "sound"];
export const DEFAULT_TRACK = "picture";

/** What a reference is encoded at when nobody said. Mirrors compile.DEFAULT_REF_SIZE.
 *
 *  Per kind, because "max" is a different ceiling for each: an image's is the
 *  reference pipeline's 2048 short edge, a video's is core's 768 reference
 *  canvas, which is already all a video ever gets. Audio has no size and is not
 *  in the table. */
export const DEFAULT_REF_SIZE = { image: "match", video: "max" };

/** The setting in force for an asset — the stored one, or its kind's default.
 *  Read this rather than `asset.ref_size`, which an older blob simply omits. */
export const refSize = (asset) => asset.ref_size || DEFAULT_REF_SIZE[asset.kind] || "match";

/** Whether an asset has a size to choose at all. */
export const sizeable = (asset) =>
  asset.role === "reference" && DEFAULT_REF_SIZE[asset.kind] !== undefined;

/** What of a reference image is actually the reference. "full" — the default —
 *  is the whole picture; the others narrow it so "her from @img-1" stops
 *  dragging the picture's background, palette and pose into the video. Read by
 *  the refiner's glossary; the DiT gets the same tensor either way. */
export const TAKES = ["full", "person", "object", "scene", "style"];

/** The narrowing in force for an asset — the stored one, or the whole picture. */
export const takes = (asset) => (TAKES.includes(asset.takes) ? asset.takes : "full");

/** Whether an asset has a narrowing to choose at all: reference images only —
 *  a keyframe is bound whole by the alignment line, and a video's narrowing is
 *  its track. */
export const takeable = (asset) => asset.kind === "image" && asset.role === "reference";

// ---- weights ----------------------------------------------------------------
//
// Which files the node loads. These used to be sockets; they are named in the
// blob now and `models.py` builds the loaders inside the subgraph. Mirrors
// `models.Weights` field for field — the backend reads exactly these keys.

/** In the order the weights popover lists them, which is the order you set them
 *  in: the two checkpoints, then the three things every mode needs, then the
 *  preview decoder, which changes nothing about the render. */
export const MODEL_FIELDS = ["fl2va", "ref2va", "clip", "vae", "audio_vae", "preview"];

export const MODEL_LABEL = {
  fl2va: "FL2VA checkpoint",
  ref2va: "Ref2VA checkpoint",
  clip: "Text encoder",
  vae: "Video VAE",
  audio_vae: "Audio VAE",
  preview: "Preview decoder",
};

/** What each field is for, said once, in the popover. */
export const MODEL_HINT = {
  fl2va: "Text-only, start/end frame and continuing shots run on these weights.",
  ref2va: "Anything with an @ reference runs on these weights.",
  clip: "H3's text encoder. Loaded as CLIPLoader type 'minimax'.",
  vae: "Decodes the picture.",
  audio_vae: "Decodes the sound. H3 always generates some, so this is never optional.",
  preview: "taeh3, from models/vae_approx — what the live preview decodes through. "
         + "Without it the preview is latent2rgb, which is colour without detail.",
};

/** UNETLoader's own list. Applies to both checkpoints — they are the same
 *  architecture at the same precision on any machine that has both. */
export const MODEL_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"];

/**
 * What `models.route` may hold. Mirrors `models.ROUTES`.
 *
 * "auto" follows the mode, which is what the node has always done. The other two
 * are a standing instruction to run everything on one checkpoint whatever the
 * mode works out to — worth having because the two are one architecture trained
 * twice, and Ref2VA handles the keyframe and text-only payloads FL2VA was
 * trained for perfectly well.
 *
 * The per-request `checkpoint` pin could already say that for one generation,
 * but it is not sticky: attaching a reference makes the pin illegal,
 * `normalizeCheckpoint` drops it, and removing the reference leaves you back on
 * auto. A route survives that, and applies to every segment of a timeline.
 */
export const ROUTES = ["auto", "fl2va", "ref2va"];

/** The next route in the cycle. Here rather than in the badge that cycles it,
 *  so the popover that lists them and the badge that steps through them cannot
 *  disagree about the order. */
export const nextRoute = (route) => ROUTES[(ROUTES.indexOf(route) + 1) % ROUTES.length];

/** Which fields a device can be pinned for: the five that become a loader.
 *  `preview` is not one — it is a filename handed to KJNodes' node, which puts
 *  its decoder wherever the sampler is. Mirrors `models.DEVICE_FIELDS`. */
export const DEVICE_FIELDS = MODEL_FIELDS.filter((field) => field !== "preview");

/** Everything but `preview` is needed to render at all — and of the two
 *  checkpoints, only the one the mode routes to. `requiredModels` answers that
 *  for a given state; this is the part that never depends on the mode. */
export const ALWAYS_REQUIRED = ["clip", "vae", "audio_vae"];

export function emptyModels() {
  return {
    fl2va: "", ref2va: "", clip: "", vae: "", audio_vae: "", preview: "",
    dtype: "default",
    // Which checkpoint everything runs on whatever the mode derives.
    route: "auto",
    // `{field: "cuda:1"}` for anything pinned to a card of its own, through
    // ComfyUI-MultiGPU. Empty is the normal state and means wherever ComfyUI
    // would have put it.
    devices: {},
  };
}

/** Coerce whatever was in the blob into a full weights block. Every field may
 *  legitimately be empty: that is what a node nobody has set up yet looks like,
 *  and it is also what a workflow saved when these were sockets loads as. */
export function parseModels(raw) {
  const out = emptyModels();
  if (!raw || typeof raw !== "object") return out;
  for (const field of MODEL_FIELDS) {
    if (typeof raw[field] === "string") out[field] = raw[field].trim();
  }
  if (MODEL_DTYPES.includes(raw.dtype)) out.dtype = raw.dtype;
  if (ROUTES.includes(raw.route)) out.route = raw.route;
  // Not validated against the machine's device list: the blob may have been
  // saved on a two-card box and opened on a one-card one, and silently dropping
  // the pin would lose the setting rather than report it. `models.loader_for`
  // refuses at queue time, naming the pack.
  if (raw.devices && typeof raw.devices === "object") {
    for (const field of DEVICE_FIELDS) {
      if (typeof raw.devices[field] === "string" && raw.devices[field].trim()) {
        out.devices[field] = raw.devices[field].trim();
      }
    }
  }
  return out;
}

/** Only what was actually picked, so a blob says nothing about fields nobody
 *  has touched — and a `dtype` left alone adds nothing at all. */
function serializeModels(models) {
  const picked = parseModels(models);
  const out = {};
  for (const field of MODEL_FIELDS) {
    if (picked[field]) out[field] = picked[field];
  }
  if (picked.dtype !== "default") out.dtype = picked.dtype;
  // Absent means "follow the mode", so the common case adds nothing.
  if (picked.route !== "auto") out.route = picked.route;
  // Absent means "wherever ComfyUI would", so a single-GPU blob adds nothing.
  if (Object.keys(picked.devices).length) out.devices = { ...picked.devices };
  return { models: out };
}

/**
 * Fill empty fields from unambiguous filename matches, in place.
 *
 * For the case that matters: a workflow saved when these were sockets loads with
 * nothing chosen, and the files are almost always already on disk under
 * recognisable names. Only ever fills a field that is empty, and only when
 * exactly one candidate matches — guessing between two is wrong half the time,
 * and the node asks instead. Returns whether it changed anything.
 */
/** The experimental T=1 image decoder, by name. It is a merged H3 VAE and loads
 *  through the same node as the real one, so nothing downstream can tell them
 *  apart — which is why both the video guess and the video weights popover have
 *  to. In a video workflow it costs multi-frame reconstruction (patch-grid
 *  ghosting, cross-frame mixing); in the pre-stage's H3 branch it is the point. */
export const IMAGE_VAE_RE = /t1[_-]?image|image[_-]vae/i;

const MODEL_HINTS = {
  fl2va: ["fl2va", "first_last"],
  ref2va: ["ref2va"],
  clip: ["minimax"],
  vae: ["minimax", "h3"],
  audio_vae: ["audio"],
  preview: ["taeh3"],
};

export function guessModels(models, files) {
  let changed = false;
  for (const field of MODEL_FIELDS) {
    if (models[field]) continue;
    const needles = MODEL_HINTS[field];
    let matched = (files?.[field] ?? []).filter((name) =>
      needles.some((needle) => name.toLowerCase().includes(needle)));
    // The two VAEs share a folder and both answer to "minimax": whichever says
    // "audio" is the audio one, and the video VAE is whatever is left.
    // The VAEs share a folder and all of them answer to "minimax": whichever
    // says "audio" is the audio one, and the experimental single-image decoder
    // (`t1_image_vae`) is never the video VAE — picking it here would quietly
    // regress every render this node makes. See PRESTAGE_HINTS.minimax, which
    // is where that file *is* the right answer.
    if (field === "vae") {
      matched = matched.filter((name) => !IMAGE_VAE_RE.test(name) && !name.toLowerCase().includes("audio"));
    }
    if (matched.length !== 1) continue;
    models[field] = matched[0];
    changed = true;
  }
  return changed;
}

/**
 * Which fields a render cannot go without: the three constants plus whichever
 * checkpoints it routes to. Mirrors `models.check`, which refuses at queue time
 * on exactly this list — a Creator passes `[checkpoint(state)]` and a Timeline
 * passes `timelineCheckpoints(timeline)`, because a chained clip legitimately
 * runs some shots on one checkpoint and some on the other.
 */
export function requiredModels(checkpoints) {
  return [...ALWAYS_REQUIRED, ...checkpoints];
}

/**
 * The checkpoints a render will actually load, after the route has had its say.
 *
 * A forced route collapses the set to one whatever the modes derived, which is
 * the point of it: "always Ref2VA" on a timeline means one loader for the whole
 * clip rather than one per checkpoint its shots happened to want.
 */
export function routedCheckpoints(models, derived) {
  const route = models?.route ?? "auto";
  return route === "auto" ? derived : [route];
}

/** Which required fields are still empty, in listing order. */
export function missingModels(models, required) {
  return required.filter((field) => !models[field])
    .sort((a, b) => MODEL_FIELDS.indexOf(a) - MODEL_FIELDS.indexOf(b));
}

// ---- turbo ------------------------------------------------------------------
//
// The turbo block is the switch's memory, not the LoRA itself. Engaged, the
// distillation LoRA is an ordinary entry in `loras` — same stack, same manager,
// same one-click disable — and this records which file the switch reaches for,
// which quality it was left at, and what the sampler row said before it was
// thrown, so switching off puts the row back rather than guessing at defaults.
// compile.py never reads it.

/** In effort order. The step counts are the H3 turbo community's numbers: 4 is
 *  the distillation target and the floor, 6 the comfort zone, 8 about as close
 *  to a native 20-step render as the LoRAs get — past 8 they over-sharpen. */
export const TURBO_QUALITIES = ["draft", "medium", "good"];
export const TURBO_STEPS = { draft: 4, medium: 6, good: 8 };

/** What the switch sets the row to. H3 samples picture and sound as one latent
 *  on two flow clocks, and at turbo step counts res_multistep leaves the audio
 *  warbling — euler + beta is the combination the turbo LoRAs were tuned
 *  against and the one that keeps the soundtrack intact. */
export const TURBO_SAMPLER = "euler";
export const TURBO_SCHEDULER = "beta";

/** Where the row returns to when the switch is thrown off and nothing was
 *  saved — the node's own declared defaults, mirrored from creator_node.py. */
export const TURBO_RESET = { steps: 20, sampler_name: "res_multistep", scheduler: "simple" };

/** The strength the switch engages a file at. The two families were distilled
 *  at different scales and their communities settled on different numbers:
 *  lightx2v's distill runs at ~0.6, larryvrh's at 1.0. A guess off the
 *  filename, and the LoRA manager's slider overrides it like any other entry. */
export const turboStrength = (name) => (/lightx2v/i.test(name || "") ? 0.6 : 1.0);

export function emptyTurbo() {
  return {
    // The file the switch engages, relative to models/loras. Picked in the
    // weights popover, because it is machine configuration like the files above
    // it: set once when the LoRA is downloaded, then thrown from the pill.
    lora: "",
    // The user said their checkpoint is a merged distill — turbo with no LoRA
    // at all, the switch owning only the sampler row. Remembered so the pill
    // engages directly on the next press instead of asking again.
    merged: false,
    quality: "medium",
    // Whether the switch is thrown. The LoRA entry itself can be removed from
    // two other places — the chip and the manager — which is why this is
    // reconciled against the stack on every commit rather than trusted.
    on: false,
    // The sampler row as it stood when the switch was thrown: {steps,
    // sampler_name, scheduler}. Null when off.
    saved: null,
  };
}

export function parseTurbo(raw) {
  const out = emptyTurbo();
  if (!raw || typeof raw !== "object") return out;
  if (typeof raw.lora === "string") out.lora = raw.lora.trim();
  out.merged = raw.merged === true;
  if (TURBO_QUALITIES.includes(raw.quality)) out.quality = raw.quality;
  out.on = raw.on === true;
  if (raw.saved && typeof raw.saved === "object") {
    out.saved = {
      steps: Number(raw.saved.steps) || TURBO_RESET.steps,
      sampler_name: typeof raw.saved.sampler_name === "string"
        ? raw.saved.sampler_name : TURBO_RESET.sampler_name,
      scheduler: typeof raw.saved.scheduler === "string"
        ? raw.saved.scheduler : TURBO_RESET.scheduler,
    };
  }
  return out;
}

/** Nothing at all until a file is picked, so every blob from before the switch
 *  existed — and every node nobody turbos — says nothing about it. */
export function serializeTurbo(turbo) {
  const picked = parseTurbo(turbo);
  if (!picked.lora && !picked.merged && !picked.on) return {};
  const out = { lora: picked.lora };
  if (picked.merged) out.merged = true;
  if (picked.quality !== "medium") out.quality = picked.quality;
  if (picked.on) out.on = true;
  if (picked.saved) out.saved = { ...picked.saved };
  return { turbo: out };
}

/** The two H3 checkpoints, which is also the granularity a LoRA belongs to:
 *  T2VA, I2VA, L2VA and FL2VA are all the same weights. */
export const CHECKPOINTS = ["fl2va", "ref2va"];
export const CHECKPOINT_LABEL = { fl2va: "FL2VA", ref2va: "Ref2VA" };
/** What `state.checkpoint` may hold: follow the mode, or pin one. */
export const CHECKPOINT_CHOICES = ["auto", ...CHECKPOINTS];
export const DEFAULT_STRENGTH = 1.0;

export function emptyState() {
  return {
    version: 1,
    prompt: "",
    // The refiner's rewrite of `prompt`, when there is one: `{body, sections?,
    // source, model, enabled}`. Stored with its `@handles` intact rather than
    // with H3's ordinals in it, so compile.py substitutes it exactly as it
    // substitutes typed text and attaching an asset re-labels it correctly.
    refined: null,
    // The two Context-IR audio fields. The timeline owns its own pair and hands
    // them down; a lone generation has nowhere else to keep them.
    soundscape: "",
    music: "",
    assets: [],
    loras: [],
    duration_s: 6,
    aspect: "16:9",
    short_edge: NATIVE_SHORT_EDGE,
    // "auto" follows the mode. Pinning it runs the same payload on the other
    // weights; compile.py decides which pins it will accept.
    checkpoint: "auto",
    // Where the finished clip lands under output/. Owned by the node for the
    // same reason the weights are: a timeline saves one file, so a segment
    // carrying its own would be a second answer to a question that has one.
    output_prefix: VIDEO_PREFIX,
    // Which files to load. Owned by the node, not by a segment — a timeline
    // segment inherits the timeline's and never carries its own.
    models: emptyModels(),
    // The turbo switch. Owned by the node for the same reason the weights are.
    turbo: emptyTurbo(),
  };
}

export function parseState(raw) {
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      const state = { ...emptyState(), ...parsed };
      // Workflows saved before LoRAs existed have no key at all, and a
      // hand-edited blob can have the wrong type in it.
      if (!Array.isArray(state.loras)) state.loras = [];
      if (!Array.isArray(state.assets)) state.assets = [];
      if (!state.refined || typeof state.refined !== "object") state.refined = null;
      for (const key of ["soundscape", "music"]) {
        if (typeof state[key] !== "string") state[key] = "";
      }
      if (!CHECKPOINT_CHOICES.includes(state.checkpoint)) state.checkpoint = "auto";
      state.output_prefix = parsePrefix(state.output_prefix, VIDEO_PREFIX);
      state.models = parseModels(state.models);
      state.turbo = parseTurbo(state.turbo);
      normalizeCheckpoint(state);
      for (const asset of state.assets) {
        if (asset?.kind !== "video") continue;
        // Workflows saved before the picture/sound split carry the two-state
        // `with_audio` boolean. compile.py still reads it; the editor works in
        // tracks from here on, so it is converted on the way in.
        if (!TRACKS.includes(asset.track)) asset.track = asset.with_audio ? "picture+sound" : DEFAULT_TRACK;
        delete asset.with_audio;
      }
      return state;
    }
  } catch {
    // A malformed blob is recoverable: fall back to empty rather than leaving
    // the node unusable. The user's text is gone either way.
  }
  return emptyState();
}

/** A blob's `output_prefix` as the UI holds it: always a string, always one
 *  `outputs.py` would accept.
 *
 *  A blob saved before this field existed has no key, and one hand-edited to
 *  something unusable has to leave the node editable — so both fall back to the
 *  node's default rather than being carried into the field as an error the user
 *  never typed. The queue refuses the blob's own value either way: this is the
 *  *editor's* copy, not the one that gets rendered with. */
function parsePrefix(raw, fallback) {
  if (raw === undefined || raw === null) return fallback;
  return cleanPrefix(raw, fallback).prefix ?? fallback;
}

/** …and back out, but only when it departs from the node's default — so a blob
 *  nobody retargeted round-trips exactly as it did before the field existed. */
function serializePrefix(prefix, fallback) {
  const clean = cleanPrefix(prefix, fallback).prefix ?? fallback;
  return clean === fallback ? {} : { output_prefix: clean };
}

/** LoRA entries, stripped to what compile.py reads. Shared by a segment's own
 *  list and the timeline's global one — they are the same kind of entry and are
 *  merged into one stack by `compile.merge_loras`. */
function serializeLoras(entries) {
  return entries.map((entry) => {
    const out = { name: entry.name, strength: round2(entry.strength) };
    if (entry.enabled === false) out.enabled = false;
    // The literal words, not a pointer at the sidecar: creator_data has to
    // still say what it means on a machine where that LoRA is missing.
    if (entry.triggers?.length) out.triggers = [...entry.triggers];
    // Absent means both checkpoints, so the common case adds nothing.
    if (!claimsBoth(entry)) out.modes = [...entry.modes];
    return out;
  });
}

/** The refiner's rewrite, stripped to what compile.py reads, or nothing.
 *
 *  An empty body is nothing at all rather than an empty rewrite: reverting
 *  should leave a blob that looks like one the refiner was never run on. */
function serializeRefined(refined) {
  const body = (refined?.body ?? "").trim();
  const sections = refined?.sections;
  if (!body && !sections) return {};
  return {
    refined: {
      ...(body ? { body } : {}),
      ...(sections ? { sections: { ...sections } } : {}),
      // Kept so the panel can say the prompt has moved on since; compile.py
      // never reads either, and both are small enough not to be worth splitting
      // out of the one object that says "this was refined".
      source: refined.source ?? "",
      ...(refined.model ? { model: refined.model } : {}),
      // Which template wrote this prose, and whether it was pinned — kept for
      // the same reason as `model`: after a reload it is the only record of
      // which form the stored rewrite is in.
      ...(refined.template ? { template: refined.template } : {}),
      ...(refined.forced ? { forced: true } : {}),
      // What the rewrite overwrote in `soundscape` and `music`, so Revert puts
      // them back rather than leaving generated prose in fields the user never
      // typed in — including after a reload, which is exactly when nobody
      // remembers what was in them.
      // Two empty strings when that is what was there: "the user had typed
      // nothing" is the fact Revert needs most, and dropping it as falsy would
      // leave the rewrite's own prose behind on exactly that case.
      ...(refined.replaced ? { replaced: { ...refined.replaced } } : {}),
      // Absent means on, so the common case adds nothing.
      ...(refined.enabled === false ? { enabled: false } : {}),
    },
  };
}

/** The parts of a state every generation has, timeline segment or not. */
function serializeCommon(state) {
  return {
    prompt: state.prompt ?? "",
    ...serializeRefined(state.refined),
    // An empty field is emitted as nothing, which is not the same as "N/A" —
    // see contextir.compose. A segment leaving them blank inherits the
    // timeline's rather than clearing them.
    ...(state.soundscape?.trim() ? { soundscape: state.soundscape } : {}),
    ...(state.music?.trim() ? { music: state.music } : {}),
    assets: state.assets.map((asset) => {
      const out = { handle: asset.handle, kind: asset.kind, role: asset.role, filename: asset.filename };
      if (asset.kind === "video") out.track = asset.track || DEFAULT_TRACK;
      // Only what departs from the backend's own default for the kind, so the
      // common setting adds nothing and an old blob round-trips unchanged.
      if (sizeable(asset) && refSize(asset) !== DEFAULT_REF_SIZE[asset.kind]) {
        out.ref_size = refSize(asset);
      }
      // Absent means the whole file, so a clip nobody trimmed adds nothing.
      if (asset.trim && asset.kind !== "image") {
        out.trim = { start: asset.trim.start, end: asset.trim.end };
      }
      // Absent means the whole picture, so an unnarrowed reference adds nothing
      // and compile.py refuses the field anywhere it means nothing.
      if (takeable(asset) && takes(asset) !== "full") {
        out.takes = takes(asset);
      }
      return out;
    }),
    loras: serializeLoras(state.loras),
    duration_s: state.duration_s,
    // Absent means "follow the mode", so the common case adds nothing.
    ...(state.checkpoint && state.checkpoint !== "auto" ? { checkpoint: state.checkpoint } : {}),
  };
}

export function serializeState(state) {
  return JSON.stringify({
    version: 1,
    ...serializeCommon(state),
    aspect: state.aspect,
    short_edge: state.short_edge,
    ...serializePrefix(state.output_prefix, VIDEO_PREFIX),
    // Not in serializeCommon: the weights belong to the node, and a timeline
    // segment goes through that function too. The turbo switch likewise.
    ...serializeModels(state.models),
    ...serializeTurbo(state.turbo),
  }, null, 2);
}

// ---- timeline ---------------------------------------------------------------
//
// A timeline is a global prompt, one canvas, and a list of segments. A segment
// is an ordinary state — same assets, same LoRAs, same checkpoint routing, so
// the same editor drives it — minus the canvas, which belongs to the timeline
// because the segments are concatenated at the end and have to match, plus one
// flag saying whether it starts from the previous segment's last frame.

export const MAX_SEGMENTS = 24;

/** Mirrors compile.RENDER_MODES. "chained" is a generation per segment,
 *  concatenated; "single" is one generation whose description holds every
 *  segment as a `[Shot n]` with its own cut time. */
export const RENDER_MODES = ["chained", "single"];
export const isSingle = (timeline) => timeline.render === "single";

/** Mirrors compile.DEFAULT_AUDIO_TAIL_S / MAX_AUDIO_TAIL_S. Short on purpose:
 *  the reference rows ride through every sampling step, and a long tail pushes
 *  the target's time origin away from the inherited start frame. */
export const DEFAULT_AUDIO_TAIL_S = 1.0;
export const MAX_AUDIO_TAIL_S = 4.0;

const clampTail = (value) => {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds <= 0) return DEFAULT_AUDIO_TAIL_S;
  return Math.min(seconds, MAX_AUDIO_TAIL_S);
};

export function emptySegment() {
  const state = emptyState();
  delete state.version;
  state.continue = false;
  // The sound seam, independent of the picture one: a hard cut whose music keeps
  // playing and a match cut that resets the sound are both ordinary.
  state.continue_audio = false;
  return state;
}

export function emptyTimeline() {
  return {
    version: 2,
    // How the segments become video. Chained by default: it is the mode with no
    // length limit, and it is what every timeline saved before this existed was.
    render: "chained",
    prompt: "",
    // The two Context-IR audio fields, global because a soundscape and a score
    // belong to the piece rather than to one shot. See contextir.py.
    soundscape: "",
    music: "",
    // The reference form's three analysis sections, in one pass only: there the
    // shots are a single generation over one merged reference pool, so the
    // analysis describes the whole clip. Chained, each segment keeps its own.
    refined: null,
    aspect: "16:9",
    short_edge: NATIVE_SHORT_EDGE,
    // Patched onto every segment, in front of whatever that segment adds. What
    // a turbo LoRA is for: you want it on the whole clip, not shot by shot.
    loras: [],
    // How much of the previous segment's sound a continuing seam inherits.
    // Mirrors compile.DEFAULT_AUDIO_TAIL_S.
    audio_tail_s: DEFAULT_AUDIO_TAIL_S,
    // Where the finished clip lands under output/. One prefix, because a
    // timeline is one file however many segments went into it.
    output_prefix: VIDEO_PREFIX,
    // One set of weights for the whole clip. Chained or not, the segments are
    // concatenated at the end and cannot come from different checkpoints of the
    // same name any more than they can come out different sizes.
    models: emptyModels(),
    // The turbo switch. Global like the LoRA it engages: a speed-up belongs to
    // the run, not to shot 3.
    turbo: emptyTurbo(),
    segments: [emptySegment()],
  };
}

/**
 * Mirror the timeline's canvas onto each segment, so a segment state answers
 * `resolved()` and `mode()` on its own and the editor needs no special case.
 * Stripped again by `serializeTimeline` — the segments do not own it.
 */
function syncCanvas(timeline) {
  for (const segment of timeline.segments) {
    segment.aspect = timeline.aspect;
    segment.short_edge = timeline.short_edge;
  }
  // Segment 1 has nothing in front of it. Kept in step here rather than guarded
  // at every read, so reordering cannot leave a stale flag behind.
  if (timeline.segments.length) {
    timeline.segments[0].continue = false;
    timeline.segments[0].continue_audio = false;
  }
  return timeline;
}

export { syncCanvas as syncTimeline };

export function parseTimeline(raw) {
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      const timeline = { ...emptyTimeline(), ...parsed };
      // Workflows saved before either existed have no key at all, and a
      // hand-edited blob can have the wrong type in it.
      if (!Array.isArray(timeline.loras)) timeline.loras = [];
      if (!RENDER_MODES.includes(timeline.render)) timeline.render = "chained";
      timeline.audio_tail_s = clampTail(timeline.audio_tail_s);
      for (const key of ["soundscape", "music"]) {
        if (typeof timeline[key] !== "string") timeline[key] = "";
      }
      if (!timeline.refined || typeof timeline.refined !== "object") timeline.refined = null;
      timeline.output_prefix = parsePrefix(timeline.output_prefix, VIDEO_PREFIX);
      timeline.models = parseModels(timeline.models);
      timeline.turbo = parseTurbo(timeline.turbo);
      const segments = Array.isArray(parsed.segments) ? parsed.segments : [];
      timeline.segments = (segments.length ? segments : [{}]).map((raw) => {
        const segment = parseState(JSON.stringify(raw ?? {}));
        delete segment.version;
        // The weights are the timeline's. A segment carrying its own would be a
        // second answer to a question that has one. The turbo switch likewise.
        delete segment.models;
        delete segment.turbo;
        segment.continue = raw?.continue === true;
        segment.continue_audio = raw?.continue_audio === true;
        return segment;
      });
      return syncCanvas(timeline);
    }
  } catch {
    // Same reasoning as parseState: an unreadable blob leaves the node usable.
  }
  return emptyTimeline();
}

export function serializeTimeline(timeline) {
  return JSON.stringify({
    version: 2,
    render: timeline.render === "single" ? "single" : "chained",
    prompt: timeline.prompt ?? "",
    // Absent means the field is not emitted at all, which is not the same as
    // "N/A" — see contextir.compose — so an empty box writes nothing.
    ...(timeline.soundscape?.trim() ? { soundscape: timeline.soundscape } : {}),
    ...(timeline.music?.trim() ? { music: timeline.music } : {}),
    // Sections only: the timeline has no body of its own, and `single_payload`
    // assembles the shots into one. `serializeRefined` drops the key when there
    // is nothing in it, which is every timeline that is not a refined one-pass.
    ...serializeRefined(timeline.refined),
    aspect: timeline.aspect,
    short_edge: timeline.short_edge,
    loras: serializeLoras(timeline.loras ?? []),
    audio_tail_s: clampTail(timeline.audio_tail_s),
    ...serializePrefix(timeline.output_prefix, VIDEO_PREFIX),
    ...serializeModels(timeline.models),
    ...serializeTurbo(timeline.turbo),
    segments: timeline.segments.map((segment, index) => {
      const out = serializeCommon(segment);
      // Absent means a hard cut, which is the default, so only continuations
      // add anything. Never on the first segment: there is nothing to continue.
      if (index > 0 && segment.continue) out.continue = true;
      if (index > 0 && segment.continue_audio) out.continue_audio = true;
      return out;
    }),
  }, null, 2);
}

/** A copy that shares nothing with the original — for "duplicate segment". */
export function cloneSegment(segment) {
  return JSON.parse(JSON.stringify(segment));
}

/**
 * Where each shot cuts in, and what the pills add up to before snapping.
 *
 * Off the raw durations rather than the snapped ones, mirroring
 * `compile.single_payload`: a one-pass render has one frame count and it is the
 * total, so there is no per-shot grid for a cut time to land on.
 */
export function cutTimes(timeline) {
  const at = [];
  let total = 0;
  for (const segment of timeline.segments) {
    at.push(total);
    total += Number(segment.duration_s) || 0;
  }
  return { at, total };
}

/** `5` -> `"00:05.000"`. Mirrors `contextir.shot_time`, which writes the real one. */
export function shotTime(seconds) {
  const ms = Math.round(Number(seconds) * 1000);
  const pad = (n, width) => String(n).padStart(width, "0");
  return `${pad(Math.floor(ms / 60000), 2)}:${pad(Math.floor(ms / 1000) % 60, 2)}.${pad(ms % 1000, 3)}`;
}

/**
 * The frames the finished clip holds.
 *
 * Chained, every segment snaps to the 17n+5 grid on its own and the results are
 * concatenated. In one pass there is a single generation, so the durations are
 * summed and snapped once — which is not the same number, and is the one the
 * sampler will actually be asked for.
 */
export function timelineFrames(timeline) {
  if (isSingle(timeline)) return framesForSeconds(cutTimes(timeline).total);
  return timeline.segments.reduce((total, segment) => total + framesForSeconds(segment.duration_s), 0);
}

/** What the finished clip will run to. */
export function timelineSeconds(timeline) {
  return secondsForFrames(timelineFrames(timeline));
}

// ---- pre-stage --------------------------------------------------------------
//
// The PreStage node's blob. Mirrors compile_image.py the way this file mirrors
// compile.py and canvas.js mirrors canvas.py: the UI shows the resolved canvas
// and refuses the illegal combinations early, and compile_image.py stays
// authoritative at queue time.

export const PRESTAGE_ARCHES = ["krea2", "ideogram4", "minimax"];
export const PRESTAGE_ARCH_LABEL = {
  krea2: "Krea 2", ideogram4: "Ideogram 4", minimax: "MiniMax H3",
};

/** The H3 branch is the video pipeline with one latent frame decoded, so its
 *  canvas, aspects and references are the video node's rather than the image
 *  models' — `isStill(state)` is the switch every per-arch rule below turns on.
 *  Mirrors compile_still.ARCH. */
export const PRESTAGE_STILL_ARCH = "minimax";
export const isStill = (state) => state?.arch === PRESTAGE_STILL_ARCH;

/** What the length pill offers, and what a fresh still samples. Every entry is
 *  a legal 17n+5 count; 5 is the cheapest clip H3 can be asked for and 124 is
 *  the bottom of its trained range. Mirrors compile_still.STILL_LENGTHS. */
export const PRESTAGE_STILL_LENGTHS = [5, 22, 39, 56, 90, 124];
export const PRESTAGE_STILL_FRAMES = 5;
export const PRESTAGE_STILL_INDEX = 0;
export const PRESTAGE_PROMPT_MODES = ["context-ir", "plain"];

/** Frames -> latent frames. Mirrors compile_still.latent_frames, which mirrors
 *  core: the VAE is causal on the 17k+5 <-> 5k+2 grid. */
export const stillLatentFrames = (frames) =>
  (frames <= 5 ? 2 : Math.floor((frames - 5) / 17) * 5 + 2);

/** H3 takes nine reference images where the Qwen edit encoder takes three. */
export const PRESTAGE_STILL_MAX_REFS = MAX_REF_IMAGES;

/** What the H3 branch writes into the sampler row: the Creator node's own
 *  defaults, because it is the Creator's sampler. The released checkpoints are
 *  CFG-distilled, which is what the 1.0 is. */
export const PRESTAGE_STILL_ROW = {
  steps: 20, cfg: 1.0, sampler_name: "res_multistep", scheduler: "simple",
};

export const PRESTAGE_CANVAS_MULTIPLE = 16;
export const PRESTAGE_MIN_EDGE = 512;
export const PRESTAGE_MAX_EDGE = 2048;
export const PRESTAGE_DEFAULT_EDGE = 1024;
export const PRESTAGE_MAX_PIXELS = 2048 * 2048;
export const PRESTAGE_MIN_RATIO = 1 / 3;
export const PRESTAGE_MAX_RATIO = 3;

// Order matters: this is the order the ratio popover lists them in. Wider than
// the video envelope on purpose — a style sheet is a legitimate still.
export const PRESTAGE_ASPECTS = [
  ["16:9", 16 / 9],
  ["3:2", 3 / 2],
  ["4:3", 4 / 3],
  ["1:1", 1],
  ["3:4", 3 / 4],
  ["2:3", 2 / 3],
  ["9:16", 9 / 16],
  ["21:9", 21 / 9],
];

/** Core's Qwen-edit encoder has exactly three image slots. */
export const PRESTAGE_MAX_REFS = 3;

/** What each Krea 2 checkpoint wants from the sampler row — what the arch and
 *  turbo pills write into the widgets, mirrored from compile_image.py. */
export const PRESTAGE_KREA_RAW = { steps: 52, cfg: 3.5, sampler_name: "euler", scheduler: "simple" };
export const PRESTAGE_KREA_TURBO = { cfg: 1.0, sampler_name: "euler", scheduler: "simple" };
export const PRESTAGE_TURBO_QUALITIES = ["draft", "medium", "good"];
export const PRESTAGE_TURBO_STEPS = { draft: 4, medium: 6, good: 8 };

/** Ideogram's official preset table (V4_QUALITY_48 / V4_DEFAULT_20 /
 *  V4_TURBO_12). The presets own steps *and* the schedule shape; the widget cfg
 *  feeds the dual-model guider, 7 being the template's number. */
export const PRESTAGE_IDEOGRAM_QUALITIES = ["quality", "default", "turbo"];
export const PRESTAGE_IDEOGRAM_STEPS = { quality: 48, default: 20, turbo: 12 };
export const PRESTAGE_IDEOGRAM_ROW = { cfg: 7.0, sampler_name: "euler" };

export const PRESTAGE_DEFAULT_DENOISE = 0.65;
export const PRESTAGE_MIN_DENOISE = 0.05;

/** Which weight fields each architecture has, in popover order. Mirrors
 *  render_image.ARCH_FIELDS. */
export const PRESTAGE_FIELDS = {
  krea2: ["model", "turbo_model", "clip", "vae"],
  ideogram4: ["model", "uncond_model", "clip", "vae"],
  // The video node's fields under the video node's names, because the H3
  // branch loads through `models.Weights` — minus the audio VAE, which a
  // picture-only render never opens. Mirrors render_still.weights_from_blob.
  minimax: ["fl2va", "ref2va", "clip", "vae", "audio_vae", "preview"],
};
export const PRESTAGE_FIELD_LABEL = {
  model: "Checkpoint",
  turbo_model: "Turbo checkpoint",
  uncond_model: "Unconditional checkpoint",
  clip: "Text encoder",
  vae: "VAE",
  fl2va: "FL2VA checkpoint",
  ref2va: "Ref2VA checkpoint",
  preview: "Preview decoder",
};
export const PRESTAGE_FIELD_HINT = {
  krea2: {
    model: "Krea 2 RAW — the undistilled base. ~52 steps at cfg 3.5, and the one to train LoRAs against.",
    turbo_model: "Krea 2 Turbo — the 8-step distillation the turbo pill swaps in. LoRAs trained on RAW apply here too.",
    clip: "Qwen3-VL 4B, loaded as CLIPLoader type 'krea2'.",
    vae: "The Qwen image VAE.",
  },
  ideogram4: {
    model: "Ideogram 4.0's conditional branch.",
    uncond_model: "The unconditional branch — Ideogram ships CFG as a second model. "
                + "Optional: without it the render runs ordinary CFG on the one checkpoint.",
    clip: "Qwen3-VL 8B, loaded as CLIPLoader type 'ideogram4'.",
    vae: "The Flux 2 VAE.",
  },
  minimax: {
    fl2va: "The same FL2VA checkpoint the Creator renders with. Used for a still with "
         + "no references, and for one that starts from an image.",
    ref2va: "The same Ref2VA checkpoint the Creator renders with. Used the moment a "
          + "reference is attached.",
    clip: "H3's text encoder. Loaded as CLIPLoader type 'minimax'.",
    vae: "The experimental single-image H3 VAE (minimax_h3_t1_image_vae_*), which "
       + "decodes one temporal latent into one picture. Its encoder is the stock H3 "
       + "encoder untouched, so the same file also encodes this render's keyframe and "
       + "references. Do not select it as the Creator's video VAE.",
    audio_vae: "Only needed when a reference clip is cited with its soundtrack, or a "
             + "reference audio file is attached — a still decodes no sound, but it can cite "
             + "some. Left unloaded otherwise.",
    preview: "taeh3, from models/vae_approx — the same live preview the video nodes use. "
           + "It decodes the whole sampled clip, not just the frame that is kept.",
  },
};

/** Filename hints for `guessPreStageModels`, per arch per field. */
const PRESTAGE_HINTS = {
  krea2: { model: ["krea2_raw"], turbo_model: ["krea2_turbo"], clip: ["qwen3vl_4b"], vae: ["qwen_image_vae"] },
  ideogram4: {
    model: ["ideogram4"], uncond_model: ["ideogram4_unconditional"],
    clip: ["qwen3vl_8b"], vae: ["flux2"],
  },
  // The checkpoints and text encoder are the video node's files under the video
  // node's names, so the needles are `MODEL_HINTS`'. The VAE is the one field
  // that is emphatically *not* the video node's: here the image decoder is the
  // right answer and the stock H3 VAE is the wrong one.
  minimax: {
    fl2va: ["fl2va", "first_last"], ref2va: ["ref2va"],
    clip: ["minimax"], vae: ["t1_image", "image_vae"],
    audio_vae: ["audio"], preview: ["taeh3"],
  },
};

// ---- per-arch geometry ------------------------------------------------------
//
// The image models generate on their own /16 grid up to 2048; the H3 branch
// generates on the video model's /32 grid at a 768 short edge, because what it
// produces is a video frame. Every geometry rule below asks which.

/** The ratio pill's options for this arch. H3's envelope is 21:9..9:16 (the six
 *  `canvas.ASPECT_PRESETS`); the image models add 3:2 and 2:3. */
export function prestageAspects(state) {
  return isStill(state) ? ASPECT_PRESETS : PRESTAGE_ASPECTS;
}

/** The resolution slider's bounds and default for this arch. */
export function prestageEdges(state) {
  return isStill(state)
    ? { min: MIN_SHORT_EDGE, max: MAX_SHORT_EDGE, step: 32, default: NATIVE_SHORT_EDGE }
    : { min: PRESTAGE_MIN_EDGE, max: PRESTAGE_MAX_EDGE, step: PRESTAGE_CANVAS_MULTIPLE,
        default: PRESTAGE_DEFAULT_EDGE };
}

export function clampPreStageEdge(state, edge) {
  const bounds = prestageEdges(state);
  const value = Math.round(Number(edge));
  if (!Number.isFinite(value)) return bounds.default;
  return Math.max(bounds.min, Math.min(bounds.max, value));
}

/** How many references this arch reads: three image slots on the Qwen edit
 *  encoder, nine on H3's own reference pipeline. */
export function prestageMaxRefs(state) {
  return isStill(state) ? PRESTAGE_STILL_MAX_REFS : PRESTAGE_MAX_REFS;
}

export function emptyPreStage() {
  return {
    version: 1,
    arch: "krea2",
    prompt: "",
    aspect: "16:9",
    short_edge: PRESTAGE_DEFAULT_EDGE,
    // {"filename", "denoise"} for img2img, or null. On the H3 branch it is the
    // start frame and the denoise is unread — a keyframe is pinned, not partly
    // denoised.
    init: null,
    // {"filename"} — the end frame, H3 only: the image the sampled clip closes
    // on. Nothing on the image models takes one.
    end: null,
    // [{handle, kind, filename, ...}] — references. Three image slots on Krea 2
    // (the Qwen edit encoder's), and on H3 the video node's whole reference
    // pipeline: nine images, three videos, three audio clips.
    refs: [],
    loras: [],
    // The image turbo is a checkpoint swap, not a LoRA — Krea Turbo *is* a
    // distilled checkpoint — but the pill keeps the H3 contract: it saves the
    // sampler row once per throw and puts it back exactly on release.
    turbo: { on: false, quality: "good", saved: null },
    // Ideogram's speed axis: which official preset shapes the schedule.
    quality: "default",
    // The H3 branch's own settings. `frames` is the clip that gets sampled and
    // `latent_index` which of its latent frames becomes the picture; the rest
    // is the video node's vocabulary. `dev` is the sweep — see prestage.js.
    minimax: emptyStill(),
    // Where the still lands under output/. Its own default folder, which is
    // what sorts stills apart from finished renders in the gallery.
    output_prefix: IMAGE_PREFIX,
    models: emptyPreStageModels(),
    // A hint for peer discovery, never authoritative — ids renumber on paste,
    // so the pre-stage pill re-derives the pairing by scan.
    peer: null,
  };
}

export function emptyPreStageModels() {
  return { krea2: {}, ideogram4: {}, minimax: {}, dtype: "default" };
}

export function emptyStill() {
  return {
    frames: PRESTAGE_STILL_FRAMES,
    latent_index: PRESTAGE_STILL_INDEX,
    prompt_mode: "context-ir",
    checkpoint: "auto",
    // DEV: the sweep. Empty lists mean "just render the settings above"; the
    // pill fills them to compare lengths, latent frames and decoders in one
    // queue. Comes out with the dev pill once the numbers are in.
    dev: { lengths: [], indices: [], vaes: [] },
  };
}

/** The H3 sub-block, coerced. Mirrors compile_still's reading of it. */
export function parseStill(raw) {
  const out = emptyStill();
  if (!raw || typeof raw !== "object") return out;
  const frames = Number(raw.frames);
  if (Number.isFinite(frames)) out.frames = framesForSeconds(Math.max(1, frames) / FPS);
  const index = Number(raw.latent_index);
  if (Number.isFinite(index)) out.latent_index = Math.round(index);
  if (PRESTAGE_PROMPT_MODES.includes(raw.prompt_mode)) out.prompt_mode = raw.prompt_mode;
  if (ROUTES.includes(raw.checkpoint)) out.checkpoint = raw.checkpoint;
  const dev = raw.dev && typeof raw.dev === "object" ? raw.dev : {};   // DEV
  for (const [key, cast] of [["lengths", Number], ["indices", Number], ["vaes", String]]) {
    if (Array.isArray(dev[key])) out.dev[key] = dev[key].map(cast).filter((v) => v === 0 || v);
  }
  return out;
}

/** Only what differs from a fresh block, so an untouched still says nothing. */
function serializeStill(still) {
  const out = { frames: still.frames, latent_index: still.latent_index };
  if (still.prompt_mode !== "context-ir") out.prompt_mode = still.prompt_mode;
  if (still.checkpoint !== "auto") out.checkpoint = still.checkpoint;
  const dev = {};                                                      // DEV
  for (const key of ["lengths", "indices", "vaes"]) {
    if (still.dev?.[key]?.length) dev[key] = [...still.dev[key]];
  }
  if (Object.keys(dev).length) out.dev = dev;
  return out;
}

/** Whether the dev sweep is on, and how many files it would write. */
export function stillSweep(still) {                                    // DEV
  const lengths = still.dev?.lengths?.length || 1;
  const indices = still.dev?.indices?.length || 1;
  const vaes = still.dev?.vaes?.length || 1;
  return { on: lengths * indices * vaes > 1, passes: lengths, images: lengths * indices * vaes };
}

export function parsePreStage(raw) {
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      const state = { ...emptyPreStage(), ...parsed };
      if (!PRESTAGE_ARCHES.includes(state.arch)) state.arch = "krea2";
      if (typeof state.prompt !== "string") state.prompt = "";
      state.output_prefix = parsePrefix(state.output_prefix, IMAGE_PREFIX);
      if (!Array.isArray(state.refs)) state.refs = [];
      state.refs = state.refs
        .filter((ref) => ref && typeof ref.filename === "string")
        .map((ref) => ({ ...ref, kind: ["image", "video", "audio"].includes(ref.kind) ? ref.kind : "image" }))
        // Only the image models cap by count alone; H3's limits are per kind
        // and are `stillRefRoom`'s, enforced where references are added.
        .slice(0, isStill(state) ? MAX_REF_FILES : prestageMaxRefs(state));
      if (!state.end || typeof state.end !== "object" || !state.end.filename) state.end = null;
      state.minimax = parseStill(state.minimax);
      // Both arches' aspect lists share six labels and the image models add two
      // of their own; a blob that switched arch by hand can be carrying one the
      // other side has never heard of.
      if (!prestageAspects(state).some(([label]) => label === state.aspect)) state.aspect = "16:9";
      // A blob that never said lands on this arch's own default rather than on
      // the other's clamped into range — 1024 is the image models' comfortable
      // size and 768 is what H3 was trained at.
      state.short_edge = parsed.short_edge === undefined
        ? prestageEdges(state).default
        : clampPreStageEdge(state, state.short_edge);
      if (!Array.isArray(state.loras)) state.loras = [];
      // UI-only, never serialized: the LoRA manager and `promptTriggers` walk
      // the video-state accessors (`checkpoint`, `references`), which want
      // these two fields to exist even though an image render has neither.
      state.assets = [];
      state.checkpoint = "auto";
      if (!state.init || typeof state.init !== "object" || !state.init.filename) state.init = null;
      if (state.init) {
        const denoise = Number(state.init.denoise);
        state.init.denoise = Number.isFinite(denoise)
          ? Math.min(1, Math.max(PRESTAGE_MIN_DENOISE, denoise)) : PRESTAGE_DEFAULT_DENOISE;
      }
      if (!PRESTAGE_IDEOGRAM_QUALITIES.includes(state.quality)) state.quality = "default";
      const turbo = state.turbo && typeof state.turbo === "object" ? state.turbo : {};
      state.turbo = {
        on: turbo.on === true,
        quality: PRESTAGE_TURBO_QUALITIES.includes(turbo.quality) ? turbo.quality : "good",
        saved: turbo.saved && typeof turbo.saved === "object" ? { ...turbo.saved } : null,
      };
      const models = state.models && typeof state.models === "object" ? state.models : {};
      state.models = emptyPreStageModels();
      for (const arch of PRESTAGE_ARCHES) {
        const side = models[arch];
        if (!side || typeof side !== "object") continue;
        for (const field of PRESTAGE_FIELDS[arch]) {
          if (typeof side[field] === "string" && side[field].trim()) {
            state.models[arch][field] = side[field].trim();
          }
        }
      }
      if (MODEL_DTYPES.includes(models.dtype)) state.models.dtype = models.dtype;
      return state;
    }
  } catch {
    // Same reasoning as parseState: an unreadable blob leaves the node usable.
  }
  return emptyPreStage();
}

export function serializePreStage(state) {
  const models = {};
  for (const arch of PRESTAGE_ARCHES) {
    const side = {};
    for (const field of PRESTAGE_FIELDS[arch]) {
      if (state.models?.[arch]?.[field]) side[field] = state.models[arch][field];
    }
    if (Object.keys(side).length) models[arch] = side;
  }
  if (state.models?.dtype && state.models.dtype !== "default") models.dtype = state.models.dtype;
  return JSON.stringify({
    version: 1,
    arch: state.arch,
    prompt: state.prompt ?? "",
    aspect: state.aspect,
    short_edge: state.short_edge,
    ...(state.init ? { init: { filename: state.init.filename, denoise: round2(state.init.denoise) } } : {}),
    ...(state.end ? { end: { filename: state.end.filename } } : {}),
    ...(state.refs.length ? {
      refs: state.refs.map((r) => ({
        handle: r.handle, filename: r.filename,
        ...(r.kind && r.kind !== "image" ? { kind: r.kind } : {}),
        ...(r.track && r.track !== DEFAULT_TRACK ? { track: r.track } : {}),
        ...(r.ref_size ? { ref_size: r.ref_size } : {}),
        ...(r.takes && r.takes !== "full" ? { takes: r.takes } : {}),
        ...(r.trim ? { trim: r.trim } : {}),
      })),
    } : {}),
    loras: serializeLoras(state.loras),
    ...(state.turbo.on || state.turbo.saved
      ? { turbo: { on: state.turbo.on, quality: state.turbo.quality,
                   ...(state.turbo.saved ? { saved: { ...state.turbo.saved } } : {}) } }
      : {}),
    ...(state.quality !== "default" ? { quality: state.quality } : {}),
    ...(isStill(state) || state.minimax?.dev?.lengths?.length
      ? { minimax: serializeStill(parseStill(state.minimax)) } : {}),
    ...serializePrefix(state.output_prefix, IMAGE_PREFIX),
    ...(Object.keys(models).length ? { models } : {}),
    ...(state.peer != null ? { peer: state.peer } : {}),
  }, null, 2);
}

/** Which directory each pre-stage weight field browses. Mirrors
 *  `render_image.FOLDERS` plus the video fields the H3 branch reuses. */
export function prestageFileLists(byFolder) {
  const dits = byFolder?.diffusion_models ?? [];
  return {
    model: dits, turbo_model: dits, uncond_model: dits, fl2va: dits, ref2va: dits,
    clip: byFolder?.text_encoders ?? [],
    vae: byFolder?.vae ?? [],
    preview: byFolder?.vae_approx ?? [],
  };
}

/** Fill empty weight fields from unambiguous filename matches — the same
 *  service `guessModels` does for the video nodes, for the same first-run. */
export function guessPreStageModels(models, byFolder) {
  const lists = prestageFileLists(byFolder);
  let changed = false;
  for (const arch of PRESTAGE_ARCHES) {
    for (const field of PRESTAGE_FIELDS[arch]) {
      if (models[arch][field]) continue;
      const needles = PRESTAGE_HINTS[arch][field];
      let matched = lists[field].filter((name) =>
        needles.some((needle) => name.toLowerCase().includes(needle)));
      // RAW vs Turbo vs unconditional share stems; whichever says the more
      // specific word belongs to the more specific field.
      if (field === "model" && arch === "ideogram4") {
        matched = matched.filter((name) => !name.toLowerCase().includes("unconditional"));
      }
      if (matched.length !== 1) continue;
      models[arch][field] = matched[0];
      changed = true;
    }
  }
  return changed;
}

/** The resolved image canvas, mirroring compile_image.resolve_canvas: /16 grid,
 *  2048² area cap, and the aspect taken from the init image when there is one. */
export function resolvedPreStage(state, initSize = null) {
  let ratio = prestageAspects(state).find(([label]) => label === state.aspect)?.[1] ?? 16 / 9;
  let fromImage = false;
  if (state.init && initSize?.width && initSize?.height) {
    ratio = initSize.width / initSize.height;
    fromImage = true;
  }
  // The H3 branch generates a video frame, so its geometry is the video
  // canvas's — same /32 grid, same 768*1344 area cap that scales with the
  // slider, same envelope. canvas.js owns it; this only routes to it.
  if (isStill(state)) {
    const [clamped] = clampRatio(ratio);
    const [width, height] = resolveCanvas(clamped, clampPreStageEdge(state, state.short_edge));
    return { width, height, ratio: clamped, fromImage };
  }
  ratio = Math.min(PRESTAGE_MAX_RATIO, Math.max(PRESTAGE_MIN_RATIO, ratio));
  const edge = Math.max(PRESTAGE_MIN_EDGE, Math.min(PRESTAGE_MAX_EDGE, Math.round(state.short_edge)));

  let width, height;
  if (ratio >= 1) { width = edge * ratio; height = edge; }
  else { width = edge; height = edge / ratio; }
  if (width * height > PRESTAGE_MAX_PIXELS) {
    const scale = Math.sqrt(PRESTAGE_MAX_PIXELS / (width * height));
    width *= scale;
    height *= scale;
  }
  // The long side is capped too — 2048 is the models' per-axis ceiling, and a
  // 3:1 sheet at a big short edge would sail past it inside the area cap.
  if (Math.max(width, height) > PRESTAGE_MAX_EDGE) {
    const scale = PRESTAGE_MAX_EDGE / Math.max(width, height);
    width *= scale;
    height *= scale;
  }
  const snap16 = (v) => Math.max(PRESTAGE_CANVAS_MULTIPLE,
    Math.floor(v / PRESTAGE_CANVAS_MULTIPLE + 0.5) * PRESTAGE_CANVAS_MULTIPLE);
  width = snap16(width);
  height = snap16(height);
  while (width * height > PRESTAGE_MAX_PIXELS && Math.max(width, height) > PRESTAGE_CANVAS_MULTIPLE) {
    if (width >= height) width -= PRESTAGE_CANVAS_MULTIPLE;
    else height -= PRESTAGE_CANVAS_MULTIPLE;
  }
  return { width, height, ratio, fromImage };
}

/** Next free ref handle: img-1, vid-1, aud-1, ... — the same identity scheme the
 *  video assets use, so the tag hues carry over and a prompt cites a still's
 *  references with the handles it would cite a shot's with. */
export function nextPreStageHandle(state, kind = "image") {
  const prefix = PREFIX[kind] ?? "img";
  const taken = new Set(state.refs.map((r) => r.handle));
  for (let n = 1; ; n += 1) {
    const handle = `${prefix}-${n}`;
    if (!taken.has(handle)) return handle;
  }
}

/** Which of a fresh pre-stage's weight fields are still empty, for the pill's
 *  warning — clip, vae and whichever DiT the turbo pill selects. */
export function missingPreStageModels(state) {
  const side = state.models[state.arch] ?? {};
  if (isStill(state)) {
    // Only the checkpoint this still routes to, the same way the video node
    // only asks for the one its mode reaches for, and the audio VAE only when
    // something attached cites sound. The preview decoder is never required:
    // without it the body previews in latent2rgb.
    return [stillCheckpoint(state), "clip", "vae",
            ...(stillNeedsAudio(state) ? ["audio_vae"] : [])]
      .filter((field) => !side[field]);
  }
  const dit = state.arch === "krea2" && state.turbo.on ? "turbo_model" : "model";
  return [dit, "clip", "vae"].filter((field) => !side[field]);
}

/** Whether anything attached to this still will be encoded as sound — a
 *  reference audio clip, or a reference video taken with its soundtrack. What
 *  decides the audio VAE is required at all. Mirrors compile_still.needs_audio. */
export function stillNeedsAudio(state) {
  return counts(stillAssets(state)).audio > 0;
}

/** A still's references in the shape the video-side accessors read.
 *
 *  The pre-stage keeps references in `refs` and keyframes in fields of their
 *  own, where the Creator keeps one flat `assets` list. Rather than a second
 *  implementation of the slot arithmetic, the H3 branch hands that list through
 *  this view — so `counts`, `capacity` and `overflow` answer for a still with
 *  the numbers they answer for a shot with. */
export const stillAssets = (state) => ({
  assets: (state.refs ?? []).map((ref) => ({ role: "reference", kind: "image", ...ref })),
});

/** How many slots a kind has left on a still, for the picker's counters. */
export const stillCapacity = (state, kind) => capacity(stillAssets(state), kind);

/** Why a still's references would not compile, or null. */
export const stillOverflow = (state) => overflow(stillAssets(state));

/** Why an attachment is refused on a still, in the video node's own words —
 *  the rule is the video node's rule: frames run on FL2VA, references on
 *  Ref2VA, and one generation runs on one checkpoint. */
export function stillBlockedReason(state, action) {
  if (action === "reference" && (state.init || state.end)) {
    return "Remove the start/end frame first — references use the Ref2VA checkpoint, frames use FL2VA.";
  }
  if ((action === "first_frame" || action === "last_frame") && (state.refs ?? []).length) {
    return "Remove the references first — start/end frames use the FL2VA checkpoint, references use Ref2VA.";
  }
  return null;
}

/** Which H3 mode a still compiles to, for the badge. The same derivation
 *  `compile._derive_mode` makes, read off the pre-stage's own two keyframe
 *  fields — a still is one of these requests, so it is one of these modes. */
export function stillMode(state) {
  if ((state.refs ?? []).length) return "REF2VA";
  if (state.init && state.end) return "FL2VA";
  if (state.init) return "I2VA";
  if (state.end) return "L2VA";
  return "T2VA";
}

/** Which H3 checkpoint a still routes to. Mirrors `compile_still._checkpoint`:
 *  references need Ref2VA, everything else runs on FL2VA, and an explicit pin
 *  wins where it is legal. */
export function stillCheckpoint(state) {
  const pin = state.minimax?.checkpoint ?? "auto";
  if (pin !== "auto") return pin;
  return state.refs.length ? "ref2va" : "fl2va";
}

/** Which of the eight identity hues (--mmc-tag-0..7) a handle wears, everywhere
 *  it appears — asset bar, prompt chip, mention menu. Derived from the handle
 *  alone so it needs no stored state and survives reloads; handles are stable
 *  across deletions, so img-2 keeps its hue after img-1 is removed. The kind
 *  offset staggers img-1 / vid-1 / aud-1 onto different hues. */
const TAG_OFFSET = { img: 0, vid: 1, aud: 2 };
export function tagIndex(handle) {
  const match = /^([A-Za-z]+)-(\d+)$/.exec(handle || "");
  if (!match) return 0;
  return (Number(match[2]) - 1 + (TAG_OFFSET[match[1]] ?? 0)) % 8;
}

/** Next free @handle for a kind: img-1, img-2, ... Stable across deletions. */
export function nextHandle(state, kind) {
  const prefix = PREFIX[kind];
  const taken = new Set(state.assets.map((a) => a.handle));
  for (let n = 1; ; n += 1) {
    const handle = `${prefix}-${n}`;
    if (!taken.has(handle)) return handle;
  }
}

// ---- loras ------------------------------------------------------------------

const round2 = (value) => Math.round((Number(value) || 0) * 100) / 100;

/** The checkpoints an entry claims. Missing or nonsense means both. */
export function loraModes(entry) {
  const claimed = (entry.modes || []).filter((m) => CHECKPOINTS.includes(m));
  return claimed.length ? claimed : [...CHECKPOINTS];
}

export const claimsBoth = (entry) => loraModes(entry).length === CHECKPOINTS.length;

/** The checkpoint the mode implies, before any pin. */
export const derivedCheckpoint = (state) => (hasReferences(state) ? "ref2va" : "fl2va");

/** Which checkpoint this state routes to, and so which LoRAs will apply.
 *  Mirrors `compile._resolve_checkpoint`. */
export function checkpoint(state) {
  const pin = state.checkpoint;
  return !pin || pin === "auto" ? derivedCheckpoint(state) : pin;
}

/** Whether the routing is the user's choice rather than the mode's. */
export const checkpointPinned = (state) => canPinCheckpoint(state) && state.checkpoint !== "auto";

/** A pin only means anything where there is a choice to make. References are
 *  encoded *for* Ref2VA — no other weights can read the blocks — so the
 *  reference modes have none. */
export const canPinCheckpoint = (state) => derivedCheckpoint(state) === "fl2va";

/** Drop a pin the mode has moved out from under. Attaching a reference turns a
 *  frame generation into a reference one, and compile.py rejects an fl2va pin on
 *  that outright; clearing it here keeps the blob queueable. */
export function normalizeCheckpoint(state) {
  if (!canPinCheckpoint(state)) state.checkpoint = "auto";
}

/** The refiner's prose for a state, or "" when there is none in play. Mirrors
 *  `compile.refined_body` — same field, same meaning for `enabled`. */
export function refinedBody(state) {
  const refined = state?.refined;
  if (!refined || refined.enabled === false) return "";
  return (refined.body || "").trim();
}

export const findLora = (state, name) => state.loras.find((l) => l.name === name) || null;

/** Applied to the routed checkpoint on the next queue, in patch order. */
export function activeLoras(state) {
  const target = checkpoint(state);
  return state.loras.filter((entry) =>
    entry.enabled !== false && loraModes(entry).includes(target) && round2(entry.strength) !== 0);
}

/** `triggers` seeds from the sidecar's trained words, which is the only moment
 *  the sidecar is consulted — from here on the entry owns its own list, so
 *  dropping a word or adding one of your own are the same edit. */
export function addLora(state, name, triggers = []) {
  if (findLora(state, name)) return null;
  const entry = {
    name, strength: DEFAULT_STRENGTH, enabled: true,
    modes: [...CHECKPOINTS], triggers: [...triggers],
  };
  state.loras.push(entry);
  return entry;
}

/**
 * The words compile.py will put in front of the prompt. Mirrors
 * `compile.collect_triggers` — same walk, same case-insensitive dedup — for the
 * same reason canvas.js mirrors canvas.py: the node has to show the composed
 * prompt before anything is queued. compile.py stays authoritative.
 */
export function promptTriggers(state) {
  const out = [];
  const seen = new Set();
  for (const entry of activeLoras(state)) {
    for (const raw of entry.triggers || []) {
      const word = String(raw).trim();
      if (!word || seen.has(word.toLowerCase())) continue;
      seen.add(word.toLowerCase());
      out.push(word);
    }
  }
  return out;
}

/**
 * The checkpoints a timeline's segments actually route to, in a fixed order.
 *
 * A global LoRA is patched onto every segment, and the segments need not agree:
 * a reference shot runs on Ref2VA and a text one on FL2VA in the same piece. So
 * "will this LoRA do anything" is a question about a set rather than about one
 * checkpoint, which is what the manager is handed instead of `checkpoint()`.
 */
export function timelineCheckpoints(timeline) {
  // One pass, one set of weights: the shots are merged into a single request, so
  // a reference anywhere makes the whole thing Ref2VA.
  if (isSingle(timeline)) {
    if (timeline.segments.some(hasReferences)) return ["ref2va"];
    const pin = timeline.segments.map((s) => s.checkpoint).find((c) => c && c !== "auto");
    return [pin || "fl2va"];
  }
  const routed = new Set(timeline.segments.map((segment) => checkpoint(segment)));
  return CHECKPOINTS.filter((name) => routed.has(name));
}

/**
 * The mode the merged one-pass request will compile to.
 *
 * `mode()` answers it for one segment; here the shots are one generation, so the
 * question is asked of all of them at once — a reference anywhere makes it
 * REF2VA, and the keyframes are the first shot's start and the last shot's end.
 */
export function singleMode(timeline) {
  const shots = timeline.segments;
  if (shots.some(hasReferences)) return "REF2VA";
  const first = frameAsset(shots[0] ?? { assets: [] }, "first_frame");
  const last = frameAsset(shots[shots.length - 1] ?? { assets: [] }, "last_frame");
  if (first && last) return "FL2VA";
  if (first) return "I2VA";
  if (last) return "L2VA";
  return "T2VA";
}

/**
 * Why this timeline could not be rendered in one pass, or null.
 *
 * Mirrors `compile.single_payload`'s refusals so a timeline switched over to one
 * pass says what is wrong with it while the shots are still in front of you,
 * rather than at queue time. compile.py stays authoritative — this only has to
 * catch the structural ones, which are the ones a chained timeline routinely has.
 */
export function singleProblem(timeline) {
  const shots = timeline.segments;
  const globalPrompt = (timeline.prompt || "").trim();

  for (const [index, shot] of shots.entries()) {
    // A refined shot has prose whatever its prompt box holds — the rewrite
    // replaces it at compile time — so an empty box is only empty if nothing
    // was written for it at all.
    const text = (refinedBody(shot) || shot.prompt || "").trim();
    if (!text && !(index === 0 && globalPrompt)) {
      return `Shot ${index + 1} has no prompt. In one pass the shots are one description `
           + `with cuts in it, so an empty one leaves a cut with nothing on the far side.`;
    }
    if (frameAsset(shot, "first_frame") && index !== 0) {
      return `Shot ${index + 1} has a start frame, but one pass opens on shot 1.`;
    }
    if (frameAsset(shot, "last_frame") && index !== shots.length - 1) {
      return `Shot ${index + 1} has an end frame, but one pass ends on shot ${shots.length}.`;
    }
  }

  const withRefs = shots.findIndex(hasReferences);
  const withFrames = shots.findIndex((s) => frameAsset(s, "first_frame") || frameAsset(s, "last_frame"));
  if (withRefs >= 0 && withFrames >= 0) {
    return `Shot ${withFrames + 1} has a start/end frame and shot ${withRefs + 1} has references. `
         + `Those are different checkpoints and one pass runs on one of them.`;
  }

  for (const [key, what] of [["checkpoint", "the checkpoint"], ["soundscape", "the soundscape"],
                             ["music", "the music"]]) {
    const seen = new Set(shots
      .map((shot) => (key === "checkpoint" ? shot.checkpoint : (shot[key] || "").trim()))
      .filter((value) => value && value !== "auto"));
    if (key !== "checkpoint" && (timeline[key] || "").trim()) seen.add((timeline[key] || "").trim());
    if (seen.size > 1) return `The shots disagree about ${what}. One pass has only one.`;
  }
  return null;
}

/** The global LoRAs that will be patched onto at least one segment. */
export function activeGlobalLoras(timeline) {
  const targets = timelineCheckpoints(timeline);
  return (timeline.loras ?? []).filter((entry) =>
    entry.enabled !== false && round2(entry.strength) !== 0
    && loraModes(entry).some((mode) => targets.includes(mode)));
}

export function removeLora(state, name) {
  state.loras = state.loras.filter((entry) => entry.name !== name);
}

// ---- assets -----------------------------------------------------------------

export const references = (state) => state.assets.filter((a) => a.role === "reference");
export const refImages = (state) => references(state).filter((a) => a.kind === "image");
// The same bucketing compile.py does: a video kept for its soundtrack alone is
// an audio reference, and never a video one.
export const soundOnly = (asset) => asset.kind === "video" && asset.track === "sound";
export const refVideos = (state) => references(state).filter((a) => a.kind === "video" && !soundOnly(a));
export const refAudios = (state) => references(state).filter((a) => a.kind === "audio" || soundOnly(a));
export const frameAsset = (state, role) => state.assets.find((a) => a.role === role) || null;

export function hasReferences(state) {
  return references(state).length > 0;
}

/** A timeline segment that starts from the previous segment's last frame. */
export const continues = (state) => state.continue === true;

/** ...and one whose sound carries on from it. Not implied by the above. */
export const continuesAudio = (state) => state.continue_audio === true;

/** Continuing *is* having a start frame — it is the previous segment's last one
 *  — which is why it locks out references exactly as a real keyframe does. */
export function hasFrames(state) {
  return continues(state) || !!(frameAsset(state, "first_frame") || frameAsset(state, "last_frame"));
}

export function mode(state) {
  if (hasReferences(state)) return "REF2VA";
  const first = frameAsset(state, "first_frame");
  const last = frameAsset(state, "last_frame");
  if (continues(state)) return last ? "FL2VA" : "I2VA";
  if (first && last) return "FL2VA";
  if (first) return "I2VA";
  if (last) return "L2VA";
  return "T2VA";
}

/** What each bucket currently holds. A video with its sound on occupies both a
 *  video slot and an audio one, which is the rule compile.py enforces. */
function counts(state) {
  const images = refImages(state).length;
  const videos = refVideos(state).length;
  const audios = refAudios(state).length
    + refVideos(state).filter((v) => v.track === "picture+sound").length;
  return { image: images, video: videos, audio: audios, files: images + videos + audios };
}

/** How many slots a kind has left, for the picker's "n / 9 slots filled". */
export function capacity(state, kind) {
  const used = counts(state);
  const max = { image: MAX_REF_IMAGES, video: MAX_REF_VIDEOS, audio: MAX_REF_AUDIOS }[kind];
  return { used: used[kind], max, filesLeft: MAX_REF_FILES - used.files };
}

/**
 * Why the references as they now stand would not compile, or null. The same
 * limits as `_derive_mode`, checked after a change has been applied, so a switch
 * that would fail at queue time can be handed back while it is still reversible.
 */
export function overflow(state) {
  const used = counts(state);
  if (used.image > MAX_REF_IMAGES) return `At most ${MAX_REF_IMAGES} reference images.`;
  if (used.video > MAX_REF_VIDEOS) return `At most ${MAX_REF_VIDEOS} reference videos.`;
  if (used.audio > MAX_REF_AUDIOS) {
    return `At most ${MAX_REF_AUDIOS} reference audio clips, counting video soundtracks.`;
  }
  if (used.files > MAX_REF_FILES) return `At most ${MAX_REF_FILES} reference files in total.`;
  return null;
}

/** The resolved geometry and duration shown on the pills. */
export function resolved(state, keyframeSize = null) {
  const frames = framesForSeconds(state.duration_s);
  let ratio = ASPECT_PRESETS.find(([label]) => label === state.aspect)?.[1] ?? 16 / 9;
  let fromImage = false;
  if (keyframeSize && keyframeSize.width && keyframeSize.height) {
    ratio = keyframeSize.width / keyframeSize.height;
    fromImage = true;
  }
  const [width, height] = resolveCanvas(ratio, state.short_edge);
  return { frames, seconds: secondsForFrames(frames), width, height, ratio, fromImage };
}

/**
 * Why the UI blocks an action, or null. Frames and references need different
 * checkpoints and cannot be combined in one pass, so each side locks the other
 * out rather than letting the backend reject the graph at queue time.
 */
export function blockedReason(state, action) {
  if (action === "reference" && continues(state)) {
    return "This segment continues from the previous one, which is a keyframe generation on FL2VA — "
         + "references need Ref2VA. Turn continuation off to attach references.";
  }
  if (action === "reference" && hasFrames(state)) {
    return "Remove the start/end frame first — references use the Ref2VA checkpoint, frames use FL2VA.";
  }
  if (action === "first_frame" && continues(state)) {
    return "This segment's start frame is the previous segment's last frame. Turn continuation off to choose one.";
  }
  if ((action === "first_frame" || action === "last_frame") && hasReferences(state)) {
    return "Remove the references first — start/end frames use the FL2VA checkpoint, references use Ref2VA.";
  }
  // Turning continuation *on* is refused by the same rule read the other way.
  if (action === "continue" && hasReferences(state)) {
    return "Remove this segment's references first — continuing from the last frame is an FL2VA "
         + "generation and references need Ref2VA.";
  }
  if (action === "continue_audio" && hasReferences(state)) {
    return "Remove this segment's references first — the reference list owns the audio slots, "
         + "so the inherited soundtrack has nowhere to go.";
  }
  if (action === "continue" && frameAsset(state, "first_frame")) {
    return "Remove this segment's start frame first — continuing would replace it with the previous "
         + "segment's last frame.";
  }
  return null;
}
