// The asset picker modal: tabs, search, upload, a grid of the input folder, and
// a slot counter that stops you selecting more than the model accepts.

import { el, ICONS, svg, icon, mountOverlay, dismissable } from "./dom.js";
import { listAssets, viewUrl, thumbUrl, upload, moveAsset, deleteAsset,
         loadPickerPrefs, savePickerPrefs } from "./api.js";
import { openTrim, trimLabel } from "./trim.js";

// "renders" is a tab, not a kind: it browses the output folder instead of a
// slice of the input one, and the files under it keep their own kinds — a
// picked render is attached as the video it is.
//
// Everything else about it is the same folder: shelves, stars, organize mode,
// drag-to-move and delete all work there exactly as they do on the input tabs,
// because a finished render needs filing more than an uploaded clip does. The
// only thing the gallery cannot do is take an upload, since renders arrive by
// being rendered. `activeAssets` is what keeps that one implementation: it is
// the list the current tab is looking at, and every organize path goes through
// it rather than reaching for `this.assets`.
const KIND_LABEL = { image: "Image", video: "Video", audio: "Audio", renders: "Renders" };
const ACCEPT = { image: "image/*", video: "video/*", audio: "audio/*" };
// What a configured video cell says about itself, short enough for the badge.
const TRACK_BADGE = { "picture+sound": "sound", "picture": "silent", "sound": "sound only" };

/**
 * @param {object} options
 * @param {string[]} options.kinds        tabs to show, in order
 * @param {string} options.kind           tab to open on
 * @param {(kind:string)=>{used:number,max:number,filesLeft:number}} options.capacity
 * @param {boolean} options.single        pick exactly one (start/end frame)
 * @param {boolean} options.viewOnly      browse and play, select nothing — the
 *   Timeline's gallery, which has no segment to attach a pick to
 * @returns {Promise<Array|null>} chosen assets, or null if cancelled
 */
export function openPicker(options) {
  return new Promise((resolve) => {
    new Picker(options, resolve).mount();
  });
}

class Picker {
  constructor(options, resolve) {
    this.options = options;
    this.resolve = resolve;
    this.kind = options.kind || options.kinds[0];
    this.query = "";
    this.selected = [];   // asset rows, in click order
    this.assets = [];
    this.renders = [];    // the output folder, only fetched when the tab exists
    this.loaded = false;
    // Which shelf the grid shows: "all", "fav", or an input subfolder. Shelves
    // are shared across tabs — a folder is a place, not a kind.
    this.shelf = "all";
    this.prefs = { favorites: [], folders: [] };
    // path -> {trim, track}. Set only for files the user opened the segment
    // editor on; everything else is attached whole and silent, as before.
    this.settings = new Map();
    // Organize mode: clicks mark files for moving or deleting instead of
    // picking them. Its own list, because `selected` means "attach to the
    // node" and is bounded by slots — marking is bounded by nothing.
    this.organize = false;
    this.marked = [];   // paths, in click order
  }

  mount() {
    this.grid = el("div", { class: "mmc-grid" });
    this.search = el("input", {
      class: "mmc-search",
      type: "search",
      placeholder: "Search...",
      oninput: (event) => { this.query = event.target.value.toLowerCase(); this.renderGrid(); },
    });

    this.tabs = this.options.kinds.map((kind) =>
      el("button", {
        class: "mmc-tab",
        "aria-selected": kind === this.kind,
        onclick: () => this.selectTab(kind),
        text: KIND_LABEL[kind],
      }));

    this.slots = el("span", { class: "mmc-slots" });
    // Children come from renderFoot: picking and organizing want different rows.
    this.foot = el("div", { class: "mmc-modal-foot" });

    this.shelfRow = el("div", { class: "mmc-shelves" });
    this.modal = el("div", { class: "mmc-modal" }, [
      el("div", { class: "mmc-modal-head" }, [
        ...this.tabs,
        el("button", { class: "mmc-close", text: "✕", onclick: () => this.close(null) }),
      ]),
      el("div", { class: "mmc-modal-bar" }, [
        this.search,
        el("button", {
          class: "mmc-organize",
          "aria-pressed": false,
          title: "Select files to move between folders or delete",
          onclick: () => this.setOrganize(!this.organize),
        }, [icon("folder", 14), el("span", { text: "Organize" })]),
        el("button", { class: "mmc-upload", text: `+  Upload ${KIND_LABEL[this.kind].toLowerCase()}`, onclick: () => this.pickFile() }),
      ]),
      this.shelfRow,
      this.grid,
      this.foot,
    ]);
    this.uploadButton = this.modal.querySelector(".mmc-upload");
    this.organizeButton = this.modal.querySelector(".mmc-organize");
    if (this.kind === "renders") this.uploadButton.style.display = "none";
    this.modal.style.position = "relative";

    this.overlay = el("div", {
      class: "mmc-overlay",
      onpointerdown: (event) => { if (event.target === this.overlay) this.close(null); },
    }, [this.modal]);

    this.unmount = mountOverlay(this.overlay, () => this.close(null));

    this.renderFoot();
    this.load();
    setTimeout(() => this.search.focus(), 30);
  }

  async load({ force = false } = {}) {
    try {
      this.assets = await listAssets({ force });
      if (this.options.kinds.includes("renders")) {
        this.renders = await listAssets({ force, root: "output" });
      }
      this.prefs = await loadPickerPrefs();
      // A mark on a file the listing no longer has is a mark on nothing.
      this.marked = this.marked.filter((p) => this.activeAssets().some((a) => a.path === p));
      this.loaded = true;
    } catch (error) {
      this.assets = [];
      this.renders = [];
      this.loaded = true;
      this.loadError = error.message;
    }
    this.renderShelves();
    this.renderGrid();
  }

  selectTab(kind) {
    if (kind === this.kind) return;
    const previous = this.kind;
    this.kind = kind;
    // Selections do not survive a tab change: they go into different slots.
    this.selected = [];
    for (const tab of this.tabs) tab.setAttribute("aria-selected", String(tab.textContent === KIND_LABEL[kind]));
    // Nothing uploads into the output folder: renders arrive by being rendered.
    // Organizing them is another matter — see the note at the top of the file.
    this.uploadButton.style.display = kind === "renders" ? "none" : "";
    if (kind !== "renders") this.uploadButton.textContent = `+  Upload ${KIND_LABEL[kind].toLowerCase()}`;
    // Shelves are shared between the input tabs — a folder is a place, not a
    // kind — but the output folder is a different place, so crossing that line
    // drops back to "all" rather than selecting a shelf that is not there.
    if ((kind === "renders") !== (previous === "renders")) {
      this.shelf = "all";
      this.marked = [];
    }
    this.renderShelves();
    this.renderGrid();
    this.renderFoot();
  }

  // ---- shelves -------------------------------------------------------------

  isFav(path) {
    return this.prefs.favorites.includes(path);
  }

  toggleFav(asset) {
    const favorites = this.isFav(asset.path)
      ? this.prefs.favorites.filter((p) => p !== asset.path)
      : [...this.prefs.favorites, asset.path];
    this.prefs = { ...this.prefs, favorites };
    savePickerPrefs(this.prefs);
    this.renderShelves();
    this.renderGrid();
  }

  /** The listing the current tab is browsing. The one place that knows the
   *  renders tab reads a different folder; everything organize-related goes
   *  through it, which is why there is only one implementation of any of it. */
  activeAssets() {
    return this.kind === "renders" ? this.renders : this.assets;
  }

  /** Which hand-made shelf names belong to the folder being browsed. Two lists
   *  because they are two folders: a shelf typed while filing renders should
   *  not appear as an empty chip over the input folder, where nothing can ever
   *  land on it. `folders` keeps its name so prefs saved before the gallery
   *  could be organized load unchanged. */
  folderKey() {
    return this.kind === "renders" ? "renderFolders" : "folders";
  }

  /** Every place a file can live: real subfolders seen in the listing (any
   *  kind — a folder is shared) plus shelves made by hand that are still
   *  empty. Sorted; nested paths are simply their own shelves. */
  folders() {
    const seen = new Set(this.prefs[this.folderKey()]);
    for (const asset of this.activeAssets()) if (asset.subfolder) seen.add(asset.subfolder);
    return [...seen].sort((a, b) => a.localeCompare(b));
  }

  setShelf(shelf) {
    this.shelf = shelf;
    this.renderShelves();
    this.renderGrid();
  }

  renderShelves() {
    if (!this.loaded || this.loadError) {
      this.shelfRow.style.display = "none";
      return;
    }
    this.shelfRow.style.display = "";
    // The renders tab is not a kind, so it counts everything the output folder
    // holds; an input tab counts only its own kind, because that is the slice
    // its grid is showing.
    const scoped = this.kind === "renders"
      ? this.renders
      : this.assets.filter((a) => a.kind === this.kind);
    const count = (test) => scoped.filter(test).length;

    const chip = ({ key, label, iconName, n, droppable }) => {
      const node = el("button", {
        class: "mmc-shelf",
        "aria-selected": this.shelf === key,
        onclick: () => this.setShelf(key),
      }, [
        ...(iconName ? [icon(iconName, 13)] : []),
        el("span", { text: label }),
        ...(n ? [el("span", { class: "mmc-shelf-n", text: String(n) })] : []),
      ]);
      if (droppable) {
        node.addEventListener("dragover", (event) => {
          if (!this.dragging) return;
          event.preventDefault();
          node.classList.add("drop");
        });
        node.addEventListener("dragleave", () => node.classList.remove("drop"));
        node.addEventListener("drop", (event) => {
          event.preventDefault();
          node.classList.remove("drop");
          this.moveTo(key);
        });
      }
      return node;
    };

    this.shelfRow.replaceChildren(
      chip({ key: "all", label: "All", n: count(() => true), droppable: true }),
      chip({ key: "fav", label: "", iconName: "star", n: count((a) => this.isFav(a.path)) }),
      ...this.folders().map((folder) => chip({
        key: folder, label: folder, iconName: "folder",
        n: count((a) => a.subfolder === folder), droppable: true,
      })),
      this.newShelfChip(),
    );
  }

  /** The trailing "+" that flips into a name field. A new shelf is only a
   *  remembered name until a file lands on it — the directory itself is
   *  created by the first upload or move. */
  newShelfChip() {
    const add = el("button", { class: "mmc-shelf mmc-shelf-new", text: "+", title: "New shelf" });
    add.addEventListener("click", () => {
      const field = el("input", {
        class: "mmc-shelf-input", type: "text", placeholder: "shelf name",
        onkeydown: (event) => {
          event.stopPropagation();
          if (event.key === "Escape") this.renderShelves();
          if (event.key !== "Enter") return;
          const name = field.value.trim().replace(/^\/+|\/+$/g, "");
          if (!name || /(^|\/)\.|\\/.test(name)) { this.warn("Shelf names cannot start with a dot."); return; }
          const key = this.folderKey();
          if (!this.prefs[key].includes(name)) {
            this.prefs = { ...this.prefs, [key]: [...this.prefs[key], name] };
            savePickerPrefs(this.prefs);
          }
          this.setShelf(name);
        },
        onblur: () => this.renderShelves(),
      });
      add.replaceWith(field);
      field.focus();
    });
    return add;
  }

  /** Drop a dragged cell onto a shelf. In organize mode a marked cell carries
   *  the whole marked set with it — drag and the Move to… menu are two doors
   *  to the same room. */
  async moveTo(folder) {
    const dragged = this.dragging;
    this.dragging = null;
    if (!dragged) return;
    const batch = this.organize && this.marked.includes(dragged.path)
      ? this.activeAssets().filter((asset) => this.marked.includes(asset.path))
      : [dragged];
    await this.moveMany(batch, folder === "all" ? "" : folder);
  }

  /** Move files into a subfolder of the folder they are in ("" is its root),
   *  carrying each one's
   *  star, segment settings, mark and selection over to its new path. Per-file
   *  failures (a name collision, say) skip that file rather than the batch. */
  async moveMany(batch, target) {
    const failures = [];
    for (const asset of batch) {
      if ((asset.subfolder || "") === target) continue;
      try {
        const path = await moveAsset(asset.path, target);
        const rename = (p) => (p === asset.path ? path : p);
        this.prefs = { ...this.prefs, favorites: this.prefs.favorites.map(rename) };
        if (this.settings.has(asset.path)) {
          this.settings.set(path, this.settings.get(asset.path));
          this.settings.delete(asset.path);
        }
        this.marked = this.marked.map(rename);
        for (const chosen of this.selected) {
          if (chosen.path === asset.path) { chosen.path = path; chosen.subfolder = target; }
        }
      } catch (error) {
        failures.push(`${asset.name}: ${error.message}`);
      }
    }
    savePickerPrefs(this.prefs);
    await this.load({ force: true });
    this.renderFoot();
    if (failures.length) {
      this.warn(failures.length === 1 ? failures[0]
        : `${failures.length} files stayed put — ${failures[0]}`);
    }
  }

  // ---- organize mode -------------------------------------------------------

  setOrganize(on) {
    if (this.organize === on) return;
    this.organize = on;
    this.marked = [];
    this.organizeButton.setAttribute("aria-pressed", String(on));
    this.renderGrid();
    this.renderFoot();
  }

  mark(asset) {
    const at = this.marked.indexOf(asset.path);
    if (at >= 0) this.marked.splice(at, 1);
    else this.marked.push(asset.path);
    this.renderGrid();
    this.renderFoot();
  }

  markedAssets() {
    return this.activeAssets().filter((asset) => this.marked.includes(asset.path));
  }

  /** The Move to… popover: every shelf, the root, and a field for a new one.
   *  Picking a destination moves the whole marked set. */
  moveMenu() {
    const menu = el("div", { class: "mmc-move-menu" });
    const go = (target) => { close(); this.moveMany(this.markedAssets(), target); };
    menu.appendChild(el("button", { class: "mmc-move-opt", onclick: () => go("") },
      [icon("image", 13), el("span", {
        text: this.kind === "renders" ? "Output folder (root)" : "Input folder (root)" })]));
    for (const folder of this.folders()) {
      menu.appendChild(el("button", { class: "mmc-move-opt", onclick: () => go(folder) },
        [icon("folder", 13), el("span", { text: folder })]));
    }
    // The same rules as newShelfChip: a name that needs rewriting is refused.
    const field = el("input", {
      class: "mmc-shelf-input", type: "text", placeholder: "New folder…",
      onkeydown: (event) => {
        event.stopPropagation();
        if (event.key === "Escape") { close(); return; }
        if (event.key !== "Enter") return;
        const name = field.value.trim().replace(/^\/+|\/+$/g, "");
        if (!name || /(^|\/)\.|\\/.test(name)) { this.warn("Folder names cannot start with a dot."); return; }
        go(name);
      },
    });
    menu.appendChild(field);
    this.modal.appendChild(menu);
    const close = dismissable(menu);
  }

  /** Delete is irreversible, so the button asks once: the first press arms it,
   *  the second (within a few seconds) fires. */
  confirmDelete() {
    if (!this.deleteButton || !this.marked.length) return;
    if (!this.deleteArmed) {
      this.deleteArmed = true;
      this.deleteButton.textContent = `Really delete ${this.marked.length}?`;
      this.deleteButton.classList.add("armed");
      this.armTimer = setTimeout(() => this.renderFoot(), 5000);
      return;
    }
    this.deleteMarked();
  }

  async deleteMarked() {
    this.deleteButton.disabled = true;
    this.deleteButton.textContent = "Deleting…";
    const failures = [];
    for (const asset of this.markedAssets()) {
      try {
        await deleteAsset(asset.path);
        this.settings.delete(asset.path);
        this.prefs = { ...this.prefs, favorites: this.prefs.favorites.filter((p) => p !== asset.path) };
        this.selected = this.selected.filter((chosen) => chosen.path !== asset.path);
        this.marked = this.marked.filter((p) => p !== asset.path);
      } catch (error) {
        failures.push(`${asset.name}: ${error.message}`);
      }
    }
    savePickerPrefs(this.prefs);
    await this.load({ force: true });
    this.renderFoot();
    if (failures.length) {
      this.warn(failures.length === 1 ? failures[0]
        : `${failures.length} files not deleted — ${failures[0]}`);
    }
  }

  visible() {
    const onShelf = this.shelf === "all" ? () => true
      : this.shelf === "fav" ? (asset) => this.isFav(asset.path)
        : (asset) => asset.subfolder === this.shelf;
    // "renders" is a tab and not a kind, so it shows every kind the output
    // folder holds — a still and the clip it seeded are both renders.
    const onKind = this.kind === "renders" ? () => true : (asset) => asset.kind === this.kind;
    return this.activeAssets().filter((asset) =>
      onKind(asset) && onShelf(asset)
      && (!this.query || asset.path.toLowerCase().includes(this.query)));
  }

  /** The slot a selection will actually take. A video kept for its soundtrack
   *  alone costs an audio slot and no video one — the same rule state.js and
   *  compile.py bucket by — so the counter has to follow it across tabs. */
  targetKind(asset) {
    return this.settings.get(asset.path)?.track === "sound" ? "audio" : asset.kind;
  }

  claimed(kind) {
    return this.selected.filter((asset) => this.targetKind(asset) === kind).length;
  }

  /**
   * Does what is selected fit in `kind`'s bucket, with `extra` more files on top?
   * `extra: 1` asks whether one more can be added; `0` re-checks a selection that
   * has just moved between buckets.
   */
  fits(kind, extra = 0) {
    if (this.options.single) return this.selected.length + extra <= 1;
    const { used, max, filesLeft } = this.options.capacity(kind);
    // filesLeft is the shared total and reads the same whichever bucket is
    // asked, so every selection counts against it, not just this bucket's.
    return used + this.claimed(kind) + extra <= max && this.selected.length + extra <= filesLeft;
  }

  room(kind) {
    return this.fits(kind, 1);
  }

  renderGrid() {
    this.grid.replaceChildren();
    if (!this.loaded) {
      this.grid.appendChild(el("div", { class: "mmc-empty", text: "Loading…" }));
      return;
    }
    if (this.loadError) {
      this.grid.appendChild(el("div", { class: "mmc-empty", text: `Could not read the input folder: ${this.loadError}` }));
      return;
    }
    const rows = this.visible();
    if (!rows.length) {
      this.grid.appendChild(el("div", {
        class: "mmc-empty",
        text: this.query
          ? `No ${this.kind === "renders" ? "renders" : `${this.kind} files`} matching “${this.query}”.`
          : this.shelf === "fav"
            ? "No favorites yet — hover a file and hit the star."
            : this.shelf !== "all"
              ? this.kind === "renders"
                ? "Nothing on this shelf yet — drag renders here, or point a node's output folder at it."
                : "Nothing on this shelf yet — drag files here, or upload while it is open."
              : this.kind === "renders"
                ? "Nothing in the output folder yet — queue a render."
                : `No ${this.kind} files in the input folder yet — upload one.`,
      }));
      return;
    }
    for (const asset of rows) this.grid.appendChild(this.cell(asset));
  }

  cell(asset) {
    const chosen = this.organize
      ? this.marked.includes(asset.path)
      : this.selected.some((a) => a.path === asset.path);
    // A div rather than a button: the segment badge is itself a button, and a
    // button inside a button is not something the DOM is willing to lay out.
    const cell = el("div", {
      class: "mmc-cell",
      role: "button",
      tabindex: "0",
      "aria-selected": chosen,
      title: `${asset.path} — double-click to view`,
      // Double-clicks are detected by hand: the first click's toggle rebuilds
      // the grid, so the second click lands on a replacement element, and
      // Firefox will not synthesise a dblclick across two different nodes.
      // The second click re-toggles first, so viewing leaves the selection
      // exactly where it stood.
      onclick: () => {
        const now = Date.now();
        const double = this.lastClick
          && this.lastClick.path === asset.path && now - this.lastClick.at < 400;
        this.lastClick = double ? null : { path: asset.path, at: now };
        if (this.organize) {
          this.mark(asset);
          if (double) this.view(asset);
          return;
        }
        if (this.options.viewOnly) {
          if (!double) this.view(asset);
          return;
        }
        this.toggle(asset);
        if (double) this.view(asset);
      },
      onkeydown: (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        if (this.organize) this.mark(asset);
        else this.toggle(asset);
      },
    });

    if (asset.kind === "image") {
      // Filenames reach the DOM only as attributes/text, never as markup.
      cell.appendChild(el("img", { src: viewUrl(asset.path, { preview: true }), loading: "lazy", alt: asset.name }));
    } else if (asset.kind === "video") {
      // A server-decoded still, not a <video>: see thumbUrl. A clip the decoder
      // cannot open answers 404, and the cell falls back to the same icon tile
      // audio uses rather than showing a broken image.
      const thumb = el("img", { src: thumbUrl(asset.path, asset.mtime), loading: "lazy", alt: asset.name });
      thumb.addEventListener("error", () => thumb.replaceWith(this.fallback(asset, "video")));
      cell.appendChild(thumb);
    } else {
      cell.appendChild(this.fallback(asset, "audio"));
    }

    cell.appendChild(el("div", { class: "mmc-check" }));
    // No segment badge while organizing: configuring a segment selects the
    // file for attachment, which is exactly not what a mark means.
    if (asset.kind !== "image" && !this.organize) cell.appendChild(this.badge(asset));
    if (asset.kind !== "audio") cell.appendChild(el("div", { class: "mmc-cell-name", text: asset.name }));

    // Stars and dragging on every tab, renders included: a finished clip is the
    // thing most worth starring, and where a render was *written* is not where
    // it has to stay — the keeper gets dragged out of the dated folder it
    // landed in and onto a shelf of its own.
    const starred = this.isFav(asset.path);
    cell.appendChild(el("button", {
      class: `mmc-cell-star${starred ? " on" : ""}`,
      title: starred ? "Remove from favorites" : "Add to favorites",
      onclick: (event) => { event.stopPropagation(); this.toggleFav(asset); },
    }, [icon("star", 13)]));

    // On the All shelf a file's home is worth a caption; on its own shelf
    // the chip above already says it.
    if (this.shelf === "all" && asset.subfolder) {
      cell.appendChild(el("div", { class: "mmc-cell-home", text: asset.subfolder }));
    }

    // Organizing is dragging: the cell rides to a shelf chip. The chips take
    // it from `this.dragging` — dataTransfer only carries strings.
    cell.draggable = true;
    cell.addEventListener("dragstart", (event) => {
      this.dragging = asset;
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", asset.path);
      this.modal.classList.add("dragging");
    });
    cell.addEventListener("dragend", () => {
      this.dragging = null;
      this.modal.classList.remove("dragging");
    });
    return cell;
  }

  /** The icon tile a cell shows when there is no picture to show: every audio
   *  file, and a video whose first frame could not be decoded. */
  fallback(asset, kind) {
    return el("div", { class: "mmc-cell-fallback" }, [svg(ICONS[kind], 26), el("span", { text: asset.name })]);
  }

  /** The segment/sound button on a video or audio cell. Hidden until the cell is
   *  hovered or selected unless it has something to say, so a grid of untouched
   *  files stays as quiet as it was before. */
  badge(asset) {
    const setting = this.settings.get(asset.path);
    const parts = [];
    if (setting?.trim) parts.push(trimLabel(setting));
    // Only once the editor has been used: until then the track is still the
    // default, and the badge would be claiming a decision nobody made.
    if (setting?.track && asset.kind === "video") parts.push(TRACK_BADGE[setting.track]);
    return el("button", {
      class: `mmc-cell-trim${parts.length ? " set" : ""}`,
      title: asset.kind === "video"
        ? "Use only part of this clip, bring its soundtrack along, or take the sound on its own"
        : "Use only part of this file",
      onclick: (event) => { event.stopPropagation(); this.editSegment(asset); },
    }, [icon("scissors", 12), el("span", { text: parts.join(" · ") || "Segment" })]);
  }

  async editSegment(asset) {
    const setting = this.settings.get(asset.path) || {};
    const result = await openTrim({
      path: asset.path,
      kind: asset.kind,
      trim: setting.trim ?? null,
      // undefined until the editor has been opened once: the track switch shows
      // the default rather than a stale "picture".
      track: setting.track,
      showTrack: asset.kind === "video",
    });
    if (!result) return;
    // Stored even when it matches the default, because "the user looked at this
    // and left the sound off" has to outrank the on-by-default rule.
    this.settings.set(asset.path, result);
    const selected = this.selected.some((a) => a.path === asset.path);
    // Switching to sound-only moves the file between buckets, and the one it
    // lands in may be full. Say so and put the choice back rather than letting a
    // selection through that compile.py would refuse.
    if (selected && !this.fits(this.targetKind(asset))) {
      this.settings.set(asset.path, { ...result, track: setting.track });
      this.renderGrid();   // the segment still changed, even if the track did not
      this.warn(`No ${this.targetKind(asset)} slot left for ${asset.name}.`);
      return;
    }
    // Configuring a file is how you say you want it: select it if it wasn't.
    if (!selected) this.toggle(asset);
    else { this.renderGrid(); this.renderFoot(); }
  }

  /** A transient line in the footer, where the slot counter already is — the
   *  picker has no other place to answer back. */
  warn(message) {
    this.slots.textContent = message;
    this.slots.classList.add("full");
    clearTimeout(this.warnTimer);
    this.warnTimer = setTimeout(() => this.renderFoot(), 4000);
  }

  /** Full size, in an overlay above the modal: a video plays with the
   *  browser's controls and its sound, an image just gets the room the grid
   *  cell could not give it. Opened by double-click on any tab. */
  view(asset) {
    let unmount;
    const media = asset.kind === "audio"
      ? el("audio", { class: "mmc-light-audio", src: viewUrl(asset.path), controls: true, autoplay: true })
      : asset.kind === "video"
        ? el("video", { class: "mmc-light-media", src: viewUrl(asset.path), controls: true, autoplay: true, loop: true })
        : el("img", { class: "mmc-light-media", src: viewUrl(asset.path), alt: asset.name });
    const overlay = el("div", {
      class: "mmc-overlay",
      onpointerdown: (event) => { if (event.target === overlay) unmount(); },
    }, [
      el("div", { class: "mmc-light" }, [media, el("div", { class: "mmc-light-name", text: asset.name })]),
    ]);
    unmount = mountOverlay(overlay, () => unmount());
  }

  toggle(asset) {
    if (this.options.viewOnly) return;
    const at = this.selected.findIndex((a) => a.path === asset.path);
    if (at >= 0) this.selected.splice(at, 1);
    else if (this.options.single) this.selected = [asset];
    else if (this.room(this.targetKind(asset))) this.selected.push(asset);
    else return;  // at capacity: the counter already says why
    this.renderGrid();
    this.renderFoot();
  }

  renderFoot() {
    clearTimeout(this.warnTimer);
    clearTimeout(this.armTimer);
    this.deleteArmed = false;
    // viewOnly has nothing to commit, but organizing still needs its buttons.
    this.foot.style.display = this.options.viewOnly && !this.organize ? "none" : "";

    if (this.organize) {
      this.slots.classList.remove("full");
      this.slots.textContent = this.marked.length
        ? `${this.marked.length} marked`
        : "Click files to mark them";
      this.deleteButton = el("button", {
        class: "mmc-del", text: "Delete", disabled: !this.marked.length,
        onclick: () => this.confirmDelete(),
      });
      this.foot.replaceChildren(
        this.slots,
        el("button", {
          class: "mmc-ghost", text: "Move to…", disabled: !this.marked.length,
          onclick: () => this.moveMenu(),
        }),
        this.deleteButton,
        el("button", { class: "mmc-add", text: "Done", onclick: () => this.setOrganize(false) }),
      );
      return;
    }

    this.deleteButton = null;
    this.addButton = el("button", { class: "mmc-add", text: "Add", onclick: () => this.commit() });
    this.foot.replaceChildren(
      this.slots,
      el("button", { class: "mmc-ghost", text: "Cancel", onclick: () => this.close(null) }),
      this.addButton,
    );
    if (this.options.single) {
      this.slots.textContent = this.selected.length ? "1 selected" : "Pick one";
      this.slots.classList.remove("full");
    } else {
      // The renders tab has no bucket of its own — a picked render is a video
      // and counts where a video counts.
      const bucket = this.kind === "renders" ? "video" : this.kind;
      const { used, max } = this.options.capacity(bucket);
      const filled = used + this.claimed(bucket);
      // A clip taken for its sound alone is not in this tab's bucket, so it is
      // reported against the one it does land in rather than silently omitted.
      const elsewhere = this.selected.length - this.claimed(bucket);
      const audio = this.options.capacity("audio");
      this.slots.textContent = `${filled} / ${max} slots filled`
        + (elsewhere ? ` · ${audio.used + this.claimed("audio")} / ${audio.max} audio` : "");
      this.slots.classList.toggle("full", filled >= max);
    }
    this.addButton.disabled = this.selected.length === 0;
  }

  pickFile() {
    const input = el("input", { type: "file", accept: ACCEPT[this.kind], multiple: !this.options.single });
    input.style.display = "none";
    input.addEventListener("change", async () => {
      const files = Array.from(input.files || []);
      input.remove();
      if (!files.length) return;
      this.uploadButton.disabled = true;
      this.uploadButton.textContent = "Uploading…";
      // An upload lands on the shelf being looked at — that is what makes a
      // shelf a place rather than a filter.
      const into = this.shelf === "all" || this.shelf === "fav" ? "" : this.shelf;
      try {
        for (const file of files) await upload(file, into);
        await this.load({ force: true });
      } catch (error) {
        this.grid.replaceChildren(el("div", { class: "mmc-empty", text: `Upload failed: ${error.message}` }));
      } finally {
        this.uploadButton.disabled = false;
        this.uploadButton.textContent = `+  Upload ${KIND_LABEL[this.kind].toLowerCase()}`;
      }
    });
    document.body.appendChild(input);
    input.click();
  }

  commit() {
    if (!this.selected.length) return;
    this.close(this.selected.map((asset) => ({ ...asset, ...(this.settings.get(asset.path) || {}) })));
  }

  close(result) {
    clearTimeout(this.warnTimer);
    clearTimeout(this.armTimer);
    this.unmount();
    this.resolve(result);
  }
}
