// The settings page: the preferences that belong to this ComfyUI rather than to
// a workflow. Opened from the rail's Settings tool, beside the Gallery.
//
// The line it draws is `settings.py`'s: a workflow says what the piece is, and
// this says how this machine writes it. Nothing here is saved into creator_data,
// so a `.json` shared with someone else renders the same shot at whatever
// quality their own copy of ComfyUI is set to.
//
// Every control writes through the moment it is touched — the same deal the LoRA
// manager has, and for the same reason: a page with a Save button has a state
// where what you see and what is stored disagree, and Done is then two different
// promises. Done here only closes.
//
// The server is the only copy. Nothing is cached between openings: the file can
// be edited by hand, and a page that showed a remembered value would be showing
// something the next render will not use.
//
// Two tabs, because the page now answers two questions that are not the same
// question: how good the file is, and where it goes. Both are this machine's
// rather than the workflow's, which is the only reason they share a page.

import { el, mountOverlay } from "./dom.js";
import { loadSettings, saveSettings } from "./api.js";
import { TOKENS, cleanPrefix, folderOf, stemOf, examplePath } from "./outputs.js";

// libx264's own quality scale: lower is better and bigger, and six points is
// roughly double the file size. Four points on it, because the encoder's full
// 0–51 makes forty useless values as reachable as the four good ones — the size
// claims below all come off that one rule, so the rows cannot drift apart.
//
// `settings.py` decides what is *allowed* (the whole scale, so a hand-edited
// file is honoured); this decides what is *offered*.
const QUALITY = [
  { crf: 28, label: "Draft",
    note: "Smallest files, about half of Standard. Fine for checking timing; "
        + "banding shows up in dark gradients." },
  { crf: 23, label: "Standard",
    note: "What libx264 picks on its own, and what this pack wrote before the "
        + "setting existed." },
  { crf: 18, label: "Fine",
    note: "About twice the size of Standard. Hard to tell from the frames the "
        + "sampler handed over." },
  { crf: 14, label: "Archival",
    note: "About three times the size of Standard. Keeps the grain and fine "
        + "texture H.264 usually eats first." },
];

export function openSettings() {
  return new Promise((resolve) => new SettingsPage(resolve).mount());
}

const TABS = [
  { key: "quality", label: "Quality" },
  { key: "folders", label: "Folders" },
];

class SettingsPage {
  constructor(resolve) {
    this.resolve = resolve;
    this.settings = null;   // until the server answers
    this.problem = null;
    this.tab = TABS[0].key;
  }

  mount() {
    this.body = el("div", { class: "mmc-set-body" });
    this.tabs = TABS.map((tab) => el("button", {
      class: "mmc-tab",
      "aria-selected": tab.key === this.tab,
      text: tab.label,
      onclick: () => this.show(tab.key),
    }));
    this.modal = el("div", { class: "mmc-modal mmc-settings" }, [
      el("div", { class: "mmc-modal-head" }, [
        ...this.tabs,
        el("button", { class: "mmc-close", text: "✕", title: "Close", onclick: () => this.close() }),
      ]),
      this.body,
      el("div", { class: "mmc-modal-foot" }, [
        el("button", { class: "mmc-add", text: "Done", onclick: () => this.close() }),
      ]),
    ]);
    this.modal.style.position = "relative";

    this.overlay = el("div", {
      class: "mmc-overlay",
      onpointerdown: (event) => { if (event.target === this.overlay) this.close(); },
    }, [this.modal]);

    this.unmount = mountOverlay(this.overlay, () => this.close());
    this.render();
    this.load();
  }

  async load() {
    try {
      this.settings = await loadSettings();
    } catch (error) {
      this.problem = `Could not read the settings — ${error.message}`;
    }
    this.render();
  }

  /**
   * Write one setting through, and take the server's answer over the click.
   *
   * Painted first so the radio moves under the pointer, then corrected if the
   * reply disagrees. The correction is the point: a value the server refused
   * must not be left on screen looking chosen.
   */
  async set(patch) {
    const previous = this.settings;
    this.settings = { ...this.settings, ...patch };
    this.problem = null;
    this.render();
    try {
      this.settings = await saveSettings(patch);
    } catch (error) {
      this.settings = previous;
      this.problem = `Not saved — ${error.message}`;
    }
    this.render();
  }

  show(tab) {
    if (tab === this.tab) return;
    this.tab = tab;
    // The problem line belongs to the control that produced it, so it does not
    // follow you to a tab where it means nothing.
    this.problem = null;
    this.render();
  }

  close() {
    this.unmount();
    this.resolve();
  }

  // ---- render ---------------------------------------------------------------

  render() {
    if (!this.settings) {
      this.body.replaceChildren(el("div", { class: "mmc-set-wait", text: this.problem ?? "Reading settings…" }));
      return;
    }
    for (const [index, tab] of TABS.entries()) {
      this.tabs[index].setAttribute("aria-selected", String(tab.key === this.tab));
    }
    this.body.replaceChildren(
      ...(this.problem ? [el("div", { class: "mmc-set-problem", text: this.problem })] : []),
      ...(this.tab === "quality" ? [this.renderQuality()] : this.renderFolders()),
    );
  }

  renderQuality() {
    const current = this.settings.video_crf;
    // A file edited by hand can hold any point on the scale. Shown as its own
    // row rather than silently rounded to the nearest tier — it is in force,
    // so it has to be visible, and picking a tier is how you leave it.
    const rows = QUALITY.some((tier) => tier.crf === current)
      ? QUALITY
      : [{ crf: current, label: "Custom",
           note: "Set by hand in the settings file. Pick one of the four below to leave it." },
         ...QUALITY];

    return this.section("Output", "Video quality",
      "How much the encoder may throw away when it writes an .mp4. Applies to "
      + "every render this ComfyUI makes, whatever workflow made it.",
      [
        el("div", { class: "mmc-set-choices" }, rows.map((tier) => el("button", {
          class: "mmc-opt mmc-set-opt",
          "aria-checked": tier.crf === current,
          onclick: () => tier.crf !== current && this.set({ video_crf: tier.crf }),
        }, [
          el("span", { class: "mmc-radio" }),
          el("span", { class: "mmc-set-opt-text" }, [
            el("span", { class: "mmc-set-opt-label", text: tier.label }),
            el("span", { class: "mmc-set-opt-note", text: tier.note }),
          ]),
          // The real encoder value, on every row. The rest of this pack shows
          // the exact filename and the exact pixel size under the friendly
          // word; a quality control that said only "Fine" would be the one
          // place in it that asks you to take an adjective on trust.
          el("span", { class: "mmc-set-value", text: `crf ${tier.crf}` }),
        ]))),
        el("div", { class: "mmc-set-foot" }, [
          el("span", {
            text: "MP4, H.264, 8-bit 4:2:0 — the file this pack has always written. "
                + "CRF is libx264's quality target: lower is better and larger, and six "
                + "points is roughly double the size. Needs ComfyUI 0.29 or newer; older "
                + "builds can only write the default.",
          }),
        ]),
      ]);
  }

  // ---- folders ---------------------------------------------------------------

  /**
   * Where the two kinds of file land. One section each, because the pack writes
   * two kinds of thing and filing them together is what makes a gallery you
   * have to read filenames in.
   *
   * This used to be a pill on every node, which meant every node was a place
   * the answer could differ and a shared workflow arrived carrying somebody
   * else's folder names. It is one answer per machine now.
   */
  renderFolders() {
    return [
      this.folderSection("video_prefix", "Renders",
        "Where finished videos land. The Creator and the Timeline both write here.",
        "mp4"),
      this.folderSection("image_prefix", "Stills",
        "Where pre-stage stills land. Their own folder, which is what lets the "
        + "gallery show stills and renders apart without being told which is which.",
        "png"),
      el("div", { class: "mmc-set-foot" }, [
        el("span", { text: "Both are relative to ComfyUI's output folder. Start ComfyUI with " }),
        el("code", { text: "--output-directory" }),
        el("span", { text: " to move that folder itself. The last part of the path names the "
                         + "files, not a folder: the counter core adds is what keeps them apart." }),
      ]),
    ];
  }

  /**
   * One folder field, with what it resolves to underneath it.
   *
   * Written through on Enter or on leaving the field rather than on every
   * keystroke — the rest of the page writes on a click, and a click is finished
   * where a half-typed path is not. What is live is the *reading*: the folder
   * and the example filename move as you type, because a prefix is two things
   * at once and nobody should have to queue a render to find out which.
   */
  folderSection(key, title, description, extension) {
    const stored = this.settings[key];
    const field = el("input", {
      class: "mmc-out-field",
      type: "text",
      value: stored,
      spellcheck: false,
      "aria-label": `${title} — folder and filename prefix`,
      onkeydown: (event) => {
        event.stopPropagation();
        if (event.key === "Enter") field.blur();
        if (event.key === "Escape") { field.value = this.settings[key]; field.blur(); }
      },
      onchange: () => commit(),
      onblur: () => commit(),
    });
    const problem = el("div", { class: "mmc-out-problem" });
    const example = el("div", { class: "mmc-out-example" });

    const paint = () => {
      const { prefix, error } = cleanPrefix(field.value, stored);
      field.classList.toggle("bad", Boolean(error));
      problem.textContent = error ?? "";
      problem.style.display = error ? "" : "none";
      example.replaceChildren(...(error ? [] : [
        // The folder and the file are shown apart because they are the two
        // halves nobody expects: "minimax/renders/H3" is a file called H3 in a
        // folder called renders, not a folder called H3.
        el("div", { class: "mmc-out-line" }, [
          el("span", { class: "mmc-out-key", text: "folder" }),
          el("span", { text: folderOf(prefix) ? `output/${folderOf(prefix)}/` : "output/" }),
        ]),
        el("div", { class: "mmc-out-line" }, [
          el("span", { class: "mmc-out-key", text: "first file" }),
          el("span", { text: examplePath(stemOf(prefix), { extension }) }),
        ]),
      ]));
      return { prefix, error };
    };

    const commit = () => {
      const { prefix, error } = paint();
      // A path that does not parse is left on screen to be fixed rather than
      // stored or silently reverted — nothing has changed on disk yet, and the
      // line under it says what is wrong.
      if (error || prefix === this.settings[key]) return;
      this.set({ [key]: prefix });
    };

    field.addEventListener("input", paint);
    paint();

    return this.section("Output", title, description, [
      el("div", { class: "mmc-set-field" }, [
        field,
        problem,
        example,
        // Core expands these when the file is written. Buttons because nobody
        // guesses the spelling of `%year%`, and a folder per shoot date is the
        // most useful thing this field does.
        el("div", { class: "mmc-out-tokens" }, TOKENS.map((token) => el("button", {
          class: "mmc-out-token",
          text: token,
          title: `Insert ${token} — filled in when the file is written`,
          onclick: () => {
            const at = field.selectionStart ?? field.value.length;
            field.value = field.value.slice(0, at) + token + field.value.slice(field.selectionEnd ?? at);
            field.focus();
            field.setSelectionRange?.(at + token.length, at + token.length);
            commit();
          },
        }))),
      ]),
    ]);
  }

  /** One setting, under a section heading. The heading repeats down the page as
   *  more of them arrive; grouping is what keeps this readable at ten. */
  section(group, title, description, controls) {
    return el("div", { class: "mmc-set-section" }, [
      el("div", { class: "mmc-note-key", text: group }),
      el("div", { class: "mmc-set-title", text: title }),
      el("div", { class: "mmc-set-desc", text: description }),
      ...controls,
    ]);
  }
}
