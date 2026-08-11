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

import { el, mountOverlay } from "./dom.js";
import { loadSettings, saveSettings } from "./api.js";

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

class SettingsPage {
  constructor(resolve) {
    this.resolve = resolve;
    this.settings = null;   // until the server answers
    this.problem = null;
  }

  mount() {
    this.body = el("div", { class: "mmc-set-body" });
    this.modal = el("div", { class: "mmc-modal mmc-settings" }, [
      el("div", { class: "mmc-modal-head" }, [
        el("button", { class: "mmc-tab", "aria-selected": true, text: "Settings" }),
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
    this.body.replaceChildren(
      ...(this.problem ? [el("div", { class: "mmc-set-problem", text: this.problem })] : []),
      this.renderQuality(),
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
