"""make_human_eval_html.py
===========================
Turn an end-to-end evaluation bundle into a SELF-CONTAINED, offline HTML page
for TWO human raters to judge whether each executed plan achieved its goal.

This produces the human-evaluation instrument for the SAGE robot-planning
benchmark.  The generated ``judge.html`` embeds all item data inline (so it can
be opened directly via ``file://``) and references frame images by their
relative path, so ``frames/`` must sit next to ``judge.html`` in the bundle.

INPUT bundle dir (default: results/human_eval/)
  items.jsonl   one JSON object per line, keys:
                  id, method, task_id, task_desc, expected_objects (list),
                  difficulty, room, scene, plan (list[str]),
                  auto_task_success (0/1/None), goal_method, goal_reason,
                  final_states (dict objType -> {isOpen, isToggled,
                  isPickedUp, parentReceptacles, ...}),
                  frame (relative path e.g. "frames/<id>.jpg", may be empty).
  frames/       the referenced jpg images.

OUTPUT
  <bundle>/judge.html

Blinding & determinism
  * The rater never sees ``method`` (it lives only in the exported CSV/JSON).
  * Items are shown in a deterministic shuffled order (FNV-1a hash of the id
    with a fixed seed) so both raters see the same sequence.

Run as:
  python scripts/make_human_eval_html.py --bundle results/human_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def load_items(items_path: Path) -> list[dict]:
    items: list[dict] = []
    with items_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] skipping malformed JSON at {items_path}:{lineno}: {exc}",
                    file=sys.stderr,
                )
    return items


def write_missing_stub(out_path: Path, bundle: Path) -> None:
    """Write a tiny placeholder when items.jsonl is absent (so we never crash)."""
    msg = (
        f"run the eval with SAGE_EVAL_BUNDLE={bundle} first"
    )
    html = (
        "<!doctype html>\n"
        "<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>SAGE human eval — no data</title>"
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;"
        "background:#f4f5f7;color:#222;margin:0;padding:48px;}"
        ".box{max-width:640px;margin:8vh auto;background:#fff;border:1px solid #dcdfe4;"
        "border-radius:12px;padding:32px 36px;box-shadow:0 1px 4px rgba(0,0,0,.06);}"
        "h1{font-size:20px;margin:0 0 12px;}code{background:#eef0f3;padding:2px 6px;"
        "border-radius:5px;font-size:90%;}</style></head><body><div class=\"box\">"
        "<h1>No evaluation items found</h1>"
        f"<p>This bundle has no <code>items.jsonl</code>. {_html_escape(msg)}.</p>"
        "<p>Then re-run "
        "<code>python scripts/make_human_eval_html.py --bundle "
        f"{_html_escape(str(bundle))}</code>.</p>"
        "</div></body></html>\n"
    )
    out_path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------
# Small text utility (used only for the stub; the live page escapes in JS)
# --------------------------------------------------------------------------
def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# --------------------------------------------------------------------------
# HTML template
# --------------------------------------------------------------------------
def build_html(items: list[dict]) -> str:
    # Embed items verbatim. json.dumps with ensure_ascii keeps it ASCII-safe;
    # close </script> sequences are neutralised so the inline script can't break.
    payload = json.dumps(items, ensure_ascii=True).replace("</", "<\\/")

    return _TEMPLATE.replace("/*__ITEMS_JSON__*/null", payload)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAGE Human Evaluation</title>
<style>
  :root {
    --bg: #f4f5f7; --card: #ffffff; --line: #dcdfe4; --ink: #1c1f24;
    --muted: #6b7280; --accent: #2563eb; --ok: #16a34a;
  }
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--ink); margin: 0; padding: 0 0 80px;
    line-height: 1.45;
  }
  .topbar {
    position: sticky; top: 0; z-index: 50; background: #ffffff;
    border-bottom: 1px solid var(--line); box-shadow: 0 1px 3px rgba(0,0,0,.06);
    padding: 10px 20px;
  }
  .topbar-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  .topbar h1 { font-size: 16px; margin: 0 12px 0 0; }
  .topbar label { font-size: 13px; color: var(--muted); }
  select, input[type=text], button {
    font: inherit; padding: 6px 10px; border: 1px solid var(--line);
    border-radius: 7px; background: #fff; color: var(--ink);
  }
  button { cursor: pointer; }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  button:hover { filter: brightness(0.97); }
  .progress { font-size: 13px; color: var(--muted); margin-left: auto; }
  .progress b { color: var(--ink); }
  .blurb {
    font-size: 12.5px; color: var(--muted); padding: 6px 20px 2px;
    max-width: 920px;
  }
  .blurb b { color: var(--ink); }
  .wrap { max-width: 920px; margin: 16px auto; padding: 0 16px; }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: 18px 20px; margin: 0 0 20px; box-shadow: 0 1px 3px rgba(0,0,0,.05);
  }
  .card.done { border-left: 5px solid var(--ok); }
  .idx { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .task { font-size: 17px; font-weight: 650; margin: 2px 0 10px; }
  .meta { font-size: 12.5px; color: var(--muted); margin-bottom: 10px; }
  .meta span { margin-right: 14px; }
  .frame { margin: 4px 0 12px; }
  .frame img { max-width: 520px; width: 100%; border-radius: 8px; border: 1px solid var(--line); display: block; }
  .noframe {
    max-width: 520px; height: 120px; display: flex; align-items: center;
    justify-content: center; color: var(--muted); background: #eef0f3;
    border: 1px dashed var(--line); border-radius: 8px; font-size: 13px;
  }
  .sect { margin: 10px 0; }
  .sect .h { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin-bottom: 3px; }
  .goal { font-size: 14px; }
  .goal .obj { display: inline-block; background: #eff6ff; color: #1e3a8a;
    border: 1px solid #bfdbfe; border-radius: 6px; padding: 1px 8px; margin: 2px 4px 2px 0; font-size: 13px; }
  ol.plan { margin: 4px 0 0; padding-left: 22px; font-size: 13.5px; }
  ol.plan li { margin: 1px 0; }
  .states { font-size: 13px; }
  .states .st { margin: 1px 0; }
  .states .stobj { font-weight: 600; }
  .auto { font-size: 12.5px; color: var(--muted); background: #fafafa;
    border: 1px dashed var(--line); border-radius: 7px; padding: 6px 10px; margin: 10px 0; }
  .judge { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--line); }
  .judge .opts { display: flex; gap: 18px; flex-wrap: wrap; align-items: center; }
  .judge label.opt { font-size: 14px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
  .judge input[type=radio] { width: 16px; height: 16px; }
  .judge .note { margin-top: 8px; }
  .judge .note input { width: 100%; }
  .empty { text-align: center; color: var(--muted); padding: 40px; }
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-row">
    <h1>SAGE Human Evaluation</h1>
    <label>Rater:
      <select id="raterSel">
        <option value="rater1">rater1</option>
        <option value="rater2">rater2</option>
      </select>
    </label>
    <label>Name (optional):
      <input type="text" id="raterName" placeholder="your name" size="14">
    </label>
    <button id="dlCsv">Download CSV</button>
    <button id="dlJson">Download JSON</button>
    <span class="progress" id="prog">judged 0 / 0</span>
  </div>
</div>
<div class="blurb">
  <b>Criteria.</b> Mark <b>1</b> only if the <b>final state</b> satisfies the full task goal.
  Mark <b>0.5</b> only if a <b>concrete sub-goal was actually achieved</b> in the final
  state (e.g. the machine is on but the mug is not placed) — <b>not</b> for merely moving
  around or attempting without achieving anything. Mark <b>0</b> if nothing required was
  achieved (e.g. the plan only navigates, or the key action — TurnOn/Pick/Place/Open —
  never appears). Judge independently from the frame, the final states, and the executed
  plan — check the plan actually contains the key action — <b>you decide</b>. Answers autosave per rater.
</div>

<div class="wrap" id="cards"></div>

<script>
"use strict";

// ---- Embedded item data (method kept here but NEVER rendered) ----
var ITEMS = /*__ITEMS_JSON__*/null;
if (!Array.isArray(ITEMS)) { ITEMS = []; }

// ---- Deterministic shuffle: FNV-1a hash of id with a fixed seed ----
var SEED = 0x9e3779b1; // fixed seed -> same order for every rater
function fnv1a(str) {
  var h = (2166136261 ^ SEED) >>> 0;
  for (var i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h >>> 0;
}
var ORDER = ITEMS.slice().sort(function (a, b) {
  var ha = fnv1a(String(a.id)), hb = fnv1a(String(b.id));
  if (ha !== hb) return ha - hb;
  return String(a.id) < String(b.id) ? -1 : 1; // stable tie-break
});

// ---- localStorage (keyed per rater) ----
function storeKey(rater) { return "sage_human_eval__" + rater; }
function loadJudgments(rater) {
  try { return JSON.parse(localStorage.getItem(storeKey(rater))) || {}; }
  catch (e) { return {}; }
}
function saveJudgments(rater, data) {
  try { localStorage.setItem(storeKey(rater), JSON.stringify(data)); }
  catch (e) { /* best-effort */ }
}
function loadName(rater) {
  try { return localStorage.getItem(storeKey(rater) + "__name") || ""; }
  catch (e) { return ""; }
}
function saveName(rater, name) {
  try { localStorage.setItem(storeKey(rater) + "__name", name || ""); }
  catch (e) { /* best-effort */ }
}

// ---- escaping ----
function esc(s) {
  s = (s === null || s === undefined) ? "" : String(s);
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;")
          .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ---- render final_states into a readable line list ----
function renderStates(fs) {
  if (!fs || typeof fs !== "object") return "<span class='states st'>(none)</span>";
  var keys = Object.keys(fs);
  if (!keys.length) return "<span class='states st'>(none)</span>";
  var rows = keys.map(function (objType) {
    var st = fs[objType] || {};
    var parts = [];
    if (st.isOpen !== undefined && st.isOpen !== null)
      parts.push("isOpen=" + st.isOpen);
    if (st.isToggled !== undefined && st.isToggled !== null)
      parts.push("isToggled=" + st.isToggled);
    if (st.isPickedUp !== undefined && st.isPickedUp !== null)
      parts.push("isPickedUp=" + st.isPickedUp);
    if (st.parentReceptacles !== undefined && st.parentReceptacles !== null) {
      var pr = st.parentReceptacles;
      if (Array.isArray(pr)) pr = "[" + pr.join(", ") + "]";
      parts.push("parentReceptacles=" + pr);
    }
    // include any other primitive fields not already shown
    Object.keys(st).forEach(function (k) {
      if (["isOpen", "isToggled", "isPickedUp", "parentReceptacles"].indexOf(k) >= 0) return;
      var v = st[k];
      if (v === null || v === undefined) return;
      if (typeof v === "object") {
        if (Array.isArray(v)) parts.push(k + "=[" + v.join(", ") + "]");
        return;
      }
      parts.push(k + "=" + v);
    });
    return "<div class='st'><span class='stobj'>" + esc(objType) + "</span>: " +
           esc(parts.join("; ")) + "</div>";
  });
  return rows.join("");
}

function renderGoal(objs) {
  if (!Array.isArray(objs) || !objs.length) return "<span class='obj'>(none)</span>";
  return objs.map(function (o) { return "<span class='obj'>" + esc(o) + "</span>"; }).join("");
}

function renderPlan(plan) {
  if (!Array.isArray(plan) || !plan.length) return "<em>(empty plan)</em>";
  return "<ol class='plan'>" + plan.map(function (s) {
    return "<li>" + esc(s) + "</li>";
  }).join("") + "</ol>";
}

// ---- state ----
var curRater = "rater1";
var judgments = {};

var cardsEl = document.getElementById("cards");
var progEl = document.getElementById("prog");
var raterSel = document.getElementById("raterSel");
var raterName = document.getElementById("raterName");

function judgedCount() {
  var n = 0;
  ORDER.forEach(function (it) {
    var j = judgments[it.id];
    if (j && j.judgment !== undefined && j.judgment !== null && j.judgment !== "") n++;
  });
  return n;
}

function updateProgress() {
  progEl.innerHTML = "judged <b>" + judgedCount() + "</b> / " + ORDER.length;
}

function markCardDone(idStr, done) {
  var card = document.getElementById("card-" + idStr);
  if (card) card.classList.toggle("done", !!done);
}

function onJudgmentChange(itemId, value) {
  if (!judgments[itemId]) judgments[itemId] = {};
  judgments[itemId].judgment = value;
  saveJudgments(curRater, judgments);
  markCardDone(cssId(itemId), true);
  updateProgress();
}

function onNoteChange(itemId, note) {
  if (!judgments[itemId]) judgments[itemId] = {};
  judgments[itemId].note = note;
  saveJudgments(curRater, judgments);
}

function cssId(id) { return String(id).replace(/[^A-Za-z0-9_-]/g, "_"); }

function renderCards() {
  if (!ORDER.length) {
    cardsEl.innerHTML = "<div class='empty'>No items to judge.</div>";
    updateProgress();
    return;
  }
  var html = ORDER.map(function (it, i) {
    var cid = cssId(it.id);
    var j = judgments[it.id] || {};
    var frame = it.frame;
    var frameHtml;
    if (frame && String(frame).trim().length) {
      frameHtml = "<img loading='lazy' src='" + esc(frame) + "' alt='final frame'>";
    } else {
      frameHtml = "<div class='noframe'>[no frame]</div>";
    }
    // Auto-checker verdict is deliberately HIDDEN from raters so the human
    // judgment stays independent of the auto signal; the two are reconciled
    // offline (auto-vs-human agreement / Cohen's kappa). Do not surface
    // auto_task_success or goal_reason in the rater UI.
    var auto = "";

    function radio(val, label) {
      var checked = (String(j.judgment) === String(val)) ? " checked" : "";
      return "<label class='opt'><input type='radio' name='j-" + cid + "' value='" +
             val + "'" + checked + "> " + label + "</label>";
    }

    var noteVal = (j.note === undefined || j.note === null) ? "" : j.note;

    return "" +
      "<div class='card" + (j.judgment !== undefined && j.judgment !== null && j.judgment !== "" ? " done" : "") +
        "' id='card-" + cid + "'>" +
        "<div class='idx'>Item " + (i + 1) + " of " + ORDER.length + "</div>" +
        "<div class='task'>" + esc(it.task_desc) + "</div>" +
        "<div class='meta'>" +
          "<span>difficulty: " + esc(it.difficulty) + "</span>" +
          "<span>room: " + esc(it.room) + "</span>" +
          "<span>scene: " + esc(it.scene) + "</span>" +
        "</div>" +
        "<div class='frame'>" + frameHtml + "</div>" +
        "<div class='sect'><div class='h'>Goal (expected objects)</div>" +
          "<div class='goal'>" + renderGoal(it.expected_objects) + "</div></div>" +
        "<div class='sect'><div class='h'>Final states</div>" +
          "<div class='states'>" + renderStates(it.final_states) + "</div></div>" +
        "<div class='sect'><div class='h'>Executed plan</div>" +
          renderPlan(it.plan) + "</div>" +
        auto +
        "<div class='judge'>" +
          "<div class='opts'>" +
            "<span class='h' style='margin-right:4px'>Goal achieved?</span>" +
            radio("1", "1 &mdash; achieved") +
            radio("0.5", "0.5 &mdash; partial") +
            radio("0", "0 &mdash; not achieved") +
          "</div>" +
          "<div class='note'><input type='text' placeholder='note (optional)' " +
            "id='note-" + cid + "' value=\"" + esc(noteVal) + "\"></div>" +
        "</div>" +
      "</div>";
  }).join("");
  cardsEl.innerHTML = html;

  // wire events
  ORDER.forEach(function (it) {
    var cid = cssId(it.id);
    var radios = document.getElementsByName("j-" + cid);
    for (var r = 0; r < radios.length; r++) {
      radios[r].addEventListener("change", (function (itemId) {
        return function (ev) { onJudgmentChange(itemId, ev.target.value); };
      })(it.id));
    }
    var noteEl = document.getElementById("note-" + cid);
    if (noteEl) {
      noteEl.addEventListener("input", (function (itemId) {
        return function (ev) { onNoteChange(itemId, ev.target.value); };
      })(it.id));
    }
  });

  updateProgress();
}

function switchRater(rater) {
  curRater = rater;
  judgments = loadJudgments(curRater);
  raterName.value = loadName(curRater);
  renderCards();
}

// ---- exports ----
function effectiveRaterId() {
  var nm = (raterName.value || "").trim();
  return nm ? (curRater + ":" + nm) : curRater;
}

function csvCell(s) {
  s = (s === null || s === undefined) ? "" : String(s);
  if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

function buildCsv() {
  var header = ["item_id", "task_id", "method", "rater", "judgment", "note"];
  var lines = [header.join(",")];
  var rid = effectiveRaterId();
  ORDER.forEach(function (it) {
    var j = judgments[it.id] || {};
    lines.push([
      csvCell(it.id), csvCell(it.task_id), csvCell(it.method), csvCell(rid),
      csvCell(j.judgment === undefined ? "" : j.judgment),
      csvCell(j.note === undefined ? "" : j.note)
    ].join(","));
  });
  return lines.join("\r\n") + "\r\n";
}

function buildJson() {
  var rid = effectiveRaterId();
  var rows = ORDER.map(function (it) {
    var j = judgments[it.id] || {};
    return {
      item_id: it.id, task_id: it.task_id, method: it.method, rater: rid,
      judgment: (j.judgment === undefined ? null : j.judgment),
      note: (j.note === undefined ? "" : j.note)
    };
  });
  return JSON.stringify({ rater: rid, n: ORDER.length, judged: judgedCount(), rows: rows }, null, 2);
}

function download(filename, text, mime) {
  var blob = new Blob([text], { type: mime });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
}

// ---- init ----
raterSel.addEventListener("change", function () { switchRater(raterSel.value); });
raterName.addEventListener("input", function () { saveName(curRater, raterName.value); });
document.getElementById("dlCsv").addEventListener("click", function () {
  download("sage_human_eval_" + effectiveRaterId().replace(/[^A-Za-z0-9_.-]/g, "_") + ".csv",
           buildCsv(), "text/csv;charset=utf-8");
});
document.getElementById("dlJson").addEventListener("click", function () {
  download("sage_human_eval_" + effectiveRaterId().replace(/[^A-Za-z0-9_.-]/g, "_") + ".json",
           buildJson(), "application/json;charset=utf-8");
});

switchRater("rater1");
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate a self-contained human-eval HTML page from a SAGE eval bundle."
    )
    ap.add_argument(
        "--bundle",
        default="results/human_eval",
        help="Bundle directory containing items.jsonl and frames/ (default: results/human_eval)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output HTML path (default: <bundle>/judge.html)",
    )
    args = ap.parse_args(argv)

    bundle = Path(args.bundle)
    out_path = Path(args.out) if args.out else (bundle / "judge.html")
    items_path = bundle / "items.jsonl"

    # Ensure the bundle dir exists so we can at least write the stub.
    bundle.mkdir(parents=True, exist_ok=True)

    if not items_path.exists():
        write_missing_stub(out_path, bundle)
        print(
            f"[make_human_eval_html] no items.jsonl in {bundle} — wrote placeholder {out_path}",
            file=sys.stderr,
        )
        return 0

    items = load_items(items_path)
    if not items:
        write_missing_stub(out_path, bundle)
        print(
            f"[make_human_eval_html] {items_path} had no usable items — wrote placeholder {out_path}",
            file=sys.stderr,
        )
        return 0

    html = build_html(items)
    out_path.write_text(html, encoding="utf-8")

    n_frames = sum(1 for it in items if it.get("frame"))
    print(f"[make_human_eval_html] wrote {out_path} ({len(items)} items, {n_frames} with frames)")
    print(f"  Open it via: file://{out_path.resolve()}")
    print("  (frames/ must sit beside judge.html for images to load)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
