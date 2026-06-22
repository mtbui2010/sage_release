#!/usr/bin/env python3
"""Emit editable .drawio (diagrams.net) versions of the two SAGE concept figures.

These mirror images/fig1_storyboard.* and images/fig2_overview.* but as native
draw.io XML, so every box / arrow / label / icon is a first-class object the user
can move and re-style in the draw.io desktop app, GIMP-import, or VS Code drawio
extension. Real AI2-THOR frames are embedded as base64 PNG data URIs, so the
files are fully self-contained (no external image dependency).

Outputs (nothing in the paper is touched):
  images/fig1_storyboard.drawio
  images/fig2_overview.drawio

Run:  python scripts/gen_drawio.py
"""
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FR = os.path.join(ROOT, "images", "frames")
OUT = os.path.join(ROOT, "images")

# ----- palette (matches the matplotlib figures) -----------------------------
OLIVE, OLIVE_E = "#d9d7b8", "#8a875a"
INSTR_E = "#1f6fb2"
SLATE, SLATE_E = "#8a93a8", "#4f5670"
RED, PINK = "#c0392b", "#fdecea"
GREEN, GREEN_BG = "#2e8b57", "#e7f4ec"
RAIL = "#cfcb9b"
GRAY = "#9aa0aa"
CODE_BG = "#f4f4ef"


def esc(s):
    """Escape text for an HTML draw.io label; keep <br> line breaks."""
    s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
         .replace('"', "&quot;"))
    # line breaks become an (XML-escaped) <br> the html=1 renderer expands
    return s.replace("[BR]", "&lt;br&gt;")


def data_uri(path):
    with open(path, "rb") as fh:
        return "data:image/png," + base64.b64encode(fh.read()).decode("ascii")


class Drawio:
    """Minimal draw.io document builder (one diagram page)."""

    def __init__(self, name, w, h):
        self.name = name
        self.w, self.h = w, h
        self.cells = []
        self._id = 1

    def _nid(self):
        self._id += 1
        return f"n{self._id}"

    def box(self, x, y, w, h, text="", fill="#ffffff", stroke="#000000",
            fontcol="#000000", fs=11, bold=False, italic=False, dashed=False,
            rounded=True, align="left", valign="middle", vertical=False,
            opacity=100, lw=1):
        style = []
        style.append("rounded=1" if rounded else "rounded=0")
        style.append("whiteSpace=wrap;html=1")
        style.append(f"fillColor={fill}" if fill else "fillColor=none")
        style.append(f"strokeColor={stroke}" if stroke else "strokeColor=none")
        style.append(f"fontColor={fontcol}")
        style.append(f"fontSize={fs}")
        fstyle = (1 if bold else 0) | (2 if italic else 0)
        if fstyle:
            style.append(f"fontStyle={fstyle}")
        if dashed:
            style.append("dashed=1;dashPattern=6 3")
        style.append(f"align={align};verticalAlign={valign}")
        style.append("spacingLeft=4;spacingRight=4")
        if vertical:
            style.append("horizontal=0")
        if opacity != 100:
            style.append(f"opacity={opacity}")
        if lw != 1:
            style.append(f"strokeWidth={lw}")
        cid = self._nid()
        self.cells.append(
            f'<mxCell id="{cid}" value="{esc(text)}" style="{";".join(style)};" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
            f'height="{h}" as="geometry"/></mxCell>')
        return cid

    def text(self, x, y, w, h, text, fs=11, bold=False, italic=False,
             fontcol="#000000", align="center", valign="middle", vertical=False):
        return self.box(x, y, w, h, text, fill=None, stroke=None,
                        fontcol=fontcol, fs=fs, bold=bold, italic=italic,
                        rounded=False, align=align, valign=valign, vertical=vertical)

    def image(self, x, y, w, h, uri, border=SLATE_E):
        cid = self._nid()
        style = (f"shape=image;imageAspect=0;aspect=fixed;verticalAlign=top;"
                 f"image={uri};")
        self.cells.append(
            f'<mxCell id="{cid}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f'</mxCell>')
        if border:
            self.box(x, y, w, h, "", fill=None, stroke=border, rounded=False, lw=1)
        return cid

    def ellipse(self, x, y, w, h, fill="#ffffff", stroke="#000000", text="", fs=9):
        cid = self._nid()
        style = (f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};"
                 f"strokeColor={stroke};fontSize={fs}")
        self.cells.append(
            f'<mxCell id="{cid}" value="{esc(text)}" style="{style};" vertex="1" '
            f'parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
        return cid

    def shape(self, x, y, w, h, shp, fill="#ffffff", stroke="#000000",
              text="", fs=9, fontcol="#000000", bold=False):
        cid = self._nid()
        fstyle = ";fontStyle=1" if bold else ""
        style = (f"shape={shp};whiteSpace=wrap;html=1;fillColor={fill};"
                 f"strokeColor={stroke};fontSize={fs};fontColor={fontcol}{fstyle}")
        self.cells.append(
            f'<mxCell id="{cid}" value="{esc(text)}" style="{style};" vertex="1" '
            f'parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
        return cid

    def line(self, x1, y1, x2, y2, stroke="#999999", dashed=False, lw=1):
        cid = self._nid()
        dash = "dashed=1;dashPattern=4 4;" if dashed else ""
        style = (f"endArrow=none;html=1;{dash}strokeColor={stroke};"
                 f"strokeWidth={lw}")
        self.cells.append(
            f'<mxCell id="{cid}" style="{style};" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
            f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')
        return cid

    def arrow(self, x1, y1, x2, y2, stroke="#555555", text="", dashed=False,
              lw=1.4, fs=9, fontcol=None, curved=False):
        cid = self._nid()
        dash = "dashed=1;dashPattern=6 3;" if dashed else ""
        crv = "curved=1;" if curved else ""
        fc = f"fontColor={fontcol};" if fontcol else ""
        style = (f"endArrow=block;endFill=1;html=1;{dash}{crv}strokeColor={stroke};"
                 f"strokeWidth={lw};fontSize={fs};{fc}labelBackgroundColor=#ffffff")
        self.cells.append(
            f'<mxCell id="{cid}" value="{esc(text)}" style="{style};" edge="1" '
            f'parent="1"><mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
            f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')
        return cid

    def xml(self):
        body = "\n        ".join(self.cells)
        return (
            f'<mxfile host="app.diagrams.net" type="device">\n'
            f'  <diagram id="{self.name}" name="{self.name}">\n'
            f'    <mxGraphModel dx="1100" dy="700" grid="1" gridSize="10" '
            f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
            f'pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" '
            f'math="0" shadow="0">\n'
            f'      <root>\n'
            f'        <mxCell id="0"/>\n'
            f'        <mxCell id="1" parent="0"/>\n'
            f'        {body}\n'
            f'      </root>\n'
            f'    </mxGraphModel>\n'
            f'  </diagram>\n'
            f'</mxfile>\n')


# ===========================================================================
# Figure 1 — gate-and-edit storyboard
# ===========================================================================
def build_fig1():
    d = Drawio("Fig1-SAGE-storyboard", 940, 560)

    # left planner rail + right header
    d.box(12, 40, 48, 430, "LLM Planner", fill=RAIL, stroke=OLIVE_E,
          fs=13, bold=True, align="center", vertical=True)
    d.text(560, 8, 368, 22, "Embodied Agent & AI2-THOR", fs=12, bold=True,
           align="right")

    # band separators + time ticks
    for ysep in (165, 305):
        d.line(70, ysep, 928, ysep, stroke="#bbbbbb", dashed=True)
    for ty, lab in [(48, "t = 0"), (175, "t = 3"), (315, "t = 8")]:
        d.text(64, ty, 44, 18, lab, fs=10, bold=True, fontcol="#595959", align="left")

    IMG_X, IMG_W, IMG_H = 745, 175, 117

    # ---- band 1 : t = 0 ----
    d.box(78, 44, 430, 34, "Cook a potato and put it into the recycle bin.",
          fill="#ffffff", stroke=INSTR_E, fs=11, italic=True, lw=1.4)
    d.box(78, 88, 520, 34,
          "Plan:  Find potato → Pick potato → Cook → Place recyclebin",
          fill=OLIVE, stroke=OLIVE_E, fs=10)
    d.arrow(60, 61, 78, 61, stroke=OLIVE_E, lw=1.4)
    d.image(IMG_X, 44, IMG_W, IMG_H, FRAMES["overview"])
    d.arrow(600, 105, IMG_X - 4, 95, stroke="#777777", text="execute", fs=8)

    # ---- band 2 : t = 3 ----
    d.box(78, 178, 430, 30, "Observation:  fridge is closed — cannot Pick the potato.",
          fill=SLATE, stroke=SLATE_E, fontcol="#ffffff", fs=10)
    d.box(78, 214, 450, 30,
          "✗  GATE: precondition violated  (¬opened(fridge) before Pick)",
          fill=PINK, stroke=RED, fontcol=RED, fs=10, bold=True, lw=1.6)
    d.box(78, 250, 560, 34,
          "✎  EDIT (suffix only):  Open fridge → Pick potato → Close fridge → …",
          fill=PINK, stroke=RED, fontcol=RED, fs=10, bold=True, dashed=True, lw=1.6)
    d.image(IMG_X, 176, IMG_W, IMG_H, FRAMES["fridge_closed"])
    d.arrow(IMG_X - 4, 210, 510, 193, stroke=SLATE_E, text="perceive", fs=8)

    # ---- band 3 : t = 8 ----
    d.box(78, 318, 380, 30, "✓  GATE: plan certified — executes safely",
          fill=GREEN_BG, stroke=GREEN, fontcol=GREEN, fs=10, bold=True, lw=1.5)
    d.box(78, 354, 450, 30, "Observation:  potato placed in the recycle bin  ✓",
          fill=SLATE, stroke=SLATE_E, fontcol="#ffffff", fs=10)
    d.image(IMG_X, 318, IMG_W, IMG_H, FRAMES["placed"])
    d.arrow(460, 333, IMG_X - 4, 350, stroke="#777777", text="execute", fs=8)

    # ---- cost inset : "LLM calls to recover" ----
    ix, iy, iw, ih = 78, 406, 470, 132
    d.box(ix, iy, iw, ih, "", fill="#fbfbfb", stroke="#c7c7c7", lw=1)
    d.text(ix + 10, iy + 6, iw - 20, 18, "LLM calls to recover  (lower is better)",
           fs=10, bold=True, align="left")
    bar0_x = ix + 84          # bars start here
    scale = 250.0 / 4.0       # px per call (axis to ~4.0)
    rows = [("SAGE", 1.18, RED, "#ffffff"),
            ("Hier. FS", 2.96, GRAY, "#000000"),
            ("ReAct", 3.71, GRAY, "#000000")]
    for i, (lab, val, col, fc) in enumerate(rows):
        by = iy + 34 + i * 30
        d.text(ix + 8, by, 72, 22, lab, fs=9, bold=(lab == "SAGE"), align="right")
        d.box(bar0_x, by, max(8, val * scale), 22, "", fill=col, stroke="#000000", rounded=False)
        d.text(bar0_x + val * scale + 4, by, 42, 22, f"{val:.2f}", fs=9, bold=True, align="left")
    d.text(bar0_x + 1.18 * scale + 60, iy + 30, 90, 26, "2.5–3.1×[BR]fewer",
           fs=10, bold=True, fontcol=RED, align="left")

    # ---- legend (right of the cost inset) ----
    leg = [("Instruction", "#ffffff", INSTR_E), ("High-level plan", OLIVE, OLIVE_E),
           ("Gate verdict", PINK, RED), ("Edited suffix", PINK, RED),
           ("Observation", SLATE, SLATE_E)]
    lx, ly = 565, 430
    for name, fc, ec in leg:
        d.box(lx, ly, 16, 16, "", fill=fc, stroke=ec, rounded=False)
        d.text(lx + 20, ly - 1, 130, 18, name, fs=9, align="left")
        ly += 22
    return d


# ===========================================================================
# Figure 2 — GAIA-style 4-panel overview
# ===========================================================================
def icon_tree(d, cx, cy):
    d.line(cx, cy + 6, cx - 14, cy + 22, stroke="#555555")
    d.line(cx, cy + 6, cx + 14, cy + 22, stroke="#555555")
    d.ellipse(cx - 7, cy - 8, 14, 14, fill="#dfe6f0", stroke=INSTR_E)
    d.ellipse(cx - 21, cy + 18, 14, 14, fill="#dfe6f0", stroke=INSTR_E)
    d.ellipse(cx + 7, cy + 18, 14, 14, fill="#dfe6f0", stroke=INSTR_E)


def icon_db(d, cx, cy):
    d.shape(cx - 13, cy - 12, 26, 30, "cylinder3", fill="#e8e2f0", stroke="#6b4f9e")


def icon_shield(d, cx, cy):
    d.shape(cx - 14, cy - 12, 28, 32, "mxgraph.basic.shield", fill="#e7f4ec",
            stroke=GREEN, text="✓", fs=12, fontcol=GREEN, bold=True)


def icon_pencil(d, cx, cy):
    d.shape(cx - 13, cy - 13, 28, 28, "rhombus", fill=PINK, stroke=RED,
            text="✎", fs=14, fontcol=RED, bold=True)


def panel(d, x0, w, num_title, icon_fn, ours=False):
    stroke = RED if ours else GRAY
    lw = 2 if ours else 1.2
    d.box(x0, 44, w, 372, "", fill="#ffffff", stroke=stroke, lw=lw)
    d.box(x0, 44, w, 30, "", fill=(PINK if ours else "#eef0f3"), stroke=stroke,
          rounded=True, lw=lw)
    d.text(x0 + 42, 44, w - 48, 30, num_title, fs=11, bold=True,
           fontcol=(RED if ours else "#2b2b2b"), align="left")
    icon_fn(d, x0 + 22, 59)


def build_fig2():
    d = Drawio("Fig2-SAGE-overview", 1010, 470)
    W = 230
    XS = [12, 258, 504, 760]

    panel(d, XS[0], W, "1. Hierarchical[BR]Decompose", icon_tree)
    panel(d, XS[1], W, "2. Retrieve[BR]+ Expand", icon_db)
    panel(d, XS[2], W, "★ 3. Symbolic[BR]Action-Gate", icon_shield, ours=True)
    panel(d, XS[3], W, "★ 4. Sub-goal[BR]Edit + Execute", icon_pencil, ours=True)

    # ---- panel 1 : decompose ----
    x = XS[0]
    d.image(x + 14, 86, 150, 100, FRAMES["overview"])
    d.text(x + 14, 188, 200, 16, "input: RGB scene + task", fs=8, fontcol="#666666", align="left")
    d.box(x + 14, 210, W - 28, 26, "“cook a potato and put it[BR]into the recycle bin”",
          fill="#f7f7f4", stroke="#cccccc", fs=8, italic=True)
    for i, g in enumerate(["g₁  find & pick potato", "g₂  cook potato",
                           "g₃  place in recycle bin"]):
        d.box(x + 14, 246 + i * 34, W - 28, 28, g, fill="#dfe6f0", stroke=INSTR_E, fs=9)

    # ---- panel 2 : retrieve + expand ----
    x = XS[1]
    d.text(x + 12, 84, W - 20, 16, "hybrid memory (seed + live episodes)",
           fs=8, fontcol="#666666", align="left")
    d.box(x + 12, 104, W - 24, 92,
          '{[BR]  "task": "put apple in fridge",[BR]  "reasoning": "open before place",'
          '[BR]  "plan_text": "Open→Pick→Place"[BR]}',
          fill=CODE_BG, stroke="#bbbbbb", fs=8, align="left")
    d.text(x + 12, 200, W - 20, 16, "+ LLM expands each sub-goal",
           fs=8, fontcol="#666666", align="left")
    d.box(x + 12, 222, W - 24, 130,
          "candidate plan[BR]"
          "MoveTo fridge[BR]Find potato[BR]Pick potato[BR]Cook potato[BR]"
          "MoveTo bin[BR]Place potato",
          fill=OLIVE, stroke=OLIVE_E, fs=9)

    # ---- panel 3 : symbolic action-gate (ours) ----
    x = XS[2]
    d.text(x + 12, 84, W - 20, 16, "verify-before-execute (no LLM)",
           fs=8, fontcol="#666666", align="left")
    # 5-field symbolic-state table
    tx, ty = x + 12, 108
    cols = ["", "arr", "fnd", "hld", "opn", "on"]
    cw = [44, 33, 33, 33, 33, 30]
    cx = tx
    for j, c in enumerate(cols):
        d.box(cx, ty, cw[j], 22, c, fill="#eceff3", stroke="#9aa0aa", rounded=False, fs=8, bold=True, align="center")
        cx += cw[j]
    table = [("Find", ["✓", "✓", "·", "·", "·"], None),
             ("Pick", ["✓", "✓", "✗", "✗", "·"], RED),
             ("Open", ["✓", "✓", "·", "✓", "·"], GREEN)]
    for i, (rl, vals, hl) in enumerate(table):
        ry = ty + 22 + i * 22
        rfill = (PINK if hl == RED else GREEN_BG if hl == GREEN else "#ffffff")
        rfc = hl or "#000000"
        cx = tx
        d.box(cx, ry, cw[0], 22, rl, fill=rfill, stroke="#9aa0aa", rounded=False, fs=8, bold=True, fontcol=rfc, align="center")
        cx += cw[0]
        for j, v in enumerate(vals):
            d.box(cx, ry, cw[j + 1], 22, v, fill=rfill, stroke="#9aa0aa", rounded=False, fs=8, fontcol=rfc, align="center")
            cx += cw[j + 1]
    d.box(x + 12, 200, W - 24, 40,
          "✗ violation[BR]Pick requires opened(fridge)",
          fill=PINK, stroke=RED, fontcol=RED, fs=9, bold=True, lw=1.6)
    d.text(x + 12, 248, W - 20, 16, "5-field state: arrived/found/", fs=8, fontcol="#888888", align="left")
    d.text(x + 12, 262, W - 20, 16, "holding/opened/on", fs=8, fontcol="#888888", align="left")

    # ---- panel 4 : sub-goal edit + execute (ours) ----
    x = XS[3]
    d.text(x + 12, 84, W - 20, 16, "✎ regenerate the failed suffix only",
           fs=8, fontcol="#666666", align="left")
    d.box(x + 12, 104, W - 24, 84,
          "repaired suffix[BR]Open fridge[BR]Pick potato[BR]Close fridge[BR]…",
          fill=CODE_BG, stroke=RED, fontcol=RED, fs=9, dashed=True)
    d.image(x + 12, 198, 100, 67, FRAMES["fridge_open"])
    d.image(x + 118, 198, 100, 67, FRAMES["placed"])
    d.text(x + 12, 268, 100, 14, "open", fs=8, fontcol="#666666")
    d.text(x + 118, 268, 100, 14, "placed", fs=8, fontcol="#666666")
    d.box(x + 12, 292, W - 24, 30, "certified plan → AI2-THOR",
          fill=GREEN_BG, stroke=GREEN, fontcol=GREEN, fs=9, bold=True)

    # ---- inter-panel arrows ----
    d.arrow(XS[0] + W, 230, XS[1], 230, stroke="#777777", text="sub-goals", fs=8)
    d.arrow(XS[1] + W, 230, XS[2], 230, stroke="#777777", text="candidate plan", fs=8)
    d.arrow(XS[2] + W, 230, XS[3], 230, stroke=RED, text="✗ violation", fs=8,
            fontcol=RED, lw=1.6)
    # feedback loop (re-verify repaired suffix) along the bottom
    d.arrow(XS[3] + W / 2, 430, XS[2] + W / 2, 430, stroke=RED, dashed=True,
            text="re-verify repaired suffix", fs=8, fontcol=RED, lw=1.4)

    # ---- footer legend ----
    d.box(20, 438, 16, 14, "", fill="#ffffff", stroke=GRAY, rounded=False)
    d.text(40, 436, 320, 18, "standard hierarchical planning", fs=9, align="left")
    d.box(360, 438, 16, 14, "", fill=PINK, stroke=RED, rounded=False)
    d.text(380, 436, 360, 18, "★ SAGE's contribution: gate + edit", fs=9,
           fontcol=RED, bold=True, align="left")
    return d


def main():
    global FRAMES
    FRAMES = {n: data_uri(os.path.join(FR, n + ".png"))
              for n in ("overview", "fridge_closed", "fridge_open", "placed")}

    f1 = build_fig1()
    with open(os.path.join(OUT, "fig1_storyboard.drawio"), "w") as fh:
        fh.write(f1.xml())
    print("wrote images/fig1_storyboard.drawio")

    f2 = build_fig2()
    with open(os.path.join(OUT, "fig2_overview.drawio"), "w") as fh:
        fh.write(f2.xml())
    print("wrote images/fig2_overview.drawio")


if __name__ == "__main__":
    main()
