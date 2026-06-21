#!/usr/bin/env python3
"""Fig 2: SAGE method overview, GAIA-style 4-panel pipeline (enriched).

Four colour-coded stage panels, each with a hand-drawn vector ICON, sub-boxes /
JSON-and-state snippets, and real AI2-THOR frames:

  1. Hierarchical Decompose   (gray)  tree icon  + input RGB scene + sub-goals
  2. Retrieve + Expand        (gray)  db icon    + retrieved few-shot JSON
  3. ★ Symbolic Action-Gate   (red)   shield✓    + full 5-field symbolic state
  4. ★ Sub-goal Edit + Execute(red)   pencil     + repaired suffix + 2 frames

Panels 3-4 are this paper's contribution (red); 1-2 are standard hierarchical
planning (gray). Arrows carry data labels (sub-goals / candidate plan /
certified plan); a red loop shows gate -> edit (violation) and edit -> gate
(re-verify).

Outputs (editable vector + raster preview); the paper is NOT touched:
  images/fig2_overview.svg   images/fig2_overview.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                Circle, Polygon, Ellipse)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FR = os.path.join(ROOT, "images", "frames")
OUT = os.path.join(ROOT, "images")

FW, FH = 7.7, 3.75

GRAY_F, GRAY_E = "#f3f3f1", "#8a8a8a"
RED_F, RED_E = "#fdecea", "#c0392b"
HEAD_G, HEAD_R = "#e4e4e0", "#f6cdc6"
GREEN = "#2e8b57"
CODE = "#eef3f7"
CODE_E = "#9bb6cc"
BLUE = "#1f6fb2"
ORANGE = "#d2730a"

W = 22.0
XS = [1.5, 26.0, 50.5, 75.0]


# ----------------------------------------------------------------------------- icons
def icon_tree(ax, cx, cy, s, col):
    nodes = [(cx, cy + 1.5 * s), (cx - 1.6 * s, cy - 1.4 * s), (cx + 1.6 * s, cy - 1.4 * s)]
    for n in nodes[1:]:
        ax.plot([nodes[0][0], n[0]], [nodes[0][1], n[1]], color=col, lw=1.0, zorder=5)
    for n in nodes:
        ax.add_patch(Circle(n, 0.7 * s, fc="white", ec=col, lw=1.2, zorder=6))


def icon_db(ax, cx, cy, s, col):
    w, h = 2.6 * s, 3.2 * s
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="white", ec=col, lw=1.2, zorder=5))
    for dy in (h / 2, h / 6, -h / 6):
        ax.add_patch(Ellipse((cx, cy + dy), w, 1.1 * s, fc="white", ec=col, lw=1.2, zorder=6))


def icon_shield(ax, cx, cy, s, col):
    pts = [(cx, cy + 2.0 * s), (cx + 1.7 * s, cy + 1.1 * s), (cx + 1.7 * s, cy - 0.6 * s),
           (cx, cy - 2.1 * s), (cx - 1.7 * s, cy - 0.6 * s), (cx - 1.7 * s, cy + 1.1 * s)]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=col, lw=1.3, zorder=5))
    ax.plot([cx - 0.8 * s, cx - 0.15 * s, cx + 0.95 * s],
            [cy - 0.1 * s, cy - 0.9 * s, cy + 0.9 * s], color=GREEN, lw=1.6, zorder=6,
            solid_capstyle="round")


def icon_pencil(ax, cx, cy, s, col):
    import math
    ang = math.radians(45)
    dx, dy = math.cos(ang), math.sin(ang)
    L, w = 3.0 * s, 0.9 * s
    bx, by = cx - L / 2 * dx, cy - L / 2 * dy
    tx, ty = cx + L / 2 * dx, cy + L / 2 * dy
    px, py = -dy * w / 2, dx * w / 2
    body = [(bx + px, by + py), (tx - 1.0 * s * dx + px, ty - 1.0 * s * dy + py),
            (tx - 1.0 * s * dx - px, ty - 1.0 * s * dy - py), (bx - px, by - py)]
    ax.add_patch(Polygon(body, closed=True, fc="#ffe6c2", ec=col, lw=1.1, zorder=5))
    tip = [(tx - 1.0 * s * dx + px, ty - 1.0 * s * dy + py), (tx, ty),
           (tx - 1.0 * s * dx - px, ty - 1.0 * s * dy - py)]
    ax.add_patch(Polygon(tip, closed=True, fc=col, ec=col, lw=1.0, zorder=6))


ICONS = {"tree": icon_tree, "db": icon_db, "shield": icon_shield, "pencil": icon_pencil}


# ----------------------------------------------------------------------------- helpers
def panel(ax, x0, num, title, icon, ours=False):
    y0, h = 5, 84
    fc, ec, lw = (RED_F, RED_E, 1.8) if ours else (GRAY_F, GRAY_E, 1.1)
    ax.add_patch(FancyBboxPatch((x0, y0), W, h, boxstyle="round,pad=0.1,rounding_size=1.5",
                 fc=fc, ec=ec, lw=lw, zorder=2))
    hf = HEAD_R if ours else HEAD_G
    ax.add_patch(FancyBboxPatch((x0 + 1, y0 + h - 12.5), W - 2, 11,
                 boxstyle="round,pad=0.05,rounding_size=1.0", fc=hf, ec=ec, lw=1.0, zorder=3))
    ICONS[icon](ax, x0 + 4.0, y0 + h - 6.8, 1.3, RED_E if ours else "#555")
    # 2-line title (centered in the space right of the icon) so long names fit
    ax.text(x0 + 14.3, y0 + h - 6.8, title, ha="center", va="center",
            fontsize=6.2, fontweight="bold", color=RED_E if ours else "black",
            zorder=4, linespacing=1.05)


def codebox(ax, x0, y0, w, h, lines, fs=5.4, ec=CODE_E):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05,rounding_size=0.6",
                 fc=CODE, ec=ec, lw=0.8, zorder=3))
    ax.text(x0 + 0.8, y0 + h - 1.3, lines, ha="left", va="top", fontsize=fs,
            family="monospace", zorder=4, linespacing=1.3)


def place_image(fig, path, x0, y0, w, label=None):
    img = mpimg.imread(path)
    r = img.shape[0] / img.shape[1]
    h_units = (w / 100.0 * FW * r) / FH * 100.0
    a = fig.add_axes([x0 / 100.0, y0 / 100.0, w / 100.0, h_units / 100.0], zorder=6)
    a.imshow(img); a.axis("off")
    a.add_patch(Rectangle((0, 0), 1, 1, transform=a.transAxes, fill=False,
                edgecolor="#4f5670", linewidth=1.0))
    if label:
        a.set_title(label, fontsize=5.0, pad=1.5)
    return h_units


def harr(ax, x, color, label=None, lcol=None):
    ax.add_patch(FancyArrowPatch((x, 47), (x + 2.9, 47), arrowstyle="-|>",
                 mutation_scale=12, lw=1.7, color=color, zorder=5))
    if label:
        ax.text(x + 1.45, 51.5, label, ha="center", va="bottom", fontsize=4.9,
                color=lcol or color, fontweight="bold")


# ----------------------------------------------------------------------------- main
def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---- panel 1: decompose ----
    panel(ax, XS[0], 1, "1. Hierarchical\nDecompose", "tree")
    place_image(fig, os.path.join(FR, "overview.png"), XS[0] + 4.5, 54, W - 9,
                label="input: RGB scene + task")
    ax.text(XS[0] + W / 2, 47, "“cook a potato, put in recycle bin”", ha="center",
            fontsize=5.4, style="italic", color="0.3")
    ax.text(XS[0] + W / 2, 41.5, "sub-goals", ha="center", fontsize=5.8, style="italic", color="0.45")
    for i, g in enumerate(["g₁  get potato", "g₂  cook potato", "g₃  recycle"]):
        ax.add_patch(FancyBboxPatch((XS[0] + 2.5, 32 - i * 7.5), W - 5, 5.6,
                     boxstyle="round,pad=0.04,rounding_size=0.5", fc="white", ec=GRAY_E, lw=0.9, zorder=3))
        ax.text(XS[0] + 3.4, 34.8 - i * 7.5, g, ha="left", va="center", fontsize=5.8, zorder=4)

    # ---- panel 2: retrieve + expand ----
    panel(ax, XS[1], 2, "2. Retrieve\n+ Expand", "db")
    ax.text(XS[1] + W / 2, 68.5, "hybrid memory  (seed + live episodes)", ha="center",
            fontsize=5.4, style="italic", color="0.4")
    codebox(ax, XS[1] + 2, 44, W - 4, 22,
            "retrieved few-shot:\n {task: \"...recycle\",\n  reasoning: \"open\n    fridge first\",\n  plan_text:\n   Find→Open→Pick\n    →Close→Place}")
    ax.text(XS[1] + W / 2, 39.5, "+ LLM expands each sub-goal", ha="center", fontsize=5.5, color="0.25")
    ax.add_patch(FancyBboxPatch((XS[1] + 2.5, 16), W - 5, 18,
                 boxstyle="round,pad=0.05,rounding_size=0.6", fc="white", ec=GRAY_E, lw=0.9, zorder=3))
    ax.text(XS[1] + W / 2, 28.5, "candidate plan", ha="center", fontsize=5.6, fontweight="bold", zorder=4)
    ax.text(XS[1] + W / 2, 22.5, "Find → Pick → Cook\n→ Place  (steps g₁…g₃)", ha="center",
            va="center", fontsize=5.3, family="monospace", color="0.2", zorder=4)

    # ---- panel 3: SYMBOLIC ACTION-GATE (ours) ----
    panel(ax, XS[2], 3, "★ 3. Symbolic\nAction-Gate", "shield", ours=True)
    ax.text(XS[2] + W / 2, 68.5, "verify-before-execute  (no LLM)", ha="center", fontsize=5.4,
            style="italic", color=RED_E)
    # full 5-field symbolic-state table
    cols = ["arrived", "found", "holding", "opened", "on"]
    tx0, ty0 = XS[2] + 2.0, 56
    cw = (W - 4) / len(cols)
    ax.add_patch(FancyBboxPatch((XS[2] + 1.6, 40), W - 3.2, 22,
                 boxstyle="round,pad=0.05,rounding_size=0.5", fc="white", ec=RED_E, lw=1.1, zorder=3))
    for j, c in enumerate(cols):
        ax.text(tx0 + cw * (j + 0.5), ty0 + 3.5, c, ha="center", fontsize=4.2,
                color="0.3", rotation=0, zorder=4)
    rows = [("Pick potato", ["✓", "✓", "✗", "✗", "·"], False, "step under test"),
            ("after Open", ["✓", "✓", "·", "✓", "·"], True, "precondition ok")]
    for i, (lab, vals, ok, note) in enumerate(rows):
        yy = ty0 - 2 - i * 6
        ax.text(tx0 + 0.2, yy, lab, ha="left", fontsize=4.6,
                color=GREEN if ok else RED_E, fontweight="bold", zorder=4)
        for j, v in enumerate(vals):
            c = RED_E if v == "✗" else (GREEN if v == "✓" else "0.45")
            ax.text(tx0 + cw * (j + 0.5), yy - 2.6, v, ha="center", fontsize=5.4, color=c, zorder=4)
    ax.add_patch(FancyBboxPatch((XS[2] + 2.5, 22), W - 5, 13,
                 boxstyle="round,pad=0.04,rounding_size=0.5", fc=RED_F, ec=RED_E, lw=1.3, zorder=3))
    ax.text(XS[2] + W / 2, 28.5, "✗ violation\nPick requires opened(fridge)", ha="center", va="center",
            fontsize=5.5, color=RED_E, fontweight="bold", zorder=4)

    # ---- panel 4: EDIT + EXECUTE (ours) ----
    panel(ax, XS[3], 4, "★ 4. Sub-goal\nEdit + Execute", "pencil", ours=True)
    ax.text(XS[3] + W / 2, 68.5, "✎ regenerate the failed suffix only", ha="center", fontsize=5.4,
            style="italic", color=RED_E)
    codebox(ax, XS[3] + 2, 53, W - 4, 13,
            "g₁ repaired suffix:\n Open→Pick→Close→…", fs=5.4, ec=RED_E)
    fw2 = (W - 6) / 2
    place_image(fig, os.path.join(FR, "fridge_open.png"), XS[3] + 2.5, 30, fw2, label="open")
    place_image(fig, os.path.join(FR, "placed.png"), XS[3] + 2.5 + fw2 + 1, 30, fw2, label="placed")
    ax.text(XS[3] + W / 2, 25, "certified plan → AI2-THOR", ha="center", fontsize=5.6,
            color=GREEN, fontweight="bold")

    # ---- inter-panel arrows with data labels ----
    harr(ax, XS[1] - 3.1, "#555", "sub-goals", "0.3")
    harr(ax, XS[2] - 3.1, "#555", "candidate\nplan", "0.3")
    harr(ax, XS[3] - 3.1, RED_E, "✗ violation", RED_E)
    # edit -> gate feedback (re-verify), dashed along the bottom
    ax.add_patch(FancyArrowPatch((XS[3] + 1, 13.5), (XS[2] + W - 1, 13.5),
                 arrowstyle="-|>", mutation_scale=9, lw=1.1, color=RED_E,
                 connectionstyle="arc3,rad=-0.32", linestyle=(0, (3, 2)), zorder=5))
    ax.text((XS[2] + XS[3]) / 2 + W / 2, 9.3, "re-verify repaired suffix", ha="center",
            fontsize=5.2, color=RED_E)

    # ---- footer legend ----
    ax.add_patch(Rectangle((1.5, 0.5), 3, 2.4, fc=GRAY_F, ec=GRAY_E, lw=1.0))
    ax.text(5.4, 1.7, "standard hierarchical planning", ha="left", va="center", fontsize=5.6)
    ax.add_patch(Rectangle((40, 0.5), 3, 2.4, fc=RED_F, ec=RED_E, lw=1.4))
    ax.text(43.9, 1.7, "★ SAGE's contribution: gate (verify-before-execute) + edit (local repair)",
            ha="left", va="center", fontsize=5.6, color=RED_E, fontweight="bold")

    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_overview.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote images/fig2_overview.svg + .png")


if __name__ == "__main__":
    main()
