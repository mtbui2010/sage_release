#!/usr/bin/env python3
"""Fig 1 (teaser): SAGE gate-and-edit storyboard.

A timeline in the style of an embodied-agent replanning figure: a left "LLM
Planner" rail, three time bands (t=0 / t=3 / t=8), colour-coded message boxes
(Instruction / High-level Plan / Gate verdict / Edited suffix / Observation),
and a real AI2-THOR observation frame on the right of each band.

The story: a naive plan would Pick the potato before opening the fridge; SAGE's
symbolic GATE catches the precondition violation and EDIT regenerates only the
failed sub-goal's suffix (Open -> Pick -> Close -> ...), yielding a certified
plan -- all without a full whole-plan replan.

Outputs (editable vector + raster preview), nothing in the paper is touched:
  images/fig1_storyboard.svg
  images/fig1_storyboard.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FR = os.path.join(ROOT, "images", "frames")
OUT = os.path.join(ROOT, "images")

FW, FH = 7.4, 4.3  # figure size (inches)

# palette
OLIVE, OLIVE_E = "#d9d7b8", "#8a875a"
INSTR_E = "#1f6fb2"
SLATE, SLATE_E = "#8a93a8", "#4f5670"
RED, PINK = "#c0392b", "#fdecea"
GREEN = "#2e8b57"
RAIL = "#cfcb9b"


def box(ax, x0, y0, w, h, text, fc, ec, fs=7.0, italic=False, bold=False, lw=1.0,
        align="left", tcol="black", dashed=False):
    ls = (0, (4, 2)) if dashed else "solid"
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.7",
                 linewidth=lw, facecolor=fc, edgecolor=ec, linestyle=ls, zorder=3))
    tx = x0 + 1.0 if align == "left" else x0 + w / 2
    ha = "left" if align == "left" else "center"
    ax.text(tx, y0 + h / 2, text, ha=ha, va="center", fontsize=fs, zorder=4,
            style="italic" if italic else "normal",
            fontweight="bold" if bold else "normal", color=tcol)


def place_image(fig, path, x0, y0, w):
    """Place a frame in 0-100x0-100 main-axis coords, preserving aspect."""
    img = mpimg.imread(path)
    r = img.shape[0] / img.shape[1]               # h/w
    w_in = w / 100.0 * FW
    h_in = w_in * r
    h_units = h_in / FH * 100.0
    a = fig.add_axes([x0 / 100.0, y0 / 100.0, w / 100.0, h_units / 100.0], zorder=5)
    a.imshow(img); a.axis("off")
    a.add_patch(Rectangle((0, 0), 1, 1, transform=a.transAxes, fill=False,
                edgecolor=SLATE_E, linewidth=1.3))
    return h_units


def robot(ax, x, y, s=1.0):
    ax.add_patch(FancyBboxPatch((x - 1.6 * s, y - 1.4 * s), 3.2 * s, 2.6 * s,
                 boxstyle="round,pad=0.02,rounding_size=0.4", fc=SLATE, ec=SLATE_E,
                 lw=1.0, zorder=6))
    ax.add_patch(Circle((x - 0.7 * s, y + 0.1 * s), 0.45 * s, fc="white", ec=SLATE_E, lw=0.8, zorder=7))
    ax.add_patch(Circle((x + 0.7 * s, y + 0.1 * s), 0.45 * s, fc="white", ec=SLATE_E, lw=0.8, zorder=7))


def arr(ax, p0, p1, color="#555", cs=None, lw=1.2, dashed=True):
    kw = dict(arrowstyle="-|>", mutation_scale=9, linewidth=lw, color=color,
              shrinkA=2, shrinkB=2, zorder=2,
              linestyle=(0, (3, 2)) if dashed else "solid")
    if cs:
        kw["connectionstyle"] = cs
    ax.add_patch(FancyArrowPatch(p0, p1, **kw))


def cost_inset(fig, ax):
    """Small horizontal bar chart embedded near the t=8 outcome: LLM calls to
    recover from a failure, SAGE vs whole-plan replanning baselines."""
    # subtle background panel on the main axis to set the inset apart
    ax.add_patch(FancyBboxPatch((8.0, 3.6), 45, 13.6, boxstyle="round,pad=0.1,rounding_size=0.8",
                 fc="#fbfbfb", ec="0.78", lw=1.0, zorder=3))
    a = fig.add_axes([0.115, 0.052, 0.345, 0.105], zorder=8)
    methods = ["SAGE", "Hier. FS", "ReAct"]
    vals = [1.18, 2.96, 3.71]
    cols = [RED, "0.62", "0.62"]
    yy = list(range(len(vals)))[::-1]                  # SAGE on top
    a.barh(yy, vals, color=cols, edgecolor="black", linewidth=0.6, height=0.66)
    for y, v in zip(yy, vals):
        a.text(v + 0.08, y, f"{v:.2f}", va="center", ha="left", fontsize=5.2, fontweight="bold")
    a.set_yticks(yy); a.set_yticklabels(methods, fontsize=5.2)
    a.set_xticks([]); a.set_xlim(0, 4.5)
    for s in ("top", "right", "bottom"):
        a.spines[s].set_visible(False)
    a.set_title("LLM calls to recover  (lower is better)", fontsize=5.6, pad=2)
    # annotate in the empty space to the right of SAGE's short bar (top row)
    a.text(4.4, yy[0], "2.5–3.1×\nfewer", ha="right", va="center", fontsize=5.3,
           color=RED, fontweight="bold", linespacing=1.0)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # left planner rail
    ax.add_patch(FancyBboxPatch((1, 8), 5.5, 86, boxstyle="round,pad=0.02,rounding_size=1.2",
                 fc=RAIL, ec=OLIVE_E, lw=1.2, zorder=3))
    ax.text(3.7, 51, "LLM Planner", rotation=90, ha="center", va="center",
            fontsize=9.5, fontweight="bold", zorder=4)

    # right column header (right-aligned so it never clips the page edge)
    ax.text(99, 97.5, "Embodied Agent & AI2-THOR", ha="right", va="center",
            fontsize=8.0, fontweight="bold")

    IMG_X, IMG_W = 79, 19

    # ---- band separators + ticks ----
    for ysep in (67.5, 36.5):
        ax.plot([7, 99], [ysep, ysep], ls=(0, (2, 2)), color="0.6", lw=0.8, zorder=1)
    for ty, tlab in [(93, "t = 0"), (65, "t = 3"), (34, "t = 8")]:
        ax.text(7.2, ty, tlab, ha="left", va="center", fontsize=7.2,
                fontweight="bold", color="0.35")

    # ================= BAND 1 : t=0 =================
    box(ax, 9, 86, 46, 6.4, "Cook a potato and put it into the recycle bin.",
        "white", INSTR_E, fs=7.2, italic=True, lw=1.3)
    box(ax, 9, 75.5, 55, 6.4,
        "Plan:  Find potato → Pick potato → Cook → Place recyclebin",
        OLIVE, OLIVE_E, fs=6.8)
    arr(ax, (6.5, 89), (9, 89), color=OLIVE_E)          # planner -> instruction
    h1 = place_image(fig, os.path.join(FR, "overview.png"), IMG_X, 71, IMG_W)
    robot(ax, 74.5, 71 + h1 / 2)
    arr(ax, (64, 78.7), (IMG_X - 0.5, 71 + h1 / 2), color="0.5")  # execute

    # ================= BAND 2 : t=3 =================
    box(ax, 9, 59, 44, 6.0, "Observation:  fridge is closed — cannot Pick the potato.",
        SLATE, SLATE_E, fs=6.8, tcol="white")
    box(ax, 9, 50.5, 47, 6.0,
        "✗  GATE: precondition violated  (¬opened(fridge) before Pick)",
        PINK, RED, fs=6.8, bold=True, tcol=RED, lw=1.4)
    box(ax, 9, 40.5, 60, 7.2,
        "✎  EDIT (suffix only):  Open fridge → Pick potato → Close fridge → …",
        PINK, RED, fs=6.8, bold=True, tcol=RED, lw=1.5, dashed=True)
    h2 = place_image(fig, os.path.join(FR, "fridge_closed.png"), IMG_X, 40, IMG_W)
    robot(ax, 74.5, 40 + h2 / 2)
    arr(ax, (IMG_X - 0.5, 40 + h2 / 2), (53, 62), color=SLATE_E)   # perceive -> obs

    # ================= BAND 3 : t=8 =================
    box(ax, 9, 27.5, 40, 6.0, "✓  GATE: plan certified — executes safely",
        "#e7f4ec", GREEN, fs=6.8, bold=True, tcol=GREEN, lw=1.4)
    box(ax, 9, 18.5, 47, 6.0, "Observation:  potato placed in the recycle bin  ✓",
        SLATE, SLATE_E, fs=6.8, tcol="white")
    h3 = place_image(fig, os.path.join(FR, "placed.png"), IMG_X, 13, IMG_W)
    robot(ax, 74.5, 13 + h3 / 2)
    arr(ax, (49, 30.5), (IMG_X - 0.5, 13 + h3 / 2), color="0.5")   # execute certified

    # ---- cost inset: "LLM calls to recover" (the quantitative payoff) ----
    cost_inset(fig, ax)

    # ---- legend (lowered to the very bottom so it clears the cost inset) ----
    leg = [("Instruction", "white", INSTR_E), ("High-level plan", OLIVE, OLIVE_E),
           ("Gate verdict", PINK, RED), ("Edited suffix", PINK, RED),
           ("Observation", SLATE, SLATE_E)]
    lx = 9
    for name, fc, ec in leg:
        ax.add_patch(FancyBboxPatch((lx, 0.6), 2.4, 2.4, boxstyle="round,pad=0.02,rounding_size=0.5",
                     fc=fc, ec=ec, lw=1.0, zorder=3))
        ax.text(lx + 3.0, 1.8, name, ha="left", va="center", fontsize=6.3)
        lx += 3.0 + 1.0 + len(name) * 1.05

    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_storyboard.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote images/fig1_storyboard.svg + .png")


if __name__ == "__main__":
    main()
