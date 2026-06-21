#!/usr/bin/env python3
"""SAGE pipeline figure for the Method section.

Left-to-right, icon-annotated so the diagram reads at a glance:
  task -> hierarchical decompose -> retrieve few-shot + LLM expand
       -> [SAGE design] GATE (verify-before-execute) --pass(checkmark)--> certified
          --violation(cross)--> EDIT (pencil: regenerate the failed sub-goal's
            suffix) -- loops back into the gate.

The two boxes that are THIS PAPER'S design (GATE + EDIT) sit on a tinted red
panel with a dashed border and a star badge "SAGE's design"; the decompose /
retrieve+expand stages are standard hierarchical-few-shot planning (gray).

Font is small (~6.7pt) per request; the figure is placed at \textwidth.
Writes: <repo_root>/ICRA/figures/method_pipeline.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, FancyBboxPatch as _Box

plt.rcParams.update({
    "font.size": 6.7,
    "font.family": "DejaVu Sans",   # has the check/cross/pencil/star glyphs
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

GRAY, GRAY_E = "#ededed", "#707070"
GATE, GATE_E = "#bcd8f0", "#1f6fb2"
EDIT, EDIT_E = "#f8d3a3", "#d2730a"
IO,   IO_E   = "#dfeede", "#3a7a3a"
SAGE_RED     = "#c0392b"
GREEN        = "#2e8b57"


def box(ax, cx, cy, w, h, text, fc, ec, bold=False, fs=6.7, lw=1.0):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle="round,pad=0.015,rounding_size=0.05",
                 linewidth=lw, facecolor=fc, edgecolor=ec, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4)
    return {"l": (cx-w/2, cy), "r": (cx+w/2, cy),
            "t": (cx, cy+h/2), "b": (cx, cy-h/2), "c": (cx, cy)}


def arr(ax, p0, p1, color="#333", cs=None, lw=1.3):
    kw = dict(arrowstyle="-|>", mutation_scale=10, linewidth=lw, color=color,
              shrinkA=2, shrinkB=2, zorder=2)
    if cs:
        kw["connectionstyle"] = cs
    ax.add_patch(FancyArrowPatch(p0, p1, **kw))


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    out = os.path.join(root, "ICRA", "figures", "method_pipeline.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 1.95))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    ym, ye, bh = 12.0, 32.0, 15.0

    # tinted panel highlighting SAGE's own design (behind everything)
    px0, py0, px1, py1 = 50.0, 1.5, 79.5, 44.0
    ax.add_patch(_Box((px0, py0), px1-px0, py1-py0,
                 boxstyle="round,pad=0.2,rounding_size=1.5",
                 facecolor="#fdecea", edgecolor=SAGE_RED, linewidth=2.2,
                 linestyle=(0, (5, 3)), zorder=0))
    ax.text((px0+px1)/2, py1-1.0, "★ SAGE's design (gate + edit)",
            color=SAGE_RED, fontsize=7.0, fontweight="bold",
            ha="center", va="top", zorder=5)

    # standard components (gray) + IO (green)
    b_task = box(ax, 8.5, ym, 14, bh, "Natural-\nlanguage task", IO, IO_E)
    b_dec  = box(ax, 27,  ym, 16, bh, "Hierarchical\ndecompose\n(sub-goals)", GRAY, GRAY_E)
    b_ret  = box(ax, 47,  ym, 18, bh, "Retrieve few-shot\n(hybrid memory)\n+ LLM expand", GRAY, GRAY_E)
    # SAGE design (colored, thicker borders)
    b_gate = box(ax, 69,  ym, 16, bh, "GATE\nverify-before-\nexecute", GATE, GATE_E, bold=True, lw=1.8)
    b_edit = box(ax, 60.5, ye, 25, 11, "✎ EDIT\nregenerate the failed\nsub-goal's suffix", EDIT, EDIT_E, bold=True, lw=1.8)
    # output
    b_out  = box(ax, 91,  ym, 14, bh, "Certified\nplan →\nAI2-THOR", IO, IO_E)

    # main path
    arr(ax, b_task["r"], b_dec["l"])
    arr(ax, b_dec["r"],  b_ret["l"])
    arr(ax, b_ret["r"],  b_gate["l"])
    # GATE -> certified (pass, green check)
    arr(ax, b_gate["r"], b_out["l"], color=GREEN)
    ax.text((b_gate["r"][0]+b_out["l"][0])/2, ym+3.5, "✓ pass",
            ha="center", va="bottom", fontsize=6.7, color=GREEN, fontweight="bold")
    # GATE -> EDIT (violation, red cross)
    arr(ax, b_gate["t"], (b_edit["r"][0], b_edit["r"][1]-1), color=SAGE_RED, cs="arc3,rad=-0.25")
    ax.text(75.0, ye-3.5, "✗ violation", ha="left", va="center",
            fontsize=6.7, color=SAGE_RED, fontweight="bold")
    # EDIT -> back into expand (re-verify the repaired suffix)
    arr(ax, (b_edit["l"][0], b_edit["l"][1]-1), (b_ret["t"][0]+2, b_ret["t"][1]),
        color=EDIT_E, cs="arc3,rad=0.25")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
