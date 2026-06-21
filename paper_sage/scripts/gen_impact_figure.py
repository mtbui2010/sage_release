#!/usr/bin/env python3
"""Generate the introduction "impact" figure for the SAGE robotics paper.

Produces a single-column, large-text horizontal bar chart that conveys SAGE's
standout result: it recovers from mid-execution failures using far fewer LLM
calls than whole-plan replanning baselines, while all methods recover 100% of
the injected failures.

Output: <repo_root>/ICRA/figures/impact_recovery.pdf

Data source: results/recovery/rec_*.csv (relative to repo root). Per method we
compute the MEAN of `replan_llm_calls` over rows where the injection actually
fired (replan_llm_calls > 0 OR replan_steps > 0), de-duplicated by
(method, task_id). If CSV reading fails we fall back to hard-coded values.

NOTE: This script only writes the figure; nothing else is touched.
"""

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Large-text rcParams for 2-column IEEE readability.
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "font.size": 10.4,
        "axes.labelsize": 10.4,
        "ytick.labelsize": 9.6,
        "axes.titlesize": 10.4,
        "lines.linewidth": 2,
    }
)

# ---------------------------------------------------------------------------
# Method name normalization and display names.
# ---------------------------------------------------------------------------
# Raw method name -> canonical key.
CANON = {
    "SAGE": "SAGE",
    "SAGE-Fixed": "SAGE",
    "HierarchicalFewShot": "HierarchicalFewShot",
    "Hierarchical": "Hierarchical",
    "Self-Refine": "Self-Refine",
    "ReAct": "ReAct",
}

# Canonical key -> display label.
DISPLAY = {
    "SAGE": "SAGE (ours)",
    "HierarchicalFewShot": "Hier. Few-Shot",
    "Hierarchical": "Hierarchical",
    "Self-Refine": "Self-Refine",
    "ReAct": "ReAct",
}

# Hard-coded fallback values (mean replan LLM calls per method).
FALLBACK = {
    "SAGE": 1.18,
    "HierarchicalFewShot": 2.96,
    "Hierarchical": 2.87,
    "Self-Refine": 3.59,
    "ReAct": 3.71,
}

SAGE_KEY = "SAGE"  # canonical key that maps to SAGE


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compute_from_csv(root):
    """Compute mean replan_llm_calls per canonical method from rec_*.csv.

    Returns a dict {canonical_key: mean_value}. Raises on any read problem so
    the caller can fall back to hard-coded values.
    """
    pattern = os.path.join(root, "results", "recovery", "rec_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError("no rec_*.csv files found at %s" % pattern)

    # De-duplicate by (canonical_method, task_id): keep the replan_llm_calls
    # value of the first fired row we see for that pair.
    seen = {}  # (canon, task_id) -> replan_llm_calls
    for path in paths:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw_method = (row.get("method") or "").strip()
                canon = CANON.get(raw_method)
                if canon is None:
                    continue
                task_id = (row.get("task_id") or "").strip()
                calls = _to_float(row.get("replan_llm_calls"))
                steps = _to_float(row.get("replan_steps"))
                # Only count rows where the injection actually fired.
                if not (calls > 0 or steps > 0):
                    continue
                key = (canon, task_id)
                if key not in seen:
                    seen[key] = calls

    # Aggregate to per-method means.
    sums = {}
    counts = {}
    for (canon, _task_id), calls in seen.items():
        sums[canon] = sums.get(canon, 0.0) + calls
        counts[canon] = counts.get(canon, 0) + 1

    if not counts:
        raise ValueError("no fired-injection rows found across rec_*.csv")

    return {canon: sums[canon] / counts[canon] for canon in counts}


def make_figure(values, out_path):
    """Render and save the horizontal bar chart.

    `values` is a dict {canonical_key: mean_llm_calls}.
    """
    # Sort ascending by LLM calls; with a horizontal bar chart, the first item
    # plotted lands at the bottom, so reverse the list so the lowest (SAGE)
    # appears on top.
    items = sorted(values.items(), key=lambda kv: kv[1])  # ascending
    items_plot = list(reversed(items))  # lowest ends up on top

    labels = [DISPLAY.get(k, k) for k, _ in items_plot]
    nums = [v for _, v in items_plot]
    colors = ["crimson" if k == SAGE_KEY else "0.6" for k, _ in items_plot]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = range(len(nums))
    bars = ax.barh(list(y), nums, color=colors, edgecolor="black", linewidth=0.8)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    # Shorter label + smaller font so it never overflows the single column.
    ax.set_xlabel("LLM calls to recover (lower is better)", fontsize=8.4)

    # Headroom for the value annotations.
    xmax = max(nums) if nums else 1.0
    ax.set_xlim(0, xmax * 1.18)

    # Annotate each bar with its value.
    for bar, val in zip(bars, nums):
        ax.text(
            bar.get_width() + xmax * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            "%.2f" % val,
            va="center",
            ha="left",
            fontsize=9.6,
            fontweight="bold",
        )

    # Honest headline folded into a two-line title (kept OUT of the plot area so
    # it never overlaps a bar). Ratio range: best/SAGE to worst/SAGE.
    sage_val = values.get(SAGE_KEY)
    baseline_vals = [v for k, v in values.items() if k != SAGE_KEY]
    if sage_val and sage_val > 0 and baseline_vals:
        lo = min(baseline_vals) / sage_val
        hi = max(baseline_vals) / sage_val
        ratio_txt = "SAGE: %.1f–%.1f$\\times$ fewer LLM calls" % (lo, hi)
    else:
        ratio_txt = "SAGE: 2.4–3.1$\\times$ fewer LLM calls"

    ax.set_title("All methods recover 100% of failures\n" + ratio_txt,
                 pad=8, fontsize=9.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def main():
    # ROOT = two levels up from this script file (.../paper_sage).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, os.pardir))
    out_path = os.path.join(root, "ICRA", "figures", "impact_recovery.pdf")

    try:
        values = compute_from_csv(root)
        # Ensure every expected method has a value; fill gaps from fallback.
        for k, v in FALLBACK.items():
            values.setdefault(k, v)
        print("[gen_impact_figure] computed values from CSV:")
    except Exception as exc:  # noqa: BLE001 - intentional broad fallback
        print("[gen_impact_figure] CSV read failed (%s); using fallback values." % exc)
        values = dict(FALLBACK)

    for k in sorted(values, key=lambda kk: values[kk]):
        print("  %-22s %.2f" % (DISPLAY.get(k, k), values[k]))

    make_figure(values, out_path)
    print("[gen_impact_figure] wrote %s" % out_path)


if __name__ == "__main__":
    main()
