#!/usr/bin/env python3
"""gen_figures.py
=================
Generate the ICRA figures from the grid CSVs. Run with miniconda python:

  DISPLAY= /home/keti/miniconda3/bin/python scripts/gen_figures.py

Outputs (PDF, vector) into paper/generated/figures/ :
  cost_quality_pareto.pdf   tokens/plan (log x) vs completeness, SAGE highlighted
  difficulty_bars.pdf       completeness by difficulty: Direct vs best baseline vs SAGE
  scaling_curve.pdf         completeness vs model size for Direct / best baseline / SAGE

SAGE is the safe_refine config (SAGE-Fixed rows relabelled); the unguarded
duplicate is dropped, matching gen_tables.py.
"""
from __future__ import annotations
import csv, glob, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 13, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "lines.linewidth": 2.0, "lines.markersize": 7, "axes.linewidth": 1.0,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
OUT = os.path.join(_ROOT, "ICRA", "figures")  # ICRA build is the canonical target
BRAND = "SAGE"  # paper-facing name; raw CSVs still use method "SAGE"

MODEL_SIZE = {"llama3.2": 3, "qwen2.5:7b": 7, "mistral-nemo": 12,
              "qwen2.5:14b": 14, "qwen2.5:14B": 14, "qwen2.5:32b-instruct": 32}
MODEL_NAME = {"llama3.2": "Llama-3.2 3B", "qwen2.5:7b": "Qwen2.5 7B",
              "mistral-nemo": "Mistral-Nemo 12B", "qwen2.5:14b": "Qwen2.5 14B",
              "qwen2.5:14B": "Qwen2.5 14B", "qwen2.5:32b-instruct": "Qwen2.5 32B"}
METHODS = ["Direct", "CoT", "Few-Shot CoT", "Self-Refine", "ReAct",
           "Hierarchical", "Hierarchical Few-Shot", "SAGE"]


def load():
    rows = []
    for f in glob.glob(os.path.join(_ROOT, "results", "grid_*_s*", "results.csv")):
        rows += list(csv.DictReader(open(f)))
    out, seen = [], set()
    for r in rows:
        if r.get("model", "").endswith(":14B"):
            r = dict(r); r["model"] = r["model"][:-1] + "b"
        if r["method"] in ("SAGE", "SAGE-Fixed"):
            r = dict(r); r["method"] = "SAGE"
            key = (r.get("model"), r.get("seed"), r.get("task_id"))
            if key in seen:
                continue
            seen.add(key)
        out.append(r)
    return out


def disp(m):
    return BRAND if m == "SAGE" else m


def fv(r, m):
    try:
        v = float(r.get(m) or "nan"); return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def mean_metric(rows, model, method, metric, difficulty=None):
    vs = [fv(r, metric) for r in rows
          if r.get("model") == model and r.get("method") == method
          and (difficulty is None or r.get("difficulty") == difficulty)]
    vs = [v for v in vs if not np.isnan(v)]
    return float(np.mean(vs)) if vs else np.nan


def models_present(rows):
    # only known models — guards against corrupted CSV rows (e.g. a stray model='1')
    return sorted({r["model"] for r in rows if r.get("model") in MODEL_SIZE},
                  key=lambda m: MODEL_SIZE.get(m, 99))


def best_baseline(rows, model, metric="completeness"):
    cand = [m for m in METHODS if m != "SAGE"]
    best, name = -1, None
    for me in cand:
        v = mean_metric(rows, model, me, metric)
        if v == v and v > best:
            best, name = v, me
    return name, best


def fig_pareto(rows):
    plt.figure(figsize=(3.5, 2.8))
    models = models_present(rows)
    # pool across models: one point per method (mean tokens, mean completeness)
    for me in METHODS:
        xs = [fv(r, "total_tokens") for r in rows if r["method"] == me]
        ys = [fv(r, "completeness") for r in rows if r["method"] == me]
        xs = [v for v in xs if not np.isnan(v)]; ys = [v for v in ys if not np.isnan(v)]
        if not xs or not ys:
            continue
        x, y = np.mean(xs), np.mean(ys)
        if me == "SAGE":
            plt.scatter([x], [y], s=70, marker="*", color="crimson", zorder=5, label=BRAND)
            plt.annotate(BRAND, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=11, color="crimson")
        else:
            plt.scatter([x], [y], s=24, color="steelblue")
            plt.annotate(me.replace("Hierarchical", "Hier."), (x, y),
                         textcoords="offset points", xytext=(3, -3), fontsize=11, color="gray")
    plt.xscale("log")
    plt.xlabel("Tokens / plan (log)"); plt.ylabel("Completeness")
    plt.title("Cost–quality trade-off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "cost_quality_pareto.pdf")); plt.close()


def fig_difficulty(rows):
    # Single panel pooled over all models (was 1 panel/model -> unreadable at
    # column width). Grouped bars: Easy/Medium/Hard x {Direct, Hier-FS, SAGE}.
    diffs = ["easy", "medium", "hard"]

    def pooled(method, d):
        vs = [fv(r, "completeness") for r in rows
              if r.get("method") == method and r.get("difficulty") == d]
        vs = [v for v in vs if not np.isnan(v)]
        return float(np.mean(vs)) if vs else np.nan

    series = [("Direct", "Direct", "0.6"),
              ("Hierarchical Few-Shot", "Hier. FS", "steelblue"),
              ("SAGE", "SAGE", "crimson")]
    # Fonts at 60% of the global size 13 (-> ~7.8) per request: keeps Fig 3
    # compact without touching the other two figures this script emits.
    FS = 7.8
    fig, ax = plt.subplots(figsize=(3.5, 1.95))
    w = 0.27
    for i, (me, lab, col) in enumerate(series):
        vals = [pooled(me, d) for d in diffs]
        ax.bar(np.arange(len(diffs)) + i * w, vals, w, label=lab,
               color=col, edgecolor="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(diffs)) + w)
    ax.set_xticklabels(["Easy", "Medium", "Hard"], fontsize=FS)
    ax.tick_params(axis="y", labelsize=FS)
    ax.set_ylabel("Completeness", fontsize=FS); ax.set_ylim(0, 1.0)
    # legend OUTSIDE (above) so it never overlaps the (tall) bars
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=FS, frameon=False, handlelength=1.2, columnspacing=1.2)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "difficulty_bars.pdf")); plt.close()


def fig_scaling(rows):
    models = models_present(rows)
    sizes = [MODEL_SIZE[m] for m in models]
    plt.figure(figsize=(3.5, 2.8))
    for me, col, mk in [("Direct", "gray", "o"), ("SAGE", "crimson", "*")]:
        ys = [mean_metric(rows, m, me, "completeness") for m in models]
        plt.plot(sizes, ys, marker=mk, color=col, label=disp(me), markersize=8 if me == "SAGE" else 5)
    # best baseline per model
    yb = [best_baseline(rows, m)[1] for m in models]
    plt.plot(sizes, yb, marker="s", color="steelblue", linestyle="--", label="best baseline", markersize=4)
    plt.xlabel("Model size (B params)"); plt.ylabel("Completeness")
    plt.title("Scaling"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "scaling_curve.pdf")); plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    if not rows:
        raise SystemExit("no grid rows")
    fig_pareto(rows); print("wrote cost_quality_pareto.pdf")
    fig_difficulty(rows); print("wrote difficulty_bars.pdf")
    if len(models_present(rows)) >= 2:
        fig_scaling(rows); print("wrote scaling_curve.pdf")
    print("models:", models_present(rows))


if __name__ == "__main__":
    main()
