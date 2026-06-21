#!/usr/bin/env python3
"""analyze_predictive.py
========================
Headline ICRA empirical finding: the zero-token, offline
\texttt{precondition_strict} score PREDICTS whether a plan will succeed when
executed in AI2-THOR. Run AFTER the sim run, with miniconda python:

  DISPLAY= /home/keti/miniconda3/bin/python scripts/analyze_predictive.py

Reads the sim result CSVs (results/sim_*/*.csv or results/*/sim_*.csv), pairs
each executed plan's offline grounding score with its binary task-success, and
computes:
  * point-biserial correlation (strict-precondition vs task_success)
  * ROC AUC of strict-precondition as a predictor of task_success
  * task-success rate binned by strict-precondition
Outputs ICRA/tables/table_predictive.tex and ICRA/figures/predictive.pdf.

Tolerant of missing sim data (prints "pending" and exits 0) so it can live in
the pipeline before the sim run completes.
"""
from __future__ import annotations
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
TBL = os.path.join(_ROOT, "ICRA", "tables")
FIG = os.path.join(_ROOT, "ICRA", "figures")


def find_sim_rows():
    rows = []
    for pat in ("results/sim_*/*.csv", "results/*/sim_*.csv", "results/sim_*.csv"):
        for f in glob.glob(os.path.join(_ROOT, pat)):
            rows += list(csv.DictReader(open(f)))
    return rows


def fnum(r, *keys):
    for k in keys:
        v = r.get(k)
        if v not in (None, "", "nan"):
            try:
                return float(v)
            except ValueError:
                pass
    return np.nan


def roc_auc(scores, labels):
    """AUC via the Mann-Whitney U statistic (no scipy)."""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    auc = (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def main():
    os.makedirs(TBL, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    rows = find_sim_rows()
    pairs = []
    for r in rows:
        ps = fnum(r, "precondition_strict", "exec_precondition_strict")
        ts = fnum(r, "exec_task_success", "task_success")
        if not np.isnan(ps) and not np.isnan(ts):
            pairs.append((ps, ts))
    if len(pairs) < 5:
        msg = "% predictive analysis pending — no sim task-success data yet\n"
        open(os.path.join(TBL, "table_predictive.tex"), "w").write(
            "\\begin{tabular}{lc}\\toprule Statistic & Value \\\\ \\midrule "
            "\\multicolumn{2}{c}{\\itshape (pending sim run)} \\\\ \\bottomrule\\end{tabular}\n")
        print("PENDING: need sim task-success data (found %d pairs)." % len(pairs))
        return
    ps = np.array([p for p, _ in pairs]); ts = np.array([t for _, t in pairs])
    tb = (ts >= 0.5).astype(int)  # binarise success
    # point-biserial = Pearson(ps, binary success)
    r_pb = float(np.corrcoef(ps, tb)[0, 1]) if tb.std() > 0 else np.nan
    auc = roc_auc(ps, tb)
    # binned success rate
    edges = [0.0, 0.5, 0.8, 0.95, 1.0001]
    labels = ["<0.5", "0.5–0.8", "0.8–0.95", "0.95–1.0"]
    binrate, binn = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ps >= lo) & (ps < hi)
        binrate.append(float(tb[m].mean()) if m.any() else np.nan)
        binn.append(int(m.sum()))

    # table
    with open(os.path.join(TBL, "table_predictive.tex"), "w") as f:
        f.write("\\begin{tabular}{lc}\n\\toprule\n")
        f.write("Predictor: offline \\texttt{precondition\\_strict} & Value \\\\\n\\midrule\n")
        f.write(f"Point-biserial corr.\\ with task-success & {r_pb:.2f} \\\\\n")
        f.write(f"ROC AUC (predicting task-success) & {auc:.2f} \\\\\n")
        f.write(f"\\# executed plans & {len(pairs)} \\\\\n\\bottomrule\n\\end{tabular}\n")

    # figure: success rate vs strict-precondition bin
    plt.figure(figsize=(3.2, 2.4))
    xs = [l for l, n in zip(labels, binn) if n > 0]
    ys = [b for b, n in zip(binrate, binn) if n > 0]
    plt.bar(range(len(xs)), ys, color="seagreen")
    plt.xticks(range(len(xs)), xs, fontsize=7)
    plt.ylabel("Task-success rate"); plt.xlabel("Offline strict-precondition")
    plt.title(f"Verifier predicts execution (AUC={auc:.2f})", fontsize=9)
    plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "predictive.pdf")); plt.close()

    print(f"point-biserial r = {r_pb:.3f} | ROC AUC = {auc:.3f} | n = {len(pairs)}")
    print("binned success rate:", dict(zip(labels, [round(b, 2) for b in binrate])))


if __name__ == "__main__":
    main()
