#!/usr/bin/env python3
"""analyze_grid.py
==================
Merge the multi-seed plan-quality grid and compute publication-grade statistics
for the SAGE / ICRA submission.

Inputs : results/grid_<model>_s<seed>/results.csv   (produced by run_grid.sh)
Outputs:
  results/grid_combined.csv            — all rows, one place
  results/grid_analysis.json           — means, 95% bootstrap CIs, permutation
                                          p-values for SAGE-Fixed vs each method
  paper/generated/table_main_ci.tex    — main results table with ±95% CI
  paper/generated/table_significance.tex — SAGE-Fixed vs baselines, per model

Statistics (no scipy dependency):
  * 95% CI of each (model, method, metric) mean via task-level bootstrap
    (resample the 38 tasks with replacement; per-task value = mean over seeds).
  * SAGE-Fixed vs baseline: paired permutation test on per-task differences
    (sign-flip, two-sided) + bootstrap CI of the mean paired difference.
    Pairing is by task (seed-averaged) within each model.

Usage:
  python scripts/analyze_grid.py
  ... --candidate SAGE-Fixed --metrics completeness precondition_strict executability
"""
from __future__ import annotations
import argparse, csv, glob, json, os
from collections import defaultdict
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

NUMERIC = ["step_ratio", "executability", "precondition", "precondition_strict",
           "redundancy", "completeness", "hallucination", "parse_ok",
           "latency_s", "llm_calls", "total_tokens", "refines",
           "verifier_rejections", "num_steps"]


def load_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_src"] = path
                rows.append(r)
    return rows


def to_float(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def per_task_table(rows, model, method, metric):
    """Return {task_id: seed-averaged metric value} for one (model, method)."""
    acc = defaultdict(list)
    for r in rows:
        if r.get("model") == model and r.get("method") == method:
            v = to_float(r.get(metric))
            if not np.isnan(v):
                acc[r.get("task_id")].append(v)
    return {t: float(np.mean(vs)) for t, vs in acc.items() if vs}


def bootstrap_ci(vals, B=10000, alpha=0.05, rng=None):
    vals = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(vals.mean()), float(lo), float(hi))


def paired_perm_test(cand, base, B=20000, rng=None):
    """Two-sided paired permutation (sign-flip) test on per-task differences.

    cand, base: dicts {task_id: value}. Returns (mean_diff, ci_lo, ci_hi, p, n)."""
    keys = sorted(set(cand) & set(base))
    d = np.array([cand[k] - base[k] for k in keys], dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    rng = rng or np.random.default_rng(0)
    obs = d.mean()
    # permutation: random sign flips
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    perm_means = (signs * d).mean(axis=1)
    p = float((np.abs(perm_means) >= abs(obs) - 1e-12).mean())
    # bootstrap CI of the mean paired difference
    idx = rng.integers(0, n, size=(B, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (float(obs), float(lo), float(hi), p, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=os.path.join(_ROOT, "results", "grid_*_s*", "results.csv"))
    ap.add_argument("--candidate", default="SAGE-Fixed")
    ap.add_argument("--metrics", nargs="+",
                    default=["completeness", "precondition_strict", "executability",
                             "precondition", "hallucination", "llm_calls", "total_tokens"])
    ap.add_argument("--out-json", default=os.path.join(_ROOT, "results", "grid_analysis.json"))
    ap.add_argument("--out-combined", default=os.path.join(_ROOT, "results", "grid_combined.csv"))
    ap.add_argument("--tex-dir", default=os.path.join(_ROOT, "paper", "generated"))
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    rows = load_rows(args.pattern)
    if not rows:
        raise SystemExit(f"No rows matched {args.pattern}")
    rng = np.random.default_rng(12345)

    models = sorted({r["model"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    seeds = sorted({r.get("seed", "") for r in rows})
    # coverage map
    cov = defaultdict(int)
    for r in rows:
        cov[(r["model"], r["method"], r.get("seed", ""))] += 1

    print(f"Loaded {len(rows)} rows | models={models} | methods={methods} | seeds={seeds}")

    # write combined csv
    fields = list(rows[0].keys())
    with open(args.out_combined, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    analysis = {"models": models, "methods": methods, "seeds": seeds,
                "n_rows": len(rows), "coverage": {f"{m}|{me}|s{s}": cov[(m, me, s)]
                                                  for (m, me, s) in cov},
                "means_ci": {}, "significance": {}}

    # per (model, method, metric): mean + bootstrap CI over seed-averaged tasks
    for model in models:
        for method in methods:
            for metric in args.metrics:
                tbl = per_task_table(rows, model, method, metric)
                if not tbl:
                    continue
                mean, lo, hi = bootstrap_ci(list(tbl.values()), B=args.boot, rng=rng)
                analysis["means_ci"][f"{model}|{method}|{metric}"] = {
                    "mean": round(mean, 4), "ci_lo": round(lo, 4),
                    "ci_hi": round(hi, 4), "n_tasks": len(tbl)}

    # significance: candidate vs every other method, per model
    for model in models:
        cand_metric_tbls = {m: per_task_table(rows, model, args.candidate, m)
                            for m in args.metrics}
        for method in methods:
            if method == args.candidate:
                continue
            for metric in args.metrics:
                base = per_task_table(rows, model, method, metric)
                cand = cand_metric_tbls[metric]
                if not base or not cand:
                    continue
                md, lo, hi, p, n = paired_perm_test(cand, base, B=max(args.boot, 20000), rng=rng)
                analysis["significance"][f"{model}|{args.candidate}_vs_{method}|{metric}"] = {
                    "mean_diff": round(md, 4), "ci_lo": round(lo, 4),
                    "ci_hi": round(hi, 4), "p_value": round(p, 5), "n_tasks": n,
                    "sig_0.05": bool(p < 0.05)}

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_combined}")

    # ── LaTeX: main table with CI (completeness + p_strict + executability) ──
    os.makedirs(args.tex_dir, exist_ok=True)
    show = [m for m in ["completeness", "precondition_strict", "executability"]
            if any(k.split("|")[2] == m for k in analysis["means_ci"])]
    method_order = [mt for mt in
                    ["Direct", "CoT", "Few-Shot CoT", "Self-Refine", "ReAct",
                     "Hierarchical", "Hierarchical Few-Shot", "SAGE", args.candidate]
                    if mt in methods]
    lines = [r"% auto-generated by analyze_grid.py — main results with 95% bootstrap CI",
             r"\begin{tabular}{ll" + "c" * len(show) + r"}", r"\toprule",
             "Model & Method & " + " & ".join(s.replace("_", r"\_") for s in show) + r" \\",
             r"\midrule"]
    for model in models:
        for method in method_order:
            cells = []
            for metric in show:
                d = analysis["means_ci"].get(f"{model}|{method}|{metric}")
                if d:
                    half = (d["ci_hi"] - d["ci_lo"]) / 2
                    cells.append(f"{d['mean']:.2f}\\,$\\pm$\\,{half:.2f}")
                else:
                    cells.append("--")
            mname = method.replace("_", r"\_")
            if method == args.candidate:
                mname = r"\textbf{" + mname + "}"
            lines.append(f"{model.replace('_', chr(92)+'_')} & {mname} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    with open(os.path.join(args.tex_dir, "table_main_ci.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {os.path.join(args.tex_dir, 'table_main_ci.tex')}")

    # ── console summary of headline significance (completeness) ──
    print("\n=== SAGE-Fixed vs baselines — completeness (paired perm test) ===")
    for model in models:
        for method in method_order:
            if method == args.candidate:
                continue
            k = f"{model}|{args.candidate}_vs_{method}|completeness"
            d = analysis["significance"].get(k)
            if d:
                star = "*" if d["sig_0.05"] else " "
                print(f"  {model:14s} vs {method:22s} Δ={d['mean_diff']:+.3f} "
                      f"[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] p={d['p_value']:.4f} {star}")


if __name__ == "__main__":
    main()
