"""Aggregate the post-Phase-4 results.csv and emit a gate verdict.

Usage:
    python scripts/analyze_long_exp.py \\
        --csv results/20260519_143552_c7c7ec/results.csv \\
        --baseline-method Direct \\
        --candidate-method SAGE-Fixed \\
        --reference-method SAGE \\
        [--out paper/generated/table_sage_ablation.tex]

The gate criterion (per-model):
    GO       : compl(cand) >= compl(baseline)
               AND p_strict(cand) >= max(p_strict(baseline), p_strict(ref)) - 0.05
    PARTIAL  : compl(cand) > compl(ref)            (cand beats SAGE-original)
               but not GO
    NO-GO    : otherwise

Aggregate verdict:
    >= 2/3 GO -> overall GO
    >= 2/3 NO-GO -> overall NO-GO
    else -> PARTIAL
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Iterable


METRICS = [
    "precondition", "precondition_strict", "executability",
    "completeness", "redundancy", "hallucination",
    "total_tokens", "llm_calls", "refines", "num_steps",
]


def _f(x: object) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def aggregate(csv_path: str) -> dict[tuple[str, str], dict[str, float]]:
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_key[(r["model"], r["method"])].append(r)
    agg: dict[tuple[str, str], dict[str, float]] = {}
    for k, rs in by_key.items():
        d: dict[str, float] = {"n": float(len(rs))}
        for m in METRICS:
            vals = [_f(r.get(m)) for r in rs if r.get(m) not in (None, "")]
            d[m] = sum(vals) / len(vals) if vals else 0.0
        # Empties tracked separately — they're the cold-start signal.
        d["empties"] = sum(1 for r in rs if _f(r.get("num_steps")) == 0)
        agg[k] = d
    return agg


def per_model_verdict(
    a: dict[tuple[str, str], dict[str, float]],
    model: str,
    baseline: str,
    candidate: str,
    reference: str,
) -> tuple[str, str]:
    """Return (verdict, one-line explanation)."""
    keys = [(model, baseline), (model, candidate), (model, reference)]
    for k in keys:
        if k not in a:
            return "MISSING", f"no rows for {k}"
    b = a[(model, baseline)]
    c = a[(model, candidate)]
    r = a[(model, reference)]
    best_ps_baseline = max(b["precondition_strict"], r["precondition_strict"])
    gap_ps = c["precondition_strict"] - best_ps_baseline
    if c["completeness"] >= b["completeness"] and gap_ps >= -0.05:
        return ("GO",
                f"compl(cand)={c['completeness']:.3f} >= compl(base)={b['completeness']:.3f}; "
                f"p_strict gap vs best={gap_ps:+.3f} (>= -0.05)")
    if c["completeness"] > r["completeness"]:
        return ("PARTIAL",
                f"compl(cand)={c['completeness']:.3f} > compl(ref-SAGE)={r['completeness']:.3f} "
                f"but compl(base)={b['completeness']:.3f} or p_strict gap={gap_ps:+.3f} blocks GO")
    return ("NO-GO",
            f"compl(cand)={c['completeness']:.3f} ≤ compl(ref-SAGE)={r['completeness']:.3f}")


def aggregate_verdict(per_model: list[tuple[str, str, str]]) -> str:
    counts = {"GO": 0, "PARTIAL": 0, "NO-GO": 0, "MISSING": 0}
    for _, v, _ in per_model:
        counts[v] = counts.get(v, 0) + 1
    total = sum(counts.values())
    if counts["GO"] >= max(2, total - 1):
        return "GO"
    if counts["NO-GO"] >= max(2, total - 1):
        return "NO-GO"
    return "PARTIAL"


def fmt_row(model: str, method: str, d: dict[str, float]) -> str:
    return (f"{model:14s} {method:14s} n={int(d['n']):3d}  "
            f"precond={d['precondition']:.3f}  "
            f"p_strict={d['precondition_strict']:.3f}  "
            f"exec={d['executability']:.3f}  "
            f"compl={d['completeness']:.3f}  "
            f"tokens={d['total_tokens']:6.0f}  "
            f"calls={d['llm_calls']:4.1f}  "
            f"refines={d['refines']:.2f}  "
            f"empties={int(d['empties']):>2d}")


LATEX_HEADER = r"""\begin{tabular}{llrrrrr}
\toprule
Model & Method & precond & p\_strict & compl & tokens & empties \\
\midrule
"""

LATEX_FOOTER = r"""\bottomrule
\end{tabular}
"""


def emit_latex(a: dict[tuple[str, str], dict[str, float]],
               models: list[str],
               methods: list[str],
               out_path: str) -> None:
    lines = [LATEX_HEADER]
    for m in models:
        first = True
        for me in methods:
            d = a.get((m, me))
            if d is None:
                continue
            model_cell = m if first else ""
            first = False
            lines.append(
                f"{model_cell} & {me} & "
                f"{d['precondition']:.2f} & {d['precondition_strict']:.2f} & "
                f"{d['completeness']:.2f} & {d['total_tokens']:.0f} & "
                f"{int(d['empties'])} \\\\\n"
            )
        lines.append("\\midrule\n")
    if lines[-1] == "\\midrule\n":
        lines.pop()
    lines.append(LATEX_FOOTER)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote LaTeX table → {out_path}")


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--baseline-method", default="Direct")
    p.add_argument("--candidate-method", default="SAGE-Fixed")
    p.add_argument("--reference-method", default="SAGE")
    p.add_argument("--models", nargs="*", default=None,
                   help="Restrict to this model set (default: all observed).")
    p.add_argument("--out",
                   default="paper/generated/table_sage_ablation.tex",
                   help="LaTeX table output path")
    args = p.parse_args(list(argv) if argv is not None else None)

    a = aggregate(args.csv)
    if not a:
        print(f"No rows in {args.csv}")
        return 1

    observed_models = sorted({k[0] for k in a})
    models = args.models or observed_models
    methods = [args.baseline_method, args.reference_method, args.candidate_method]

    print(f"\nAggregate rows in {args.csv}:\n")
    for model in models:
        for method in methods:
            d = a.get((model, method))
            if d is None:
                print(f"  {model:14s} {method:14s}  (MISSING)")
                continue
            print("  " + fmt_row(model, method, d))
        print()

    print("\nPer-model gate verdict:\n")
    per_model = []
    for model in models:
        verdict, explanation = per_model_verdict(
            a, model,
            baseline=args.baseline_method,
            candidate=args.candidate_method,
            reference=args.reference_method,
        )
        per_model.append((model, verdict, explanation))
        print(f"  {model:14s}: {verdict:8s} — {explanation}")

    overall = aggregate_verdict(per_model)
    print(f"\nOverall verdict: {overall}")

    emit_latex(a, models, methods, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
