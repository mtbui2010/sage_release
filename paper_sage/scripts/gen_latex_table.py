"""gen_latex_table.py
=======================
Generate paper-ready LaTeX tables from a results.csv produced by
run_benchmark.py. Three table types:

  --table main       Mean per (model, method) — PreCond, PreCondStrict,
                     Exec, StepRatio, Halluc., Tokens, Latency.
  --table ablation   Same as main but restricted to SAGE-* methods.
  --table sim        task_success, step_success_rate, total_reward
                     (reads sim_<model>.csv files in the run dir).

Output is a self-contained LaTeX `tabular` block ready to paste into a
\\begin{table} environment. Use \\input{table_main.tex} from your
LaTeX source for clean composition.

Usage
-----
  python scripts/gen_latex_table.py \\
      --csv results/<run_id>/results.csv \\
      --table main \\
      --out paper/table_main.tex

  # Sim table (different CSV layout):
  python scripts/gen_latex_table.py \\
      --csv results/<run_id>/sim_qwen2.5_7b.csv \\
      --table sim \\
      --out paper/table_sim.tex
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Iterable


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(rows: list[dict], key: str) -> float:
    vals = []
    for r in rows:
        v = r.get(key)
        if v in (None, "", "None"):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            pass
    return sum(vals) / len(vals) if vals else 0.0


def _escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&")


def _emit_main(rows: list[dict]) -> str:
    """Main table: rows grouped by (model, method)."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        groups[(r["model"], r["method"])].append(r)

    # Column metrics + format
    cols = [
        ("PreCond",       "precondition",        "{:.2f}"),
        ("PreCondStrict", "precondition_strict", "{:.2f}"),
        ("Exec",          "executability",       "{:.2f}"),
        ("StepR",         "step_ratio",          "{:.2f}"),
        ("Compl.",        "completeness",        "{:.2f}"),
        ("Halluc.",       "hallucination",       "{:.2f}"),
        ("Tokens",        "total_tokens",        "{:.0f}"),
        ("Lat(s)",        "latency_s",           "{:.1f}"),
    ]

    lines = []
    lines.append(r"\begin{tabular}{ll" + "r" * len(cols) + "}")
    lines.append(r"\toprule")
    header = ["Model", "Method"] + [c[0] for c in cols]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    last_model = None
    for (model, method) in sorted(groups.keys()):
        rs = groups[(model, method)]
        cells = []
        if model != last_model:
            cells.append(_escape(model))
            last_model = model
        else:
            cells.append("")
        cells.append(_escape(method))
        for _, key, fmt in cols:
            cells.append(fmt.format(_mean(rs, key)))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _emit_ablation(rows: list[dict]) -> str:
    abl = [r for r in rows if r.get("method", "").startswith("SAGE")]
    if not abl:
        return "% no SAGE-* rows found — run `make ablate` first."
    return _emit_main(abl)


def _emit_sim(rows: list[dict]) -> str:
    """Sim table: assumes evaluate_sim.py CSV with exec_* columns."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("method", "?")].append(r)

    cols = [
        ("TaskSucc", "exec_task_success", "{:.2%}"),
        ("StepSucc", "exec_step_success", "{:.2f}"),
        ("Reward",   "exec_total_reward", "{:.1f}"),
        ("Replans",  "exec_replans",      "{:.1f}"),
        ("Lat(s)",   "exec_latency_s",    "{:.1f}"),
    ]
    lines = []
    lines.append(r"\begin{tabular}{l" + "r" * len(cols) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(["Method"] + [c[0] for c in cols]) + r" \\")
    lines.append(r"\midrule")
    for method in sorted(groups):
        rs = groups[method]
        cells = [_escape(method)]
        for _, key, fmt in cols:
            cells.append(fmt.format(_mean(rs, key)))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--csv", required=True, help="Input results.csv")
    p.add_argument("--table", choices=["main", "ablation", "sim"], default="main")
    p.add_argument("--out", default="", help="Output .tex (stdout if empty)")
    args = p.parse_args()

    rows = _read_csv(args.csv)
    if not rows:
        print(f"[ERROR] empty CSV: {args.csv}", file=sys.stderr)
        return 1

    if args.table == "main":
        body = _emit_main(rows)
    elif args.table == "ablation":
        body = _emit_ablation(rows)
    else:
        body = _emit_sim(rows)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(body + "\n")
        print(f"Wrote {args.out} ({body.count(chr(10))+1} lines)")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
