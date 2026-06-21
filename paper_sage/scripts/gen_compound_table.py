#!/usr/bin/env python3
"""Emit ICRA/tables/table_compound.tex from the compound stress-test results.

Reads results/compound_<model>/results.csv (one per model) and reports, per model,
goal completeness for SAGE (SAGE) vs the strongest baseline (Hierarchical
Few-Shot) and the one-shot Direct baseline, plus SAGE's mean plan length and
precondition-validity. Regenerate after the mistral run lands.
"""
import csv, glob, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [
    ("results/compound_full",          r"\texttt{qwen2.5:7b}"),
    ("results/compound_qwen2.5_14b",   r"\texttt{qwen2.5:14b}"),
    ("results/compound_llama3.2",      r"\texttt{llama3.2}"),
    ("results/compound_mistral-nemo",  r"\texttt{mistral-nemo}"),
]

def fv(r, k):
    try: return float(r.get(k) or "nan")
    except: return float("nan")
def mean(v):
    v = [x for x in v if x == x]; return sum(v)/len(v) if v else float("nan")
def m2(x): return "SAGE" if x == "SAGE-Fixed" else x

rows_out = []
for run, label in RUNS:
    path = os.path.join(ROOT, run, "results.csv")
    if not os.path.exists(path):
        continue
    agg = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        me = m2(r["method"])
        agg[me]["c"].append(fv(r, "completeness"))
        agg[me]["s"].append(fv(r, "num_steps"))
        agg[me]["p"].append(fv(r, "precondition_strict"))
    if "SAGE" not in agg:
        continue
    g  = mean(agg["SAGE"]["c"])
    hf = mean(agg["Hierarchical Few-Shot"]["c"])
    d  = mean(agg["Direct"]["c"])
    delta = g - hf
    rows_out.append(
        f"{label} & {d:.3f} & {hf:.3f} & \\textbf{{{g:.3f}}} & "
        f"\\textbf{{{delta:+.3f}}} & {mean(agg['SAGE']['s']):.1f} & "
        f"{mean(agg['SAGE']['p']):.3f} \\\\"
    )

body = "\n".join(rows_out)
tex = (
    "\\begin{tabular}{lcccccc}\n\\toprule\n"
    "Model & Direct & Hier-FS & \\textbf{SAGE} & $\\Delta$ & "
    "SAGE steps & SAGE p-valid \\\\\n\\midrule\n"
    f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
)
out = os.path.join(ROOT, "ICRA", "tables", "table_compound.tex")
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w").write(tex)
print(f"wrote {out} ({len(rows_out)} model rows)")
print(tex)
