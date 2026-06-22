#!/usr/bin/env python3
"""gen_tables.py
================
Generate the LaTeX result tables for the ICRA paper from the grid CSVs
(and, when present, the sim CSVs). Run with the miniconda python (numpy).

  python scripts/gen_tables.py

Outputs into paper/generated/ (also copied to ICRA/tables/ by the Makefile/
assembly step):
  table_completeness.tex   completeness, 8 methods x N models, mean +/- 95% CI
  table_pstrict.tex        precondition_strict, same shape
  table_scaling.tex        Direct vs best-baseline vs SAGE across model sizes
  table_cost.tex           llm_calls + tokens per method (pooled over models)

Design decisions baked in:
  * SAGE is COMBINED into one column: the safe_refine=True config
    (labelled SAGE-Fixed in the raw CSVs) is shown as "SAGE"; the unguarded
    duplicate is dropped (the two are empirically identical).
  * 95% CIs come from a task-level bootstrap (per-task value = mean over seeds).
"""
from __future__ import annotations
import csv, glob, os
from collections import defaultdict
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
OUT = os.path.join(_ROOT, "ICRA", "tables")  # ICRA build is the canonical target
BRAND = "SAGE"  # paper-facing name; raw CSVs still use method "SAGE"

# model id -> (display name, param label, sort order)
MODEL_META = {
    "llama3.2":      ("Llama-3.2", "3B", 0),
    "qwen2.5:7b":    ("Qwen2.5", "7B", 1),
    "mistral-nemo":  ("Mistral-Nemo", "12B", 2),
    "qwen2.5:14b":   ("Qwen2.5", "14B", 3),
    "qwen2.5:14B":   ("Qwen2.5", "14B", 3),
    "qwen2.5:32b-instruct": ("Qwen2.5", "32B", 4),
}
# display method order; "SAGE" is synthesised from SAGE-Fixed
METHOD_ORDER = ["Direct", "CoT", "Few-Shot CoT", "Self-Refine", "ReAct",
                "Hierarchical", "Hierarchical Few-Shot", "SAGE"]


def load():
    rows = []
    for f in glob.glob(os.path.join(_ROOT, "results", "grid_*_s*", "results.csv")):
        rows += list(csv.DictReader(open(f)))
    # Collapse to one SAGE: both SAGE (unguarded) and SAGE-Fixed (safe_refine)
    # map to method "SAGE"; they are empirically identical, so we de-duplicate
    # by (model, seed, task_id) keeping the first. This works whether one or
    # both configs were run for a given model.
    out, seen = [], set()
    for r in rows:
        # unify the 14B model id across hosts (local tag :14B vs remote :14b)
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


def fval(r, m):
    try:
        v = float(r.get(m) or "nan");  return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def per_task(rows, model, method, metric):
    acc = defaultdict(list)
    for r in rows:
        if r.get("model") == model and r.get("method") == method:
            v = fval(r, metric)
            if not np.isnan(v):
                acc[r.get("task_id")].append(v)
    return {t: float(np.mean(v)) for t, v in acc.items() if v}


def ci(vals, B=10000, rng=None):
    a = np.array([v for v in vals if not np.isnan(v)], float)
    if len(a) == 0:
        return (np.nan, np.nan)
    rng = rng or np.random.default_rng(0)
    m = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    return float(a.mean()), float((np.percentile(m, 97.5) - np.percentile(m, 2.5)) / 2)


def models_present(rows):
    # Only known models — guards against corrupted/garbage CSV rows (e.g. a
    # race-interleaved row with model='1') that would otherwise KeyError on
    # MODEL_META[m] downstream. Mirrors gen_figures.models_present.
    ms = {r["model"] for r in rows if r.get("model") in MODEL_META}
    return sorted(ms, key=lambda m: MODEL_META[m][2])


def metric_table(rows, metric, caption_metric):
    rng = np.random.default_rng(12345)
    models = models_present(rows)
    best = {}  # (model) -> best non-SAGE mean, to underline / compare
    cell = defaultdict(dict)
    for mdl in models:
        means = {}
        for me in METHOD_ORDER:
            tbl = per_task(rows, mdl, me, metric)
            if tbl:
                mean, half = ci(list(tbl.values()), rng=rng)
                cell[me][mdl] = (mean, half, len(tbl))
                means[me] = mean
        nong = {k: v for k, v in means.items() if k != "SAGE"}
        best[mdl] = max(nong.values()) if nong else np.nan
    # build LaTeX
    head = "Method & " + " & ".join(
        f"{MODEL_META[m][0]}\\,{MODEL_META[m][1]}" for m in models) + r" \\"
    lines = [f"% auto: {caption_metric}, mean $\\pm$ 95% CI (task bootstrap). SAGE = safe_refine config.",
             r"\begin{tabular}{l" + "c" * len(models) + "}", r"\toprule", head, r"\midrule"]
    for me in METHOD_ORDER:
        cells = []
        for mdl in models:
            if mdl in cell[me]:
                mean, half, n = cell[me][mdl]
                s = f"{mean:.2f}\\,{{\\scriptsize$\\pm${half:.2f}}}"
                if me == "SAGE":
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            else:
                cells.append("--")
        nm = me if me != "SAGE" else r"\textbf{SAGE (ours)}"
        lines.append(f"{nm} & " + " & ".join(cells) + r" \\")
        if me == "Hierarchical Few-Shot":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def scaling_table(rows):
    """SAGE vs Direct vs best-baseline across model sizes (completeness)."""
    rng = np.random.default_rng(7)
    models = models_present(rows)
    def m(model, method, metric):
        t = per_task(rows, model, method, metric)
        return ci(list(t.values()), rng=rng)[0] if t else np.nan
    lines = [r"% auto: scaling — completeness / p_strict across model size",
             r"\begin{tabular}{ll" + "c" * len(models) + "}", r"\toprule",
             "Metric & Method & " + " & ".join(f"{MODEL_META[x][1]}" for x in models) + r" \\",
             r"\midrule"]
    for metric, lab in [("completeness", "Completeness"), ("precondition_strict", "P-strict")]:
        for i, me in enumerate(["Direct", "Hierarchical Few-Shot", "SAGE"]):
            first = f"\\multirow{{3}}{{*}}{{{lab}}}" if i == 0 else ""
            vals = " & ".join(f"{m(x, me, metric):.2f}" if m(x, me, metric)==m(x, me, metric) else "--" for x in models)
            nm = me if me != "SAGE" else r"\textbf{SAGE}"
            lines.append(f"{first} & {nm} & {vals} " + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def cost_table(rows):
    rng = np.random.default_rng(3)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("llm_calls", "total_tokens"):
            v = fval(r, k)
            if not np.isnan(v):
                agg[r["method"]][k].append(v)
    lines = [r"% auto: cost per method, pooled over models",
             r"\begin{tabular}{lcc}", r"\toprule",
             r"Method & LLM calls & Tokens/plan \\", r"\midrule"]
    for me in METHOD_ORDER:
        a = agg.get(me)
        if not a:
            continue
        c = np.mean(a["llm_calls"]) if a["llm_calls"] else float("nan")
        t = np.mean(a["total_tokens"]) if a["total_tokens"] else float("nan")
        nm = me if me != "SAGE" else r"\textbf{SAGE}"
        lines.append(f"{nm} & {c:.1f} & {t:.0f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    if not rows:
        raise SystemExit("no grid rows found")
    outs = {
        "table_completeness.tex": metric_table(rows, "completeness", "completeness"),
        "table_pstrict.tex":      metric_table(rows, "precondition_strict", "precondition\\_strict"),
        "table_scaling.tex":      scaling_table(rows),
        "table_cost.tex":         cost_table(rows),
    }
    for name, txt in outs.items():
        with open(os.path.join(OUT, name), "w") as f:
            f.write(txt)
        print(f"wrote {os.path.join(OUT, name)}")
    print(f"models present: {models_present(rows)}")


if __name__ == "__main__":
    main()
