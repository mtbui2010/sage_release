#!/usr/bin/env python3
"""gen_supplementary.py
=======================
Emit the detailed LaTeX tables that back the SAGE supplementary / appendix PDF
(``ICRA/supplementary.tex``).  Reads only the result CSVs / JSONs under
``results/`` and writes one ``\\input``-able tabular per file into
``ICRA/supp_tables/``.  No preamble is emitted -- each file is a bare
``\\begin{tabular} ... \\end{tabular}`` so it can be ``\\input`` from any
floating environment.

Run with the miniconda python (numpy available, scipy NOT required)::

    python scripts/gen_supplementary.py

Design / conventions (kept in sync with scripts/gen_tables.py and the deep
analysis in results/analysis/deep_findings.md):

* Model id normalization: ``qwen2.5:14B`` -> ``qwen2.5:14b``.
* Method collapse: ``SAGE-Fixed`` (safe_refine config) -> ``SAGE``; the two
  configs are empirically identical, so de-duplicate by (model, seed, task_id)
  keeping the first.
* Paper-facing relabel: method ``SAGE`` is shown as ``SAGE`` in all headers.
* Every table is best-effort: if a CSV / JSON is missing the table is SKIPPED
  with a printed warning rather than crashing.  This lets the supplementary PDF
  compile against placeholders until all runs land.

The script prints one summary line per table written (or skipped).
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected but stay defensive
    np = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
RESULTS = os.path.join(_ROOT, "results")
OUT = os.path.join(_ROOT, "ICRA", "supp_tables")

BRAND = "SAGE"  # paper-facing name; raw CSVs still use method "SAGE"

# model id -> (display name, param label, sort order)
MODEL_META = {
    "llama3.2":             ("Llama-3.2", "3B", 0),
    "qwen2.5:7b":           ("Qwen2.5", "7B", 1),
    "mistral-nemo":         ("Mistral-Nemo", "12B", 2),
    "qwen2.5:14b":          ("Qwen2.5", "14B", 3),
    "qwen2.5:14B":          ("Qwen2.5", "14B", 3),
    "qwen2.5:32b-instruct": ("Qwen2.5", "32B", 4),
}

# display method order for the offline grid; "SAGE"/"SAGE" comes last.
METHOD_ORDER = ["Direct", "CoT", "Few-Shot CoT", "Self-Refine", "ReAct",
                "Hierarchical", "Hierarchical Few-Shot", "SAGE"]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def norm_model(m: str) -> str:
    if m and m.endswith(":14B"):
        return m[:-1] + "b"
    return m


def model_disp(m: str) -> str:
    meta = MODEL_META.get(m)
    if meta:
        return f"{meta[0]}\\,{meta[1]}"
    return m.replace("_", "\\_") if m else "?"


def model_sort_key(m: str):
    return MODEL_META.get(m, ("zzz", "", 99))[2]


def method_disp(me: str) -> str:
    return BRAND if me == "SAGE" else me


def fval(row, key):
    """float() a CSV cell, returning None for blanks / non-numeric."""
    try:
        v = float(row.get(key, "") or "")
    except (TypeError, ValueError):
        return None
    if v != v:  # nan
        return None
    return v


def mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt(v, nd=3):
    return "--" if v is None else f"{v:.{nd}f}"


def read_csv(path):
    try:
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return None


def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_table(name, body, summary):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        fh.write(body if body.endswith("\n") else body + "\n")
    print(f"[ok]   {name}: {summary}")


def skip(name, why):
    print(f"[skip] {name}: {why}")


# --------------------------------------------------------------------------- #
# data loaders
# --------------------------------------------------------------------------- #
def load_grid():
    """All offline grid rows, normalized + SAGE-collapsed + de-duplicated.

    Mirrors scripts/gen_tables.py:load() so the supplementary numbers match the
    main paper exactly.
    """
    # Leak-free: the three MEMORY methods overlap seed memory with the test set,
    # so we take them from the leave-one-out runs (loo_*) instead of the grid;
    # non-memory baselines are leak-free by construction and come from the grid.
    MEM = {"Few-Shot CoT", "Hierarchical Few-Shot", "SAGE", "SAGE-Fixed"}
    rows = []
    for f in glob.glob(os.path.join(RESULTS, "grid_*_s*", "results.csv")):
        if "_quarantine" in f:
            continue
        r = read_csv(f)
        if r:
            rows += [x for x in r if x.get("method") not in MEM]   # baselines only
    for f in glob.glob(os.path.join(RESULTS, "loo_*", "results.csv")):
        r = read_csv(f)
        if r:
            rows += [x for x in r if x.get("method") in MEM]       # memory methods (LOO)
    out, seen = [], set()
    for r in rows:
        r = dict(r)
        r["model"] = norm_model(r.get("model", ""))
        if r.get("method") in ("SAGE", "SAGE-Fixed"):
            r["method"] = "SAGE"
            key = (r.get("model"), r.get("seed"), r.get("task_id"))
            if key in seen:
                continue
            seen.add(key)
        # drop obviously corrupt rows (model not recognised AND no metrics)
        out.append(r)
    return out


def models_present(rows):
    # only recognised models — drops corrupt CSV rows (e.g. a stray model='1')
    ms = {r.get("model") for r in rows if r.get("model") in MODEL_META}
    return sorted(ms, key=model_sort_key)


def cell_mean(rows, model, method, metric):
    """Per-task mean over seeds, then mean over tasks (so seeds are not
    double-counted)."""
    per_task = defaultdict(list)
    for r in rows:
        if r.get("model") == model and r.get("method") == method:
            v = fval(r, metric)
            if v is not None:
                per_task[r.get("task_id")].append(v)
    task_means = [mean(v) for v in per_task.values()]
    return mean(task_means)


# --------------------------------------------------------------------------- #
# (a) full offline grid: methods x models, multiple metrics
# --------------------------------------------------------------------------- #
GRID_METRICS = [
    ("completeness", "Compl."),
    ("precondition_strict", "P-strict"),
    ("executability", "Exec."),
    ("total_tokens", "Tokens"),
    ("llm_calls", "Calls"),
]


def table_grid_full(rows):
    name = "supp_grid_full.tex"
    if not rows:
        return skip(name, "no grid rows found")
    models = models_present(rows)
    if not models:
        return skip(name, "no models present in grid rows")

    # one column-group per model, each with the 5 metric sub-columns
    ncols_per_model = len(GRID_METRICS)
    col_spec = "l" + ("c" * ncols_per_model) * len(models)

    lines = [
        "% auto-generated by scripts/gen_supplementary.py -- FULL offline grid.",
        "% Rows = 8 planning methods (SAGE = collapsed SAGE-Fixed).",
        "% Per model: completeness, precondition_strict, executability,",
        "% mean total_tokens, mean llm_calls. Do not edit by hand.",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
    ]
    # top header row: model names spanning their metric sub-columns
    top = ["Method"]
    for m in models:
        top.append("\\multicolumn{%d}{c}{%s}" % (ncols_per_model, model_disp(m)))
    lines.append(" & ".join(top) + " \\\\")
    # cmidrule under each model group
    cmids = []
    start = 2
    for _ in models:
        end = start + ncols_per_model - 1
        cmids.append("\\cmidrule(lr){%d-%d}" % (start, end))
        start = end + 1
    lines.append(" ".join(cmids))
    # sub-header row of metric labels
    sub = [""]
    for _ in models:
        sub += [lab for _, lab in GRID_METRICS]
    lines.append(" & ".join(sub) + " \\\\")
    lines.append("\\midrule")

    n_written = 0
    for me in METHOD_ORDER:
        cells = [method_disp(me)]
        any_cell = False
        bold = (me == "SAGE")
        for m in models:
            for metric, _ in GRID_METRICS:
                v = cell_mean(rows, m, me, metric)
                if v is None:
                    cells.append("--")
                    continue
                any_cell = True
                if metric == "total_tokens":
                    s = f"{v:.0f}"
                elif metric == "llm_calls":
                    s = f"{v:.1f}"
                else:
                    s = f"{v:.3f}"
                if bold:
                    s = "\\textbf{" + s + "}"
                cells.append(s)
        if not any_cell:
            continue
        if bold:
            cells[0] = "\\textbf{" + cells[0] + " (ours)}"
            lines.append("\\midrule")
        lines.append(" & ".join(cells) + " \\\\")
        n_written += 1

    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(name, "\n".join(lines),
                f"{n_written} methods x {len(models)} models x {ncols_per_model} metrics")


# --------------------------------------------------------------------------- #
# (b) completeness by difficulty x method (pooled over models)
# --------------------------------------------------------------------------- #
DIFFS = ["easy", "medium", "hard"]


def table_difficulty(rows):
    name = "supp_difficulty.tex"
    if not rows:
        return skip(name, "no grid rows found")

    # pool over models: per (method, difficulty) average per-task means
    def pooled(method, diff):
        per_task = defaultdict(list)
        for r in rows:
            if r.get("method") != method:
                continue
            if (r.get("difficulty") or "").lower() != diff:
                continue
            v = fval(r, "completeness")
            if v is not None:
                per_task[(r.get("model"), r.get("task_id"))].append(v)
        return mean([mean(v) for v in per_task.values()])

    lines = [
        "% auto-generated: completeness by difficulty, pooled over all models.",
        "\\begin{tabular}{l" + "c" * len(DIFFS) + "}",
        "\\toprule",
        "Method & " + " & ".join(d.capitalize() for d in DIFFS) + " \\\\",
        "\\midrule",
    ]
    n_written = 0
    for me in METHOD_ORDER:
        vals = [pooled(me, d) for d in DIFFS]
        if all(v is None for v in vals):
            continue
        bold = (me == "SAGE")
        nm = ("\\textbf{" + method_disp(me) + " (ours)}") if bold else method_disp(me)
        cells = []
        for v in vals:
            s = fmt(v, 3)
            if bold and v is not None:
                s = "\\textbf{" + s + "}"
            cells.append(s)
        if bold:
            lines.append("\\midrule")
        lines.append(nm + " & " + " & ".join(cells) + " \\\\")
        n_written += 1
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(name, "\n".join(lines), f"{n_written} methods x {len(DIFFS)} difficulties")


# --------------------------------------------------------------------------- #
# (c) auto-verifier 3-domain transfer
# --------------------------------------------------------------------------- #
def _rule_to_tex(action, rule):
    """Render an induced precondition dict as a readable conjunction."""
    # human names for the boolean features used across the three domains
    POS = {
        "did_find": "found", "holding": "holding", "pickupable": "pickupable",
        "openable": "openable", "toggleable": "toggleable",
        "receptacle": "receptacle", "arrived_target": "arrived",
        "target_open": "open",
        # alfworld vocabulary
        "at_receptacle": "at-recep", "receptacle_open": "recep-open",
        "in_inventory": "in-inv", "is_openable": "openable",
        "is_receptacle": "receptacle",
    }
    if not rule:
        return "$\\top$"
    parts = []
    for k, v in rule.items():
        name = POS.get(k, k.replace("_", "-"))
        lit = name if int(v) == 1 else "\\neg " + name
        parts.append(lit)
    return "$" + " \\wedge ".join(parts) + "$"


def table_transfer():
    name = "supp_transfer.tex"
    ithor = read_json(os.path.join(RESULTS, "autoverify", "induced_rules.json"))
    proc = read_json(os.path.join(RESULTS, "autoverify", "induced_rules_procthor.json"))
    alf = read_json(os.path.join(RESULTS, "autoverify", "induced_rules_alfworld.json"))
    if not (ithor or proc or alf):
        return skip(name, "no induced_rules*.json found")

    lines = [
        "% auto-generated: cross-domain transfer of the induced verifier.",
        "% Same induction pipeline, zero manual rules. Do not commit results.",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lll ll}",
        "\\toprule",
        "Action & \\multicolumn{2}{c}{iTHOR (in-domain)} "
        "& \\multicolumn{2}{c}{ProcTHOR (unseen)} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        " & Induced precond. & Acc. & Induced precond. & Acc. \\\\",
        "\\midrule",
    ]
    ir = (ithor or {}).get("rules", {})
    pr = (proc or {}).get("rules", {})
    actions = list(ir.keys()) or list(pr.keys())
    for a in actions:
        i = ir.get(a)
        p = pr.get(a)
        i_rule = _rule_to_tex(a, i["rule"]) if i else "--"
        i_acc = f"{i['accuracy']:.2f}" if i else "--"
        p_rule = _rule_to_tex(a, p["rule"]) if p else "--"
        p_acc = f"{p['accuracy']:.2f}" if p else "--"
        lines.append(
            f"\\texttt{{{a}}} & {i_rule} & {i_acc} & {p_rule} & {p_acc} \\\\")
    lines.append("\\midrule")

    def gm(d, k, default="--"):
        if not d:
            return default
        return d.get(k, default)

    i_match = gm(ithor, "handwritten_match")
    p_match = gm(proc, "handwritten_match")
    i_pred = (ithor or {}).get("heldout_predacc", {}).get("mean")
    p_pred = (proc or {}).get("heldout_predacc", {}).get("mean")
    i_n = gm(ithor, "n_transitions")
    p_n = gm(proc, "n_transitions")
    lines.append(
        f"Match vs.\\ ref & \\multicolumn{{2}}{{c}}{{{i_match}}} "
        f"& \\multicolumn{{2}}{{c}}{{{p_match}}} \\\\")
    lines.append(
        f"Held-out pred-acc & \\multicolumn{{2}}{{c}}{{{fmt(i_pred,3)}}} "
        f"& \\multicolumn{{2}}{{c}}{{{fmt(p_pred,3)}}} \\\\")
    lines.append(
        f"\\#transitions & \\multicolumn{{2}}{{c}}{{{i_n}}} "
        f"& \\multicolumn{{2}}{{c}}{{{p_n}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    # Append an ALFWorld OOD block as a second tabular in the same file so a
    # single \input covers all three domains.
    if alf:
        ar = alf.get("rules", {})
        a_pred = alf.get("heldout_predacc", {})
        lines += [
            "",
            "\\vspace{4pt}",
            "% ALFWorld (held-out, different action vocabulary)",
            "\\begin{tabular}{llc}",
            "\\toprule",
            "\\multicolumn{3}{c}{ALFWorld val (OOD, distinct vocabulary)} \\\\",
            "\\midrule",
            "Action & Induced precond. & Held-out pred-acc \\\\",
            "\\midrule",
        ]
        per = a_pred.get("per_action", {})
        for a, info in ar.items():
            rule = _rule_to_tex(a, info.get("rule", {}))
            pa = per.get(a)
            lines.append(f"\\texttt{{{a}}} & {rule} & {fmt(pa,3)} \\\\")
        lines.append("\\midrule")
        lines.append(
            f"Overall & \\multicolumn{{1}}{{c}}{{--}} & {fmt(a_pred.get('mean'),3)} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]

    write_table(name, "\n".join(lines),
                f"iTHOR={'y' if ithor else 'n'} ProcTHOR={'y' if proc else 'n'} "
                f"ALFWorld={'y' if alf else 'n'}")


# --------------------------------------------------------------------------- #
# (d) SOTA baselines (LLM+P, SayCan) vs SAGE
# --------------------------------------------------------------------------- #
def load_sota():
    rows = []
    for f in glob.glob(os.path.join(RESULTS, "sota_*", "results.csv")):
        r = read_csv(f)
        if r:
            for x in r:
                x = dict(x)
                x["model"] = norm_model(x.get("model", ""))
                rows.append(x)
    return rows


def table_sota(grid_rows):
    name = "supp_sota.tex"
    sota = load_sota()
    if not sota:
        return skip(name, "no sota_*/results.csv found")

    sota_models = sorted({r.get("model") for r in sota if r.get("model")},
                         key=model_sort_key)
    metrics = [("completeness", "Compl."), ("precondition_strict", "P-strict")]

    def sota_cell(model, method, metric):
        vals = [fval(r, metric) for r in sota
                if r.get("model") == model and r.get("method") == method]
        return mean([v for v in vals if v is not None])

    lines = [
        "% auto-generated: SOTA baselines (LLM+P, SayCan) vs SAGE.",
        "% SAGE numbers come from the offline grid (method SAGE).",
        "% NOTE: SayCan precondition_strict=1.0 is by construction -- it uses the",
        "% symbolic verifier as its affordance filter.",
        "\\begin{tabular}{ll" + "c" * len(sota_models) + "}",
        "\\toprule",
        "Method & Metric & " + " & ".join(model_disp(m) for m in sota_models) + " \\\\",
        "\\midrule",
    ]
    methods = [("LLM+P", False), ("SayCan", False), ("SAGE", True)]
    for me, is_sage in methods:
        for j, (metric, mlab) in enumerate(metrics):
            first = ("\\multirow{2}{*}{%s}" % method_disp(me)) if j == 0 else ""
            cells = []
            for m in sota_models:
                if is_sage:
                    v = cell_mean(grid_rows, m, "SAGE", metric)
                else:
                    v = sota_cell(m, me, metric)
                s = fmt(v, 3)
                if is_sage and v is not None:
                    s = "\\textbf{" + s + "}"
                cells.append(s)
            if is_sage and j == 0:
                first = "\\multirow{2}{*}{\\textbf{%s}}" % method_disp(me)
            lines.append(first + " & " + mlab + " & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    write_table(name, "\n".join(lines),
                f"3 methods x {len(metrics)} metrics x {len(sota_models)} models")


# --------------------------------------------------------------------------- #
# sim helpers (gc / gate files)
# --------------------------------------------------------------------------- #
SIM_MODELS = ["llama3.2", "qwen2.5:7b", "mistral-nemo",
              "qwen2.5:14b", "qwen2.5:32b-instruct"]


def sim_path(model, suffix):
    # sim CSV filenames replace ':' with '_' (e.g. qwen2.5:7b -> qwen2.5_7b)
    fname = model.replace(":", "_")
    return os.path.join(RESULTS, "sim", f"sim_{fname}{suffix}.csv")


# --------------------------------------------------------------------------- #
# (e) safety gate: gate-OFF (gc) vs gate-ON (gate)
# --------------------------------------------------------------------------- #
def table_gate():
    name = "supp_gate.tex"
    found_any = False
    rows = []  # (model_disp, prevented, repairs, off_step, on_step)
    for m in SIM_MODELS:
        gate = read_csv(sim_path(m, "_gate"))
        gc = read_csv(sim_path(m, "_gc"))
        if not gate:
            continue
        found_any = True
        prevented = sum(int(fval(r, "gate_prevented") or 0) for r in gate)
        repairs = sum(int(fval(r, "gate_repairs") or 0) for r in gate)
        # gate-ON step success: mean exec_step_success over Direct+Hierarchical
        on_vals = [fval(r, "exec_step_success") for r in gate
                   if r.get("method") in ("Direct", "Hierarchical")]
        on_step = mean([v for v in on_vals if v is not None])
        # gate-OFF step success: same methods from the gc file
        off_step = None
        if gc:
            off_vals = [fval(r, "exec_step_success") for r in gc
                        if r.get("method") in ("Direct", "Hierarchical")]
            off_step = mean([v for v in off_vals if v is not None])
        rows.append((model_disp(m), prevented, repairs, off_step, on_step))

    if not found_any:
        return skip(name, "no sim_*_gate.csv found")

    lines = [
        "% auto-generated: runtime safety gate. gate-OFF from sim_<model>_gc.csv,",
        "% gate-ON from sim_<model>_gate.csv (Direct + Hierarchical pooled).",
        "% step_success = mean exec_step_success over those two baselines.",
        "\\begin{tabular}{lrrcc}",
        "\\toprule",
        "Model & Prevented & Repairs & Step (gate-OFF) & Step (gate-ON) \\\\",
        "\\midrule",
    ]
    for disp, prev, rep, off, on in rows:
        lines.append(
            f"{disp} & {prev} & {rep} & {fmt(off,3)} & {fmt(on,3)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(name, "\n".join(lines), f"{len(rows)} models with gate runs")


# --------------------------------------------------------------------------- #
# (f) grounded sim execution: exec_step_success per model x {Direct,Hier,SAGE}
# --------------------------------------------------------------------------- #
def table_sim_exec():
    name = "supp_sim_exec.tex"
    methods = ["Direct", "Hierarchical", "SAGE"]
    per_model = {}  # model -> {method: mean exec_step_success}
    pooled = defaultdict(list)
    for m in SIM_MODELS:
        gc = read_csv(sim_path(m, "_gc"))
        if not gc:
            continue
        row = {}
        for me in methods:
            vals = [fval(r, "exec_step_success") for r in gc
                    if r.get("method") == me]
            mu = mean([v for v in vals if v is not None])
            row[me] = mu
            if mu is not None:
                pooled[me].append(mu)
        if any(v is not None for v in row.values()):
            per_model[m] = row

    if not per_model:
        return skip(name, "no sim_*_gc.csv found")

    lines = [
        "% auto-generated: grounded step-level execution success in AI2-THOR.",
        "% exec_step_success = fraction of plan steps that execute successfully,",
        "% mean over the 38 simulator-verified GT tasks (sim_<model>_gc.csv).",
        "\\begin{tabular}{l" + "c" * len(methods) + "}",
        "\\toprule",
        "Model & " + " & ".join(method_disp(x) for x in methods) + " \\\\",
        "\\midrule",
    ]
    for m in sorted(per_model.keys(), key=model_sort_key):
        row = per_model[m]
        best = max((v for v in row.values() if v is not None), default=None)
        cells = []
        for me in methods:
            v = row.get(me)
            s = fmt(v, 3)
            if v is not None and best is not None and abs(v - best) < 1e-9:
                s = "\\textbf{" + s + "}"
            cells.append(s)
        lines.append(model_disp(m) + " & " + " & ".join(cells) + " \\\\")
    # pooled row
    lines.append("\\midrule")
    pcells = []
    pmeans = {me: mean(pooled[me]) for me in methods}
    pbest = max((v for v in pmeans.values() if v is not None), default=None)
    for me in methods:
        v = pmeans[me]
        s = fmt(v, 3)
        if v is not None and pbest is not None and abs(v - pbest) < 1e-9:
            s = "\\textbf{" + s + "}"
        pcells.append(s)
    lines.append("\\textbf{Pooled} & " + " & ".join(pcells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(name, "\n".join(lines), f"{len(per_model)} models x {len(methods)} methods")


# --------------------------------------------------------------------------- #
# (g) failure taxonomy (from deep_findings.md, else reference existing table)
# --------------------------------------------------------------------------- #
def table_failuretax():
    name = "supp_failuretax.tex"
    md_path = os.path.join(RESULTS, "analysis", "deep_findings.md")
    existing = os.path.join(_ROOT, "ICRA", "tables", "table_failure_taxonomy.tex")

    parsed = _parse_failuretax_md(md_path)
    if parsed:
        rows, total = parsed
        lines = [
            "% auto-generated from results/analysis/deep_findings.md",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Failure category & Direct & Hierarchical & SAGE \\\\",
            "\\midrule",
        ]
        for cat, d, h, s in rows:
            lines.append(f"{cat} & {d} & {h} & {s} \\\\")
        if total:
            lines.append("\\midrule")
            lines.append(f"Total failed (of 375) & {total[0]} & {total[1]} & {total[2]} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write_table(name, "\n".join(lines), f"{len(rows)} categories from deep_findings.md")
        return

    # fall back: re-emit a copy of the committed table if present
    if os.path.exists(existing):
        try:
            with open(existing) as fh:
                body = fh.read()
            write_table(name, "% copied from ICRA/tables/table_failure_taxonomy.tex\n" + body,
                        "copied existing table_failure_taxonomy.tex")
            return
        except OSError:
            pass
    skip(name, "no deep_findings.md taxonomy and no existing table")


def _parse_failuretax_md(md_path):
    """Pull the 'Category x method' markdown table out of deep_findings.md.

    Returns (rows, total) where rows = [(category, direct, hier, sage), ...]
    and total = (direct, hier, sage) or None. Returns None if not parseable.
    """
    try:
        with open(md_path) as fh:
            text = fh.read()
    except OSError:
        return None
    rows = []
    total = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 4:
            continue
        cat = cells[0]
        # skip header / separator rows
        if cat.lower() in ("failure category", "") or set(cat) <= set("-: "):
            continue
        nums = []
        ok = True
        for c in cells[1:]:
            c2 = c.replace("**", "").strip()
            try:
                nums.append(int(c2))
            except ValueError:
                ok = False
                break
        if not ok or len(nums) != 3:
            continue
        clean_cat = cat.replace("**", "").strip()
        if clean_cat.lower().startswith("total"):
            total = tuple(nums)
        else:
            rows.append((clean_cat, nums[0], nums[1], nums[2]))
    if not rows:
        return None
    return rows, total


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"writing supplementary tables to {OUT}")
    grid = load_grid()
    if grid:
        print(f"loaded {len(grid)} grid rows; models = {models_present(grid)}")
    else:
        print("WARNING: no grid rows found -- grid-derived tables will be skipped")

    table_grid_full(grid)          # (a)
    table_difficulty(grid)         # (b)
    table_transfer()               # (c)
    table_sota(grid)               # (d)
    table_gate()                   # (e)
    table_sim_exec()               # (f)
    table_failuretax()             # (g)

    print("done.")


if __name__ == "__main__":
    main()
