#!/usr/bin/env python3
"""Emit ICRA/tables/table_completeness.tex with LEAK-FREE (leave-one-out) numbers.

Non-memory baselines come from the original grid (already clean: they ignore seed
memory). The three memory methods (Few-Shot CoT, Hierarchical Few-Shot, SAGE)
come from the loo_* runs (MEM_LEAVE_ONE_OUT=1). SAGE is deduped by
(model,seed,task). Four models with LOO data; 32B is dropped (no LOO re-run).
Reports mean completeness +/- 95% task-bootstrap CI is omitted here (kept simple:
mean only) to avoid recomputing bootstrap; the supplement carries CIs.
"""
import csv, glob, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEM = {"Few-Shot CoT", "Hierarchical Few-Shot", "SAGE"}
MODELS = [("llama3.2", "Llama\\,3B"),
          ("qwen2.5:7b", "Qwen\\,7B"),
          ("mistral-nemo", "Mist.\\,12B"),
          ("qwen2.5:14b", "Qwen\\,14B")]
ROWS = ["Direct", "CoT", "Few-Shot CoT", "Self-Refine", "ReAct",
        "Hierarchical", "Hierarchical Few-Shot", "SAGE"]

def fv(r, k):
    try: return float(r.get(k) or "nan")
    except: return float("nan")
def mean(v):
    v = [x for x in v if x == x]; return sum(v)/len(v) if v else float("nan")
def m2(x): return "SAGE" if x == "SAGE-Fixed" else x
def nm(m): return "qwen2.5:14b" if m == "qwen2.5:14B" else m

# collect: per (method, model) completeness list
data = defaultdict(lambda: defaultdict(list))
# non-memory baselines from grid
for f in glob.glob(os.path.join(ROOT, "results/grid_*/results.csv")):
    if "_quarantine" in f: continue
    for r in csv.DictReader(open(f)):
        me = m2(r.get("method")); mdl = nm(r.get("model"))
        if me in MEM: continue
        data[me][mdl].append(fv(r, "completeness"))
# memory methods from loo
seen = set()
for f in glob.glob(os.path.join(ROOT, "results/loo_*/results.csv")):
    for r in csv.DictReader(open(f)):
        me = m2(r.get("method")); mdl = nm(r.get("model"))
        if me == "SAGE":
            k = (mdl, r.get("seed"), r.get("task_id"))
            if k in seen: continue
            seen.add(k)
        data[me][mdl].append(fv(r, "completeness"))

# column maximum (bold the genuinely-best method per model — honest presentation)
hdr = " & ".join(lab for _, lab in MODELS)
colmax = {}
for key, _ in MODELS:
    colmax[key] = max((mean(data[me].get(key, [])) for me in ROWS
                       if mean(data[me].get(key, [])) == mean(data[me].get(key, []))),
                      default=float("nan"))
lines = []
for me in ROWS:
    disp = "SAGE (ours)" if me == "SAGE" else me
    cells = []
    for key, _ in MODELS:
        v = mean(data[me].get(key, []))
        if v != v:
            cells.append("--")
        elif abs(v - colmax[key]) < 1e-9:
            cells.append(f"\\textbf{{{v:.3f}}}")  # bold = best in column
        else:
            cells.append(f"{v:.3f}")
    if me == "SAGE":
        lines.append("\\midrule")
        lines.append("SAGE (ours) & " + " & ".join(cells) + " \\\\")
    else:
        lines.append(f"{disp} & " + " & ".join(cells) + " \\\\")

tex = (
    "% auto (leave-one-out, leak-free): completeness mean. "
    "Memory methods from loo_* runs; baselines from grid.\n"
    "\\begin{tabular}{lcccc}\n\\toprule\n"
    f"Method & {hdr} \\\\\n\\midrule\n" + "\n".join(lines) +
    "\n\\bottomrule\n\\end{tabular}\n"
)
out = os.path.join(ROOT, "ICRA", "tables", "table_completeness.tex")
open(out, "w").write(tex)
print(f"wrote {out}")
print(tex)
