#!/usr/bin/env python3
"""Emit ICRA/tables/table_recovery.tex — the failure-recovery COST table.

Reads results/recovery/rec_*.csv (held-out injected-failure runs) and reports,
per method, the cost of recovering from the injected mid-execution failure:
recovery rate, latency, and LLM calls. All methods recover ~100%; SAGE does so
at far fewer LLM calls by repairing only the failed sub-goal's suffix.
Counts only rows where the injection actually fired (a replan was triggered).
"""
import csv, glob, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def fv(r, k):
    try: return float(r.get(k) or "nan")
    except: return float("nan")
def mean(v):
    v = [x for x in v if x == x]; return sum(v)/len(v) if v else float("nan")

rows = []
for f in glob.glob(os.path.join(ROOT, "results/recovery/rec_*.csv")):
    rows += list(csv.DictReader(open(f)))

agg = defaultdict(lambda: defaultdict(list)); seen = set()
for r in rows:
    me = "SAGE" if r["method"] == "SAGE-Fixed" else r["method"]
    if fv(r, "replan_llm_calls") > 0 or fv(r, "replan_steps") > 0:
        k = (me, r.get("task_id"))
        if k in seen: continue
        seen.add(k)
        agg[me]["ok"].append(fv(r, "replan_ok"))
        agg[me]["lat"].append(fv(r, "replan_latency_s"))
        agg[me]["calls"].append(fv(r, "replan_llm_calls"))

# display name -> internal key
ORDER = [("SAGE (ours)", "SAGE"),
         ("Hierarchical Few-Shot", "HierarchicalFewShot"),
         ("Hierarchical", "Hierarchical"),
         ("Self-Refine", "Self-Refine"),
         ("ReAct", "ReAct")]
gc = mean(agg["SAGE"]["calls"])
lines = []
for disp, key in ORDER:
    a = agg.get(key)
    if not a or not a["ok"]:
        continue
    ok, lat, calls = mean(a["ok"]), mean(a["lat"]), mean(a["calls"])
    ratio = calls / gc if gc else float("nan")
    name = f"\\textbf{{{disp}}}" if key == "SAGE" else disp
    cells = (f"{ok:.2f} & {lat:.2f} & {calls:.2f} & {ratio:.1f}$\\times$")
    if key == "SAGE":
        cells = (f"\\textbf{{{ok:.2f}}} & \\textbf{{{lat:.2f}}} & "
                 f"\\textbf{{{calls:.2f}}} & \\textbf{{1.0$\\times$}}")
    lines.append(f"{name} & {len(a['ok'])} & {cells} \\\\")

tex = (
    "% auto: failure-recovery COST on injected-failure tasks (held-out)\n"
    "\\begin{tabular}{lccccc}\n\\toprule\n"
    "Method & $n$ & Recovery & Latency (s) & LLM calls & vs SAGE \\\\\n"
    "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
)
out = os.path.join(ROOT, "ICRA", "tables", "table_recovery.tex")
open(out, "w").write(tex)
print(f"wrote {out} ({len(lines)} methods)")
print(tex)
