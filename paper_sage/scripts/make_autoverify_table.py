#!/usr/bin/env python3
"""make_autoverify_table.py  (Phase 2: auto/learned verifier — transfer table)
=============================================================================
Merge per-domain induced-rule JSONs (iTHOR, ProcTHOR, [ALFWorld]) produced by
induce_verifier.py into the FOCUS deliverable cross-domain table
``ICRA/tables/table_autoverifier_transfer.tex``:

  columns  = the domains (iTHOR | ProcTHOR | ...)
  rows     = per-action induced precondition + per-action accuracy,
             then the overall match (X/16) and held-out pred-acc per domain.

This is the "the verifier is learned, not authored, and ports automatically"
table: the constancy of the induced precondition column across domains is the
whole transfer claim. Pure stdlib; no LLM.

Run (after inducing both domains):
  /home/keti/miniconda3/bin/python scripts/make_autoverify_table.py \
      --ithor   results/autoverify/induced_rules.json \
      --procthor results/autoverify/induced_rules_procthor.json \
      --out ICRA/tables/table_autoverifier_transfer.tex
"""
from __future__ import annotations
import argparse, json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

ACTIONS = ["Pick", "Open", "Close", "TurnOn", "TurnOff", "Place", "PutIn"]
NICE = {"did_find": "found", "holding": "holding", "arrived_target": "arrived",
        "target_open": "open", "pickupable": "pickupable", "openable": "openable",
        "toggleable": "toggleable", "receptacle": "receptacle",
        "in_inventory": "in\\_inv", "at_loc": "at\\_loc", "recep_open": "open",
        "sliceable": "sliceable"}


def cond_str(rule):
    if not rule:
        return "--"
    return " $\\wedge$ ".join(
        (("" if v else "$\\neg$") + NICE.get(k, k)) for k, v in rule.items())


def load(path):
    if not path or not os.path.isfile(path):
        return None
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ithor", default=os.path.join(_ROOT, "results", "autoverify", "induced_rules.json"))
    ap.add_argument("--procthor", default=os.path.join(_ROOT, "results", "autoverify", "induced_rules_procthor.json"))
    ap.add_argument("--alfworld", default="")
    ap.add_argument("--out", default=os.path.join(_ROOT, "ICRA", "tables", "table_autoverifier_transfer.tex"))
    args = ap.parse_args()

    domains = []
    for label, path in [("iTHOR (in-domain)", args.ithor),
                        ("ProcTHOR (unseen)", args.procthor),
                        ("ALFWorld (new vocab)", args.alfworld)]:
        d = load(path)
        if d is not None:
            domains.append((label, d))
    if not domains:
        print("No induced JSONs found; nothing to merge. Run induce_verifier.py first.")
        return

    ncol = len(domains)
    colspec = "l" + "ll" * ncol  # action + (cond, acc) per domain
    lines = [r"% auto: cross-domain transfer of the induced verifier (do not commit results)",
             r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    # header
    hdr = ["Action"]
    for label, _ in domains:
        hdr.append("\\multicolumn{2}{c}{" + label + "}")
    lines.append(" & ".join(hdr) + r" \\")
    sub = [""] + ["Induced precond. & Acc."] * ncol
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")

    for act in ACTIONS:
        cells = [f"\\texttt{{{act}}}"]
        for _, d in domains:
            r = d.get("rules", {}).get(act, {})
            cells.append(cond_str(r.get("rule", {})))
            acc = r.get("accuracy")
            cells.append(f"{acc:.2f}" if isinstance(acc, (int, float)) else "--")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    # match row
    mrow = ["Match vs.\\ ref"]
    for _, d in domains:
        mrow.append("\\multicolumn{2}{c}{" + str(d.get("handwritten_match", "--")) + "}")
    lines.append(" & ".join(mrow) + r" \\")
    # held-out pred-acc row
    prow = ["Held-out pred-acc"]
    for _, d in domains:
        ho = d.get("heldout_predacc", {}).get("mean", "--")
        prow.append("\\multicolumn{2}{c}{" + (f"{ho:.3f}" if isinstance(ho, (int, float)) else str(ho)) + "}")
    lines.append(" & ".join(prow) + r" \\")
    # n transitions row
    nrow = ["\\#transitions"]
    for _, d in domains:
        nrow.append("\\multicolumn{2}{c}{" + str(d.get("n_transitions", "--")) + "}")
    lines.append(" & ".join(nrow) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("wrote", args.out, f"({ncol} domains)")


if __name__ == "__main__":
    main()
