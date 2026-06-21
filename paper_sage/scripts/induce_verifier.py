#!/usr/bin/env python3
"""induce_verifier.py  (Phase 2: auto/learned verifier)
======================================================
Induce per-action precondition rules from the collected (pre-state, action,
success) transitions, instead of hand-writing them. Demonstrates that the
verifier's logic can be LEARNED from interaction data and therefore ported to a
new domain automatically.

Method (conjunctive precondition mining, no sklearn):
  For each action and each boolean feature f, the value v in {0,1} is a REQUIRED
  precondition iff every observed SUCCESS has f==v while at least one FAILURE has
  f!=v (i.e. violating it is sufficient to fail). The induced rule is the AND of
  all such (f==v). We then score the rule (predict success = all required met)
  against the observed labels and compare to the hand-written verifier.

Run (after collect_transitions.py):
  /home/keti/miniconda3/bin/python scripts/induce_verifier.py \
      --in results/autoverify/transitions.csv
Outputs: results/autoverify/induced_rules.json + ICRA/tables/table_autoverifier.tex
"""
from __future__ import annotations
import argparse, csv, json, os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
FEATURES = ["did_find", "holding", "arrived_target", "target_open",
            "pickupable", "openable", "toggleable", "receptacle"]
# Hand-written verifier preconditions (for the match comparison).
HANDWRITTEN = {
    "Pick":    {"did_find": 1, "holding": 0, "pickupable": 1},
    "Open":    {"did_find": 1, "openable": 1},
    "Close":   {"did_find": 1, "openable": 1},
    "TurnOn":  {"did_find": 1, "toggleable": 1},
    "TurnOff": {"did_find": 1, "toggleable": 1},
    "Place":   {"holding": 1, "arrived_target": 1},
    "PutIn":   {"holding": 1, "target_open": 1, "receptacle": 1},
}


def induce(rows, features=None):
    feats = features or FEATURES
    rules = {}
    by_act = defaultdict(list)
    for r in rows:
        by_act[r["action"]].append(r)
    for act, rs in by_act.items():
        succ = [r for r in rs if r["success"] == "1"]
        fail = [r for r in rs if r["success"] == "0"]
        if not succ:
            rules[act] = {"rule": {}, "note": "no positive examples", "n": len(rs)}
            continue
        req = {}
        for f in feats:
            svals = {r[f] for r in rs if r["success"] == "1"}
            if len(svals) == 1:                       # constant among successes
                v = svals.pop()
                # is violating it sufficient to fail? (some failure has f != v)
                if any(r[f] != v for r in fail):
                    req[f] = int(v)
        # score: predict success iff all required features match
        def pred(r, req=req):
            return all(int(r[f]) == v for f, v in req.items())
        correct = sum(1 for r in rs if int(pred(r)) == int(r["success"] == "1"))
        rules[act] = {"rule": req, "accuracy": round(correct / len(rs), 3),
                      "n": len(rs), "n_pos": len(succ), "n_neg": len(fail)}
    return rules


def rule_accuracy(rule, rows):
    """Predictive accuracy of a conjunctive rule on (possibly held-out) rows."""
    if not rows:
        return float("nan")
    correct = 0
    for r in rows:
        pred = all(int(r[f]) == v for f, v in rule.items())
        correct += int(pred == (r["success"] == "1"))
    return correct / len(rows)


def holdout_predacc(rows, features, by="scene"):
    """Leave-one-{scene/house}-out predictive accuracy: for each held-out unit,
    induce rules on the rest and score the rule on the held-out unit's rows.
    Reference-free transfer metric (Claim B/C). Returns per-action mean acc and
    an overall mean. Pure stdlib."""
    # ProcTHOR rows carry the house id inside the 'scene' label (procthor_val_N),
    # so 'house' and 'scene' both resolve to the per-house unit column here.
    if rows and by not in rows[0]:
        by = "scene"
    units = sorted({r[by] for r in rows})
    per_act = defaultdict(list)
    if len(units) < 2:
        return {"mean": float("nan"), "per_action": {}, "n_units": len(units)}
    for held in units:
        train = [r for r in rows if r[by] != held]
        test = [r for r in rows if r[by] == held]
        tr_rules = induce(train, features)
        by_act_test = defaultdict(list)
        for r in test:
            by_act_test[r["action"]].append(r)
        for act, trs in by_act_test.items():
            rule = tr_rules.get(act, {}).get("rule", {})
            acc = rule_accuracy(rule, trs)
            if acc == acc:  # not NaN
                per_act[act].append(acc)
    per_action = {a: round(sum(v) / len(v), 3) for a, v in per_act.items() if v}
    allv = [x for v in per_act.values() for x in v]
    return {"mean": round(sum(allv) / len(allv), 3) if allv else float("nan"),
            "per_action": per_action, "n_units": len(units)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(_ROOT, "results", "autoverify", "transitions.csv"))
    ap.add_argument("--out-json", default=os.path.join(_ROOT, "results", "autoverify", "induced_rules.json"))
    ap.add_argument("--tex", default=os.path.join(_ROOT, "ICRA", "tables", "table_autoverifier.tex"))
    ap.add_argument("--features", default="",
                    help="comma-separated feature list (default: the THOR-7 vocab). "
                         "Lets the same learner serve a different domain vocabulary.")
    ap.add_argument("--ref-json", default="",
                    help="JSON file with a reference precondition dict to score against "
                         "instead of the built-in HANDWRITTEN (used by ALFWorld).")
    ap.add_argument("--holdout-by", default="scene", choices=["scene", "house"],
                    help="column for leave-one-out held-out predictive accuracy")
    args = ap.parse_args()
    if not os.path.isfile(args.inp):
        print("PENDING: no transitions yet at", args.inp); return
    rows = list(csv.DictReader(open(args.inp)))

    features = [s.strip() for s in args.features.split(",") if s.strip()] or FEATURES
    reference = HANDWRITTEN
    if args.ref_json and os.path.isfile(args.ref_json):
        reference = json.load(open(args.ref_json))

    rules = induce(rows, features)

    # match vs reference (hand-written, or a scoring-only domain reference)
    matches, total = 0, 0
    for act, hw in reference.items():
        ind = rules.get(act, {}).get("rule", {})
        # a precondition matches if induced agrees on the reference keys
        for k, v in hw.items():
            total += 1
            if ind.get(k) == v:
                matches += 1

    # reference-free transfer metric: leave-one-unit-out predictive accuracy
    held = holdout_predacc(rows, features, by=args.holdout_by)

    json.dump({"rules": rules, "handwritten_match": f"{matches}/{total}",
               "n_transitions": len(rows), "features": features,
               "holdout_by": args.holdout_by, "heldout_predacc": held},
              open(args.out_json, "w"), indent=2)

    print(f"Induced precondition rules from {len(rows)} transitions:")
    for act in reference:
        d = rules.get(act, {})
        print(f"  {act:8s} rule={d.get('rule')} acc={d.get('accuracy')} (n={d.get('n')})")
    print(f"Reference-precondition match: {matches}/{total}")
    print(f"Held-out (leave-one-{args.holdout_by}-out) mean pred-acc: {held['mean']} "
          f"over {held['n_units']} units")

    # tex table
    os.makedirs(os.path.dirname(args.tex), exist_ok=True)
    lines = [r"% auto: induced vs hand-written verifier preconditions",
             r"\begin{tabular}{llc}", r"\toprule",
             r"Action & Induced precondition & Acc. \\", r"\midrule"]
    nice = {"did_find": "found", "holding": "holding", "arrived_target": "arrived",
            "target_open": "open", "pickupable": "pickupable", "openable": "openable",
            "toggleable": "toggleable", "receptacle": "receptacle"}
    for act in reference:
        d = rules.get(act, {})
        rule = d.get("rule", {})
        cond = " $\\wedge$ ".join(
            (("" if v else "$\\neg$") + nice.get(k, k)) for k, v in rule.items()) or "--"
        acc = d.get("accuracy", float("nan"))
        lines.append(f"\\texttt{{{act}}} & {cond} & {acc:.2f} \\\\")
    lines += [r"\midrule",
              f"\\multicolumn{{3}}{{l}}{{\\footnotesize Match vs.\\ hand-written: {matches}/{total} conditions"
              f"; held-out pred-acc {held['mean']}}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    open(args.tex, "w").write("\n".join(lines) + "\n")
    print("wrote", args.tex)


if __name__ == "__main__":
    main()
