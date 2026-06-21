#!/usr/bin/env python3
"""Reconcile human ratings against the partial-credit auto-checker.

Run AFTER raters export their CSVs from judge.html. Each CSV has columns:
    item_id, task_id, method, rater, judgment, note      (judgment in {0,0.5,1})

Usage:
    python scripts/reconcile_auto_human.py rater1.csv rater2.csv [...]
    # defaults to results/human_eval/sage_human_eval_*.csv if no args

Reports (stdlib only — no SciPy, matching the paper's stats convention):
  * per-method means: human (rater-averaged), auto_partial, auto_binary
  * inter-rater agreement: exact %, Cohen's kappa, linear-weighted kappa
  * auto-vs-human: Pearson r, MAE, binarized Cohen's kappa
  * largest auto-vs-human disagreements (for manual inspection)
Writes results/human_eval/reconcile.csv and reconcile_summary.json.
"""
import csv, glob, json, math, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BUNDLE = os.path.join(ROOT, "results", "human_eval")
METHODS = ("Direct", "SAGE", "Hierarchical")


def load_auto():
    p = os.path.join(BUNDLE, "auto_partial.csv")
    auto = {}
    for r in csv.DictReader(open(p)):
        auto[r["id"]] = {"partial": float(r["auto_partial"]),
                         "binary": float(r["auto_binary"] or 0),
                         "method": r["method"], "task_id": r["task_id"]}
    return auto


_FIELDS = ["item_id", "task_id", "method", "rater", "judgment", "note"]


def load_human(paths):
    # rater -> {item_id: judgment}.  Tolerates CSVs exported without a header row.
    raters = collections.defaultdict(dict)
    for p in paths:
        with open(p) as f:
            first = f.readline()
            f.seek(0)
            has_header = "item_id" in first
            reader = csv.DictReader(f) if has_header else csv.DictReader(f, fieldnames=_FIELDS)
            for r in reader:
                j = r.get("judgment", "")
                if j == "" or j is None:
                    continue
                rid = r.get("rater") or os.path.basename(p)
                try:
                    raters[rid][r["item_id"]] = float(j)
                except (ValueError, TypeError):
                    pass
    return raters


def cohen_kappa(a, b, weighted=False):
    """Cohen's kappa for paired ordinal labels a,b over {0,0.5,1}."""
    cats = [0.0, 0.5, 1.0]
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    if n == 0:
        return float("nan")
    # confusion + marginals
    conf = [[0] * 3 for _ in range(3)]
    for x, y in zip(a, b):
        conf[idx[x]][idx[y]] += 1
    row = [sum(conf[i]) for i in range(3)]
    col = [sum(conf[i][j] for i in range(3)) for j in range(3)]

    def w(i, j):
        if not weighted:
            return 1.0 if i == j else 0.0
        return 1.0 - abs(i - j) / 2.0  # linear weight over 3 ordinal levels

    po = sum(w(i, j) * conf[i][j] for i in range(3) for j in range(3)) / n
    pe = sum(w(i, j) * row[i] * col[j] for i in range(3) for j in range(3)) / (n * n)
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def snap(v):
    return min([0.0, 0.5, 1.0], key=lambda c: abs(c - v))


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(BUNDLE, "sage_human_eval_*.csv")))
    if not paths:
        print("No human CSVs found. Export them from judge.html first, e.g.\n"
              f"  {BUNDLE}/sage_human_eval_rater1.csv")
        return
    print("Human CSVs:", [os.path.basename(p) for p in paths])
    auto = load_auto()
    raters = load_human(paths)
    rids = list(raters)
    print("Raters:", rids)

    # items judged by ALL raters
    common = set(auto)
    for rid in rids:
        common &= set(raters[rid])
    common = sorted(common)
    print(f"Items judged by all raters AND scored by auto: {len(common)}/{len(auto)}\n")
    if not common:
        print("No overlap yet — raters still in progress.")
        return

    # human consensus per item = mean over raters
    human = {i: sum(raters[r][i] for r in rids) / len(rids) for i in common}

    # ---- per-method means ----
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # n, human, partial, binary
    for i in common:
        a = agg[auto[i]["method"]]
        a[0] += 1; a[1] += human[i]; a[2] += auto[i]["partial"]; a[3] += auto[i]["binary"]
    print(f"{'method':13s} {'n':>3s} {'human':>7s} {'auto_partial':>12s} {'auto_binary':>11s}")
    rows_summary = {}
    for m in METHODS:
        if m not in agg: continue
        n, h, p, b = agg[m]
        print(f"{m:13s} {n:>3d} {h/n:>7.3f} {p/n:>12.3f} {b/n:>11.3f}")
        rows_summary[m] = {"n": n, "human": h/n, "auto_partial": p/n, "auto_binary": b/n}

    # ---- inter-rater agreement (if >=2 raters) ----
    print()
    if len(rids) >= 2:
        a = [raters[rids[0]][i] for i in common]
        b = [raters[rids[1]][i] for i in common]
        exact = sum(1 for x, y in zip(a, b) if x == y) / len(a)
        print(f"Inter-rater ({rids[0]} vs {rids[1]}): exact={exact:.3f} "
              f"kappa={cohen_kappa(a,b):.3f} weighted-kappa={cohen_kappa(a,b,True):.3f}")
    else:
        print("Inter-rater agreement: need >=2 raters (only one so far).")

    # ---- auto vs human ----
    hv = [human[i] for i in common]
    pv = [auto[i]["partial"] for i in common]
    bv = [auto[i]["binary"] for i in common]
    mae_p = sum(abs(h - p) for h, p in zip(hv, pv)) / len(hv)
    mae_b = sum(abs(h - p) for h, p in zip(hv, bv)) / len(hv)
    # binarized kappa: success = score>=0.5
    hb = [1.0 if x >= 0.5 else 0.0 for x in hv]
    pb = [1.0 if x >= 0.5 else 0.0 for x in pv]
    print(f"\nAuto_partial vs human:  Pearson r={pearson(pv,hv):.3f}  MAE={mae_p:.3f}  "
          f"binarized-kappa={cohen_kappa([snap(x) for x in pb],[snap(x) for x in hb]):.3f}")
    print(f"Auto_binary  vs human:  Pearson r={pearson(bv,hv):.3f}  MAE={mae_b:.3f}")
    print("(lower MAE / higher r for partial vs binary ⇒ partial credit tracks humans better)")

    # ---- largest disagreements ----
    disag = sorted(common, key=lambda i: -abs(human[i] - auto[i]["partial"]))[:12]
    print("\nLargest auto_partial–human gaps:")
    for i in disag:
        print(f"  {i:22s} human={human[i]:.2f} auto={auto[i]['partial']:.2f}")

    # ---- write outputs ----
    with open(os.path.join(BUNDLE, "reconcile.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "method", "task_id", "human_mean", "auto_partial",
                    "auto_binary"] + [f"rater_{r}" for r in rids])
        for i in common:
            w.writerow([i, auto[i]["method"], auto[i]["task_id"], f"{human[i]:.3f}",
                        auto[i]["partial"], auto[i]["binary"]]
                       + [raters[r].get(i, "") for r in rids])
    json.dump({"per_method": rows_summary, "n_common": len(common),
               "raters": rids,
               "auto_partial_vs_human": {"pearson": pearson(pv, hv), "mae": mae_p},
               "auto_binary_vs_human": {"pearson": pearson(bv, hv), "mae": mae_b}},
              open(os.path.join(BUNDLE, "reconcile_summary.json"), "w"), indent=2)
    print(f"\nwrote {BUNDLE}/reconcile.csv + reconcile_summary.json")


if __name__ == "__main__":
    main()
