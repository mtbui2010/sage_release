#!/usr/bin/env python3
"""Partial-credit auto scorer for the human-eval bundle.

Re-scores each captured item in results/human_eval/items.jsonl with *fractional*
goal credit (0 / 0.5 / 1, finer for 3-goal tasks) instead of the binary
auto_task_success, by decomposing each task into atomic sub-goals and counting
how many the captured final state satisfies.

Faithful to the human-judged plans: it scores the SAME captured final_states +
executed plan the raters see — it does NOT re-run the simulator. Reuses
goal_checker.py's helper predicates so the sub-goals match the capture-time logic.

Multi-step tasks (coffee, boil, cook, microwave, dishes, watch-tv, read, sleep,
get-fridge, get-clothes, compounds) get true partial credit. Inherently single-
condition tasks (turn on stove, light off, sit, grab soap) keep the binary verdict.

Output: results/human_eval/auto_partial.csv and a per-method summary.
Run:  python scripts/auto_partial_score.py
"""
import csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PP   = os.path.abspath(os.path.join(ROOT, "..", "pyplanner"))
sys.path.insert(0, PP)
sys.path.insert(0, os.path.join(PP, "apps"))
sys.path.insert(0, os.path.join(PP, "apps", "evaluate"))

from goal_checker import (  # noqa: E402  (reuse capture-time predicates)
    _is_on, _is_open, _was_grabbed, _was_turned_on, _was_placed_on,
)
import goal_checker as gc  # noqa: E402

BUNDLE = os.path.join(ROOT, "results", "human_eval")


def has_action(steps, *acts):
    acts = tuple(a.lower() for a in acts)
    return any(s.get("action", "").lower() in acts for s in steps)


def placed(item, recept, om, steps):
    return _was_placed_on(item, recept, steps, om)


# Atomic sub-goals per GoalFn name. Each entry: (label, predicate(om, steps)->bool).
# Mirrors the conditions the capture-time GoalFns evaluate internally.
SUBGOALS = {
    "_goal_make_coffee": [
        ("mug on machine", lambda om, s: placed("Mug", "CoffeeMachine", om, s)),
        ("machine on",     lambda om, s: _is_on("CoffeeMachine", om) or _was_turned_on("CoffeeMachine", s)),
    ],
    "_goal_boil_water": [
        ("pot on stove", lambda om, s: placed("Pot", "Stove", om, s) or placed("Pot", "StoveBurner", om, s)),
        ("burner on",    lambda om, s: _was_turned_on("Stove", s) or _is_on("StoveBurner", om) or _is_on("StoveKnob", om)),
    ],
    "_goal_cook_egg": [
        ("egg in pan", lambda om, s: placed("Egg", "Pan", om, s) or placed("Egg", "Pot", om, s)),
        ("stove on",   lambda om, s: _was_turned_on("Stove", s) or _is_on("StoveBurner", om)),
    ],
    "_goal_microwave": [
        ("food in microwave", lambda om, s: placed("", "Microwave", om, s)),
        ("microwave on",      lambda om, s: _is_on("Microwave", om) or _was_turned_on("Microwave", s)),
    ],
    "_goal_setup_dishes": [
        ("plate on table", lambda om, s: placed("Plate", "DiningTable", om, s) or placed("Plate", "Table", om, s)),
        ("cup on table",   lambda om, s: placed("Cup", "DiningTable", om, s) or placed("Cup", "Table", om, s)),
    ],
    "_goal_get_fridge": [
        # capture-lenient: "accessed" = fridge referenced in a step or left open
        ("fridge accessed", lambda om, s: _is_open("Fridge", om) or any("fridge" in x.get("object", "").lower() for x in s)),
        ("item grabbed",    lambda om, s: has_action(s, "Pick", "Grab", "PickupObject")),
    ],
    "_goal_get_clothes": [
        ("dresser opened", lambda om, s: _is_open("Dresser", om) or _is_open("Drawer", om) or (has_action(s, "Open") and any(o in x.get("object", "").lower() for x in s for o in ("dresser", "drawer")))),
        ("clothes grabbed", lambda om, s: has_action(s, "Pick", "Grab", "PickupObject")),
    ],
    "_goal_watch_tv": [
        ("tv on",         lambda om, s: _is_on("Television", om) or _is_on("TV", om) or _was_turned_on("Television", s) or _was_turned_on("TV", s)),
        ("agent sitting", lambda om, s: has_action(s, "Sit", "SitOn")),
    ],
    "_goal_read_book": [
        ("book grabbed",  lambda om, s: _was_grabbed("Book", s)),
        ("agent sitting", lambda om, s: has_action(s, "Sit", "SitOn")),
    ],
    "_goal_sleep": [  # used by B04 "turn off light AND lie" (B02 overridden to lie-only)
        ("light off",    lambda om, s: has_action(s, "TurnOff") and not _is_on("LightSwitch", om)),
        ("agent lying",  lambda om, s: has_action(s, "LieOn", "LieDown", "Lie")),
    ],
}

# Inline compound tasks (registry lambdas) keyed by task_id.
SUBGOALS_BY_TASK = {
    "B02": [  # "Lie down on the bed" — lie only (capture _goal_sleep is lying-only)
        ("agent lying", lambda om, s: has_action(s, "LieOn", "LieDown", "Lie")),
    ],
    "B07": [  # task = set alarm + turn OFF light + lie down (3 true sub-goals;
              # capture binary ignores the light, so partial is legitimately stricter)
        ("alarm grabbed", lambda om, s: _was_grabbed("Alarm", s) or _was_grabbed("AlarmClock", s)),
        ("light off",     lambda om, s: has_action(s, "TurnOff") and not _is_on("LightSwitch", om)),
        ("agent lying",   lambda om, s: has_action(s, "LieOn", "LieDown", "Lie")),
    ],
    "B08": [  # capture B08 = pillow placed on bed
        ("pillow on bed", lambda om, s: placed("Pillow", "Bed", om, s) or any(x.get("action") in ("Place", "PutIn") and "bed" in x.get("target", "").lower() for x in s)),
    ],
    "L06": [  # turn on lamp + sit
        ("lamp on",       lambda om, s: _is_on("FloorLamp", om) or _is_on("DeskLamp", om) or _was_turned_on("Lamp", s)),
        ("agent sitting", lambda om, s: has_action(s, "Sit", "SitOn")),
    ],
    "L07": [  # clean living room: place >=2 clutter items
        ("first item placed",  lambda om, s: len(gc._placements_from_steps(s)) >= 1),
        ("second item placed", lambda om, s: len(gc._placements_from_steps(s)) >= 2),
    ],
    "L08": [  # movie night: TV on, lamp dimmed/off, sit
        ("tv on",         lambda om, s: _is_on("Television", om) or _was_turned_on("Television", s) or _was_turned_on("TV", s)),
        ("lamp off",      lambda om, s: has_action(s, "TurnOff")),
        ("agent sitting", lambda om, s: has_action(s, "Sit", "SitOn")),
    ],
}


def parse_plan(plan):
    """['TurnOn CoffeeMachine', ...] -> [{'action','object','target'}, ...]."""
    out = []
    for step in plan or []:
        if isinstance(step, dict):
            out.append(step); continue
        parts = str(step).split()
        if not parts:
            continue
        out.append({"action": parts[0], "object": parts[1] if len(parts) > 1 else "",
                    "target": parts[2] if len(parts) > 2 else ""})
    return out


def score_item(it):
    """Return (partial_score, n_met, n_total, detail, mode)."""
    tid = it["task_id"]
    om = it.get("final_states") or {}
    steps = parse_plan(it.get("plan"))
    binary = float(it.get("auto_task_success") or 0.0)

    subs = SUBGOALS_BY_TASK.get(tid)
    if subs is None:
        fn = gc.GOAL_CONDITIONS.get(tid)
        name = getattr(fn, "__name__", "") if fn else ""
        subs = SUBGOALS.get(name)

    if not subs:
        # inherently single-condition / unlisted -> keep binary verdict
        return binary, int(round(binary)), 1, ["(single-condition: binary)"], "binary"

    met = 0; detail = []
    for label, pred in subs:
        try:
            ok = bool(pred(om, steps))
        except Exception as e:
            ok = False; label += f" [err:{e}]"
        met += ok
        detail.append(f"{label}={'Y' if ok else 'N'}")
    n = len(subs)
    return met / n, met, n, detail, "partial"


def main():
    items = [json.loads(l) for l in open(os.path.join(BUNDLE, "items.jsonl"))]
    rows = []
    for it in items:
        sc, met, n, detail, mode = score_item(it)
        rows.append({
            "id": it["id"], "method": it["method"], "task_id": it["task_id"],
            "auto_binary": it.get("auto_task_success"),
            "auto_partial": round(sc, 3), "n_met": met, "n_total": n,
            "mode": mode, "subgoals": "; ".join(detail),
            "goal_reason": it.get("goal_reason", ""),
        })

    out = os.path.join(BUNDLE, "auto_partial.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # summary
    import collections
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0])  # method -> [n, sum_binary, sum_partial]
    diverge = 0
    for r in rows:
        a = agg[r["method"]]
        a[0] += 1; a[1] += float(r["auto_binary"] or 0); a[2] += r["auto_partial"]
        # sanity: binarized partial (all met) should match binary
        if (1.0 if r["auto_partial"] >= 0.999 else 0.0) != float(r["auto_binary"] or 0):
            diverge += 1

    print(f"wrote {out}  ({len(rows)} items)\n")
    print(f"{'method':13s} {'n':>3s} {'auto_binary':>11s} {'auto_partial':>12s} {'Δ(partial-bin)':>14s}")
    for m in ("Direct", "SAGE", "Hierarchical"):
        if m not in agg: continue
        n, sb, sp = agg[m]
        print(f"{m:13s} {n:>3d} {sb/n:>11.3f} {sp/n:>12.3f} {(sp-sb)/n:>+14.3f}")
    print(f"\nbinarized-partial vs binary mismatch: {diverge}/{len(rows)} "
          f"(low = decomposition consistent with capture checker)")
    # show the items that gained partial credit
    gained = [r for r in rows if r["mode"] == "partial" and 0 < r["auto_partial"] < 1
              and float(r["auto_binary"] or 0) == 0]
    print(f"\nItems lifted 0 -> partial ({len(gained)}):")
    for r in sorted(gained, key=lambda x: -x["auto_partial"])[:20]:
        print(f"  {r['id']:22s} {r['auto_partial']:.2f}  [{r['subgoals']}]")


if __name__ == "__main__":
    main()
