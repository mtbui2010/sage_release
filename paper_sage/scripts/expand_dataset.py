#!/usr/bin/env python3
"""expand_dataset.py
====================
Grow the 38-task AI2-THOR benchmark with ADDITIONAL, object-grounded,
simulator-verified tasks for the ICRA submission.

Quality discipline (so generated tasks can never pollute the benchmark):
  1. Inspect each scene in the LIVE simulator → real affordances
     (pickupable / openable / toggleable / receptacles).
  2. Instantiate templated candidate tasks from those real objects, with
     reference plans in the canonical Navigate/Grab/Place/PutIn/Open/Close/
     TurnOn/TurnOff vocabulary.
  3. EXECUTE every candidate's reference plan in AI2-THOR; keep a task only
     if every interaction step returns success=True (i.e. the reference plan
     is genuinely executable). This is the same bar the original 38 met.
  4. De-duplicate against existing task descriptions; assign room-prefixed
     task_ids that continue the existing scheme.

Run (thor_server must be up on :5555):
  DISPLAY=:0 PYTHONPATH=... /home/keti/miniconda3/bin/python \
      scripts/expand_dataset.py --target-new 40 --out <merged.json>

Outputs:
  results/expansion/new_tasks.json   — verified new tasks only
  results/expansion/report.json      — per-candidate verification report
  <out>                              — merged dataset (original 38 + new)
"""
from __future__ import annotations
import argparse, json, os, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYPLANNER = os.path.abspath(os.path.join(_HERE, "..", "..", "pyplanner"))
sys.path.insert(0, _PYPLANNER)
sys.path.insert(0, os.path.join(_PYPLANNER, "apps"))
sys.path.insert(0, os.path.join(_PYPLANNER, "apps", "evaluate"))

from thor_app.sim_client import ThorClient                      # noqa: E402
from make_dataset_from_sim import inspect_scene                  # noqa: E402

ROOM_BY_SCENE_RANGE = [  # (low, high, room, id_prefix)
    (1, 30, "kitchen", "K"), (200, 230, "living_room", "L"),
    (300, 330, "bedroom", "B"), (400, 430, "bathroom", "A"),
]


def scene_room(scene: str):
    n = int("".join(ch for ch in scene if ch.isdigit()))
    for lo, hi, room, pre in ROOM_BY_SCENE_RANGE:
        if lo <= n <= hi:
            return room, pre
    return "kitchen", "K"


def _step(action, obj="", target="", reason=""):
    return {"action": action, "object": obj, "target": target, "reason": reason}


# Receptacles we trust as "open-and-put-in" containers for hard tasks.
CONTAINER_TYPES = {"Fridge", "Microwave", "Cabinet", "Box", "Safe"}
# Large open surfaces reliable as "place X on Y" targets. We keep ONLY surfaces
# where "place <object> on <surface>" is a commonsense-plausible household goal,
# and exclude sim-valid-but-implausible targets (a garbage can, a chair/armchair,
# a bathtub, a safe) that produced odd templated tasks in an earlier version.
RECEPTACLE_SURFACES = {
    "CounterTop", "DiningTable", "CoffeeTable", "SideTable", "Desk", "Dresser",
    "Shelf", "ShelvingUnit", "TVStand", "Sofa", "Bed", "Ottoman", "Stool",
    "Sink", "SinkBasin",
}
# Don't ask to pick obviously-immovable or trivial things even if flagged.
PICK_BLOCKLIST = {"Floor", "Wall", "Window", "Mirror", "Painting"}
# Toggleables whose NL goal is unnatural ("turn on the candle") even if the
# simulator marks them toggleable — excluded from the easy turn-on template.
TOGGLE_BLOCKLIST = {"Candle"}


def _goto(obj, reason):
    """Stored GT-style nav step (legacy 'Navigate', normalised downstream)."""
    return _step("Navigate", obj, reason=reason)


def _sim_goto_find(obj):
    """Sim-vocabulary [MoveTo, Find] that the current thor_server requires
    before any interaction with `obj`."""
    return [_step("MoveTo", obj), _step("Find", obj)]


_GRAB_ALIAS = {"Grab": "Pick"}  # stored uses legacy Grab; sim wants Pick


def gen_candidates(info: dict, room: str):
    """Yield (task_desc, difficulty, expected_objects, stored_steps, sim_steps).

    stored_steps follow the existing GT style (Navigate/Grab, no explicit Find,
    so step_ratio stays comparable to the original 38). sim_steps are the
    MoveTo/Find/interact sequence the live simulator needs for verification.
    """
    pick = [p for p in info["pickupable"] if p not in PICK_BLOCKLIST]
    openable = info["openable"]
    toggle = [t for t in info["toggleable"] if t not in TOGGLE_BLOCKLIST]
    recept = [r for r in info["receptacles"] if r in RECEPTACLE_SURFACES]
    cands = []

    def emit(desc, diff, exp, ops):
        """ops: list of (verb, obj, target). Build stored + sim sequences."""
        stored, sim = [], []
        for verb, obj, tgt in ops:
            if verb == "Navigate":
                stored.append(_goto(obj, f"Go to {obj}"))
                sim.append(_step("MoveTo", obj))
            elif verb in ("Grab", "Pick"):
                stored.append(_step("Grab", obj, reason=f"Pick up {obj}"))
                sim += [_step("Find", obj), _step("Pick", obj)]
            else:  # interaction needs Find first in the sim
                stored.append(_step(verb, obj, tgt, reason=f"{verb} {obj}"))
                sim += [_step("Find", tgt or obj), _step(verb, obj, tgt)]
        cands.append((desc, diff, exp, stored, sim))

    # ── easy: single interaction ──
    for t in toggle[:3]:
        emit(f"Turn on the {camel_to_words(t)}", "easy", [t],
             [("Navigate", t, ""), ("TurnOn", t, "")])
    for o in openable[:2]:
        emit(f"Open the {camel_to_words(o)}", "easy", [o],
             [("Navigate", o, ""), ("Open", o, "")])
    for p in pick[:2]:
        emit(f"Pick up the {camel_to_words(p)}", "easy", [p],
             [("Navigate", p, ""), ("Grab", p, "")])

    # ── medium: pick-and-place (try several combos; sim verification filters) ──
    for p in pick[:6]:
        for r in recept[:2]:
            if r == p:
                continue
            # sim 'Place' takes the receptacle as its object (held item implicit)
            emit(f"Place the {camel_to_words(p)} on the {camel_to_words(r)}",
                 "medium", [p, r],
                 [("Navigate", p, ""), ("Grab", p, ""),
                  ("Navigate", r, ""), ("Place", r, "")])

    # ── hard: open container, put object in, close ──
    for o in [c for c in openable if c in CONTAINER_TYPES][:2]:
        for p in pick[:4]:
            if p == o:
                continue
            # sim 'PutIn' takes the container as its object (held item implicit)
            emit(f"Put the {camel_to_words(p)} in the {camel_to_words(o)} and close it",
                 "hard", [p, o],
                 [("Navigate", o, ""), ("Open", o, ""),
                  ("Navigate", p, ""), ("Grab", p, ""),
                  ("Navigate", o, ""), ("PutIn", o, ""), ("Close", o, "")])
    return cands


def camel_to_words(s: str) -> str:
    out = []
    for i, ch in enumerate(s):
        if ch.isupper() and i > 0 and not s[i - 1].isupper():
            out.append(" ")
        out.append(ch.lower())
    return "".join(out)


def verify(client: ThorClient, scene: str, steps: list[dict]) -> tuple[bool, str]:
    """Execute reference steps; return (ok, reason). ok iff every INTERACTION
    step succeeds (Navigate may legitimately no-op on some servers)."""
    r = client.reset(scene)
    if isinstance(r, dict) and r.get("success") is False and r.get("msg"):
        return False, f"reset failed: {r.get('msg')}"
    interaction = {"Find", "Pick", "Place", "PutIn", "Open", "Close", "TurnOn", "TurnOff", "Wash"}
    for st in steps:
        try:
            resp = client.step(st["action"], st.get("object", ""), st.get("target", ""))
        except Exception as e:
            return False, f"{st['action']}({st.get('object')}) raised {type(e).__name__}"
        if st["action"] in interaction and not resp.get("success", False):
            return False, f"{st['action']}({st.get('object')}) -> {resp.get('msg','fail')[:60]}"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(_PYPLANNER, "eval_dataset_gt.json"))
    ap.add_argument("--sim-host", default="localhost")
    ap.add_argument("--sim-port", type=int, default=5555)
    ap.add_argument("--target-new", type=int, default=40)
    ap.add_argument("--reselect", action="store_true",
                    help="Skip the simulator and re-select from a previously saved "
                         "results/expansion/verified_pool.json (offline, instant).")
    ap.add_argument("--out", default=os.path.join(_PYPLANNER, "eval_dataset_expanded.json"))
    ap.add_argument("--outdir", default=os.path.join(_HERE, "..", "results", "expansion"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base = json.load(open(args.base))
    base_samples = base.get("samples", base)
    scenes = sorted({s["scene"] for s in base_samples},
                    key=lambda s: int("".join(c for c in s if c.isdigit())))
    seen_desc = {s["task_desc"].strip().lower() for s in base_samples}
    # next id index per prefix
    maxid = {}
    for s in base_samples:
        tid = s["task_id"]
        pre = "".join(c for c in tid if c.isalpha())
        num = int("".join(c for c in tid if c.isdigit()) or 0)
        maxid[pre] = max(maxid.get(pre, 0), num)

    pool_path = os.path.join(args.outdir, "verified_pool.json")
    report = []
    if args.reselect and os.path.exists(pool_path):
        # Offline re-selection from a previously-verified pool (no simulator).
        verified_pool = json.load(open(pool_path))
        print(f"[reselect] loaded {len(verified_pool)} verified tasks from {pool_path}")
    else:
        client = ThorClient(host=args.sim_host, port=args.sim_port)
        # Verify the FULL candidate pool across all scenes, then select a balanced
        # subset (so the expansion isn't dominated by easy tasks emitted first).
        verified_pool = []
        for scene in scenes:
            room, pre = scene_room(scene)
            try:
                info = inspect_scene(client, scene)
            except Exception as e:
                report.append({"scene": scene, "error": str(e)[:120]}); continue
            for desc, diff, exp, steps, sim_steps in gen_candidates(info, room):
                if desc.strip().lower() in seen_desc:
                    continue
                seen_desc.add(desc.strip().lower())
                ok, reason = verify(client, scene, sim_steps)
                report.append({"scene": scene, "task": desc, "difficulty": diff,
                               "verified": ok, "reason": reason})
                json.dump(report, open(os.path.join(args.outdir, "report.json"), "w"), indent=2)
                print(f"  [{'OK ' if ok else 'xx '}] {scene:12s} {diff:6s} {desc[:46]:46s} | {reason[:34]}",
                      flush=True)
                if ok:
                    verified_pool.append({"scene": scene, "room": room, "pre": pre,
                                          "desc": desc, "diff": diff, "exp": exp,
                                          "steps": steps, "vis_pool": info["visible_objects"]})
        # Save the full verified pool for reproducibility / re-selection.
        json.dump(verified_pool, open(pool_path, "w"), indent=2)

    # Balanced selection: match the original difficulty mix (14e:16m:8h) AND
    # spread each difficulty round-robin across rooms so the expansion does not
    # skew toward object-rich kitchen scenes.
    from collections import defaultdict as _dd
    quota = {"easy": round(args.target_new * 14 / 38),
             "medium": round(args.target_new * 16 / 38),
             "hard": args.target_new - round(args.target_new * 14 / 38) - round(args.target_new * 16 / 38)}
    pool_dr = _dd(list)
    for c in verified_pool:
        pool_dr[(c["diff"], c["room"])].append(c)
    rooms = sorted({c["room"] for c in verified_pool})
    selected = []
    for d in ("easy", "medium", "hard"):
        queues = {r: list(pool_dr[(d, r)]) for r in rooms}
        need, ri = quota[d], 0
        while need > 0 and any(queues.values()):
            r = rooms[ri % len(rooms)]
            if queues[r]:
                selected.append(queues[r].pop(0)); need -= 1
            ri += 1
    # top up from any leftovers (round-robin by room) if a tier ran short
    if len(selected) < args.target_new:
        leftover_by_room = _dd(list)
        for c in verified_pool:
            if c not in selected:
                leftover_by_room[c["room"]].append(c)
        ri = 0
        while len(selected) < args.target_new and any(leftover_by_room.values()):
            r = rooms[ri % len(rooms)]
            if leftover_by_room[r]:
                selected.append(leftover_by_room[r].pop(0))
            ri += 1

    new_tasks = []
    for c in selected:
        maxid[c["pre"]] = maxid.get(c["pre"], 0) + 1
        tid = f"{c['pre']}{maxid[c['pre']]:02d}"
        vis = sorted(set(c["exp"]) | set(c["vis_pool"][:4]))
        new_tasks.append({
            "task_id": tid, "task_desc": c["desc"], "room": c["room"], "scene": c["scene"],
            "obs": f"You are in the {c['room']}. Relevant objects are visible.",
            "visible_objects": vis, "reference_steps": c["steps"],
            "expected_objects": c["exp"], "difficulty": c["diff"], "fail_injection": None,
            "source": "expand_dataset.sim_verified",
            # Provenance carried INTO the task so the dataset itself evidences that
            # the reference plan was executed in AI2-THOR and every interaction
            # step returned success=True (same bar as the curated 38).
            "_meta": {
                "gt_source": "simulator_execution",
                "candidate_source": "template",
                "task_success": True,
                "grounded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "verifier": "ai2thor-5.0.0",
                "warnings": [],
            },
        })
    from collections import Counter as _C
    print(f"\nVerified pool: {len(verified_pool)} "
          f"(by diff: {dict(_C(c['diff'] for c in verified_pool))}); "
          f"selected {len(new_tasks)} "
          f"(rooms: {dict(_C(t['room'] for t in new_tasks))}, "
          f"diff: {dict(_C(t['difficulty'] for t in new_tasks))}).")

    json.dump({"version": "expand-1.0", "samples": new_tasks},
              open(os.path.join(args.outdir, "new_tasks.json"), "w"), indent=2)
    json.dump(report, open(os.path.join(args.outdir, "report.json"), "w"), indent=2)
    merged = {"version": base.get("version", "1.0"),
              "samples": base_samples + new_tasks}
    json.dump(merged, open(args.out, "w"), indent=2)
    nv = sum(1 for r in report if r.get("verified"))
    print(f"\nVerified {len(new_tasks)} new tasks "
          f"({nv}/{len([r for r in report if 'verified' in r])} candidates passed).")
    print(f"Merged dataset: {len(base_samples)} + {len(new_tasks)} = "
          f"{len(base_samples)+len(new_tasks)} → {args.out}")


if __name__ == "__main__":
    main()
