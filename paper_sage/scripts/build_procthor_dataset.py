#!/usr/bin/env python3
"""build_procthor_dataset.py
=============================
Build an OUT-OF-DISTRIBUTION evaluation dataset on procedurally-generated
ProcTHOR-10k *val*-split houses, using the EXACT SAME task-generation and
simulator-verification discipline as ``scripts/expand_dataset.py`` (iTHOR).

This is the Stage-0 builder for the "Generalization to unseen ProcTHOR
layouts" experiment (see ICRA/plans/generalization-procthor.md). It does NOT
edit any pyplanner module, any baseline planner, or make_dataset_from_sim.py —
the no-edit-to-method/infra property is itself the evidence for the paper's
"no re-tuning" claim.

Quality discipline (identical bar to the 38 curated iTHOR tasks):
  1. Reset the LIVE simulator to each ProcTHOR val house and read real
     affordances from get_objects (pickupable/openable/toggleable/receptacle).
  2. Instantiate templated candidate tasks from ONLY the observed objects,
     with reference plans in the canonical ROBOT_ACTIONS vocabulary.
  3. EXECUTE every candidate's reference plan in AI2-THOR; keep a task only if
     every interaction step returns success=True (genuinely executable).
  4. De-duplicate descriptions; assign house-prefixed task_ids.
  5. Save the full verified pool so re-selection is offline/instant.

The candidate templates and the executable-verification gate are reused from
``expand_dataset`` VERBATIM (imported, not duplicated) — only the scene loop
and the reset entry point differ (ProcTHOR uses split+house_index instead of a
FloorPlan scene string).

Run (a ProcTHOR-capable thor_server must be listening; here :5558):
    OLLAMA_TIMEOUT=300 python \
        scripts/build_procthor_dataset.py \
          --sim-host localhost --sim-port 5558 \
          --split val --num-houses 40 --target-new 70 \
          --out ../pyplanner/eval_dataset_procthor.json \
          --outdir results/procthor_expansion

Offline re-selection from a previously-saved verified pool (no simulator):
    ... build_procthor_dataset.py --reselect --target-new 60 ...

Outputs:
  <out>                                   — verified OOD task set (same schema
                                            as eval_dataset_expanded.json, plus
                                            sim_type/split/house_index fields)
  results/procthor_expansion/verified_pool.json — full verified pool
  results/procthor_expansion/report.json        — per-candidate verify log
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYPLANNER = os.path.abspath(os.path.join(_HERE, "..", "..", "pyplanner"))
sys.path.insert(0, _PYPLANNER)
sys.path.insert(0, os.path.join(_PYPLANNER, "apps"))
sys.path.insert(0, os.path.join(_PYPLANNER, "apps", "evaluate"))

from sim_client import SimClient                                   # noqa: E402
# Reuse the candidate templates + the executable-verification gate VERBATIM
# from the iTHOR expander. Same action vocabulary → no edits needed.
from expand_dataset import gen_candidates, camel_to_words          # noqa: E402


# ── ProcTHOR scene inspector ──────────────────────────────────────────
# Mirrors make_dataset_from_sim.inspect_scene, but resets via
# (split, house_index) instead of a FloorPlan scene string. Kept local so
# make_dataset_from_sim.py stays untouched.
def inspect_house(client: SimClient, split: str, house_index: int) -> dict:
    """Reset to a ProcTHOR house and return its full object inventory.

    Returns the same dict shape inspect_scene produces (the fields
    gen_candidates consumes: pickupable/openable/toggleable/receptacles +
    visible_objects/obs/object_map).
    """
    resp = client.reset(simulator_type="procthor", split=split,
                        house_index=house_index)
    if resp.get("status") != "ok":
        raise RuntimeError(
            f"Cannot load procthor {split}:{house_index}: {resp.get('msg','')}"
        )

    obj_resp = client.get_objects()
    if obj_resp.get("status") != "ok":
        raise RuntimeError(f"get_objects failed: {obj_resp.get('msg','')}")

    objects = obj_resp.get("objects", [])

    object_map: dict[str, dict] = {}
    for o in objects:
        t = o.get("objectType")
        if t is None:
            continue
        d = o.get("distance", 1e9)
        if t not in object_map or d < object_map[t].get("distance", 1e9):
            object_map[t] = o

    visible    = sorted({o["objectType"] for o in objects if o.get("visible")})
    pickupable = sorted({t for t, o in object_map.items() if o.get("pickupable")})
    openable   = sorted({t for t, o in object_map.items() if o.get("openable")})
    toggleable = sorted({t for t, o in object_map.items() if o.get("toggleable")})
    receptacles = sorted({t for t, o in object_map.items() if o.get("receptacle")})

    return {
        "scene":           f"procthor_{split}_{house_index}",
        "obs":             obj_resp.get("obs", resp.get("obs", "")),
        "visible_objects": obj_resp.get("visible_objects", visible),
        "all_objects":     objects,
        "pickupable":      pickupable,
        "openable":        openable,
        "toggleable":      toggleable,
        "receptacles":     receptacles,
        "object_map":      object_map,
        "split":           split,
        "house_index":     house_index,
    }


def verify_house(client: SimClient, split: str, house_index: int,
                steps: list[dict]) -> tuple[bool, str]:
    """ProcTHOR analogue of expand_dataset.verify: reset the house, then
    execute the reference plan; ok iff every INTERACTION step succeeds.

    (We cannot reuse expand_dataset.verify directly because its reset takes a
    scene string; the candidate loop / interaction check are identical here.)
    """
    r = client.reset(simulator_type="procthor", split=split,
                    house_index=house_index)
    if r.get("status") != "ok":
        return False, f"reset failed: {r.get('msg','')[:60]}"
    interaction = {"Find", "Pick", "Place", "PutIn", "Open", "Close",
                "TurnOn", "TurnOff", "Wash"}
    for st in steps:
        try:
            resp = client.step(st["action"], st.get("object", ""),
                            st.get("target", ""))
        except Exception as e:
            return False, f"{st['action']}({st.get('object')}) raised {type(e).__name__}"
        if st["action"] in interaction and not resp.get("success", False):
            return False, f"{st['action']}({st.get('object')}) -> {resp.get('msg','fail')[:50]}"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-host", default="localhost")
    ap.add_argument("--sim-port", type=int, default=5558)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-houses", type=int, default=40,
                    help="House indices 0..num-houses-1 of the chosen split.")
    ap.add_argument("--house-indices", type=int, nargs="+", default=None,
                    help="Explicit house indices (overrides --num-houses).")
    ap.add_argument("--target-new", type=int, default=70)
    ap.add_argument("--reselect", action="store_true",
                    help="Skip the simulator and re-select from a previously "
                        "saved verified_pool.json (offline, instant).")
    ap.add_argument("--out", default=os.path.join(_PYPLANNER, "eval_dataset_procthor.json"))
    ap.add_argument("--outdir", default=os.path.join(_HERE, "..", "results", "procthor_expansion"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pool_path = os.path.join(args.outdir, "verified_pool.json")
    report_path = os.path.join(args.outdir, "report.json")

    house_indices = (args.house_indices if args.house_indices is not None
                    else list(range(args.num_houses)))

    seen_desc: set[str] = set()
    report: list[dict] = []

    if args.reselect and os.path.exists(pool_path):
        verified_pool = json.load(open(pool_path))
        print(f"[reselect] loaded {len(verified_pool)} verified tasks from {pool_path}")
    else:
        client = SimClient.for_thor(host=args.sim_host, port=args.sim_port,
                                    timeout_ms=120_000)
        verified_pool = []
        for hi in house_indices:
            # Room label is cosmetic for ProcTHOR multi-room houses; tag by house.
            room = "house"
            pre = "P"
            try:
                info = inspect_house(client, args.split, hi)
            except Exception as e:
                report.append({"house_index": hi, "error": str(e)[:120]})
                json.dump(report, open(report_path, "w"), indent=2)
                print(f"  [xx ] house {hi:3d}  inspect error: {str(e)[:60]}", flush=True)
                continue
            n_pick = len(info["pickupable"]); n_open = len(info["openable"])
            n_tog = len(info["toggleable"]); n_rec = len(info["receptacles"])
            print(f"  house {hi:3d}: pick={n_pick} open={n_open} tog={n_tog} "
                f"recept={n_rec} (visible={len(info['visible_objects'])})",
                flush=True)
            for desc, diff, exp, steps, sim_steps in gen_candidates(info, room):
                key = desc.strip().lower()
                if key in seen_desc:
                    continue
                seen_desc.add(key)
                ok, reason = verify_house(client, args.split, hi, sim_steps)
                report.append({"house_index": hi, "task": desc,
                            "difficulty": diff, "verified": ok, "reason": reason})
                json.dump(report, open(report_path, "w"), indent=2)
                print(f"    [{'OK ' if ok else 'xx '}] h{hi:<3d} {diff:6s} "
                    f"{desc[:46]:46s} | {reason[:30]}", flush=True)
                if ok:
                    verified_pool.append({
                        "house_index": hi, "split": args.split,
                        "scene": f"procthor_{args.split}_{hi}",
                        "room": room, "pre": pre, "desc": desc, "diff": diff,
                        "exp": exp, "steps": steps,
                        "vis_pool": info["visible_objects"],
                        "obs": info["obs"],
                    })
        json.dump(verified_pool, open(pool_path, "w"), indent=2)

    # ── Balanced selection: match the original 14e:16m:8h mix, spread across
    #    houses round-robin so the set isn't dominated by object-rich houses. ──
    quota = {"easy": round(args.target_new * 14 / 38),
            "medium": round(args.target_new * 16 / 38),
            "hard": args.target_new - round(args.target_new * 14 / 38)
                    - round(args.target_new * 16 / 38)}
    pool_dr = defaultdict(list)
    for c in verified_pool:
        pool_dr[(c["diff"], c["house_index"])].append(c)
    houses = sorted({c["house_index"] for c in verified_pool})
    selected = []
    for d in ("easy", "medium", "hard"):
        queues = {h: list(pool_dr[(d, h)]) for h in houses}
        need, ri = quota[d], 0
        while need > 0 and any(queues.values()):
            h = houses[ri % len(houses)]
            if queues[h]:
                selected.append(queues[h].pop(0)); need -= 1
            ri += 1
    # top up from leftovers (round-robin by house) if a tier ran short
    if len(selected) < args.target_new:
        leftover_by_house = defaultdict(list)
        for c in verified_pool:
            if c not in selected:
                leftover_by_house[c["house_index"]].append(c)
        ri = 0
        while len(selected) < args.target_new and any(leftover_by_house.values()):
            h = houses[ri % len(houses)]
            if leftover_by_house[h]:
                selected.append(leftover_by_house[h].pop(0))
            ri += 1

    # ── Emit samples (same schema as eval_dataset_expanded.json + OOD meta) ──
    maxid = defaultdict(int)
    new_tasks = []
    for c in selected:
        hi = c["house_index"]
        maxid[hi] += 1
        tid = f"P{hi:02d}_{maxid[hi]:02d}"
        vis = sorted(set(c["exp"]) | set(c["vis_pool"][:4]))
        new_tasks.append({
            "task_id": tid,
            "task_desc": c["desc"],
            "room": c["room"],
            "scene": c["scene"],          # cosmetic label procthor_val_<hi>
            "obs": c.get("obs") or f"You are in a ProcTHOR house. Relevant objects are visible.",
            "visible_objects": vis,
            "reference_steps": c["steps"],
            "expected_objects": c["exp"],
            "difficulty": c["diff"],
            "fail_injection": None,
            "source": "build_procthor_dataset.sim_verified",
            # ── OOD / ProcTHOR routing fields (consumed by evaluate_sim) ──
            "sim_type": "procthor",
            "split": c["split"],
            "house_index": hi,
            "_meta": {
                "simulator_type": "procthor",
                "split": c["split"],
                "house_index": hi,
            },
        })

    json.dump({"version": "procthor-1.0", "samples": new_tasks},
            open(args.out, "w"), indent=2)
    json.dump(report, open(report_path, "w"), indent=2)

    n_houses_used = len({t["house_index"] for t in new_tasks})
    print(f"\nVerified pool: {len(verified_pool)} "
        f"(by diff: {dict(Counter(c['diff'] for c in verified_pool))}); "
        f"selected {len(new_tasks)} across {n_houses_used} houses "
        f"(diff: {dict(Counter(t['difficulty'] for t in new_tasks))}).")
    nv = sum(1 for r in report if r.get("verified"))
    ntot = len([r for r in report if "verified" in r])
    print(f"Candidate pass rate: {nv}/{ntot}")
    print(f"Dataset → {args.out}  ({len(new_tasks)} tasks)")


if __name__ == "__main__":
    main()
