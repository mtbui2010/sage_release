#!/usr/bin/env python3
"""collect_transitions.py  (Phase 2: auto/learned verifier)
==========================================================
Probe AI2-THOR systematically to collect (pre-state features, action, success)
transitions, so the verifier's precondition rules can be INDUCED from
interaction data instead of hand-written. Uses the simulator + SimClient only
(NO Ollama) -> safe to run in parallel with the LLM grid.

For each scene we drive the agent into controlled pre-states (did-Find? holding?
arrived-target? object affordances) and then attempt an interaction, recording
whether the simulator accepts it. The induced rules (see induce_verifier.py)
should match the hand-written verifier on AI2-THOR and, crucially, can be
re-induced for a new domain (ProcTHOR / ALFWorld) without manual rule authoring.

The probe driver (``collect_one``) is domain-agnostic: it reads affordances from
``inspect_*``'s object_map, so the SAME probes run on iTHOR FloorPlans and on
ProcTHOR procedurally-generated houses. Only the reset/inspect helper differs.

Run (iTHOR, the 9-scene anchor):
  DISPLAY=:0 PYTHONPATH=...:apps:apps/evaluate \
      /home/keti/miniconda3/bin/python scripts/collect_transitions.py \
      --sim-type thor --sim-port 5556 \
      --scenes FloorPlan1 FloorPlan2 FloorPlan3 FloorPlan201 FloorPlan203 \
               FloorPlan301 FloorPlan303 FloorPlan401 FloorPlan403 \
      --objs-per-aff 5 --out results/autoverify/transitions.csv

Run (ProcTHOR transfer, unseen val houses, SAME probes):
  DISPLAY=:0 PYTHONPATH=...:apps:apps/evaluate \
      /home/keti/miniconda3/bin/python scripts/collect_transitions.py \
      --sim-type procthor --sim-port 5557 \
      --houses 0 1 2 3 4 5 6 7 8 9 --objs-per-aff 5 \
      --out results/autoverify/transitions_procthor.csv
"""
from __future__ import annotations
import argparse, csv, os, sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.abspath(os.path.join(_HERE, "..", "..", "pyplanner"))
for p in (_PY, os.path.join(_PY, "apps"), os.path.join(_PY, "apps", "evaluate")):
    sys.path.insert(0, p)
from thor_app.sim_client import ThorClient, SimClient   # noqa: E402
from make_dataset_from_sim import inspect_scene          # noqa: E402

INTERACTIONS = ["Pick", "Open", "Close", "TurnOn", "TurnOff", "Place", "PutIn"]

FIELDS = ["scene", "action", "object", "did_find", "holding",
          "arrived_target", "target_open",
          "pickupable", "openable", "toggleable", "receptacle",
          "success", "msg"]


def aff(obj_map, otype):
    o = obj_map.get(otype, {})
    return dict(pickupable=int(bool(o.get("pickupable"))),
                openable=int(bool(o.get("openable"))),
                toggleable=int(bool(o.get("toggleable"))),
                receptacle=int(bool(o.get("receptacle"))))


_CLIENT = {"c": None, "host": "localhost", "port": 5555, "sim_type": "thor"}


def _client():
    if _CLIENT["c"] is None:
        if _CLIENT["sim_type"] == "procthor":
            _CLIENT["c"] = SimClient.for_procthor(host=_CLIENT["host"], port=_CLIENT["port"])
        else:
            _CLIENT["c"] = ThorClient(host=_CLIENT["host"], port=_CLIENT["port"])
    return _CLIENT["c"]


# ── ProcTHOR inspector: identical to inspect_scene but resets by house index ──

def inspect_house(client, split, house_index):
    """ProcTHOR analogue of make_dataset_from_sim.inspect_scene: reset a val/train
    house by index, then return the SAME object inventory dict shape so the
    probe driver (collect_one) is unchanged. The only domain-specific line is the
    reset call (split + house_index instead of a FloorPlan string)."""
    resp = client.reset(simulator_type="procthor", split=split, house_index=house_index)
    if resp.get("status") != "ok":
        raise RuntimeError(f"Cannot load procthor {split}:{house_index}: {resp.get('msg','')}")
    obj_resp = client.get_objects()
    if obj_resp.get("status") != "ok":
        raise RuntimeError(f"get_objects failed: {obj_resp.get('msg','')}")
    objects = obj_resp["objects"]
    object_map = {}
    for o in objects:
        t = o["objectType"]
        if t not in object_map or o.get("distance", 1e9) < object_map[t].get("distance", 1e9):
            object_map[t] = o
    visible    = sorted({o["objectType"] for o in objects if o.get("visible")})
    pickupable = sorted({t for t, o in object_map.items() if o.get("pickupable")})
    openable   = sorted({t for t, o in object_map.items() if o.get("openable")})
    toggleable = sorted({t for t, o in object_map.items() if o.get("toggleable")})
    receptacles= sorted({t for t, o in object_map.items() if o.get("receptacle")})
    label = f"procthor_{split}_{client.house_index if client.house_index is not None else house_index}"
    return {
        "scene": label, "obs": obj_resp.get("obs", ""),
        "visible_objects": obj_resp.get("visible_objects", visible),
        "all_objects": objects, "pickupable": pickupable, "openable": openable,
        "toggleable": toggleable, "receptacles": receptacles, "object_map": object_map,
        "reset_kwargs": dict(simulator_type="procthor", split=split, house_index=house_index),
    }


def attempt(reset_kwargs, setup, action, obj, target=""):
    """Reset, run setup steps, then the probe action. Resilient to simulator
    timeouts/hangs: on exception, drop the (now-broken) socket, reconnect, and
    return a skip marker so collection continues."""
    try:
        c = _client()
        if "house_index" in reset_kwargs or reset_kwargs.get("simulator_type") == "procthor":
            c.reset(**reset_kwargs)
        else:
            c.reset(reset_kwargs["scene"])
        for a, o, t in setup:
            c.step(a, o, t)
        r = c.step(action, obj, target)
        return int(bool(r.get("success"))), str(r.get("msg", ""))[:50]
    except Exception as e:
        _CLIENT["c"] = None  # force fresh socket next time (ZMQ REQ broke)
        return None, f"EXC {type(e).__name__}"


def collect_one(scene_label, info, reset_kwargs, objs_per_aff, writer, flusher, counter):
    """Domain-agnostic probe driver. Runs the full contrastive probe battery on a
    single already-inspected scene/house and writes rows. ``info`` is the dict
    returned by inspect_scene / inspect_house; ``reset_kwargs`` is whatever
    attempt() needs to re-enter that scene/house. Returns the running total.

    Identical for iTHOR and ProcTHOR — this is the single source of truth for the
    probe protocol (§1.3 of the plan)."""
    om = info["object_map"]
    pick = info["pickupable"][:objs_per_aff]
    openable = info["openable"][:objs_per_aff]
    toggle = info["toggleable"][:objs_per_aff]
    recept = [r for r in info["receptacles"] if r not in ("Floor",)][:objs_per_aff]
    a_filler = pick[0] if pick else None  # a held object for "holding" probes

    n = counter["n"]

    def rec(action, obj, did_find, holding, arrived, target_open, success, msg, tgt=""):
        nonlocal n
        if success is None:      # simulator hung on this probe → skip
            return
        a = aff(om, tgt if (action in ("Place", "PutIn") and tgt) else obj)
        row = dict(scene=scene_label, action=action, object=obj, did_find=did_find,
                   holding=holding, arrived_target=arrived, target_open=target_open,
                   success=success, msg=msg, **a)
        writer.writerow(row); flusher(); n += 1

    # ---- Pick / Open / Close / TurnOn / TurnOff: need Find(obj) ----
    for act, pool in [("Pick", pick), ("Open", openable), ("Close", openable),
                      ("TurnOn", toggle), ("TurnOff", toggle)]:
        for o in pool:
            # without Find
            s, m = attempt(reset_kwargs, [("MoveTo", o, "")], act, o)
            rec(act, o, 0, 0, 0, 0, s, m)
            # with Find
            pre = [("MoveTo", o, ""), ("Find", o, "")]
            if act == "Close":
                pre += [("Open", o, "")]
            if act == "TurnOff":
                pre += [("TurnOn", o, "")]
            s, m = attempt(reset_kwargs, pre, act, o)
            rec(act, o, 1, 0, 0, 0, s, m)
            # Pick while already holding something
            if act == "Pick" and a_filler and a_filler != o:
                pre2 = [("MoveTo", a_filler, ""), ("Find", a_filler, ""), ("Pick", a_filler, ""),
                        ("MoveTo", o, ""), ("Find", o, "")]
                s, m = attempt(reset_kwargs, pre2, "Pick", o)
                rec("Pick", o, 1, 1, 0, 0, s, m)

    # ---- Place(receptacle): need holding + arrived(receptacle) ----
    for r in recept:
        if not a_filler:
            break
        grab = [("MoveTo", a_filler, ""), ("Find", a_filler, ""), ("Pick", a_filler, "")]
        # holding + arrived
        s, m = attempt(reset_kwargs, grab + [("MoveTo", r, "")], "Place", r)
        rec("Place", r, 0, 1, 1, 0, s, m, tgt=r)
        # not holding + arrived
        s, m = attempt(reset_kwargs, [("MoveTo", r, "")], "Place", r)
        rec("Place", r, 0, 0, 1, 0, s, m, tgt=r)
        # ---- (A.i) BALANCED arrived_target negative: holding but NOT arrived ----
        # Matched pair: same receptacle, same holding=1, vary arrived_target.
        # Pick the filler (so holding=1) but do NOT MoveTo r → arrived=0 → expect
        # fail. One per receptacle makes arrived_target constant-among-successes.
        s, m = attempt(reset_kwargs, grab, "Place", r)
        rec("Place", r, 0, 1, 0, 0, s, m, tgt=r)

    # ---- PutIn(container): need holding + container open ----
    containers = [c for c in openable if aff(om, c)["receptacle"]][:objs_per_aff]
    for c in containers:
        if not a_filler:
            break
        grab = [("MoveTo", a_filler, ""), ("Find", a_filler, ""), ("Pick", a_filler, "")]
        # (A.ii) MATCHED PAIR holding 'openable' affordance FIXED, varying target_open:
        #   open container  → PutIn → expect success (holding=1, openable=1, target_open=1)
        #   closed container→ PutIn → expect fail    (holding=1, openable=1, target_open=0)
        # Same container instance in both arms → 'openable' is constant across the
        # pair, so the miner cannot use it to separate success from failure and is
        # forced onto target_open. (Repeated per container to raise n_pos.)
        s, m = attempt(reset_kwargs, grab + [("MoveTo", c, ""), ("Find", c, ""), ("Open", c, "")], "PutIn", c)
        rec("PutIn", c, 1, 1, 0, 1, s, m, tgt=c)
        s, m = attempt(reset_kwargs, grab + [("MoveTo", c, ""), ("Find", c, "")], "PutIn", c)
        rec("PutIn", c, 1, 1, 0, 0, s, m, tgt=c)

    # ---- contrastive probes (give each precondition both +/- examples) ----
    negobj = next((p for p in info["pickupable"]
                   if not aff(om, p)["toggleable"] and not aff(om, p)["openable"]), None)
    if negobj:
        pre = [("MoveTo", negobj, ""), ("Find", negobj, "")]
        s, m = attempt(reset_kwargs, pre, "TurnOn", negobj)   # wrong affordance
        rec("TurnOn", negobj, 1, 0, 0, 0, s, m)
        s, m = attempt(reset_kwargs, pre, "Open", negobj)
        rec("Open", negobj, 1, 0, 0, 0, s, m)
        s, m = attempt(reset_kwargs, pre, "Close", negobj)
        rec("Close", negobj, 1, 0, 0, 0, s, m)
        s, m = attempt(reset_kwargs, pre, "TurnOff", negobj)
        rec("TurnOff", negobj, 1, 0, 0, 0, s, m)
    # Pick a found-but-NON-pickupable fixed object -> negative for pickupable.
    nonpick = next((t for t, o in om.items()
                    if not o.get("pickupable")
                    and (o.get("openable") or o.get("toggleable") or o.get("receptacle"))), None)
    if nonpick:
        pre = [("MoveTo", nonpick, ""), ("Find", nonpick, "")]
        s, m = attempt(reset_kwargs, pre, "Pick", nonpick)
        rec("Pick", nonpick, 1, 0, 0, 0, s, m)
    # PutIn negatives: (a) container open but NOT holding -> holding precondition.
    if containers and a_filler:
        c0 = containers[0]
        s, m = attempt(reset_kwargs, [("MoveTo", c0, ""), ("Find", c0, ""), ("Open", c0, "")], "PutIn", c0)
        rec("PutIn", c0, 1, 0, 0, 1, s, m, tgt=c0)
    # (A.ii) break the openable<->target_open correlation: holding + arrived at a
    # NON-openable receptacle, attempt PutIn. openable=0 here, so the miner sees a
    # receptacle PutIn whose success/failure does NOT track openable -> openable
    # is no longer constant-among-successes, target_open is.
    nonopen_recept = next((r for r in info["receptacles"]
                           if r not in ("Floor",) and not aff(om, r)["openable"]), None)
    if a_filler and nonopen_recept:
        grab = [("MoveTo", a_filler, ""), ("Find", a_filler, ""), ("Pick", a_filler, "")]
        s, m = attempt(reset_kwargs, grab + [("MoveTo", nonopen_recept, ""), ("Find", nonopen_recept, "")],
                       "PutIn", nonopen_recept)
        # target_open is irrelevant on a non-openable receptacle; record 1 (the
        # "openness" precondition is vacuously satisfiable) so a success here does
        # not force target_open off. If the sim rejects PutIn on a flat receptacle
        # this is simply a failure row, which is fine.
        rec("PutIn", nonopen_recept, 1, 1, 0, 1, s, m, tgt=nonopen_recept)
    if a_filler and negobj:
        grab = [("MoveTo", a_filler, ""), ("Find", a_filler, ""), ("Pick", a_filler, "")]
        s, m = attempt(reset_kwargs, grab + [("MoveTo", negobj, ""), ("Find", negobj, "")], "PutIn", negobj)
        rec("PutIn", negobj, 1, 1, 0, 0, s, m, tgt=negobj)
    if a_filler and recept:
        r = recept[0]
        grab = [("MoveTo", a_filler, ""), ("Find", a_filler, ""), ("Pick", a_filler, "")]
        s, m = attempt(reset_kwargs, grab, "Place", r)        # holding but NOT arrived
        rec("Place", r, 0, 1, 0, 0, s, m, tgt=r)
        if negobj and negobj != a_filler:                      # place onto non-receptacle
            s, m = attempt(reset_kwargs, grab + [("MoveTo", negobj, "")], "Place", negobj)
            rec("Place", negobj, 0, 1, 1, 0, s, m, tgt=negobj)

    counter["n"] = n
    print(f"[{scene_label}] collected, running total = {n}", flush=True)
    return n


def balance_report(out_csv):
    """Self-check (§1.3): for every (action, feature) report ±-counts among
    successes and failures, and warn if any candidate precondition lacks a clean
    contrastive pair (a success and a failure that differ in that feature). Cited
    in the paper as 'every mined condition had a contrastive pair.'"""
    from collections import Counter
    rows = list(csv.DictReader(open(out_csv)))
    feats = ["did_find", "holding", "arrived_target", "target_open",
             "pickupable", "openable", "toggleable", "receptacle"]
    by_act = defaultdict(list)
    for r in rows:
        by_act[r["action"]].append(r)
    print("\n=== --balance contrastive self-check ===")
    for act in INTERACTIONS:
        rs = by_act.get(act, [])
        succ = [r for r in rs if r["success"] == "1"]
        fail = [r for r in rs if r["success"] == "0"]
        print(f"[{act}] n={len(rs)} pos={len(succ)} neg={len(fail)}")
        for f in feats:
            sc = Counter(r[f] for r in succ)
            fc = Counter(r[f] for r in fail)
            # a clean contrast exists if some success value v has a failure with !=v
            ok = any(any(rf[f] != v for rf in fail) for v in {r[f] for r in succ}) if succ else False
            flag = "" if ok or not succ else "  <-- NO CONTRAST"
            print(f"    {f:14s} succ{dict(sc)} fail{dict(fc)}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-type", choices=["thor", "procthor"], default="thor",
                    help="thor = iTHOR FloorPlans (--scenes); procthor = val houses (--houses)")
    ap.add_argument("--scenes", nargs="+",
                    default=["FloorPlan1", "FloorPlan2", "FloorPlan3",
                             "FloorPlan201", "FloorPlan203", "FloorPlan301",
                             "FloorPlan303", "FloorPlan401", "FloorPlan403"])
    ap.add_argument("--houses", nargs="+", type=int,
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    help="ProcTHOR val-split house indices (used when --sim-type procthor)")
    ap.add_argument("--split", default="val", help="ProcTHOR split (default val)")
    ap.add_argument("--sim-host", default="localhost")
    ap.add_argument("--sim-port", type=int, default=5556)
    ap.add_argument("--objs-per-aff", type=int, default=5)
    ap.add_argument("--balance", action="store_true",
                    help="after collection, print a per-feature ±-count contrastive self-check")
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results", "autoverify", "transitions.csv"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    _CLIENT["host"], _CLIENT["port"] = args.sim_host, args.sim_port
    _CLIENT["sim_type"] = args.sim_type
    client = _client()

    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
    counter = {"n": 0}
    flush = f.flush

    if args.sim_type == "procthor":
        units = [("house", hi) for hi in args.houses]
    else:
        units = [("scene", sc) for sc in args.scenes]

    for kind, unit in units:
        try:
            if kind == "house":
                info = inspect_house(client, args.split, unit)
                reset_kwargs = info["reset_kwargs"]
                scene_label = info["scene"]
            else:
                info = inspect_scene(client, unit)
                reset_kwargs = {"scene": unit}
                scene_label = unit
        except Exception as e:
            print(f"[skip] {kind}:{unit}: {e}")
            _CLIENT["c"] = None
            client = _client()
            continue
        try:
            collect_one(scene_label, info, reset_kwargs, args.objs_per_aff, w, flush, counter)
        except Exception as e:
            print(f"[err] {scene_label}: {type(e).__name__}: {e}")
            _CLIENT["c"] = None
            client = _client()

    f.close()
    print(f"Wrote {counter['n']} transitions -> {args.out}")
    if args.balance:
        balance_report(args.out)


if __name__ == "__main__":
    main()
