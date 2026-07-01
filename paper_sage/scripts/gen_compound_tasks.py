#!/usr/bin/env python3
"""Generate a HARDER, method-agnostic stress-test benchmark by composing existing
single-goal AI2-THOR tasks into multi-goal compound tasks.

Rationale (see ICRA/plans/why-completeness-small.md): the 75-task completeness
metric saturates (65% of instances at 1.0), leaving no headroom to discriminate
methods. Composing same-scene tasks into 2- and 3-goal sequences lengthens the
horizon and unions the expected-object set, so completeness regains headroom.

The difficulty axis is purely structural (task composition); it references nothing
about SAGE. Ground-truth reference = concatenation of the already
simulator-verified sub-plans, re-checked here through the symbolic verifier, so no
LLM is used to author references (no new leakage, no hand-authoring bias).

Output: a dataset JSON in the same schema as eval_dataset_expanded.json.
"""
from __future__ import annotations
import argparse, json, os, sys
from itertools import combinations

PYP = "/path/to/pyplanner"
sys.path.insert(0, PYP); sys.path.insert(0, os.path.join(PYP, "apps"))
from pyplanner.verifier import normalize_plan, simulate  # noqa: E402


def _union(seqs):
    out = []
    for s in seqs:
        for x in (s or []):
            if x not in out:
                out.append(x)
    return out


def compose(members: list[dict], degree: int) -> dict:
    ids = [m["task_id"] for m in members]
    descs = [m["task_desc"].rstrip(".") for m in members]
    if degree == 2:
        desc = f"First, {descs[0].lower()}. Then, {descs[1].lower()}."
    else:
        desc = (f"First, {descs[0].lower()}. Then, {descs[1].lower()}. "
                f"Finally, {descs[2].lower()}.")
    ref = []
    for m in members:
        ref.extend(m.get("reference_steps") or [])
    return {
        "task_id": "C{}_".format(degree) + "+".join(ids),
        "task_desc": desc,
        "room": members[0].get("room", ""),
        "scene": members[0].get("scene", ""),
        "obs": members[0].get("obs", ""),
        "visible_objects": _union(m.get("visible_objects") for m in members),
        "reference_steps": ref,
        "expected_objects": _union(m.get("expected_objects") for m in members),
        "difficulty": f"compound{degree}",
        "fail_injection": {},
        "_meta": {"gt_source": "composition", "members": ids, "degree": degree},
    }


def ref_ok(sample: dict) -> bool:
    """Re-verify the concatenated reference through the symbolic verifier."""
    steps = normalize_plan([
        {"action": s.get("action", ""),
         "object": s.get("object", "") or s.get("target", "")}
        for s in sample["reference_steps"]
    ])
    rep = simulate(steps, visible_objects=sample.get("visible_objects") or [])
    return rep.ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=f"{PYP}/eval_dataset_expanded.json")
    ap.add_argument("--out", default="/tmp/compound_stress.json")
    ap.add_argument("--per-scene-deg2", type=int, default=6,
                    help="max degree-2 compounds per scene")
    ap.add_argument("--per-scene-deg3", type=int, default=4,
                    help="max degree-3 compounds per scene")
    ap.add_argument("--require-valid-ref", action="store_true",
                    help="keep only compounds whose concatenated reference "
                         "passes the symbolic verifier")
    args = ap.parse_args()

    data = json.load(open(args.inp))
    by_scene: dict[str, list[dict]] = {}
    for s in data["samples"]:
        by_scene.setdefault(s.get("scene", "?"), []).append(s)

    out, kept, dropped = [], 0, 0
    for scene, members in by_scene.items():
        members = sorted(members, key=lambda m: m["task_id"])
        for deg, cap in ((2, args.per_scene_deg2), (3, args.per_scene_deg3)):
            if len(members) < deg:
                continue
            made = 0
            for combo in combinations(members, deg):
                if made >= cap:
                    break
                samp = compose(list(combo), deg)
                if args.require_valid_ref and not ref_ok(samp):
                    dropped += 1
                    continue
                out.append(samp); kept += 1; made += 1

    json.dump({"version": "compound-stress-1", "samples": out},
              open(args.out, "w"))
    n2 = sum(1 for s in out if s["difficulty"] == "compound2")
    n3 = sum(1 for s in out if s["difficulty"] == "compound3")
    print(f"scenes: {len(by_scene)}  kept: {kept} (deg2={n2}, deg3={n3})  "
          f"dropped(invalid ref): {dropped}")
    print(f"mean expected_objects: "
          f"{sum(len(s['expected_objects']) for s in out)/max(kept,1):.1f}  "
          f"mean ref_steps: "
          f"{sum(len(s['reference_steps']) for s in out)/max(kept,1):.1f}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
