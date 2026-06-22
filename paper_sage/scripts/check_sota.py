#!/usr/bin/env python3
"""check_sota.py — offline (NO-LLM) acceptance gate for the LLM+P and SayCan
SOTA baselines.

Run this BEFORE spending any GPU on a benchmark, exactly like smoke_test.py.
It exercises everything that does NOT need an LLM:

  LLM+P:
    * the fixed domain loads and parses
    * hand-written PDDL problems for known GT tasks solve with pyperplan
    * the grounded plan round-trips to STEP_SCHEMA dicts
    * the round-tripped plan PASSES the symbolic verifier (so the classical
      analogue is consistent with the rule verifier — the fair-comparison
      property)
    * pddl_plan_to_steps covers every domain operator

  SayCan:
    * candidate enumeration is non-empty and well-formed
    * the verifier affordance oracle zeroes precondition-violating candidates
    * a hand-fed score map drives greedy decoding to a verifier-valid plan
      (we monkeypatch _score_candidates so NO LLM is called)

Install prerequisite (pure-python, no native build):
    pip install pyperplan

Usage:
    python scripts/check_sota.py
Exit code 0 = all gates pass.
"""
from __future__ import annotations

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_PAPER_ROOT, ".."))
_PYPLANNER = os.path.join(_REPO_ROOT, "pyplanner")
sys.path.insert(0, _PYPLANNER)

import pyplanner  # noqa: E402
from pyplanner.llmp import (  # noqa: E402
    load_domain, pddl_plan_to_steps, _solve_pddl, _extract_problem_block,
    _OP_TO_ACTION,
)
from pyplanner.saycan import SayCanPlanner, _parse_scores  # noqa: E402
from pyplanner.verifier import simulate, SymbolicState, verify_step  # noqa: E402
from pyplanner.base import ROBOT_ACTIONS  # noqa: E402


_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ok   {name}")
    else:
        _FAIL += 1
        print(f"  FAIL {name}  {detail}")


# ─────────────────────────────────────────────────────────────────────
# Hand-written PDDL problems for 3 known GT tasks (no LLM).
# These mirror eval_dataset_gt.json tasks K01 (coffee), and two fetch tasks.
# ─────────────────────────────────────────────────────────────────────
PROBLEMS = {
    # Turn on the coffee machine (goal: appliance on). Requires find+turnon.
    "coffee_on": (
        """(define (problem coffee)
  (:domain household)
  (:objects StartPos Kitchen CoffeeMachine - obj)
  (:init (arrived StartPos) (hand-empty))
  (:goal (and (turned-on CoffeeMachine))))""",
        # Expected: must contain a TurnOn CoffeeMachine, preceded by Find.
        {"TurnOn"},
    ),
    # Fetch the apple, hold it (goal: holding Apple). Requires find+pick.
    "fetch_apple": (
        """(define (problem fetch)
  (:domain household)
  (:objects StartPos Kitchen Apple - obj)
  (:init (arrived StartPos) (hand-empty))
  (:goal (and (holding Apple))))""",
        {"Pick"},
    ),
    # Open the fridge (goal: opened Fridge). Requires find+open.
    "open_fridge": (
        """(define (problem openf)
  (:domain household)
  (:objects StartPos Kitchen Fridge - obj)
  (:init (arrived StartPos) (hand-empty))
  (:goal (and (opened Fridge))))""",
        {"Open"},
    ),
}


def gate_llmp() -> None:
    print("\n[LLM+P]")
    dom = load_domain()
    check("domain loads non-empty", bool(dom) and "define (domain household" in dom)

    # Operator coverage: every domain (:action ...) maps to a ROBOT_ACTION.
    import re
    ops = set(re.findall(r"\(:action\s+([a-zA-Z0-9\-]+)", dom))
    missing = ops - set(_OP_TO_ACTION)
    check("every domain operator is back-mappable", not missing,
          f"unmapped: {sorted(missing)}")
    bad_actions = {a for a, _ in _OP_TO_ACTION.values()} - set(ROBOT_ACTIONS)
    check("every mapped action is a ROBOT_ACTION", not bad_actions,
          f"not in ROBOT_ACTIONS: {sorted(bad_actions)}")

    try:
        import pyperplan  # noqa: F401
        have_pyperplan = True
    except Exception as e:
        have_pyperplan = False
        print(f"  ---- pyperplan NOT installed ({e}); skipping solve gates.")
        print("  ---- install with: pip install pyperplan")

    if have_pyperplan:
        for tname, (prob, must_have) in PROBLEMS.items():
            try:
                grounded = _solve_pddl(dom, prob)
                steps = pddl_plan_to_steps(grounded)
                check(f"{tname}: solver returns a plan", bool(grounded),
                      f"grounded={grounded}")
                check(f"{tname}: round-trips to steps", bool(steps))
                acts = {s["action"] for s in steps}
                check(f"{tname}: plan contains {must_have}", must_have <= acts,
                      f"got actions {sorted(acts)}")
                # The classical plan must also satisfy the rule verifier — this
                # is the fair-comparison invariant (same world model).
                rep = simulate(steps, stop_on_error=False)
                check(f"{tname}: round-tripped plan passes the verifier", rep.ok,
                      rep.summary())
            except Exception as e:
                check(f"{tname}: solve+roundtrip", False,
                      f"{type(e).__name__}: {e}")

    # _extract_problem_block tolerates markdown + prose.
    noisy = "Here you go:\n```pddl\n" + PROBLEMS["fetch_apple"][0] + "\n```\nDone."
    blk = _extract_problem_block(noisy)
    check("extract_problem_block strips markdown/prose",
          blk is not None and blk.strip().startswith("(define"))
    check("extract_problem_block returns None on garbage",
          _extract_problem_block("no pddl here") is None)


def gate_saycan() -> None:
    print("\n[SayCan]")
    # Construct without contacting any backend (host is never dialed offline).
    planner = SayCanPlanner(host="http://localhost:1", model="x", provider="ollama")

    st = SymbolicState()
    st.visible.update(["Apple", "Fridge", "CoffeeMachine"])
    cands = planner._candidates(st, ["Apple", "Fridge", "CoffeeMachine"])
    check("candidates non-empty", bool(cands))
    check("candidates well-formed",
          all("action" in c and c["action"] in ROBOT_ACTIONS for c in cands))

    # Affordance oracle: Pick before any Find must be infeasible (~0).
    aff_pick = planner._affordance({"action": "Pick", "object": ""}, st)
    check("Pick-before-Find is infeasible", aff_pick <= 0.01, f"aff={aff_pick}")
    # After Find, Pick becomes feasible.
    st2 = st.copy()
    st2.found = "Apple"
    aff_pick2 = planner._affordance({"action": "Pick", "object": "Apple"}, st2)
    check("Pick-after-Find is feasible", aff_pick2 >= 0.99, f"aff={aff_pick2}")

    # Greedy decode with a SCRIPTED scorer (NO LLM): score the verifier-correct
    # next action highest at each step, drive a fetch task to completion.
    target_seq = [("Find", "Apple"), ("Pick", "Apple"),
                  ("MoveTo", "DiningTable"), ("Place", "DiningTable")]

    def scripted_scorer(task, obs, history, state, cands):
        # Prefer the next action in target_seq that is feasible right now.
        want = None
        for a, o in target_seq:
            # skip ones we already did (by position in history)
            done_pairs = [(h["action"], h.get("object", "")) for h in history]
            if (a, o) in done_pairs:
                continue
            want = (a, o)
            break
        scores = {}
        for i, c in enumerate(cands):
            key = (c["action"], c.get("object", ""))
            scores[i] = 0.95 if want and key == want else 0.05
        done = want is None
        return scores, done, 0, 0

    planner._score_candidates = scripted_scorer  # type: ignore[assignment]
    plan, metrics = planner.generate_plan(
        "Put the apple on the dining table", "obs",
        ["Apple", "DiningTable"])
    check("scripted greedy produced a plan", bool(plan), f"plan={plan}")
    rep = simulate(plan, stop_on_error=False)
    check("scripted greedy plan passes the verifier", rep.ok, rep.summary())
    seq = [(s["action"], s["object"]) for s in plan]
    check("greedy recovered the intended fetch sequence",
          seq[:4] == target_seq, f"seq={seq}")
    check("no LLM calls counted (scripted)", metrics.llm_calls == 0,
          f"calls={metrics.llm_calls}")

    # _parse_scores robustness.
    sc, done = _parse_scores('{"scores": {"0": 0.9, "1": 0.2}, "done": false}', 3)
    check("parse_scores reads json", sc.get(0) == 0.9 and not done)
    sc2, _ = _parse_scores("garbage no json", 2)
    check("parse_scores defaults on garbage", sc2.get(0) == 0.3 and sc2.get(1) == 0.3)


def gate_registry() -> None:
    print("\n[Registry]")
    check("LLM+P registered", "LLM+P" in pyplanner.REGISTRY)
    check("SayCan registered", "SayCan" in pyplanner.REGISTRY)
    try:
        p1 = pyplanner.get("LLM+P", model="x")
        check("get('LLM+P') constructs", p1.name == "LLM+P")
    except Exception as e:
        check("get('LLM+P') constructs", False, f"{type(e).__name__}: {e}")
    try:
        p2 = pyplanner.get("SayCan", model="x")
        check("get('SayCan') constructs", p2.name == "SayCan")
    except Exception as e:
        check("get('SayCan') constructs", False, f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 60)
    print("SOTA baselines offline acceptance gate (no LLM)")
    print("=" * 60)
    try:
        gate_registry()
        gate_llmp()
        gate_saycan()
    except Exception:
        traceback.print_exc()
        return 2
    print("\n" + "=" * 60)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
