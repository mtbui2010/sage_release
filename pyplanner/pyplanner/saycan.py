# pyplanner/saycan.py
# ─────────────────────────────────────────────────────────────────────
# SayCan-style planner (Ahn et al., 2022, "Do As I Can, Not As I Say").
#
# Affordance-grounded greedy decoding.  At each step we:
#   1. Enumerate candidate next actions over the visible objects /
#      affordances (the (action, object) grid that ROBOT_ACTIONS allows).
#   2. Score each candidate by the classic SayCan product:
#          score = p_LLM(candidate | task, history) * p_affordance(candidate)
#      where p_affordance is the feasibility of the candidate in the CURRENT
#      symbolic state.  We use pyplanner.verifier as the affordance oracle:
#      a candidate that would VIOLATE a precondition gets feasibility ~= 0,
#      a feasible candidate gets feasibility ~= 1 (with a small floor so a
#      strongly-preferred-but-borderline action is not hard-zeroed).
#   3. Pick the argmax, append it, advance the symbolic state, and repeat
#      until the LLM signals Done / a goal is reached / a step cap is hit.
#
# Honest-SayCan notes:
#   - The original SayCan uses a LEARNED value function for p_LLM and a
#     LEARNED affordance model for p_affordance.  pyplanner has neither a
#     trained value head nor a learned affordance net, so we instantiate the
#     SAME value*affordance decomposition with (a) an LLM scoring call for the
#     value term and (b) the rule verifier as a deterministic affordance
#     oracle.  This is the faithful structural analogue available here and is
#     declared as such in the paper.
#   - LLM cost is bounded: ONE scoring call per decoding step (the candidate
#     set is ranked in a single call), so total cost is comparable to
#     Hierarchical (1 + N calls), not Self-Refine-like.
#
# Registry key: "SayCan".
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import re
import time

from pyplanner.base import (
    ACTIONS_STR, STEP_SCHEMA,
    BasePlanner, PlanMetrics,
)
from pyplanner.verifier import (
    SymbolicState, verify_step, _apply, DEFAULT_CONTAINERS,
)

# Actions whose `object` field is meaningfully a *target* the LLM must choose.
# Pick is included with object = the last-found item (object field is documented
# as ignored at runtime, but we carry it for readability / scoring).
_OBJECT_ACTIONS = {
    "MoveTo", "Find", "Place", "PutIn",
    "Open", "Close", "TurnOn", "TurnOff",
}
_NOARG_ACTIONS = {"Pick"}

# Rooms / furniture that are valid MoveTo destinations even if not in the
# visible-object list (the visible list is pickupable-biased in the dataset).
_COMMON_PLACES = [
    "Kitchen", "LivingRoom", "Bedroom", "Bathroom",
    "DiningTable", "CounterTop", "Sofa", "Sink", "CoffeeTable",
]

_MAX_STEPS_DEFAULT = 18
_FEASIBLE_VALUE = 1.0       # affordance for verifier-OK candidates
_INFEASIBLE_VALUE = 0.001   # affordance for verifier-rejected candidates


_SCORE_SYSTEM = f"""You are the value head of a SayCan household robot.
Given the task, the actions taken so far, and the robot's current symbolic
state, you must score how USEFUL each candidate next action is for making
progress toward completing the task.  Higher = more useful right now.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

You will be given a numbered list of candidate next actions.  Return ONLY a
JSON object mapping each candidate index (as a string) to a usefulness score
in [0,1], plus a "done" boolean that is true ONLY if the task is already
fully complete and no further action is needed.  No markdown, no prose:
{{"scores": {{"0": 0.9, "1": 0.1, "2": 0.4}}, "done": false}}"""


def _parse_scores(raw: str, n: int) -> tuple[dict[int, float], bool]:
    """Parse the LLM scoring response into {idx: score} and a done flag.

    Robust to markdown fences and partial JSON; missing indices default to a
    small uniform prior so a malformed response never hard-stops decoding.
    """
    raw = re.sub(r"```json|```", "", raw or "").strip()
    scores: dict[int, float] = {}
    done = False
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except Exception:
                data = None
    if isinstance(data, dict):
        done = bool(data.get("done", False))
        sc = data.get("scores", data)
        if isinstance(sc, dict):
            for k, v in sc.items():
                try:
                    scores[int(k)] = max(0.0, min(1.0, float(v)))
                except (ValueError, TypeError):
                    continue
    if not scores:
        # Fallback: scrape "idx: value" pairs.
        for m in re.finditer(r'"?(\d+)"?\s*[:=]\s*([01]?\.?\d+)', raw):
            try:
                scores[int(m.group(1))] = max(0.0, min(1.0, float(m.group(2))))
            except ValueError:
                continue
    # Uniform prior for any unscored candidate.
    for i in range(n):
        scores.setdefault(i, 0.3)
    return scores, done


class SayCanPlanner(BasePlanner):
    name = "SayCan"
    description = (
        "SayCan-style affordance-grounded greedy decoding: at each step the "
        "next action is argmax of value(LLM)×affordance(verifier). The "
        "symbolic verifier is the affordance oracle — precondition-violating "
        "candidates get feasibility~0. One scoring call per step."
    )

    def __init__(self, *args, max_steps: int = _MAX_STEPS_DEFAULT,
                 containers: set[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = int(max_steps)
        self._containers = containers if containers is not None else DEFAULT_CONTAINERS

    # ── candidate enumeration ──
    def _candidates(self, state: SymbolicState,
                    visible_objects: list[str]) -> list[dict]:
        """Enumerate plausible next (action, object) candidates.

        We do NOT pre-filter by feasibility here — feasibility is the
        affordance term scored later (verifier).  We only bound the candidate
        set to keep the scoring prompt small: object-actions are paired with
        visible objects + common places; Pick is a single no-arg candidate.
        """
        objs = list(dict.fromkeys((visible_objects or []) + _COMMON_PLACES))
        cands: list[dict] = []

        # MoveTo: any place/furniture/container.
        for o in objs:
            cands.append({"action": "MoveTo", "object": o})
        # Find: any visible object (not rooms — but harmless if included).
        for o in (visible_objects or []):
            cands.append({"action": "Find", "object": o})
        # Pick: single no-arg candidate (object carried from last found).
        cands.append({"action": "Pick", "object": state.found or ""})
        # Interactions on visible objects / current handle.
        interact_targets = list(dict.fromkeys(
            (visible_objects or []) + ([state.found] if state.found else [])
        ))
        for o in interact_targets:
            for a in ("Open", "Close", "TurnOn", "TurnOff"):
                cands.append({"action": a, "object": o})
        # Place / PutIn on the currently-arrived receptacle / found container.
        if state.holding:
            if state.arrived:
                cands.append({"action": "Place", "object": state.arrived})
            for o in interact_targets:
                cands.append({"action": "PutIn", "object": o})

        # De-duplicate while preserving order; bound the set for prompt size.
        seen = set()
        uniq = []
        for c in cands:
            key = (c["action"], c["object"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        return uniq[:60]

    # ── affordance term from the verifier ──
    def _affordance(self, cand: dict, state: SymbolicState) -> float:
        ok, _code, _reason = verify_step(cand, state, containers=self._containers)
        return _FEASIBLE_VALUE if ok else _INFEASIBLE_VALUE

    # ── LLM value term: one scoring call over the whole candidate set ──
    def _score_candidates(self, task: str, obs: str, history: list[dict],
                          state: SymbolicState,
                          cands: list[dict]) -> tuple[dict[int, float], bool, int, int]:
        hist_str = "\n".join(
            f"  {i+1}. {s.get('action')} {s.get('object','')}".rstrip()
            for i, s in enumerate(history)
        ) or "  (none yet)"
        cand_str = "\n".join(
            f"  {i}: {c['action']} {c.get('object','')}".rstrip()
            for i, c in enumerate(cands)
        )
        user = (
            f"Task: {task}\n\n"
            f"Robot symbolic state:\n  {state.as_text()}\n\n"
            f"Actions taken so far:\n{hist_str}\n\n"
            f"Candidate next actions:\n{cand_str}\n\n"
            "Score each candidate's usefulness now and set done appropriately."
        )
        raw, in_tok, out_tok = self._chat(
            [{"role": "system", "content": _SCORE_SYSTEM},
             {"role": "user", "content": user}],
            temperature=0.0,
        )
        scores, done = _parse_scores(raw, len(cands))
        return scores, done, in_tok, out_tok

    def generate_plan(self, task, obs, visible_objects):
        t0 = time.perf_counter()
        state = SymbolicState()
        if visible_objects:
            state.visible.update(visible_objects)
        plan: list[dict] = []
        llm_calls = 0
        in_tok = out_tok = 0
        notes = ""

        for _step in range(self.max_steps):
            cands = self._candidates(state, visible_objects)
            if not cands:
                break
            try:
                scores, done, it, ot = self._score_candidates(
                    task, obs, plan, state, cands)
                in_tok += it
                out_tok += ot
                llm_calls += 1
            except Exception as e:
                notes = f"score_error:{type(e).__name__}"
                break
            if done:
                break

            # The classic SayCan product: value(LLM) × affordance(verifier),
            # argmax. Feasible candidates have affordance 1.0; precondition-
            # violating candidates have affordance ~0, so they are effectively
            # gated out unless every candidate is infeasible.
            best_i, best_val = -1, -1.0
            for i, c in enumerate(cands):
                aff = self._affordance(c, state)
                val = scores.get(i, 0.3)
                combined = val * aff
                if combined > best_val:
                    best_val, best_i = combined, i

            if best_i < 0:
                break
            chosen = cands[best_i]

            # Safety: never append a hard-infeasible step (affordance ~0); if
            # the best candidate is still infeasible, decoding is stuck → stop.
            if self._affordance(chosen, state) <= _INFEASIBLE_VALUE:
                notes = notes or "stuck_all_infeasible"
                break

            plan.append({"action": chosen["action"], "object": chosen.get("object", "")})
            _apply(chosen, state)

        ok = bool(plan)
        metrics = PlanMetrics(
            method=self.name,
            model=self.model,
            backend=self.provider,
            latency_s=time.perf_counter() - t0,
            llm_calls=llm_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            num_steps=len(plan),
            parse_ok=ok,
            notes=notes,
            extra={"saycan_steps": len(plan), "saycan_calls": llm_calls},
        )
        print(f"[{self.name}] {len(plan)} steps, {llm_calls} calls in "
              f"{metrics.latency_s:.1f}s" + (f" ({notes})" if notes else ""))
        return plan, metrics

    def replan(self, task, completed, failed_step, failure_reason,
               obs, visible_objects):
        """Resume greedy decoding from the state AFTER the completed prefix.

        We re-simulate the completed steps to reconstruct the symbolic state,
        then continue SayCan decoding for the remaining suffix only.
        """
        t0 = time.perf_counter()
        state = SymbolicState()
        if visible_objects:
            state.visible.update(visible_objects)
        # Replay the completed prefix to recover state (apply unconditionally —
        # these steps actually executed in the world).
        for s in completed:
            _apply(s, state)

        plan: list[dict] = []
        llm_calls = 0
        in_tok = out_tok = 0
        notes = f"replan_after:{failed_step.get('action')}"

        history = list(completed)
        for _step in range(self.max_steps):
            cands = self._candidates(state, visible_objects)
            if not cands:
                break
            try:
                scores, done, it, ot = self._score_candidates(
                    task, obs, history, state, cands)
                in_tok += it
                out_tok += ot
                llm_calls += 1
            except Exception as e:
                notes = f"score_error:{type(e).__name__}"
                break
            if done:
                break

            best_i, best_val = -1, -1.0
            for i, c in enumerate(cands):
                aff = self._affordance(c, state)
                val = scores.get(i, 0.3)
                combined = val * aff
                if combined > best_val:
                    best_val, best_i = combined, i
            if best_i < 0:
                break
            chosen = cands[best_i]
            if self._affordance(chosen, state) <= _INFEASIBLE_VALUE:
                break
            step = {"action": chosen["action"], "object": chosen.get("object", "")}
            plan.append(step)
            history.append(step)
            _apply(chosen, state)

        metrics = PlanMetrics(
            method=self.name,
            model=self.model,
            backend=self.provider,
            latency_s=time.perf_counter() - t0,
            llm_calls=llm_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            num_steps=len(plan),
            parse_ok=bool(plan),
            notes=notes,
        )
        return plan, metrics


__all__ = ["SayCanPlanner"]
