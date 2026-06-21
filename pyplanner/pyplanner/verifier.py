# pyplanner/verifier.py
# ─────────────────────────────────────────────────────────────────────
# Symbolic precondition verifier for SAGE planner.
#
# A lightweight, rule-based, LLM-free checker that decides — given the
# robot's symbolic state — whether the next action is admissible.
# It mirrors the semantics documented in pyplanner/base.STEP_SCHEMA:
#
#     MoveTo  <room|furniture>  → sets state.arrived
#     Find    <object>          → sets state.found (requires arrived)
#     Pick                      → requires found; sets state.holding
#     Place   <receptacle>      → requires holding AND arrived==receptacle
#     Open / Close / TurnOn / TurnOff → requires found; toggles state.opened
#     PutIn   <container>       → requires holding AND found(container)
#     Wash / Sit / LieOn / Serve / Wait → soft preconditions
#
# Why rule-based and not LLM:
#   - O(1) per step, zero token cost
#   - Deterministic — same plan always gets the same verdict
#   - Used both as a gate BEFORE execution (catches hallucinations) and
#     as a feedback signal to drive a single LLM refinement pass.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from pyplanner.base import ROBOT_ACTIONS


# Subsets of actions sharing the same precondition pattern.
_INTERACT_ACTIONS  = {"Pick", "Open", "Close", "TurnOn", "TurnOff", "Wash"}
_PLACE_ACTIONS     = {"Place", "PutIn"}
_NAV_ACTIONS       = {"MoveTo"}
_FIND_ACTIONS      = {"Find"}
_BODY_ACTIONS      = {"Sit", "LieOn"}
_FREE_ACTIONS      = {"Serve", "Wait"}
# Cross-domain object-treatment actions (ALFWorld heat/cool/clean), only present
# in ROBOT_ACTIONS when opted in via PYPLANNER_EXTRA_ACTIONS. They act on the held
# object (precondition: holding) and leave it in hand.
_TREAT_ACTIONS     = {"Heat", "Cool", "Clean"}

# Common containers that gate access to objects inside them.
DEFAULT_CONTAINERS = {
    "Fridge", "Cabinet", "Drawer", "Microwave", "Oven",
    "Safe", "Box", "GarbageCan",
}


# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────
@dataclass
class SymbolicState:
    """Tracks robot symbolic state across plan simulation.

    Fields mirror the implicit variables documented in STEP_SCHEMA:
        arrived  : last MoveTo destination (room or furniture)
        found    : last Find target (interactable handle)
        holding  : object currently in the gripper, or None
        opened   : set of currently-opened containers
        on       : set of currently-turned-on appliances
    """
    arrived: str | None       = None
    found:   str | None       = None
    holding: str | None       = None
    opened:  set[str]         = field(default_factory=set)
    on:      set[str]         = field(default_factory=set)
    visible: set[str]         = field(default_factory=set)

    def copy(self) -> "SymbolicState":
        return SymbolicState(
            arrived=self.arrived,
            found=self.found,
            holding=self.holding,
            opened=set(self.opened),
            on=set(self.on),
            visible=set(self.visible),
        )

    def as_text(self) -> str:
        """Human-readable snapshot for prompts."""
        return (
            f"arrived: {self.arrived or '∅'} | "
            f"found: {self.found or '∅'} | "
            f"holding: {self.holding or '∅'} | "
            f"opened: {sorted(self.opened) or '∅'} | "
            f"on: {sorted(self.on) or '∅'}"
        )


@dataclass
class StepViolation:
    """One precondition violation report."""
    index:    int
    step:     dict
    code:     str    # short machine-readable tag
    reason:   str    # human-readable explanation
    severity: str = "error"   # "error" or "warning"


@dataclass
class VerifierReport:
    """Aggregate result of simulating a plan against SymbolicState."""
    ok:          bool
    violations:  list[StepViolation] = field(default_factory=list)
    final_state: SymbolicState | None = None
    num_steps:   int                  = 0

    def first_error_index(self) -> int | None:
        for v in self.violations:
            if v.severity == "error":
                return v.index
        return None

    def summary(self, limit: int = 5) -> str:
        if self.ok:
            return f"OK ({self.num_steps} steps verified)"
        head = self.violations[:limit]
        body = "\n".join(
            f"  step {v.index+1}: [{v.code}] {v.reason} "
            f"({v.step.get('action')} {v.step.get('object','')})".rstrip()
            for v in head
        )
        more = f"\n  ... +{len(self.violations) - limit} more" if len(self.violations) > limit else ""
        return f"FAIL ({len(self.violations)} violation(s)):\n{body}{more}"


# ─────────────────────────────────────────────────────────────────────
# Step normalization
# ─────────────────────────────────────────────────────────────────────
# Legacy aliases that appear in eval_dataset_gt.json and external sources.
_ALIAS_MAP = {
    "navigate": "MoveTo",
    "moveto":   "MoveTo",
    "goto":     "MoveTo",
    "grab":     "Pick",
    "pickup":   "Pick",
    "pick":     "Pick",
    "place":    "Place",
    "put":      "Place",
    "putin":    "PutIn",
    "open":     "Open",
    "close":    "Close",
    "turnon":   "TurnOn",
    "turnoff":  "TurnOff",
    "wash":     "Wash",
    "find":     "Find",
    "search":   "Find",
    "locate":   "Find",
    "sit":      "Sit",
    "lieon":    "LieOn",
    "serve":    "Serve",
    "wait":     "Wait",
}


def normalize_action(action: str) -> str:
    """Map legacy / dataset action names to canonical ROBOT_ACTIONS."""
    if not action:
        return ""
    key = action.replace(" ", "").replace("_", "").lower()
    return _ALIAS_MAP.get(key, action)


def normalize_step(step: dict) -> dict:
    """Return a copy of step with action normalized."""
    out = dict(step)
    out["action"] = normalize_action(step.get("action", ""))
    return out


def normalize_plan(steps: Iterable[dict]) -> list[dict]:
    return [normalize_step(s) for s in steps]


# ─────────────────────────────────────────────────────────────────────
# Transition function
# ─────────────────────────────────────────────────────────────────────
def _apply(step: dict, state: SymbolicState) -> None:
    """Mutate `state` as if `step` had been executed successfully.

    Called only AFTER verify_step has approved the step.
    """
    a = step.get("action", "")
    o = step.get("object", "") or ""

    if a == "MoveTo":
        state.arrived = o
        # Moving away from an interactable resets 'found' unless the new
        # destination IS the same handle (rare, e.g. MoveTo Fridge twice).
        if state.found and state.found != o:
            state.found = None

    elif a == "Find":
        state.found = o

    elif a == "Pick":
        # Object field is documented as ignored at runtime; default to last found.
        held = o or state.found
        if held:
            state.holding = held
        state.found = None  # the object has left the scene

    elif a in _PLACE_ACTIONS:
        state.holding = None
        # state.arrived stays; robot is still at the receptacle.

    elif a == "Open":
        if o:
            state.opened.add(o)

    elif a == "Close":
        state.opened.discard(o)

    elif a == "TurnOn":
        if o:
            state.on.add(o)

    elif a == "TurnOff":
        state.on.discard(o)

    elif a in _TREAT_ACTIONS:
        # Heat/Cool/Clean treat the held object; it stays in hand, no tracked
        # state field changes (the 5-field state has no heated/cooled/clean flag).
        pass

    # Sit / LieOn / Serve / Wait / Wash have no symbolic side-effect we track.


# ─────────────────────────────────────────────────────────────────────
# Per-step verification
# ─────────────────────────────────────────────────────────────────────
def verify_step(
    step: dict,
    state: SymbolicState,
    containers: set[str] | None = None,
) -> tuple[bool, str | None, str | None]:
    """Check whether `step` is admissible in `state`.

    Returns (ok, code, reason).
        ok=True  → step is valid; reason is None
        ok=False → reason describes the violation, code is a short tag
    """
    containers = containers if containers is not None else DEFAULT_CONTAINERS

    a = step.get("action", "")
    o = step.get("object", "") or ""

    if a not in ROBOT_ACTIONS:
        return False, "unknown_action", f"action '{a}' not in ROBOT_ACTIONS"

    if a in _NAV_ACTIONS:
        if not o:
            return False, "empty_object", "MoveTo requires a destination object"
        return True, None, None

    if a in _FIND_ACTIONS:
        if not o:
            return False, "empty_object", "Find requires a target object"
        # Soft: warn if object not in visible list, but allow (LLM may know
        # about objects revealed by opening a container we just opened).
        return True, None, None

    if a == "Pick":
        if state.holding:
            return (
                False,
                "already_holding",
                f"cannot Pick while already holding '{state.holding}'",
            )
        if state.found is None:
            return (
                False,
                "no_find_before_pick",
                "Pick requires a preceding Find of the target object",
            )
        return True, None, None

    if a in _PLACE_ACTIONS:
        if not state.holding:
            return False, "place_without_holding", f"{a} requires holding an object"
        if not o:
            return False, "empty_object", f"{a} requires a receptacle/container"
        if a == "Place" and state.arrived != o:
            return (
                False,
                "place_wrong_location",
                f"Place requires MoveTo '{o}' first (currently at '{state.arrived}')",
            )
        if a == "PutIn" and o in containers and o not in state.opened:
            return (
                False,
                "putin_closed_container",
                f"PutIn requires container '{o}' to be Open",
            )
        return True, None, None

    if a in {"Open", "Close", "TurnOn", "TurnOff", "Wash"}:
        if state.found is None and state.arrived != o:
            return (
                False,
                "interact_without_find",
                f"{a} requires a preceding Find of '{o or 'target'}'",
            )
        if a == "Close" and o and o not in state.opened:
            return (
                False,
                "close_not_opened",
                f"Close '{o}' but it is not Open",
            )
        if a == "TurnOff" and o and o not in state.on:
            return (
                False,
                "turnoff_not_on",
                f"TurnOff '{o}' but it is not On",
            )
        return True, None, None

    if a in _TREAT_ACTIONS:
        # Heat/Cool/Clean require the object to be in hand (you cannot cool an
        # object you are not holding). This is the precondition the gate enforces.
        if not state.holding:
            return False, "treat_without_holding", f"{a} requires holding the object first"
        return True, None, None

    if a in _BODY_ACTIONS or a in _FREE_ACTIONS:
        return True, None, None

    # Unreachable — ROBOT_ACTIONS check above covers all cases.
    return True, None, None


# ─────────────────────────────────────────────────────────────────────
# Whole-plan simulation
# ─────────────────────────────────────────────────────────────────────
def simulate(
    plan:            list[dict],
    initial_state:   SymbolicState | None = None,
    visible_objects: list[str] | None     = None,
    containers:      set[str] | None      = None,
    stop_on_error:   bool                 = True,
) -> VerifierReport:
    """Replay a plan symbolically and report all precondition violations.

    Args:
        plan:            list of step dicts (will be normalized in place via copy)
        initial_state:   starting SymbolicState; new empty state if None
        visible_objects: optional list of objects assumed visible at start
        containers:      override default container set
        stop_on_error:   if True, halt at first error; else collect all errors
                          and keep simulating optimistically (do NOT apply the
                          rejected step's effects).

    Returns:
        VerifierReport with violations and the final SymbolicState.
    """
    state = initial_state.copy() if initial_state else SymbolicState()
    if visible_objects:
        state.visible.update(visible_objects)

    plan = normalize_plan(plan)
    violations: list[StepViolation] = []

    for i, step in enumerate(plan):
        ok, code, reason = verify_step(step, state, containers=containers)
        if not ok:
            violations.append(
                StepViolation(index=i, step=step, code=code or "violation",
                              reason=reason or "precondition failed")
            )
            if stop_on_error:
                break
            # If continuing, skip side-effects of the rejected step.
            continue
        _apply(step, state)

    return VerifierReport(
        ok=not violations,
        violations=violations,
        final_state=state,
        num_steps=len(plan),
    )


# ─────────────────────────────────────────────────────────────────────
# Feedback formatter for LLM refinement
# ─────────────────────────────────────────────────────────────────────
def format_violations_for_llm(
    report: VerifierReport,
    plan:   list[dict],
    limit:  int = 6,
) -> str:
    """Render violations as a compact, actionable instruction block.

    Designed to be appended to a refine prompt: tells the LLM exactly
    which steps failed, why, and what state the robot was in.
    """
    if report.ok:
        return "All preconditions satisfied."

    lines = ["The previous plan violates these preconditions:"]
    for v in report.violations[:limit]:
        a = v.step.get("action", "?")
        o = v.step.get("object", "")
        lines.append(f"  step {v.index+1} ({a} {o}): {v.reason}")
    if len(report.violations) > limit:
        lines.append(f"  ... and {len(report.violations) - limit} more.")

    lines.append(
        "\nFix the plan so each step satisfies its preconditions. "
        "Common fixes:"
    )
    lines.append("  - Insert MoveTo <container> + Open before Find for items inside containers.")
    lines.append("  - Insert Find <object> before Pick / Open / Close / TurnOn / TurnOff.")
    lines.append("  - Insert MoveTo <receptacle> before Place; the receptacle must match.")
    lines.append("  - Do not Pick while already holding something — Place the held item first.")
    return "\n".join(lines)


__all__ = [
    "SymbolicState",
    "StepViolation",
    "VerifierReport",
    "DEFAULT_CONTAINERS",
    "normalize_action",
    "normalize_step",
    "normalize_plan",
    "verify_step",
    "simulate",
    "format_violations_for_llm",
]
