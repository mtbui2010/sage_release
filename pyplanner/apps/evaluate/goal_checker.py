"""
goal_checker.py
===============
Verify whether an executed plan actually achieved the user's intended goal.

The problem
-----------
Step success ≠ task success:
  - All steps return success=True  →  robot did the actions
  - But was the GOAL achieved?     →  unknown without verification

Example:
  Prompt: "make coffee"
  Steps:  Navigate Mug ✓, Grab Mug ✓, TurnOn Microwave ✓
  Result: 3/3 steps OK  —  but coffee was never brewed → TASK FAILED

Two-layer verification
----------------------
Layer 1  GoalCondition  — per-task state assertions checked via get_objects().
         Fast, deterministic. Checks things like:
           - Is the mug placed on/near the coffee machine?
           - Is the coffee machine on?
           - Is the fridge open?

Layer 2  LLM judge      — if GoalCondition is ambiguous or not defined.
         Sends final observation + executed steps to LLM and asks:
           "Did the robot achieve the task goal?"

Confidence
----------
Each check returns a GoalVerdict with:
  success    : bool
  confidence : float  0.0–1.0
  method     : "goal_condition" | "llm_judge" | "combined"
  reason     : short explanation
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import pyplanner
    from pyplanner import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_BACKEND
    from pyplanner.base import LLMBackend
except ImportError:
    sys.path.insert(0, os.path.join(_HERE, "..", "pyplanner"))
    import pyplanner
    from pyplanner import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_BACKEND
    from pyplanner.base import LLMBackend


# ═════════════════════════════════════════════════════════════════════
# Verdict dataclass
# ═════════════════════════════════════════════════════════════════════

@dataclass
class GoalVerdict:
    success:    bool
    confidence: float   # 0.0 = no idea, 1.0 = certain
    method:     str     # "goal_condition" | "llm_judge" | "combined" | "no_check"
    reason:     str     # human-readable explanation
    details:    dict    # extra data (state snapshot, llm response, etc.)

    def to_dict(self) -> dict:
        return {
            "success":    self.success,
            "confidence": round(self.confidence, 3),
            "method":     self.method,
            "reason":     self.reason,
        }


# ═════════════════════════════════════════════════════════════════════
# Layer 1: GoalCondition — per-task state assertions
# ═════════════════════════════════════════════════════════════════════

# A GoalFn takes the object_map snapshot and executed_steps list,
# returns (success: bool, confidence: float, reason: str)
GoalFn = Callable[[dict, list[dict]], tuple[bool, float, str]]


def _obj(name: str, om: dict) -> dict | None:
    """Get object metadata by partial name match."""
    name_l = name.lower().replace("_", "")
    for ot, meta in om.items():
        if name_l in ot.lower():
            return meta
    return None


def _is_on(obj_name: str, om: dict) -> bool:
    o = _obj(obj_name, om)
    return bool(o and o.get("isToggled"))


def _is_open(obj_name: str, om: dict) -> bool:
    o = _obj(obj_name, om)
    return bool(o and o.get("isOpen"))


def _was_grabbed(obj_name: str, steps: list[dict]) -> bool:
    """Check if object was grabbed in executed steps."""
    name_l = obj_name.lower().replace("_", "")
    return any(
        s.get("action") in ("Pick", "PickupObject")
        and name_l in s.get("object", "").lower()
        for s in steps
    )


def _placements_from_steps(steps: list[dict]) -> list[tuple[str, str]]:
    """Schema-aware (item, receptacle) placements, tracking the held item across
    Pick->Place. Handles BOTH conventions: (a) Place object=item, target=receptacle;
    (b) the canonical ROBOT_ACTIONS schema Place object=receptacle, item implicit
    (the most recently picked object). Only SUCCESSFUL executed steps are passed in,
    so a recorded Place means AI2-THOR actually performed the placement."""
    held = None
    out: list[tuple[str, str]] = []
    for s in steps:
        a = s.get("action", "")
        o = s.get("object", "").lower().replace("_", "")
        t = s.get("target", "").lower().replace("_", "")
        if a in ("Pick", "PickupObject", "Grab"):
            held = o
        elif a in ("Place", "PutIn", "Serve"):
            if t:                       # schema (a): explicit item + target
                out.append((o, t))
            elif held:                  # schema (b): held item placed onto object(=receptacle)
                out.append((held, o))
            else:                       # unknown item — record receptacle only
                out.append(("", o))
            held = None
    return out


def _was_placed_on(obj_name: str, target_name: str, steps: list[dict],
                   om: dict | None = None) -> bool:
    """True if `obj_name` was placed on/in `target_name`. Uses schema-aware step
    tracking (item carried across Pick->Place); optionally confirmed by final
    metadata (parentReceptacles) when an object_map is supplied."""
    obj_l    = obj_name.lower().replace("_", "")
    target_l = target_name.lower().replace("_", "")
    for item, recept in _placements_from_steps(steps):
        item_ok = (not obj_l) or (obj_l in item) or (item and item in obj_l)
        if item_ok and (target_l in recept or recept in target_l) and recept:
            return True
    # metadata confirmation: does an object of type obj sit in a receptacle of type target?
    if om:
        o = _obj(obj_name, om)
        if o:
            for pid in (o.get("parentReceptacles") or []):
                if target_l in str(pid).lower():
                    return True
    return False


def _was_turned_on(obj_name: str, steps: list[dict]) -> bool:
    name_l = obj_name.lower().replace("_", "")
    return any(
        s.get("action") == "TurnOn"
        and name_l in s.get("object", "").lower()
        for s in steps
    )


def _was_washed(obj_name: str, steps: list[dict]) -> bool:
    name_l = obj_name.lower().replace("_", "")
    return any(
        s.get("action") == "Wash"
        and name_l in s.get("object", "").lower()
        for s in steps
    )


# ── Goal condition definitions ────────────────────────────────────────
# Each entry: task_id → GoalFn
# GoalFn(object_map, executed_steps) → (success, confidence, reason)

def _goal_make_coffee(om, steps):
    mug_placed   = _was_placed_on("mug", "coffee", steps) or _was_placed_on("mug", "coffeemachine", steps)
    machine_on   = _was_turned_on("coffee", steps) or _is_on("coffeemachine", om)
    if mug_placed and machine_on:
        return True, 0.95, "Mug placed on coffee machine and machine turned on"
    if machine_on and not mug_placed:
        return False, 0.9, "Coffee machine on but mug not placed — coffee not brewed"
    if mug_placed and not machine_on:
        return False, 0.85, "Mug placed but coffee machine not turned on"
    return False, 0.7, "Neither mug placed nor machine turned on"

def _goal_boil_water(om, steps):
    pot_on_stove = _was_placed_on("pot", "stove", steps)
    stove_on     = _was_turned_on("stove", steps) or _is_on("stoveburner", om)
    water_added  = _was_placed_on("pot", "sink", steps) or any(
        s.get("action") == "TurnOn" and "faucet" in s.get("object","").lower() for s in steps
    )
    if pot_on_stove and stove_on:
        return True, 0.9, "Pot on stove and burner on"
    if stove_on and not pot_on_stove:
        return False, 0.9, "Stove on but pot not placed on it"
    return False, 0.75, "Stove not turned on"

def _goal_turn_on_stove(om, steps):
    stove_on = _was_turned_on("stove", steps) or _is_on("stoveburner", om)
    return (stove_on, 0.95, "Stove burner is on") if stove_on \
      else (False, 0.95, "Stove burner was not turned on")

def _goal_cook_egg(om, steps):
    egg_in_pan = _was_placed_on("egg", "pan", steps)
    stove_on   = _was_turned_on("stove", steps) or _is_on("stoveburner", om)
    if egg_in_pan and stove_on:
        return True, 0.9, "Egg in pan and stove on"
    if egg_in_pan:
        return False, 0.85, "Egg in pan but stove not on"
    return False, 0.85, "Egg not placed in pan"

def _goal_get_fridge(om, steps):
    fridge_opened = _is_open("fridge", om) or _was_grabbed("", steps)  # any grab after open
    item_grabbed  = any(s.get("action") == "Pick" for s in steps)
    if item_grabbed and any("fridge" in s.get("object","").lower() or _is_open("fridge", om) for s in steps):
        return True, 0.85, "Item retrieved from fridge"
    if _is_open("fridge", om) and item_grabbed:
        return True, 0.8, "Fridge opened and item grabbed"
    return False, 0.7, "No item grabbed from fridge"

def _goal_setup_dishes(om, steps):
    plate_placed = _was_placed_on("plate", "dining", steps) or _was_placed_on("plate", "table", steps)
    cup_placed   = _was_placed_on("cup", "dining", steps)   or _was_placed_on("cup", "table", steps)
    if plate_placed and cup_placed:
        return True, 0.95, "Plate and cup placed on dining table"
    if plate_placed:
        return False, 0.8, "Plate placed but cup missing"
    return False, 0.8, "No dishes placed on table"

def _goal_wash_apple(om, steps):
    washed = _was_washed("apple", steps)
    faucet = any(s.get("action") == "TurnOn" and "faucet" in s.get("object","").lower() for s in steps)
    if washed:
        return True, 0.95, "Apple washed at sink"
    if faucet and _was_grabbed("apple", steps):
        return True, 0.75, "Apple grabbed and faucet turned on (likely washed)"
    return False, 0.85, "Apple not washed"

def _goal_microwave(om, steps):
    item_in_mw = any(
        s.get("action") in ("Place","PutIn") and "microwave" in s.get("target","").lower()
        for s in steps
    )
    mw_on = _was_turned_on("microwave", steps) or _is_on("microwave", om)
    if item_in_mw and mw_on:
        return True, 0.95, "Food placed in microwave and microwave on"
    if mw_on and not item_in_mw:
        return False, 0.9, "Microwave on but food not placed inside"
    return False, 0.85, "Microwave not activated"

def _goal_watch_tv(om, steps):
    tv_on = _was_turned_on("tv", steps) or _was_turned_on("television", steps) or _is_on("television", om)
    sat   = any(s.get("action") == "Sit" for s in steps)
    if tv_on and sat:
        return True, 0.95, "TV on and agent sitting"
    if tv_on:
        return False, 0.8, "TV on but agent not sitting"
    return False, 0.85, "TV not turned on"

def _goal_read_book(om, steps):
    grabbed = _was_grabbed("book", steps)
    sat     = any(s.get("action") == "Sit" for s in steps)
    if grabbed and sat:
        return True, 0.9, "Book grabbed and agent sitting"
    if grabbed:
        return False, 0.75, "Book grabbed but agent not sitting"
    return False, 0.85, "Book not picked up"

def _goal_sleep(om, steps):
    light_off = any(s.get("action") == "TurnOff" for s in steps)
    lying     = any(s.get("action") == "LieOn"   for s in steps)
    if lying:
        return True, 0.9, "Agent lying on bed"
    if light_off:
        return False, 0.6, "Light off but agent not lying down"
    return False, 0.8, "Agent did not lie down"

def _goal_alarm_clock(om, steps):
    grabbed = _was_grabbed("alarm", steps)
    return (True, 0.9, "Alarm clock picked up") if grabbed \
      else (False, 0.9, "Alarm clock not interacted with")

def _goal_get_clothes(om, steps):
    opened  = _is_open("dresser", om) or any(s.get("action") == "Open" and "dresser" in s.get("object","").lower() for s in steps)
    grabbed = _was_grabbed("clothes", steps) or _was_grabbed("cloth", steps)
    if grabbed:
        return True, 0.9, "Clothes retrieved"
    if opened:
        return False, 0.75, "Dresser opened but clothes not taken"
    return False, 0.85, "Dresser not opened"

def _goal_brush_teeth(om, steps):
    grabbed  = _was_grabbed("toothbrush", steps)
    washed   = _was_washed("toothbrush", steps)
    faucet   = any(s.get("action") == "TurnOn" and "faucet" in s.get("object","").lower() for s in steps)
    if grabbed and (washed or faucet):
        return True, 0.9, "Toothbrush used at sink"
    if grabbed:
        return False, 0.75, "Toothbrush grabbed but sink not used"
    return False, 0.85, "Toothbrush not picked up"

def _goal_wash_hands(om, steps):
    faucet_on  = any(s.get("action") == "TurnOn"  and "faucet" in s.get("object","").lower() for s in steps)
    faucet_off = any(s.get("action") == "TurnOff" and "faucet" in s.get("object","").lower() for s in steps)
    soap       = _was_grabbed("soap", steps) or _was_washed("soap", steps) or _was_washed("hands", steps)
    if faucet_on and faucet_off:
        return True, 0.9, "Faucet on and off — hands likely washed"
    if faucet_on:
        return False, 0.7, "Faucet on but not turned off"
    return False, 0.85, "Faucet not used"

def _goal_shower(om, steps):
    shower_on = _was_turned_on("shower", steps) or _is_on("shower", om)
    return (True, 0.95, "Shower turned on") if shower_on \
      else (False, 0.95, "Shower not turned on")


# ── Registry: task_id → GoalFn ────────────────────────────────────────
GOAL_CONDITIONS: dict[str, GoalFn] = {
    "K01": _goal_make_coffee,
    "K02": _goal_boil_water,
    "K03": _goal_get_fridge,
    "K04": _goal_boil_water,
    "K05": _goal_cook_egg,
    "K06": _goal_wash_apple,
    "K07": _goal_microwave,
    "K08": _goal_setup_dishes,
    "K09": _goal_make_coffee,
    "L01": _goal_watch_tv,
    "L02": lambda om, s: (any(st.get("action")=="Sit" for st in s), 0.95, "Agent sat on sofa"),
    "L03": _goal_read_book,
    "L04": _goal_watch_tv,
    "L05": _goal_read_book,
    "L06": lambda om, s: (
        (any(st.get("action")=="TurnOn" and "lamp" in st.get("object","").lower() for st in s)
         and any(st.get("action")=="Sit" for st in s)),
        0.9, "Lamp on and sitting"
    ),
    "L07": lambda om, s: (
        any(st.get("action") in ("Place","PutIn") for st in s), 0.8, "Items placed"
    ),
    "L08": lambda om, s: (
        (any(st.get("action")=="TurnOn"  and "tv"   in st.get("object","").lower() for st in s)
         and any(st.get("action")=="TurnOff" and "lamp" in st.get("object","").lower() for st in s)
         and any(st.get("action")=="Sit" for st in s)),
        0.95, "TV on, lamp off, sitting"
    ),
    "B01": lambda om, s: (any(st.get("action")=="TurnOff" for st in s), 0.9, "Light turned off"),
    "B02": _goal_sleep,
    "B03": _goal_alarm_clock,
    "B04": _goal_sleep,
    "B05": _goal_alarm_clock,
    "B06": _goal_get_clothes,
    "B07": lambda om, s: (
        _goal_sleep(om, s)[0] and _goal_alarm_clock(om, s)[0], 0.9, "Alarm set and lying down"
    ),
    "B08": lambda om, s: (
        any(st.get("action") in ("Place","PutIn") and "bed" in st.get("target","").lower() for st in s),
        0.9, "Pillow placed on bed"
    ),
    "A01": _goal_shower,
    "A02": lambda om, s: (_was_grabbed("soap", s), 0.9, "Soap picked up"),
    "A03": lambda om, s: (any(st.get("action")=="TurnOn" for st in s), 0.9, "Light turned on"),
    "A04": _goal_wash_hands,
    "A05": _goal_brush_teeth,
    "A06": lambda om, s: (
        _was_placed_on("towel", "rack", s) or _was_placed_on("towel", "towelrack", s),
        0.9, "Towel placed on rack"
    ),
    "A07": lambda om, s: (
        _goal_wash_hands(om, s)[0] and _goal_brush_teeth(om, s)[0], 0.9, "Hands washed and teeth brushed"
    ),
    "A08": lambda om, s: (
        _goal_shower(om, s)[0] and any(
            st.get("action") in ("Place","PutIn") and "mat" in st.get("target","").lower() for st in s
        ), 0.9, "Shower on and towel placed near shower"
    ),
    "X01": _goal_microwave,
    "X02": lambda om, s: (
        _was_placed_on("book", "table", s) and any(
            st.get("action")=="TurnOn" and "lamp" in st.get("object","").lower() for st in s
        ), 0.9, "Book on table and lamp on"
    ),
    "X03": lambda om, s: (
        any(st.get("action")=="TurnOn" and "faucet" in st.get("object","").lower() for st in s)
        and _was_grabbed("cup", s), 0.9, "Cup grabbed and water running"
    ),
    "X04": lambda om, s: (
        any(st.get("action")=="Open"  and "window"  in st.get("object","").lower() for st in s)
        and any(st.get("action")=="Close" and "curtain" in st.get("object","").lower() for st in s),
        0.95, "Window opened and curtains closed"
    ),
    "X05": lambda om, s: (
        _was_placed_on("soap", "cabinet", s) or any(
            st.get("action")=="PutIn" and "soap" in st.get("object","").lower() for st in s
        ), 0.95, "Soap placed in cabinet"
    ),
}


# ═════════════════════════════════════════════════════════════════════
# Layer 2: LLM judge
# ═════════════════════════════════════════════════════════════════════

LLM_JUDGE_SYSTEM = """You are a robot task evaluator.
Given:
  1. The original user task (goal)
  2. The final environment observation after execution
  3. The list of steps that were successfully executed

Decide whether the robot achieved the goal.
Be strict: partial completion = failure.

Respond with ONLY valid JSON:
{"success": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}

Examples:
  Task: "make coffee"
  Steps: [Navigate Mug, Grab Mug, TurnOn CoffeeMachine] but mug not placed on machine
  → {"success": false, "confidence": 0.9, "reason": "Mug not placed on coffee machine before brewing"}

  Task: "wash hands"
  Steps: [Navigate Sink, TurnOn Faucet, Wash Soap, TurnOff Faucet]
  → {"success": true, "confidence": 0.95, "reason": "Faucet used and soap washed — hands washed"}"""


def llm_judge(
    task_desc:     str,
    final_obs:     str,
    executed_steps:list[dict],
    provider:      str = DEFAULT_BACKEND,
    model:         str = DEFAULT_MODEL,
    host:          str = DEFAULT_HOST,
    api_key:       str = "",
) -> GoalVerdict:
    """Ask LLM to judge whether the goal was achieved."""
    steps_text = "\n".join(
        f"  {i+1}. {s.get('action')} {s.get('object','')}"
        + (f" → {s['target']}" if s.get("target") else "")
        for i, s in enumerate(executed_steps)
    ) or "  (no steps executed)"

    user_msg = (
        f"Task: {task_desc}\n\n"
        f"Final environment observation:\n{final_obs}\n\n"
        f"Steps successfully executed:\n{steps_text}\n\n"
        f"Did the robot achieve the goal?"
    )

    try:
        backend = LLMBackend(provider=provider, model=model, host=host, api_key=api_key)
        content, _, _ = backend.chat(
            [{"role": "system", "content": LLM_JUDGE_SYSTEM},
             {"role": "user",   "content": user_msg}],
            temperature=0.1,
        )
        import re
        content_clean = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content_clean)
        return GoalVerdict(
            success    = bool(data.get("success", False)),
            confidence = float(data.get("confidence", 0.5)),
            method     = "llm_judge",
            reason     = str(data.get("reason", "")),
            details    = {"raw_response": content[:300]},
        )
    except Exception as e:
        return GoalVerdict(
            success    = False,
            confidence = 0.0,
            method     = "llm_judge",
            reason     = f"LLM judge error: {e}",
            details    = {"error": str(e)},
        )


# ═════════════════════════════════════════════════════════════════════
# Combined checker — Layer 1 first, Layer 2 as fallback
# ═════════════════════════════════════════════════════════════════════

def check_goal(
    task_id:        str,
    task_desc:      str,
    executed_steps: list[dict],
    final_obs:      str,
    object_map:     dict,
    # LLM judge config (used as fallback)
    use_llm_judge:  bool = True,
    provider:       str  = DEFAULT_BACKEND,
    model:          str  = DEFAULT_MODEL,
    host:           str  = DEFAULT_HOST,
    api_key:        str  = "",
    # Threshold: use LLM judge if goal_condition confidence < this
    llm_fallback_threshold: float = 0.8,
) -> GoalVerdict:
    """
    Main entry point. Checks goal in two layers:
      1. GoalCondition (fast, deterministic)
      2. LLM judge (fallback when condition is ambiguous or undefined)

    Args:
        task_id:        e.g. "K01"
        task_desc:      original user prompt
        executed_steps: steps that returned success=True in simulator
        final_obs:      environment observation after all steps
        object_map:     objectType → metadata from get_objects()
        use_llm_judge:  enable LLM fallback
        llm_fallback_threshold: use LLM if gc_confidence < this value

    Returns:
        GoalVerdict with success, confidence, method, reason
    """
    gc_fn = GOAL_CONDITIONS.get(task_id)

    # ── Layer 1: GoalCondition ─────────────────────────────────────
    if gc_fn is not None:
        try:
            success, confidence, reason = gc_fn(object_map, executed_steps)
            verdict = GoalVerdict(
                success    = success,
                confidence = confidence,
                method     = "goal_condition",
                reason     = reason,
                details    = {"task_id": task_id},
            )

            # High confidence → return directly
            if confidence >= llm_fallback_threshold:
                return verdict

            # Low confidence + LLM available → combine
            if use_llm_judge:
                llm_v = llm_judge(task_desc, final_obs, executed_steps,
                                   provider, model, host, api_key)
                # Weighted combination: gc weight by its confidence, llm by its confidence
                w_gc  = confidence
                w_llm = llm_v.confidence
                total = w_gc + w_llm
                if total > 0:
                    score = (w_gc * int(success) + w_llm * int(llm_v.success)) / total
                    combined_success = score >= 0.5
                    combined_conf    = (w_gc * confidence + w_llm * llm_v.confidence) / total
                else:
                    combined_success = success
                    combined_conf    = confidence

                return GoalVerdict(
                    success    = combined_success,
                    confidence = round(combined_conf, 3),
                    method     = "combined",
                    reason     = f"GC({confidence:.2f}): {reason}  |  LLM({llm_v.confidence:.2f}): {llm_v.reason}",
                    details    = {
                        "gc_success":    success,
                        "gc_confidence": confidence,
                        "llm_success":   llm_v.success,
                        "llm_confidence":llm_v.confidence,
                    },
                )

            return verdict

        except Exception as e:
            # GoalCondition crashed — fall through to LLM
            pass

    # ── Layer 2: LLM judge only (no GoalCondition defined) ─────────
    if use_llm_judge:
        return llm_judge(task_desc, final_obs, executed_steps,
                          provider, model, host, api_key)

    # ── No check available ──────────────────────────────────────────
    return GoalVerdict(
        success    = len(executed_steps) > 0,   # optimistic: if steps ran, assume ok
        confidence = 0.3,
        method     = "no_check",
        reason     = f"No GoalCondition defined for '{task_id}' and LLM judge disabled",
        details    = {},
    )


# ═════════════════════════════════════════════════════════════════════
# Convenience: batch check all samples in a dataset
# ═════════════════════════════════════════════════════════════════════

def verify_dataset(
    dataset_path: str,
    use_llm_judge: bool = False,
    provider: str = DEFAULT_BACKEND,
    model:    str = DEFAULT_MODEL,
    host:     str = DEFAULT_HOST,
    api_key:  str = "",
    verbose:  bool = False,
) -> list[dict]:
    """
    Load a dataset (eval_dataset_gt.json or similar) and add goal verdicts.
    Returns list of samples with added 'goal_verdict' field.
    """
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", data)

    results = []
    for s in samples:
        # Reconstruct object_map from visible_objects (best effort without live sim)
        om = {ot: {"objectType": ot, "isToggled": False, "isOpen": False}
              for ot in s.get("visible_objects", [])}

        verdict = check_goal(
            task_id        = s["task_id"],
            task_desc      = s["task_desc"],
            executed_steps = s.get("reference_steps", []),
            final_obs      = s.get("obs", ""),
            object_map     = om,
            use_llm_judge  = use_llm_judge,
            provider       = provider,
            model          = model,
            host           = host,
            api_key        = api_key,
        )

        if verbose:
            icon = "✅" if verdict.success else "❌"
            print(f"  {icon} [{s['task_id']}] {verdict.method}  "
                  f"conf={verdict.confidence:.2f}  {verdict.reason[:60]}")

        sample_out = dict(s)
        sample_out["goal_verdict"] = verdict.to_dict()
        results.append(sample_out)

    return results


# ═════════════════════════════════════════════════════════════════════
# Quick demo / CLI
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify goal completion for a dataset")
    parser.add_argument("--dataset",  default="eval_dataset_gt.json")
    parser.add_argument("--llm",      action="store_true", help="Enable LLM judge fallback")
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--host",     default=DEFAULT_HOST)
    parser.add_argument("--provider", default=DEFAULT_BACKEND)
    parser.add_argument("--api-key",  default="")
    parser.add_argument("--out",      default="eval_dataset_verified.json")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    print(f"\n  Goal Verification")
    print(f"  Dataset  : {args.dataset}")
    print(f"  LLM judge: {'on' if args.llm else 'off (GoalCondition only)'}")

    results = verify_dataset(
        dataset_path  = args.dataset,
        use_llm_judge = args.llm,
        provider      = args.provider,
        model         = args.model,
        host          = args.host,
        api_key       = args.api_key,
        verbose       = True,
    )

    success_rate = sum(1 for r in results if r["goal_verdict"]["success"]) / len(results)
    methods      = {}
    for r in results:
        m = r["goal_verdict"]["method"]
        methods[m] = methods.get(m, 0) + 1

    print(f"\n  Results: {len(results)} tasks  |  goal_success_rate={success_rate:.0%}")
    print(f"  Methods: {methods}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"version":"1.0","samples":results}, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {args.out}\n")