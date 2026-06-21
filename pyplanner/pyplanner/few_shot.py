# planners/few_shot.py
# Method 4 — Few-Shot Chain-of-Thought
#
# Provides 3 hand-crafted example (task → reasoning → plan) pairs in the prompt
# so the model can mimic the pattern rather than invent it.
#
# Advantages over plain CoT:
# - The examples anchor object names and action vocabulary to known-good patterns
# - Less likely to generate hallucinated actions or wrong action order
# - Still a single LLM call — same latency profile as CoT
#
# The examples cover: kitchen (grab+use appliance), bathroom (multi-step routine),
# and living room (navigate+interact). This gives the model diverse context.

import re
import time
from pyplanner.few_shot_examples import FEW_SHOT_EXAMPLES

from pyplanner.base import (
    ACTIONS_STR, JSON_EXAMPLE, STEP_SCHEMA, ROBOT_ACTIONS,
    BasePlanner, PlanMetrics, parse_steps,
)


SYSTEM_PROMPT = f"""You are a household assistant robot planner.
Study the examples below, then generate a plan for the new task using the SAME format.

Available robot actions (use ONLY these names, exact spelling):
{ACTIONS_STR}

{STEP_SCHEMA}

{FEW_SHOT_EXAMPLES}

OUTPUT FORMAT — STRICT.
You MUST output two sections in this exact order, with these exact headers:

Reasoning:
<one short paragraph>

Plan:
<a single JSON object — see example>

The Plan section MUST be valid JSON of the form {{"steps": [...]}}.
Do NOT use markdown bold (**) on the headers.
Do NOT number the steps as English sentences.
Do NOT wrap the JSON in code fences.
Do NOT add any text after the JSON closes.

Example of the required output shape:
Reasoning:
Mug is on the shelf in the kitchen; need to grab it and start the coffee machine.

Plan:
{JSON_EXAMPLE}"""


# Markdown formatting that LLMs love to add around headers; stripped before regex.
_MD_HEADER_RE = re.compile(r"\*\*\s*(Reasoning|Plan)\s*:\s*\*\*", re.IGNORECASE)
# A numbered or bulleted step line, e.g. "1. Pick up the Apple", "- Find the mug".
_NL_STEP_RE   = re.compile(
    r"^\s*(?:\d+\.|[-*])\s+(.{4,200})$",
    re.MULTILINE,
)


def _strip_markdown(raw: str) -> str:
    """Normalise `**Plan:**` → `Plan:` so downstream regexes match."""
    return _MD_HEADER_RE.sub(lambda m: m.group(1) + ":", raw)


def _extract(raw: str):
    raw = _strip_markdown(raw)
    reasoning = ""
    rm = re.search(r"Reasoning:\s*(.*?)(?=Plan:|$)", raw, re.DOTALL | re.IGNORECASE)
    if rm:
        reasoning = rm.group(1).strip()
    plan_raw = raw
    # Prefer the JSON-object form; tolerate the LLM leading with prose under Plan:.
    pm = re.search(r"Plan:\s*([\s\S]*)", raw, re.IGNORECASE)
    if pm:
        plan_raw = pm.group(1).strip()
    return reasoning, plan_raw


# Common natural-language verb phrases the LLM emits → canonical actions.
_VERB_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(navigate|move|go|head|walk|approach)\b", re.I), "MoveTo"),
    (re.compile(r"\b(find|locate|search)\b",                    re.I), "Find"),
    (re.compile(r"\b(pick|grab|take|grasp)\b",                  re.I), "Pick"),
    (re.compile(r"\b(place|put down|set down|drop off)\b",      re.I), "Place"),
    (re.compile(r"\bput in(?:to)?\b",                           re.I), "PutIn"),
    (re.compile(r"\bopen\b",                                    re.I), "Open"),
    (re.compile(r"\bclose\b",                                   re.I), "Close"),
    (re.compile(r"\b(turn on|switch on|power on|start|activate)\b", re.I), "TurnOn"),
    (re.compile(r"\b(turn off|switch off|power off|stop)\b",    re.I), "TurnOff"),
    (re.compile(r"\bwash\b",                                    re.I), "Wash"),
    (re.compile(r"\b(sit|sit down)\b",                          re.I), "Sit"),
    (re.compile(r"\b(lie on|lay on|lie down)\b",                re.I), "LieOn"),
    (re.compile(r"\bserve\b",                                   re.I), "Serve"),
    (re.compile(r"\bwait\b",                                    re.I), "Wait"),
]


def _nl_lines_to_steps(plan_text: str, visible: list[str] | None = None) -> list[dict]:
    """Last-resort parser: turn numbered NL lines into canonical step dicts.

    Used only when the LLM ignored the JSON-only instruction and emitted a
    numbered or bulleted list. We extract a verb hint and the longest noun
    phrase that matches either the visible-objects set or a CamelCase token
    in the line.
    """
    # Match CamelCase visible objects against natural-language forms by also
    # storing a space-stripped lower-case key ("coffeemachine" → "CoffeeMachine"
    # can then match "coffee machine" once we strip spaces from the candidate line).
    vis_set = {v.lower(): v for v in (visible or [])}
    steps: list[dict] = []
    for m in _NL_STEP_RE.finditer(plan_text):
        line = m.group(1).strip(" .;:,")
        if not line:
            continue
        verb = ""
        for pat, canonical in _VERB_HINTS:
            if pat.search(line):
                verb = canonical
                break
        if not verb:
            continue
        obj = ""
        line_lower      = line.lower()
        line_no_spaces  = line_lower.replace(" ", "")
        # Prefer visible objects mentioned in the line — space-tolerant.
        for v_lower, v_orig in vis_set.items():
            if v_lower in line_lower or v_lower in line_no_spaces:
                obj = v_orig
                break
        # Otherwise grab the first CamelCase token (LLM convention).
        if not obj:
            cam = re.search(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", line)
            if cam:
                obj = cam.group(1)
        # Otherwise the first capitalised content word that is NOT a verb hint.
        if not obj:
            for cap in re.finditer(r"\b([A-Z][a-zA-Z]{2,})\b", line):
                tok = cap.group(1)
                if not any(p.search(tok) for p, _ in _VERB_HINTS):
                    obj = tok
                    break
        if verb == "Pick" and not obj:
            steps.append({"action": verb, "object": ""})
            continue
        if obj:
            steps.append({"action": verb, "object": obj})
    return steps


def _parse_with_fallback(plan_raw: str, visible: list[str] | None = None) -> list[dict]:
    """Try JSON parse first; if that yields nothing, fall back to NL extraction."""
    steps = parse_steps(plan_raw)
    if steps:
        return steps
    return _nl_lines_to_steps(plan_raw, visible=visible)


class FewShotPlanner(BasePlanner):
    name        = "Few-Shot CoT"
    description = (
        "3 hand-crafted examples anchor the model's output style. "
        "Single call like CoT but more consistent object/action naming."
    )

    def generate_plan(self, task, obs, visible_objects):
        t0 = time.perf_counter()
        user_msg = (
            self._context_str(task, obs, visible_objects)
            + "\n\nNow generate the Reasoning and Plan for this task:"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        try:
            raw, in_tok, out_tok = self._chat(messages, temperature=0.2)
            reasoning, plan_raw  = _extract(raw)
            steps = parse_steps(plan_raw)
            fallback_used = False
            if not steps:
                # Small LLMs sometimes ignore the JSON instruction and emit a
                # numbered natural-language list — recover what we can rather
                # than scoring the method at zero. See git blame for context.
                steps = _nl_lines_to_steps(plan_raw, visible=visible_objects)
                fallback_used = bool(steps)
            ok = bool(steps)
        except Exception as e:
            reasoning, steps = str(e), []
            in_tok, out_tok, ok = 0, 0, False
            fallback_used = False

        metrics = PlanMetrics(
            method        = self.name,
            model         = self.model,
            backend       = self.provider,
            latency_s     = time.perf_counter() - t0,
            llm_calls     = 1,
            input_tokens  = in_tok,
            output_tokens = out_tok,
            num_steps     = len(steps),
            parse_ok      = ok,
            extra         = {"reasoning": reasoning, "fallback_used": fallback_used},
        )
        tag = " (NL-fallback)" if fallback_used else ""
        print(f"[{self.name}]{tag} {len(steps)} steps in {metrics.latency_s:.1f}s")
        return steps, metrics

    def replan(self, task, completed, failed_step, failure_reason, obs, visible_objects):
        t0 = time.perf_counter()
        user_msg = (
            self._replan_context(task, completed, failed_step, failure_reason, obs, visible_objects)
            + "\n\nNow generate the Reasoning and remaining Plan:"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        try:
            raw, in_tok, out_tok = self._chat(messages, temperature=0.2)
            reasoning, plan_raw  = _extract(raw)
            steps = parse_steps(plan_raw)
            fallback_used = False
            if not steps:
                steps = _nl_lines_to_steps(plan_raw, visible=visible_objects)
                fallback_used = bool(steps)
            ok = bool(steps)
        except Exception as e:
            reasoning, steps = str(e), []
            in_tok, out_tok, ok = 0, 0, False
            fallback_used = False

        metrics = PlanMetrics(
            method        = self.name,
            model         = self.model,
            backend       = self.provider,
            latency_s     = time.perf_counter() - t0,
            llm_calls     = 1,
            input_tokens  = in_tok,
            output_tokens = out_tok,
            num_steps     = len(steps),
            parse_ok      = ok,
            extra         = {"reasoning": reasoning, "fallback_used": fallback_used},
        )
        return steps, metrics
