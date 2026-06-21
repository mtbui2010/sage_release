# planners/cot.py
# Method 2 — Chain-of-Thought (CoT) Planning
#
# Two-phase in a SINGLE LLM call:
#   Phase A: "think step by step" free-form reasoning
#   Phase B: extract structured JSON from that reasoning
#
# Produces more logical plans at the cost of ~2x tokens.
# The reasoning trace is stored in metrics.extra["reasoning"].

import time

from pyplanner.base import (
    ACTIONS_STR, JSON_EXAMPLE, STEP_SCHEMA,
    BasePlanner, PlanMetrics, parse_steps,
)

SYSTEM_PROMPT = f"""You are a household assistant robot planner.
Your job is to turn a natural-language household task into a structured action plan.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

You MUST follow this two-part output format — no exceptions:

<reasoning>
Think through the task step by step:
1. What is the goal?
2. What objects are needed?
3. In what order must the actions happen?
4. What could go wrong?
</reasoning>

<plan>
{JSON_EXAMPLE}
</plan>

Do not output anything outside these two XML tags."""


def _extract(raw: str):
    """Split <reasoning> and <plan> blocks from LLM output."""
    import re
    reasoning = ""
    rm = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL)
    if rm:
        reasoning = rm.group(1).strip()

    plan_raw = raw
    pm = re.search(r"<plan>(.*?)</plan>", raw, re.DOTALL)
    if pm:
        plan_raw = pm.group(1).strip()

    return reasoning, plan_raw


class CoTPlanner(BasePlanner):
    name        = "CoT"
    description = (
        "Chain-of-Thought: LLM reasons freely first (<reasoning>), "
        "then outputs structured JSON (<plan>). Better logic, ~2x tokens."
    )

    def generate_plan(self, task, obs, visible_objects):
        t0 = time.perf_counter()
        user_msg = (
            self._context_str(task, obs, visible_objects)
            + "\n\nThink carefully, then generate the complete plan:"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        try:
            raw, in_tok, out_tok = self._chat(messages, temperature=0.3)
            reasoning, plan_raw  = _extract(raw)
            steps = parse_steps(plan_raw)
            ok    = bool(steps)
        except Exception as e:
            reasoning, steps, raw = str(e), [], ""
            in_tok, out_tok, ok   = 0, 0, False

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
            extra         = {"reasoning": reasoning},
        )
        print(f"[{self.name}] {len(steps)} steps in {metrics.latency_s:.1f}s | reasoning: {len(reasoning)} chars")
        return steps, metrics

    def replan(self, task, completed, failed_step, failure_reason, obs, visible_objects):
        t0 = time.perf_counter()
        user_msg = (
            self._replan_context(task, completed, failed_step, failure_reason, obs, visible_objects)
            + "\n\nThink carefully about the failure, then generate the remaining steps:"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        try:
            raw, in_tok, out_tok = self._chat(messages, temperature=0.3)
            reasoning, plan_raw  = _extract(raw)
            steps = parse_steps(plan_raw)
            ok    = bool(steps)
        except Exception as e:
            reasoning, steps = str(e), []
            in_tok, out_tok, ok = 0, 0, False

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
            extra         = {"reasoning": reasoning},
        )
        return steps, metrics
