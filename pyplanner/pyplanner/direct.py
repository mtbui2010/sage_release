# planners/direct.py
# Method 1 — Direct Planning (baseline)
#
# Single LLM call: system prompt + context → JSON plan.
# Fastest, least tokens. Serves as the baseline for comparison.

import time

from pyplanner.base import (
    ACTIONS_STR, JSON_EXAMPLE, STEP_SCHEMA,
    BasePlanner, PlanMetrics, _approx_tokens, parse_steps,
)

SYSTEM_PROMPT = f"""You are a household assistant robot planner.
The user will describe what they want the robot to do in natural language.
Your job is to generate a step-by-step action plan for the robot.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

Return ONLY valid JSON with no markdown and no explanation:
{JSON_EXAMPLE}"""


class DirectPlanner(BasePlanner):
    name        = "Direct"
    description = (
        "Baseline: single LLM call, system prompt → JSON plan. "
        "Fastest but no explicit reasoning."
    )

    def generate_plan(self, task, obs, visible_objects):
        t0 = time.perf_counter()
        user_msg = self._context_str(task, obs, visible_objects) + "\n\nGenerate the complete step-by-step plan:"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        try:
            raw, in_tok, out_tok = self._chat(messages)
            steps = parse_steps(raw)
            ok    = bool(steps)
        except Exception as e:
            steps, raw, in_tok, out_tok, ok = [], str(e), 0, 0, False

        metrics = PlanMetrics(
            method       = self.name,
            model        = self.model,
            backend       = self.provider,
            latency_s    = time.perf_counter() - t0,
            llm_calls    = 1,
            input_tokens = in_tok,
            output_tokens= out_tok,
            num_steps    = len(steps),
            parse_ok     = ok,
        )
        print(f"[{self.name}] {len(steps)} steps in {metrics.latency_s:.1f}s")
        return steps, metrics

    def replan(self, task, completed, failed_step, failure_reason, obs, visible_objects):
        t0 = time.perf_counter()
        user_msg = self._replan_context(task, completed, failed_step, failure_reason, obs, visible_objects)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        try:
            raw, in_tok, out_tok = self._chat(messages)
            steps = parse_steps(raw)
            ok    = bool(steps)
        except Exception as e:
            steps, raw, in_tok, out_tok, ok = [], str(e), 0, 0, False

        metrics = PlanMetrics(
            method       = self.name,
            model        = self.model,
            backend       = self.provider,
            latency_s    = time.perf_counter() - t0,
            llm_calls    = 1,
            input_tokens = in_tok,
            output_tokens= out_tok,
            num_steps    = len(steps),
            parse_ok     = ok,
        )
        return steps, metrics
