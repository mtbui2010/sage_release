# planners/self_refine.py
# Method 5 — Self-Refine Planning
#
# Iterative improvement loop:
#   1. Generate initial plan (same as Direct)
#   2. Critique: LLM reviews the plan and lists issues
#   3. Refine: LLM rewrites plan addressing the critique
#   4. Repeat 2–3 for `max_iterations` rounds
#
# Stops early if the critique says "no issues" or plan is stable.
# Produces highest-quality plans at the cost of 1 + 2×N calls.
#
# Reference: Madaan et al. "Self-Refine: Iterative Refinement with Self-Feedback" (2023)

import re
import time

from pyplanner.base import (
    ACTIONS_STR, JSON_EXAMPLE, STEP_SCHEMA,
    BasePlanner, PlanMetrics, parse_steps,
)

PLAN_SYSTEM = f"""You are a household assistant robot planner.
Generate a step-by-step action plan for the robot.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

Return ONLY valid JSON:
{JSON_EXAMPLE}"""

CRITIQUE_SYSTEM = f"""You are a robot plan reviewer.
Given a robot action plan, identify problems such as:
- Missing Navigate/Find before interacting with objects
- Wrong action order (e.g. using object before picking it up)
- Redundant or missing steps
- Wrong object names
- Steps that are impossible given visible objects

Available robot actions: {', '.join(ACTIONS_STR.split())}

If the plan is correct, respond with exactly: NO_ISSUES
Otherwise list the problems concisely (2–5 bullet points)."""

REFINE_SYSTEM = f"""You are a household assistant robot planner.
You will receive an original plan and a critique. Fix the issues and return an improved plan.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

Return ONLY valid JSON:
{JSON_EXAMPLE}"""


def _is_no_issues(critique: str) -> bool:
    return "NO_ISSUES" in critique.upper()


def _steps_equal(a: list[dict], b: list[dict]) -> bool:
    if len(a) != len(b):
        return False
    return all(
        x.get("action") == y.get("action") and x.get("object") == y.get("object")
        for x, y in zip(a, b)
    )


def _steps_to_text(steps: list[dict]) -> str:
    import json as _json
    return _json.dumps({"steps": steps}, indent=2)


class SelfRefinePlanner(BasePlanner):
    name        = "Self-Refine"
    description = (
        "Generate → Critique → Refine loop (default 2 iterations). "
        "Highest quality plans, 1+2N LLM calls."
    )

    def __init__(self, host: str, model: str, provider: str = "ollama", api_key: str = "", max_iterations: int = 2, **kwargs):
        super().__init__(host=host, model=model, provider=provider, api_key=api_key)
        self.max_iterations = max_iterations

    def _initial_plan(self, task, obs, visible_objects) -> tuple[list[dict], str, int, int]:
        obj_str = ", ".join(visible_objects[:30]) if visible_objects else "none"
        user_msg = (
            f"User request: {task}\n\n"
            f"Scene observation:\n{obs}\n\n"
            f"Visible objects: {obj_str}\n\n"
            "Generate the complete step-by-step plan:"
        )
        raw, in_tok, out_tok = self._chat(
            [{"role": "system", "content": PLAN_SYSTEM},
             {"role": "user",   "content": user_msg}],
            temperature=0.2,
        )
        return parse_steps(raw), raw, in_tok, out_tok

    def _critique(self, task, steps, obs, visible_objects) -> tuple[str, int, int]:
        obj_str = ", ".join(visible_objects[:30]) if visible_objects else "none"
        user_msg = (
            f"Task: {task}\n"
            f"Visible objects: {obj_str}\n\n"
            f"Plan to review:\n{_steps_to_text(steps)}\n\n"
            "Review the plan and list issues (or say NO_ISSUES):"
        )
        raw, in_tok, out_tok = self._chat(
            [{"role": "system", "content": CRITIQUE_SYSTEM},
             {"role": "user",   "content": user_msg}],
            temperature=0.1,
        )
        return raw.strip(), in_tok, out_tok

    def _refine(self, task, steps, critique, obs, visible_objects) -> tuple[list[dict], int, int]:
        obj_str = ", ".join(visible_objects[:30]) if visible_objects else "none"
        user_msg = (
            f"Task: {task}\n"
            f"Visible objects: {obj_str}\n\n"
            f"Original plan:\n{_steps_to_text(steps)}\n\n"
            f"Critique:\n{critique}\n\n"
            "Fix all issues and return an improved plan:"
        )
        raw, in_tok, out_tok = self._chat(
            [{"role": "system", "content": REFINE_SYSTEM},
             {"role": "user",   "content": user_msg}],
            temperature=0.2,
        )
        return parse_steps(raw), in_tok, out_tok

    def generate_plan(self, task, obs, visible_objects):
        t0         = time.perf_counter()
        llm_calls  = 0
        total_in   = 0
        total_out  = 0
        critiques  = []
        iterations = 0

        # ── Initial plan ──
        try:
            steps, _, in_tok, out_tok = self._initial_plan(task, obs, visible_objects)
            llm_calls += 1
            total_in  += in_tok
            total_out += out_tok
        except Exception as e:
            metrics = PlanMetrics(
                method=self.name, model=self.model,
            backend       = self.provider,
                latency_s=time.perf_counter() - t0,
                llm_calls=1, parse_ok=False,
                extra={"error": str(e)},
            )
            return [], metrics

        # ── Refine loop ──
        for i in range(self.max_iterations):
            if not steps:
                break
            iterations += 1

            try:
                critique, in2, out2 = self._critique(task, steps, obs, visible_objects)
                llm_calls += 1
                total_in  += in2
                total_out += out2
                critiques.append(critique)

                if _is_no_issues(critique):
                    break  # early stop

                prev_steps = steps
                steps, in3, out3 = self._refine(task, steps, critique, obs, visible_objects)
                llm_calls += 1
                total_in  += in3
                total_out += out3

                if _steps_equal(prev_steps, steps):
                    break  # stable — no point continuing
            except Exception as e:
                critiques.append(f"[Error in iter {i+1}] {e}")
                break

        metrics = PlanMetrics(
            method        = self.name,
            model         = self.model,
            backend       = self.provider,
            latency_s     = time.perf_counter() - t0,
            llm_calls     = llm_calls,
            input_tokens  = total_in,
            output_tokens = total_out,
            num_steps     = len(steps),
            parse_ok      = bool(steps),
            notes         = f"{iterations} refine iterations",
            extra         = {"critiques": critiques, "iterations": iterations},
        )
        print(f"[{self.name}] {len(steps)} steps, {iterations} refinements, {metrics.latency_s:.1f}s")
        return steps, metrics

    def replan(self, task, completed, failed_step, failure_reason, obs, visible_objects):
        replan_task = (
            f"{task}. "
            f"Previous step failed — {failed_step.get('action')} {failed_step.get('object')}: {failure_reason}. "
            "Generate only the remaining steps, fixing the failure."
        )
        steps, metrics = self.generate_plan(replan_task, obs, visible_objects)
        return steps, metrics
