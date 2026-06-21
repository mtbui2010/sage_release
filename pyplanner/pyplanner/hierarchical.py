# planners/hierarchical.py
# Method 4 — Hierarchical Planning
#
# Two-level decomposition:
#   Level 1 (High-level): LLM breaks task into 2–5 sub-goals (natural language)
#   Level 2 (Low-level):  For each sub-goal, LLM expands into concrete action steps
#
# Benefits:
# - Each LLM call is simpler → fewer hallucinations
# - Sub-goals provide natural checkpoints for partial replanning
# - The high-level plan is stored in metrics for inspection
#
# Costs: 1 + N_subgoals LLM calls.

import json
import re
import time

from pyplanner.base import (
    ACTIONS_STR, JSON_EXAMPLE, STEP_SCHEMA,
    BasePlanner, PlanMetrics, parse_steps,
)

HI_SYSTEM = """You are a household task planner (high-level).
Break the given task into 2–5 ordered sub-goals (natural language).
Each sub-goal should be a distinct phase of the task.

Return ONLY valid JSON — no markdown, no explanation:
{"subgoals": ["sub-goal 1", "sub-goal 2", "sub-goal 3"]}"""

LO_SYSTEM = f"""You are a household assistant robot (low-level executor).
You will receive ONE sub-goal and must expand it into concrete robot action steps.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

Return ONLY valid JSON:
{JSON_EXAMPLE}"""


def _parse_subgoals(raw: str) -> list[str]:
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return [str(s) for s in data.get("subgoals", [])]
        if isinstance(data, list):
            return [str(s) for s in data]
    except json.JSONDecodeError:
        pass
    # fallback: extract quoted strings
    return re.findall(r'"([^"]{5,})"', raw)


class HierarchicalPlanner(BasePlanner):
    name        = "Hierarchical"
    description = (
        "Two-level: LLM 1 decomposes into sub-goals, "
        "LLM 2 expands each sub-goal into actions. "
        "Better structure, 1+N calls."
    )

    def _expand(
        self,
        subgoal: str,
        task: str,
        obs: str,
        visible_objects: list[str],
    ) -> tuple[list[dict], int, int]:
        obj_str = ", ".join(visible_objects[:30]) if visible_objects else "none"
        user_msg = (
            f"Overall task: {task}\n"
            f"Current sub-goal: {subgoal}\n\n"
            f"Scene observation:\n{obs}\n\n"
            f"Visible objects: {obj_str}\n\n"
            "Expand this sub-goal into action steps:"
        )
        raw, in_tok, out_tok = self._chat(
            [{"role": "system", "content": LO_SYSTEM},
             {"role": "user",   "content": user_msg}],
            temperature=0.2,
        )
        return parse_steps(raw), in_tok, out_tok

    def generate_plan(self, task, obs, visible_objects):
        t0          = time.perf_counter()
        llm_calls   = 0
        total_in    = 0
        total_out   = 0
        all_steps   = []
        subgoals    = []

        obj_str = ", ".join(visible_objects[:30]) if visible_objects else "none"

        # ── Level 1: decompose ──
        try:
            hi_msg = (
                f"Task: {task}\n"
                f"Scene observation: {obs}\n"
                f"Visible objects: {obj_str}\n\n"
                "Break this into ordered sub-goals:"
            )
            raw_hi, in1, out1 = self._chat(
                [{"role": "system", "content": HI_SYSTEM},
                 {"role": "user",   "content": hi_msg}],
                temperature=0.3,
            )
            subgoals   = _parse_subgoals(raw_hi)
            llm_calls += 1
            total_in  += in1
            total_out += out1
        except Exception as e:
            subgoals = [task]  # fallback: treat full task as single sub-goal
            print(f"[{self.name}] Hi-level error: {e}, falling back to 1 sub-goal")

        # ── Level 2: expand each sub-goal ──
        for sg in subgoals:
            try:
                steps, in2, out2 = self._expand(sg, task, obs, visible_objects)
                all_steps  += steps
                llm_calls  += 1
                total_in   += in2
                total_out  += out2
            except Exception as e:
                print(f"[{self.name}] Lo-level error for '{sg}': {e}")

        # Deduplicate consecutive identical Navigate steps
        deduped = []
        for s in all_steps:
            if (deduped and s["action"] == "MoveTo"
                    and deduped[-1]["action"] == "MoveTo"
                    and deduped[-1]["object"] == s["object"]):
                continue
            deduped.append(s)

        metrics = PlanMetrics(
            method        = self.name,
            model         = self.model,
            backend       = self.provider,
            latency_s     = time.perf_counter() - t0,
            llm_calls     = llm_calls,
            input_tokens  = total_in,
            output_tokens = total_out,
            num_steps     = len(deduped),
            parse_ok      = bool(deduped),
            extra         = {"subgoals": subgoals},
        )
        print(f"[{self.name}] {len(subgoals)} sub-goals → {len(deduped)} steps in {metrics.latency_s:.1f}s")
        return deduped, metrics

    def replan(self, task, completed, failed_step, failure_reason, obs, visible_objects):
        # For replan: only re-decompose the remaining task
        completed_desc = ", ".join(
            f"{s.get('action')} {s.get('object')}" for s in completed
        ) or "nothing yet"
        remaining_task = (
            f"{task}. Already done: {completed_desc}. "
            f"Last action failed ({failed_step.get('action')} {failed_step.get('object')}): {failure_reason}. "
            "Generate only the remaining steps."
        )
        steps, metrics = self.generate_plan(remaining_task, obs, visible_objects)
        metrics.method = self.name
        return steps, metrics
