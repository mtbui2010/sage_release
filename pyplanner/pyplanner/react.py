# planners/react.py
# Method 3 — ReAct (Reasoning + Acting) Planning
#
# Interleaved loop: Thought → Action proposal → simulated Observation → next Thought
# Each iteration proposes ONE action step. The loop runs until the LLM signals DONE
# or max_steps is reached.
#
# This mirrors how ReAct is used in robotics: the planner doesn't commit to a full
# plan up front, but decides the next action given current state.
# Here we simulate the "observation" from the LLM itself (no real env step).
#
# Costs: N × LLM calls (one per step). Slower but most grounded.

import re
import time

from pyplanner.base import (
    ACTIONS_STR, BasePlanner, PlanMetrics, _approx_tokens,
)

SYSTEM_PROMPT = f"""You are a household assistant robot that reasons and acts one step at a time.

Available robot actions:
{ACTIONS_STR}

At each turn you will receive the task, completed steps so far, and current scene observation.
You must output EXACTLY one of:

  Thought: <one sentence reasoning about what to do next>
  Action: <action_name> | <object> | <target_or_empty> | <reason>

OR, when the task is complete:

  Thought: The task is complete.
  Done

Rules:
- Action fields are pipe-separated: action | object | target | reason
- target is empty string if not applicable
- Use exact action names from the list above
- Always Navigate or Find before interacting
- Output NOTHING else — no JSON, no markdown"""

EXTRACT_RE = re.compile(
    r"Action:\s*(\w+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]*)\s*\|\s*(.+)",
    re.IGNORECASE,
)


def _parse_action(raw: str) -> dict | None:
    m = EXTRACT_RE.search(raw)
    if not m:
        return None
    return {
        "action": m.group(1).strip(),
        "object": m.group(2).strip(),
        "target": m.group(3).strip(),
        "reason": m.group(4).strip(),
    }


def _is_done(raw: str) -> bool:
    return bool(re.search(r"\bDone\b", raw, re.IGNORECASE))


class ReActPlanner(BasePlanner):
    name        = "ReAct"
    description = (
        "ReAct: generates ONE action per LLM call with explicit Thought. "
        "Most grounded but slowest (N calls for N steps)."
    )

    def __init__(self, host: str, model: str, provider: str = "ollama", api_key: str = "", max_steps: int = 15, **kwargs):
        super().__init__(host=host, model=model, provider=provider, api_key=api_key)
        self.max_steps = max_steps

    def _run_loop(
        self,
        task: str,
        initial_obs: str,
        visible_objects: list[str],
        seed_steps: list[dict] = [],
    ) -> tuple[list[dict], PlanMetrics]:
        t0          = time.perf_counter()
        steps       = list(seed_steps)
        llm_calls   = 0
        total_in    = 0
        total_out   = 0
        thoughts    = []

        obj_str = ", ".join(visible_objects[:30]) if visible_objects else "none"

        for _ in range(self.max_steps - len(seed_steps)):
            # Build conversation history
            completed_str = "\n".join(
                f"  {i+1}. {s['action']} {s['object']}"
                + (f" → {s['target']}" if s.get("target") else "")
                for i, s in enumerate(steps)
            ) or "  (none yet)"

            user_msg = (
                f"Task: {task}\n\n"
                f"Completed steps so far:\n{completed_str}\n\n"
                f"Current scene observation:\n{initial_obs}\n\n"
                f"Visible objects: {obj_str}\n\n"
                "What is your next Thought and Action? (or Done if finished)"
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ]

            try:
                raw, in_tok, out_tok = self._chat(messages, temperature=0.2)
                llm_calls += 1
                total_in  += in_tok
                total_out += out_tok
            except Exception as e:
                thoughts.append(f"[Error] {e}")
                break

            # Extract thought
            tm = re.search(r"Thought:\s*(.+)", raw, re.IGNORECASE)
            if tm:
                thoughts.append(tm.group(1).strip())

            if _is_done(raw):
                break

            step = _parse_action(raw)
            if step:
                steps.append(step)
            else:
                # Couldn't parse — try to continue
                thoughts.append(f"[parse fail] {raw[:80]}")
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
            extra         = {"thoughts": thoughts},
        )
        return steps, metrics

    def generate_plan(self, task, obs, visible_objects):
        steps, metrics = self._run_loop(task, obs, visible_objects)
        print(f"[{self.name}] {len(steps)} steps, {metrics.llm_calls} LLM calls, {metrics.latency_s:.1f}s")
        return steps, metrics

    def replan(self, task, completed, failed_step, failure_reason, obs, visible_objects):
        # Seed with completed steps, re-run loop for remaining
        replan_obs = (
            f"{obs}\n[FAILURE] Step '{failed_step.get('action')} {failed_step.get('object')}' "
            f"failed: {failure_reason}. Adjust your plan accordingly."
        )
        steps, metrics = self._run_loop(task, replan_obs, visible_objects, seed_steps=completed)
        # Return only the NEW steps (after completed)
        new_steps = steps[len(completed):]
        metrics.num_steps = len(new_steps)
        return new_steps, metrics
