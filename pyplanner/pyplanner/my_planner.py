# pyplanner/my_planner.py
# Method — HierarchicalFewShot (custom)
#
# Combines Hierarchical planning with dynamic few-shot example retrieval:
#
#   Step 1 — Retrieval:
#     Parse all examples from few_shot_examples.py and rank by word-overlap
#     similarity to the input task. Select the top-3 closest examples.
#
#   Step 2 — High-level decomposition (LLM call 1):
#     Given the task + retrieved examples, ask the LLM to break the task
#     into 2–5 sub-goals (natural language).
#
#   Step 3 — Low-level expansion (LLM call per sub-goal):
#     For each sub-goal, ask the LLM to expand it into concrete action
#     steps, still guided by the retrieved examples.
#
# Key differences vs existing methods:
#   - vs FewShotPlanner   : only 3 retrieved examples (not all), chosen by similarity
#   - vs HierarchicalPlanner: few-shot context anchors BOTH decomposition and expansion
#
# LLM calls: 1 (decompose) + N_subgoals (expand) = 1 + N total.

from __future__ import annotations

import json
import re
import time
from typing import NamedTuple

from pyplanner.base import (
    ACTIONS_STR, JSON_EXAMPLE, STEP_SCHEMA,
    BasePlanner, PlanMetrics, parse_steps,
)
from pyplanner.few_shot_examples import FEW_SHOT_EXAMPLES


# ══════════════════════════════════════════════════════════════════════
# Example parsing & retrieval
# ══════════════════════════════════════════════════════════════════════

class _Example(NamedTuple):
    task:      str
    reasoning: str
    plan_text: str   # raw JSON string of the plan


def _parse_examples(raw: str) -> list[_Example]:
    """Parse FEW_SHOT_EXAMPLES string into a list of _Example objects."""
    examples: list[_Example] = []
    # Split on === EXAMPLE N === headers
    blocks = re.split(r"===\s*EXAMPLE\s+\d+\s*===", raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Extract Task
        tm = re.search(r"Task:\s*(.+?)(?=\nIntent|\nReasoning|\nPlan:|$)",
                       block, re.DOTALL | re.IGNORECASE)
        task = tm.group(1).strip() if tm else ""

        # Extract Reasoning
        rm = re.search(r"Reasoning:\s*(.*?)(?=\nPlan:|$)",
                       block, re.DOTALL | re.IGNORECASE)
        reasoning = rm.group(1).strip() if rm else ""

        # Extract Plan (JSON block)
        pm = re.search(r"Plan:\s*(\{.*?\})\s*$", block, re.DOTALL | re.IGNORECASE)
        plan_text = pm.group(1).strip() if pm else ""

        if task and plan_text:
            examples.append(_Example(task=task, reasoning=reasoning, plan_text=plan_text))
    return examples


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, strip punctuation."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _similarity(query: str, example_task: str) -> float:
    """Jaccard similarity between query and example task word sets."""
    q = _tokenize(query)
    e = _tokenize(example_task)
    if not q or not e:
        return 0.0
    return len(q & e) / len(q | e)


def retrieve_top_k(query: str, examples: list[_Example], k: int = 3) -> list[_Example]:
    """Return the k examples with highest word-overlap similarity to query."""
    scored = sorted(examples, key=lambda ex: _similarity(query, ex.task), reverse=True)
    return scored[:k]


def _format_examples(examples: list[_Example]) -> str:
    """Format retrieved examples into a prompt block."""
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"=== RETRIEVED EXAMPLE {i} ===\n"
            f"Task: {ex.task}\n\n"
            f"Reasoning:\n{ex.reasoning}\n\n"
            f"Plan:\n{ex.plan_text}"
        )
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════
# Prompt templates
# ══════════════════════════════════════════════════════════════════════

_HI_SYSTEM = """\
You are a household task planner (high-level).
Study the retrieved examples below to understand task structure, then break the
given task into 2–5 ordered sub-goals (natural language phrases).

Return ONLY valid JSON — no markdown, no explanation:
{"subgoals": ["sub-goal 1", "sub-goal 2", ...]}"""

_LO_SYSTEM = f"""\
You are a household assistant robot (low-level executor).
You will receive ONE sub-goal and must expand it into concrete robot action steps.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

Study the retrieved examples for correct patterns (especially container rules),
then expand the sub-goal.

Return ONLY valid JSON:
{JSON_EXAMPLE}"""


# ══════════════════════════════════════════════════════════════════════
# Planner
# ══════════════════════════════════════════════════════════════════════

class HierarchicalFewShotPlanner(BasePlanner):
    name = "HierarchicalFewShot"
    description = (
        "Hierarchical decomposition guided by the 3 most similar few-shot examples. "
        "Retrieves examples by word-overlap similarity. 1+N LLM calls."
    )

    def __init__(self, top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._top_k   = top_k
        self._all_examples = _parse_examples(FEW_SHOT_EXAMPLES)

    def _retrieve(self, task: str) -> tuple[list[_Example], str]:
        """Return (selected examples, formatted prompt block)."""
        selected = retrieve_top_k(task, self._all_examples, self._top_k)
        return selected, _format_examples(selected)

    def _decompose(
        self,
        task: str,
        obs: str,
        visible_objects: list[str],
        examples_block: str,
    ) -> tuple[list[str], int, int]:
        obj_str  = ", ".join(visible_objects[:30]) if visible_objects else "none"
        user_msg = (
            f"{examples_block}\n\n"
            f"=== NEW TASK ===\n"
            f"Task: {task}\n"
            f"Scene observation: {obs}\n"
            f"Visible objects: {obj_str}\n\n"
            "Break this into ordered sub-goals:"
        )
        raw, in_tok, out_tok = self._chat(
            [{"role": "system", "content": _HI_SYSTEM},
             {"role": "user",   "content": user_msg}],
            temperature=0.3,
        )
        subgoals = _parse_subgoals(raw)
        return subgoals, in_tok, out_tok

    def _expand(
        self,
        subgoal: str,
        task: str,
        obs: str,
        visible_objects: list[str],
        examples_block: str,
    ) -> tuple[list[dict], int, int]:
        obj_str  = ", ".join(visible_objects[:30]) if visible_objects else "none"
        user_msg = (
            f"{examples_block}\n\n"
            f"=== NEW SUB-GOAL ===\n"
            f"Overall task: {task}\n"
            f"Current sub-goal: {subgoal}\n\n"
            f"Scene observation:\n{obs}\n\n"
            f"Visible objects: {obj_str}\n\n"
            "Expand this sub-goal into action steps:"
        )
        raw, in_tok, out_tok = self._chat(
            [{"role": "system", "content": _LO_SYSTEM},
             {"role": "user",   "content": user_msg}],
            temperature=0.2,
        )
        return parse_steps(raw), in_tok, out_tok

    # ── Public API ─────────────────────────────────────────────────────

    def generate_plan(
        self,
        task: str,
        obs: str,
        visible_objects: list[str],
    ) -> tuple[list[dict], PlanMetrics]:
        t0        = time.perf_counter()
        llm_calls = 0
        total_in  = 0
        total_out = 0
        subgoals: list[str] = []

        # 1. Retrieve similar examples
        selected, examples_block = self._retrieve(task)
        retrieved_tasks = [ex.task for ex in selected]

        # 2. High-level decomposition
        try:
            subgoals, in1, out1 = self._decompose(task, obs, visible_objects, examples_block)
            llm_calls += 1
            total_in  += in1
            total_out += out1
        except Exception as e:
            subgoals = [task]
            print(f"[{self.name}] Decompose error: {e} — falling back to 1 sub-goal")

        # 3. Low-level expansion per sub-goal
        all_steps: list[dict] = []
        for sg in subgoals:
            try:
                steps, in2, out2 = self._expand(sg, task, obs, visible_objects, examples_block)
                all_steps  += steps
                llm_calls  += 1
                total_in   += in2
                total_out  += out2
            except Exception as e:
                print(f"[{self.name}] Expand error for '{sg}': {e}")

        # Deduplicate consecutive identical MoveTo steps
        deduped: list[dict] = []
        for s in all_steps:
            if (deduped
                    and s["action"] == "MoveTo"
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
            extra         = {
                "subgoals":        subgoals,
                "retrieved_tasks": retrieved_tasks,
            },
        )
        print(
            f"[{self.name}] retrieved={retrieved_tasks} | "
            f"{len(subgoals)} sub-goals → {len(deduped)} steps in {metrics.latency_s:.1f}s"
        )
        return deduped, metrics

    def replan(
        self,
        task: str,
        completed: list[dict],
        failed_step: dict,
        failure_reason: str,
        obs: str,
        visible_objects: list[str],
    ) -> tuple[list[dict], PlanMetrics]:
        completed_desc = ", ".join(
            f"{s.get('action')} {s.get('object')}" for s in completed
        ) or "nothing yet"
        remaining_task = (
            f"{task}. Already done: {completed_desc}. "
            f"Last action failed ({failed_step.get('action')} {failed_step.get('object')}): "
            f"{failure_reason}. Generate only the remaining steps."
        )
        steps, metrics = self.generate_plan(remaining_task, obs, visible_objects)
        metrics.method = self.name
        return steps, metrics


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

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
    return re.findall(r'"([^"]{5,})"', raw)
