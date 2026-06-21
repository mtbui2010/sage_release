# planners/llm_router.py
# Method 7 — LLM Router (Ollama + External API verification)
#
# Two-model pipeline:
#   Step 1 — Ollama (local, fast) generates an initial plan
#   Step 2 — External API (OpenAI GPT-4o or Google Gemini) acts as a critic/verifier
#             that either approves the plan or rewrites problematic steps
#
# This exploits: local model's speed + frontier model's reasoning quality
# for verification only (cheaper than full generation on paid API).
#
# Supported backends: "openai" (GPT-4o-mini default) or "gemini" (gemini-2.5-flash default)
# Set OPENAI_API_KEY or GEMINI_API_KEY in environment before use.
#
# Fallback: if external API fails, returns the local plan with a note in metrics.

import json
import os
import re
import time

import requests

from pyplanner.base import (
    ACTIONS_STR, JSON_EXAMPLE, STEP_SCHEMA,
    BasePlanner, PlanMetrics, parse_steps,
)
from pyplanner.direct import DirectPlanner

# ── External API defaults ──
OPENAI_URL  = "https://api.openai.com/v1/chat/completions"
GEMINI_URL  = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

OPENAI_MODEL  = "gpt-4o-mini"
GEMINI_MODEL  = "gemini-2.5-flash"

VERIFY_SYSTEM = f"""You are a robot plan verifier.
Given a task description and a robot action plan, check for:
1. Missing Navigate/Find before object interactions
2. Wrong action order (e.g. using an object before picking it up)
3. Actions not in the allowed list
4. Steps that are logically impossible

Allowed actions: {', '.join(a for a in ACTIONS_STR.split(chr(10)) if a.strip())}

If the plan is correct, respond with exactly:
APPROVED

If there are issues, respond with a corrected plan in this exact JSON format and nothing else:
{JSON_EXAMPLE}"""


def _steps_to_text(steps):
    return json.dumps({"steps": steps}, indent=2)


def _call_openai(steps: list[dict], task: str, api_key: str, model: str) -> tuple[list[dict] | None, int, int, str]:
    """Returns (corrected_steps_or_None, in_tokens, out_tokens, note)."""
    user_msg = f"Task: {task}\n\nPlan to verify:\n{_steps_to_text(steps)}\n\nVerify and fix if needed:"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens":  1000,
    }
    try:
        resp = requests.post(OPENAI_URL, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage   = data.get("usage", {})
        in_tok  = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)

        if "APPROVED" in content.upper():
            return None, in_tok, out_tok, "openai: approved"

        corrected = parse_steps(content)
        if corrected:
            return corrected, in_tok, out_tok, f"openai: rewrote {len(steps)}→{len(corrected)} steps"
        return None, in_tok, out_tok, "openai: parse failed, keeping original"
    except Exception as e:
        return None, 0, 0, f"openai error: {e}"


def _call_gemini(steps: list[dict], task: str, api_key: str, model: str) -> tuple[list[dict] | None, int, int, str]:
    """Returns (corrected_steps_or_None, in_tokens, out_tokens, note)."""
    user_msg = f"Task: {task}\n\nPlan to verify:\n{_steps_to_text(steps)}\n\nVerify and fix if needed:"
    body = {
        "system_instruction": {"parts": [{"text": VERIFY_SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000},
    }
    url = GEMINI_URL.format(model=model) + f"?key={api_key}"
    try:
        resp = requests.post(url, json=body, timeout=30)
        resp.raise_for_status()
        data    = resp.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        usage   = data.get("usageMetadata", {})
        in_tok  = usage.get("promptTokenCount", 0)
        out_tok = usage.get("candidatesTokenCount", 0)

        if "APPROVED" in content.upper():
            return None, in_tok, out_tok, "gemini: approved"

        corrected = parse_steps(content)
        if corrected:
            return corrected, in_tok, out_tok, f"gemini: rewrote {len(steps)}→{len(corrected)} steps"
        return None, in_tok, out_tok, "gemini: parse failed, keeping original"
    except Exception as e:
        return None, 0, 0, f"gemini error: {e}"


class LLMRouterPlanner(BasePlanner):
    name        = "LLM Router"
    description = (
        "Ollama generates plan locally → external API (OpenAI/Claude) verifies & fixes. "
        "Best of both: local speed + frontier reasoning. Requires API key."
    )

    def __init__(
        self,
        host: str,
        model: str,
        provider: str          = "ollama",
        api_key: str           = "",
        verifier_backend: str  = "openai",
        verifier_model: str    = "",
        openai_api_key: str    = "",
        gemini_api_key: str    = "",
        **kwargs,
    ):
        super().__init__(host=host, model=model, provider=provider, api_key=api_key)
        self.backend           = verifier_backend
        self.verifier_model    = verifier_model
        self.openai_key  = openai_api_key or os.getenv("OPENAI_API_KEY",  "")
        self.gemini_key  = gemini_api_key  or os.getenv("GEMINI_API_KEY",   "")
        self._local            = DirectPlanner(host=host, model=model, provider=provider, api_key=api_key)

    def _verify(self, steps: list[dict], task: str) -> tuple[list[dict], int, int, str]:
        if self.backend == "gemini":
            key   = self.gemini_key
            model = self.verifier_model or GEMINI_MODEL
            if not key:
                return steps, 0, 0, "no GEMINI_API_KEY — skipped verification"
            corrected, i, o, note = _call_gemini(steps, task, key, model)
        else:
            key   = self.openai_key
            model = self.verifier_model or OPENAI_MODEL
            if not key:
                return steps, 0, 0, "no OPENAI_API_KEY — skipped verification"
            corrected, i, o, note = _call_openai(steps, task, key, model)

        return (corrected if corrected is not None else steps), i, o, note

    def generate_plan(self, task, obs, visible_objects):
        t0 = time.perf_counter()

        # ── Phase 1: local generation ──
        local_steps, local_m = self._local.generate_plan(task, obs, visible_objects)

        # ── Phase 2: external verification ──
        final_steps, v_in, v_out, note = self._verify(local_steps, task)

        metrics = PlanMetrics(
            method        = self.name,
            model         = f"{self.model}+{self.backend}",
            latency_s     = time.perf_counter() - t0,
            llm_calls     = local_m.llm_calls + 1,
            input_tokens  = local_m.input_tokens  + v_in,
            output_tokens = local_m.output_tokens + v_out,
            num_steps     = len(final_steps),
            parse_ok      = bool(final_steps),
            notes         = note,
            extra         = {
                "local_steps":    local_steps,
                "verifier_note":  note,
                "verifier_backend": self.backend,
                "steps_changed":  final_steps != local_steps,
            },
        )
        print(f"[{self.name}] local:{len(local_steps)} → final:{len(final_steps)} | {note} | {metrics.latency_s:.1f}s")
        return final_steps, metrics

    def replan(self, task, completed, failed_step, failure_reason, obs, visible_objects):
        t0 = time.perf_counter()
        local_steps, local_m = self._local.replan(
            task, completed, failed_step, failure_reason, obs, visible_objects
        )
        final_steps, v_in, v_out, note = self._verify(local_steps, task)

        metrics = PlanMetrics(
            method        = self.name,
            model         = f"{self.model}+{self.backend}",
            latency_s     = time.perf_counter() - t0,
            llm_calls     = local_m.llm_calls + 1,
            input_tokens  = local_m.input_tokens  + v_in,
            output_tokens = local_m.output_tokens + v_out,
            num_steps     = len(final_steps),
            parse_ok      = bool(final_steps),
            notes         = note,
        )
        return final_steps, metrics
