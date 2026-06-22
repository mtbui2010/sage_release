"""
evaluate.py
===========
Run all (or selected) pyplanner methods against eval_dataset.json,
compute quantitative metrics for each sample, and save results to CSV.

Metrics computed per (method × sample):
  Plan quality (without environment):
    executability     — % steps with valid action + non-empty object
    precondition      — % interact-steps preceded by Navigate/Find
    redundancy        — % consecutive duplicate steps (lower = better)
    completeness      — % expected objects mentioned in plan
    hallucination     — % steps referencing objects not in visible list
    parse_ok          — 1 if plan parsed to non-empty list, else 0
    num_steps         — total generated steps
    ref_steps         — reference plan step count
    step_ratio        — num_steps / ref_steps  (1.0 = perfect match)

  Efficiency:
    latency_s         — wall-clock seconds
    llm_calls         — LLM round-trips
    input_tokens      — prompt tokens
    output_tokens     — completion tokens
    total_tokens      — input + output
    tokens_per_step   — total_tokens / num_steps

  Robustness (only for samples with fail_injection):
    replan_ok         — 1 if replan returned a non-empty plan
    replan_latency_s  — seconds for replan call
    replan_steps      — number of steps in replan
    replan_llm_calls  — LLM calls in replan
    step_overlap      — fraction of replan steps also in original plan

  Aggregate:
    quality_score     — weighted combination of quality metrics (0–1)
    efficiency_score  — 1 − normalised(latency × llm_calls) (0–1)
    overall_score     — 0.6 × quality + 0.25 × efficiency + 0.15 × robustness

Usage:
    # All methods (requires Ollama running):
    python evaluate.py

    # Select methods:
    python evaluate.py --methods Direct CoT

    # Custom dataset / model / output:
    python evaluate.py --dataset eval_dataset.json \\
                       --model llama3.2 \\
                       --host http://192.168.1.11:11434 \\
                       --out results.csv

    # Dry-run (no LLM calls — stub mode for CI / testing):
    python evaluate.py --dry-run

    # Use OpenAI instead of Ollama:
    python evaluate.py --provider openai --model gpt-4o-mini \\
                       --api-key sk-...

    # Verbose per-sample output:
    python evaluate.py --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any

# ── Resolve pyplanner (works with or without pip install) ─────────────
try:
    import pyplanner
except ModuleNotFoundError:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, "..", "pyplanner"))
    import pyplanner

from pyplanner import REGISTRY, DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_BACKEND

# ── Import scoring helpers from make_dataset ──────────────────────────
try:
    from make_dataset import (
        score_executability, score_precondition, score_redundancy,
        score_completeness, score_hallucination, compute_quality_score,
        VALID_ACTIONS,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_dataset import (
        score_executability, score_precondition, score_redundancy,
        score_completeness, score_hallucination, compute_quality_score,
        VALID_ACTIONS,
    )




# ═════════════════════════════════════════════════════════════════════
# Connection check
# ═════════════════════════════════════════════════════════════════════

def check_connection(provider: str, host: str, model: str, api_key: str) -> tuple[bool, str]:
    """
    Ping the LLM backend before evaluation starts.
    Returns (ok: bool, error_message: str).
    """
    import urllib.request, urllib.error

    if provider == "ollama":
        try:
            url = host.rstrip("/") + "/api/tags"
            # Some reverse proxies (e.g. localhost:11434) reject the
            # default "Python-urllib/x.y" User-Agent with HTTP 403. Send a
            # browser-ish UA so the pre-flight check matches what the actual
            # requests-based chat in base.py already gets through with.
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            available = [m["name"] for m in data.get("models", [])]
            # Check if requested model is present
            if not any(model.split(":")[0] in m for m in available):
                hint = ""
                if available:
                    close = [m for m in available if any(p in m for p in model.split(":")[0].split())]
                    hint  = f"\n  Available models: {available[:6]}"
                    if close:
                        hint = f"\n  Did you mean: {close[0]}?" + hint
                else:
                    hint = "\n  No models pulled yet. Run:  ollama pull " + model
                return False, (
                    f"Model \"{model}\" not found in Ollama.{hint}\n"
                    f"  Pull it with:  ollama pull {model}"
                )
            return True, f"Ollama OK — model \"{model}\" available"
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", str(e))
            return False, (
                f"Cannot reach Ollama at {host}\n"
                f"  Error  : {reason}\n"
                f"  Fix    : run  ollama serve  (or check --host URL)\n"
                f"  Example: python evaluate.py --host http://192.168.1.11:11434"
            )
        except Exception as e:
            return False, f"Ollama check failed: {e}"

    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            return False, (
                "OpenAI API key not set.\n"
                "  Fix: python evaluate.py --api-key sk-...\n"
                "       or:  export OPENAI_API_KEY=sk-..."
            )
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
            return True, "OpenAI API key valid"
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return False, "OpenAI API key is invalid (HTTP 401 Unauthorized)."
            return False, f"OpenAI API error: HTTP {e.code}"
        except Exception as e:
            return False, f"OpenAI connection error: {e}"

    elif provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not key:
            return False, (
                "Gemini API key not set.\n"
                "  Fix: python evaluate.py --api-key AIza...\n"
                "       or:  export GEMINI_API_KEY=AIza..."
            )
        return True, "Gemini key present (not pinged)"

    return True, ""

# ═════════════════════════════════════════════════════════════════════
# Result dataclass
# ═════════════════════════════════════════════════════════════════════
@dataclass
class SampleResult:
    # Identity
    method:     str = ""
    model:      str = ""
    backend:    str = ""
    task_id:    str = ""
    task_desc:  str = ""
    room:       str = ""
    difficulty: str = ""

    # Plan quality
    parse_ok:       float = 0.0
    num_steps:      int   = 0
    ref_steps:      int   = 0
    step_ratio:     float = 0.0
    executability:  float = 0.0
    precondition:   float = 0.0
    redundancy:     float = 0.0
    completeness:   float = 0.0
    hallucination:  float = 0.0
    quality_score:  float = 0.0

    # Efficiency
    latency_s:       float = 0.0
    llm_calls:       int   = 0
    input_tokens:    int   = 0
    output_tokens:   int   = 0
    total_tokens:    int   = 0
    tokens_per_step: float = 0.0
    efficiency_score: float = 0.0

    # Robustness (replan)
    has_fail_injection: int   = 0
    replan_ok:          float = 0.0
    replan_latency_s:   float = 0.0
    replan_steps:       int   = 0
    replan_llm_calls:   int   = 0
    step_overlap:       float = 0.0
    robustness_score:   float = 0.0

    # Aggregate
    overall_score: float = 0.0

    # Error tracking
    error: str = ""

    def to_row(self) -> dict:
        return asdict(self)


CSV_COLUMNS = [
    "method", "model", "backend", "task_id", "task_desc", "room", "difficulty",
    # quality
    "parse_ok", "num_steps", "ref_steps", "step_ratio",
    "executability", "precondition", "redundancy", "completeness", "hallucination",
    "quality_score",
    # efficiency
    "latency_s", "llm_calls", "input_tokens", "output_tokens", "total_tokens",
    "tokens_per_step", "efficiency_score",
    # robustness
    "has_fail_injection", "replan_ok", "replan_latency_s", "replan_steps",
    "replan_llm_calls", "step_overlap", "robustness_score",
    # aggregate
    "overall_score",
    # meta
    "error",
]


# ═════════════════════════════════════════════════════════════════════
# Scoring helpers
# ═════════════════════════════════════════════════════════════════════

def score_step_overlap(original: list[dict], replan: list[dict]) -> float:
    """Fraction of replan steps whose (action, object) pair appeared in original."""
    if not replan:
        return 0.0
    orig_pairs = {(s.get("action",""), s.get("object","")) for s in original}
    overlap = sum(
        1 for s in replan
        if (s.get("action",""), s.get("object","")) in orig_pairs
    )
    return round(overlap / len(replan), 4)


def compute_efficiency_score(latency_s: float, llm_calls: int,
                              max_latency: float = 60.0, max_calls: int = 20) -> float:
    """Normalised efficiency in [0,1]. Higher = faster / fewer calls."""
    norm_lat   = min(latency_s / max_latency, 1.0)
    norm_calls = min(llm_calls / max_calls,   1.0)
    raw = 1.0 - 0.7 * norm_lat - 0.3 * norm_calls
    return round(max(0.0, raw), 4)


def compute_robustness_score(replan_ok: float, step_overlap: float,
                              replan_latency_s: float) -> float:
    """Score for replan quality in [0,1]."""
    if replan_ok == 0:
        return 0.0
    lat_penalty = min(replan_latency_s / 60.0, 1.0) * 0.2
    return round(max(0.0, replan_ok * 0.6 + (1 - step_overlap) * 0.2 - lat_penalty + 0.2), 4)


def compute_overall_score(quality: float, efficiency: float, robustness: float,
                           has_fail: bool) -> float:
    if has_fail:
        return round(0.55 * quality + 0.20 * efficiency + 0.25 * robustness, 4)
    else:
        return round(0.65 * quality + 0.35 * efficiency, 4)


# ═════════════════════════════════════════════════════════════════════
# Dry-run stub planner (no LLM calls)
# ═════════════════════════════════════════════════════════════════════

class _DryRunPlanner:
    """Returns a minimal synthetic plan without hitting an LLM."""

    name        = "DryRun"
    description = "Stub planner for testing — no LLM calls."
    provider    = "none"
    model       = "none"

    def generate_plan(self, task, obs, visible_objects):
        from pyplanner.base import PlanMetrics
        steps = [
            {"action": "MoveTo", "object": visible_objects[0] if visible_objects else "object",
             "target": "", "reason": "stub"},
            {"action": "Pick",     "object": visible_objects[0] if visible_objects else "object",
             "target": "", "reason": "stub"},
        ]
        m = PlanMetrics(method=self.name, model=self.model, backend=self.provider,
                        latency_s=0.01, llm_calls=1, input_tokens=50, output_tokens=20,
                        num_steps=len(steps), parse_ok=True)
        return steps, m

    def replan(self, task, completed, failed_step, failure_reason, obs, visible_objects):
        from pyplanner.base import PlanMetrics
        steps = [{"action": "MoveTo", "object": "sink", "target": "", "reason": "stub replan"}]
        m = PlanMetrics(method=self.name, model=self.model, backend=self.provider,
                        latency_s=0.01, llm_calls=1, input_tokens=60, output_tokens=20,
                        num_steps=1, parse_ok=True)
        return steps, m


# ═════════════════════════════════════════════════════════════════════
# Core evaluation loop
# ═════════════════════════════════════════════════════════════════════

def evaluate_sample(
    planner,
    sample: dict,
    verbose: bool = False,
) -> SampleResult:
    res = SampleResult(
        method     = planner.name,
        model      = getattr(planner, "model", ""),
        backend    = getattr(planner, "provider", ""),
        task_id    = sample["task_id"],
        task_desc  = sample["task_desc"][:60],
        room       = sample["room"],
        difficulty = sample["difficulty"],
        ref_steps  = len(sample["reference_steps"]),
        has_fail_injection = 1 if sample.get("fail_injection") else 0,
    )

    try:
        # ── generate_plan ──────────────────────────────────────────────
        steps, metrics = planner.generate_plan(
            task            = sample["task_desc"],
            obs             = sample["obs"],
            visible_objects = sample["visible_objects"],
        )

        # Surface any LLM-level error stored in metrics.notes
        if metrics.notes and not metrics.parse_ok:
            raise RuntimeError(f"LLM error: {metrics.notes}")

        res.parse_ok       = 1.0 if (steps and metrics.parse_ok) else 0.0
        res.num_steps      = len(steps)
        res.step_ratio     = round(res.num_steps / res.ref_steps, 4) if res.ref_steps else 0.0
        res.latency_s      = round(metrics.latency_s,  4)
        res.llm_calls      = metrics.llm_calls
        res.input_tokens   = metrics.input_tokens
        res.output_tokens  = metrics.output_tokens
        res.total_tokens   = metrics.total_tokens
        res.tokens_per_step = round(metrics.tokens_per_step, 2)

        # quality metrics
        res.executability = score_executability(steps)
        res.precondition  = score_precondition(steps)
        res.redundancy    = score_redundancy(steps)
        res.completeness  = score_completeness(steps, sample["expected_objects"])
        res.hallucination = score_hallucination(steps, sample["visible_objects"])
        res.quality_score = compute_quality_score({
            "executability": res.executability,
            "precondition":  res.precondition,
            "completeness":  res.completeness,
            "redundancy":    res.redundancy,
            "hallucination": res.hallucination,
        })
        res.efficiency_score = compute_efficiency_score(res.latency_s, res.llm_calls)

        # ── replan (only if fail_injection present) ────────────────────
        fi = sample.get("fail_injection")
        if fi and fi.get("at_step") is not None:
            at_step = fi["at_step"]
            completed_steps = steps[:at_step] if at_step <= len(steps) else steps
            failed_step     = steps[at_step]  if at_step < len(steps) else {"action":"Wait","object":""}

            replan_steps, replan_m = planner.replan(
                task            = sample["task_desc"],
                completed       = completed_steps,
                failed_step     = failed_step,
                failure_reason  = fi["failure_reason"],
                obs             = sample["obs"],
                visible_objects = sample["visible_objects"],
            )

            res.replan_ok        = 1.0 if (replan_steps and replan_m.parse_ok) else 0.0
            res.replan_latency_s = round(replan_m.latency_s, 4)
            res.replan_steps     = len(replan_steps)
            res.replan_llm_calls = replan_m.llm_calls
            res.step_overlap     = score_step_overlap(steps, replan_steps)
            res.robustness_score = compute_robustness_score(
                res.replan_ok, res.step_overlap, res.replan_latency_s
            )

        res.overall_score = compute_overall_score(
            res.quality_score, res.efficiency_score, res.robustness_score,
            has_fail=bool(fi and fi.get("at_step") is not None),
        )

    except Exception as e:
        res.error = str(e)[:200]
        # Always print the error (not just in verbose mode)
        # so silent failures are immediately visible
        err_short = str(e)[:100]
        print(f"\n  ⚠  [{res.task_id}] {res.method} error: {err_short}")
        if "Connection refused" in str(e) or "Cannot connect" in str(e) or "timeout" in str(e).lower():
            print("     ↳  LLM server unreachable — check connection or use --dry-run")
        if verbose:
            traceback.print_exc()

    if verbose:
        status = "✅" if not res.error else "❌"
        print(
            f"  {status} [{res.task_id}] {res.task_desc[:40]:40s}"
            f"  steps={res.num_steps:2d}/{res.ref_steps:2d}"
            f"  Q={res.quality_score:.2f}  E={res.efficiency_score:.2f}"
            f"  {res.latency_s:.1f}s"
            + (f"  ERR={res.error[:30]}" if res.error else "")
        )

    return res


def run_evaluation(
    dataset_path:  str,
    methods:       list[str],
    host:          str,
    model:         str,
    provider:      str,
    api_key:       str,
    out_path:      str,
    dry_run:       bool,
    verbose:       bool,
    method_kwargs: dict[str, dict],
) -> str:
    """
    Run evaluation, return path to CSV file.
    """
    # ── Load dataset ──────────────────────────────────────────────────
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"] if "samples" in data else data
    print(f"\n📂  Loaded {len(samples)} samples from {dataset_path}")

    # ── Connection check ──────────────────────────────────────────────
    if not dry_run:
        ok, msg = check_connection(provider, host, model, api_key)
        if ok:
            print(f"  ✅  Connection OK — {msg}")
        else:
            print(f"\n{'═'*60}")
            print(f"  ❌  CONNECTION FAILED")
            print(f"{'─'*60}")
            for line in msg.splitlines():
                print(f"  {line}")
            print(f"{'═'*60}")
            print("\n  Tip: use --dry-run to test without LLM\n")
            sys.exit(1)

    # ── Build planners ────────────────────────────────────────────────
    planners = []
    if dry_run:
        stub = _DryRunPlanner()
        stub.name = "DryRun"
        planners = [stub]
        print("🔧  Dry-run mode — no LLM calls will be made")
    else:
        for name in methods:
            kwargs = method_kwargs.get(name, {})
            kwargs.update(provider=provider, api_key=api_key)
            try:
                p = pyplanner.get(name, host=host, model=model, **kwargs)
                planners.append(p)
                print(f"  ✅  {name:15}  ({provider}/{model})")
            except Exception as e:
                print(f"  ❌  {name}: {e}")

    if not planners:
        print("No planners initialised — aborting.")
        return ""

    # ── Run evaluation ────────────────────────────────────────────────
    all_results: list[SampleResult] = []
    total = len(planners) * len(samples)
    done  = 0

    for planner in planners:
        print(f"\n▶  Evaluating [{planner.name}]  ({len(samples)} samples)")
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 3

        for sample in samples:
            result = evaluate_sample(planner, sample, verbose=verbose)
            all_results.append(result)
            done += 1

            # Fail-fast: abort if too many consecutive errors (likely a connection issue)
            if result.error:
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"\n  ❌  {consecutive_errors} consecutive errors — aborting [{planner.name}]")
                    print(f"     Last error: {result.error}")
                    print(f"     Tip: run  python evaluate.py --dry-run  to test without LLM")
                    break
            else:
                consecutive_errors = 0

            if not verbose:
                pct = done / total * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                err_indicator = f"  ⚠ {consecutive_errors} err" if consecutive_errors else ""
                print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}{err_indicator}", end="", flush=True)
        if not verbose:
            print()

    # ── Write CSV ─────────────────────────────────────────────────────
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r.to_row())

    # ── Print summary ─────────────────────────────────────────────────
    _print_summary(all_results)
    print(f"\n✅  Results saved → {out_path}")
    print(f"    Rows: {len(all_results)}  ·  Columns: {len(CSV_COLUMNS)}\n")
    return out_path


def _print_summary(results: list[SampleResult]):
    from collections import defaultdict
    by_method: dict[str, list[SampleResult]] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    cols = ["method", "model", "backend",
            "quality_score", "efficiency_score", "robustness_score", "overall_score",
            "latency_s", "llm_calls", "tokens_per_step",
            "parse_ok", "executability", "precondition", "completeness",
            "redundancy", "hallucination", "step_ratio", "errors"]

    header = (
        f"\n{'─'*115}\n"
        f"  {'Method':<16} {'Model':<14} {'Backend':<10}"
        f" {'Quality':>7} {'Effic':>6} {'Robust':>7} {'Overall':>8}"
        f" {'Latency':>8} {'Calls':>5} {'Tok/step':>8}"
        f" {'Parse%':>7} {'Exec%':>6} {'PreCond':>7} {'Compl':>6}"
        f" {'Redund':>6} {'Halluc':>6} {'StepR':>6} {'Errors':>6}"
        f"\n{'─'*115}"
    )
    print(header)

    for method, rows in by_method.items():
        n   = len(rows)
        err = sum(1 for r in rows if r.error)

        def avg(attr): return round(sum(getattr(r, attr) for r in rows) / n, 3)

        print(
            f"  {method:<16} {rows[0].model:<14} {rows[0].backend:<10}"
            f" {avg('quality_score'):>7.3f}"
            f" {avg('efficiency_score'):>6.3f}"
            f" {avg('robustness_score'):>7.3f}"
            f" {avg('overall_score'):>8.3f}"
            f" {avg('latency_s'):>8.2f}s"
            f" {avg('llm_calls'):>5.1f}"
            f" {avg('tokens_per_step'):>8.0f}"
            f" {avg('parse_ok')*100:>7.1f}%"
            f" {avg('executability')*100:>6.1f}%"
            f" {avg('precondition')*100:>7.1f}%"
            f" {avg('completeness')*100:>6.1f}%"
            f" {avg('redundancy')*100:>6.1f}%"
            f" {avg('hallucination')*100:>6.1f}%"
            f" {avg('step_ratio'):>6.2f}"
            f" {err:>5}/{n}"
        )

    print(f"{'─'*115}")


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    all_methods = list(REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Evaluate pyplanner methods quantitatively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",  default="eval_dataset.json",
                        help="Path to evaluation dataset JSON (default: eval_dataset.json)")
    parser.add_argument("--methods",  nargs="+", default=all_methods,
                        choices=all_methods + ["all"],
                        help=f"Methods to evaluate (default: all). Choices: {all_methods}")
    parser.add_argument("--host",     default=DEFAULT_HOST,
                        help=f"Ollama host URL (default: {DEFAULT_HOST})")
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--provider", default=DEFAULT_BACKEND,
                        choices=["ollama", "openai", "gemini"],
                        help="LLM provider (default: ollama)")
    parser.add_argument("--api-key",  default="",
                        help="API key for openai/gemini (or set env var)")
    parser.add_argument("--out",      default="eval_results.csv",
                        help="Output CSV path (default: eval_results.csv)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Stub mode — no LLM calls (for testing)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print per-sample results")
    # Method-specific kwargs
    parser.add_argument("--react-max-steps",      type=int, default=15)
    parser.add_argument("--refine-iterations",    type=int, default=2)
    args = parser.parse_args()

    # Handle "all" shorthand
    methods = all_methods if "all" in args.methods else args.methods

    method_kwargs = {
        "ReAct":       {"max_steps":        args.react_max_steps},
        "Self-Refine": {"max_iterations":   args.refine_iterations},
    }

    # Auto-generate dataset if missing
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        print(f"⚠  Dataset not found: {dataset_path}")
        print("   Running make_dataset.py to generate it...")
        try:
            from make_dataset import build_dataset, validate_dataset
            samples = build_dataset()
            warnings = validate_dataset(samples)
            if warnings:
                for w in warnings:
                    print(f"   ⚠  {w}")
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump({"version": "1.0", "samples": samples}, f, indent=2, ensure_ascii=False)
            print(f"   ✅  Generated {len(samples)} samples → {dataset_path}")
        except Exception as e:
            print(f"   ❌  Failed to auto-generate dataset: {e}")
            sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  PyPlanner Evaluation")
    print(f"{'═'*60}")
    print(f"  Methods  : {methods}")
    print(f"  Provider : {args.provider}")
    print(f"  Model    : {args.model}")
    print(f"  Host     : {args.host}")
    print(f"  Output   : {args.out}")
    if args.dry_run:
        print("  Mode     : DRY RUN (no LLM)")
    print(f"{'═'*60}")

    t_start = time.perf_counter()
    out_path = run_evaluation(
        dataset_path  = dataset_path,
        methods       = methods,
        host          = args.host,
        model         = args.model,
        provider      = args.provider,
        api_key       = args.api_key,
        out_path      = args.out,
        dry_run       = args.dry_run,
        verbose       = args.verbose,
        method_kwargs = method_kwargs,
    )
    elapsed = time.perf_counter() - t_start
    if out_path:
        print(f"  Total time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()