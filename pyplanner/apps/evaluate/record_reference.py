"""
record_reference.py
===================
Generate EXECUTABLE ground-truth reference plans by actually running
candidate action sequences inside AI2-THOR and keeping only steps
that return success=True from the simulator.

Why this is needed
------------------
LLM-generated or hand-written plans have no guarantee of executability.
True ground truth requires simulator feedback:
  - step.success == True  → action worked
  - env.done == True      → task completed

Strategy
--------
For each task we use a "greedy template executor":

  1. Load the scene in AI2-THOR.
  2. Take the candidate steps (from LLM or hand-written).
  3. Execute each step via ThorClient.step().
  4. Record the step only if success=True.
  5. If a step fails, try alternative object names (real scene names).
  6. After all steps, record task_success = any step succeeded AND
     final reward > 0 or env signals done.

The saved reference plan is the EXECUTED sequence — not what was
planned, but what actually worked in the simulator.

Output
------
  eval_dataset_gt.json  — dataset with ground-truth executable plans
  record_report.json    — per-task execution report

Usage
-----
    # Requires: python thor_server.py  running in a separate terminal

    python record_reference.py
    python record_reference.py --scenes FloorPlan1 FloorPlan2
    python record_reference.py --ref-source llm --model llama3.2
    python record_reference.py --max-retries 3 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict

_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from thor_app.sim_client import ThorClient
except ImportError:
    sys.path.insert(0, _HERE)
    from thor_app.sim_client import ThorClient

try:
    import pyplanner
    from pyplanner import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_BACKEND
    from pyplanner.base import parse_steps
except ImportError:
    sys.path.insert(0, os.path.join(_HERE, "..", "pyplanner"))
    import pyplanner
    from pyplanner import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_BACKEND
    from pyplanner.base import parse_steps

try:
    from make_dataset import SAMPLES_RAW
    from thor_app.evaluate.make_dataset_from_sim import (
        inspect_scene, _find_real_object,
        generate_llm_reference_steps, _check_llm_connection,
    )
except ImportError:
    sys.path.insert(0, _HERE)
    from make_dataset import SAMPLES_RAW
    from thor_app.evaluate.make_dataset_from_sim import (
        inspect_scene, _find_real_object,
        generate_llm_reference_steps, _check_llm_connection,
    )


# ═════════════════════════════════════════════════════════════════════
# Execution result per task
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionRecord:
    task_id:          str
    task_desc:        str
    scene:            str
    room:             str
    difficulty:       str
    # Executed (ground-truth) plan
    gt_steps:         list[dict] = field(default_factory=list)
    # Execution outcome
    task_success:     bool  = False
    steps_attempted:  int   = 0
    steps_succeeded:  int   = 0
    total_reward:     float = 0.0
    # Metadata
    candidate_source: str   = ""   # "manual" | "llm"
    n_candidate_steps:int   = 0
    exec_latency_s:   float = 0.0
    warnings:         list[str] = field(default_factory=list)

    @property
    def step_success_rate(self) -> float:
        return self.steps_succeeded / self.steps_attempted if self.steps_attempted else 0.0


# ═════════════════════════════════════════════════════════════════════
# Core: execute candidate steps, keep only successful ones
# ═════════════════════════════════════════════════════════════════════

def _try_step(client: ThorClient, action: str, obj: str, target: str = "") -> dict:
    """Execute one step and return result dict."""
    return client.step(action, obj, target)


def _resolve_alternatives(obj_name: str, scene_info: dict) -> list[str]:
    """
    Return list of object name variants to try if the original fails.
    e.g. "coffee_machine" → ["CoffeeMachine", "coffee_machine", "Machine"]
    """
    om = scene_info["object_map"]
    candidates = []

    real = _find_real_object(obj_name, om)
    if real:
        candidates.append(real)

    # Also try original name (some servers accept snake_case)
    if obj_name not in candidates:
        candidates.append(obj_name)

    return candidates


def execute_candidate_plan(
    client:      ThorClient,
    scene:       str,
    scene_info:  dict,
    candidate_steps: list[dict],
    max_retries: int = 2,
    verbose:     bool = False,
) -> tuple[list[dict], dict]:
    """
    Execute candidate steps in simulator.
    For each step, try object name alternatives if first attempt fails.

    Returns:
        (gt_steps, outcome)
        gt_steps — only the steps that actually succeeded
        outcome  — {task_success, steps_attempted, steps_succeeded, total_reward}
    """
    gt_steps        = []
    steps_attempted = 0
    steps_succeeded = 0
    total_reward    = 0.0
    task_done       = False

    for step in candidate_steps:
        action = step.get("action", "Wait")
        obj    = step.get("object", "")
        target = step.get("target", "")
        reason = step.get("reason", "")

        if action == "Wait":
            # Wait always succeeds — include it
            gt_steps.append({"action": action, "object": "", "target": "", "reason": reason})
            steps_succeeded += 1
            continue

        # Resolve object name alternatives
        obj_variants    = _resolve_alternatives(obj,    scene_info) if obj    else [""]
        target_variants = _resolve_alternatives(target, scene_info) if target else [""]

        step_ok = False

        for obj_try in obj_variants:
            for target_try in target_variants:
                steps_attempted += 1
                result = _try_step(client, action, obj_try, target_try)

                if verbose:
                    status = "✓" if result.get("success") else "✗"
                    print(f"    {status} {action} {obj_try}"
                          + (f" → {target_try}" if target_try else "")
                          + (f"  ({result.get('msg','')})" if not result.get("success") else ""))

                if result.get("done"):
                    task_done = True
                    gt_steps.append({
                        "action": action, "object": obj_try,
                        "target": target_try, "reason": reason,
                    })
                    total_reward  += result.get("reward", 0.0)
                    steps_succeeded += 1
                    step_ok = True
                    break

                if result.get("success"):
                    gt_steps.append({
                        "action": action, "object": obj_try,
                        "target": target_try, "reason": reason,
                    })
                    total_reward  += result.get("reward", 0.0)
                    steps_succeeded += 1
                    step_ok = True
                    break

            if step_ok or task_done:
                break

        if task_done:
            break

    # Final reward check — some tasks signal success only at the end
    if not task_done and total_reward > 0:
        task_done = True

    return gt_steps, {
        "task_success":    task_done,
        "steps_attempted": steps_attempted,
        "steps_succeeded": steps_succeeded,
        "total_reward":    round(total_reward, 4),
    }


# ═════════════════════════════════════════════════════════════════════
# Build ground-truth dataset
# ═════════════════════════════════════════════════════════════════════

def record_ground_truth(
    client:        ThorClient,
    ref_source:    str = "manual",   # "manual" | "llm"
    scenes:        list[str] | None = None,
    planner_host:  str = DEFAULT_HOST,
    planner_model: str = DEFAULT_MODEL,
    provider:      str = DEFAULT_BACKEND,
    api_key:       str = "",
    max_retries:   int = 2,
    verbose:       bool = False,
) -> tuple[list[dict], list[ExecutionRecord]]:
    """
    For every sample in SAMPLES_RAW:
      1. Load scene, inspect real objects
      2. Get candidate steps (manual or LLM)
      3. Execute in simulator, keep only successful steps
      4. Save as ground-truth dataset

    Returns: (samples, records)
    """
    # Collect scenes
    scenes_needed = set(scenes) if scenes else set(r["scene"] for r in SAMPLES_RAW)

    # ── Inspect all scenes once ────────────────────────────────────
    print(f"\n🔍  Inspecting {len(scenes_needed)} scenes...")
    scene_cache: dict[str, dict] = {}
    for i, scene in enumerate(sorted(scenes_needed)):
        print(f"  [{i+1}/{len(scenes_needed)}] {scene}...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            scene_cache[scene] = inspect_scene(client, scene)
            print(f"✅  {len(scene_cache[scene]['object_map'])} objects  "
                  f"{len(scene_cache[scene]['visible_objects'])} visible  "
                  f"({time.perf_counter()-t0:.1f}s)")
        except Exception as e:
            print(f"❌  {e}")

    # ── Process each sample ────────────────────────────────────────
    print(f"\n▶  Recording ground-truth plans ({ref_source} candidates)...")
    samples = []
    records = []

    for raw in SAMPLES_RAW:
        scene = raw["scene"]
        if scene not in scene_cache:
            continue

        info = scene_cache[scene]
        rec  = ExecutionRecord(
            task_id          = raw["task_id"],
            task_desc        = raw["task_desc"],
            scene            = scene,
            room             = raw["room"],
            difficulty       = raw["difficulty"],
            candidate_source = ref_source,
        )

        if verbose:
            print(f"\n  [{raw['task_id']}] {raw['task_desc'][:55]}")

        # ── Get candidate steps ──────────────────────────────────
        if ref_source == "llm":
            try:
                candidate_steps, _ = generate_llm_reference_steps(
                    task_desc     = raw["task_desc"],
                    scene_info    = info,
                    planner_host  = planner_host,
                    planner_model = planner_model,
                    provider      = provider,
                    api_key       = api_key,
                )
            except Exception as e:
                rec.warnings.append(f"LLM failed: {e}, using manual")
                candidate_steps = raw["reference_steps"]
        else:
            candidate_steps = raw["reference_steps"]

        rec.n_candidate_steps = len(candidate_steps)

        # ── Reset scene fresh before execution ────────────────────
        reset_resp = client.reset(scene)
        if reset_resp.get("status") != "ok":
            rec.warnings.append(f"Scene reset failed: {reset_resp.get('msg','')}")
            records.append(rec)
            continue

        # ── Execute candidates in simulator ───────────────────────
        t0 = time.perf_counter()
        gt_steps, outcome = execute_candidate_plan(
            client          = client,
            scene           = scene,
            scene_info      = info,
            candidate_steps = candidate_steps,
            max_retries     = max_retries,
            verbose         = verbose,
        )
        rec.exec_latency_s    = round(time.perf_counter() - t0, 3)
        rec.gt_steps          = gt_steps
        rec.task_success      = outcome["task_success"]
        rec.steps_attempted   = outcome["steps_attempted"]
        rec.steps_succeeded   = outcome["steps_succeeded"]
        rec.total_reward      = outcome["total_reward"]

        status = "✅" if rec.task_success else ("⚠ " if gt_steps else "❌")
        print(f"  {status} [{raw['task_id']}] "
              f"{len(candidate_steps)}→{len(gt_steps)} steps  "
              f"reward={rec.total_reward:.1f}  "
              f"{rec.exec_latency_s:.1f}s"
              + (f"  ⚠ {rec.warnings[-1][:40]}" if rec.warnings else ""))

        # ── Build dataset sample ──────────────────────────────────
        # Map expected_objects to real names
        real_expected = []
        for exp in raw["expected_objects"]:
            real = _find_real_object(exp, info["object_map"])
            real_expected.append(real if real else exp)

        # visible_objects from the initial reset (what the agent actually sees)
        real_visible = reset_resp.get("visible_objects",
                                       info["visible_objects"])

        sample = {
            "task_id":          raw["task_id"],
            "task_desc":        raw["task_desc"],
            "room":             raw["room"],
            "scene":            scene,
            "obs":              reset_resp.get("obs", info["obs"]),
            "visible_objects":  real_visible,
            "reference_steps":  gt_steps,          # ← EXECUTED steps only
            "expected_objects": real_expected,
            "difficulty":       raw["difficulty"],
            "fail_injection":   raw.get("fail_injection") or {},
            "_meta": {
                "gt_source":         "simulator_execution",
                "candidate_source":  ref_source,
                "candidate_steps":   len(candidate_steps),
                "executed_steps":    len(gt_steps),
                "task_success":      rec.task_success,
                "step_success_rate": round(rec.step_success_rate, 3),
                "total_reward":      rec.total_reward,
                "grounded_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                "warnings":          rec.warnings,
            },
        }
        samples.append(sample)
        records.append(rec)

    return samples, records


# ═════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════

def print_summary(samples: list[dict], records: list[ExecutionRecord]):
    total          = len(records)
    succeeded      = sum(1 for r in records if r.task_success)
    partial        = sum(1 for r in records if r.gt_steps and not r.task_success)
    failed         = sum(1 for r in records if not r.gt_steps)
    avg_gt_steps   = sum(len(s["reference_steps"]) for s in samples) / len(samples) if samples else 0
    avg_reward     = sum(r.total_reward for r in records) / total if total else 0
    avg_lat        = sum(r.exec_latency_s for r in records) / total if total else 0

    print(f"\n{'─'*60}")
    print(f"  Ground-truth recording summary — {total} tasks")
    print(f"{'─'*60}")
    print(f"  Task success   : {succeeded}/{total} ({100*succeeded/total:.0f}%)")
    print(f"  Partial (steps ok, task not done): {partial}")
    print(f"  Failed (0 steps executed) : {failed}")
    print(f"  Avg GT steps   : {avg_gt_steps:.1f}  (executed steps only)")
    print(f"  Avg reward     : {avg_reward:.2f}")
    print(f"  Avg exec time  : {avg_lat:.1f}s / task")

    if failed > 0:
        print(f"\n  ❌ Failed tasks:")
        for r in records:
            if not r.gt_steps:
                warn = r.warnings[-1][:60] if r.warnings else "no steps succeeded"
                print(f"     [{r.task_id}] {r.task_desc[:40]}  — {warn}")

    print(f"{'─'*60}\n")


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Record executable ground-truth plans in AI2-THOR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sim-host",    default="localhost")
    parser.add_argument("--sim-port",    type=int, default=5555)
    parser.add_argument("--ref-source",  default="manual",
                        choices=["manual", "llm"],
                        help="Source of candidate steps before execution. "
                             "'manual' = use hand-written steps from make_dataset.py. "
                             "'llm'    = ask LLM to generate candidates, then execute.")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--host",        default=DEFAULT_HOST,
                        help="Ollama/LLM host URL")
    parser.add_argument("--provider",    default=DEFAULT_BACKEND,
                        choices=["ollama", "openai", "gemini"])
    parser.add_argument("--api-key",     default="")
    parser.add_argument("--scenes",      nargs="+", default=None)
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Object name variants to try per step (default: 2)")
    parser.add_argument("--out",         default="eval_dataset_gt.json",
                        help="Output dataset file")
    parser.add_argument("--report",      default="record_report.json",
                        help="Execution report JSON")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Show plan without executing in simulator")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  Ground-Truth Recorder — AI2-THOR")
    print(f"{'═'*60}")
    print(f"  Simulator  : {args.sim_host}:{args.sim_port}")
    print(f"  Candidate  : {args.ref_source}")
    if args.ref_source == "llm":
        print(f"  LLM        : {args.provider}/{args.model} @ {args.host}")
    print(f"  Output     : {args.out}")
    print(f"\n  ℹ  reference_steps = steps that returned success=True in simulator")
    print(f"     (NOT LLM output — simulator is the source of truth)")

    # ── Simulator check ──────────────────────────────────────────────
    print(f"\n🔌  Connecting to ThorServer...")
    try:
        client = ThorClient(host=args.sim_host, port=args.sim_port)
        if not client.connected:
            print(f"  ❌  ThorServer not responding at {args.sim_host}:{args.sim_port}")
            print(f"      Fix: python thor_server.py")
            sys.exit(1)
        print(f"  ✅  Connected")
    except Exception as e:
        print(f"  ❌  {e}")
        sys.exit(1)

    # ── LLM check (only for llm ref source) ─────────────────────────
    if args.ref_source == "llm":
        print(f"\n🧠  Checking LLM ({args.provider}/{args.model})...")
        ok, msg = _check_llm_connection(args.provider, args.host, args.model, args.api_key)
        if ok:
            print(f"  ✅  {msg}")
        else:
            print(f"\n{'═'*60}\n  ❌  LLM CONNECTION FAILED\n{'─'*60}")
            for line in msg.splitlines():
                print(f"  {line}")
            print(f"{'═'*60}")
            print("  Tip: use --ref-source manual (no LLM needed)\n")
            sys.exit(1)

    if args.dry_run:
        print("\n  Dry-run mode — no simulator execution")
        from make_dataset import build_dataset
        samples = build_dataset()
        print(f"  Would process {len(samples)} samples")
        for s in samples[:3]:
            print(f"    [{s['task_id']}] {s['task_desc'][:50]}  ({len(s['reference_steps'])} candidate steps)")
        return

    # ── Record ───────────────────────────────────────────────────────
    t_start = time.perf_counter()
    samples, records = record_ground_truth(
        client        = client,
        ref_source    = args.ref_source,
        scenes        = args.scenes,
        planner_host  = args.host,
        planner_model = args.model,
        provider      = args.provider,
        api_key       = args.api_key,
        max_retries   = args.max_retries,
        verbose       = args.verbose,
    )
    elapsed = time.perf_counter() - t_start

    print_summary(samples, records)

    # ── Write dataset ────────────────────────────────────────────────
    if samples:
        dataset = {
            "version":         "1.0",
            "grounded":        True,
            "gt_source":       "simulator_execution",
            "candidate_source": args.ref_source,
            "generated":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "samples":         samples,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        size_kb = os.path.getsize(args.out) // 1024
        print(f"✅  Dataset saved → {args.out}  ({size_kb} KB)")

    # ── Write report ─────────────────────────────────────────────────
    report = {
        "total_tasks":    len(records),
        "task_success":   sum(1 for r in records if r.task_success),
        "total_time_s":   round(elapsed, 1),
        "tasks": [
            {
                "task_id":          r.task_id,
                "task_desc":        r.task_desc,
                "scene":            r.scene,
                "difficulty":       r.difficulty,
                "task_success":     r.task_success,
                "candidate_steps":  r.n_candidate_steps,
                "gt_steps":         len(r.gt_steps),
                "step_success_rate":round(r.step_success_rate, 3),
                "total_reward":     r.total_reward,
                "exec_latency_s":   r.exec_latency_s,
                "warnings":         r.warnings,
            }
            for r in records
        ],
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅  Report saved  → {args.report}")
    print(f"    Total time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()