#!/usr/bin/env python3
"""evaluate_alfworld.py — execute pyplanner plans on the ALFWorld TextWorld benchmark.

Purpose
-------
Cross-domain execution harness that mirrors the AI2-THOR driver in
``pyplanner/apps/evaluate/evaluate_sim.py`` but targets **ALFWorld TextWorld**
games instead of the THOR simulator. For each ALFWorld game it:

  1. resets the env and builds the initial ``obs`` + ``visible_objects`` from the
     first observation and the admissible-command set,
  2. calls ``planner.generate_plan(task, obs, visible_objects) -> (steps, PlanMetrics)``,
  3. executes the plan step-by-step: ``translate_step`` (alfworld_adapter) ->
     ``env.step([cmd])`` -> update ``AlfworldState`` from the returned observation,
  4. on a non-progress step ("Nothing happens" / no state change) calls
     ``planner.replan(...)`` with a failure_reason, capped by ``--max-replan``,
  5. (optional ``--verify-gate``) verifies each translated step's pyplanner-level
     preconditions with ``pyplanner.verifier`` BEFORE dispatch, pre-emptively
     repairing predicted violations — same semantics as evaluate_sim's gate.

Success detection
-----------------
ALFWorld's ``env.step([cmd])`` returns ``(obs, scores, dones, infos)`` (batched).
  * ``exec_task_success`` = ``dones[0]`` OR ``infos['won'][0]`` — ALFWorld's own
    goal signal. This is the authoritative metric.
  * ``exec_step_success`` = fraction of *translated* commands that produced a
    state change, i.e. the observation was NOT a "Nothing happens"-style
    rejection. (Find/idle steps that translate to None are NOT counted as
    executed commands; they are skips.)

Outputs a CSV (one row per game×method) plus an ``aggregate.json`` of per-method
means (sibling of the CSV).

Known fragilities (inherited from alfworld_adapter; the human must run-test)
----------------------------------------------------------------------------
* Find/Pick mismatch — ``Find`` is a skip; the next ``Pick`` takes the pending
  object from the *current* location. A Find of a non-co-located object mis-fires.
* ``toggle`` has no on/off direction — TurnOn and TurnOff both map to ``toggle``.
* State tracking is observation-text parsing; phrasing drift degrades it.
* ALFWorld's success is per-GAME goal achievement, not per-step reward; a plan
  can execute every step yet not satisfy the goal (exec_task_success=0 with high
  exec_step_success). Lead analysis with both, per the workspace's sim findings.

Usage
-----
    python evaluate_alfworld.py --methods SAGE Direct \
        --model llama3.2 --config configs/alfworld_base_config.yaml \
        --max-games 20 --out results/alfworld/run.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any

# ── Resolve paths so pyplanner + the sibling adapter import cleanly ──────────
_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import pyplanner  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "pyplanner"))
    import pyplanner  # noqa: F401

from pyplanner import REGISTRY, DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_BACKEND  # noqa: E402

# Sibling adapter (same directory).
try:
    from alfworld_adapter import (
        AlfworldState,
        make_alfworld_env,
        translate_step,
        _extract_ids,
    )
except ImportError:
    sys.path.insert(0, _HERE)
    from alfworld_adapter import (  # type: ignore
        AlfworldState,
        make_alfworld_env,
        translate_step,
        _extract_ids,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SAGE variant flag table — copied verbatim from evaluate_sim.py / run_benchmark
# so ALFWorld results are directly comparable to the plan-quality + THOR grids.
# ═══════════════════════════════════════════════════════════════════════════
_SAGE_FLAGS = {
    "SAGE":            dict(enable_verifier=True,  enable_local_repair=True,  enable_memory=True,  safe_refine=False),
    "SAGE-Fixed":      dict(enable_verifier=True,  enable_local_repair=True,  enable_memory=True,  safe_refine=True),
    "SAGE-NoVerifier": dict(enable_verifier=False, enable_local_repair=True,  enable_memory=True,  safe_refine=False),
    "SAGE-NoRepair":   dict(enable_verifier=True,  enable_local_repair=False, enable_memory=True,  safe_refine=False),
    "SAGE-NoMemory":   dict(enable_verifier=True,  enable_local_repair=True,  enable_memory=False, safe_refine=False),
}


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass — mirrors evaluate_sim's column philosophy where sensible
# ═══════════════════════════════════════════════════════════════════════════
CSV_COLUMNS = [
    "method", "model", "game_id", "task_type", "task_desc",
    "num_steps", "exec_task_success", "exec_step_success",
    "exec_steps_done", "exec_replans", "latency_s",
    "gate_on", "gate_prevented", "gate_repairs", "error",
]


@dataclass
class AlfworldSampleResult:
    method:            str   = ""
    model:             str   = ""
    game_id:           str   = ""
    task_type:         str   = ""
    task_desc:         str   = ""
    num_steps:         int   = 0
    exec_task_success: float = 0.0   # dones[0] / infos['won']
    exec_step_success: float = 0.0   # fraction of commands that changed state
    exec_steps_done:   int   = 0     # translated commands actually dispatched
    exec_replans:      int   = 0
    latency_s:         float = 0.0   # planning + execution wall time
    gate_on:           int   = 0
    gate_prevented:    int   = 0
    gate_repairs:      int   = 0
    error:             str   = ""

    def to_row(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# Batched-env helpers (ALFWorld init_env(batch_size=1) returns lists/tuples)
# ═══════════════════════════════════════════════════════════════════════════
def _unwrap(x: Any) -> Any:
    if isinstance(x, (list, tuple)):
        return x[0] if x else x
    return x


def _admissible(infos: Any) -> list[str]:
    if not isinstance(infos, dict):
        return []
    for key in ("admissible_commands", "admissible_actions"):
        if key in infos:
            val = _unwrap(infos[key])
            if isinstance(val, (list, tuple)):
                return [str(c) for c in val]
    return []


def _won(infos: Any) -> bool:
    if not isinstance(infos, dict):
        return False
    return bool(_unwrap(infos.get("won")))


def _is_rejection(obs: str) -> bool:
    """ALFWorld signals a non-progressing command with a 'Nothing happens'-style
    observation. Mirrors the success oracle used by the collectors."""
    low = (obs or "").lower()
    return (
        "nothing happens" in low
        or "you can't" in low
        or "can't see" in low
        or "not able" in low
        or obs is None
        or obs.strip() == ""
    )


def _parse_task_desc(obs: str) -> str:
    """ALFWorld prepends 'Your task is to: <goal>' to the first observation."""
    if not obs:
        return ""
    m = None
    low = obs.lower()
    idx = low.find("your task is to:")
    if idx >= 0:
        return obs[idx + len("your task is to:"):].strip().split("\n")[0].strip()
    return obs.strip().split("\n")[-1].strip()[:200]


def _visible_objects_from(obs: str, admissible: list[str]) -> list[str]:
    """Build a CamelCase-ish visible-objects list (what pyplanner planners expect
    for the prompt) from the initial obs + admissible 'go to'/'take' targets."""
    import re

    found: list[str] = []
    seen: set[str] = set()
    for src in [obs or ""] + list(admissible or []):
        for tok in _extract_ids(src):
            stem = re.sub(r"\s*\d+$", "", tok).strip()
            # CamelCase the stem: 'dining table' -> 'DiningTable'
            camel = "".join(w.capitalize() for w in stem.split())
            if camel and camel not in seen:
                seen.add(camel)
                found.append(camel)
    return found


# ═══════════════════════════════════════════════════════════════════════════
# Core: execute one plan in ALFWorld
# ═══════════════════════════════════════════════════════════════════════════
def execute_plan_in_alfworld(
    env,
    planner,
    task_desc: str,
    plan: list[dict],
    state: AlfworldState,
    obs: str,
    admissible: list[str],
    visible_objects: list[str],
    max_steps: int = 40,
    max_replan: int = 3,
    verbose: bool = False,
    verify_gate: bool = False,
) -> dict:
    """Translate + execute a plan step-by-step in a freshly-reset ALFWorld game.

    Returns a dict with: task_success, step_success, steps_done, replans, error,
    gate_on, gate_prevented, gate_repairs.
    """
    completed: list[dict] = []
    current_plan = list(plan)
    replan_count = 0
    cmds_dispatched = 0
    cmds_progressed = 0
    task_done = _won({}) or False
    error = ""

    # ── verify-before-execute gate setup (import-guarded, same as evaluate_sim) ─
    gate_prevented = 0
    gate_repairs = 0
    _gate_ready = False
    _vsim = _vverify = _vnstep = _vnplan = None
    if verify_gate:
        try:
            from pyplanner.verifier import (
                simulate as _vsim,
                verify_step as _vverify,
                normalize_step as _vnstep,
                normalize_plan as _vnplan,
            )
            _gate_ready = True
        except Exception as _ve:
            if verbose:
                print(f"    verify-gate disabled (verifier import failed: {_ve})")
            _gate_ready = False
    _consecutive_prevented = 0

    # Expose the latest admissible set to translate_step (it reads it off state).
    state._last_admissible = admissible  # type: ignore[attr-defined]

    steps_budget = max_steps
    try:
        while current_plan and steps_budget > 0:
            step = current_plan.pop(0)
            action = step.get("action", "Wait")
            obj = step.get("object", "")

            if verbose:
                print(f"    -> {action} {obj}".rstrip())

            # ── pre-execution verifier gate ──────────────────────────────────
            if _gate_ready:
                try:
                    _state = _vsim(
                        _vnplan(completed),
                        visible_objects=visible_objects,
                        stop_on_error=False,
                    ).final_state
                    _ok, _code, _reason = _vverify(_vnstep(step), _state)
                except Exception as _ge:
                    _ok, _code, _reason = True, None, None
                    if verbose:
                        print(f"    gate eval error ({_ge}); executing normally")

                if not _ok:
                    gate_prevented += 1
                    _consecutive_prevented += 1
                    if verbose:
                        print(f"    gate blocked: [{_code}] {_reason} "
                              f"(prevented #{gate_prevented})")
                    if (planner is not None
                            and replan_count < max_replan
                            and _consecutive_prevented <= max_replan):
                        replan_count += 1
                        gate_repairs += 1
                        try:
                            new_steps, _ = planner.replan(
                                task=task_desc,
                                completed=completed,
                                failed_step=step,
                                failure_reason=f"[pre-exec verifier] {_reason}",
                                obs=obs,
                                visible_objects=visible_objects,
                            )
                            current_plan = list(new_steps)
                        except Exception as _re:
                            if verbose:
                                print(f"    pre-emptive replan error: {_re}")
                        continue
                    else:
                        continue
                else:
                    _consecutive_prevented = 0

            # ── translate -> command ─────────────────────────────────────────
            state._last_admissible = admissible  # type: ignore[attr-defined]
            cmd = translate_step(step, state)
            if cmd is None:
                # Find / idle / un-translatable → no-op skip. Still record the
                # step as logically completed so the gate state stays in sync.
                completed.append(step)
                continue

            # ── dispatch to env ──────────────────────────────────────────────
            try:
                raw_obs, scores, dones, infos = env.step([cmd])
            except Exception as e:
                error = f"env.step EXC: {type(e).__name__}: {e}"[:150]
                break
            obs = str(_unwrap(raw_obs) or "")
            admissible = _admissible(infos)
            cmds_dispatched += 1
            steps_budget -= 1

            progressed = not _is_rejection(obs)
            if progressed:
                cmds_progressed += 1
                # update tracked state ONLY on accepted commands
                state.update(obs, admissible)
                state._last_admissible = admissible  # type: ignore[attr-defined]
                completed.append(step)

            if bool(_unwrap(dones)) or _won(infos):
                task_done = True
                break

            if not progressed:
                # Non-progress → replan the suffix.
                if replan_count < max_replan and planner is not None:
                    replan_count += 1
                    msg = obs.strip()[:80] or "Nothing happens"
                    if verbose:
                        print(f"    failed: {msg} — replanning "
                              f"({replan_count}/{max_replan})")
                    try:
                        new_steps, _ = planner.replan(
                            task=task_desc,
                            completed=completed,
                            failed_step=step,
                            failure_reason=msg,
                            obs=obs,
                            visible_objects=visible_objects,
                        )
                        current_plan = list(new_steps)
                    except Exception as e:
                        if verbose:
                            print(f"    replan error: {e}")
                        break
                else:
                    break  # budget exhausted / no planner

    except Exception as e:
        error = str(e)[:150]
        if verbose:
            traceback.print_exc()

    step_success = round(cmds_progressed / cmds_dispatched, 4) if cmds_dispatched else 0.0
    return {
        "task_success": 1.0 if task_done else 0.0,
        "step_success": step_success,
        "steps_done": cmds_dispatched,
        "replans": replan_count,
        "error": error,
        "gate_on": 1 if (verify_gate and _gate_ready) else 0,
        "gate_prevented": gate_prevented,
        "gate_repairs": gate_repairs,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Evaluate one game: reset -> plan -> execute
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_game(
    env,
    planner,
    game_idx: int,
    max_steps: int = 40,
    max_replan: int = 3,
    verbose: bool = False,
    verify_gate: bool = False,
) -> AlfworldSampleResult:
    res = AlfworldSampleResult(
        method=getattr(planner, "name", ""),
        model=getattr(planner, "model", ""),
        game_id=f"alf_{game_idx}",
    )
    t0 = time.perf_counter()

    # ── reset + build initial obs/visible_objects ─────────────────────────────
    try:
        raw_obs, infos = env.reset()
    except Exception as e:
        res.error = f"reset EXC: {type(e).__name__}: {e}"[:150]
        res.latency_s = round(time.perf_counter() - t0, 3)
        return res

    obs = str(_unwrap(raw_obs) or "")
    admissible = _admissible(infos)
    task_desc = _parse_task_desc(obs)
    res.task_desc = task_desc[:60]
    # task_type is the ALFWorld goal verb family when discoverable; best-effort.
    res.task_type = (task_desc.split()[0].lower() if task_desc else "")
    visible_objects = _visible_objects_from(obs, admissible)

    state = AlfworldState()
    state.update(obs, admissible)

    # ── plan ──────────────────────────────────────────────────────────────────
    try:
        steps, metrics = planner.generate_plan(
            task=task_desc,
            obs=obs,
            visible_objects=visible_objects,
        )
        if metrics.notes and not metrics.parse_ok:
            raise RuntimeError(f"LLM error: {metrics.notes}")
    except Exception as e:
        res.error = str(e)[:200]
        res.latency_s = round(time.perf_counter() - t0, 3)
        print(f"\n  [{res.game_id}] {res.method} plan error: {str(e)[:80]}")
        return res

    res.num_steps = len(steps)
    if verbose:
        print(f"  [{res.game_id}] task='{task_desc[:50]}'  {len(steps)} steps")

    # ── execute ────────────────────────────────────────────────────────────────
    exec_result = execute_plan_in_alfworld(
        env=env,
        planner=planner,
        task_desc=task_desc,
        plan=steps,
        state=state,
        obs=obs,
        admissible=admissible,
        visible_objects=visible_objects,
        max_steps=max_steps,
        max_replan=max_replan,
        verbose=verbose,
        verify_gate=verify_gate,
    )
    res.exec_task_success = exec_result["task_success"]
    res.exec_step_success = exec_result["step_success"]
    res.exec_steps_done = exec_result["steps_done"]
    res.exec_replans = exec_result["replans"]
    res.gate_on = exec_result["gate_on"]
    res.gate_prevented = exec_result["gate_prevented"]
    res.gate_repairs = exec_result["gate_repairs"]
    if exec_result["error"]:
        res.error = (res.error + " | " + exec_result["error"]).strip(" |")[:200]
    res.latency_s = round(time.perf_counter() - t0, 3)

    if verbose:
        status = "OK " if res.exec_task_success else "x  "
        print(f"  {status}[{res.game_id}] success={res.exec_task_success} "
              f"step_succ={res.exec_step_success} "
              f"steps={res.exec_steps_done}/{res.num_steps} "
              f"replans={res.exec_replans} {res.latency_s:.1f}s")
    return res


# ═══════════════════════════════════════════════════════════════════════════
# Planner construction — copied from evaluate_sim.py so methods match exactly
# ═══════════════════════════════════════════════════════════════════════════
def build_planners(methods, host, model, provider, api_key):
    """Instantiate planners exactly as evaluate_sim.py / run_benchmark.py do."""
    # GT file used to seed SAGE memory (same 38-task curated set), resolved
    # relative to this script: paper_sage/scripts/ -> ../../pyplanner/eval_dataset_gt.json
    _gt = os.path.join(
        os.path.dirname(os.path.dirname(_HERE)),  # remote_dir
        "pyplanner", "eval_dataset_gt.json",
    )
    planners = []
    for name in methods:
        try:
            if name.startswith("SAGE"):
                from pyplanner.sage import SAGEPlanner
                flags = _SAGE_FLAGS.get(name)
                if flags is None:
                    raise ValueError(f"unknown SAGE variant '{name}'")
                p = SAGEPlanner(
                    host=host, model=model, provider=provider, api_key=api_key,
                    gt_path=_gt if flags["enable_memory"] else "",
                    live_path="",
                    **flags,
                )
            else:
                if name not in REGISTRY:
                    raise ValueError(f"unknown method '{name}'")
                p = REGISTRY[name](host=host, model=model,
                                   provider=provider, api_key=api_key)
            planners.append(p)
            print(f"  OK  {name:16}  ({provider}/{model})")
        except Exception as e:
            print(f"  XX  {name}: {e}")
    return planners


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate + write
# ═══════════════════════════════════════════════════════════════════════════
def aggregate(results: list[AlfworldSampleResult]) -> dict:
    from collections import defaultdict

    by_method: dict[str, list[AlfworldSampleResult]] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    metrics = [
        "exec_task_success", "exec_step_success", "exec_steps_done",
        "exec_replans", "latency_s", "num_steps",
        "gate_prevented", "gate_repairs",
    ]
    out: dict[str, Any] = {}
    for method, rows in by_method.items():
        n = len(rows)
        agg = {"n_games": n,
               "n_errors": sum(1 for r in rows if r.error)}
        for m in metrics:
            vals = [getattr(r, m) for r in rows]
            agg[m + "_mean"] = round(sum(vals) / n, 4) if n else 0.0
        out[method] = agg
    return out


def _print_summary(results: list[AlfworldSampleResult]):
    agg = aggregate(results)
    print(f"\n{'-' * 96}")
    print(f"  {'Method':<16} {'Games':>6} {'TaskSucc':>9} {'StepSucc':>9} "
          f"{'Steps':>6} {'Replan':>7} {'Lat(s)':>8} {'Err':>5}")
    print(f"{'-' * 96}")
    for method, a in agg.items():
        print(f"  {method:<16} {a['n_games']:>6} "
              f"{a['exec_task_success_mean']:>9.3f} "
              f"{a['exec_step_success_mean']:>9.3f} "
              f"{a['exec_steps_done_mean']:>6.1f} "
              f"{a['exec_replans_mean']:>7.2f} "
              f"{a['latency_s_mean']:>8.1f} "
              f"{a['n_errors']:>5}")
    print(f"{'-' * 96}")


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════
def run(args) -> str:
    print(f"\n{'=' * 65}")
    print("  PyPlanner × ALFWorld (TextWorld) Execution Evaluation")
    print(f"{'=' * 65}")
    print(f"  Methods    : {args.methods}")
    print(f"  Provider   : {args.provider}  |  Model: {args.model}")
    print(f"  LLM host   : {args.host}")
    print(f"  Config     : {args.config}")
    print(f"  Max games  : {args.max_games}  |  Max steps/ep: {args.max_steps}")
    print(f"  Verify-gate: {'ON' if args.verify_gate else 'off'}")
    print(f"  Output     : {args.out}")
    print(f"{'=' * 65}")

    # ── build env (import-guarded; clear error on missing alfworld) ───────────
    try:
        env = make_alfworld_env(args.config, split="eval_out_of_distribution")
    except Exception as e:
        print(f"\n  ENV SETUP FAILED: {e}")
        print("  (alfworld/textworld must be installed and $ALFWORLD_DATA set;")
        print("   see results/analysis/alfworld_transfer_notes.md)")
        return ""

    # ── build planners ─────────────────────────────────────────────────────────
    planners = build_planners(
        args.methods, args.host, args.model, args.provider, args.api_key
    )
    if not planners:
        print("No planners initialised — aborting.")
        return ""

    # NOTE: ALFWorld's TW env cycles games on reset(); each planner re-walks the
    # same game sequence from index 0. To keep the per-method game sets aligned
    # we reset the env between planners by re-instantiating it (a fresh env
    # restarts the game cursor). Re-instantiation is cheap in text mode.
    all_results: list[AlfworldSampleResult] = []
    for pi, planner in enumerate(planners):
        if pi > 0:
            try:
                env = make_alfworld_env(args.config, split="eval_out_of_distribution")
            except Exception as e:
                print(f"  could not re-create env for {planner.name}: {e}")
                break
        print(f"\n>  [{planner.name}]  up to {args.max_games} games")
        consecutive_errors = 0
        for gi in range(args.max_games):
            res = evaluate_game(
                env=env,
                planner=planner,
                game_idx=gi,
                max_steps=args.max_steps,
                max_replan=args.max_replan,
                verbose=args.verbose,
                verify_gate=args.verify_gate,
            )
            all_results.append(res)
            if res.error:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    print(f"\n  3 consecutive errors — aborting [{planner.name}]")
                    print(f"     Last: {res.error[:80]}")
                    break
            else:
                consecutive_errors = 0
            if not args.verbose:
                done = gi + 1
                pct = done / args.max_games * 100
                bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
                print(f"\r  [{bar}] {pct:5.1f}%  {done}/{args.max_games}",
                      end="", flush=True)
        if not args.verbose:
            print()

    # ── write CSV ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r.to_row())

    # ── write aggregate.json (sibling of the CSV) ──────────────────────────────
    agg = aggregate(all_results)
    agg_path = os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "aggregate.json"
    )
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    _print_summary(all_results)
    print(f"\n  Results  -> {args.out}")
    print(f"  Aggregate-> {agg_path}")
    print(f"  Rows: {len(all_results)}  ·  Columns: {len(CSV_COLUMNS)}\n")
    return args.out


def main():
    all_methods = list(REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Execute pyplanner plans on the ALFWorld TextWorld benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--methods", nargs="+", default=all_methods)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default="ollama",
                        choices=["ollama", "openai", "gemini"])
    parser.add_argument("--api-key", default="")
    parser.add_argument("--config",
                        default="configs/alfworld_base_config.yaml",
                        help="path to alfworld base_config.yaml")
    parser.add_argument("--max-games", type=int, default=20,
                        help="number of ALFWorld games to evaluate per method")
    parser.add_argument("--max-steps", type=int, default=40,
                        help="per-episode env-step cap")
    parser.add_argument("--max-replan", type=int, default=3)
    parser.add_argument("--out", default="results/alfworld/alfworld_results.csv")
    parser.add_argument("--verify-gate", action="store_true",
                        help="Runtime safety monitor: verify each translated "
                             "step's pyplanner-level preconditions BEFORE "
                             "executing it (same semantics as evaluate_sim). "
                             "Predicted violations are repaired pre-emptively "
                             "and never dispatched. Adds gate_* columns.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if "all" in args.methods:
        args.methods = all_methods

    t_start = time.perf_counter()
    out = run(args)
    if out:
        print(f"  Total time: {time.perf_counter() - t_start:.1f}s\n")


if __name__ == "__main__":
    main()
