"""run_benchmark.py
======================
End-to-end benchmark driver for the SAGE paper.

Runs a configurable list of planners (`--methods`) against the curated
38-task AI2-THOR evaluation set, using one or more Ollama models
(`--models`), and emits:

  * results/{run_id}/results.csv         — long-form per-(model,method,task) rows
  * results/{run_id}/aggregate.json      — per-(model,method) means
  * results/{run_id}/config.json         — full run configuration

Two evaluation modes:

  --mode plan
      Plan-quality only (no simulator).  Uses the offline metrics from
      apps/evaluate/evaluate.py: executability, precondition, redundancy,
      completeness, hallucination, plus latency/tokens.

  --mode sim
      Full execution in AI2-THOR via the existing apps/evaluate
      machinery.  Requires `thor_server.py` to be running.

Method names match pyplanner.REGISTRY keys exactly:
  Direct, CoT, Few-Shot CoT, Self-Refine, ReAct, Hierarchical,
  LLM Router, Hierarchical Few-Shot, SAGE, LLM+P, SayCan

Published-style SOTA baselines (reviewer comparison set):
  LLM+P   — NL→PDDL→pyperplan classical solve→steps (Liu et al. 2023).
            Surfaces extra metrics llmp_error / llmp_solve_s and may fall
            back to a Direct LLM call when classical solving fails.
  SayCan  — affordance-grounded greedy decoding, value(LLM)×affordance(verifier)
            (Ahn et al. 2022).
  Both route through REGISTRY automatically (they do not start with "SAGE"),
  so they need no make_planner change. Example (note the comma form for the
  space-containing names; the two SOTA names have no spaces):
    python scripts/run_benchmark.py --mode plan --models qwen2.5:7b \\
        --methods-csv "Direct,SAGE,LLM+P,SayCan" --run-id sota_demo

Ablation variants of SAGE are exposed by the special method names:
  SAGE-NoVerifier, SAGE-NoRepair, SAGE-NoMemory
which instantiate SAGEPlanner with one component disabled.

Usage
-----
  # Plan-quality benchmark on three models, default method set:
  python scripts/run_benchmark.py --mode plan \\
      --models llama3.2 qwen2.5:7b mistral-nemo \\
      --methods Direct "Few-Shot CoT" Hierarchical "Hierarchical Few-Shot" SAGE

  # With AI2-THOR (slower):
  python scripts/run_benchmark.py --mode sim --models llama3.2 \\
      --methods Direct SAGE

  # Full ablation:
  python scripts/run_benchmark.py --methods SAGE SAGE-NoVerifier \\
      SAGE-NoRepair SAGE-NoMemory

"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field

# Resolve workspace paths so the script is runnable from anywhere.
_HERE       = os.path.dirname(os.path.abspath(__file__))
_PAPER_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT  = os.path.abspath(os.path.join(_PAPER_ROOT, ".."))
_PYPLANNER  = os.path.join(_REPO_ROOT, "pyplanner")

sys.path.insert(0, _PYPLANNER)
sys.path.insert(0, os.path.join(_PYPLANNER, "apps"))

import pyplanner  # noqa: E402
from pyplanner import REGISTRY, DEFAULT_HOST  # noqa: E402
from pyplanner.sage import SAGEPlanner  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Ablation helpers
# ─────────────────────────────────────────────────────────────────────
def make_sage(model: str, host: str, gt_path: str, live_path: str,
               variant: str) -> SAGEPlanner:
    """Construct SAGE with the requested ablation flag set."""
    flags = {
        "SAGE":            dict(enable_verifier=True,  enable_local_repair=True,  enable_memory=True,  safe_refine=False),
        "SAGE-Fixed":      dict(enable_verifier=True,  enable_local_repair=True,  enable_memory=True,  safe_refine=True),
        "SAGE-NoVerifier": dict(enable_verifier=False, enable_local_repair=True,  enable_memory=True,  safe_refine=False),
        "SAGE-NoRepair":   dict(enable_verifier=True,  enable_local_repair=False, enable_memory=True,  safe_refine=False),
        "SAGE-NoMemory":   dict(enable_verifier=True,  enable_local_repair=True,  enable_memory=False, safe_refine=False),
    }[variant]
    return SAGEPlanner(
        host=host, model=model, provider="ollama",
        gt_path=gt_path if flags["enable_memory"] else "",
        live_path=live_path if flags["enable_memory"] else "",
        **flags,
    )


def make_planner(method: str, model: str, host: str,
                 gt_path: str, live_path: str):
    if method.startswith("SAGE"):
        return make_sage(model, host, gt_path, live_path, method)
    if method not in REGISTRY:
        raise SystemExit(
            f"Unknown method '{method}'. Available: "
            + ", ".join(REGISTRY) + ", SAGE-NoVerifier, SAGE-NoRepair, SAGE-NoMemory"
        )
    return REGISTRY[method](host=host, model=model, provider="ollama")


# ─────────────────────────────────────────────────────────────────────
# Plan-quality metrics (mirrors apps/evaluate/evaluate.py)
# ─────────────────────────────────────────────────────────────────────
from pyplanner.base import ROBOT_ACTIONS  # noqa: E402
from pyplanner.verifier import normalize_plan, simulate as _verifier_simulate  # noqa: E402

_INTERACT = {"Pick", "Open", "Close", "TurnOn", "TurnOff", "Wash", "Place", "PutIn"}


def _executability(steps: list[dict]) -> float:
    if not steps:
        return 0.0
    ok = sum(1 for s in steps
             if s.get("action") in ROBOT_ACTIONS and (s.get("object") or s.get("action") == "Pick"))
    return ok / len(steps)


def _precondition(steps: list[dict]) -> float:
    if not steps:
        return 0.0
    n_interact = 0
    n_ok       = 0
    seen_nav   = False
    seen_find  = False
    for s in steps:
        a = s.get("action")
        if a == "MoveTo":
            seen_nav = True
            seen_find = False
        elif a == "Find":
            seen_find = True
        if a in _INTERACT:
            n_interact += 1
            if seen_nav or seen_find:
                n_ok += 1
    return n_ok / n_interact if n_interact else 1.0


def _redundancy(steps: list[dict]) -> float:
    if len(steps) < 2:
        return 0.0
    dup = sum(
        1 for i in range(1, len(steps))
        if steps[i].get("action") == steps[i-1].get("action")
        and steps[i].get("object") == steps[i-1].get("object")
    )
    return dup / len(steps)


def _completeness(steps: list[dict], expected: list[str]) -> float:
    if not expected:
        return 1.0
    plan_text = " ".join(s.get("object", "") for s in steps).lower()
    hit = sum(1 for o in expected if o.lower() in plan_text)
    return hit / len(expected)


def _precondition_strict(steps: list[dict], visible: list[str]) -> float:
    """Strict precondition score from the symbolic verifier.

    Returns 1 - (#violations / #steps). A plan is fully valid only if EVERY
    step's preconditions hold in the simulated symbolic state — far stricter
    than the legacy heuristic in `_precondition`, which only checks that
    each interaction is preceded by *some* Navigate or Find (ignoring object
    identity, holding state, container open/close, etc.).

    Used to expose SAGE's actual advantage: baselines often score 1.0 on
    the heuristic metric but fail the strict check.
    """
    if not steps:
        return 0.0
    rep = _verifier_simulate(steps, visible_objects=visible, stop_on_error=False)
    n_viol = len(rep.violations)
    return max(0.0, 1.0 - n_viol / len(steps))


def _hallucination(steps: list[dict], visible: list[str]) -> float:
    if not steps:
        return 0.0
    vis = {v.lower() for v in (visible or [])}
    if not vis:
        return 0.0  # cannot judge
    halls = 0
    total = 0
    for s in steps:
        o = (s.get("object") or "").lower()
        if not o:
            continue
        total += 1
        # Allow rooms / furniture / containers — heuristically: anything in
        # ROBOT_ACTIONS context is fine if it matches any visible substring.
        if not any(o in v or v in o for v in vis):
            halls += 1
    return halls / total if total else 0.0


def plan_quality(steps: list[dict], sample: dict) -> dict:
    steps = normalize_plan(steps)
    ref   = sample.get("reference_steps") or []
    ratio = len(steps) / max(1, len(ref))
    visible = sample.get("visible_objects") or []
    return {
        "num_steps":           len(steps),
        "ref_steps":           len(ref),
        "step_ratio":          round(ratio, 3),
        "executability":       round(_executability(steps), 3),
        "precondition":        round(_precondition(steps), 3),
        "precondition_strict": round(_precondition_strict(steps, visible), 3),
        "redundancy":          round(_redundancy(steps), 3),
        "completeness":        round(_completeness(steps, sample.get("expected_objects") or []), 3),
        "hallucination":       round(_hallucination(steps, visible), 3),
        "parse_ok":            int(bool(steps)),
    }


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────
def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", [])


RESULT_FIELDS = [
    "run_id", "seed", "model", "method", "task_id", "difficulty", "room",
    # quality
    "num_steps", "ref_steps", "step_ratio",
    "executability", "precondition", "precondition_strict",
    "redundancy", "completeness", "hallucination", "parse_ok",
    # cost
    "latency_s", "llm_calls", "input_tokens", "output_tokens",
    "total_tokens", "tokens_per_step",
    # extras
    "subgoals", "refines", "verifier_rejections",
    # SOTA-baseline extras (LLM+P): translate/solve failure tax + solver time.
    # Blank/0 for every other method.
    "llmp_fallback", "llmp_error", "llmp_solve_s",
    "error",
]


def run_plan_mode(args, samples: list[dict], gt_path: str,
                  live_path: str, out_dir: str,
                  csv_filename: str = "results.csv") -> str:
    csv_path = os.path.join(out_dir, csv_filename)
    fieldnames = RESULT_FIELDS
    written = 0
    # Resume support: collect already-done (model, method, task_id) cells and
    # append rather than overwrite, so interruptions never cost completed work.
    done_cells: set = set()
    mode = "w"
    if getattr(args, "resume", False) and os.path.isfile(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as rf:
            for r in csv.DictReader(rf):
                done_cells.add((r.get("model"), r.get("method"), r.get("task_id")))
        mode = "a"
        print(f"[resume] {len(done_cells)} cells already done in {csv_path}")
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            w.writeheader()

        for model in args.models:
            # Pay the cold-load cost once per model so the first method
            # doesn't get a partially-loaded model returning unparseable text.
            try:
                from pyplanner.base import LLMBackend
                _warmup_backend = LLMBackend(host=args.host, model=model,
                                             provider="ollama")
                _warmup_backend.chat(
                    [{"role": "user", "content": "ok"}], temperature=0.0
                )
                print(f"\n══ warmed up model={model} ══")
            except Exception as e:
                print(f"\n══ warmup model={model} failed: {e} (continuing) ══")

            for method in args.methods:
                print(f"\n══ model={model}  method={method} ══")
                try:
                    planner = make_planner(method, model, args.host,
                                           gt_path, live_path)
                    # Multi-seed robustness: forward the run seed to the
                    # planner's Ollama backend so temperature>0 sampling is
                    # reproducible per-seed. Every planner exposes exactly one
                    # self._backend (verified), so this one assignment covers
                    # all methods including SAGE. seed=None → legacy behaviour.
                    if args.seed is not None and getattr(planner, "_backend", None) is not None:
                        planner._backend.seed = args.seed
                except Exception as e:
                    print(f"  [SKIP] could not construct: {e}")
                    continue

                for s in samples:
                    if args.max_samples and written and written % args.max_samples == 0:
                        break
                    if (model, method, s.get("task_id")) in done_cells:
                        continue
                    task = s.get("task_desc", "")
                    obs  = s.get("obs", "")
                    vis  = s.get("visible_objects", []) or []
                    row = {
                        "run_id": args.run_id,
                        "seed": args.seed if args.seed is not None else "",
                        "model": model, "method": method,
                        "task_id": s.get("task_id"), "difficulty": s.get("difficulty"),
                        "room": s.get("room"),
                    }
                    try:
                        t0 = time.perf_counter()
                        steps, metrics = planner.generate_plan(task, obs, vis)
                        elapsed = time.perf_counter() - t0
                        q = plan_quality(steps, s)
                        row.update(q)
                        m = metrics.to_dict() if metrics else {}
                        row["latency_s"]      = m.get("latency_s", round(elapsed, 3))
                        row["llm_calls"]      = m.get("llm_calls", 0)
                        row["input_tokens"]   = m.get("input_tokens", 0)
                        row["output_tokens"]  = m.get("output_tokens", 0)
                        row["total_tokens"]   = m.get("total_tokens", 0)
                        row["tokens_per_step"] = m.get("tokens_per_step", 0)
                        extra = getattr(metrics, "extra", {}) or {}
                        row["subgoals"]             = len(extra.get("subgoals", []) or [])
                        row["refines"]              = extra.get("refines", 0)
                        row["verifier_rejections"]  = extra.get("verifier_rejections", 0)
                        # SOTA-baseline (LLM+P) transparency metrics. These keys
                        # are only present in LLM+P's metrics.extra; default to
                        # 0/"" for every other method so the CSV stays aligned.
                        row["llmp_fallback"]        = int(bool(extra.get("llmp_fallback", False)))
                        row["llmp_error"]           = extra.get("llmp_error", "") or ""
                        row["llmp_solve_s"]         = extra.get("llmp_solve_s", 0)
                        row["error"] = ""
                        print(f"  {s.get('task_id')}: steps={q['num_steps']:>2} "
                              f"q={q['precondition']:.2f}/{q['executability']:.2f} "
                              f"calls={row['llm_calls']} t={row['latency_s']:.1f}s")
                    except Exception as e:
                        row.update({k: 0 for k in fieldnames if k not in row})
                        row["error"] = f"{type(e).__name__}: {e}"[:200]
                        print(f"  {s.get('task_id')}: ERROR — {row['error']}")
                    w.writerow(row)
                    f.flush()
                    written += 1
    print(f"\nWrote {written} rows → {csv_path}")
    return csv_path


def aggregate(csv_path: str, out_dir: str) -> str:
    import csv as _csv
    from collections import defaultdict
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            by_key[(row["model"], row["method"])].append(row)

    out: list[dict] = []
    metrics = ["step_ratio", "executability", "precondition", "precondition_strict",
               "redundancy", "completeness", "hallucination", "parse_ok",
               "latency_s", "llm_calls", "total_tokens", "refines",
               "verifier_rejections",
               # SOTA-baseline (LLM+P): mean fallback rate + mean solve time.
               # For non-LLM+P methods these average to 0 (always blank/0).
               "llmp_fallback", "llmp_solve_s"]
    for (model, method), rows in by_key.items():
        agg = {"model": model, "method": method, "n": len(rows)}
        for m in metrics:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r.get(m) or 0))
                except ValueError:
                    pass
            agg[m + "_mean"] = round(sum(vals) / len(vals), 3) if vals else 0.0
        agg["error_rate"] = round(
            sum(1 for r in rows if r.get("error")) / len(rows), 3
        ) if rows else 0.0
        out.append(agg)
    path = os.path.join(out_dir, "aggregate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote aggregate → {path}")
    return path


def patch_into(existing_csv: str, new_csv: str,
               pairs_to_replace: list[tuple[str, str]]) -> tuple[int, int]:
    """Splice fresh rows for (model, method) pairs into an existing CSV.

    Returns (dropped_count, added_count). A timestamped backup of the
    existing CSV is written alongside before the in-place replacement.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = f"{existing_csv}.bak.{ts}"
    import shutil as _sh
    _sh.copy2(existing_csv, backup)
    print(f"[patch] backup → {backup}")

    pairs_set = {tuple(p) for p in pairs_to_replace}
    surviving = []
    dropped   = 0
    with open(existing_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        for row in reader:
            if (row.get("model"), row.get("method")) in pairs_set:
                dropped += 1
            else:
                surviving.append(row)
    # Union schema so that new columns (e.g. precondition_strict) added
    # since the original run are preserved when patching.
    fieldnames = list(RESULT_FIELDS)
    for f_ in old_fields:
        if f_ not in fieldnames:
            fieldnames.append(f_)

    added = 0
    new_rows: list[dict] = []
    with open(new_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            new_rows.append(row)
            added += 1

    with open(existing_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in surviving:
            writer.writerow(r)
        for r in new_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    return dropped, added


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--mode", choices=["plan", "sim"], default="plan")
    p.add_argument("--dataset", default=os.path.join(_PYPLANNER, "eval_dataset_gt.json"))
    p.add_argument("--gt-path", default=os.path.join(_PYPLANNER, "eval_dataset_gt.json"),
                   help="GT file used to seed SAGE memory")
    p.add_argument("--live-path", default=os.path.join(_PAPER_ROOT, "data", "memory.jsonl"))
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--models", nargs="+", default=["llama3.2"])
    p.add_argument("--methods", nargs="+",
                   default=["Direct", "Few-Shot CoT", "Hierarchical",
                            "Hierarchical Few-Shot", "SAGE"])
    p.add_argument("--methods-csv", default="",
                   help="Comma-separated alternative to --methods. Use this when "
                        "passing method names that contain spaces (e.g. \"Few-Shot "
                        "CoT,Hierarchical Few-Shot\") so that shell word-splitting "
                        "does not break the names apart. If set, OVERRIDES --methods.")
    p.add_argument("--out-root", default=os.path.join(_PAPER_ROOT, "results"))
    p.add_argument("--run-id", default="")
    # ── sim-mode pass-through flags (forwarded to evaluate_sim.py) ──
    # Enable a one-liner ProcTHOR (or alternate-host/port) sim run without
    # editing evaluate_sim's defaults. Plan mode ignores these.
    p.add_argument("--sim-host", default="localhost",
                   help="[sim mode] ThorServer host forwarded to evaluate_sim.")
    p.add_argument("--sim-port", type=int, default=5555,
                   help="[sim mode] ThorServer ZMQ port forwarded to evaluate_sim.")
    p.add_argument("--sim-type", default="thor", choices=["thor", "procthor"],
                   help="[sim mode] Simulator family forwarded to evaluate_sim. "
                        "'procthor' resets houses by (split, house_index) read "
                        "from each sample.")
    p.add_argument("--max-replan", type=int, default=3,
                   help="[sim mode] Replan budget forwarded to evaluate_sim.")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Cap per (model, method) for quick debugging")
    p.add_argument("--task-ids", nargs="+", default=[],
                   help="If non-empty, only run on samples whose task_id is in this set.")
    p.add_argument("--seed", type=int, default=None,
                   help="Ollama sampling seed forwarded to every planner's backend. "
                        "Used for the multi-seed robustness study; runs at each "
                        "method's native temperature but with reproducible sampling. "
                        "Default None = legacy non-seeded behaviour.")
    p.add_argument("--resume", action="store_true",
                   help="If the run's results.csv already exists, append to it and "
                        "skip (model, method, task_id) cells already present. Makes "
                        "the grid robust to interruptions and lets a run on a larger "
                        "dataset reuse rows from an earlier subset run.")
    p.add_argument("--into", default="",
                   help="Patch results back into an existing run_id instead of "
                        "creating a new one. The existing CSV is backed up to "
                        "results.csv.bak.<ts>, then rows for the specified "
                        "(model, method) pairs are replaced with fresh ones.")
    args = p.parse_args()

    # --methods-csv wins if set (space-safe alternative to --methods).
    if getattr(args, "methods_csv", ""):
        args.methods = [m.strip() for m in args.methods_csv.split(",") if m.strip()]

    # ── --into branch: validate target, build pairs to replace ──
    patch_target: str | None = None
    if args.into:
        patch_target = os.path.join(args.out_root, args.into)
        if not os.path.isdir(patch_target):
            raise SystemExit(f"--into target not found: {patch_target}")
        if not os.path.isfile(os.path.join(patch_target, "results.csv")):
            raise SystemExit(f"--into target has no results.csv: {patch_target}")
        # Reuse the existing run_id so plot scripts keep working.
        args.run_id = args.into
        out_dir = patch_target
    else:
        args.run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        out_dir = os.path.join(args.out_root, args.run_id)
        os.makedirs(out_dir, exist_ok=True)

    # In --into mode, persist a patch-config record alongside (don't overwrite
    # the original config.json which describes the seminal run).
    cfg_name = f"patch_{time.strftime('%Y%m%d_%H%M%S')}.json" if args.into else "config.json"
    with open(os.path.join(out_dir, cfg_name), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    samples = load_dataset(args.dataset)
    if args.task_ids:
        wanted = set(args.task_ids)
        samples = [s for s in samples if s.get("task_id") in wanted]
        missing = wanted - {s.get("task_id") for s in samples}
        if missing:
            raise SystemExit(f"--task-ids not found in dataset: {sorted(missing)}")
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Dataset: {len(samples)} samples loaded from {args.dataset}")

    if args.mode == "sim":
        # Delegate to apps/evaluate/evaluate_sim.py rather than re-implementing.
        if args.into:
            raise SystemExit("--into is only supported with --mode plan")
        sim_script = os.path.join(_PYPLANNER, "apps", "evaluate", "evaluate_sim.py")
        if not os.path.exists(sim_script):
            raise SystemExit(f"Cannot find {sim_script}")
        import subprocess
        # evaluate_sim.py imports `thor_app.*` (the pre-rename name of apps/)
        # and the top-level `make_dataset`. Put both the pyplanner repo root
        # (so `thor_app` resolves via the thor_app->apps symlink) and apps/
        # (for the bare `make_dataset` import) on PYTHONPATH for the child.
        sim_env = dict(os.environ)
        extra_paths = [_PYPLANNER, os.path.join(_PYPLANNER, "apps")]
        sim_env["PYTHONPATH"] = os.pathsep.join(
            extra_paths + ([sim_env["PYTHONPATH"]] if sim_env.get("PYTHONPATH") else [])
        )
        for model in args.models:
            csv_out = os.path.join(out_dir, f"sim_{model.replace(':','_')}.csv")
            cmd = [sys.executable, sim_script,
                   "--model", model, "--host", args.host,
                   "--methods", *args.methods,
                   "--dataset", args.dataset,
                   "--sim-host", args.sim_host,
                   "--sim-port", str(args.sim_port),
                   "--sim-type", args.sim_type,
                   "--max-replan", str(args.max_replan),
                   "--out", csv_out]
            print(f"\n  $ {' '.join(cmd)}")
            subprocess.run(cmd, check=False, env=sim_env)
        return 0

    # Plan-quality mode.
    if patch_target:
        # Run into a temp CSV beside the existing one, then splice.
        tmp_csv = run_plan_mode(args, samples, args.gt_path,
                                args.live_path, out_dir,
                                csv_filename="results.csv.new")
        pairs = [(m, me) for m in args.models for me in args.methods]
        dropped, added = patch_into(
            os.path.join(patch_target, "results.csv"),
            tmp_csv, pairs,
        )
        os.remove(tmp_csv)
        print(f"\n[patch] replaced {dropped} stale row(s), added {added} fresh row(s)")
        print(f"[patch] patched: {[f'{m}/{me}' for m, me in pairs]}")
        aggregate(os.path.join(patch_target, "results.csv"), patch_target)
        print(f"\nDone. Patched into run id: {args.run_id}")
        print(f"Inspect:  {patch_target}")
        return 0

    csv_path = run_plan_mode(args, samples, args.gt_path,
                             args.live_path, out_dir)
    aggregate(csv_path, out_dir)
    print(f"\nDone. Run id: {args.run_id}")
    print(f"Inspect:  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
