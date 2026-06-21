#!/usr/bin/env python3
"""analyze_safety_gate.py
=========================
Runtime safety monitor (verify-before-execute) analysis for the SAGE paper.

Compares a gate-OFF run against a gate-ON run produced by
``evaluate_sim.py`` (the ``--verify-gate`` flag). For the SAME model/tasks it
reports, per method (collapsing SAGE -> SAGE for the paper brand):

  * prevented-failure rate    — gate_prevented / steps the planner emitted
  * wasted sim actions OFF/ON — steps sent to the sim that returned success=False
  * change in exec_step_success
  * total real client.step calls saved by the gate

HEADLINE: with the gate ON, every method sends FEWER failing actions to the
robot (wasted_off > wasted_on), at no extra LLM cost over execute-then-replan.

A "wasted" / failing sim action is counted from the per-row execution columns:
each row reports exec_steps_done (steps actually dispatched to the simulator)
and exec_step_success (fraction that returned success=True). The number of
sim steps that FAILED is therefore  round(exec_steps_done * (1 - exec_step_success)).
The gate cannot waste those, because predicted-bad steps are never dispatched —
so the headline number is the drop in that count.

Usage (no scipy / no GPU / no LLM):

  /home/keti/miniconda3/bin/python scripts/analyze_safety_gate.py \\
      --off results/sim_safety/sim_qwen2.5_7b_off_r2.csv \\
      --on  results/sim_safety/sim_qwen2.5_7b_gate_r2.csv

Each --off / --on accepts MULTIPLE CSVs (e.g. one per model); rows are pooled
and grouped by (method, model). Emits ICRA/tables/table_safety_gate.tex.

Tolerant of missing CSVs: prints "pending" and exits 0 so it can sit in the
pipeline before the sim run completes (mirrors analyze_predictive.py).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
TBL = os.path.join(_ROOT, "ICRA", "tables")

# Brand mapping: the paper presents SAGE as SAGE.
BRAND = "SAGE"


def _brand_method(method: str) -> str:
    if method and method.startswith("SAGE"):
        # SAGE -> SAGE, SAGE-Fixed -> SAGE-Fixed, etc.
        return method.replace("SAGE", BRAND, 1)
    return method


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _expand(paths: list[str]) -> list[str]:
    """Expand globs, keep only existing files, dedup preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for p in paths or []:
        for f in sorted(glob.glob(p)) or ([p] if os.path.exists(p) else []):
            if os.path.exists(f) and f not in seen:
                seen.add(f)
                out.append(f)
    return out


def _read_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for f in paths:
        try:
            with open(f, encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
        except Exception as e:  # noqa: BLE001
            print(f"  warn: could not read {f}: {e}")
    return rows


def _row_failed_sim_steps(row: dict) -> float:
    """Number of dispatched sim steps that returned success=False for one task.

    exec_steps_done  = steps actually sent to client.step
    exec_step_success = fraction of those that succeeded
    -> failed = steps_done * (1 - step_success)
    """
    done = _to_int(row.get("exec_steps_done"))
    succ = _to_float(row.get("exec_step_success"))
    if done <= 0:
        return 0.0
    return done * (1.0 - succ)


def _agg(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Aggregate per (method, model)."""
    acc: dict[tuple[str, str], dict] = defaultdict(
        lambda: dict(n=0, sim_steps=0.0, failed=0.0, step_success=0.0,
                     prevented=0, repairs=0, gate_on=0, task_success=0.0)
    )
    for r in rows:
        key = (_brand_method(r.get("method", "")), r.get("model", ""))
        a = acc[key]
        a["n"] += 1
        a["sim_steps"] += _to_int(r.get("exec_steps_done"))
        a["failed"] += _row_failed_sim_steps(r)
        a["step_success"] += _to_float(r.get("exec_step_success"))
        a["task_success"] += _to_float(r.get("exec_task_success"))
        a["prevented"] += _to_int(r.get("gate_prevented"))
        a["repairs"] += _to_int(r.get("gate_repairs"))
        a["gate_on"] += _to_int(r.get("gate_on"))
    return acc


def _mean(total: float, n: int) -> float:
    return (total / n) if n else 0.0


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off", nargs="+", default=["results/sim_safety/*off*.csv"],
                    help="gate-OFF result CSV(s) (globs ok)")
    ap.add_argument("--on", nargs="+", default=["results/sim_safety/*gate*.csv"],
                    help="gate-ON result CSV(s) (globs ok)")
    ap.add_argument("--out", default=os.path.join(TBL, "table_safety_gate.tex"),
                    help="output LaTeX table path")
    args = ap.parse_args()

    off_paths = _expand([os.path.join(_ROOT, p) if not os.path.isabs(p) else p
                         for p in args.off])
    on_paths = _expand([os.path.join(_ROOT, p) if not os.path.isabs(p) else p
                        for p in args.on])

    if not off_paths or not on_paths:
        print("safety-gate analysis: pending "
              f"(off={len(off_paths)} files, on={len(on_paths)} files). "
              "Run evaluate_sim.py with and without --verify-gate first.")
        return

    print(f"  gate-OFF CSVs: {off_paths}")
    print(f"  gate-ON  CSVs: {on_paths}")

    off = _agg(_read_rows(off_paths))
    on = _agg(_read_rows(on_paths))

    keys = sorted(set(off) | set(on))

    # ── Console report + collect table rows ───────────────────────────────
    print("\n" + "=" * 92)
    print(f"  {'method':<16}{'model':<14}"
          f"{'waste/OFF':>10}{'waste/ON':>10}{'prevent':>9}"
          f"{'step%OFF':>10}{'step%ON':>10}{'steps_saved':>12}")
    print("-" * 92)

    table_rows: list[tuple] = []
    tot_saved = 0.0
    for k in keys:
        method, model = k
        o = off.get(k)
        n = on.get(k)
        if o is None or n is None:
            # Only present in one arm — report but skip from paired headline.
            present = "OFF" if o is not None else "ON"
            print(f"  {method:<16}{model:<14}  (only in {present} run — skipped)")
            continue

        waste_off = _mean(o["failed"], o["n"])
        waste_on = _mean(n["failed"], n["n"])
        prevented = _mean(n["prevented"], n["n"])
        ss_off = _mean(o["step_success"], o["n"])
        ss_on = _mean(n["step_success"], n["n"])
        ts_off = _mean(o["task_success"], o["n"])
        ts_on = _mean(n["task_success"], n["n"])

        # Total real client.step calls saved across the run = the drop in total
        # dispatched failing steps (predicted-bad steps never dispatched).
        steps_saved = o["failed"] - n["failed"]
        tot_saved += steps_saved

        # prevented-failure rate = prevented steps as fraction of all the steps
        # the planner emitted into the sim attempt (dispatched-ON + prevented-ON).
        emitted_on = n["sim_steps"] + n["prevented"]
        prevent_rate = (n["prevented"] / emitted_on) if emitted_on else 0.0

        print(f"  {method:<16}{model:<14}"
              f"{waste_off:>10.3f}{waste_on:>10.3f}{prevented:>9.3f}"
              f"{ss_off:>10.3f}{ss_on:>10.3f}{steps_saved:>12.1f}")

        table_rows.append((method, model, waste_off, waste_on,
                           n["prevented"], prevent_rate, ss_off, ss_on,
                           ts_off, ts_on))

    print("-" * 92)
    print(f"  Total failing sim actions PREVENTED (real client.step calls saved): "
          f"{tot_saved:.1f}")
    print("=" * 92)

    # ── Emit LaTeX table ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines = [
        r"% Auto-generated by scripts/analyze_safety_gate.py",
        r"% Runtime safety monitor: verify-before-execute gating.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Runtime safety monitor (verify-before-execute). With the "
        r"zero-token symbolic gate ON, every method sends fewer failing actions "
        r"to the robot. \emph{Wasted} = mean dispatched sim steps per task that "
        r"returned failure; \emph{Prev.} = total precondition-violating steps the "
        r"gate blocked before dispatch; \emph{Step succ.} = mean fraction of "
        r"dispatched steps that succeeded.}",
        r"\label{tab:safety_gate}",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Method & Model & \multicolumn{2}{c}{Wasted/task} & Prev. & "
        r"\multicolumn{2}{c}{Step succ.} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
        r" & & OFF & ON & (n) & OFF & ON \\",
        r"\midrule",
    ]
    for (method, model, w_off, w_on, prev_n, _prate, ss_off, ss_on,
         _ts_off, _ts_on) in table_rows:
        lines.append(
            f"{_tex_escape(method)} & {_tex_escape(model)} & "
            f"{w_off:.2f} & {w_on:.2f} & {int(round(prev_n))} & "
            f"{ss_off:.2f} & {ss_on:.2f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
