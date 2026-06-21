"""seed_memory.py
====================
Pre-build SAGE's hybrid memory store from eval_dataset_gt.json.

The GT file contains 38 simulator-verified household plans.  This script
parses them, normalizes action vocabulary (Navigate→MoveTo, Grab→Pick),
and writes a JSONL "live pool" file that SAGE will load alongside the
canonical seed loader.  Running it is OPTIONAL — SAGE loads seeds
directly when given `gt_path` — but it's useful when you want the
warm-start memory to live in one explicit file (for reproducibility or
to extend it with hand-written examples).

Usage
-----
    python scripts/seed_memory.py \\
        --gt   ../pyplanner/eval_dataset_gt.json \\
        --out  data/memory.jsonl

    # Quick inspection of what was loaded:
    python scripts/seed_memory.py --gt ../pyplanner/eval_dataset_gt.json --dry-run

"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running without installing the workspace package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pyplanner"))

from pyplanner.memory_retriever import load_seed_examples  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--gt",  default="../pyplanner/eval_dataset_gt.json",
                   help="Path to eval_dataset_gt.json")
    p.add_argument("--out", default="data/memory.jsonl",
                   help="Output JSONL file for the live pool")
    p.add_argument("--dry-run", action="store_true",
                   help="Print loaded examples, write nothing")
    args = p.parse_args()

    seeds = load_seed_examples(args.gt)
    if not seeds:
        print(f"[ERROR] No examples loaded from {args.gt}")
        return 1
    print(f"Loaded {len(seeds)} seed examples from {args.gt}")

    if args.dry_run:
        for ex in seeds[:5]:
            print(f"  • [{ex.source}] {ex.task}")
            print(f"      plan: {ex.plan_text[:120]}{'...' if len(ex.plan_text) > 120 else ''}")
        return 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in seeds:
            f.write(json.dumps({
                "task":      ex.task,
                "reasoning": ex.reasoning,
                "plan_text": ex.plan_text,
                "source":    ex.source,
            }) + "\n")
    print(f"Wrote {len(seeds)} entries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
