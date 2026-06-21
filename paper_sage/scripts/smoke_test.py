"""smoke_test.py
================
Offline (no-LLM) sanity checks for SAGE's verifier, memory retriever,
and planner construction. Catches schema/import regressions in CI or
after refactors. Run before invoking any expensive LLM benchmark.

  python scripts/smoke_test.py

Exit code is 0 on success, 1 on any failure.
"""
from __future__ import annotations

import os
import sys

_HERE       = os.path.dirname(os.path.abspath(__file__))
_PAPER_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_PYPLANNER  = os.path.abspath(os.path.join(_PAPER_ROOT, "..", "pyplanner"))
sys.path.insert(0, _PYPLANNER)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    raise SystemExit(1)


def test_registry() -> None:
    import pyplanner
    assert "SAGE" in pyplanner.REGISTRY, "SAGE not in REGISTRY"
    _ok(f"REGISTRY contains SAGE among {len(pyplanner.REGISTRY)} methods")


def test_verifier_basic() -> None:
    from pyplanner.verifier import simulate, normalize_plan
    good = [
        {"action": "MoveTo", "object": "Kitchen"},
        {"action": "Find",   "object": "Apple"},
        {"action": "Pick",   "object": "Apple"},
        {"action": "MoveTo", "object": "DiningTable"},
        {"action": "Place",  "object": "DiningTable"},
    ]
    r = simulate(good)
    assert r.ok, f"expected good plan to pass, got: {r.summary()}"
    _ok("valid plan accepted")

    bad = [
        {"action": "Pick",  "object": "Apple"},
        {"action": "Place", "object": "DiningTable"},
    ]
    r = simulate(bad)
    assert not r.ok, "expected bad plan to be rejected"
    _ok(f"invalid plan rejected: {r.violations[0].code}")

    legacy = [{"action": "Navigate", "object": "Mug"},
              {"action": "Grab",     "object": "Mug"}]
    norm = normalize_plan(legacy)
    assert norm[0]["action"] == "MoveTo" and norm[1]["action"] == "Pick"
    _ok("legacy actions (Navigate/Grab) normalized")


def test_verifier_container_rule() -> None:
    from pyplanner.verifier import simulate
    # Container rule: object inside Fridge needs Open first.
    plan = [
        {"action": "MoveTo", "object": "Fridge"},
        {"action": "Find",   "object": "Fridge"},
        {"action": "Open",   "object": "Fridge"},
        {"action": "Find",   "object": "Apple"},
        {"action": "Pick",   "object": "Apple"},
        {"action": "Find",   "object": "Fridge"},
        {"action": "Close",  "object": "Fridge"},
    ]
    r = simulate(plan)
    assert r.ok, r.summary()
    _ok("container-rule plan accepted (Open before Find, Close after Pick)")


def test_memory_retriever() -> None:
    from pyplanner.memory_retriever import MemoryRetriever, MemoryRetrieverConfig
    gt = os.path.join(_PYPLANNER, "eval_dataset_gt.json")
    if not os.path.exists(gt):
        _fail(f"eval_dataset_gt.json not found at {gt}")
    mr = MemoryRetriever(MemoryRetrieverConfig(gt_path=gt, top_k=3))
    sizes = mr.size
    assert sizes["seed"] > 0, "no seed examples loaded"
    _ok(f"loaded {sizes['seed']} seed examples")
    hits = mr.retrieve("Make a cup of coffee", k=3)
    assert hits, "retrieval returned nothing"
    assert any("coffee" in h.task.lower() for h in hits), \
        "top-3 for 'make coffee' should contain a coffee task"
    _ok(f"top hit for 'make coffee' → {hits[0].task[:50]!r}")


def test_sage_construction() -> None:
    from pyplanner.sage import SAGEPlanner
    gt = os.path.join(_PYPLANNER, "eval_dataset_gt.json")
    g = SAGEPlanner(gt_path=gt, model="llama3.2", top_k=2)
    assert g.memory is not None and g.memory.size["seed"] > 0
    _ok("SAGE constructed with seeded memory")


def test_sage_ablation_flags() -> None:
    from pyplanner.sage import SAGEPlanner
    gt = os.path.join(_PYPLANNER, "eval_dataset_gt.json")
    for flag in ("enable_verifier", "enable_memory", "enable_local_repair"):
        kw = {flag: False, "gt_path": gt, "model": "llama3.2"}
        g = SAGEPlanner(**kw)
        assert getattr(g, flag) is False
        _ok(f"ablation flag {flag}=False respected")


def test_extra_actions_gated() -> None:
    """Heat/Cool/Clean are OFF by default (THOR vocab untouched) and ON only via
    PYPLANNER_EXTRA_ACTIONS, with the verifier requiring 'holding'. The env must be
    set before import, so the enabled case runs in a subprocess."""
    import subprocess
    import sys

    from pyplanner.base import ROBOT_ACTIONS, STEP_SCHEMA
    assert "Heat" not in ROBOT_ACTIONS and len(ROBOT_ACTIONS) == 14, \
        "default THOR vocab must be unchanged"
    assert "Cross-domain" not in STEP_SCHEMA
    _ok("default vocab unchanged (Heat/Cool/Clean absent)")

    code = (
        "from pyplanner.base import ROBOT_ACTIONS;"
        "from pyplanner.verifier import verify_step, SymbolicState;"
        "assert all(a in ROBOT_ACTIONS for a in ('Heat','Cool','Clean'));"
        "s=SymbolicState();"
        "ok,c,_=verify_step({'action':'Cool','object':'Bread'},s); assert not ok and c=='treat_without_holding';"
        "s.holding='Bread'; ok2,_,_=verify_step({'action':'Cool','object':'Bread'},s); assert ok2;"
        "print('subproc-ok')"
    )
    env = dict(os.environ, PYPLANNER_EXTRA_ACTIONS="Heat,Cool,Clean")
    env["PYTHONPATH"] = _PYPLANNER + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "subproc-ok" in r.stdout, \
        f"gated extra-actions test failed: {r.stderr.strip()[:200]}"
    _ok("PYPLANNER_EXTRA_ACTIONS enables Heat/Cool/Clean + 'holding' precondition")


def main() -> int:
    print("SAGE smoke test")
    print("=================")
    tests = [
        ("registry",            test_registry),
        ("verifier basic",      test_verifier_basic),
        ("verifier container",  test_verifier_container_rule),
        ("memory retriever",    test_memory_retriever),
        ("sage construction",  test_sage_construction),
        ("sage ablation",      test_sage_ablation_flags),
        ("extra-actions gated", test_extra_actions_gated),
    ]
    for name, fn in tests:
        print(f"\n[{name}]")
        fn()
    print("\nAll smoke tests passed ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
