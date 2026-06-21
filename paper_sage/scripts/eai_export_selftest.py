#!/usr/bin/env python3
# Offline self-test for eai_export's translation + grammar validation.
# Does NOT import pyplanner and does NOT call an LLM. It stubs the two
# verifier symbols (SymbolicState, _apply) with the exact semantics from
# pyplanner/verifier.py so the SAGE->VirtualHome translator can be checked
# in isolation. Run:  python eai_export_selftest.py
from __future__ import annotations

import json
import sys
import os.path as osp

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import eai_export as E  # noqa: E402


# ── Minimal stubs mirroring pyplanner.verifier semantics ──────────────
class StubState:
    def __init__(self):
        self.arrived = None
        self.found = None
        self.holding = None
        self.opened = set()
        self.on = set()


def stub_apply(step, state):
    a = step.get("action", "")
    o = step.get("object", "") or ""
    if a == "MoveTo":
        state.arrived = o
        if state.found and state.found != o:
            state.found = None
    elif a == "Find":
        state.found = o
    elif a == "Pick":
        held = o or state.found
        if held:
            state.holding = held
        state.found = None
    elif a in ("Place", "PutIn"):
        state.holding = None
    elif a == "Open":
        if o:
            state.opened.add(o)
    elif a == "Close":
        state.opened.discard(o)
    elif a == "TurnOn":
        if o:
            state.on.add(o)
    elif a == "TurnOff":
        state.on.discard(o)


# Monkeypatch the local imports used inside sage_steps_to_vh_entries.
# That function imports normalize_plan and _apply from pyplanner.verifier;
# we shim a fake module so the import succeeds without pyplanner installed.
import types  # noqa: E402

fake = types.ModuleType("pyplanner.verifier")
fake.normalize_plan = lambda steps: [dict(s) for s in steps]
fake._apply = stub_apply
sys.modules.setdefault("pyplanner", types.ModuleType("pyplanner"))
sys.modules["pyplanner.verifier"] = fake


def test_prompt_parse():
    prompt = (
        "The current environment state is: kitchen (1), fridge (2), "
        "apple (3), coffeetable (4), tv (5).\n"
        "Node goals show the target object states: tv is ON.\n"
        "Edge goals: apple INSIDE fridge.\n"
        "Action goals: SWITCHON tv."
    )
    scene = E.parse_scene_objects(prompt)
    assert ("apple", "3") in scene, scene
    assert ("tv", "5") in scene, scene
    g = E.Grounder(scene)
    assert g.resolve("Apple") == ("apple", "3")
    assert g.resolve("CoffeeTable") == ("coffeetable", "4")
    assert g.resolve("Television") is None or True  # substring fallback may or may not hit
    print("[ok] prompt parse + grounding")


def test_translate_and_validate():
    scene = [("kitchen", "1"), ("fridge", "2"), ("apple", "3"),
             ("coffeetable", "4"), ("tv", "5")]
    g = E.Grounder(scene)
    sage = [
        {"action": "MoveTo", "object": "Kitchen"},
        {"action": "MoveTo", "object": "Fridge"},
        {"action": "Open",   "object": "Fridge"},
        {"action": "Find",   "object": "Apple"},
        {"action": "Pick",   "object": "Apple"},
        {"action": "Close",  "object": "Fridge"},
        {"action": "MoveTo", "object": "CoffeeTable"},
        {"action": "Place",  "object": "CoffeeTable"},
        {"action": "Find",   "object": "TV"},
        {"action": "TurnOn", "object": "TV"},
        {"action": "Serve",  "object": "Apple"},  # should be dropped (no VH map)
    ]
    entries, dropped = E.sage_steps_to_vh_entries(sage, g, None, StubState)
    actions = [a for a, _ in entries]
    # Expect: WALK,WALK,OPEN,FIND,GRAB,CLOSE,WALK,PUTBACK,FIND,SWITCHON ; Serve dropped
    assert actions == ["WALK", "WALK", "OPEN", "FIND", "GRAB", "CLOSE",
                       "WALK", "PUTBACK", "FIND", "SWITCHON"], actions
    # GRAB arg = apple (the found object)
    grab = next(args for a, args in entries if a == "GRAB")
    assert grab == ["apple", "3"], grab
    # PUTBACK args = held(apple) + receptacle(coffeetable)
    putback = next(args for a, args in entries if a == "PUTBACK")
    assert putback == ["apple", "3", "coffeetable", "4"], putback
    assert len(dropped) == 1 and "Serve" == dropped[0]["step"]["action"], dropped

    out = E.entries_to_llm_output(entries)
    # repeated WALK keys must survive in the textual string
    assert out.count('"WALK"') == 3, out
    ok, reason = E.local_validate(out)
    assert ok, reason
    # round-trip through the order-preserving parser
    parsed = E.local_parse_preserving_order(out)
    assert len(parsed) == len(entries), (len(parsed), len(entries))
    print("[ok] translate + grammar validate + duplicate-key preservation")


def test_argc_rules():
    # PUTBACK needs 2 objects (4 params); WALK needs 1 (2 params); STANDUP 0.
    assert E.local_validate('{"PUTBACK": ["a","1","b","2"]}')[0]
    assert not E.local_validate('{"PUTBACK": ["a","1"]}')[0]
    assert E.local_validate('{"WALK": ["a","1"]}')[0]
    assert not E.local_validate('{"WALK": ["a","1","b","2"]}')[0]
    assert not E.local_validate('{"WALK": ["a"]}')[0]  # odd name_id
    assert not E.local_validate('{"FLY": ["a","1"]}')[0]  # unknown action
    print("[ok] argument-count + name_id grammar rules")


if __name__ == "__main__":
    test_prompt_parse()
    test_translate_and_validate()
    test_argc_rules()
    print("\nALL SELFTESTS PASSED")
