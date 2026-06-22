"""
make_dataset.py
===============
Build a static evaluation dataset for pyplanner benchmarking.

Each sample contains:
  - task description + room context
  - simulated environment observation
  - visible objects in the scene
  - reference (gold-standard) action plan
  - expected key objects the plan must address
  - difficulty tag (easy / medium / hard)
  - optional failure injection for replan evaluation

Output: eval_dataset.json  (default)

Usage:
    python make_dataset.py
    python make_dataset.py --out my_dataset.json
    python make_dataset.py --summary        # print stats only
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field


# ── Valid action vocabulary (must match pyplanner) ────────────────────
VALID_ACTIONS = {
    "MoveTo", "Find", "Pick", "Place", "PutIn",
    "Open", "Close", "TurnOn", "TurnOff",
    "Wash", "Sit", "LieOn", "Serve", "Wait",
}


@dataclass
class Step:
    action: str
    object: str
    target: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FailInjection:
    """Simulates a step failure for replan evaluation."""
    at_step:        int    # 0-based index of the step that fails
    failure_reason: str    # error message returned by the environment


@dataclass
class EvalSample:
    task_id:          str
    task_desc:        str
    room:             str          # kitchen | living_room | bedroom | bathroom
    scene:            str          # FloorPlan<N>
    obs:              str          # simulated environment observation text
    visible_objects:  list[str]
    reference_steps:  list[Step]
    expected_objects: list[str]    # objects a valid plan MUST reference
    difficulty:       str          # easy | medium | hard
    fail_injection:   FailInjection | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reference_steps"] = [s for s in d["reference_steps"]]
        if d["fail_injection"] is None:
            d["fail_injection"] = {}
        return d


# ═════════════════════════════════════════════════════════════════════
# DATASET DEFINITION
# All 51 samples across 4 rooms × 3 difficulty levels
# ═════════════════════════════════════════════════════════════════════

def _s(action, obj, target="", reason="") -> dict:
    return {"action": action, "object": obj, "target": target, "reason": reason}


def _fail(at_step: int, reason: str) -> dict:
    return {"at_step": at_step, "failure_reason": reason}


SAMPLES_RAW: list[dict] = [

    # ── KITCHEN — easy ────────────────────────────────────────────────
    {
        "task_id":   "K01",
        "task_desc": "Make a cup of coffee using the coffee machine",
        "room":      "kitchen",
        "scene":     "FloorPlan1",
        "obs":       "Kitchen. Counter top visible. Coffee machine on counter. Mug on shelf.",
        "visible_objects": ["coffee_machine", "mug", "counter_top", "shelf", "fridge"],
        "reference_steps": [
            _s("MoveTo", "mug",            reason="Move to the mug"),
            _s("Pick",     "mug",            reason="Pick up the mug"),
            _s("MoveTo", "coffee_machine", reason="Move to the coffee machine"),
            _s("Place",    "mug",  "coffee_machine", "Position mug under dispenser"),
            _s("TurnOn",   "coffee_machine", reason="Start brewing"),
        ],
        "expected_objects": ["mug", "coffee_machine"],
        "difficulty": "easy",
    },
    {
        "task_id":   "K02",
        "task_desc": "Turn on the stove burner",
        "room":      "kitchen",
        "scene":     "FloorPlan2",
        "obs":       "Kitchen with stove top. Burner is off. Pot nearby.",
        "visible_objects": ["stove_burner", "pot", "counter_top"],
        "reference_steps": [
            _s("MoveTo", "stove_burner", reason="Move to the stove"),
            _s("TurnOn",   "stove_burner", reason="Switch burner on"),
        ],
        "expected_objects": ["stove_burner"],
        "difficulty": "easy",
    },
    {
        "task_id":   "K03",
        "task_desc": "Open the fridge and retrieve a tomato",
        "room":      "kitchen",
        "scene":     "FloorPlan1",
        "obs":       "Kitchen. Fridge on the wall. Tomato visible inside fridge.",
        "visible_objects": ["fridge", "counter_top", "tomato"],
        "reference_steps": [
            _s("MoveTo", "fridge",  reason="Move to the fridge"),
            _s("Open",     "fridge",  reason="Open fridge door"),
            _s("Pick",     "tomato",  reason="Pick up the tomato"),
            _s("Close",    "fridge",  reason="Close fridge door"),
        ],
        "expected_objects": ["fridge", "tomato"],
        "difficulty": "easy",
    },

    # ── KITCHEN — medium ──────────────────────────────────────────────
    {
        "task_id":   "K04",
        "task_desc": "Boil water in a pot on the stove",
        "room":      "kitchen",
        "scene":     "FloorPlan2",
        "obs":       "Kitchen. Stove visible. Pot on counter. Sink accessible.",
        "visible_objects": ["pot", "stove_burner", "sink", "faucet", "counter_top"],
        "reference_steps": [
            _s("MoveTo", "pot",          reason="Move to the pot"),
            _s("Pick",     "pot",          reason="Pick up the pot"),
            _s("MoveTo", "sink",         reason="Go to the sink"),
            _s("TurnOn",   "faucet",       reason="Fill pot with water"),
            _s("TurnOff",  "faucet",       reason="Stop water"),
            _s("MoveTo", "stove_burner", reason="Move to the stove"),
            _s("Place",    "pot", "stove_burner", "Put pot on burner"),
            _s("TurnOn",   "stove_burner", reason="Heat the water"),
        ],
        "expected_objects": ["pot", "faucet", "stove_burner"],
        "difficulty": "medium",
        "fail_injection": _fail(3, "Faucet not reachable from current position"),
    },
    {
        "task_id":   "K05",
        "task_desc": "Cook an egg — place it in a pan on the stove",
        "room":      "kitchen",
        "scene":     "FloorPlan3",
        "obs":       "Kitchen. Stove with pan. Egg on counter.",
        "visible_objects": ["egg", "pan", "stove_burner", "counter_top"],
        "reference_steps": [
            _s("MoveTo", "egg",          reason="Move to the egg"),
            _s("Pick",     "egg",          reason="Pick up the egg"),
            _s("MoveTo", "pan",          reason="Move to the pan"),
            _s("Place",    "egg", "pan",   "Place egg in pan"),
            _s("MoveTo", "stove_burner", reason="Go to stove"),
            _s("TurnOn",   "stove_burner", reason="Start cooking"),
        ],
        "expected_objects": ["egg", "pan", "stove_burner"],
        "difficulty": "medium",
    },
    {
        "task_id":   "K06",
        "task_desc": "Wash an apple at the sink",
        "room":      "kitchen",
        "scene":     "FloorPlan1",
        "obs":       "Kitchen. Apple on counter. Sink with faucet accessible.",
        "visible_objects": ["apple", "sink", "faucet", "counter_top"],
        "reference_steps": [
            _s("MoveTo", "apple",  reason="Move to the apple"),
            _s("Pick",     "apple",  reason="Pick up the apple"),
            _s("MoveTo", "sink",   reason="Go to the sink"),
            _s("TurnOn",   "faucet", reason="Turn on water"),
            _s("Wash",     "apple",  reason="Wash the apple under water"),
            _s("TurnOff",  "faucet", reason="Turn off water"),
        ],
        "expected_objects": ["apple", "sink", "faucet"],
        "difficulty": "medium",
    },
    {
        "task_id":   "K07",
        "task_desc": "Heat food in the microwave",
        "room":      "kitchen",
        "scene":     "FloorPlan1",
        "obs":       "Kitchen. Microwave on counter. Plate with food on table.",
        "visible_objects": ["microwave", "plate", "food", "counter_top"],
        "reference_steps": [
            _s("MoveTo", "plate",     reason="Move to the plate"),
            _s("Pick",     "plate",     reason="Pick up the plate"),
            _s("MoveTo", "microwave", reason="Move to the microwave"),
            _s("Open",     "microwave", reason="Open microwave door"),
            _s("Place",    "plate", "microwave", "Put plate inside"),
            _s("Close",    "microwave", reason="Close the door"),
            _s("TurnOn",   "microwave", reason="Start heating"),
        ],
        "expected_objects": ["plate", "microwave"],
        "difficulty": "medium",
        "fail_injection": _fail(2, "Microwave door is already open"),
    },

    # ── KITCHEN — hard ────────────────────────────────────────────────
    {
        "task_id":   "K08",
        "task_desc": "Set up the dining table with plates and cups",
        "room":      "kitchen",
        "scene":     "FloorPlan4",
        "obs":       "Kitchen with dining table. Plates in cabinet. Cups on shelf.",
        "visible_objects": ["plate", "cup", "cabinet", "shelf", "dining_table"],
        "reference_steps": [
            _s("MoveTo", "cabinet",      reason="Go to the cabinet"),
            _s("Open",     "cabinet",      reason="Open cabinet"),
            _s("Pick",     "plate",        reason="Take a plate"),
            _s("MoveTo", "dining_table", reason="Go to the table"),
            _s("Place",    "plate", "dining_table", "Set plate on table"),
            _s("MoveTo", "shelf",        reason="Go to the shelf"),
            _s("Pick",     "cup",          reason="Take a cup"),
            _s("MoveTo", "dining_table", reason="Return to the table"),
            _s("Place",    "cup", "dining_table", "Set cup on table"),
            _s("MoveTo", "cabinet",      reason="Close cabinet"),
            _s("Close",    "cabinet",      reason="Close the cabinet"),
        ],
        "expected_objects": ["plate", "cup", "dining_table"],
        "difficulty": "hard",
        "fail_injection": _fail(1, "Cabinet is locked"),
    },
    {
        "task_id":   "K09",
        "task_desc": "Prepare a coffee and place it on the dining table",
        "room":      "kitchen",
        "scene":     "FloorPlan1",
        "obs":       "Kitchen. Coffee machine, mug, dining table all visible.",
        "visible_objects": ["coffee_machine", "mug", "dining_table", "counter_top"],
        "reference_steps": [
            _s("MoveTo", "mug",            reason="Go to mug"),
            _s("Pick",     "mug",            reason="Pick up mug"),
            _s("MoveTo", "coffee_machine", reason="Go to coffee machine"),
            _s("Place",    "mug", "coffee_machine", "Position mug"),
            _s("TurnOn",   "coffee_machine", reason="Brew coffee"),
            _s("Pick",     "mug",            reason="Pick up filled mug"),
            _s("MoveTo", "dining_table",   reason="Go to dining table"),
            _s("Serve",    "mug", "dining_table", "Place coffee on table"),
        ],
        "expected_objects": ["mug", "coffee_machine", "dining_table"],
        "difficulty": "hard",
    },

    # ── LIVING ROOM — easy ────────────────────────────────────────────
    {
        "task_id":   "L01",
        "task_desc": "Turn on the television",
        "room":      "living_room",
        "scene":     "FloorPlan201",
        "obs":       "Living room. TV mounted on wall. Remote on coffee table.",
        "visible_objects": ["television", "remote_control", "sofa", "coffee_table"],
        "reference_steps": [
            _s("MoveTo", "television", reason="Move to the TV"),
            _s("TurnOn",   "television", reason="Switch TV on"),
        ],
        "expected_objects": ["television"],
        "difficulty": "easy",
    },
    {
        "task_id":   "L02",
        "task_desc": "Sit on the sofa",
        "room":      "living_room",
        "scene":     "FloorPlan201",
        "obs":       "Living room with sofa. Pillows on the sofa.",
        "visible_objects": ["sofa", "pillow", "coffee_table"],
        "reference_steps": [
            _s("MoveTo", "sofa", reason="Move to the sofa"),
            _s("Sit",      "sofa", reason="Sit down"),
        ],
        "expected_objects": ["sofa"],
        "difficulty": "easy",
    },
    {
        "task_id":   "L03",
        "task_desc": "Pick up a book from the bookshelf",
        "room":      "living_room",
        "scene":     "FloorPlan201",
        "obs":       "Living room. Bookshelf with books on the wall.",
        "visible_objects": ["bookshelf", "book", "sofa", "lamp"],
        "reference_steps": [
            _s("MoveTo", "bookshelf", reason="Go to bookshelf"),
            _s("Pick",     "book",      reason="Pick up a book"),
        ],
        "expected_objects": ["bookshelf", "book"],
        "difficulty": "easy",
    },

    # ── LIVING ROOM — medium ──────────────────────────────────────────
    {
        "task_id":   "L04",
        "task_desc": "Watch TV — turn it on and sit on the sofa",
        "room":      "living_room",
        "scene":     "FloorPlan201",
        "obs":       "Living room. TV off. Sofa in center. Remote on table.",
        "visible_objects": ["television", "remote_control", "sofa", "coffee_table"],
        "reference_steps": [
            _s("MoveTo", "television", reason="Move to TV"),
            _s("TurnOn",   "television", reason="Turn TV on"),
            _s("MoveTo", "sofa",       reason="Move to sofa"),
            _s("Sit",      "sofa",       reason="Sit down to watch"),
        ],
        "expected_objects": ["television", "sofa"],
        "difficulty": "medium",
    },
    {
        "task_id":   "L05",
        "task_desc": "Read a book — pick one up and sit on the sofa",
        "room":      "living_room",
        "scene":     "FloorPlan201",
        "obs":       "Living room. Bookshelf on wall. Sofa visible.",
        "visible_objects": ["book", "bookshelf", "sofa", "lamp"],
        "reference_steps": [
            _s("MoveTo", "bookshelf", reason="Go to bookshelf"),
            _s("Pick",     "book",      reason="Pick up a book"),
            _s("MoveTo", "sofa",      reason="Go to the sofa"),
            _s("Sit",      "sofa",      reason="Sit to read"),
        ],
        "expected_objects": ["book", "sofa"],
        "difficulty": "medium",
        "fail_injection": _fail(1, "Book out of reach from current position"),
    },
    {
        "task_id":   "L06",
        "task_desc": "Turn on the lamp and sit on the sofa",
        "room":      "living_room",
        "scene":     "FloorPlan202",
        "obs":       "Living room. Floor lamp in corner. Sofa in center.",
        "visible_objects": ["lamp", "sofa", "coffee_table", "television"],
        "reference_steps": [
            _s("MoveTo", "lamp", reason="Move to the lamp"),
            _s("TurnOn",   "lamp", reason="Turn lamp on"),
            _s("MoveTo", "sofa", reason="Go to the sofa"),
            _s("Sit",      "sofa", reason="Sit down"),
        ],
        "expected_objects": ["lamp", "sofa"],
        "difficulty": "medium",
    },

    # ── LIVING ROOM — hard ────────────────────────────────────────────
    {
        "task_id":   "L07",
        "task_desc": "Clean the living room — pick up clutter and place items away",
        "room":      "living_room",
        "scene":     "FloorPlan202",
        "obs":       "Messy living room. Book on floor. Remote on sofa. Cup on table.",
        "visible_objects": ["book", "remote_control", "cup", "bookshelf",
                            "coffee_table", "sofa"],
        "reference_steps": [
            _s("MoveTo", "book",           reason="Go to book on floor"),
            _s("Pick",     "book",           reason="Pick up book"),
            _s("MoveTo", "bookshelf",      reason="Go to bookshelf"),
            _s("Place",    "book", "bookshelf", "Put book away"),
            _s("MoveTo", "remote_control", reason="Go to remote on sofa"),
            _s("Pick",     "remote_control", reason="Pick up remote"),
            _s("MoveTo", "coffee_table",   reason="Go to coffee table"),
            _s("Place",    "remote_control", "coffee_table", "Place remote on table"),
        ],
        "expected_objects": ["book", "remote_control", "bookshelf"],
        "difficulty": "hard",
        "fail_injection": _fail(3, "Bookshelf is full, cannot place book"),
    },
    {
        "task_id":   "L08",
        "task_desc": "Set up for movie night: turn on TV, dim lamp, sit on sofa",
        "room":      "living_room",
        "scene":     "FloorPlan201",
        "obs":       "Living room. TV off. Lamp on. Sofa in center.",
        "visible_objects": ["television", "lamp", "sofa", "remote_control"],
        "reference_steps": [
            _s("MoveTo", "television", reason="Go to TV"),
            _s("TurnOn",   "television", reason="Turn TV on"),
            _s("MoveTo", "lamp",       reason="Go to lamp"),
            _s("TurnOff",  "lamp",       reason="Dim the room"),
            _s("MoveTo", "sofa",       reason="Go to sofa"),
            _s("Sit",      "sofa",       reason="Sit down to watch"),
        ],
        "expected_objects": ["television", "lamp", "sofa"],
        "difficulty": "hard",
    },

    # ── BEDROOM — easy ────────────────────────────────────────────────
    {
        "task_id":   "B01",
        "task_desc": "Turn off the bedroom light",
        "room":      "bedroom",
        "scene":     "FloorPlan301",
        "obs":       "Bedroom. Ceiling light on. Bed in center.",
        "visible_objects": ["light_switch", "bed", "dresser"],
        "reference_steps": [
            _s("MoveTo", "light_switch", reason="Go to light switch"),
            _s("TurnOff",  "light_switch", reason="Turn off the light"),
        ],
        "expected_objects": ["light_switch"],
        "difficulty": "easy",
    },
    {
        "task_id":   "B02",
        "task_desc": "Lie down on the bed",
        "room":      "bedroom",
        "scene":     "FloorPlan301",
        "obs":       "Bedroom. Bed with pillows. Dresser on the side.",
        "visible_objects": ["bed", "pillow", "dresser", "alarm_clock"],
        "reference_steps": [
            _s("MoveTo", "bed",   reason="Move to the bed"),
            _s("LieOn",    "bed",   reason="Lie down on the bed"),
        ],
        "expected_objects": ["bed"],
        "difficulty": "easy",
    },
    {
        "task_id":   "B03",
        "task_desc": "Pick up the alarm clock",
        "room":      "bedroom",
        "scene":     "FloorPlan301",
        "obs":       "Bedroom. Alarm clock on nightstand.",
        "visible_objects": ["alarm_clock", "bed", "nightstand"],
        "reference_steps": [
            _s("MoveTo", "alarm_clock", reason="Move to alarm clock"),
            _s("Pick",     "alarm_clock", reason="Pick it up"),
        ],
        "expected_objects": ["alarm_clock"],
        "difficulty": "easy",
    },

    # ── BEDROOM — medium ──────────────────────────────────────────────
    {
        "task_id":   "B04",
        "task_desc": "Go to sleep — turn off light and lie on the bed",
        "room":      "bedroom",
        "scene":     "FloorPlan301",
        "obs":       "Bedroom at night. Light is on. Bed unmade.",
        "visible_objects": ["light_switch", "bed", "pillow", "alarm_clock"],
        "reference_steps": [
            _s("MoveTo", "light_switch", reason="Go to light switch"),
            _s("TurnOff",  "light_switch", reason="Turn light off"),
            _s("MoveTo", "bed",          reason="Go to the bed"),
            _s("LieOn",    "bed",          reason="Lie down to sleep"),
        ],
        "expected_objects": ["light_switch", "bed"],
        "difficulty": "medium",
    },
    {
        "task_id":   "B05",
        "task_desc": "Set the alarm clock on the nightstand",
        "room":      "bedroom",
        "scene":     "FloorPlan301",
        "obs":       "Bedroom. Alarm clock on bed. Nightstand next to bed.",
        "visible_objects": ["alarm_clock", "bed", "nightstand"],
        "reference_steps": [
            _s("MoveTo", "alarm_clock", reason="Go to alarm clock"),
            _s("Pick",     "alarm_clock", reason="Pick up alarm clock"),
            _s("MoveTo", "nightstand",  reason="Go to nightstand"),
            _s("Place",    "alarm_clock", "nightstand", "Place on nightstand"),
        ],
        "expected_objects": ["alarm_clock", "nightstand"],
        "difficulty": "medium",
        "fail_injection": _fail(2, "Nightstand surface is occupied"),
    },
    {
        "task_id":   "B06",
        "task_desc": "Get clothes from the dresser",
        "room":      "bedroom",
        "scene":     "FloorPlan302",
        "obs":       "Bedroom. Dresser with drawers. Clothes inside.",
        "visible_objects": ["dresser", "clothes", "bed", "mirror"],
        "reference_steps": [
            _s("MoveTo", "dresser", reason="Go to dresser"),
            _s("Open",     "dresser", reason="Open drawer"),
            _s("Pick",     "clothes", reason="Take clothes out"),
            _s("Close",    "dresser", reason="Close drawer"),
        ],
        "expected_objects": ["dresser", "clothes"],
        "difficulty": "medium",
    },

    # ── BEDROOM — hard ────────────────────────────────────────────────
    {
        "task_id":   "B07",
        "task_desc": "Prepare for bed: set alarm, turn off light, lie down",
        "room":      "bedroom",
        "scene":     "FloorPlan301",
        "obs":       "Bedroom. Light on. Alarm clock on shelf. Bed visible.",
        "visible_objects": ["alarm_clock", "light_switch", "bed", "nightstand", "shelf"],
        "reference_steps": [
            _s("MoveTo", "alarm_clock",  reason="Go to alarm clock"),
            _s("Pick",     "alarm_clock",  reason="Pick up alarm clock"),
            _s("MoveTo", "nightstand",   reason="Go to nightstand"),
            _s("Place",    "alarm_clock", "nightstand", "Set clock down"),
            _s("MoveTo", "light_switch", reason="Go to light switch"),
            _s("TurnOff",  "light_switch", reason="Turn off light"),
            _s("MoveTo", "bed",          reason="Go to bed"),
            _s("LieOn",    "bed",          reason="Lie down"),
        ],
        "expected_objects": ["alarm_clock", "light_switch", "bed"],
        "difficulty": "hard",
    },
    {
        "task_id":   "B08",
        "task_desc": "Make the bed and place the pillow neatly",
        "room":      "bedroom",
        "scene":     "FloorPlan302",
        "obs":       "Bedroom. Pillow on the floor. Bed unmade.",
        "visible_objects": ["pillow", "bed", "dresser"],
        "reference_steps": [
            _s("MoveTo", "pillow",     reason="Go to pillow on floor"),
            _s("Pick",     "pillow",     reason="Pick up pillow"),
            _s("MoveTo", "bed",        reason="Go to the bed"),
            _s("Place",    "pillow", "bed", "Place pillow on bed"),
        ],
        "expected_objects": ["pillow", "bed"],
        "difficulty": "hard",
        "fail_injection": _fail(0, "Pillow is under the bed, not reachable"),
    },

    # ── BATHROOM — easy ───────────────────────────────────────────────
    {
        "task_id":   "A01",
        "task_desc": "Turn on the shower",
        "room":      "bathroom",
        "scene":     "FloorPlan402",
        "obs":       "Bathroom. Shower stall. Faucet handle visible.",
        "visible_objects": ["shower_faucet", "toilet", "sink", "towel"],
        "reference_steps": [
            _s("MoveTo", "shower_faucet", reason="Go to shower"),
            _s("TurnOn",   "shower_faucet", reason="Turn shower on"),
        ],
        "expected_objects": ["shower_faucet"],
        "difficulty": "easy",
    },
    {
        "task_id":   "A02",
        "task_desc": "Pick up the soap from the sink",
        "room":      "bathroom",
        "scene":     "FloorPlan401",
        "obs":       "Bathroom. Sink with soap dispenser. Mirror above.",
        "visible_objects": ["soap", "sink", "mirror", "towel"],
        "reference_steps": [
            _s("MoveTo", "sink", reason="Go to the sink"),
            _s("Pick",     "soap", reason="Pick up soap"),
        ],
        "expected_objects": ["soap", "sink"],
        "difficulty": "easy",
    },
    {
        "task_id":   "A03",
        "task_desc": "Turn on the bathroom light",
        "room":      "bathroom",
        "scene":     "FloorPlan401",
        "obs":       "Bathroom. Light switch next to door. Light is off.",
        "visible_objects": ["light_switch", "sink", "toilet", "towel_rack"],
        "reference_steps": [
            _s("MoveTo", "light_switch", reason="Go to switch"),
            _s("TurnOn",   "light_switch", reason="Turn light on"),
        ],
        "expected_objects": ["light_switch"],
        "difficulty": "easy",
    },

    # ── BATHROOM — medium ─────────────────────────────────────────────
    {
        "task_id":   "A04",
        "task_desc": "Wash hands with soap at the sink",
        "room":      "bathroom",
        "scene":     "FloorPlan401",
        "obs":       "Bathroom. Sink with faucet. Soap on counter.",
        "visible_objects": ["sink", "faucet", "soap", "towel"],
        "reference_steps": [
            _s("MoveTo", "sink",   reason="Go to sink"),
            _s("TurnOn",   "faucet", reason="Turn on water"),
            _s("Pick",     "soap",   reason="Pick up soap"),
            _s("Wash",     "soap",   reason="Wash hands with soap"),
            _s("TurnOff",  "faucet", reason="Turn off water"),
        ],
        "expected_objects": ["sink", "faucet", "soap"],
        "difficulty": "medium",
        "fail_injection": _fail(1, "Faucet handle is stuck"),
    },
    {
        "task_id":   "A05",
        "task_desc": "Brush teeth — toothbrush, use sink, brush",
        "room":      "bathroom",
        "scene":     "FloorPlan401",
        "obs":       "Bathroom. Toothbrush on counter. Sink accessible.",
        "visible_objects": ["toothbrush", "toothpaste", "sink", "faucet", "mirror"],
        "reference_steps": [
            _s("MoveTo", "toothbrush", reason="Go to toothbrush"),
            _s("Pick",     "toothbrush", reason="Pick up toothbrush"),
            _s("MoveTo", "sink",       reason="Go to sink"),
            _s("TurnOn",   "faucet",     reason="Turn on water"),
            _s("Wash",     "toothbrush", reason="Wet the toothbrush"),
            _s("TurnOff",  "faucet",     reason="Turn off water"),
        ],
        "expected_objects": ["toothbrush", "sink", "faucet"],
        "difficulty": "medium",
    },
    {
        "task_id":   "A06",
        "task_desc": "Take a towel and hang it on the rack",
        "room":      "bathroom",
        "scene":     "FloorPlan401",
        "obs":       "Bathroom. Towel on the floor. Towel rack on wall.",
        "visible_objects": ["towel", "towel_rack", "sink", "toilet"],
        "reference_steps": [
            _s("MoveTo", "towel",      reason="Go to towel"),
            _s("Pick",     "towel",      reason="Pick up towel"),
            _s("MoveTo", "towel_rack", reason="Go to towel rack"),
            _s("Place",    "towel", "towel_rack", "Hang towel on rack"),
        ],
        "expected_objects": ["towel", "towel_rack"],
        "difficulty": "medium",
    },

    # ── BATHROOM — hard ───────────────────────────────────────────────
    {
        "task_id":   "A07",
        "task_desc": "Full morning routine: light on, wash hands, brush teeth",
        "room":      "bathroom",
        "scene":     "FloorPlan401",
        "obs":       "Bathroom. Light off. Toothbrush and soap on counter. Sink ready.",
        "visible_objects": ["light_switch", "toothbrush", "soap", "sink",
                            "faucet", "mirror", "towel"],
        "reference_steps": [
            _s("MoveTo", "light_switch", reason="Go to switch"),
            _s("TurnOn",   "light_switch", reason="Turn on light"),
            _s("MoveTo", "sink",         reason="Go to sink"),
            _s("TurnOn",   "faucet",       reason="Start water for handwashing"),
            _s("Pick",     "soap",         reason="Pick up soap"),
            _s("Wash",     "soap",         reason="Wash hands"),
            _s("TurnOff",  "faucet",       reason="Stop water"),
            _s("MoveTo", "toothbrush",   reason="Go to toothbrush"),
            _s("Pick",     "toothbrush",   reason="Pick up toothbrush"),
            _s("MoveTo", "sink",         reason="Return to sink"),
            _s("TurnOn",   "faucet",       reason="Start water for brushing"),
            _s("Wash",     "toothbrush",   reason="Wet the brush"),
            _s("TurnOff",  "faucet",       reason="Stop water"),
        ],
        "expected_objects": ["light_switch", "soap", "toothbrush", "faucet"],
        "difficulty": "hard",
        "fail_injection": _fail(3, "Soap dispenser is empty"),
    },
    {
        "task_id":   "A08",
        "task_desc": "Prepare the shower: turn on, adjust, place towel nearby",
        "room":      "bathroom",
        "scene":     "FloorPlan402",
        "obs":       "Bathroom. Shower stall. Towel on rack. Mat on floor.",
        "visible_objects": ["shower_faucet", "towel", "towel_rack", "bath_mat"],
        "reference_steps": [
            _s("MoveTo", "towel",         reason="Go to towel"),
            _s("Pick",     "towel",         reason="Take towel"),
            _s("MoveTo", "bath_mat",      reason="Go near shower"),
            _s("Place",    "towel", "bath_mat", "Place towel near shower"),
            _s("MoveTo", "shower_faucet", reason="Go to shower"),
            _s("TurnOn",   "shower_faucet", reason="Turn shower on"),
        ],
        "expected_objects": ["shower_faucet", "towel"],
        "difficulty": "hard",
    },

    # ── CROSS-ROOM edge cases ─────────────────────────────────────────
    {
        "task_id":   "X01",
        "task_desc": "Put an apple in the microwave and heat it",
        "room":      "kitchen",
        "scene":     "FloorPlan1",
        "obs":       "Kitchen. Apple on counter. Microwave plugged in.",
        "visible_objects": ["apple", "microwave", "counter_top"],
        "reference_steps": [
            _s("MoveTo", "apple",     reason="Go to apple"),
            _s("Pick",     "apple",     reason="Pick up apple"),
            _s("MoveTo", "microwave", reason="Go to microwave"),
            _s("Open",     "microwave", reason="Open microwave door"),
            _s("PutIn",    "apple", "microwave", "Put apple in microwave"),
            _s("Close",    "microwave", reason="Close the door"),
            _s("TurnOn",   "microwave", reason="Start heating"),
        ],
        "expected_objects": ["apple", "microwave"],
        "difficulty": "medium",
    },
    {
        "task_id":   "X02",
        "task_desc": "Place a book on the coffee table and turn on the lamp",
        "room":      "living_room",
        "scene":     "FloorPlan201",
        "obs":       "Living room. Book on sofa. Coffee table nearby. Lamp in corner.",
        "visible_objects": ["book", "sofa", "coffee_table", "lamp"],
        "reference_steps": [
            _s("MoveTo", "book",          reason="Go to book"),
            _s("Pick",     "book",          reason="Pick up book"),
            _s("MoveTo", "coffee_table",  reason="Go to coffee table"),
            _s("Place",    "book", "coffee_table", "Place book on table"),
            _s("MoveTo", "lamp",          reason="Go to lamp"),
            _s("TurnOn",   "lamp",          reason="Turn lamp on"),
        ],
        "expected_objects": ["book", "coffee_table", "lamp"],
        "difficulty": "medium",
    },
    {
        "task_id":   "X03",
        "task_desc": "Fill a cup with water from the sink",
        "room":      "kitchen",
        "scene":     "FloorPlan2",
        "obs":       "Kitchen. Cup on table. Sink with faucet accessible.",
        "visible_objects": ["cup", "sink", "faucet", "counter_top"],
        "reference_steps": [
            _s("MoveTo", "cup",    reason="Go to cup"),
            _s("Pick",     "cup",    reason="Pick up cup"),
            _s("MoveTo", "sink",   reason="Go to sink"),
            _s("TurnOn",   "faucet", reason="Turn on water"),
            _s("Wait",     "",       reason="Fill the cup"),
            _s("TurnOff",  "faucet", reason="Turn off water"),
        ],
        "expected_objects": ["cup", "faucet"],
        "difficulty": "easy",
    },
    {
        "task_id":   "X04",
        "task_desc": "Open the window and then close the curtains",
        "room":      "living_room",
        "scene":     "FloorPlan202",
        "obs":       "Living room. Window with curtains. Curtains open.",
        "visible_objects": ["window", "curtain", "sofa", "lamp"],
        "reference_steps": [
            _s("MoveTo", "window",  reason="Go to the window"),
            _s("Open",     "window",  reason="Open the window"),
            _s("MoveTo", "curtain", reason="Go to curtains"),
            _s("Close",    "curtain", reason="Close the curtains"),
        ],
        "expected_objects": ["window", "curtain"],
        "difficulty": "easy",
    },
    {
        "task_id":   "X05",
        "task_desc": "Place soap in the cabinet under the sink",
        "room":      "bathroom",
        "scene":     "FloorPlan401",
        "obs":       "Bathroom. Soap on counter. Cabinet under sink.",
        "visible_objects": ["soap", "sink", "cabinet", "towel"],
        "reference_steps": [
            _s("MoveTo", "soap",    reason="Go to soap"),
            _s("Pick",     "soap",    reason="Pick up soap"),
            _s("MoveTo", "cabinet", reason="Go to cabinet"),
            _s("Open",     "cabinet", reason="Open cabinet"),
            _s("PutIn",    "soap", "cabinet", "Place soap inside"),
            _s("Close",    "cabinet", reason="Close cabinet"),
        ],
        "expected_objects": ["soap", "cabinet"],
        "difficulty": "medium",
        "fail_injection": _fail(3, "Cabinet is already full"),
    },
]


# ═════════════════════════════════════════════════════════════════════
# SCORING HELPERS  (used by evaluate.py)
# ═════════════════════════════════════════════════════════════════════

def score_executability(steps: list[dict], valid_actions: set[str] = VALID_ACTIONS) -> float:
    """Fraction of steps with a valid action AND non-empty object."""
    if not steps:
        return 0.0
    ok = sum(
        1 for s in steps
        if s.get("action", "") in valid_actions and bool(s.get("object", "").strip())
    )
    return round(ok / len(steps), 4)


def score_precondition(steps: list[dict]) -> float:
    """
    Fraction of interact-steps that are immediately preceded by Navigate/Find.
    Interact actions: Pick, Place, PutIn, Open, Close, TurnOn, TurnOff, Wash, Sit, LieOn, Serve
    """
    INTERACT = {"Pick","Place","PutIn","Open","Close","TurnOn","TurnOff","Wash","Sit","LieOn","Serve"}
    NAV      = {"MoveTo", "Find"}
    if not steps:
        return 1.0
    interact_steps = [(i, s) for i, s in enumerate(steps) if s.get("action") in INTERACT]
    if not interact_steps:
        return 1.0
    ok = 0
    for i, _ in interact_steps:
        if i > 0 and steps[i - 1].get("action") in NAV:
            ok += 1
    return round(ok / len(interact_steps), 4)


def score_redundancy(steps: list[dict]) -> float:
    """Fraction of consecutive duplicate (action, object) pairs."""
    if len(steps) < 2:
        return 0.0
    dupes = sum(
        1 for i in range(1, len(steps))
        if steps[i].get("action") == steps[i-1].get("action")
        and steps[i].get("object") == steps[i-1].get("object")
    )
    return round(dupes / (len(steps) - 1), 4)


def score_completeness(steps: list[dict], expected_objects: list[str]) -> float:
    """Fraction of expected objects mentioned in at least one step."""
    if not expected_objects:
        return 1.0
    mentioned = {s.get("object", "").lower() for s in steps}
    covered = sum(1 for obj in expected_objects if obj.lower() in mentioned)
    return round(covered / len(expected_objects), 4)


def score_hallucination(steps: list[dict], visible_objects: list[str]) -> float:
    """Fraction of steps referencing objects NOT in visible_objects list."""
    if not steps:
        return 0.0
    visible_lower = {v.lower() for v in visible_objects}
    hallucinated = sum(
        1 for s in steps
        if s.get("object", "").strip()
        and s["object"].lower() not in visible_lower
    )
    return round(hallucinated / len(steps), 4)


def compute_quality_score(metrics: dict) -> float:
    """
    Weighted aggregate quality score in [0, 1].
    Higher = better plan quality.
    """
    weights = {
        "executability":  0.30,
        "precondition":   0.25,
        "completeness":   0.25,
        "redundancy":    -0.10,   # penalise redundancy
        "hallucination": -0.10,   # penalise hallucination
    }
    score = (
        weights["executability"]  * metrics.get("executability",  0.0)
      + weights["precondition"]   * metrics.get("precondition",   0.0)
      + weights["completeness"]   * metrics.get("completeness",   0.0)
      + weights["redundancy"]     * metrics.get("redundancy",     0.0)
      + weights["hallucination"]  * metrics.get("hallucination",  0.0)
    )
    return round(max(0.0, min(1.0, score + 0.2)), 4)   # shift so baseline ~0.5


# ═════════════════════════════════════════════════════════════════════
# BUILD + VALIDATE
# ═════════════════════════════════════════════════════════════════════

def build_dataset() -> list[dict]:
    samples = []
    for raw in SAMPLES_RAW:
        fi = raw.get("fail_injection")
        sample = {
            "task_id":          raw["task_id"],
            "task_desc":        raw["task_desc"],
            "room":             raw["room"],
            "scene":            raw["scene"],
            "obs":              raw["obs"],
            "visible_objects":  raw["visible_objects"],
            "reference_steps":  raw["reference_steps"],
            "expected_objects": raw["expected_objects"],
            "difficulty":       raw["difficulty"],
            "fail_injection":   fi if fi else {},
        }
        samples.append(sample)
    return samples


def validate_dataset(samples: list[dict]) -> list[str]:
    """Return list of validation warnings."""
    warnings = []
    ids_seen = set()
    for s in samples:
        tid = s["task_id"]
        if tid in ids_seen:
            warnings.append(f"Duplicate task_id: {tid}")
        ids_seen.add(tid)

        for step in s["reference_steps"]:
            if step["action"] not in VALID_ACTIONS:
                warnings.append(f"{tid}: invalid action '{step['action']}'")
            if not step.get("object") and step["action"] not in {"Wait"}:
                warnings.append(f"{tid}: step '{step['action']}' has empty object")

        if not s["expected_objects"]:
            warnings.append(f"{tid}: expected_objects is empty")

        if s["difficulty"] not in {"easy", "medium", "hard"}:
            warnings.append(f"{tid}: unknown difficulty '{s['difficulty']}'")

    return warnings


def print_summary(samples: list[dict]):
    from collections import Counter
    diff_counts = Counter(s["difficulty"] for s in samples)
    room_counts = Counter(s["room"]       for s in samples)
    fail_count  = sum(1 for s in samples if s.get("fail_injection"))
    avg_steps   = sum(len(s["reference_steps"]) for s in samples) / len(samples)

    print(f"\n{'─'*45}")
    print(f"  Dataset summary — {len(samples)} samples")
    print(f"{'─'*45}")
    print(f"  Difficulty:  easy={diff_counts['easy']}  medium={diff_counts['medium']}  hard={diff_counts['hard']}")
    print(f"  Rooms:       {dict(room_counts)}")
    print(f"  With fail injection: {fail_count}/{len(samples)}")
    print(f"  Avg reference steps: {avg_steps:.1f}")
    print(f"{'─'*45}\n")


def main():
    parser = argparse.ArgumentParser(description="Build pyplanner evaluation dataset")
    parser.add_argument("--out",     default="eval_dataset.json", help="Output JSON file")
    parser.add_argument("--summary", action="store_true",         help="Print summary only, no file write")
    args = parser.parse_args()

    samples = build_dataset()
    warnings = validate_dataset(samples)

    if warnings:
        print(f"\n⚠  {len(warnings)} validation warnings:")
        for w in warnings:
            print(f"   {w}")
    else:
        print(f"\n✅  Dataset validated — no warnings")

    print_summary(samples)

    if not args.summary:
        out_path = args.out
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "samples": samples}, f, indent=2, ensure_ascii=False)
        print(f"✅  Saved {len(samples)} samples → {out_path}")
        print(f"    Size: {os.path.getsize(out_path) // 1024} KB\n")


if __name__ == "__main__":
    main()
