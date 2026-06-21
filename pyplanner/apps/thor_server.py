# thor_server.py
# Unified AI2-THOR server supporting both iTHOR and ProcTHOR via a single ZMQ socket.
#
# Run:
#   python thor_server.py               # default port 5555
#   python thor_server.py --port 5556   # custom port
#
# Install deps:
#   pip install ai2thor pyzmq numpy opencv-python
#   pip install prior   # only needed for ProcTHOR scenes

from __future__ import annotations
import argparse
import json
import base64
import math
import random
import logging

import zmq
import numpy as np
from ai2thor.controller import Controller

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ThorServer")


# ══════════════════════════════════════════════════════════════════════
# CONTROLLER FACTORIES
# ══════════════════════════════════════════════════════════════════════

def make_ithor_controller(scene: str = "FloorPlan1") -> Controller:
    """Create a standard iTHOR controller for the given FloorPlan scene."""
    return Controller(
        scene=scene,
        agentMode="default",
        visibilityDistance=1.5,
        gridSize=0.25,
        snapToGrid=True,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=640,
        height=480,
        fieldOfView=90,
    )


def make_robotthor_controller(scene: str = "FloorPlan_Train1_1") -> Controller:
    """Create a RoboTHOR controller for the given scene."""
    return Controller(
        scene=scene,
        agentMode="locobot",
        visibilityDistance=1.5,
        gridSize=0.25,
        snapToGrid=True,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=640,
        height=480,
        fieldOfView=90,
    )


def make_procthor_controller(house=None, split: str = "train", house_index: int | None = None) -> tuple[Controller, list[dict]]:
    """
    Create a ProcTHOR controller.
    If house_index is given, loads that specific house from the split.
    Otherwise loads a random house from the split.
    Requires the `prior` package: pip install prior
    """
    try:
        import prior
    except ImportError as exc:
        raise ImportError(
            "The 'prior' package is required for ProcTHOR. "
            "Install it with:  pip install prior"
        ) from exc

    dataset = prior.load_dataset("procthor-10k")
    if house is None:
        split_data = dataset[split]
        if house_index is not None:
            house = split_data[int(house_index)]
            log.info(f"ProcTHOR: loading {split}[{house_index}]")
        else:
            house = split_data[random.randint(0, len(split_data) - 1)]
            log.info(f"ProcTHOR: loading random house from {split}")

    ctrl = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        gridSize=0.25,
        snapToGrid=True,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=640,
        height=480,
        fieldOfView=90,
    )
    ctrl.reset(scene=house)
    rooms = house.get("rooms", []) if isinstance(house, dict) else []
    return ctrl, rooms


# Task label → FloorPlan name (iTHOR only; ProcTHOR ignores this)
TASK_SCENES: dict[str, str] = {
    "CoffeeSetupMug":            "FloorPlan1",
    "TurnOnStove":               "FloorPlan2",
    "BoilPot":                   "FloorPlan3",
    "TurnOnMicrowave":           "FloorPlan1",
    "OpenFridge":                "FloorPlan1",
    "PickPlaceCounterToCabinet": "FloorPlan2",
    "WashDishes":                "FloorPlan3",
    "WatchTV":                   "FloorPlan201",
    "ReadBook":                  "FloorPlan201",
    "GoToSleep":                 "FloorPlan301",
    "BrushTeeth":                "FloorPlan401",
}


# ══════════════════════════════════════════════════════════════════════
# SKILL PRIMITIVES  (each returns (success: bool, message: str))
# ══════════════════════════════════════════════════════════════════════

def _find_object(controller: Controller, object_type: str) -> dict | None:
    """Return the nearest object whose objectType matches (partial, case-insensitive)."""
    if not object_type or not object_type.strip():
        return None
    event = controller.step("Pass")
    key   = object_type.lower().replace(" ", "").replace("_", "")

    candidates = [
        o for o in event.metadata["objects"]
        if key in o["objectType"].lower()
        or o["objectType"].lower() in key
        or key in o["objectId"].lower()
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x["distance"])


def _navigate_to(controller: Controller, object_id: str) -> tuple[bool, str]:
    """Teleport agent to the closest interactable pose for object_id.

    Strategy:
    1. GetInteractablePoses with broad horizons/standings
    2. Fallback: GetReachablePositions → closest reachable grid pos → face object
    3. Last resort: try PickupObject from current position
    """
    agent_event = controller.step("Pass")
    agent_pos   = agent_event.metadata["agent"]["position"]

    # 1. Try GetInteractablePoses with broad parameters
    event = controller.step(
        action="GetInteractablePoses",
        objectId=object_id,
        horizons=[-30, -15, 0, 15, 30, 45, 60],
        standings=[True, False],
    )
    poses = event.metadata.get("actionReturn") or []
    if poses:
        best = min(poses, key=lambda p: (p["x"] - agent_pos["x"]) ** 2
                                      + (p["z"] - agent_pos["z"]) ** 2)
        ev2 = controller.step(
            action="TeleportFull",
            position={"x": best["x"], "y": best["y"], "z": best["z"]},
            rotation={"x": 0, "y": best["rotation"], "z": 0},
            horizon=best["horizon"],
            standing=best.get("standing", True),
        )
        if ev2.metadata["lastActionSuccess"]:
            return True, f"Navigated to {object_id}"

    # 2. Fallback: use GetReachablePositions
    obj_meta = next(
        (o for o in agent_event.metadata["objects"] if o["objectId"] == object_id),
        None,
    )
    if obj_meta:
        op  = obj_meta["position"]
        cy  = agent_pos["y"]
        ev_r = controller.step(action="GetReachablePositions")
        reachable = ev_r.metadata.get("actionReturn") or []
        if reachable:
            best_pos = min(reachable,
                           key=lambda p: (p["x"] - op["x"]) ** 2 + (p["z"] - op["z"]) ** 2)
            angle = math.degrees(math.atan2(op["x"] - best_pos["x"],
                                            op["z"] - best_pos["z"])) % 360
            for horizon in [30, 15, 0, 45]:
                ev3 = controller.step(
                    action="TeleportFull",
                    position={"x": best_pos["x"], "y": cy, "z": best_pos["z"]},
                    rotation={"x": 0, "y": angle, "z": 0},
                    horizon=horizon,
                    standing=True,
                )
                if ev3.metadata["lastActionSuccess"]:
                    return True, f"Navigated near {object_id}"

    # 3. Last resort: stay in place, let PickupObject try anyway
    return True, f"Could not navigate to {object_id}, attempting pickup from current position"


def _navigate_to_room_center(controller: Controller, room_type: str, house_rooms: list) -> tuple[bool, str]:
    """Teleport agent to the centroid of the named room using its floorPolygon."""
    filtered = _filter_rooms(house_rooms)
    key = room_type.lower().replace(" ", "").replace("_", "")
    match = next(
        (r for r in filtered
         if r.get("roomType", "").lower().replace(" ", "") == key
         or _room_label(r, filtered).lower().replace(" ", "") == key),
        None,
    )
    if not match:
        return False, f"Room '{room_type}' not found in house"

    polygon = match.get("floorPolygon", [])
    if not polygon:
        return False, f"Room '{room_type}' has no floor polygon"

    ax2 = _floor_axis(polygon)
    cx = sum(p["x"]  for p in polygon) / len(polygon)
    cz = sum(p[ax2]  for p in polygon) / len(polygon)
    # Snap to 0.25 grid (AI2-THOR requirement)
    cx = round(round(cx / 0.25) * 0.25, 4)
    cz = round(round(cz / 0.25) * 0.25, 4)

    event = controller.step("Pass")
    cy    = event.metadata["agent"]["position"]["y"]
    agent = event.metadata["agent"]

    # Build spiral of candidate grid points around centroid, all inside the polygon
    GRID = 0.25
    candidates = [(cx, cz)]
    for r in range(1, 20):
        for dx in range(-r, r + 1):
            for dz in [-r, r] if abs(dx) < r else [range(-r, r + 1)]:
                if isinstance(dz, range):
                    for dz2 in dz:
                        px = round(cx + dx * GRID, 4)
                        pz = round(cz + dz2 * GRID, 4)
                        if _point_in_polygon(px, pz, polygon):
                            candidates.append((px, pz))
                else:
                    px = round(cx + dx * GRID, 4)
                    pz = round(cz + dz * GRID, 4)
                    if _point_in_polygon(px, pz, polygon):
                        candidates.append((px, pz))

    for px, pz in candidates:
        event = controller.step(
            action="TeleportFull",
            position={"x": px, "y": cy, "z": pz},
            rotation=agent["rotation"],
            horizon=agent["cameraHorizon"],
            standing=True,
            forceAction=False,
        )
        if event.metadata["lastActionSuccess"]:
            return True, f"Moved to {room_type} ({px:.2f}, {pz:.2f})"

    return False, f"No free grid position found in {room_type}"


def skill_navigate(controller: Controller, object_type: str,
                   house_rooms: list | None = None, ctx: dict | None = None, **_) -> tuple[bool, str]:
    """MoveTo <room | furniture>. Updates ctx['arrived'] on success."""
    rooms = house_rooms or []
    if rooms:
        filtered = _filter_rooms(rooms)
        key = object_type.lower().replace(" ", "").replace("_", "")
        room_match = next(
            (r for r in filtered
             if r.get("roomType", "").lower().replace(" ", "") == key
             or _room_label(r, filtered).lower().replace(" ", "") == key),
            None,
        )
        if room_match:
            ok, msg = _navigate_to_room_center(controller, object_type, rooms)
            if ok and ctx is not None:
                ctx["arrived"] = room_match.get("roomType", object_type)
            return ok, msg

    obj = _find_object(controller, object_type)
    if not obj:
        return False, f"Cannot find '{object_type}' in scene"

    # If house has rooms, verify the object is in the current room
    filtered = _filter_rooms(rooms)
    if filtered:
        event = controller.step("Pass")
        cur_room_label = _room_from_event(event, rooms)
        opos = obj.get("position", {})
        ox, oz = opos.get("x", 0), opos.get("z", 0)
        obj_room_label = None
        for r in filtered:
            if _point_in_polygon(ox, oz, r.get("floorPolygon", [])):
                obj_room_label = _room_label(r, filtered)
                break
        if obj_room_label is None:
            nr = _nearest_room_entry(ox, oz, filtered)
            obj_room_label = _room_label(nr, filtered) if nr else "another room"
        if obj_room_label.lower().replace(" ", "") != cur_room_label.lower().replace(" ", ""):
            return False, (
                f"'{object_type}' is in {obj_room_label}, not current room ({cur_room_label}). "
                f"MoveTo {obj_room_label} first."
            )

    ok, msg = _navigate_to(controller, obj["objectId"])
    if ok and ctx is not None:
        ctx["arrived"] = obj["objectType"]
    return ok, msg


def skill_find(controller: Controller, object_type: str,
               house_rooms: list | None = None, ctx: dict | None = None, **_) -> tuple[bool, str]:
    """Find closest matching object in the current room. Updates ctx['found_type'/'found_id']."""
    event = controller.step("Pass")
    all_objs = event.metadata["objects"]

    # Restrict to current room polygon if available
    rooms = _filter_rooms(house_rooms or [])
    current_poly = None
    if rooms:
        cur_room = _room_from_event(event, house_rooms)
        cur_key  = cur_room.lower().replace(" ", "").replace("_", "")
        for r in rooms:
            if _room_label(r, rooms).lower().replace(" ", "").replace("_", "") == cur_key \
               or r.get("roomType", "").lower().replace(" ", "") == cur_key:
                current_poly = r.get("floorPolygon", [])
                break

    key = object_type.lower().replace("_", "").replace(" ", "")
    candidates = []
    for o in all_objs:
        if o["objectType"].lower() != key and \
           o["objectType"].lower().replace(" ", "") != key:
            continue
        if current_poly:
            opos = o.get("position", {})
            if not _point_in_polygon(opos.get("x", 0), opos.get("z", 0), current_poly):
                continue
        candidates.append(o)

    if not candidates:
        return False, f"'{object_type}' not found in current room"

    obj = min(candidates, key=lambda o: o.get("distance", float("inf")))

    # Rotate agent to face the found object
    agent = event.metadata["agent"]
    pos   = agent["position"]
    rot   = agent["rotation"]["y"]
    dx = obj["position"]["x"] - pos["x"]
    dz = obj["position"]["z"] - pos["z"]
    target_angle = math.degrees(math.atan2(dx, dz)) % 360
    diff = (target_angle - rot + 360) % 360
    if diff > 180:
        steps = int((360 - diff) / 90) + 1
        for _ in range(steps):
            controller.step("RotateLeft")
    else:
        steps = int(diff / 90) + 1
        for _ in range(steps):
            controller.step("RotateRight")

    if ctx is not None:
        ctx["found_type"] = obj["objectType"]
        ctx["found_id"]   = obj["objectId"]
    return True, f"Found {obj['objectType']} at {obj['distance']:.2f}m"


def skill_grab(controller: Controller, object_type: str,
               ctx: dict | None = None, **_) -> tuple[bool, str]:
    # Assert found
    found_id   = ctx.get("found_id")   if ctx else None
    found_type = ctx.get("found_type") if ctx else None
    if not found_id:
        return False, "No object found — use Find first"

    # Assert not already holding
    event = controller.step("Pass")
    held = [o for o in event.metadata["objects"] if o.get("isPickedUp")]
    if held:
        return False, f"Already holding {held[0]['objectType']} — Place it first"

    # Navigate to found object
    nav_ok, nav_msg = _navigate_to(controller, found_id)
    if not nav_ok:
        return False, nav_msg

    # Try normal pickup first, then forceAction to handle visibility edge cases
    for force in (False, True):
        event = controller.step(action="PickupObject", objectId=found_id, forceAction=force)
        if event.metadata["lastActionSuccess"]:
            if ctx is not None:
                ctx["found_type"] = None
                ctx["found_id"]   = None
            return True, f"Picked up {found_type or object_type}"
    return False, event.metadata.get("errorMessage", "Pickup failed")


def skill_place(controller: Controller, object_type: str,
                ctx: dict | None = None, **_) -> tuple[bool, str]:
    """Place held object onto arrived location. Asserts holding + arrived."""
    # Assert holding
    event = controller.step("Pass")
    held = [o for o in event.metadata["objects"] if o.get("isPickedUp")]
    if not held:
        return False, "Not holding any object — Pick something up first"
    held_obj = held[0]

    # Assert arrived
    arrived = ctx.get("arrived") if ctx else None
    receptacle_name = object_type or arrived
    if not receptacle_name:
        return False, "No arrived location — MoveTo a receptacle first"

    recep = _find_object(controller, receptacle_name)
    if not recep:
        return False, f"Cannot find receptacle '{receptacle_name}'"

    surface_types = {"floor", "counter", "table", "desk", "shelf",
                     "bed", "sofa", "chair", "sink", "basin", "tub"}
    is_surface = any(s in recep["objectType"].lower() for s in surface_types)
    if not recep.get("receptacle") and not is_surface:
        return False, f"'{recep['objectType']}' is not a receptacle"

    nav_ok, nav_msg = _navigate_to(controller, recep["objectId"])
    if not nav_ok:
        return False, nav_msg

    event = controller.step(action="PutObject", objectId=recep["objectId"],
                            forceAction=True, placeStationary=True)
    if event.metadata["lastActionSuccess"]:
        if ctx is not None:
            ctx["arrived"] = None
        return True, f"Placed {held_obj['objectType']} on/in {receptacle_name}"

    event2 = controller.step(action="PutObject", objectId=recep["objectId"], placeStationary=True)
    if event2.metadata["lastActionSuccess"]:
        if ctx is not None:
            ctx["arrived"] = None
        return True, f"Placed {held_obj['objectType']} on/in {receptacle_name}"

    return False, event2.metadata.get("errorMessage", event.metadata.get("errorMessage", "Place failed"))


def _assert_found(ctx: dict | None, action: str):
    """Return (objectId, objectType, error_msg). error_msg is None if found."""
    if ctx is None or not ctx.get("found_id"):
        return None, None, f"{action} requires Find first — no object found"
    return ctx["found_id"], ctx.get("found_type", ""), None


def skill_open(controller: Controller, object_type: str,
               ctx: dict | None = None, **_) -> tuple[bool, str]:
    found_id, found_type, err = _assert_found(ctx, "Open")
    if err:
        return False, err
    nav_ok, nav_msg = _navigate_to(controller, found_id)
    if not nav_ok:
        return False, nav_msg
    event = controller.step(action="OpenObject", objectId=found_id)
    if event.metadata["lastActionSuccess"]:
        return True, f"Opened {found_type}"
    return False, event.metadata.get("errorMessage", "Open failed")


def skill_close(controller: Controller, object_type: str,
                ctx: dict | None = None, **_) -> tuple[bool, str]:
    found_id, found_type, err = _assert_found(ctx, "Close")
    if err:
        return False, err
    nav_ok, nav_msg = _navigate_to(controller, found_id)
    if not nav_ok:
        return False, nav_msg
    event = controller.step(action="CloseObject", objectId=found_id)
    if event.metadata["lastActionSuccess"]:
        return True, f"Closed {found_type}"
    return False, event.metadata.get("errorMessage", "Close failed")


def skill_turnon(controller: Controller, object_type: str,
                 ctx: dict | None = None, **_) -> tuple[bool, str]:
    found_id, found_type, err = _assert_found(ctx, "TurnOn")
    if err:
        return False, err
    nav_ok, nav_msg = _navigate_to(controller, found_id)
    if not nav_ok:
        return False, nav_msg
    event = controller.step(action="ToggleObjectOn", objectId=found_id)
    if event.metadata["lastActionSuccess"]:
        return True, f"Turned on {found_type}"
    return False, event.metadata.get("errorMessage", "Toggle on failed")


def skill_turnoff(controller: Controller, object_type: str,
                  ctx: dict | None = None, **_) -> tuple[bool, str]:
    found_id, found_type, err = _assert_found(ctx, "TurnOff")
    if err:
        return False, err
    nav_ok, nav_msg = _navigate_to(controller, found_id)
    if not nav_ok:
        return False, nav_msg
    event = controller.step(action="ToggleObjectOff", objectId=found_id)
    if event.metadata["lastActionSuccess"]:
        return True, f"Turned off {found_type}"
    return False, event.metadata.get("errorMessage", "Toggle off failed")


def skill_wash(controller: Controller, object_type: str, **_) -> tuple[bool, str]:
    grab_ok, grab_msg = skill_grab(controller, object_type)
    if not grab_ok:
        return False, grab_msg
    sink = _find_object(controller, "Sink") or _find_object(controller, "SinkBasin")
    if not sink:
        return False, "Cannot find sink in scene"
    nav_ok, nav_msg = _navigate_to(controller, sink["objectId"])
    if not nav_ok:
        return False, nav_msg
    event = controller.step("Pass")
    held  = [o for o in event.metadata["objects"] if o.get("isPickedUp")]
    if held:
        controller.step(
            action="PutObject",
            objectId=held[0]["objectId"],
            receptacleObjectId=sink["objectId"],
        )
    return True, f"Washed {object_type} in sink"


def skill_sit(controller: Controller, object_type: str, **_) -> tuple[bool, str]:
    obj = _find_object(controller, object_type)
    if not obj:
        return False, f"Cannot find '{object_type}'"
    nav_ok, nav_msg = _navigate_to(controller, obj["objectId"])
    if not nav_ok:
        return False, nav_msg
    return True, f"Agent is sitting on {object_type}"


def skill_wait(controller: Controller, **_) -> tuple[bool, str]:
    controller.step("Pass")
    return True, "Agent waited"


SKILLS: dict[str, callable] = {
    "MoveTo": skill_navigate,
    "Find":     skill_find,
    "Pick":     skill_grab,
    "Place":    skill_place,
    "PutIn":    skill_place,
    "Open":     skill_open,
    "Close":    skill_close,
    "TurnOn":   skill_turnon,
    "TurnOff":  skill_turnoff,
    "Wash":     skill_wash,
    "Sit":      skill_sit,
    "LieOn":    skill_sit,
    "Serve":    skill_place,
    "Wait":     skill_wait,
}


# ══════════════════════════════════════════════════════════════════════
# OBSERVATION HELPERS
# ══════════════════════════════════════════════════════════════════════

def _floor_axis(polygon: list[dict]) -> str:
    """
    ProcTHOR prior dataset stores floorPolygon as {x, y, z} where the second
    floor axis may be 'y' (z=0 constant) OR 'z' (y=0 constant).
    Pick whichever has more variance.
    """
    y_vals = [p.get("y", 0) for p in polygon]
    z_vals = [p.get("z", 0) for p in polygon]
    y_range = max(y_vals) - min(y_vals) if y_vals else 0
    z_range = max(z_vals) - min(z_vals) if z_vals else 0
    return "y" if y_range > z_range else "z"


# Structural/non-interactable object types — excluded from nearby/visible lists
_STRUCTURAL = {"Wall", "Floor", "Ceiling", "Room", "Doorway", "Window",
               "WallDecal", "ShowerDoor", "ShowerCurtain"}

# Canonical room types to display — one of each per house
_MAIN_ROOM_TYPES = ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"]
# Aliases: map variant names → canonical type
_ROOM_TYPE_ALIAS = {
    "livingroom": "LivingRoom",
    "restroom":   "Bathroom",
    "toilet":     "Bathroom",
    "washroom":   "Bathroom",
}


def _polygon_area(polygon: list[dict]) -> float:
    """Shoelace area on the floor plane."""
    if len(polygon) < 3:
        return 0.0
    ax2 = _floor_axis(polygon)
    n   = len(polygon)
    return abs(sum(
        polygon[i]["x"] * polygon[(i + 1) % n][ax2]
        - polygon[(i + 1) % n]["x"] * polygon[i][ax2]
        for i in range(n)
    ) / 2)


def _canonical_type(rt: str) -> str:
    """Normalise a roomType string to one of _MAIN_ROOM_TYPES, or '' if unknown."""
    if rt in _MAIN_ROOM_TYPES:
        return rt
    return _ROOM_TYPE_ALIAS.get(rt.lower(), "")


def _all_valid_rooms(rooms: list[dict]) -> list[dict]:
    """All room entries that belong to a main type — used for polygon hit-testing."""
    return [r for r in rooms if _canonical_type(r.get("roomType", ""))]


def _filter_rooms(rooms: list[dict]) -> list[dict]:
    """Return exactly one room per main type (largest by polygon area) — used for display/navigation."""
    by_type: dict[str, list[dict]] = {}
    for r in rooms:
        ct = _canonical_type(r.get("roomType", ""))
        if ct:
            by_type.setdefault(ct, []).append(r)
    result = []
    for rt in _MAIN_ROOM_TYPES:
        candidates = by_type.get(rt, [])
        if candidates:
            best = max(candidates, key=lambda r: _polygon_area(r.get("floorPolygon", [])))
            result.append(best)
    return result


def _point_in_polygon(x: float, z: float, polygon: list[dict]) -> bool:
    """Ray-casting point-in-polygon test on the floor plane (X + auto-detected axis)."""
    if not polygon:
        return False
    ax2 = _floor_axis(polygon)
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, zi = polygon[i]["x"], polygon[i][ax2]
        xj, zj = polygon[j]["x"], polygon[j][ax2]
        if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _room_label(room: dict, all_rooms: list[dict]) -> str:
    """Return display label: 'Kitchen' if unique type, 'livingroom3' if duplicated."""
    rt = room.get("roomType", "unknown")
    count = sum(1 for r in all_rooms if r.get("roomType") == rt)
    if count > 1:
        rid = room.get("id", "")
        num = rid.split("|")[-1] if "|" in rid else rid
        return f"{rt.lower()}{num}"
    return rt


def _nearest_room_entry(x: float, z: float, rooms: list[dict]) -> dict | None:
    """Return the room dict whose centroid is closest to (x, z)."""
    best_room = None
    best_dist = float("inf")
    for room in rooms:
        polygon = room.get("floorPolygon", [])
        if not polygon:
            continue
        ax2 = _floor_axis(polygon)
        cx = sum(p["x"]  for p in polygon) / len(polygon)
        cz = sum(p[ax2]  for p in polygon) / len(polygon)
        d  = (x - cx) ** 2 + (z - cz) ** 2
        if d < best_dist:
            best_dist = d
            best_room = room
    return best_room


def _room_from_event(event, house_rooms: list | None = None) -> str:
    """
    Detect the agent's current room.
    - ProcTHOR: uses house_rooms (from prior dataset) + point-in-polygon,
                falls back to nearest centroid if polygon test fails.
    - iTHOR: derived from sceneName (single-room scenes).
    """
    rooms = _filter_rooms(house_rooms or [])
    if rooms:
        pos = event.metadata["agent"]["position"]
        ax, az = pos["x"], pos["z"]
        for room in rooms:
            polygon = room.get("floorPolygon", [])
            if polygon and _point_in_polygon(ax, az, polygon):
                return _room_label(room, rooms)
        # Fallback: nearest room centroid
        nr = _nearest_room_entry(ax, az, rooms)
        return _room_label(nr, rooms) if nr else "unknown room"

    # iTHOR: single-room scene — derive from scene name
    scene_name = event.metadata.get("sceneName", "")
    try:
        n = int("".join(filter(str.isdigit, scene_name)))
        if   1   <= n <= 30:  return "Kitchen"
        elif 201 <= n <= 230: return "Living room"
        elif 301 <= n <= 330: return "Bedroom"
        elif 401 <= n <= 430: return "Bathroom"
    except ValueError:
        pass
    return scene_name or "unknown"


def build_robot_state(event, house_rooms: list | None = None,
                      found_type: str | None = None,
                      arrived: str | None = None) -> str:
    """Robot-centric state: room, location, holding, found, arrived, env (furniture per room)."""
    agent  = event.metadata["agent"]
    pos    = agent["position"]
    rooms  = house_rooms or []
    room   = _room_from_event(event, rooms)

    held = [o for o in event.metadata["objects"] if o.get("isPickedUp")]
    held_str = held[0]["objectType"] if held else "nothing"

    lines = [
        f"Current room: {room}",
        f"Location: ({pos['x']:.2f}, {pos['z']:.2f}), facing {agent['rotation']['y']:.0f}°",
        f"Holding: {held_str}",
        f"Found: {found_type or 'nothing'}",
        f"Arrived: {arrived or 'none'}",
    ]

    # Env: furniture (non-pickupable) per room
    disp_rooms = _filter_rooms(rooms)
    all_poly   = _all_valid_rooms(rooms)
    if disp_rooms:
        all_objs = event.metadata["objects"]
        furn_per_room: dict[str, set] = {_room_label(r, disp_rooms): set() for r in disp_rooms}
        for o in all_objs:
            if o.get("pickupable") or o["objectType"] in _STRUCTURAL:
                continue
            opos = o.get("position", {})
            ox, oz = opos.get("x", 0), opos.get("z", 0)
            assigned_ct = None
            for r in all_poly:
                if _point_in_polygon(ox, oz, r.get("floorPolygon", [])):
                    assigned_ct = _canonical_type(r.get("roomType", ""))
                    break
            if assigned_ct is None:
                nr = _nearest_room_entry(ox, oz, all_poly)
                if nr:
                    assigned_ct = _canonical_type(nr.get("roomType", ""))
            if assigned_ct:
                for dr in disp_rooms:
                    if _canonical_type(dr.get("roomType", "")) == assigned_ct:
                        furn_per_room[_room_label(dr, disp_rooms)].add(o["objectType"])
                        break
        lines.append("Env:")
        for lbl, types in furn_per_room.items():
            if types:
                lines.append(f"  {lbl}: {', '.join(sorted(types))}")

    if not event.metadata["lastActionSuccess"]:
        lines.append(f"Last action failed: {event.metadata.get('errorMessage', '')}")
    return "\n".join(lines)



def build_env_state(event, house_rooms: list | None = None) -> str:
    """Environment state: pickupable objects per room."""
    raw          = house_rooms or []
    disp_rooms   = _filter_rooms(raw)      # one per type — display labels
    all_poly     = _all_valid_rooms(raw)   # all polygons — for hit-testing
    all_objects  = event.metadata["objects"]

    def _is_env_obj(o: dict) -> bool:
        return bool(o.get("pickupable")) and o["objectType"] not in _STRUCTURAL

    def _fmt_room(label: str, obj_types: list[str]) -> list[str]:
        if not obj_types:
            return []
        lines = [f"{label}:"]
        for i in range(0, len(obj_types), 6):
            lines.append("  - " + ", ".join(obj_types[i:i + 6]))
        return lines

    def _display_label_for_type(ct: str) -> str | None:
        """Map canonical type → display label from disp_rooms."""
        for r in disp_rooms:
            if _canonical_type(r.get("roomType", "")) == ct:
                return _room_label(r, disp_rooms)
        return None

    if disp_rooms:
        labels        = [_room_label(r, disp_rooms) for r in disp_rooms]
        seen_per_room = {lbl: set() for lbl in labels}

        for o in all_objects:
            if not _is_env_obj(o):
                continue
            opos = o.get("position", {})
            ox, oz = opos.get("x", 0), opos.get("z", 0)
            assigned_ct = None
            # Test against ALL valid polygons (not just the representative one)
            for r in all_poly:
                if _point_in_polygon(ox, oz, r.get("floorPolygon", [])):
                    assigned_ct = _canonical_type(r.get("roomType", ""))
                    break
            if assigned_ct is None:
                # Fallback: nearest among all valid polygons
                nr = _nearest_room_entry(ox, oz, all_poly)
                if nr:
                    assigned_ct = _canonical_type(nr.get("roomType", ""))
            if assigned_ct:
                lbl = _display_label_for_type(assigned_ct)
                if lbl and lbl in seen_per_room:
                    seen_per_room[lbl].add(o["objectType"])

        lines = []
        for lbl in labels:
            lines += _fmt_room(lbl, sorted(seen_per_room[lbl]))
        return "\n".join(lines)

    else:
        scene_name = event.metadata.get("sceneName", "Scene")
        obj_types  = sorted({o["objectType"] for o in all_objects if _is_env_obj(o)})
        return "\n".join(_fmt_room(scene_name, obj_types))


def build_obs(event, house_rooms: list | None = None,
              found_type: str | None = None, arrived: str | None = None) -> str:
    """Combined obs for LLM context: robot state + environment state."""
    robot = build_robot_state(event, house_rooms, found_type, arrived)
    env   = build_env_state(event, house_rooms)
    return robot + "\n\nEnvironment:\n" + env


def get_visible_objects(event) -> list[str]:
    seen, result = set(), []
    for o in event.metadata["objects"]:
        if o["visible"] and o["objectType"] not in seen and o["objectType"] not in _STRUCTURAL:
            seen.add(o["objectType"])
            result.append(o["objectType"])
    return sorted(result)


def get_visible_objects_meta(event) -> list[dict]:
    seen, result = set(), []
    for o in event.metadata["objects"]:
        if o["visible"] and o["objectType"] not in seen and o["objectType"] not in _STRUCTURAL:
            seen.add(o["objectType"])
            result.append({
                "name":       o["objectType"],
                "pickupable": o.get("pickupable", False),
                "receptacle": o.get("receptacle", False),
                "openable":   o.get("openable", False),
                "toggleable": o.get("toggleable", False),
                "isOpen":     o.get("isOpen"),
                "isToggled":  o.get("isToggled"),
            })
    return sorted(result, key=lambda x: x["name"])


def to_python(obj):
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.bool_):      return bool(obj)
    if isinstance(obj, dict):          return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_python(v) for v in obj]
    return obj


# ══════════════════════════════════════════════════════════════════════
# SERVER
# ══════════════════════════════════════════════════════════════════════

class ThorServer:
    """
    Single ZMQ REP server that handles both iTHOR and ProcTHOR.

    Protocol — send JSON with `cmd` key:

        ping
            → {"status": "ok"}

        reset   {"task": <scene_or_task_name>, "simulator_type": "thor"|"procthor"}
            → {"status": "ok", "obs": ..., "visible_objects": [...], ...}

        step    {"action": <skill>, "object": <type>, "target": <type>}
            → {"status": "ok"|"error", "obs": ..., "success": bool, ...}

        get_frame
            → {"status": "ok", "frame": <base64-jpg>}

        get_objects
            → {"status": "ok", "objects": [...]}

        nav     {"action": "MoveAhead"|"RotateLeft"| ...}
            → {"status": "ok", ...}

        get_state
            → {"status": "ok", "position": ..., "rotation": ..., ...}
    """

    def __init__(self, port: int = 5555):
        self.controller:    Controller | None = None
        self.sim_type:      str  = "thor"
        self.current_scene: str  = ""
        self.step_count:    int  = 0
        self.total_reward:  float = 0.0
        self.house_rooms:   list  = []   # ProcTHOR room list from prior dataset
        self.found_type:    str | None = None
        self.found_id:      str | None = None
        self.arrived:       str | None = None

        self.ctx    = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        log.info(f"Listening on port {port}")
        log.info(f"Available skills: {list(SKILLS.keys())}")

    # ── internal helpers ──────────────────────────────────────────────

    def _stop_controller(self):
        if self.controller is not None:
            try:
                self.controller.stop()
            except Exception as exc:
                log.warning(f"Error stopping controller: {exc}")
            self.controller = None

    def _make_controller(
        self, task: str, simulator_type: str,
        split: str = "train", house_index: int | None = None,
    ) -> tuple[Controller, str, list]:
        """
        Create the right controller for the requested simulator type.
        Returns (controller, resolved_scene_name, house_rooms).
        house_rooms is a list of room dicts from prior (ProcTHOR only), else [].
        """
        sim = simulator_type.lower().strip()
        if sim in ("thor", "ithor"):
            scene = TASK_SCENES.get(task, task)
            ctrl  = make_ithor_controller(scene)
            return ctrl, scene, []

        elif sim == "robotthor":
            scene = task if task else "FloorPlan_Train1_1"
            ctrl  = make_robotthor_controller(scene)
            return ctrl, scene, []

        elif sim == "procthor":
            ctrl, rooms = make_procthor_controller(split=split, house_index=house_index)
            scene = f"procthor_{split}_{house_index}" if house_index is not None else f"procthor_{split}_random"
            return ctrl, scene, rooms

        else:
            raise ValueError(f"Unknown simulator type '{simulator_type}'. Use 'thor', 'robotthor', or 'procthor'.")

    # ── command handlers ──────────────────────────────────────────────

    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd", "")

        # ── ping ──────────────────────────────────────────────────────
        if cmd == "ping":
            return {"status": "ok", "sim_type": self.sim_type}

        # ── reset ─────────────────────────────────────────────────────
        if cmd == "reset":
            task        = msg.get("task", "FloorPlan1")
            sim_type    = msg.get("simulator_type", "thor")
            split       = msg.get("split", "train")
            house_index = msg.get("house_index", None)
            self._stop_controller()
            try:
                log.info(f"Resetting — sim_type={sim_type}, task={task}, split={split}, house_index={house_index}")
                ctrl, scene, rooms = self._make_controller(task, sim_type, split=split, house_index=house_index)
                self.controller    = ctrl
                self.sim_type      = sim_type
                self.current_scene = scene
                self.house_rooms   = rooms
                self.step_count    = 0
                self.total_reward  = 0.0
                self.found_type    = None
                self.found_id      = None
                self.arrived       = None

                event = self.controller.step("Pass")
                return {
                    "status":                 "ok",
                    "obs":                    build_obs(event, self.house_rooms),
                    "robot_state":            build_robot_state(event, self.house_rooms),
                    "env_state":              build_env_state(event, self.house_rooms),
                    "visible_objects":        get_visible_objects(event),
                    "visible_objects_meta":   get_visible_objects_meta(event),
                    "scene":                  scene,
                    "sim_type":               sim_type,
                    "available_actions":      list(SKILLS.keys()),
                }
            except Exception as exc:
                import traceback
                return {"status": "error", "msg": str(exc), "trace": traceback.format_exc()}

        # ── step ──────────────────────────────────────────────────────
        if cmd == "step":
            if self.controller is None:
                return {"status": "error", "msg": "Call reset first"}

            action      = msg.get("action", "Wait")
            object_type = msg.get("object", "")
            target      = msg.get("target", "")

            skill_fn = SKILLS.get(action)
            if not skill_fn:
                return {
                    "status":  "error",
                    "msg":     f"Unknown action '{action}'. Available: {list(SKILLS.keys())}",
                    "obs":          build_obs(_e := self.controller.step("Pass"), self.house_rooms),
                    "robot_state":  build_robot_state(_e, self.house_rooms),
                    "env_state":    build_env_state(_e, self.house_rooms),
                    "success": False,
                    "reward":  0.0,
                    "done":    False,
                }
            try:
                log.info(f"Executing: {action} {object_type}")
                ctx = {
                    "found_type": self.found_type,
                    "found_id":   self.found_id,
                    "arrived":    self.arrived,
                }
                success, message = skill_fn(
                    self.controller,
                    object_type=object_type,
                    house_rooms=self.house_rooms,
                    ctx=ctx,
                )
                # Persist updated found/arrived
                self.found_type = ctx.get("found_type")
                self.found_id   = ctx.get("found_id")
                self.arrived    = ctx.get("arrived")

                event  = self.controller.step("Pass")
                reward = 1.0 if success else 0.0
                self.total_reward += reward
                self.step_count   += 1
                log.info(f"{'OK' if success else 'FAIL'}: {message}")
                return {
                    "status":               "ok",
                    "obs":                  build_obs(event, self.house_rooms, self.found_type, self.arrived),
                    "robot_state":          build_robot_state(event, self.house_rooms, self.found_type, self.arrived),
                    "env_state":            build_env_state(event, self.house_rooms),
                    "success":              success,
                    "reward":               reward,
                    "total_reward":         self.total_reward,
                    "done":                 False,
                    "msg":                  message,
                    "step_count":           self.step_count,
                    "found":                self.found_type,
                    "arrived":              self.arrived,
                    "visible_objects":      get_visible_objects(event),
                    "visible_objects_meta": get_visible_objects_meta(event),
                }
            except Exception as exc:
                import traceback
                log.error(f"Exception during step: {exc}")
                return {
                    "status":  "error",
                    "msg":     str(exc),
                    "obs":     "",
                    "success": False,
                    "reward":  0.0,
                    "done":    False,
                }

        # ── get_frame ─────────────────────────────────────────────────
        if cmd == "get_frame":
            if self.controller is None:
                return {"status": "error", "msg": "No controller — call reset first"}
            try:
                import cv2
                event     = self.controller.step("Pass")
                frame_bgr = cv2.cvtColor(event.frame, cv2.COLOR_RGB2BGR)
                _, buf    = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return {
                    "status":       "ok",
                    "frame":        base64.b64encode(buf).decode(),
                    "total_reward": self.total_reward,
                    "step_count":   self.step_count,
                }
            except Exception as exc:
                return {"status": "error", "msg": str(exc)}

        # ── get_objects ───────────────────────────────────────────────
        if cmd == "get_objects":
            if self.controller is None:
                return {"status": "error", "msg": "No controller"}
            event   = self.controller.step("Pass")
            objects = [
                {
                    "objectType":  o["objectType"],
                    "objectId":    o["objectId"],
                    "visible":     o.get("visible", False),
                    "pickupable":  o.get("pickupable", False),
                    "openable":    o.get("openable", False),
                    "toggleable":  o.get("toggleable", False),
                    "receptacle":  o.get("receptacle", False),
                    "isOpen":      o.get("isOpen", False),
                    "isToggled":   o.get("isToggled", False),
                    "isPickedUp":  o.get("isPickedUp", False),
                    "parentReceptacles":   o.get("parentReceptacles") or [],
                    "receptacleObjectIds": o.get("receptacleObjectIds") or [],
                    "distance":    round(o.get("distance", 999.0), 2),
                    "position":    o.get("position", {}),
                }
                for o in event.metadata["objects"]
            ]
            return {
                "status":               "ok",
                "objects":              objects,
                "scene":                self.current_scene,
                "obs":                  build_obs(event, self.house_rooms),
                "visible_objects":      get_visible_objects(event),
                "visible_objects_meta": get_visible_objects_meta(event),
            }

        # ── nav ───────────────────────────────────────────────────────
        if cmd == "nav":
            if self.controller is None:
                return {"status": "error", "msg": "No controller"}
            VALID_NAV = {
                "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
                "RotateLeft", "RotateRight", "LookUp", "LookDown",
            }
            action = msg.get("action", "MoveAhead")
            if action not in VALID_NAV:
                return {"status": "error", "msg": f"Invalid nav action '{action}'. Valid: {sorted(VALID_NAV)}"}
            event = self.controller.step(action)
            return {
                "status":               "ok",
                "action":               action,
                "success":              event.metadata["lastActionSuccess"],
                "obs":                  build_obs(event, self.house_rooms),
                "visible_objects":      get_visible_objects(event),
                "visible_objects_meta": get_visible_objects_meta(event),
                "position":             event.metadata["agent"]["position"],
                "rotation":             event.metadata["agent"]["rotation"],
            }

        # ── get_state ─────────────────────────────────────────────────
        if cmd == "get_state":
            if self.controller is None:
                return {"status": "error", "msg": "No controller"}
            event = self.controller.step("Pass")
            agent = event.metadata["agent"]
            held  = [o["objectType"] for o in event.metadata["objects"] if o.get("isPickedUp")]
            return {
                "status":       "ok",
                "position":     agent["position"],
                "rotation":     agent["rotation"],
                "held_objects": held,
                "step_count":   self.step_count,
                "total_reward": self.total_reward,
                "sim_type":     self.sim_type,
                "scene":        self.current_scene,
            }

        return {"status": "error", "msg": f"Unknown command '{cmd}'"}

    # ── main loop ─────────────────────────────────────────────────────

    def run(self):
        log.info("Ready. Waiting for requests…")
        while True:
            try:
                raw  = self.socket.recv_string()
                msg  = json.loads(raw)
                resp = self.handle(msg)
                self.socket.send_string(json.dumps(to_python(resp)))
            except KeyboardInterrupt:
                log.info("Shutting down.")
                break
            except Exception as exc:
                log.error(f"Unhandled error: {exc}")
                try:
                    self.socket.send_string(json.dumps({"status": "error", "msg": str(exc)}))
                except Exception:
                    pass

        self._stop_controller()
        self.ctx.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified AI2-THOR ZMQ server")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port (default: 5555)")
    args = parser.parse_args()
    ThorServer(port=args.port).run()