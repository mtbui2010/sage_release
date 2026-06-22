#!/usr/bin/env python3
"""Capture real AI2-THOR observation frames for the SAGE concept figures.

Renders a small storyboard of the "cook a potato and put it in the recycle bin"
scenario in FloorPlan1 (kitchen), saving RGB frames to images/frames/*.png:

  overview        kitchen establishing shot
  fridge_closed   agent facing a CLOSED fridge  (why a naive Pick fails)
  fridge_open     after Open(fridge)             (the EDIT inserts this)
  hold            agent holding the potato       (after Pick succeeds)
  placed          potato dropped in the garbage/recycle can (task done)

Best-effort: if any single action fails we still save the current frame and move
on, so the script always produces something usable. Run with:

  DISPLAY=:0 python scripts/capture_thor_frames.py
"""
import math
import os

import numpy as np
from PIL import Image

from ai2thor.controller import Controller

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "images", "frames")
os.makedirs(OUT, exist_ok=True)


def save(event, name):
    arr = event.frame  # HxWx3 uint8 RGB
    Image.fromarray(arr).save(os.path.join(OUT, name + ".png"))
    print("  saved", name, arr.shape)


def obj(event, otype):
    for o in event.metadata["objects"]:
        if o["objectType"] == otype:
            return o
    return None


def face(ctrl, target_pos, horizon=0, standoff=0.0):
    """Teleport the agent to a reachable position ~`standoff` metres from
    `target_pos` (0 = nearest) and rotate to look *exactly* at it. `horizon` > 0
    tilts the camera down (useful for low objects like a floor garbage can)."""
    ev = ctrl.step(action="GetReachablePositions")
    pts = ev.metadata["actionReturn"]

    def d(p):
        return math.hypot(p["x"] - target_pos["x"], p["z"] - target_pos["z"])

    if standoff > 0:
        best = min(pts, key=lambda p: abs(d(p) - standoff))
    else:
        best = min(pts, key=lambda p: (p["x"] - target_pos["x"]) ** 2 + (p["z"] - target_pos["z"]) ** 2)
    dx, dz = target_pos["x"] - best["x"], target_pos["z"] - best["z"]
    yaw = (math.degrees(math.atan2(dx, dz))) % 360  # exact, not snapped to 90
    ev = ctrl.step(action="Teleport", position=best, rotation=dict(x=0, y=yaw, z=0),
                   horizon=horizon, standing=True)
    return ev


def main():
    ctrl = Controller(
        scene="FloorPlan1", agentMode="default", visibilityDistance=1.5,
        gridSize=0.25, snapToGrid=True, renderDepthImage=False,
        renderInstanceSegmentation=False, width=900, height=600, fieldOfView=90,
    )
    try:
        ev = ctrl.step(action="Pass")
        save(ev, "overview")

        fridge = obj(ev, "Fridge")
        if fridge:
            ev = face(ctrl, fridge["position"])
            save(ev, "fridge_closed")
            ev = ctrl.step(action="OpenObject", objectId=fridge["objectId"], forceAction=True)
            if not ev.metadata["lastActionSuccess"]:
                print("  open failed:", ev.metadata["errorMessage"][:80])
            save(ev, "fridge_open")

        # Pick a potato (look around the fridge / counters for it).
        potato = obj(ev, "Potato")
        if potato:
            ev = face(ctrl, potato["position"])
            ev = ctrl.step(action="PickupObject", objectId=potato["objectId"], forceAction=True)
            if not ev.metadata["lastActionSuccess"]:
                print("  pick failed:", ev.metadata["errorMessage"][:80])
            save(ev, "hold")

        # Place into the garbage/recycle can.
        can = obj(ev, "GarbageCan")
        if can:
            ev = face(ctrl, can["position"], horizon=20, standoff=1.3)  # stand back, look down
            held = ev.metadata.get("inventoryObjects") or []
            if held:
                ev = ctrl.step(action="PutObject", objectId=can["objectId"], forceAction=True)
                if not ev.metadata["lastActionSuccess"]:
                    print("  put failed:", ev.metadata["errorMessage"][:80])
            save(ev, "placed")
    finally:
        ctrl.stop()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
