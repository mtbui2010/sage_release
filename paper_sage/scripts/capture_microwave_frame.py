#!/usr/bin/env python3
"""Capture a 'potato in the microwave' AI2-THOR frame to replace the stale
garbage-can frame in the SAGE figures (task changed: recycle bin -> microwave).

Same scene/camera settings as capture_thor_frames.py for visual consistency.
Saves images/frames/microwave.png (microwave open, potato inside).

Run:  DISPLAY=:0 /home/keti/miniconda3/bin/python scripts/capture_microwave_frame.py
"""
import math, os
from PIL import Image
from ai2thor.controller import Controller

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "images", "frames")
os.makedirs(OUT, exist_ok=True)


def obj(ev, t):
    for o in ev.metadata["objects"]:
        if o["objectType"] == t:
            return o
    return None


def face(ctrl, pos, horizon=0, standoff=0.0):
    ev = ctrl.step(action="GetReachablePositions")
    pts = ev.metadata["actionReturn"]
    d = lambda p: math.hypot(p["x"] - pos["x"], p["z"] - pos["z"])
    best = (min(pts, key=lambda p: abs(d(p) - standoff)) if standoff > 0
            else min(pts, key=lambda p: (p["x"]-pos["x"])**2 + (p["z"]-pos["z"])**2))
    dx, dz = pos["x"] - best["x"], pos["z"] - best["z"]
    yaw = math.degrees(math.atan2(dx, dz)) % 360
    return ctrl.step(action="Teleport", position=best, rotation=dict(x=0, y=yaw, z=0),
                     horizon=horizon, standing=True)


def save(ev, name):
    Image.fromarray(ev.frame).save(os.path.join(OUT, name + ".png"))
    print("  saved", name, ev.frame.shape)


def main():
    ctrl = Controller(scene="FloorPlan1", agentMode="default", visibilityDistance=1.5,
                      gridSize=0.25, snapToGrid=True, width=900, height=600, fieldOfView=90)
    try:
        ev = ctrl.step(action="Pass")
        # grab a potato first
        potato = obj(ev, "Potato")
        if potato:
            ev = face(ctrl, potato["position"])
            ev = ctrl.step(action="PickupObject", objectId=potato["objectId"], forceAction=True)
            print("  pick potato:", ev.metadata["lastActionSuccess"])
        mw = obj(ev, "Microwave")
        if not mw:
            print("  NO microwave in scene!"); return
        ev = face(ctrl, mw["position"], horizon=10, standoff=1.0)
        ev = ctrl.step(action="OpenObject", objectId=mw["objectId"], forceAction=True)
        print("  open microwave:", ev.metadata["lastActionSuccess"])
        held = ev.metadata.get("inventoryObjects") or []
        if held:
            ev = ctrl.step(action="PutObject", objectId=mw["objectId"], forceAction=True)
            print("  put potato in microwave:", ev.metadata["lastActionSuccess"])
        # re-face for a clean composed shot of the open microwave with potato inside
        ev = face(ctrl, mw["position"], horizon=8, standoff=0.9)
        save(ev, "microwave")
        # also a closed+on variant ("heated")
        ev = ctrl.step(action="CloseObject", objectId=mw["objectId"], forceAction=True)
        ev = ctrl.step(action="ToggleObjectOn", objectId=mw["objectId"], forceAction=True)
        ev = face(ctrl, mw["position"], horizon=8, standoff=0.9)
        save(ev, "microwave_on")
    finally:
        ctrl.stop()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
