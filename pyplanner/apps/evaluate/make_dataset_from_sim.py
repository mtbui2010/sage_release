"""
make_dataset_from_sim.py
========================
Generate an evaluation dataset where every visible_objects list and
obs string comes directly from AI2-THOR — not hand-written.

For reference_steps, two strategies are available (--ref-mode):
  "manual"  — use the hand-crafted steps from make_dataset.py as a starting
               point, then validate each step's objects against the real scene.
               Fast. Steps may still be imperfect but objects are guaranteed real.

  "llm"     — call the LLM to write reference_steps given the real scene context.
               Slower but fully grounded. Requires Ollama/OpenAI/Anthropic.

Output: eval_dataset_sim.json  (drop-in replacement for eval_dataset.json)

Usage:
    # Requires: python thor_server.py  running in a separate terminal

    # Validate & enrich existing hand-written dataset (fast):
    python make_dataset_from_sim.py

    # Fully regenerate reference steps with LLM:
    python make_dataset_from_sim.py --ref-mode llm --model llama3.2

    # Custom scenes only:
    python make_dataset_from_sim.py --scenes FloorPlan1 FloorPlan2

    # Dry-run: show what would be generated without writing file:
    python make_dataset_from_sim.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# ── Resolve paths ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from thor_app.sim_client import ThorClient
except ImportError:
    sys.path.insert(0, _HERE)
    from thor_app.sim_client import ThorClient

try:
    import pyplanner
    from pyplanner.base import ROBOT_ACTIONS
    from pyplanner import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_BACKEND
except ImportError:
    sys.path.insert(0, os.path.join(_HERE, "..", "pyplanner"))
    import pyplanner
    from pyplanner import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_BACKEND
    from pyplanner.base import ROBOT_ACTIONS

VALID_ACTIONS = set(ROBOT_ACTIONS)

try:
    from make_dataset import SAMPLES_RAW, VALID_ACTIONS as _VA
    VALID_ACTIONS = _VA
except ImportError:
    sys.path.insert(0, _HERE)
    from make_dataset import SAMPLES_RAW, VALID_ACTIONS as _VA
    VALID_ACTIONS = _VA


# ═════════════════════════════════════════════════════════════════════
# Scene inspector: load a scene and extract full object metadata
# ═════════════════════════════════════════════════════════════════════

def inspect_scene(client: ThorClient, scene: str) -> dict:
    """
    Reset simulator to scene and return full object inventory.

    Returns:
        {
          "scene":           str,
          "obs":             str,             # human-readable observation
          "visible_objects": list[str],       # objectTypes that are visible
          "all_objects":     list[dict],      # full metadata for all objects
          "pickupable":      list[str],       # objectTypes that can be picked up
          "openable":        list[str],       # objectTypes that can be opened
          "toggleable":      list[str],       # objectTypes that can be toggled on/off
          "receptacles":     list[str],       # objectTypes that can receive objects
          "object_map":      dict[str, dict], # objectType -> metadata (nearest instance)
        }
    """
    # Reset to scene
    resp = client.reset(scene)
    if resp.get("status") != "ok":
        raise RuntimeError(f"Cannot load scene {scene}: {resp.get('msg','')}")

    # Get full object list
    obj_resp = client.get_objects()
    if obj_resp.get("status") != "ok":
        raise RuntimeError(f"get_objects failed: {obj_resp.get('msg','')}")

    objects = obj_resp["objects"]

    # Build category sets (deduplicated by objectType, keep nearest instance)
    object_map: dict[str, dict] = {}
    for o in objects:
        t = o["objectType"]
        if t not in object_map or o["distance"] < object_map[t]["distance"]:
            object_map[t] = o

    visible    = sorted({o["objectType"] for o in objects if o["visible"]})
    pickupable = sorted({t for t, o in object_map.items() if o["pickupable"]})
    openable   = sorted({t for t, o in object_map.items() if o["openable"]})
    toggleable = sorted({t for t, o in object_map.items() if o["toggleable"]})
    receptacles= sorted({t for t, o in object_map.items() if o["receptacle"]})

    return {
        "scene":           scene,
        "obs":             obj_resp.get("obs", resp.get("obs", "")),
        "visible_objects": obj_resp.get("visible_objects", visible),
        "all_objects":     objects,
        "pickupable":      pickupable,
        "openable":        openable,
        "toggleable":      toggleable,
        "receptacles":     receptacles,
        "object_map":      object_map,
    }


# ═════════════════════════════════════════════════════════════════════
# Step validator: check each reference step against real scene objects
# ═════════════════════════════════════════════════════════════════════

def _snake_to_camel(name: str) -> str:
    """coffee_machine → CoffeeMachine"""
    return "".join(w.capitalize() for w in name.replace("-", "_").split("_"))


def _find_real_object(name: str, object_map: dict) -> str | None:
    """
    Try to match a step object name (snake_case or free text) to a real
    objectType in the scene. Returns matched objectType or None.

    Matching strategy (in order):
      1. Exact match (case-insensitive)
      2. CamelCase conversion: coffee_machine → CoffeeMachine
      3. Substring match: "stove" matches "StoveBurner"
      4. Reverse substring: "alarm_clock" in "AlarmClock"
    """
    if not name:
        return ""   # empty object is fine (e.g. Wait action)

    name_lower = name.lower().replace("_", "").replace(" ", "")

    for ot in object_map:
        if ot.lower() == name_lower:
            return ot

    camel = _snake_to_camel(name).lower()
    for ot in object_map:
        if ot.lower() == camel:
            return ot

    # Substring both ways
    for ot in object_map:
        ot_lower = ot.lower()
        if name_lower in ot_lower or ot_lower in name_lower:
            return ot

    return None


def validate_steps_against_scene(
    steps: list[dict],
    scene_info: dict,
) -> tuple[list[dict], list[str]]:
    """
    Validate reference steps, replacing object names with real scene names.

    Returns:
        (validated_steps, warnings)
        validated_steps — steps with object/target names mapped to real objectTypes
        warnings        — list of issues found
    """
    object_map = scene_info["object_map"]
    validated  = []
    warnings   = []

    for i, step in enumerate(steps):
        action = step.get("action", "")
        obj    = step.get("object", "")
        target = step.get("target", "")
        reason = step.get("reason", "")

        # Action must be valid
        if action not in VALID_ACTIONS:
            warnings.append(f"Step {i+1}: unknown action '{action}'")
            validated.append(step)
            continue

        # Skip object validation for Wait
        if action == "Wait":
            validated.append({"action": action, "object": "", "target": "", "reason": reason})
            continue

        # Resolve object
        real_obj = _find_real_object(obj, object_map)
        if real_obj is None:
            warnings.append(
                f"Step {i+1} ({action}): object '{obj}' not found in scene "
                f"'{scene_info['scene']}' — keeping original name"
            )
            real_obj = obj   # keep original if not found

        # Resolve target (for Place/PutIn)
        real_target = ""
        if target:
            real_target = _find_real_object(target, object_map)
            if real_target is None:
                warnings.append(
                    f"Step {i+1} ({action}): target '{target}' not found in scene"
                )
                real_target = target

        validated.append({
            "action": action,
            "object": real_obj,
            "target": real_target,
            "reason": reason,
        })

    return validated, warnings


# ═════════════════════════════════════════════════════════════════════
# LLM reference step generator
# ═════════════════════════════════════════════════════════════════════

LLM_REF_SYSTEM = """You are a robot task planner writing GROUND TRUTH reference plans.
You will be given:
  - A task description
  - The EXACT list of objects available in the scene (objectType names from AI2-THOR)
  - Their properties (pickupable, openable, toggleable, receptacle)

Write the minimal correct action plan using ONLY objects from the provided list.
Use exact objectType names — no snake_case, no abbreviations.

Available actions: Navigate, Find, Grab, Place, PutIn, Open, Close, TurnOn, TurnOff, Wash, Sit, LieOn, Serve, Wait

Rules:
- Always Navigate or Find BEFORE interacting with an object
- Use the exact objectType name from the list (e.g. "CoffeeMachine" not "coffee_machine")
- Maximum 12 steps
- Only include steps that are actually needed

Return ONLY valid JSON, no markdown:
{"steps": [
  {"action": "MoveTo", "object": "Mug", "target": "", "reason": "Move to the mug"},
  {"action": "Pick",     "object": "Mug", "target": "", "reason": "Pick up the mug"}
]}"""


def generate_llm_reference_steps(
    task_desc:    str,
    scene_info:   dict,
    planner_host: str,
    planner_model:str,
    provider:     str,
    api_key:      str,
) -> tuple[list[dict], str]:
    """
    Ask LLM to write reference_steps using the real scene object list.
    Returns (steps, raw_response).
    """
    from pyplanner.base import LLMBackend, parse_steps

    # Build a rich object description for the prompt
    obj_lines = []
    om = scene_info["object_map"]
    for ot, meta in sorted(om.items()):
        props = []
        if meta["pickupable"]:  props.append("pickupable")
        if meta["openable"]:    props.append("openable")
        if meta["toggleable"]:  props.append("toggleable")
        if meta["receptacle"]:  props.append("receptacle")
        if meta["visible"]:     props.append("visible")
        prop_str = f" [{', '.join(props)}]" if props else ""
        obj_lines.append(f"  {ot}{prop_str}")

    obj_block = "\n".join(obj_lines)

    user_msg = (
        f"Task: {task_desc}\n\n"
        f"Scene: {scene_info['scene']}\n"
        f"Current observation: {scene_info['obs'][:300]}\n\n"
        f"Objects in scene:\n{obj_block}\n\n"
        f"Write the minimal reference plan using ONLY objects from the list above:"
    )

    backend = LLMBackend(provider=provider, model=planner_model,
                         host=planner_host, api_key=api_key)
    content, _, _ = backend.chat(
        [{"role": "system", "content": LLM_REF_SYSTEM},
         {"role": "user",   "content": user_msg}],
        temperature=0.1,
    )
    steps = parse_steps(content)
    return steps, content


# ═════════════════════════════════════════════════════════════════════
# Main dataset builder
# ═════════════════════════════════════════════════════════════════════

def build_dataset_from_sim(
    client:        ThorClient,
    ref_mode:      str = "manual",   # "manual" or "llm"
    scenes:        list[str] | None = None,
    planner_host:  str = DEFAULT_HOST,
    planner_model: str = DEFAULT_MODEL,
    provider:      str = DEFAULT_BACKEND,
    api_key:       str = "",
    verbose:       bool = False,
) -> tuple[list[dict], dict]:
    """
    Build dataset by loading each scene in AI2-THOR and grounding objects.

    Returns: (samples, stats)
    """
    # Collect unique scenes needed
    if scenes is not None:
        scenes_to_load = set(scenes)
    else:
        scenes_to_load = set(r["scene"] for r in SAMPLES_RAW)

    # ── Step 1: Inspect all scenes ──────────────────────────────────
    print(f"\n🔍  Inspecting {len(scenes_to_load)} scenes in AI2-THOR...")
    scene_cache: dict[str, dict] = {}

    for i, scene in enumerate(sorted(scenes_to_load)):
        print(f"  [{i+1}/{len(scenes_to_load)}] Loading {scene}...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            scene_cache[scene] = inspect_scene(client, scene)
            elapsed = time.perf_counter() - t0
            n_obj = len(scene_cache[scene]["object_map"])
            n_vis = len(scene_cache[scene]["visible_objects"])
            print(f"✅  {n_obj} objects  {n_vis} visible  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"❌  {e}")

    # ── Step 2: Build samples ───────────────────────────────────────
    print(f"\n📋  Building samples ({ref_mode} reference mode)...")
    samples     = []
    all_warnings: list[str] = []
    stats = defaultdict(int)

    for raw in SAMPLES_RAW:
        scene = raw["scene"]

        # Skip if scene wasn't loaded (or filtered out)
        if scene not in scene_cache:
            if verbose:
                print(f"  ⚠  Skip {raw['task_id']}: scene {scene} not in cache")
            stats["skipped"] += 1
            continue

        info = scene_cache[scene]

        # ── Ground visible_objects from real scene ──
        # Use the actual visible objects from AI2-THOR reset
        real_visible = info["visible_objects"]

        # ── Ground obs from real scene ──
        real_obs = info["obs"]

        # ── Ground expected_objects ──
        # Map hand-written expected object names to real scene objectTypes
        real_expected = []
        for exp_obj in raw["expected_objects"]:
            real_name = _find_real_object(exp_obj, info["object_map"])
            if real_name:
                real_expected.append(real_name)
            else:
                real_expected.append(exp_obj)   # keep original if not found
                all_warnings.append(
                    f"{raw['task_id']}: expected '{exp_obj}' not in scene {scene}"
                )

        # ── Reference steps ──
        if ref_mode == "llm":
            print(f"  🧠  {raw['task_id']}: generating ref steps via LLM...", end=" ", flush=True)
            try:
                ref_steps, _ = generate_llm_reference_steps(
                    task_desc    = raw["task_desc"],
                    scene_info   = info,
                    planner_host = planner_host,
                    planner_model= planner_model,
                    provider     = provider,
                    api_key      = api_key,
                )
                print(f"✅  {len(ref_steps)} steps")
                stats["llm_generated"] += 1
            except Exception as e:
                print(f"⚠  LLM failed ({e}), falling back to manual")
                ref_steps = raw["reference_steps"]
                stats["llm_fallback"] += 1
        else:
            ref_steps = raw["reference_steps"]

        # ── Validate + remap reference_steps object names ──
        validated_steps, step_warnings = validate_steps_against_scene(ref_steps, info)
        if step_warnings:
            for w in step_warnings:
                all_warnings.append(f"{raw['task_id']}: {w}")
            stats["steps_with_warnings"] += 1
        else:
            stats["steps_clean"] += 1

        if verbose and step_warnings:
            for w in step_warnings:
                print(f"    ⚠  {w}")

        # ── Build sample ──
        sample = {
            "task_id":          raw["task_id"],
            "task_desc":        raw["task_desc"],
            "room":             raw["room"],
            "scene":            scene,
            "obs":              real_obs,                          # ← from real scene
            "visible_objects":  real_visible,                     # ← from real scene
            "reference_steps":  validated_steps,                  # ← validated/remapped
            "expected_objects": real_expected,                    # ← remapped to real names
            "difficulty":       raw["difficulty"],
            "fail_injection":   raw.get("fail_injection") or {},
            # Provenance metadata
            "_meta": {
                "ref_mode":          ref_mode,
                "scene_n_objects":   len(info["object_map"]),
                "scene_n_visible":   len(real_visible),
                "step_warnings":     len(step_warnings),
                "grounded_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }
        samples.append(sample)
        stats["ok"] += 1

    stats["total_warnings"] = len(all_warnings)
    return samples, dict(stats), all_warnings


# ═════════════════════════════════════════════════════════════════════
# Diff: compare hand-written vs grounded dataset
# ═════════════════════════════════════════════════════════════════════

def diff_datasets(manual: list[dict], grounded: list[dict]) -> list[dict]:
    """
    Show which visible_objects and step objects changed between
    the hand-written and grounded versions.
    """
    manual_map   = {s["task_id"]: s for s in manual}
    diffs = []
    for s in grounded:
        tid = s["task_id"]
        orig = manual_map.get(tid)
        if not orig:
            continue
        d = {"task_id": tid, "scene": s["scene"], "changes": []}

        # Visible objects diff
        orig_vis = set(orig["visible_objects"])
        new_vis  = set(s["visible_objects"])
        if orig_vis != new_vis:
            added   = sorted(new_vis - orig_vis)
            removed = sorted(orig_vis - new_vis)
            if added:
                d["changes"].append(f"visible+: {added}")
            if removed:
                d["changes"].append(f"visible-: {removed}")

        # Step object diff
        orig_objs = {st["object"] for st in orig["reference_steps"] if st["object"]}
        new_objs  = {st["object"] for st in s["reference_steps"]  if st["object"]}
        renamed = []
        for oo in orig_objs:
            if oo not in new_objs:
                # find what it was renamed to
                matches = [no for no in new_objs if oo.lower().replace("_","") in no.lower()]
                if matches:
                    renamed.append(f"{oo} → {matches[0]}")
        if renamed:
            d["changes"].append(f"remapped: {renamed}")

        if d["changes"]:
            diffs.append(d)

    return diffs


# ═════════════════════════════════════════════════════════════════════
# Summary printer
# ═════════════════════════════════════════════════════════════════════

def print_summary(samples: list[dict], stats: dict, warnings: list[str]):
    from collections import Counter
    diff_cnt  = Counter(s["difficulty"] for s in samples)
    room_cnt  = Counter(s["room"]       for s in samples)
    avg_vis   = sum(len(s["visible_objects"])  for s in samples) / len(samples) if samples else 0
    avg_obj   = sum(s["_meta"]["scene_n_objects"] for s in samples) / len(samples) if samples else 0
    avg_steps = sum(len(s["reference_steps"]) for s in samples) / len(samples) if samples else 0

    print(f"\n{'─'*55}")
    print(f"  Dataset summary — {len(samples)} grounded samples")
    print(f"{'─'*55}")
    print(f"  Difficulty:    easy={diff_cnt['easy']}  medium={diff_cnt['medium']}  hard={diff_cnt['hard']}")
    print(f"  Rooms:         {dict(room_cnt)}")
    print(f"  Avg visible:   {avg_vis:.1f} objects/scene (from real AI2-THOR)")
    print(f"  Avg total:     {avg_obj:.1f} objects/scene (all in scene)")
    print(f"  Avg ref steps: {avg_steps:.1f}")
    print(f"  Ref mode:      {samples[0]['_meta']['ref_mode'] if samples else 'N/A'}")
    print(f"  Stats:         {stats}")
    if warnings:
        print(f"\n  ⚠  {len(warnings)} warnings:")
        for w in warnings[:10]:
            print(f"     {w}")
        if len(warnings) > 10:
            print(f"     ... and {len(warnings)-10} more")
    print(f"{'─'*55}\n")


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════


def _check_llm_connection(provider: str, host: str, model: str, api_key: str) -> tuple[bool, str]:
    """
    Verify the LLM backend is reachable before starting generation.
    Returns (ok, message). Mirrors the logic in evaluate.py check_connection().
    """
    import urllib.request, urllib.error

    if provider == "ollama":
        try:
            url = host.rstrip("/") + "/api/tags"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            available = [m["name"] for m in data.get("models", [])]
            model_base = model.split(":")[0]
            if not any(model_base in m for m in available):
                hint = f"\n  Available: {available[:6]}" if available else "\n  No models pulled yet"
                close = [m for m in available if model_base in m]
                if close:
                    hint = f"\n  Did you mean: {close[0]}?" + hint
                return False, (
                    f"Model \"{model}\" not found in Ollama.{hint}\n"
                    f"  Pull it:  ollama pull {model}"
                )
            return True, f"Ollama OK — model \"{model}\" available"
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", str(e))
            return False, (
                f"Cannot reach Ollama at {host}\n"
                f"  Error  : {reason}\n"
                f"  Fix    : run  ollama serve\n"
                f"  Or set : --host http://<your-ollama-server>:11434"
            )
        except Exception as e:
            return False, f"Ollama check failed: {e}"

    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            return False, (
                "OpenAI API key not set.\n"
                "  Fix: --api-key sk-...  or  export OPENAI_API_KEY=sk-..."
            )
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
            return True, "OpenAI API key valid"
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return False, "OpenAI API key invalid (HTTP 401 Unauthorized)."
            return False, f"OpenAI error: HTTP {e.code}"
        except Exception as e:
            return False, f"OpenAI connection error: {e}"

    elif provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not key:
            return False, (
                "Gemini API key not set.\n"
                "  Fix: --api-key AIza...  or  export GEMINI_API_KEY=AIza..."
            )
        return True, "Gemini key present"

    return True, f"Unknown provider '{provider}'"


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground-truth evaluation dataset from AI2-THOR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sim-host",   default="localhost")
    parser.add_argument("--sim-port",   type=int, default=5555)
    parser.add_argument("--ref-mode",   default="manual", choices=["manual", "llm"],
                        help="How to generate reference_steps: 'manual' (validate existing) "
                             "or 'llm' (regenerate with LLM from real scene)")
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--host",       default=DEFAULT_HOST)
    parser.add_argument("--provider",   default=DEFAULT_BACKEND,
                        choices=["ollama","openai","gemini"])
    parser.add_argument("--api-key",    default="")
    parser.add_argument("--scenes",     nargs="+", default=None,
                        help="Only process specific scenes, e.g. FloorPlan1 FloorPlan2")
    parser.add_argument("--out",        default="eval_dataset_sim.json")
    parser.add_argument("--diff",       action="store_true",
                        help="Show diff between hand-written and grounded datasets")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print what would be generated, don't write file")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    # ── Simulator check ──────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  Dataset Generator — AI2-THOR grounded")
    print(f"{'═'*55}")
    print(f"  Simulator : {args.sim_host}:{args.sim_port}")
    print(f"  Ref mode  : {args.ref_mode}")
    if args.ref_mode == "llm":
        print(f"  LLM       : {args.provider}/{args.model}")

    # ── Simulator connection ─────────────────────────────────────────
    print(f"\n🔌  Connecting to ThorServer...")
    try:
        client = ThorClient(host=args.sim_host, port=args.sim_port)
        if not client.connected:
            print(f"  ❌  ThorServer not responding at {args.sim_host}:{args.sim_port}")
            print(f"      Fix: run  python thor_server.py  in a separate terminal")
            sys.exit(1)
        print(f"  ✅  Simulator connected")
    except Exception as e:
        print(f"  ❌  Connection failed: {e}")
        print(f"      Fix: run  python thor_server.py  in a separate terminal")
        sys.exit(1)

    # ── LLM connection check (only needed for --ref-mode llm) ─────────
    if args.ref_mode == "llm":
        print(f"\n🧠  Checking LLM connection ({args.provider}/{args.model})...")
        ok, msg = _check_llm_connection(args.provider, args.host, args.model, args.api_key)
        if ok:
            print(f"  ✅  {msg}")
        else:
            print(f"\n{'═'*60}")
            print(f"  ❌  LLM CONNECTION FAILED")
            print(f"{'─'*60}")
            for line in msg.splitlines():
                print(f"  {line}")
            print(f"{'═'*60}")
            print(f"  Tip: use --ref-mode manual to skip LLM entirely\n")
            sys.exit(1)

    # ── Build dataset ────────────────────────────────────────────────
    t_start = time.perf_counter()
    samples, stats, warnings = build_dataset_from_sim(
        client        = client,
        ref_mode      = args.ref_mode,
        scenes        = args.scenes,
        planner_host  = args.host,
        planner_model = args.model,
        provider      = args.provider,
        api_key       = args.api_key,
        verbose       = args.verbose,
    )
    elapsed = time.perf_counter() - t_start

    print_summary(samples, stats, warnings)

    # ── Diff ─────────────────────────────────────────────────────────
    if args.diff:
        try:
            from make_dataset import build_dataset as build_manual
            manual_samples = build_manual()
            diffs = diff_datasets(manual_samples, samples)
            if diffs:
                print(f"  Changes vs hand-written dataset ({len(diffs)} samples changed):")
                for d in diffs:
                    print(f"    [{d['task_id']}] {d['scene']}")
                    for c in d["changes"]:
                        print(f"      {c}")
            else:
                print("  ✅  No differences from hand-written dataset")
        except Exception as e:
            print(f"  ⚠  Diff failed: {e}")

    if args.dry_run:
        print(f"  Dry-run — not writing file. {len(samples)} samples in {elapsed:.1f}s\n")
        return

    # ── Write output ──────────────────────────────────────────────────
    out = {
        "version":    "1.0",
        "grounded":   True,
        "ref_mode":   args.ref_mode,
        "sim_host":   args.sim_host,
        "generated":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "samples":    samples,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(args.out) // 1024
    print(f"✅  Saved {len(samples)} samples → {args.out}  ({size_kb} KB)  [{elapsed:.1f}s]\n")


if __name__ == "__main__":
    main()