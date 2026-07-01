#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# eai_export.py — OFFLINE exporter: run SAGE/SAGE on the Embodied Agent
# Interface (EAI) VirtualHome `action_sequencing` track and emit
# validly-formatted LLM responses that EAI's evaluator can score.
#
# No simulator, no GPU, no Unity. The whole EAI action_sequencing track
# is scored offline against EAI's symbolic transition model. This script
# only needs:
#   - the EAI prompt file produced by:
#       eai-eval --dataset virtualhome --eval-type action_sequencing \
#                --mode generate_prompts
#   - pyplanner installed (pip install -e ../pyplanner) for SAGEPlanner.
#
# It writes responses in EAI's exact expected layout:
#       <out_dir>/virtualhome/action_sequencing/<model>_outputs.json
# where each file is a JSON LIST of:
#       {"identifier": <file_id>, "llm_output": "<json-action-string>"}
#
# Then score with:
#   eai-eval --dataset virtualhome --eval-type action_sequencing \
#            --mode evaluate_results --llm-response-path <out_dir> \
#            --output-dir results/eai_sage
#
# ── EAI response schema (verified against the EAI source) ─────────────
#   src/virtualhome_eval/evaluation/action_sequencing/scripts/
#       evaluate_results.py   → reads {model}_outputs.json; per item:
#                               file_id = item["identifier"]
#                               actions = item["llm_output"]   (a STRING)
#                               strips ```json, then
#                               load_json_preserving_order(actions)
#   simulation/evolving_graph/eval_utils.py:
#       load_json_preserving_order(s):
#           regex r'"(\w+)"\s*:\s*(\[[^\]]+\])' over the string, in textual
#           order → list[ {ACTION: [args...]} ]  (so DUPLICATE action keys
#           survive — the output must be a JSON-object-LOOKING string with
#           one "ACTION": [...] entry per step, repeats allowed).
#       check_name_id_format : each action's params list length must be EVEN
#                              (name,id pairs).
#       check_action_grammar : len(params)//2 == valid_actions[ACTION][1]
#                              (#object args). 0 / 1 / 2 object args.
#       json_to_action       : each [name,id] -> "[ACTION] <name> (gid)" via
#                              relevant_name_to_id[f"{name}_{id}"]; the id we
#                              emit MUST be an id that appears in the scene's
#                              "<object_in_scene>" list ("name (id)").
#
#   The one_shot prompt example output is a JSON object:
#       {"FIND": ["sink","sink_id"], "PUTBACK": ["cup","cup_id","sink","sink_id"]}
#
# ── SAGE → VirtualHome action map ────────────────────────────────────
#   MoveTo  -> WALK     <obj>                 (1 arg)
#   Find    -> FIND     <obj>                 (1 arg)
#   Pick    -> GRAB     <held>                (1 arg; held = state.holding)
#   Place   -> PUTBACK  <held> <recep>        (2 args)
#   PutIn   -> PUTIN    <held> <container>    (2 args)
#   Open    -> OPEN     <obj>                 (1 arg)
#   Close   -> CLOSE    <obj>                 (1 arg)
#   TurnOn  -> SWITCHON <obj>                 (1 arg)
#   TurnOff -> SWITCHOFF<obj>                 (1 arg)
#   Wash    -> WASH     <obj>                 (1 arg)
#   Sit     -> SIT      <obj>                 (1 arg)
#   LieOn   -> LIE      <obj>                 (1 arg)
#   Serve/Wait -> dropped (no VH equivalent; not affordance-bearing)
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys
import traceback
from collections import OrderedDict


# ── EAI valid_actions (#object-args), mirrored from eval_utils.valid_actions ──
# Only the second tuple element (expected #object args) matters for grammar.
VH_ARGC = {
    "DRINK": 1, "EAT": 1, "CUT": 1, "TOUCH": 1, "LOOKAT": 1,
    "WATCH": 1, "READ": 1, "TYPE": 1, "PUSH": 1, "PULL": 1, "MOVE": 1,
    "SQUEEZE": 1, "SLEEP": 0, "WAKEUP": 0, "RINSE": 1, "SCRUB": 1,
    "WASH": 1, "GRAB": 1, "SWITCHOFF": 1, "SWITCHON": 1, "CLOSE": 1,
    "FIND": 1, "WALK": 1, "OPEN": 1, "POINTAT": 1, "PUTBACK": 2,
    "PUTIN": 2, "PUTOBJBACK": 1, "RUN": 1, "SIT": 1, "STANDUP": 0,
    "TURNTO": 1, "WIPE": 1, "PUTON": 1, "PUTOFF": 1, "GREET": 1,
    "DROP": 1, "LIE": 1, "POUR": 2,
}

# SAGE action -> VirtualHome action token. (SAGE names are post-normalize_plan.)
SAGE_TO_VH = {
    "MoveTo":  "WALK",
    "Find":    "FIND",
    "Pick":    "GRAB",
    "Place":   "PUTBACK",
    "PutIn":   "PUTIN",
    "Open":    "OPEN",
    "Close":   "CLOSE",
    "TurnOn":  "SWITCHON",
    "TurnOff": "SWITCHOFF",
    "Wash":    "WASH",
    "Sit":     "SIT",
    "LieOn":   "LIE",
    # Serve / Wait have no VH counterpart -> skipped.
}


# ─────────────────────────────────────────────────────────────────────
# Prompt parsing
# ─────────────────────────────────────────────────────────────────────
# Scene objects in the EAI prompt are rendered as  "name (id)".  We collect
# every such (name, id) and build a name->id resolver so SAGE's CamelCase
# object names can be grounded back to a real scene id.
# EAI renders scene objects in TWO formats across sections:
#   "Objects in the scene":  'washing_machine, id: 1001, properties: [...]'
#   "Edges":                 '<clothes_jacket> (1003) is NEAR to <basket> (1000)'
# We harvest both so the grounder sees every (name, id) in the scene.
_OBJ_RES = [
    re.compile(r"([A-Za-z_][\w]*)\s*,\s*id:\s*(\d+)"),   # name, id: NNN
    re.compile(r"<([A-Za-z_][\w]*)>\s*\((\d+)\)"),        # <name> (NNN)
    re.compile(r"([A-Za-z_][\w]*)\s*\((\d+)\)"),          # name (NNN) fallback
]


def parse_scene_objects(prompt_text: str) -> "list[tuple[str,str]]":
    """Return ordered, de-duplicated (name, id) pairs found in the prompt,
    across both EAI rendering formats (see _OBJ_RES)."""
    seen = set()
    out: list[tuple[str, str]] = []
    for rx in _OBJ_RES:
        for name, oid in rx.findall(prompt_text):
            key = (name, oid)
            if key in seen:
                continue
            seen.add(key)
            out.append((name, oid))
    return out


# Map an EAI node-goal STATE -> an imperative SAGE clause.
_STATE_VERB = {
    "ON": "turn on {a}", "OFF": "turn off {a}",
    "OPEN": "open {a}", "CLOSED": "close {a}",
    "CLEAN": "wash {a}", "PLUGGED_IN": "turn on {a}",
}
# Map an EAI edge-goal RELATION -> an imperative SAGE clause.
_REL_VERB = {"ON": "put {a} on {b}", "INSIDE": "put {a} inside {b}"}


def _goal_block(lines: "list[str]", header: str, enders: "tuple[str,...]") -> "list[str]":
    """Return the content lines under `header` up to the first `enders` marker."""
    out, grab = [], False
    for ln in lines:
        low = ln.lower()
        if header in low:
            grab = True
            continue
        if grab and any(e in low for e in enders):
            break
        if grab and ln and not ln.startswith("-"):
            out.append(ln)
    return out


def build_task_from_prompt(prompt_text: str) -> str:
    """Synthesize an IMPERATIVE natural-language task for SAGE from EAI goals.

    EAI gives node goals ('X is STATE'), edge goals ('A is REL to B') and
    action goals. A raw concatenation makes SAGE under-plan (it expects a
    THOR-style imperative), so we translate each goal into an action clause
    SAGE can decompose, e.g. 'put clothes_jacket on washing_machine; turn on
    washing_machine'.
    """
    lines = [ln.strip() for ln in prompt_text.splitlines()]
    instrs: list[str] = []

    for ln in _goal_block(lines, "node goals are", ("edge goals are", "action goals are", "please output", "output:")):
        m = re.match(r"(.+?)\s+is\s+([A-Za-z_]+)\s*$", ln)
        if m:
            a, st = m.group(1).strip(), m.group(2).strip().upper()
            if st in _STATE_VERB:
                instrs.append(_STATE_VERB[st].format(a=a))

    for ln in _goal_block(lines, "edge goals are", ("action goals are", "please output", "output:")):
        m = re.match(r"(.+?)\s+is\s+([A-Za-z_]+)\s+to\s+(.+?)\s*$", ln)
        if m:
            a, rel, b = m.group(1).strip(), m.group(2).strip().upper(), m.group(3).strip()
            if rel in _REL_VERB:
                instrs.append(_REL_VERB[rel].format(a=a, b=b))

    for ln in _goal_block(lines, "action goals are", ("please output", "output:", "-----")):
        if "no action" in ln.lower():
            continue
        instrs.append(ln)

    # De-duplicate while preserving order.
    seen, uniq = set(), []
    for c in instrs:
        if c.lower() not in seen:
            seen.add(c.lower())
            uniq.append(c)

    if not uniq:
        return re.sub(r"\s+", " ", " ".join(l for l in lines if l)).strip()[:600]
    return "In a household scene, complete these goals: " + "; ".join(uniq) + "."


def _norm(s: str) -> str:
    """Loose normalization for matching SAGE object names to scene names."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


class Grounder:
    """Resolve a SAGE object string to a scene (name, id) pair.

    EAI's json_to_action looks up relevant_name_to_id[f"{name}_{id}"], where
    name/id are exactly what we emit and must correspond to a real scene
    object. So we must emit the scene's own name spelling + a scene id.
    """

    def __init__(self, scene_objects: "list[tuple[str,str]]"):
        self.scene = scene_objects
        self.by_norm: dict[str, tuple[str, str]] = {}
        for name, oid in scene_objects:
            self.by_norm.setdefault(_norm(name), (name, oid))

    def resolve(self, sage_obj: str) -> "tuple[str,str] | None":
        if not sage_obj:
            return None
        n = _norm(sage_obj)
        if n in self.by_norm:
            return self.by_norm[n]
        # Substring fallback: SAGE 'CoffeeTable' vs scene 'coffeetable', or
        # SAGE 'Fridge' vs scene 'fridge'/'refrigerator'.
        for key, val in self.by_norm.items():
            if n and (n in key or key in n):
                return val
        return None


# ─────────────────────────────────────────────────────────────────────
# SAGE plan -> VirtualHome action entries
# ─────────────────────────────────────────────────────────────────────
def sage_steps_to_vh_entries(steps, grounder, simulate_fn, SymbolicState):
    """Translate normalized SAGE steps into a list of (ACTION, [args]) tuples.

    Pick/Place args are reconstructed from the verifier's SymbolicState by
    replaying the plan: GRAB's object = the object being held; PUTBACK/PUTIN
    args = (held_object, receptacle).

    Returns (entries, dropped) where entries is list[(action, [name,id,...])]
    suitable to render to the EAI llm_output string. Steps whose objects
    cannot be grounded to the scene are dropped (they would crash
    json_to_action's relevant_name_to_id lookup).
    """
    from pyplanner.verifier import normalize_plan, _apply  # local import

    steps = normalize_plan(steps)
    state = SymbolicState()
    entries: list[tuple[str, list[str]]] = []
    dropped: list[dict] = []

    for s in steps:
        a = s.get("action", "")
        o = s.get("object", "") or ""
        vh = SAGE_TO_VH.get(a)
        if vh is None:
            dropped.append({"step": s, "why": "no_vh_mapping"})
            _apply(s, state)
            continue

        if a == "Pick":
            # Match verifier._apply precedence: held = explicit object else last found.
            held = o or state.found
            g = grounder.resolve(held)
            if g is None:
                dropped.append({"step": s, "why": f"ungrounded_pick:{held}"})
                _apply(s, state)
                continue
            entries.append(("GRAB", [g[0], g[1]]))

        elif a in ("Place", "PutIn"):
            held = state.holding or ""
            gh = grounder.resolve(held)
            gr = grounder.resolve(o)
            if gh is None or gr is None:
                dropped.append({"step": s, "why": f"ungrounded_place:{held}->{o}"})
                _apply(s, state)
                continue
            entries.append((vh, [gh[0], gh[1], gr[0], gr[1]]))

        else:
            # 1-object actions: WALK/FIND/OPEN/CLOSE/SWITCHON/SWITCHOFF/WASH/SIT/LIE
            g = grounder.resolve(o)
            if g is None:
                dropped.append({"step": s, "why": f"ungrounded:{o}"})
                _apply(s, state)
                continue
            entries.append((vh, [g[0], g[1]]))

        _apply(s, state)

    return entries, dropped


def entries_to_llm_output(entries) -> str:
    """Render (ACTION,[args]) entries as the EAI llm_output STRING.

    EAI parses the string with a per-key regex that PRESERVES ORDER and
    keeps DUPLICATE keys, so we hand-build a JSON-object-looking string
    rather than json.dumps a dict (which would collapse duplicate WALKs).
    """
    parts = []
    for action, args in entries:
        arg_json = json.dumps(args)            # ["name","id"]  with double quotes
        parts.append(f'"{action}": {arg_json}')
    return "{" + ", ".join(parts) + "}"


# ─────────────────────────────────────────────────────────────────────
# Local mirror of EAI grammar validation (for --validate without importing
# the EAI package). Mirrors load_json_preserving_order + check_*.
# ─────────────────────────────────────────────────────────────────────
_KV_RE = re.compile(r'"(\w+)"\s*:\s*(\[[^\]]*\])')


def local_parse_preserving_order(s: str):
    s = re.sub(r"\s+", " ", s.strip())
    out = []
    for key, val in _KV_RE.findall(s):
        out.append({key: json.loads(val)})
    return out


def local_validate(llm_output: str) -> "tuple[bool,str|None]":
    try:
        parsed = local_parse_preserving_order(llm_output)
    except Exception as e:  # noqa: BLE001
        return False, f"parse_error:{e}"
    if not parsed:
        return False, "empty_after_parse"
    for d in parsed:
        for act, params in d.items():
            if act not in VH_ARGC:
                return False, f"unknown_action:{act}"
            params = [p for p in params if p != ""]
            if len(params) % 2 != 0:
                return False, f"name_id_odd:{act}:{params}"
            if len(params) // 2 != VH_ARGC[act]:
                return False, f"argc_mismatch:{act}:{params}!={VH_ARGC[act]}"
    return True, None


def eai_validate(llm_output: str):
    """Validate with EAI's own functions if the package is importable.

    Returns (ok, reason) or None if EAI is not importable (caller falls
    back to local_validate).
    """
    try:
        from virtualhome_eval.simulation.evolving_graph.eval_utils import (  # type: ignore
            load_json_preserving_order, check_name_id_format, check_action_grammar,
        )
    except Exception:
        return None
    s = llm_output
    if s.startswith("```json"):
        s = s[7:]
    s = s.strip().replace("\n", "").replace("'", '"')
    try:
        actions = load_json_preserving_order(s)
    except Exception as e:  # noqa: BLE001
        return False, f"eai_parse_error:{e}"
    ok, msg = check_name_id_format(actions)
    if not ok:
        return False, msg
    ok, msg = check_action_grammar(actions)
    if not ok:
        return False, msg
    return True, None


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def find_prompt_file(prompt_path: str) -> str:
    """Accept either a direct file or a dir; locate the action_sequencing prompts JSON."""
    if osp.isfile(prompt_path):
        return prompt_path
    cands = []
    for root, _dirs, files in os.walk(prompt_path):
        for f in files:
            if f.endswith(".json") and "action_sequencing" in osp.join(root, f).lower():
                cands.append(osp.join(root, f))
    if not cands:
        # fall back to any json under the dir
        for root, _dirs, files in os.walk(prompt_path):
            for f in files:
                if f.endswith(".json"):
                    cands.append(osp.join(root, f))
    if not cands:
        raise FileNotFoundError(f"No prompt JSON found under {prompt_path}")
    # prefer the one literally containing 'prompt'
    cands.sort(key=lambda p: (("prompt" not in p.lower()), len(p)))
    return cands[0]


def load_prompts(prompt_file: str):
    data = json.load(open(prompt_file, "r"))
    # EAI generate_prompts emits a list of {"identifier","llm_prompt"}.
    if isinstance(data, dict):
        # some versions key by identifier -> prompt
        data = [{"identifier": k, "llm_prompt": v} for k, v in data.items()]
    items = []
    for d in data:
        ident = d.get("identifier") or d.get("file_id") or d.get("id")
        prompt = d.get("llm_prompt") or d.get("prompt") or d.get("text") or ""
        if ident is None:
            continue
        items.append((str(ident), prompt))
    return items


def main():
    ap = argparse.ArgumentParser(description="Export SAGE/SAGE responses for EAI VirtualHome action_sequencing.")
    ap.add_argument("--prompts", required=True,
                    help="EAI prompt JSON file OR a dir produced by generate_prompts.")
    ap.add_argument("--out-dir", required=True,
                    help="Root output dir. Files go to <out>/virtualhome/action_sequencing/<model>_outputs.json")
    ap.add_argument("--model-name", default="sage",
                    help="Used in the {model}_outputs.json filename and to tag the run.")
    ap.add_argument("--limit", type=int, default=0, help="Process only N tasks from --start (0 = all).")
    ap.add_argument("--start", type=int, default=0, help="Index to start from (for sharding).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Ollama sampling seed forwarded to the planner backend (multi-seed study).")
    ap.add_argument("--host", default=None, help="Override LLM host (default: pyplanner DEFAULT_HOST).")
    ap.add_argument("--llm-model", default=None, help="Override LLM model (default: pyplanner DEFAULT_MODEL).")
    ap.add_argument("--provider", default=None, help="LLM provider: ollama|openai|gemini.")
    ap.add_argument("--no-verifier", action="store_true",
                    help="Disable SAGE's symbolic verifier (verifier-OFF ablation).")
    ap.add_argument("--gt-path", default="", help="Path to eval_dataset_gt.json for memory seeding.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do not call the LLM; emit a trivial valid plan (parse pipeline smoke test).")
    ap.add_argument("--validate-only", default="",
                    help="Path to an existing {model}_outputs.json to validate (no generation).")
    args = ap.parse_args()

    # Validate-only mode: just run the grammar check over an existing file.
    if args.validate_only:
        data = json.load(open(args.validate_only, "r"))
        n_ok = n_bad = 0
        for item in data:
            res = eai_validate(item["llm_output"])
            if res is None:
                res = local_validate(item["llm_output"])
            ok, reason = res
            if ok:
                n_ok += 1
            else:
                n_bad += 1
                print(f"  INVALID {item.get('identifier')}: {reason}")
        print(f"[validate] ok={n_ok} bad={n_bad} (validator={'EAI' if eai_validate('{}') is not None else 'local'})")
        return

    # Import pyplanner lazily so --validate-only works without it.
    try:
        from pyplanner.sage import SAGEPlanner
        from pyplanner.verifier import SymbolicState, simulate, normalize_plan
        from pyplanner.base import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_BACKEND
    except Exception as e:  # noqa: BLE001
        print(f"[fatal] cannot import pyplanner: {e}\n"
              f"        pip install -e /path/to/pyplanner", file=sys.stderr)
        sys.exit(2)

    host = args.host or DEFAULT_HOST
    llm_model = args.llm_model or DEFAULT_MODEL
    provider = args.provider or DEFAULT_BACKEND

    prompt_file = find_prompt_file(args.prompts)
    print(f"[info] prompt file: {prompt_file}")
    items = load_prompts(prompt_file)
    if args.start:
        items = items[args.start:]
    if args.limit:
        items = items[: args.limit]
    print(f"[info] tasks to process: {len(items)}")

    planner = None
    if not args.dry_run:
        planner = SAGEPlanner(
            host=host, model=llm_model, provider=provider,
            enable_verifier=not args.no_verifier,
            gt_path=args.gt_path,
        )
        # Multi-seed robustness: forward the sampling seed to the backend (same
        # mechanism as run_benchmark.py). seed=None keeps ollama's default.
        if args.seed is not None and getattr(planner, "_backend", None) is not None:
            planner._backend.seed = args.seed
        print(f"[info] SAGE host={host} model={llm_model} provider={provider} "
              f"verifier={'ON' if not args.no_verifier else 'OFF'} seed={args.seed}")

    out_records = []
    n_valid = n_invalid = n_err = 0
    per_task_dropped = 0

    for idx, (ident, prompt) in enumerate(items):
        try:
            scene = parse_scene_objects(prompt)
            grounder = Grounder(scene)
            task = build_task_from_prompt(prompt)
            visible = [name for name, _id in scene]
            # Give SAGE the actual scene inventory so it can ground its steps.
            obs = "Scene objects: " + ", ".join(name for name, _id in scene)

            if args.dry_run:
                # Emit a minimal valid plan grounded to the first scene object.
                if scene:
                    name, oid = scene[0]
                    entries = [("WALK", [name, oid]), ("FIND", [name, oid])]
                else:
                    entries = []
                dropped = []
            else:
                steps, _metrics = planner.generate_plan(task, obs, visible)
                entries, dropped = sage_steps_to_vh_entries(
                    steps, grounder, simulate, SymbolicState
                )

            per_task_dropped += len(dropped)
            llm_output = entries_to_llm_output(entries)

            # In-line validation so we never write a file EAI will choke on.
            res = eai_validate(llm_output)
            if res is None:
                res = local_validate(llm_output)
            ok, reason = res
            if ok:
                n_valid += 1
            else:
                n_invalid += 1
                print(f"  [warn] {ident}: invalid llm_output ({reason}); n_entries={len(entries)}")

            out_records.append({"identifier": ident, "llm_output": llm_output})
            print(f"[{idx+1}/{len(items)}] {ident}: entries={len(entries)} "
                  f"dropped={len(dropped)} valid={ok}")
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"  [error] {ident}: {e}")
            traceback.print_exc()
            # Still write an empty (will be flagged invalid by EAI) so the
            # identifier set stays aligned with the prompts.
            out_records.append({"identifier": ident, "llm_output": "{}"})

    out_subdir = osp.join(args.out_dir, "virtualhome", "action_sequencing")
    os.makedirs(out_subdir, exist_ok=True)
    out_file = osp.join(out_subdir, f"{args.model_name}_outputs.json")
    with open(out_file, "w") as f:
        json.dump(out_records, f, indent=2)

    print("\n── summary ─────────────────────────────────────────────")
    print(f"  wrote          : {out_file}")
    print(f"  tasks          : {len(out_records)}")
    print(f"  grammar-valid  : {n_valid}")
    print(f"  grammar-invalid: {n_invalid}")
    print(f"  hard errors    : {n_err}")
    print(f"  steps dropped  : {per_task_dropped} (ungrounded / no-VH-mapping)")
    print(f"\n  next:\n    eai-eval --dataset virtualhome --eval-type action_sequencing \\\n"
          f"             --mode evaluate_results --llm-response-path {args.out_dir} \\\n"
          f"             --output-dir results/eai_sage")


if __name__ == "__main__":
    main()
