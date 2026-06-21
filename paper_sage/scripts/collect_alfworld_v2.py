#!/usr/bin/env python3
"""collect_alfworld_v2.py — ALFWorld auto-verifier transition collector (clean, obs-tracked).

Cross-domain transfer for SAGE's learned verifier (Claim C): re-induce per-action
preconditions in ALFWorld's DIFFERENT (text) action vocabulary, ZERO manual rules.

Success oracle = ALFWorld's `admissible_commands` (the precondition-satisfied actions in
the current state). We walk the env and, per state, record POSITIVE transitions (a
type-valid command that IS admissible → success=1) and NEGATIVE ones (constructed
type-valid command NOT admissible → success=0), tagged with observable pre-state features.

State is tracked from the OBSERVATION TEXT (reliable), not inferred from commands:
  "You arrive at <R>."        -> at = R
  "You pick up the <O> ..."   -> holding = O
  "You put the <O> ..."       -> holding = None
  "You open the <R>."         -> open.add(R)   ; "You close the <R>." -> open.discard(R)

Features: at_receptacle, receptacle_open, holding, in_inventory(==holding),
is_openable, is_receptacle.  NO LLM. CPU only. Parallel via disjoint --game-start.
"""
from __future__ import annotations
import argparse, csv, os, re, sys

OPENABLE = {"fridge", "microwave", "cabinet", "drawer", "safe", "box"}
# always-puttable surfaces (no open needed)
SURFACE  = {"countertop", "shelf", "sinkbasin", "garbagecan", "diningtable",
            "sidetable", "coffeetable", "dresser", "bed", "sofa", "armchair",
            "toilet", "stoveburner", "bathtubbasin", "desk", "ottoman", "cart"}
FIELDS = ["scene", "game", "action", "object", "at_receptacle", "receptacle_open",
          "holding", "in_inventory", "is_openable", "is_receptacle", "success", "msg"]


def _stem(name): return re.sub(r"\s*\d+$", "", name.strip()).strip().lower()
def _unwrap(x):  return (x[0] if x else x) if isinstance(x, (list, tuple)) else x
def _admissible(infos):
    a = infos.get("admissible_commands") if isinstance(infos, dict) else None
    a = _unwrap(a) if a is not None else None
    return list(a) if a else []


def make_env(config_path):
    import yaml, importlib
    cfg = yaml.safe_load(open(config_path)); cfg.setdefault("env", {})
    Env = getattr(importlib.import_module("alfworld.agents.environment.alfred_tw_env"), "AlfredTWEnv")
    return Env(cfg, train_eval="eval_out_of_distribution").init_env(batch_size=1)


def parse_recepts(obs):
    seen, out = set(), []
    for f in re.findall(r"\b([a-z]+ \d+)\b", obs.lower()):
        if f not in seen:
            seen.add(f); out.append(f)
    return out


def collect_game(env, gi, writer, flush, counter):
    obs, infos = env.reset()
    obs = str(_unwrap(obs) or "")
    adm = _admissible(infos)
    recepts   = parse_recepts(obs)
    openables = [r for r in recepts if _stem(r) in OPENABLE]
    surfaces  = [r for r in recepts if _stem(r) in SURFACE]
    state = {"at": None, "open": set(), "holding": None}

    def track(o):
        m = re.search(r"you arrive at ([a-z]+ \d+)", o.lower())
        if m: state["at"] = m.group(1)
        m = re.search(r"you pick up the ([a-z]+ \d+)", o.lower())
        if m: state["holding"] = m.group(1)
        if re.search(r"you (put|move) the ", o.lower()): state["holding"] = None
        m = re.search(r"you open the ([a-z]+ \d+)", o.lower())
        if m: state["open"].add(m.group(1))
        m = re.search(r"you close the ([a-z]+ \d+)", o.lower())
        if m: state["open"].discard(m.group(1))

    def step(cmd):
        nonlocal obs, adm
        try:
            o, s, d, infos = env.step([cmd]); o = str(_unwrap(o) or "")
            obs = o; adm = _admissible(infos); track(o); return True
        except Exception:
            return False

    def goto(r):
        # go-to is deterministic in ALFWorld; set location explicitly so the
        # at_receptacle feature is exact even when the env emits no 'arrive' line.
        if f"go to {r}" in adm:
            step(f"go to {r}")
        state["at"] = r

    def feats(target, is_recep):
        st = _stem(target)
        return dict(at_receptacle=int(state["at"] == target),
                    receptacle_open=int(target in state["open"]),
                    holding=int(state["holding"] is not None),
                    in_inventory=int(state["holding"] is not None),
                    is_openable=int(st in OPENABLE),
                    is_receptacle=int(is_recep))

    def rec(action, target, success, msg, is_recep=True):
        row = dict(scene=f"alf_{gi}", game=f"alf_{gi}", action=action, object=target,
                   success=int(bool(success)), msg=str(msg)[:40], **feats(target, is_recep))
        writer.writerow(row); flush(); counter["n"] += 1

    # ── OPEN/CLOSE on openables, with at/¬at and open-state ± ──────────────
    for r in openables[:6]:
        rec("open", r, f"open {r}" in adm, "pre-goto")           # neg: ¬at
        goto(r)
        rec("open", r, f"open {r}" in adm, "at")                  # pos: at ∧ openable ∧ ¬open
        if f"open {r}" in adm:
            step(f"open {r}")
            rec("close", r, f"close {r}" in adm, "open->close")   # pos close: at ∧ open
            rec("open", r, f"open {r}" in adm, "already-open")     # neg open: already open
    # close NEGATIVE on a not-yet-opened openable (receptacle_open=0)
    for r in openables[:6]:
        if r not in state["open"]:
            goto(r)
            if r not in state["open"]:
                rec("close", r, f"close {r}" in adm, "close-notopen")  # neg close: ¬open

    # ── OPEN on NON-openable surfaces (neg for is_openable) ────────────────
    for r in surfaces[:5]:
        goto(r)
        rec("open", r, f"open {r}" in adm, "surface-nonopenable")  # neg: ¬openable

    # ── TAKE / PUT (holding ± , put on a SURFACE so open-state isn't needed) ──
    # gather several takes across receptacles for coverage
    n_take = 0
    for r in (surfaces[:4] + openables[:4]):
        if n_take >= 3: break
        goto(r)
        if _stem(r) in OPENABLE and f"open {r}" in adm: step(f"open {r}")
        takes = [c for c in adm if c.startswith("take ")]
        if not takes: continue
        tc = takes[0]; mm = re.match(r"take (.+) from (.+)", tc); ob = mm.group(1) if mm else tc[5:]
        # put features describe the TARGET receptacle, not the held object.
        # put BEFORE take = ¬holding negative (at receptacle r, hand empty)
        rec("put", r, any(c.startswith("move ") for c in adm), "put-not-holding", is_recep=True)
        rec("take", ob, True, "take-pos", is_recep=False)         # pos: at ∧ ¬holding
        step(tc)                                                   # now holding (obs-tracked)
        rec("take", ob, tc in adm, "take-while-holding", is_recep=False)  # neg: holding
        # PUT positive: go to a surface (always puttable) and put (target = surface)
        placed = False
        for s in surfaces[:5]:
            goto(s)
            puts = [c for c in adm if c.startswith("move ")]
            rec("put", s, bool(puts), "put-holding-surface", is_recep=True)  # pos: holding ∧ at(surface)
            if puts:
                step(puts[0]); placed = True; break
        n_take += 1
    print(f"[game_{gi}] total={counter['n']} at={state['at']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/alfworld_base_config.yaml")
    ap.add_argument("--num-games", type=int, default=40)
    ap.add_argument("--game-start", type=int, default=0)
    ap.add_argument("--out", default="results/autoverify/transitions_alfworld.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    env = make_env(args.config)
    f = open(args.out, "w", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
    counter = {"n": 0}
    for _ in range(args.game_start):
        try: env.reset()
        except Exception: pass
    for gi in range(args.game_start, args.game_start + args.num_games):
        try: collect_game(env, gi, w, f.flush, counter)
        except Exception as e: print(f"[skip] game {gi}: {type(e).__name__} {e}", flush=True)
    f.close()
    print(f"Wrote {counter['n']} transitions over {args.num_games} games -> {args.out}")


if __name__ == "__main__":
    main()
