#!/usr/bin/env python3
"""collect_transitions_alfworld.py  (Phase 2: auto/learned verifier — CROSS-DOMAIN)
==================================================================================
Probe ALFWorld (TextWorld backend) systematically to collect
``(pre-state features, action, success)`` transitions in a DIFFERENT action
vocabulary than AI2-THOR, so SAGE's precondition verifier can be RE-INDUCED from
interaction data with **zero hand-written rules** in a domain it was never
authored for. This is the load-bearing evidence for Claim C of
``ICRA/plans/auto-verifier-transfer.md`` (the verifier is LEARNED, not authored,
and ports across embodiments/vocabularies).

It mirrors the contrastive-probe discipline of the iTHOR collector
(``collect_transitions.py``): for every action and every candidate precondition
feature we generate matched +/- pre-states (drive the agent there with
``goto`` / ``take`` / ``open`` setup commands) then attempt the probe action and
record whether the environment accepts it. **No LLM is used.** The success oracle
is ALFWorld's own ``admissible_commands`` set + reward/win signal + observation
("Nothing happens" style rejections), exactly the kind of deterministic
zero-cost grounding signal the verifier needs.

The output CSV has the SAME shape philosophy as the THOR collector
(``game, action, object, <feature flags>, success, msg``) so the UNCHANGED
``induce_verifier.py`` can consume it via ``--features`` + ``--ref-json``.

ALFWorld action vocabulary probed (TextWorld surface forms):
    goto <recep> | take <obj> from <recep> | put <obj> in/on <recep>
    open <recep> | close <recep> | toggle <obj>
    clean <obj> with <recep> | heat <obj> with <recep> | cool <obj> with <recep>

Candidate FEATURE SET (booleans parsed from the OBSERVABLE text state only — the
inventory string, the current observation, and the admissible-command set — never
from knowing the answer):
    in_inventory      the probed object is currently in the agent's inventory
    at_receptacle     the agent is currently AT the target receptacle/location
    receptacle_open   the target receptacle is currently open
    holding_any       the agent is holding at least one object
    obj_visible       the probed object is visible in the current location text
    is_openable       static affordance inferred from receptacle name table
    is_toggleable     static affordance inferred from object name table
    is_receptacle     the target is a receptacle (vs a movable object)

Run (single process):
  ALFWORLD_DATA=~/.cache/alfworld \
  python scripts/collect_transitions_alfworld.py \
      --config ~/.cache/alfworld/configs/base_config.yaml \
      --num-games 20 --game-start 0 \
      --out results/autoverify/transitions_alfworld_0.csv

Run (CPU parallelism: several processes over DISJOINT game ranges, then merge):
  see results/analysis/alfworld_transfer_notes.md
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import traceback

# ---------------------------------------------------------------------------
# Defensive, import-guarded environment bring-up. The MAIN session debug-runs
# this; every alfworld/textworld import is isolated so a single missing piece
# yields a clear message instead of an opaque traceback.
# ---------------------------------------------------------------------------


def _die(msg: str, exc: "Exception | None" = None) -> "NoReturn":  # type: ignore[name-defined]
    print("=" * 70, file=sys.stderr)
    print("[collect_transitions_alfworld] FATAL:", msg, file=sys.stderr)
    if exc is not None:
        print("  underlying error:", repr(exc), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.exit(2)


def _load_config(config_path: "str | None"):
    """Locate + parse base_config.yaml. Honour --config, then $ALFWORLD_CONFIG,
    then a couple of conventional locations under $ALFWORLD_DATA / the installed
    package. Returns the parsed config dict (yaml). Fails LOUDLY but cleanly."""
    try:
        import yaml  # noqa: F401
    except Exception as e:  # pragma: no cover - defensive
        _die("PyYAML is required to read base_config.yaml (pip install pyyaml).", e)
    import yaml

    candidates = []
    if config_path:
        candidates.append(config_path)
    if os.environ.get("ALFWORLD_CONFIG"):
        candidates.append(os.environ["ALFWORLD_CONFIG"])
    data_root = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    candidates += [
        os.path.join(data_root, "configs", "base_config.yaml"),
        os.path.join(data_root, "base_config.yaml"),
    ]
    # config shipped inside the installed package
    try:
        import alfworld  # noqa: F401
        pkg_dir = os.path.dirname(alfworld.__file__)
        candidates += [
            os.path.join(pkg_dir, "configs", "base_config.yaml"),
            os.path.join(os.path.dirname(pkg_dir), "configs", "base_config.yaml"),
        ]
    except Exception:
        pass

    for c in candidates:
        if c and os.path.isfile(c):
            try:
                with open(c, "r") as fh:
                    cfg = yaml.safe_load(fh)
                print(f"[config] using {c}", flush=True)
                return cfg, c
            except Exception as e:  # pragma: no cover - defensive
                _die(f"Found config at {c} but failed to parse it.", e)
    _die(
        "Could not locate base_config.yaml. Pass --config <path>, or set "
        "$ALFWORLD_CONFIG, or place it under $ALFWORLD_DATA/configs/. "
        f"Tried: {[c for c in candidates if c]}"
    )


def _make_env(cfg, config_path):
    """Construct the lightweight TextWorld ALFWorld env. We getattr the env class
    so we tolerate the class living at slightly different import paths across
    alfworld 0.4.x. The TextWorld path needs NO simulator / GPU render."""
    env_mod = None
    try:
        import alfworld.agents.environment as env_mod  # type: ignore
    except Exception as e:  # pragma: no cover - defensive
        _die(
            "Cannot import alfworld.agents.environment. Is alfworld installed "
            "and is $ALFWORLD_DATA set with downloaded game data?",
            e,
        )

    # Preferred class for the no-sim text path, with fallbacks. alfworld 0.4.x
    # does NOT re-export the env classes on the package; they live in submodules
    # (e.g. alfworld.agents.environment.alfred_tw_env.AlfredTWEnv). Try the
    # package attr first, then the per-class submodule.
    import importlib
    _submods = {
        "AlfredTWEnv":   "alfworld.agents.environment.alfred_tw_env",
        "AlfredThorEnv": "alfworld.agents.environment.alfred_thor_env",
        "AlfredHybrid":  "alfworld.agents.environment.alfred_hybrid",
    }
    env_cls = None
    for name in ("AlfredTWEnv", "AlfredThorEnv", "AlfredHybrid"):
        env_cls = getattr(env_mod, name, None)
        if env_cls is None:
            try:
                env_cls = getattr(importlib.import_module(_submods[name]), name, None)
            except Exception:
                env_cls = None
        if env_cls is not None:
            print(f"[env] using {name}", flush=True)
            break
    if env_cls is None:
        _die(
            "No ALFWorld env class found (looked for AlfredTWEnv / AlfredThorEnv / "
            "AlfredHybrid on alfworld.agents.environment and its submodules)."
        )

    # alfworld's constructor signature has drifted across versions; try the
    # documented (config, train_eval) form first, then config-only.
    last_exc = None
    for kwargs in (
        dict(config=cfg, train_eval="eval_out_of_distribution"),
        dict(config=cfg, train_eval="train"),
    ):
        try:
            tw = env_cls(**kwargs)
            break
        except TypeError:
            tw = None
        except Exception as e:  # data/path errors surface here
            last_exc = e
            tw = None
    else:
        tw = None
    if tw is None:
        try:
            tw = env_cls(cfg)  # positional, config-only
        except Exception as e:
            _die(
                "Failed to construct the ALFWorld env. The likeliest causes are "
                "(a) game data not finished downloading to $ALFWORLD_DATA, or "
                "(b) a constructor-signature mismatch for this alfworld version. "
                f"Config used: {config_path}",
                e or last_exc,
            )

    # init_env() returns a (batched, batch_size=1) gym-like env in TW mode.
    try:
        env = tw.init_env(batch_size=1)
    except Exception as e:
        _die("env.init_env(batch_size=1) failed.", e)
    return env


# ---------------------------------------------------------------------------
# Static affordance tables — inferred from the OBJECT/RECEPTACLE NAME ONLY (the
# same "static affordance flag" discipline the THOR collector uses via
# obj_map[...]['openable'/'toggleable'/...]). These are NOT preconditions and are
# NOT the answer; they are observable-from-the-name affordance hints the miner is
# free to use or discard. ALFWorld receptacle/object types are well known.
# ---------------------------------------------------------------------------
OPENABLE_RECEPS = {
    "fridge", "microwave", "cabinet", "drawer", "safe", "box",
    "garbagecan", "kettle",  # kettle/garbagecan have lids in some layouts
}
TOGGLEABLE_OBJS = {
    "desklamp", "floorlamp", "lamp", "television", "tv", "faucet",
    "microwave", "stoveburner", "toaster", "blender", "candle",
}
# Receptacle-ish surfaces/containers (where put/take happen).
RECEPTACLE_TYPES = {
    "fridge", "microwave", "cabinet", "drawer", "safe", "box", "sink",
    "sinkbasin", "countertop", "shelf", "shelfunit", "diningtable",
    "coffeetable", "sidetable", "desk", "bed", "sofa", "armchair",
    "dresser", "ottoman", "toilet", "toiletpaperhanger", "towelholder",
    "handtowelholder", "bathtub", "bathtubbasin", "stoveburner", "garbagecan",
    "cart", "tvstand", "coffeemachine", "dishwasher", "laundryhamper",
}

# ALFWorld names objects/receptacles as "<type> <id>", e.g. "cabinet 4",
# "apple 1", "diningtable 1". We bucket by the alphabetic stem.


def _type_of(name: str) -> str:
    """'cabinet 4' -> 'cabinet'; 'desklamp 1' -> 'desklamp'."""
    if not name:
        return ""
    head = name.strip().split()[0].lower()
    return "".join(ch for ch in head if ch.isalpha())


def is_openable(name: str) -> int:
    return int(_type_of(name) in OPENABLE_RECEPS)


def is_toggleable(name: str) -> int:
    return int(_type_of(name) in TOGGLEABLE_OBJS)


def is_receptacle(name: str) -> int:
    t = _type_of(name)
    return int(t in RECEPTACLE_TYPES or t in OPENABLE_RECEPS)


# ---------------------------------------------------------------------------
# Observable-text state parsing. Everything below reads ONLY the observation
# string, the inventory string, and the admissible-command list — i.e. exactly
# what an agent can observe — never privileged game state.
# ---------------------------------------------------------------------------


def _unwrap(x):
    """init_env(batch_size=1) returns batched lists/tuples; unwrap element 0."""
    if isinstance(x, (list, tuple)):
        return x[0] if x else x
    return x


def _admissible(infos) -> "list[str]":
    """Pull the admissible-commands list out of the (batched) infos dict, across
    the key spellings alfworld has used."""
    if not isinstance(infos, dict):
        return []
    for key in ("admissible_commands", "admissible_actions"):
        if key in infos:
            val = infos[key]
            val = _unwrap(val)
            if isinstance(val, (list, tuple)):
                return [str(c) for c in val]
    return []


def _won(infos) -> bool:
    if not isinstance(infos, dict):
        return False
    w = infos.get("won")
    w = _unwrap(w)
    return bool(w)


class Probe:
    """Thin stateful wrapper around the batched TW env that tracks the observable
    text state needed for feature extraction, and exposes one method per probe
    primitive. Resilient: any env error sets ``self.broken`` and the caller skips
    to the next game."""

    def __init__(self, env):
        self.env = env
        self.broken = False
        self.obs = ""
        self.infos = {}
        self.admissible = []

    # ---- low-level ----
    def reset(self) -> bool:
        try:
            obs, infos = self.env.reset()
            self.obs = str(_unwrap(obs) or "")
            self.infos = infos if isinstance(infos, dict) else {}
            self.admissible = _admissible(self.infos)
            self.broken = False
            return True
        except Exception as e:
            print(f"[reset] EXC {type(e).__name__}: {e}", flush=True)
            self.broken = True
            return False

    def step(self, cmd: str):
        """Run one text command. Returns (success:int, msg:str). SUCCESS oracle:
        a command is a success transition iff it was admissible AND the env
        accepted it (observation is not a 'nothing happens' rejection). An
        inadmissible command is recorded as a failure WITHOUT stepping the env
        (so we never corrupt the trajectory with rejected actions)."""
        if self.broken:
            return None, "broken"
        admissible_now = cmd in self.admissible
        if not admissible_now:
            # Inadmissible = a clean precondition-violation negative. Do not step.
            return 0, "inadmissible"
        try:
            obs, scores, dones, infos = self.env.step([cmd])
            obs = str(_unwrap(obs) or "")
            self.infos = infos if isinstance(infos, dict) else {}
            self.admissible = _admissible(self.infos)
            self.obs = obs
        except Exception as e:
            print(f"[step] EXC {type(e).__name__}: {e}", flush=True)
            self.broken = True
            return None, f"EXC {type(e).__name__}"
        low = obs.lower()
        rejected = (
            "nothing happens" in low
            or "you can't" in low
            or "can't see" in low
            or "not able" in low
            or obs.strip() == ""
        )
        ok = 0 if rejected else 1
        return ok, obs.strip()[:50].replace("\n", " ")

    def admissible_contains(self, prefix: str) -> "list[str]":
        return [c for c in self.admissible if c.startswith(prefix)]

    # ---- observable-state feature helpers ----
    def inventory(self) -> str:
        """Best-effort inventory string. ALFWorld exposes inventory via the
        'inventory' command being admissible; we read the obs it returns. We do
        NOT permanently advance state with it (inventory is a no-op look)."""
        # Track holding via what we've taken; but also confirm from text.
        return self._inv_text

    _inv_text = ""

    def refresh_inventory(self):
        if self.broken:
            return
        # 'inventory' is a pure observation (no state change).
        if "inventory" in [c.strip() for c in self.admissible]:
            try:
                obs, _, _, infos = self.env.step(["inventory"])
                self._inv_text = str(_unwrap(obs) or "")
                self.infos = infos if isinstance(infos, dict) else self.infos
                self.admissible = _admissible(self.infos) or self.admissible
            except Exception:
                pass


# ---- feature extraction from observable text + admissible set ----------------


def in_inventory(p: Probe, obj: str) -> int:
    p.refresh_inventory()
    inv = p.inventory().lower()
    if obj and obj.lower() in inv:
        return 1
    # Fallback: if "put <obj> ..." is admissible, the obj is in hand.
    if obj and any(c.startswith(f"put {obj.lower()}") for c in (s.lower() for s in p.admissible)):
        return 1
    return 0


def holding_any(p: Probe) -> int:
    p.refresh_inventory()
    inv = p.inventory().lower()
    if "you are carrying" in inv and "nothing" not in inv:
        return 1
    if "nothing" in inv:
        return 0
    # Fallback on admissible "put" commands existing.
    return int(any(c.lower().startswith("put ") for c in p.admissible))


def at_receptacle(p: Probe, recep: str) -> int:
    """The agent is AT a receptacle iff the current observation describes being
    at/facing it OR receptacle-local commands (open/close/put-on-this) reference
    it in the admissible set."""
    if not recep:
        return 0
    r = recep.lower()
    low = p.obs.lower()
    if (f"on the {r}" in low or f"at the {r}" in low or f"the {r} is" in low
            or f"arrive at {r}" in low or f"you are at the {r}" in low):
        return 1
    # If we can open/close/put-into THIS receptacle right now, we're at it.
    for c in (s.lower() for s in p.admissible):
        if (c == f"open {r}" or c == f"close {r}"
                or c.startswith(f"put ") and c.endswith(f" {r}")
                or (c.startswith("take ") and c.endswith(f"from {r}"))):
            return 1
    return 0


def receptacle_open(p: Probe, recep: str) -> int:
    if not recep:
        return 0
    r = recep.lower()
    low = p.obs.lower()
    if f"the {r} is open" in low or f"{r} is open" in low:
        return 1
    if f"the {r} is closed" in low or f"{r} is closed" in low:
        return 0
    # If "close <recep>" is admissible it must currently be open; if "open
    # <recep>" is admissible it is currently closed.
    adm = [c.lower() for c in p.admissible]
    if f"close {r}" in adm:
        return 1
    if f"open {r}" in adm:
        return 0
    # Non-openable receptacle: treat as vacuously "open" (accessible).
    return 1 if not is_openable(recep) else 0


def obj_visible(p: Probe, obj: str) -> int:
    if not obj:
        return 0
    o = obj.lower()
    if o in p.obs.lower():
        return 1
    return int(any(o in c.lower() for c in p.admissible))


FIELDS = [
    "game", "action", "object",
    # dynamic, observable-from-text features
    "in_inventory", "at_receptacle", "receptacle_open", "holding_any", "obj_visible",
    # static affordance flags inferable from the name
    "is_openable", "is_toggleable", "is_receptacle",
    "success", "msg",
]


def _features(p: Probe, obj: str, target: str):
    """Compute the full candidate feature vector for a probe of <obj> w.r.t. an
    optional <target> receptacle. ``obj`` is the thing acted on; ``target`` is
    the receptacle for take/put/clean/heat/cool, else ''."""
    recep = target or (obj if is_receptacle(obj) else "")
    aff_name = recep if recep else obj
    return dict(
        in_inventory=in_inventory(p, obj),
        at_receptacle=at_receptacle(p, recep),
        receptacle_open=receptacle_open(p, recep),
        holding_any=holding_any(p),
        obj_visible=obj_visible(p, obj),
        is_openable=is_openable(recep) if recep else is_openable(obj),
        is_toggleable=is_toggleable(obj),
        is_receptacle=is_receptacle(aff_name),
    )


# ---------------------------------------------------------------------------
# Enumerate probe targets from the admissible command set of a freshly-reset
# game. We parse the goto/open/take/put commands to discover the receptacles and
# objects present, then drive contrastive probes against them.
# ---------------------------------------------------------------------------


def _discover(p: Probe):
    receps, objs = set(), set()
    for c in p.admissible:
        toks = c.split()
        if not toks:
            continue
        verb = toks[0]
        if verb == "goto" and len(toks) >= 2:
            receps.add(" ".join(toks[1:]))
        elif verb == "open" and len(toks) >= 2:
            receps.add(" ".join(toks[1:]))
        elif verb == "take" and "from" in toks:
            i = toks.index("from")
            objs.add(" ".join(toks[1:i]))
            receps.add(" ".join(toks[i + 1:]))
        elif verb == "put" and ("in" in toks or "on" in toks):
            kw = "in" if "in" in toks else "on"
            i = toks.index(kw)
            objs.add(" ".join(toks[1:i]))
            receps.add(" ".join(toks[i + 1:]))
    return sorted(receps), sorted(objs)


def _goto(p: Probe, recep: str):
    """Drive to a receptacle if a goto for it is admissible. Returns success."""
    cmd = f"goto {recep}"
    if cmd in p.admissible:
        s, _ = p.step(cmd)
        return s == 1
    return False


def _take_first_available(p: Probe):
    """Take any currently-takeable object (for holding_any probes). Returns the
    object name taken or None."""
    for c in p.admissible:
        if c.startswith("take ") and " from " in c:
            s, _ = p.step(c)
            if s == 1:
                toks = c.split()
                i = toks.index("from")
                return " ".join(toks[1:i])
    return None


def collect_one_game(game_idx: int, p: Probe, writer, flush, counter, objs_per_aff: int):
    """Run the full contrastive probe battery on ONE freshly reset game. Mirrors
    collect_one() in the iTHOR collector: for each action and each candidate
    feature, produce matched +/- pre-states by driving the agent with goto/take/
    open setup commands, then attempt the probe and record admissibility/success.
    """
    n = counter["n"]

    def rec(action, obj, target, feats, success, msg):
        nonlocal n
        if success is None:  # env hung/broke → skip this probe
            return
        row = dict(game=f"game_{game_idx}", action=action, object=obj,
                   success=success, msg=msg, **feats)
        writer.writerow(row)
        flush()
        n += 1

    receps, objs = _discover(p)
    openable_receps = [r for r in receps if is_openable(r)][:objs_per_aff]
    closed_candidates = openable_receps
    toggle_objs = [o for o in objs if is_toggleable(o)][:objs_per_aff]
    take_pairs = []  # (obj, recep) discovered from take commands
    for c in p.admissible:
        if c.startswith("take ") and " from " in c:
            toks = c.split()
            i = toks.index("from")
            take_pairs.append((" ".join(toks[1:i]), " ".join(toks[i + 1:])))
    take_pairs = take_pairs[: objs_per_aff * 2]

    # =====================================================================
    # GOTO — admissible from anywhere; positive baseline (always-accepted nav).
    # Records obj_visible +/- naturally (target far vs near).
    # =====================================================================
    for r in receps[:objs_per_aff]:
        cmd = f"goto {r}"
        feats = _features(p, r, "")
        if cmd in p.admissible:
            s, m = p.step(cmd)
            rec("goto", r, "", feats, s, m)

    # =====================================================================
    # OPEN / CLOSE — contrastive on at_receptacle and receptacle_open.
    #   open: at recep + closed -> success ;  not-at recep -> inadmissible (fail)
    #   open: already-open recep -> inadmissible (fail)  [receptacle_open=1]
    #   close: at recep + open -> success ; closed recep -> fail
    # =====================================================================
    for r in openable_receps:
        if not p.reset():
            return
        # not at recep yet → open should be inadmissible → failure (at=0)
        feats = _features(p, r, "")
        s, m = p.step(f"open {r}")  # likely inadmissible from start
        rec("open", r, "", feats, s, m)

        if not _goto(p, r):
            continue
        # at recep, currently closed → open should succeed
        feats = _features(p, r, "")
        s, m = p.step(f"open {r}")
        rec("open", r, "", feats, s, m)

        # now it is open: a second open is inadmissible → failure (recep_open=1)
        feats = _features(p, r, "")
        s, m = p.step(f"open {r}")
        rec("open", r, "", feats, s, m)

        # close: at recep + open → success
        feats = _features(p, r, "")
        s, m = p.step(f"close {r}")
        rec("close", r, "", feats, s, m)

        # close again: now closed → inadmissible → failure (recep_open=0)
        feats = _features(p, r, "")
        s, m = p.step(f"close {r}")
        rec("close", r, "", feats, s, m)

    # =====================================================================
    # TAKE — contrastive on at_receptacle and in_inventory(/holding_any).
    #   take obj from r: not at r            -> inadmissible (fail, at=0)
    #   take obj from r: at r, not holding   -> success      (at=1, in_inv=0)
    #   take obj from r: already in inventory-> inadmissible (fail, in_inv=1)
    # =====================================================================
    for (obj, r) in take_pairs[:objs_per_aff]:
        if not p.reset():
            return
        # away from r → take inadmissible
        feats = _features(p, obj, r)
        s, m = p.step(f"take {obj} from {r}")
        rec("take", obj, r, feats, s, m)

        if not _goto(p, r):
            continue
        # at r, open if needed
        if is_openable(r) and f"open {r}" in p.admissible:
            p.step(f"open {r}")
        # at r, not yet holding obj → take should succeed
        feats = _features(p, obj, r)
        s, m = p.step(f"take {obj} from {r}")
        took_ok = s == 1
        rec("take", obj, r, feats, s, m)

        if took_ok:
            # obj now in inventory → re-take is inadmissible (in_inventory=1)
            feats = _features(p, obj, r)
            s, m = p.step(f"take {obj} from {r}")
            rec("take", obj, r, feats, s, m)

    # =====================================================================
    # PUT — contrastive on holding_any / in_inventory and at_receptacle.
    #   put obj in/on r: holding obj + at r -> success (holding=1, at=1)
    #   put obj in/on r: NOT holding        -> inadmissible (fail, holding=0)
    #   put obj in/on r: holding but not at r -> inadmissible (fail, at=0)
    # =====================================================================
    for (obj, src) in take_pairs[:objs_per_aff]:
        if not p.reset():
            return
        # not holding anything → any put is inadmissible
        # pick a plausible destination receptacle
        dests = [r for r in receps if r != src]
        dest = dests[0] if dests else src
        feats = _features(p, obj, dest)
        # find an admissible put for obj if one exists pre-pickup (should not)
        put_cmd = next((c for c in p.admissible
                        if c.startswith(f"put {obj} ") and c.endswith(dest)), None)
        s, m = p.step(put_cmd or f"put {obj} in/on {dest}")
        rec("put", obj, dest, feats, s, m)

        # now acquire obj
        if not _goto(p, src):
            continue
        if is_openable(src) and f"open {src}" in p.admissible:
            p.step(f"open {src}")
        st, _ = p.step(f"take {obj} from {src}")
        if st != 1:
            continue

        # holding obj but NOT at dest → put inadmissible (at=0)
        feats = _features(p, obj, dest)
        put_cmd = next((c for c in p.admissible
                        if c.startswith(f"put {obj} ") and c.endswith(dest)), None)
        s, m = p.step(put_cmd or f"put {obj} in/on {dest}")
        rec("put", obj, dest, feats, s, m)

        # goto dest, open if needed → holding + at dest → put succeeds
        if not _goto(p, dest):
            continue
        if is_openable(dest) and f"open {dest}" in p.admissible:
            p.step(f"open {dest}")
        feats = _features(p, obj, dest)
        put_cmd = next((c for c in p.admissible
                        if c.startswith(f"put {obj} ") and c.endswith(dest)), None)
        s, m = p.step(put_cmd or f"put {obj} in/on {dest}")
        rec("put", obj, dest, feats, s, m)

    # =====================================================================
    # TOGGLE — contrastive on is_toggleable and at_receptacle/obj_visible.
    #   toggle obj: toggleable obj, at it      -> success
    #   toggle obj: non-toggleable obj         -> inadmissible (fail)
    # =====================================================================
    for o in toggle_objs:
        if not p.reset():
            return
        # navigate near the object's receptacle if discoverable
        host = next((r for (oo, r) in take_pairs if oo == o), None)
        if host:
            _goto(p, host)
        feats = _features(p, o, "")
        cmd = next((c for c in p.admissible if c.startswith(f"toggle {o}")), None)
        s, m = p.step(cmd or f"toggle {o}")
        rec("toggle", o, "", feats, s, m)
    # non-toggleable negative: try toggling a plain object
    nontoggle = next((o for o in objs if not is_toggleable(o)), None)
    if nontoggle:
        feats = _features(p, nontoggle, "")
        s, m = p.step(f"toggle {nontoggle}")  # inadmissible → fail
        rec("toggle", nontoggle, "", feats, s, m)

    # =====================================================================
    # CLEAN / HEAT / COOL — contrastive on holding_any + at_receptacle.
    # These appear in admissible only when holding obj AND at the right
    # appliance/receptacle. We probe both the satisfied and unsatisfied states.
    #   clean obj with sink:  holding + at sink     -> success (if admissible)
    #   clean obj with sink:  not holding           -> inadmissible (fail)
    # =====================================================================
    for verb, appliance_kw in (("clean", "sinkbasin"), ("heat", "microwave"), ("cool", "fridge")):
        # negative: with nothing in hand, the verb is inadmissible
        if not p.reset():
            return
        appliances = [r for r in receps if _type_of(r) == appliance_kw]
        any_obj, src = (take_pairs[0] if take_pairs else (None, None))
        appliance = appliances[0] if appliances else None
        if any_obj is None or appliance is None:
            continue
        feats = _features(p, any_obj, appliance)
        s, m = p.step(f"{verb} {any_obj} with {appliance}")  # not holding → fail
        rec(verb, any_obj, appliance, feats, s, m)

        # positive: acquire the object, go to appliance, attempt the verb
        if not _goto(p, src):
            continue
        if is_openable(src) and f"open {src}" in p.admissible:
            p.step(f"open {src}")
        st, _ = p.step(f"take {any_obj} from {src}")
        if st != 1:
            continue
        if not _goto(p, appliance):
            continue
        feats = _features(p, any_obj, appliance)
        cmd = next((c for c in p.admissible
                    if c.startswith(f"{verb} {any_obj} with")), None)
        s, m = p.step(cmd or f"{verb} {any_obj} with {appliance}")
        rec(verb, any_obj, appliance, feats, s, m)

    counter["n"] = n
    print(f"[game_{game_idx}] collected, running total = {n}", flush=True)


def balance_report(out_csv):
    """Per-(action,feature) ±-count contrastive self-check, mirroring the THOR
    collector's --balance. Cited in the paper as 'every mined condition had a
    contrastive pair'."""
    from collections import Counter, defaultdict
    rows = list(csv.DictReader(open(out_csv)))
    feats = ["in_inventory", "at_receptacle", "receptacle_open", "holding_any",
             "obj_visible", "is_openable", "is_toggleable", "is_receptacle"]
    by_act = defaultdict(list)
    for r in rows:
        by_act[r["action"]].append(r)
    print("\n=== --balance contrastive self-check (ALFWorld) ===")
    for act in sorted(by_act):
        rs = by_act[act]
        succ = [r for r in rs if r["success"] == "1"]
        fail = [r for r in rs if r["success"] == "0"]
        print(f"[{act}] n={len(rs)} pos={len(succ)} neg={len(fail)}")
        for f in feats:
            sc = Counter(r[f] for r in succ)
            fc = Counter(r[f] for r in fail)
            ok = any(any(rf[f] != v for rf in fail) for v in {r[f] for r in succ}) if succ else False
            flag = "" if ok or not succ else "  <-- NO CONTRAST"
            print(f"    {f:16s} succ{dict(sc)} fail{dict(fc)}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="path to alfworld base_config.yaml (else $ALFWORLD_CONFIG "
                         "/ $ALFWORLD_DATA/configs/base_config.yaml / packaged copy)")
    ap.add_argument("--num-games", type=int, default=20,
                    help="number of games to probe in this process")
    ap.add_argument("--game-start", type=int, default=0,
                    help="how many games to skip first (for DISJOINT parallel "
                         "ranges across processes -> CPU parallelism)")
    ap.add_argument("--objs-per-aff", type=int, default=4,
                    help="cap on receptacles/objects probed per affordance per game")
    ap.add_argument("--balance", action="store_true",
                    help="after collection, print the per-feature ±-count self-check")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results",
        "autoverify", "transitions_alfworld.csv"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    cfg, config_path = _load_config(args.config)
    env = _make_env(cfg, config_path)
    p = Probe(env)

    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    counter = {"n": 0}

    # Skip game-start games (DISJOINT ranges). ALFWorld's TW env cycles games on
    # reset; we advance the cursor by resetting game-start times first.
    for _ in range(max(0, args.game_start)):
        if not p.reset():
            print("[warn] reset failed while skipping to game-start; aborting", flush=True)
            f.close()
            return

    collected_games = 0
    for g in range(args.game_start, args.game_start + args.num_games):
        if not p.reset():
            print(f"[skip] game {g}: reset failed", flush=True)
            # try to recover by continuing; env may be exhausted
            continue
        try:
            collect_one_game(g, p, w, f.flush, counter, args.objs_per_aff)
            collected_games += 1
        except Exception as e:
            print(f"[err] game {g}: {type(e).__name__}: {e}", flush=True)
            # leave a fresh reset for the next iteration
            p.broken = True

    f.close()
    print(f"Wrote {counter['n']} transitions over {collected_games} games -> {args.out}",
          flush=True)
    if args.balance:
        balance_report(args.out)


if __name__ == "__main__":
    main()
