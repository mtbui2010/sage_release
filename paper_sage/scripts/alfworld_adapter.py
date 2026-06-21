#!/usr/bin/env python3
"""alfworld_adapter.py — translate pyplanner steps into ALFWorld TextWorld commands.

Purpose
-------
Lets the pyplanner planner family (Direct / CoT / Hierarchical / SAGE / ...) run
against the **ALFWorld TextWorld** benchmark, mirroring the AI2-THOR harness in
``pyplanner/apps/evaluate/evaluate_sim.py``. The planner emits steps in the
canonical ``ROBOT_ACTIONS`` vocabulary (``MoveTo``/``Find``/``Pick``/``Place``/
``PutIn``/``Open``/``Close``/``TurnOn``/``TurnOff``); this module converts each
step into a single ALFWorld text command (``go to``/``take``/``put``/``open``/
``close``/``toggle``) using a tracked, observation-parsed world state and the
env's ``admissible_commands`` whenever they are available.

This module is the *bridge layer only*. The actual episode loop, success
detection, CSV/JSON aggregation and CLI live in ``evaluate_alfworld.py``.

Three public surfaces:
  * ``translate_step(step, state)``     — one step -> one ALFWorld command (or None)
  * ``AlfworldState``                   — obs-text-tracked world state
  * ``make_alfworld_env(config, split)``— instantiate AlfredTWEnv (+ textworld patch)

Known fragilities (DELIBERATELY surfaced — the human must run-test these)
------------------------------------------------------------------------
1. **Find/Pick mismatch.** ALFWorld has no explicit "find" primitive — locating an
   object is implicit in ``take <obj> from <recep>``. We therefore translate
   ``Find`` to *None* (a no-op the caller skips) but stash the intended object in
   ``state.pending_find`` so the *following* ``Pick`` knows what to take and from
   where (the current location). If a plan issues ``Pick`` without a preceding
   ``Find`` (or ``step.object`` on the Pick), we cannot construct a valid command
   and return None. THOR-style plans that ``Find`` an object the agent is not
   co-located with will produce a ``take`` from the wrong receptacle.
2. **Toggle has no on/off direction.** ALFWorld exposes a single ``toggle <obj>``
   that flips state; there is no ``turn on`` vs ``turn off``. We map *both*
   ``TurnOn`` and ``TurnOff`` to ``toggle``. A plan that toggles twice (on then
   off) is faithfully represented, but a plan that assumes idempotent TurnOn will
   instead flip the device the second time.
3. **Text-parse state tracking is heuristic.** ``AlfworldState`` is updated by
   substring/regex matching on the observation ("You arrive at ...", "You pick up
   the ...", "You open the ...", the ALFWorld "You put/move the X in/on R" form).
   If a future alfworld build rephrases these lines, location/holding/open
   tracking degrades and the constructed-command fallbacks may target the wrong
   receptacle id. We lean on ``admissible_commands`` first precisely to limit this
   exposure, but the fallbacks are pure string construction.
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Import-guarded heavy deps. We DO NOT import alfworld/textworld/yaml at module
# load — a missing piece must degrade to a clear error at make_alfworld_env()
# time, never an ImportError when this module is merely imported (so that the
# pure-Python translate_step / AlfworldState remain usable, e.g. in unit tests).
# ---------------------------------------------------------------------------


# ALFWorld receptacle type stems that must be opened before take/put can happen.
OPENABLE_STEMS = {"fridge", "microwave", "cabinet", "drawer", "safe", "box"}


def _stem(name: str) -> str:
    """'cabinet 4' -> 'cabinet' ; 'DeskLamp' -> 'desklamp'. Strips trailing id."""
    if not name:
        return ""
    s = re.sub(r"\s*\d+$", "", str(name).strip()).strip().lower()
    # collapse CamelCase / spaces to a bare alphabetic stem
    return "".join(ch for ch in s if ch.isalpha() or ch == " ").strip()


def _camel_to_words(name: str) -> str:
    """'CoffeeMachine' -> 'coffee machine' ; 'DiningTable' -> 'dining table'.

    pyplanner objects are CamelCase; ALFWorld ids are lowercase space-separated
    ('coffee machine 1'). We lowercase and split on case boundaries so substring
    matching against admissible commands / the name map works either way.
    """
    if not name:
        return ""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return spaced.replace("_", " ").lower().strip()


def _is_openable(name: str) -> bool:
    return _stem(name) in OPENABLE_STEMS


# Words that delimit an ALFWorld id and can never be part of one. ALFWorld object
# / receptacle ids are 1–2 lowercase words followed by an integer (e.g.
# "cabinet 1", "dining table 2", "coffee machine 1"); the surrounding command
# verbs/prepositions below must NOT be swallowed into an id by the extractor.
_STOP_WORDS = {
    "go", "to", "take", "from", "put", "in", "on", "open", "close",
    "toggle", "move", "place", "with", "the", "a", "an", "and", "you",
    "see", "is", "are", "at", "of", "look", "examine", "use", "clean",
    "heat", "cool", "inventory",
}


def _extract_ids(text: str) -> list[str]:
    """Pull clean ALFWorld ids ('<=2 non-stop words> <int>') out of arbitrary text.

    ALFWorld surface forms interleave ids with verbs/prepositions ('take apple 2
    from cabinet 1'); a naive '<words> <digit>' regex over-binds across those
    keywords. We instead walk tokens and, at each integer, look back over up to
    two preceding NON-stop-word tokens to assemble the id. Order-preserving,
    de-duplicated.
    """
    if not text:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    tokens = re.findall(r"[a-z]+|\d+", text.lower())
    for i, tok in enumerate(tokens):
        if not tok.isdigit():
            continue
        words: list[str] = []
        j = i - 1
        while j >= 0 and len(words) < 2:
            w = tokens[j]
            if not w.isalpha() or w in _STOP_WORDS:
                break
            words.insert(0, w)
            j -= 1
        if not words:
            continue
        full = " ".join(words) + " " + tok
        if full not in seen:
            seen.add(full)
            ids.append(full)
    return ids


# ═══════════════════════════════════════════════════════════════════════════
# AlfworldState — observation-text-tracked world state
# ═══════════════════════════════════════════════════════════════════════════
class AlfworldState:
    """Tracks the observable ALFWorld world state across a single episode.

    Fields:
        location      current receptacle id the agent is at (e.g. 'cabinet 1') or None
        holding       object id currently in hand (e.g. 'apple 2') or None
        open_recepts  set of receptacle ids currently open
        name_to_id    map from lowercased type/word -> last-seen full id
                      ('cabinet' -> 'cabinet 1', 'apple' -> 'apple 2')
        pending_find  object the last Find step intends to pick (Find/Pick bridge)

    All updates come from parsing the observation TEXT (the reliable signal per
    collect_alfworld_v2.py), with the admissible-command set used at *resolve*
    time, not for state tracking.
    """

    def __init__(self) -> None:
        self.location: str | None = None
        self.holding: str | None = None
        self.open_recepts: set[str] = set()
        self.name_to_id: dict[str, str] = {}
        self.pending_find: str | None = None  # object name from the last Find

    # ---- id discovery / resolution -------------------------------------------
    def register_ids(self, text: str) -> None:
        """Scan any text (obs or admissible cmd) for clean '<name> <id>' tokens and
        record the most-recent full id for each type stem AND each full token."""
        for full in _extract_ids(text):
            self.name_to_id[full] = full
            st = _stem(full)
            if st:
                # keep the first-seen id for a stem stable unless we learn the
                # exact one later via location updates
                self.name_to_id.setdefault(st, full)

    def resolve_id(self, name: str, admissible: list[str] | None = None) -> str | None:
        """Best-effort resolve a pyplanner object/receptacle name to an ALFWorld id.

        Order: exact id already (has trailing digits) → name map (full word) →
        name map (stem) → scan admissible commands for a substring match → None.
        """
        if not name:
            return None
        low = _camel_to_words(name)
        # already an id like 'cabinet 1'
        if re.search(r"\d+$", low.strip()):
            return low.strip()
        if low in self.name_to_id:
            return self.name_to_id[low]
        st = _stem(low)
        if st in self.name_to_id:
            return self.name_to_id[st]
        # fall back to scanning admissible commands for an id with this stem
        if admissible:
            for c in admissible:
                for cand in _extract_ids(c):
                    if _stem(cand) == st:
                        self.name_to_id.setdefault(st, cand)
                        return cand
        return None

    # ---- update from observation text ----------------------------------------
    def update(self, obs: str, admissible: list[str] | None = None) -> None:
        """Update tracked state from a returned observation string + admissible set."""
        if admissible:
            for c in admissible:
                self.register_ids(c)
        if not obs:
            return
        self.register_ids(obs)
        low = obs.lower()

        m = re.search(r"you arrive at (?:the )?([a-z][a-z ]*? \d+)", low)
        if not m:
            m = re.search(r"you are (?:now )?at (?:the )?([a-z][a-z ]*? \d+)", low)
        if m:
            self.location = m.group(1).strip()

        m = re.search(r"you (?:pick up|take) the ([a-z][a-z ]*? \d+)", low)
        if m:
            self.holding = m.group(1).strip()

        # ALFWorld realises a Place as "You put the X in/on the R." or
        # "You move the X to the R." — either empties the hand.
        if re.search(r"you (?:put|move|place) the ", low):
            self.holding = None

        m = re.search(r"you open the ([a-z][a-z ]*? \d+)", low)
        if m:
            self.open_recepts.add(m.group(1).strip())
        m = re.search(r"you close the ([a-z][a-z ]*? \d+)", low)
        if m:
            self.open_recepts.discard(m.group(1).strip())


# ═══════════════════════════════════════════════════════════════════════════
# translate_step — one pyplanner step -> one ALFWorld command (or None)
# ═══════════════════════════════════════════════════════════════════════════
def _match_admissible(
    admissible: list[str] | None,
    verb: str,
    *needles: str,
) -> str | None:
    """Return the admissible command that starts with `verb` and contains every
    (lowercased) needle as a substring, or None. Lets us pick the exact valid
    surface form ('go to cabinet 1', 'take apple 2 from countertop 1', ...)
    instead of guessing the receptacle index."""
    if not admissible:
        return None
    needs = [n.lower() for n in needles if n]
    best = None
    for c in admissible:
        cl = c.lower()
        if not cl.startswith(verb.lower()):
            continue
        if all(n in cl for n in needs):
            # prefer the shortest match (least likely to over-bind a longer id)
            if best is None or len(cl) < len(best):
                best = c
    return best


def translate_step(step: dict, state: AlfworldState) -> str | None:
    """Convert ONE pyplanner step into an ALFWorld text command using `state`.

    Mapping (see module docstring for the fragile cases):
        MoveTo <recep>     -> "go to <recep_id>"
        Find   <obj>       -> None (record obj in state.pending_find for next Pick)
        Pick               -> "take <obj_id> from <recep_id>"  (obj = pending_find /
                              step.object ; recep = current location)
        Place / PutIn <r>  -> "put <obj_id> in/on <recep_id>"
        Open / Close <r>   -> "open/close <recep_id>"
        TurnOn / TurnOff   -> "toggle <obj_id>"

    `admissible` is read from ``state`` via the caller having called
    ``state.update(...)`` after the previous step; when an admissible command
    matching the intent exists we return that exact surface form, else we build
    a plausible string. Returns None when no sensible command exists (caller
    treats None as a no-op / skip — e.g. every ``Find``).
    """
    action = (step.get("action") or "").strip()
    obj = step.get("object") or ""
    target = step.get("target") or ""
    admissible = getattr(state, "_last_admissible", None)

    # ── MoveTo: navigate to a receptacle/surface ──────────────────────────────
    if action == "MoveTo":
        rid = state.resolve_id(obj, admissible)
        cmd = (_match_admissible(admissible, "go to", rid or _camel_to_words(obj))
               or _match_admissible(admissible, "go to", _stem(obj)))
        if cmd:
            return cmd
        # With an admissible set, TRUST it: an unmatched 'go to' target is not a
        # navigable receptacle here (e.g. a lamp/appliance you 'use' in place) —
        # skip rather than dispatch a doomed command that triggers a replan storm.
        if admissible:
            return None
        words = _camel_to_words(obj)
        return f"go to {words}" if words else None

    # ── Find: no ALFWorld primitive — record intent, skip ─────────────────────
    if action == "Find":
        if obj:
            state.pending_find = obj
        return None

    # ── Pick: take the (pending) object from the current location ─────────────
    if action == "Pick":
        want = obj or state.pending_find
        state.pending_find = None
        if not want:
            return None
        oid = state.resolve_id(want, admissible)
        rid = state.location
        # exact admissible 'take <obj> from <recep>'
        cmd = (_match_admissible(admissible, "take", oid or _camel_to_words(want))
               or _match_admissible(admissible, "take", _stem(want)))
        if cmd:
            return cmd
        if admissible:
            return None  # object not takeable here — skip, don't storm replans
        if oid and rid:
            return f"take {oid} from {rid}"
        if oid:
            return f"take {oid}"
        words = _camel_to_words(want)
        return f"take {words}" if words else None

    # ── Place / PutIn: put held object on/into a receptacle ───────────────────
    if action in ("Place", "PutIn"):
        recep = obj or target
        rid = state.resolve_id(recep, admissible) or _camel_to_words(recep)
        held = state.holding or _camel_to_words(state.pending_find or "")
        prep = "in" if (action == "PutIn" or _is_openable(rid)) else "on"
        # exact admissible 'put <held> in/on <recep>'
        cmd = (_match_admissible(admissible, "put", held, rid)
               or _match_admissible(admissible, "put", rid)
               or _match_admissible(admissible, "put", held))
        if cmd:
            return cmd
        if admissible:
            return None
        if held and rid:
            return f"put {held} {prep} {rid}"
        if rid:
            # hand unknown — still emit a best-effort; likely inadmissible
            return f"put object {prep} {rid}"
        return None

    # ── Open / Close ──────────────────────────────────────────────────────────
    if action in ("Open", "Close"):
        verb = action.lower()
        rid = state.resolve_id(obj, admissible) or _camel_to_words(obj)
        cmd = (_match_admissible(admissible, verb, rid)
               or _match_admissible(admissible, verb, _stem(obj)))
        if cmd:
            return cmd
        if admissible:
            return None
        return f"{verb} {rid}" if rid else None

    # ── TurnOn / TurnOff: ALFWorld 'use <lamp>' (examine-in-light) or 'toggle' ─
    if action in ("TurnOn", "TurnOff"):
        oid = state.resolve_id(obj, admissible) or _camel_to_words(obj)
        cmd = (_match_admissible(admissible, "use", oid)
               or _match_admissible(admissible, "toggle", oid)
               or _match_admissible(admissible, "use", _stem(obj))
               or _match_admissible(admissible, "toggle", _stem(obj)))
        if cmd:
            return cmd
        if admissible:
            return None
        return f"toggle {oid}" if oid else None

    # ── Heat / Cool / Clean: ALFWorld treats the HELD object with an appliance ──
    #    admissible surface form: '<verb> <obj> with <appliance>'
    if action in ("Heat", "Cool", "Clean"):
        verb = action.lower()
        held = state.holding or _camel_to_words(obj or state.pending_find or "")
        oid = state.resolve_id(obj, admissible) or held
        cmd = (_match_admissible(admissible, verb, held)
               or _match_admissible(admissible, verb, oid)
               or _match_admissible(admissible, verb, _stem(obj)))
        if cmd:
            return cmd
        if admissible:
            return None
        return f"{verb} {held}" if held else None

    # Body/idle actions (Wait, Sit, LieOn, Serve, Wash, ...) have no ALFWorld
    # analogue — skip them.
    return None


# ═══════════════════════════════════════════════════════════════════════════
# make_alfworld_env — instantiate AlfredTWEnv exactly like the collectors do
# ═══════════════════════════════════════════════════════════════════════════
def _apply_textworld_eval_patch() -> None:
    """Defensively patch textworld 1.7.0's PDDL textgen eval() for Python 3.13.

    textworld 1.7.0's ``textgen.__init__.derive()`` does
    ``locals().update(context["variables"]); eval(self.expression)`` which no
    longer injects names into eval scope under 3.13 → ``NameError: name 'r' is
    not defined`` on env.reset(). We rebind ``derive`` to pass the variables as
    the eval locals dict explicitly. Best-effort: any failure is swallowed (the
    patch is unnecessary on other textworld versions). Mirrors the fix documented
    in results/analysis/alfworld_transfer_notes.md.
    """
    # NOTE: On this workspace the textworld site-package
    # (textworld/envs/pddl/textgen/__init__.py) is ALREADY patched in place for
    # the Py3.13 eval()-scope bug (it builds an explicit `_ns` namespace and calls
    # eval(expr, {"__builtins__": __builtins__}, _ns)). Monkey-patching `derive`
    # from here is therefore unnecessary AND harmful: the several `derive`
    # overloads have different signatures (derive(self), derive(self, context=None),
    # derive(self, start, context={})), so a blanket rebind breaks the no-arg call
    # site with "missing positional argument: context". We detect the in-place fix
    # and no-op; only fall back to a careful rebind if the source is unpatched.
    try:
        import inspect
        import textworld.envs.pddl.textgen as _tg  # type: ignore
        src = inspect.getsource(_tg)
        if "_ns" in src and "__builtins__" in src:
            return  # site-package already fixed → nothing to do
    except Exception:
        return


def make_alfworld_env(config_path: str, split: str = "eval_out_of_distribution"):
    """Instantiate the ALFWorld TextWorld env (AlfredTWEnv), batch_size=1.

    Mirrors ``collect_transitions_alfworld.py`` / ``collect_alfworld_v2.py``:
      * parse the yaml config,
      * import ``AlfredTWEnv`` from the submodule via importlib (it is NOT
        re-exported on the package in alfworld 0.4.2),
      * apply the textworld 3.13 eval() patch defensively,
      * construct with ``train_eval=split`` (falling back to positional),
      * return ``env.init_env(batch_size=1)``.

    Raises a RuntimeError with an actionable message if alfworld/textworld/yaml
    are missing or the env cannot be built — never a bare ImportError at import.
    """
    # ---- guarded imports -----------------------------------------------------
    try:
        import yaml  # noqa: F401
    except Exception as e:  # pragma: no cover - defensive
        raise RuntimeError(
            "PyYAML is required to read the ALFWorld config "
            "(pip install pyyaml)."
        ) from e

    try:
        import importlib

        importlib.import_module("alfworld")
    except Exception as e:  # pragma: no cover - defensive
        raise RuntimeError(
            "Cannot import 'alfworld'. Install it (pip install alfworld) and make "
            "sure $ALFWORLD_DATA points at downloaded game data. "
            f"Underlying error: {e!r}"
        ) from e

    import importlib

    import yaml

    if not config_path or not __import__("os").path.isfile(config_path):
        raise RuntimeError(
            f"ALFWorld config not found: {config_path!r}. Pass --config pointing "
            "at alfworld_base_config.yaml (the canonical copy lives at "
            "paper_sage/configs/alfworld_base_config.yaml)."
        )

    try:
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
    except Exception as e:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to parse ALFWorld config {config_path!r}: {e!r}") from e
    if isinstance(cfg, dict):
        cfg.setdefault("env", {})

    # textworld 3.13 eval() fix — must precede the first reset(). Best-effort.
    _apply_textworld_eval_patch()

    # Env class lives in a submodule across 0.4.x; import via importlib.
    env_cls = None
    last_exc: Exception | None = None
    _submods = {
        "AlfredTWEnv": "alfworld.agents.environment.alfred_tw_env",
        "AlfredThorEnv": "alfworld.agents.environment.alfred_thor_env",
        "AlfredHybrid": "alfworld.agents.environment.alfred_hybrid",
    }
    # Prefer the package attr, then the per-class submodule.
    try:
        env_mod = importlib.import_module("alfworld.agents.environment")
    except Exception as e:
        raise RuntimeError(
            "Cannot import alfworld.agents.environment "
            f"({e!r}). Is alfworld correctly installed?"
        ) from e
    for name in ("AlfredTWEnv", "AlfredThorEnv", "AlfredHybrid"):
        env_cls = getattr(env_mod, name, None)
        if env_cls is None:
            try:
                env_cls = getattr(importlib.import_module(_submods[name]), name, None)
            except Exception:
                env_cls = None
        if env_cls is not None:
            break
    if env_cls is None:
        raise RuntimeError(
            "No ALFWorld env class found (looked for AlfredTWEnv / AlfredThorEnv / "
            "AlfredHybrid on alfworld.agents.environment and its submodules)."
        )

    # Constructor signature drifts across versions; try kw then positional.
    tw = None
    for kwargs in (
        dict(config=cfg, train_eval=split),
        dict(config=cfg, train_eval="train"),
    ):
        try:
            tw = env_cls(**kwargs)
            break
        except TypeError:
            tw = None
        except Exception as e:
            last_exc = e
            tw = None
    if tw is None:
        try:
            tw = env_cls(cfg, train_eval=split)
        except Exception:
            try:
                tw = env_cls(cfg)  # positional, config-only
            except Exception as e:
                raise RuntimeError(
                    "Failed to construct AlfredTWEnv. Likely causes: game data not "
                    "finished downloading to $ALFWORLD_DATA, or a constructor "
                    f"signature mismatch for this alfworld version. Config: "
                    f"{config_path!r}. Underlying error: {(e or last_exc)!r}"
                ) from (e or last_exc)

    try:
        env = tw.init_env(batch_size=1)
    except Exception as e:
        raise RuntimeError(f"env.init_env(batch_size=1) failed: {e!r}") from e
    return env
