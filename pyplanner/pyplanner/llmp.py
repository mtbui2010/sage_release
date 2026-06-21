# pyplanner/llmp.py
# ─────────────────────────────────────────────────────────────────────
# LLM+P — the classic "LLM translates to PDDL, a classical planner solves"
# baseline (Liu et al., 2023, arXiv:2304.11477), implemented honestly on the
# pyplanner action vocabulary.
#
# Pipeline per task (.generate_plan):
#   1. NL  -> PDDL problem  (ONE LLM call).  The PDDL *domain* is FIXED and
#      hand-authored (pddl/household_domain.pddl) — the LLM only emits the
#      (:objects ...) (:init ...) (:goal ...) blocks.  This is the strongest,
#      most honest form of LLM+P (domain fixed, problem translated).
#   2. SOLVE domain+problem with pyperplan (pure-python classical planner).
#   3. PDDL plan -> STEP_SCHEMA step dicts (pure Python, deterministic).
#
# Robust fallback (CRITICAL — classical solvers fail often on LLM PDDL):
#   On parse error / solver failure / timeout / empty plan, fall back to a
#   single Direct-style LLM call so the method ALWAYS returns a usable plan,
#   recording meta['llmp_fallback'] = True and a typed meta['llmp_error'].
#
# Imports are guarded: this module imports cleanly even if pyperplan is not
# installed (pyperplan is imported lazily inside _solve_pddl).
#
# Registry key: "LLM+P".
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import tempfile
import time

from pyplanner.base import (
    ACTIONS_STR, JSON_EXAMPLE, STEP_SCHEMA,
    BasePlanner, PlanMetrics, parse_steps,
)

# ─────────────────────────────────────────────────────────────────────
# Locate the fixed PDDL domain.
# Resolution order:
#   1. $PYPLANNER_PDDL_DOMAIN (explicit override)
#   2. <pyplanner repo root>/pddl/household_domain.pddl   (installed location)
#   3. embedded fallback string (so the module is self-contained even if the
#      .pddl file was not copied alongside the package)
# ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_DEFAULT_DOMAIN_PATH = os.path.join(_REPO_ROOT, "pddl", "household_domain.pddl")

# Embedded copy — kept byte-compatible with pddl/household_domain.pddl so the
# planner works even on a minimal install that did not ship the data file.
_EMBEDDED_DOMAIN = """\
(define (domain household)
  (:requirements :strips :typing :negative-preconditions)
  (:types obj - object)
  (:predicates
    (arrived ?l - obj) (found ?o - obj) (holding ?o - obj)
    (hand-empty) (opened ?c - obj) (turned-on ?a - obj))
  (:action moveto
    :parameters (?to - obj ?from - obj ?f - obj)
    :precondition (and (arrived ?from) (found ?f))
    :effect (and (arrived ?to) (not (arrived ?from)) (not (found ?f))))
  (:action moveto-nofound
    :parameters (?to - obj ?from - obj)
    :precondition (and (arrived ?from))
    :effect (and (arrived ?to) (not (arrived ?from))))
  (:action find
    :parameters (?o - obj ?at - obj)
    :precondition (and (arrived ?at))
    :effect (and (found ?o)))
  (:action pick
    :parameters (?o - obj)
    :precondition (and (found ?o) (hand-empty))
    :effect (and (holding ?o) (not (hand-empty)) (not (found ?o))))
  (:action place
    :parameters (?o - obj ?recep - obj)
    :precondition (and (holding ?o) (arrived ?recep))
    :effect (and (hand-empty) (not (holding ?o))))
  (:action putin
    :parameters (?o - obj ?c - obj)
    :precondition (and (holding ?o) (found ?c) (opened ?c))
    :effect (and (hand-empty) (not (holding ?o))))
  (:action open
    :parameters (?c - obj)
    :precondition (and (found ?c) (not (opened ?c)))
    :effect (and (opened ?c)))
  (:action close
    :parameters (?c - obj)
    :precondition (and (found ?c) (opened ?c))
    :effect (and (not (opened ?c))))
  (:action turnon
    :parameters (?a - obj)
    :precondition (and (found ?a) (not (turned-on ?a)))
    :effect (and (turned-on ?a)))
  (:action turnoff
    :parameters (?a - obj)
    :precondition (and (found ?a) (turned-on ?a))
    :effect (and (not (turned-on ?a))))
)
"""


def load_domain(path: str | None = None) -> str:
    """Return the fixed domain PDDL string.

    Prefers $PYPLANNER_PDDL_DOMAIN, then the on-disk data file, then the
    embedded fallback.  Never raises — a missing file degrades to embedded.
    """
    candidates = []
    if path:
        candidates.append(path)
    env_p = os.getenv("PYPLANNER_PDDL_DOMAIN", "")
    if env_p:
        candidates.append(env_p)
    candidates.append(_DEFAULT_DOMAIN_PATH)
    for p in candidates:
        try:
            if p and os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as fh:
                    txt = fh.read().strip()
                if txt:
                    return txt
        except OSError:
            continue
    return _EMBEDDED_DOMAIN


# Map grounded PDDL operator name → (ROBOT_ACTION, which-param-is-object).
# pyperplan grounds actions as "(opname arg1 arg2 ...)"; we back-map to the
# {"action","object"} STEP_SCHEMA dicts.  The object slot follows STEP_SCHEMA:
#   MoveTo  -> destination (?to)        Find  -> target (?o)
#   Pick    -> object ignored at runtime, but we keep the picked object name
#   Place   -> receptacle (?recep)      PutIn -> container (?c)
#   Open/Close/TurnOn/TurnOff -> target (?c / ?a)
_OP_TO_ACTION: dict[str, tuple[str, int]] = {
    "moveto":          ("MoveTo",  0),   # ?to
    "moveto-nofound":  ("MoveTo",  0),   # ?to
    "find":            ("Find",    0),   # ?o
    "pick":            ("Pick",    0),   # ?o (object field documented as ignored)
    "place":           ("Place",   1),   # ?recep
    "putin":           ("PutIn",   1),   # ?c
    "open":            ("Open",    0),   # ?c
    "close":           ("Close",   0),   # ?c
    "turnon":          ("TurnOn",  0),   # ?a
    "turnoff":         ("TurnOff", 0),   # ?a
}


# ─────────────────────────────────────────────────────────────────────
# Prompting:  NL + visible_objects -> PDDL problem (objects/init/goal only)
# ─────────────────────────────────────────────────────────────────────
_PROBLEM_SYSTEM = """\
You are a PDDL translator for a household robot.  A FIXED PDDL domain named
`household` is already defined; you must NOT redefine it.  Translate the
user's request into a PDDL PROBLEM only.

The domain predicates you may use are EXACTLY:
  (arrived ?l)    robot is at location/furniture/container ?l
  (found ?o)      robot has located object ?o
  (holding ?o)    object ?o is in the gripper
  (hand-empty)    gripper is free
  (opened ?c)     container ?c is open
  (turned-on ?a)  appliance ?a is on

The domain actions (for your reasoning only — do NOT emit them) are:
  moveto, find, pick, place, putin, open, close, turnon, turnoff.

RULES for the problem you emit:
- Declare EVERY constant you reference in (:objects ...) with type - obj.
  Use CamelCase names exactly as in the visible-object list (Apple, Mug,
  Fridge, CoffeeMachine, DiningTable, Kitchen, ...).  Object names must be
  valid PDDL identifiers (letters, digits, hyphen — no spaces, no quotes).
- The (:init ...) MUST contain exactly one (arrived <Start>) fact and a
  (hand-empty) fact, plus any (opened ...) / (turned-on ...) that already
  hold.  Use a generic start location named `StartPos` if none is obvious.
- The (:goal ...) is a conjunction of the target predicates that make the
  task complete.  Examples:
    * "bring the apple to the table" -> (and (arrived DiningTable) (holding Apple))
      (or, if it should be put down) (and (arrived DiningTable) (hand-empty))
      Prefer encoding the final placed state as (and (... )); if the object
      must end ON a receptacle, use a (holding ...) -> place ... so keep the
      goal as (and (arrived <Recep>) (hand-empty)) plus the object having been
      picked.  When in doubt, make the goal `(holding <Object>)` for fetch
      tasks and `(turned-on <Appliance>)` for power-on tasks.
    * "turn on the coffee machine" -> (turned-on CoffeeMachine)
    * "open the fridge" -> (opened Fridge)
- Keep it minimal and SOLVABLE: only include objects relevant to the task.

Return ONLY the PDDL problem, no markdown, no commentary:
(define (problem household-task)
  (:domain household)
  (:objects ...)
  (:init ...)
  (:goal (and ...)))"""

_PROBLEM_EXAMPLE = """\
Example.
Task: Bring the apple from the fridge to the dining table.
Visible: Fridge, Apple, DiningTable, Kitchen
PDDL problem:
(define (problem household-task)
  (:domain household)
  (:objects StartPos Kitchen Fridge Apple DiningTable - obj)
  (:init (arrived StartPos) (hand-empty))
  (:goal (and (arrived DiningTable) (hand-empty))))"""


def nl_to_problem_prompt(task: str, obs: str, visible_objects: list[str]) -> list[dict]:
    """Build the chat messages that ask the LLM for a PDDL problem only."""
    obj_str = ", ".join(visible_objects[:30]) if visible_objects else "none listed"
    user = (
        f"{_PROBLEM_EXAMPLE}\n\n"
        f"Now translate THIS task.\n"
        f"Task: {task}\n"
        f"Observation:\n{obs}\n"
        f"Visible objects: {obj_str}\n\n"
        f"PDDL problem:"
    )
    return [
        {"role": "system", "content": _PROBLEM_SYSTEM},
        {"role": "user", "content": user},
    ]


# ─────────────────────────────────────────────────────────────────────
# Direct-style fallback prompt (used when classical solving fails)
# ─────────────────────────────────────────────────────────────────────
_FALLBACK_SYSTEM = f"""You are a household assistant robot planner.
Generate a step-by-step action plan for the robot.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

Return ONLY valid JSON with no markdown and no explanation:
{JSON_EXAMPLE}"""


# ─────────────────────────────────────────────────────────────────────
# PDDL problem extraction / sanitisation
# ─────────────────────────────────────────────────────────────────────
def _extract_problem_block(raw: str) -> str | None:
    """Pull the first balanced `(define (problem ...) ...)` s-expression.

    Tolerates markdown fences and leading/trailing prose.  Returns None if no
    balanced define-problem block is present.
    """
    if not raw:
        return None
    txt = re.sub(r"```(?:pddl|lisp)?", "", raw).replace("```", "")
    # Find "(define (problem" and read until parentheses balance.
    m = re.search(r"\(\s*define\s*\(\s*problem", txt, re.IGNORECASE)
    if not m:
        return None
    start = m.start()
    depth = 0
    for i in range(start, len(txt)):
        c = txt[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return txt[start:i + 1]
    return None  # unbalanced — let caller treat as parse failure


def _strip_comments(pddl: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in pddl.splitlines())


# ─────────────────────────────────────────────────────────────────────
# Solving with pyperplan (lazy import + guard)
# ─────────────────────────────────────────────────────────────────────
def _resolve_search_and_heuristic(search: str, heuristic: str):
    """Return (search_fn, heuristic_cls_or_None) using pyperplan's public maps.

    Primary path: pyperplan.planner.SEARCHES / HEURISTICS — the documented,
    stable name→callable maps used by pyperplan's own CLI.  Fallback path:
    direct submodule imports, for layout drift across versions.

    Raises ImportError if pyperplan is absent entirely.
    """
    try:
        from pyperplan import planner as _pp_planner  # type: ignore
    except Exception as e:
        _pp_planner = None
        _planner_err = e
    else:
        _planner_err = None

    search_fn = None
    heur_cls = None

    if _pp_planner is not None:
        searches = getattr(_pp_planner, "SEARCHES", {}) or {}
        heuristics = getattr(_pp_planner, "HEURISTICS", {}) or {}
        # pyperplan CLI search keys: bfs, dfs, ehs, astar, wastar, gbf, ids, sat
        search_fn = searches.get(search) or searches.get("wastar") \
            or searches.get("astar") or searches.get("bfs")
        # pyperplan CLI heuristic keys: blind, hadd, hff, hmax, hsa, landmark, lmcut
        heur_cls = heuristics.get(heuristic) or heuristics.get("hadd") \
            or heuristics.get("hff") or heuristics.get("blind")

    if search_fn is None:
        # Fallback: import the search functions directly from submodules.
        try:
            from pyperplan.search.breadth_first_search import breadth_first_search
            search_fn = breadth_first_search
        except Exception:
            pass
        try:
            from pyperplan.search.a_star import (  # noqa
                astar_search, weighted_astar_search, greedy_best_first_search,
            )
            search_fn = {
                "astar": astar_search, "wastar": weighted_astar_search,
                "gbf": greedy_best_first_search,
            }.get(search, search_fn or weighted_astar_search)
        except Exception:
            pass

    if heur_cls is None:
        try:
            from pyperplan.heuristics.relaxation import (  # noqa
                hAddHeuristic, hFFHeuristic, hMaxHeuristic,
            )
            heur_cls = {
                "hadd": hAddHeuristic, "hff": hFFHeuristic, "hmax": hMaxHeuristic,
            }.get(heuristic, hAddHeuristic)
        except Exception:
            try:
                from pyperplan.heuristics.blind import BlindHeuristic
                heur_cls = BlindHeuristic
            except Exception:
                heur_cls = None

    if search_fn is None:
        raise ImportError(
            "pyperplan present but no usable search function found "
            f"(planner import: {_planner_err})"
        )
    return search_fn, heur_cls


def _solve_pddl(domain_str: str, problem_str: str,
                search: str = "wastar", heuristic: str = "hadd") -> list[str]:
    """Solve a domain+problem with pyperplan; return grounded action strings.

    Each returned element is like "(moveto kitchen fridge apple)" — lowercased
    op name + grounded args, matching pyperplan's Operator.name.

    Raises ImportError if pyperplan is absent; RuntimeError on parse/solve
    failure or unsolvable problem (empty plan).  The caller catches both and
    falls back.
    """
    try:
        from pyperplan.pddl.parser import Parser
        from pyperplan.grounding import ground
    except Exception as e:  # ImportError or internal layout change
        raise ImportError(f"pyperplan unavailable: {e}") from e

    search_fn, heur_cls = _resolve_search_and_heuristic(search, heuristic)

    # pyperplan's Parser reads from files; write temp files.
    dtf = tempfile.NamedTemporaryFile("w", suffix="_domain.pddl",
                                      delete=False, encoding="utf-8")
    ptf = tempfile.NamedTemporaryFile("w", suffix="_problem.pddl",
                                      delete=False, encoding="utf-8")
    try:
        dtf.write(domain_str)
        dtf.flush()
        dtf.close()
        ptf.write(problem_str)
        ptf.flush()
        ptf.close()

        parser = Parser(dtf.name, ptf.name)
        domain = parser.parse_domain()
        problem = parser.parse_problem(domain)
        task = ground(problem)

        sf_name = getattr(search_fn, "__name__", "")
        # Blind / uninformed searches take only the task; heuristic searches
        # take (task, heuristic_instance).  Detect by trying the heuristic form
        # first when a heuristic class is available, then degrade gracefully.
        plan = None
        heuristic_searches = {
            "astar_search", "weighted_astar_search",
            "greedy_best_first_search", "enforced_hillclimbing_search",
        }
        # Hard wall-clock cap on the classical search: some LLM-authored PDDL
        # problems make pyperplan loop ~indefinitely (it has no internal timeout).
        # SIGALRM works because the planner runs in the main thread; on timeout we
        # raise so the caller falls back to a Direct-style plan. (No-op on platforms
        # without SIGALRM.)
        import signal as _sig
        _solve_to = int(float(os.getenv("LLMP_SOLVE_TIMEOUT", "20")))
        _has_alarm = hasattr(_sig, "SIGALRM")
        def _on_to(signum, frame):
            raise RuntimeError(f"pyperplan solve exceeded {_solve_to}s — abandoning")
        _old = None
        if _has_alarm:
            _old = _sig.signal(_sig.SIGALRM, _on_to); _sig.alarm(_solve_to)
        try:
            if heur_cls is not None and sf_name in heuristic_searches:
                try:
                    plan = search_fn(task, heur_cls(task))
                except TypeError:
                    plan = search_fn(task)
            else:
                try:
                    plan = search_fn(task)
                except TypeError:
                    # Search unexpectedly needs a heuristic — supply one.
                    if heur_cls is not None:
                        plan = search_fn(task, heur_cls(task))
                    else:
                        raise
        finally:
            if _has_alarm:
                _sig.alarm(0); _sig.signal(_sig.SIGALRM, _old)

        if not plan:
            raise RuntimeError("solver returned no plan (unsolvable or empty)")

        # plan is an iterable of Operator; .name is "(opname a b c)".
        out = []
        for op in plan:
            name = getattr(op, "name", None) or str(op)
            out.append(str(name).strip())
        if not out:
            raise RuntimeError("solver plan had no operators")
        return out
    finally:
        for f in (dtf, ptf):
            try:
                os.unlink(f.name)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────
# Grounded PDDL actions -> STEP_SCHEMA dicts
# ─────────────────────────────────────────────────────────────────────
def pddl_plan_to_steps(grounded: list[str]) -> list[dict]:
    """Map grounded pyperplan action strings to STEP_SCHEMA step dicts.

    Input items look like "(moveto kitchen fridge apple)".  Output items are
    {"action": <ROBOT_ACTION>, "object": <CamelCaseArg>}.  Unknown operators
    are dropped (covered by _OP_TO_ACTION; check_llmp asserts full coverage).

    Object names are restored from the lowercased grounded args by best-effort
    title-casing only when the original casing is lost; pyperplan lowercases
    everything, so we map back to the canonical visible-object casing via an
    optional `casing` table supplied by the planner.
    """
    steps: list[dict] = []
    for raw in grounded:
        toks = raw.strip().strip("()").split()
        if not toks:
            continue
        op = toks[0].lower()
        args = toks[1:]
        mapped = _OP_TO_ACTION.get(op)
        if mapped is None:
            continue
        action, obj_idx = mapped
        obj = args[obj_idx] if obj_idx < len(args) else ""
        steps.append({"action": action, "object": obj})
    return steps


def _recase_steps(steps: list[dict], visible_objects: list[str]) -> list[dict]:
    """Restore canonical CamelCase object names lost to pyperplan lowercasing."""
    table = {v.lower(): v for v in (visible_objects or [])}
    out = []
    for s in steps:
        o = s.get("object", "") or ""
        canon = table.get(o.lower())
        if canon is None and o:
            # Title-case fallback (kitchen -> Kitchen, coffeemachine -> Coffeemachine).
            canon = o[:1].upper() + o[1:]
        out.append({"action": s["action"], "object": canon or o})
    return out


# ─────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────
class LLMPPlanner(BasePlanner):
    name = "LLM+P"
    description = (
        "LLM+P (Liu et al. 2023): the LLM translates the task into a PDDL "
        "problem against a fixed hand-authored household domain; a classical "
        "planner (pyperplan) solves it; the grounded plan is mapped back to "
        "robot steps. Falls back to a Direct LLM call when classical solving "
        "fails."
    )

    def __init__(self, *args, domain_path: str | None = None,
                 search: str = "wastar", heuristic: str = "hadd",
                 repair_retries: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._domain_path = domain_path
        self._domain_str = load_domain(domain_path)
        self._search = search
        self._heuristic = heuristic
        # Capped syntactic-repair retries (re-prompt with parser error). Keep
        # at 1 so LLM+P does not become "unlimited refine"; report the count.
        self._repair_retries = max(0, int(repair_retries))

    # ── Direct-style fallback (always returns a usable plan) ──
    def _fallback_direct(self, task, obs, visible_objects):
        user = self._context_str(task, obs, visible_objects) + \
            "\n\nGenerate the complete step-by-step plan:"
        raw, in_tok, out_tok = self._chat(
            [{"role": "system", "content": _FALLBACK_SYSTEM},
             {"role": "user", "content": user}],
            temperature=0.2,
        )
        return parse_steps(raw), in_tok, out_tok

    def _translate_and_solve(self, task, obs, visible_objects):
        """Returns (steps, in_tok, out_tok, llm_calls, error_code).

        error_code is "" on classical success, else a typed tag.  steps is []
        when classical solving failed (caller then runs the fallback).
        """
        in_tok = out_tok = 0
        llm_calls = 0
        last_err = ""
        problem_str = None

        msgs = nl_to_problem_prompt(task, obs, visible_objects)
        # Translation + capped syntactic-repair retries.
        for attempt in range(self._repair_retries + 1):
            try:
                raw, it, ot = self._chat(msgs, temperature=0.0)
            except Exception as e:
                return [], in_tok, out_tok, llm_calls, f"translate_call_error:{type(e).__name__}"
            in_tok += it
            out_tok += ot
            llm_calls += 1

            block = _extract_problem_block(raw)
            if block is None:
                last_err = "no_problem_block"
                # Re-prompt with the failure, if retries remain.
                msgs = msgs + [
                    {"role": "assistant", "content": raw[:1500]},
                    {"role": "user", "content":
                        "That was not a parseable PDDL problem. Re-emit ONLY a "
                        "single (define (problem ...) (:domain household) "
                        "(:objects ...) (:init ...) (:goal (and ...))) block."},
                ]
                continue

            problem_str = _strip_comments(block)
            try:
                grounded = _solve_pddl(
                    self._domain_str, problem_str,
                    search=self._search, heuristic=self._heuristic,
                )
            except ImportError as e:
                # pyperplan absent — cannot do classical solving at all.
                return [], in_tok, out_tok, llm_calls, f"pyperplan_missing:{e}"[:120]
            except Exception as e:
                last_err = f"solve_error:{type(e).__name__}"
                # Re-prompt with the solver error, if retries remain.
                msgs = msgs + [
                    {"role": "assistant", "content": block[:1500]},
                    {"role": "user", "content":
                        f"The classical planner rejected that problem "
                        f"({type(e).__name__}: {str(e)[:160]}). Re-emit a "
                        f"corrected, SOLVABLE PDDL problem (declare every "
                        f"constant in :objects, include (hand-empty) and one "
                        f"(arrived ...) in :init, keep the goal reachable)."},
                ]
                continue

            steps = pddl_plan_to_steps(grounded)
            steps = _recase_steps(steps, visible_objects)
            if not steps:
                last_err = "empty_mapped_plan"
                continue
            return steps, in_tok, out_tok, llm_calls, ""

        return [], in_tok, out_tok, llm_calls, last_err or "translate_failed"

    def generate_plan(self, task, obs, visible_objects):
        t0 = time.perf_counter()
        llm_calls = 0
        in_tok = out_tok = 0
        fallback = False
        error_code = ""

        try:
            steps, it, ot, calls, error_code = self._translate_and_solve(
                task, obs, visible_objects)
            in_tok += it
            out_tok += ot
            llm_calls += calls
        except Exception as e:
            steps, error_code = [], f"unexpected:{type(e).__name__}"

        # Fallback: classical pipeline produced nothing usable.
        if not steps:
            fallback = True
            try:
                fsteps, it, ot = self._fallback_direct(task, obs, visible_objects)
                in_tok += it
                out_tok += ot
                llm_calls += 1
                steps = fsteps
            except Exception as e:
                steps = []
                error_code = (error_code + f"|fallback_error:{type(e).__name__}").strip("|")

        ok = bool(steps)
        metrics = PlanMetrics(
            method=self.name,
            model=self.model,
            backend=self.provider,
            latency_s=time.perf_counter() - t0,
            llm_calls=llm_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            num_steps=len(steps),
            parse_ok=ok,
            notes=error_code,
            extra={
                "llmp_fallback": fallback,
                "llmp_error": error_code,
                "llmp_solve_s": round(time.perf_counter() - t0, 3),
            },
        )
        tag = "FALLBACK" if fallback else "PDDL"
        print(f"[{self.name}] {tag} {len(steps)} steps in {metrics.latency_s:.1f}s"
              + (f" (err={error_code})" if error_code else ""))
        return steps, metrics

    def replan(self, task, completed, failed_step, failure_reason,
               obs, visible_objects):
        """Re-translate the residual goal and re-solve.

        Mirrors LLM-DP's re-solve-on-divergence.  We append a note about the
        completed prefix and failure to the task so the LLM asserts the already
        achieved effects in :init; on classical failure, fall back to a Direct
        replan so a usable suffix is always returned.
        """
        t0 = time.perf_counter()
        done = ", ".join(
            f"{s.get('action')} {s.get('object')}" for s in completed
        ) or "nothing yet"
        residual_task = (
            f"{task}\n(Already completed: {done}. The step "
            f"'{failed_step.get('action')} {failed_step.get('object')}' failed: "
            f"{failure_reason}. Plan ONLY the remaining steps; assert completed "
            f"effects in :init.)"
        )

        llm_calls = 0
        in_tok = out_tok = 0
        fallback = False
        error_code = ""
        try:
            steps, it, ot, calls, error_code = self._translate_and_solve(
                residual_task, obs, visible_objects)
            in_tok += it
            out_tok += ot
            llm_calls += calls
        except Exception as e:
            steps, error_code = [], f"unexpected:{type(e).__name__}"

        if not steps:
            fallback = True
            try:
                user = self._replan_context(task, completed, failed_step,
                                            failure_reason, obs, visible_objects)
                raw, it, ot = self._chat(
                    [{"role": "system", "content": _FALLBACK_SYSTEM},
                     {"role": "user", "content": user}],
                    temperature=0.2,
                )
                in_tok += it
                out_tok += ot
                llm_calls += 1
                steps = parse_steps(raw)
            except Exception as e:
                steps = []
                error_code = (error_code + f"|fallback_error:{type(e).__name__}").strip("|")

        metrics = PlanMetrics(
            method=self.name,
            model=self.model,
            backend=self.provider,
            latency_s=time.perf_counter() - t0,
            llm_calls=llm_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            num_steps=len(steps),
            parse_ok=bool(steps),
            notes=error_code,
            extra={"llmp_fallback": fallback, "llmp_error": error_code},
        )
        return steps, metrics


__all__ = [
    "LLMPPlanner",
    "load_domain",
    "nl_to_problem_prompt",
    "pddl_plan_to_steps",
]
