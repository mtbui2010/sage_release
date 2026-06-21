# Datasheet — SAGE AI2-THOR Household Planning Benchmark

This datasheet follows the structure proposed in *Datasheets for Datasets*
(Gebru et al., 2021). It documents the 75-task AI2-THOR (iTHOR) household
planning benchmark used to evaluate the SAGE embodied planner, together with
the auxiliary splits and out-of-distribution sets shipped alongside it.

The benchmark file is `eval_dataset_expanded.json` (curated + generated, 75
tasks). The curated-only file is `eval_dataset_gt.json` (38 tasks).

---

## Motivation

**For what purpose was the dataset created?**
To provide a small, fully simulator-grounded benchmark of household robot
planning tasks on AI2-THOR (iTHOR) scenes, with a *verified reference plan* per
task. It supports two evaluation modes: (1) reference-free **plan-quality**
scoring against the reference, and (2) **execution** of generated plans in the
live AI2-THOR simulator. The reference plans were not authored by an LLM; the
verification gate is deterministic and LLM-free.

**Who created the dataset and on behalf of whom?**
The authors of the SAGE research artifact (`paper_sage`), layered on the
`pyplanner` library. Scenes and the simulator are AI2-THOR, from the Allen
Institute for AI (see `LICENSE_NOTES.md`).

---

## Composition

**What do the instances represent?**
Each instance is a single household task in one AI2-THOR scene. The record
schema (see `eval_dataset_gt.json`) is:

| Field | Meaning |
|-------|---------|
| `task_id` | Room-prefixed id (`K`/`L`/`B`/`A` = kitchen/living-room/bedroom/bathroom). |
| `task_desc` | Natural-language instruction. |
| `room` | One of `kitchen`, `living_room`, `bedroom`, `bathroom`. |
| `scene` | AI2-THOR `FloorPlan<N>`. |
| `obs` | Text observation (agent pose, holding, nearby objects with distances). |
| `visible_objects` | Object types visible/relevant in the scene. |
| `reference_steps` | The verified reference plan (list of `{action, object, target, reason}`). |
| `expected_objects` | Objects a valid plan must reference (used for completeness scoring). |
| `difficulty` | `easy`, `medium`, or `hard`. |
| `fail_injection` | Optional `{at_step, failure_reason}` for replan/recovery evaluation; `{}` if none. |
| `source` | Provenance tag (`expand_dataset.sim_verified` on generated tasks; absent on curated). |
| `_meta` | Grounding record: `gt_source`, `candidate_source`, sim outcome, timestamp, `verifier` version. |

**How many instances are there?**
75 tasks total:
- **38 curated** — hand-authored (`SAMPLES_RAW` in
  `pyplanner/apps/make_dataset.py`), then simulator-verified.
  `_meta.candidate_source = "manual"`, `_meta.gt_source = "simulator_execution"`.
- **37 generated** — template-generated then simulator-verified
  (`scripts/expand_dataset.py`). `source = "expand_dataset.sim_verified"`,
  `_meta.candidate_source = "template"`, `_meta.gt_source = "simulator_execution"`.

**What is the action vocabulary?**
Canonical robot actions: `MoveTo, Find, Pick, Place, PutIn, Open, Close,
TurnOn, TurnOff` plus body/idle actions (`Wash, Sit, LieOn, Serve, Wait`). The
dataset stores some steps in a **legacy vocabulary** (`Navigate` for
`MoveTo`, `Grab` for `Pick`); these are normalized at load time through
`pyplanner.verifier.normalize_plan` before scoring or execution.

**Is any information missing?**
The `obs` text is a synthesized snapshot, not a full simulator state dump. The
generated tasks carry a terse templated `obs` (`"You are in the {room}.
Relevant objects are visible."`) rather than a pose/distance listing.

**Are instances related?**
Tasks within a scene/room share objects. The auxiliary **COMPOUND** benchmark
(`scripts/gen_compound_tasks.py`) explicitly composes 2- and 3-goal sequences
from same-scene single-goal tasks (74 tasks), with references formed by
concatenating already-verified sub-plans and re-checking them through the
symbolic verifier.

**Recommended splits.**
- For memory-based methods, a **leave-one-out (leak-free)** retrieval protocol
  is used so a task is never retrieved as its own exemplar.
- The **37 generated tasks are held out** from SAGE's seed memory (which stays
  on the curated 38), so evaluation on them is leakage-free by construction (see
  `scripts/run_grid.sh`).

**Auxiliary / OOD sets shipped alongside.**
- **ProcTHOR OOD**: a 70-task set on ProcTHOR-10k *val* houses built with the
  identical template-then-verify discipline (`scripts/build_procthor_dataset.py`,
  `eval_dataset_procthor.json`).
- **COMPOUND**: 74 multi-goal compositions (above).
- **ALFWorld verifier-induction study**: transitions collected in ALFWorld used
  to *induce* the verifier's precondition rules from data
  (`scripts/induce_verifier.py`); this is an analysis study, not part of the
  75-task benchmark.

**Errors / noise / redundancy?** See **Limitations**.

---

## Collection process

**How were the 38 curated tasks acquired?**
Hand-authored across 4 rooms × 3 difficulty tiers (plus cross-room edge cases),
covering single- and multi-step household goals with optional failure
injections. The raw definitions live in `SAMPLES_RAW`
(`pyplanner/apps/make_dataset.py`). Each candidate plan was then executed in
AI2-THOR; the kept reference and its outcome are recorded in `_meta`
(`gt_source = "simulator_execution"`).

**How were the 37 generated tasks acquired (`scripts/expand_dataset.py`)?**
1. **Affordance discovery** — reset the *live* AI2-THOR simulator to each scene
   and read real affordances (pickupable / openable / toggleable / receptacles)
   via `inspect_scene`.
2. **Templated candidate instantiation** — from the observed objects only,
   emit candidate tasks in three templates: easy (single interaction:
   turn-on / open / pick), medium (place object on surface), hard (open
   container, put object in, close). Surfaces and toggleables are filtered by
   commonsense whitelists/blocklists (see Limitations).
3. **Simulator-execution gate** — execute every candidate's reference plan in
   AI2-THOR and keep it **only if every interaction step returns
   `success=True`**. This is the same bar the curated 38 met.
4. **De-duplication and balanced selection** — drop descriptions already in the
   base set; select a subset that matches the base difficulty mix and spreads
   round-robin across rooms so the expansion is not dominated by object-rich
   kitchen scenes.

**Verification bar (both subsets).** A task is included only if its reference
plan executes in AI2-THOR with all interaction steps succeeding. Provenance is
written into `_meta` (`verifier: "ai2thor-5.0.0"`).

**Over what timeframe / what environment?**
AI2-THOR **5.0.0** (iTHOR), FloorPlans across the four room ranges (kitchen
1–30, living-room 200–230, bedroom 300–330, bathroom 400–430). Curated tasks
were grounded in March 2026; generated tasks carry their own `grounded_at`
timestamps.

---

## Preprocessing / cleaning / labeling

- **Action normalization**: legacy `Navigate`/`Grab` are mapped to
  `MoveTo`/`Pick` via `pyplanner.verifier.normalize_plan` before scoring or
  execution. The raw stored vocabulary is preserved in the JSON so step counts
  stay comparable to the original curation.
- **Labels**: the `reference_steps` plan, `expected_objects`, and `difficulty`
  are the labels. `expected_objects` drives the completeness metric; references
  drive execution and plan-quality metrics.
- **Provenance retained**: the `_meta` block (and `source`) is kept on every
  generated task so the dataset itself evidences how it was grounded.

---

## Uses

**Intended uses.**
- Benchmarking task planners (Direct, CoT, Few-Shot CoT, Self-Refine, ReAct,
  Hierarchical, Hierarchical Few-Shot, and SAGE/SAGE) on plan quality and on
  live AI2-THOR execution.
- Studying replanning/recovery via `fail_injection`.
- Studying generalization (ProcTHOR OOD) and longer-horizon robustness
  (COMPOUND).

**Discouraged / out-of-scope uses.**
- **Do not** use the 37 generated tasks as held-in exemplars for a
  memory/retrieval method that is also evaluated on them — they are held out for
  leak-free evaluation.
- **Do not** treat goal-condition (end-task) success as a reliable signal on
  this benchmark — see Limitations (3); report step-level execution success.
- **Do not** treat results as evidence of real-robot performance — this is
  simulation only.
- The benchmark is small (75 tasks) and is intended for controlled comparison,
  not for training large models.

---

## Distribution

The dataset ships as JSON files in the artifact (`eval_dataset_gt.json`,
`eval_dataset_expanded.json`, and the auxiliary `eval_dataset_procthor.json`).
The generator scripts are released so the data is fully reproducible (see
`REPRODUCE.md`). Recommended licensing is in `LICENSE_NOTES.md`. AI2-THOR scenes
themselves remain under AI2-THOR's license.

---

## Maintenance

- The dataset is regenerable from pinned code (`make_dataset.py`,
  `expand_dataset.py`) against AI2-THOR 5.0.0; the full verified candidate pool
  is saved (`results/expansion/verified_pool.json`) so re-selection is offline
  and deterministic.
- Adding a task: extend `SAMPLES_RAW` (curated) or rerun the generator with a
  new `--target-new`. Any new task must pass the same simulator-execution gate.
- Provenance (`_meta`) must be preserved on every record.

---

## Limitations (stated honestly)

1. **The 37 generated tasks are templated and intentionally simple** — atomic
   turn-on / pick / open, place-object-on-surface, and put-in-container-and-close.
   They broaden coverage but do not add long-horizon complexity (the COMPOUND
   set exists for that).
2. **Earlier generator versions produced a few commonsense-implausible
   place-targets** (e.g. placing an object on a garbage can or armchair). Those
   receptacles were removed from the generator's surface whitelist
   (`RECEPTACLE_SURFACES`), and toggle/pick blocklists were added; the released
   set uses the cleaned whitelists.
3. **Goal-condition (end-task) success checkers in the simulator are
   unreliable on this benchmark.** The paper treats this as an open measurement
   finding and leads with **step-level execution success** rather than
   end-task success.
4. **Simulation only.** All grounding and execution are in AI2-THOR; there is
   **no real-robot evaluation**.
5. **Small scale.** 75 tasks; the curated `obs` is a synthesized snapshot and
   the generated `obs` is terse/templated.
