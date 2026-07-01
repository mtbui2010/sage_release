# SAGE — Symbolic Action-Gating and Editing for LLM Task Planners

Reference implementation, benchmark, and reproduction code for the paper
**“SAGE: Symbolic Action-Gating and Editing for LLM Task Planners.”**

> ### 📄 Reviewers — start here
> **Full supplementary tables & analysis → [`online_appendix.md`](online_appendix.md)**
> (claims→evidence roadmap, per-model result grids, failure-recovery cost, the
> non-circular safety monitor, human-judged end-task success, on-device (Jetson)
> results, verbatim prompts — renders inline in your browser).

## Why SAGE

Headline results (qwen2.5:7b unless noted; full per-model tables in
[`online_appendix.md`](online_appendix.md)):

| Axis | Baselines | **SAGE** |
|---|---|---|
| Completeness on **hard compound tasks** | 0.78 (Hier-FS) | **0.85**  (+0.06 … +0.23 across models) |
| **Failure-recovery cost** (equal 100% recovery) | 2.9–3.7 LLM calls | **1.18**  (2.4–3.3× fewer) |
| **Runtime safety** (verify-before-execute) | — | **+0.107** step-success for Direct; **206** unsafe actions blocked |
| **Human-judged end-task success** | 0.40 / 0.28 | **0.53**  (κ = 0.65) |
| **Verifier cost on the edge** (Jetson AGX Orin) | — | **0.008 ms/plan**, zero tokens |

### How it works: gate + edit

```
                        ┌───────────────────── EDIT (on violation): ─────────────────────┐
                        │            regenerate only the failed sub-goal's suffix         │
                        ▼                                                                 │
  task ─► decompose ─► expand ─►  ┌───────────────────────────┐  pass ─► certified plan ─► execute
          (sub-goals)  (per goal) │  GATE: symbolic verifier   │                              │ fail
                        ▲         │  verify preconditions       │◄─────────── recover ◄────────┘
                        │         │  (LLM-free, 0 tokens, O(|π|))│
                        └─ retrieve└───────────────────────────┘
                           (hybrid memory)          │ violation → typed reason
                                                     └──────────────► EDIT (above)
```

The symbolic **gate** is LLM-free (zero tokens, `O(|π|)`); the sub-goal-local
**edit** regenerates only the failed sub-goal's suffix, keeping completed and
untouched future work. Results above are non-circular where applicable (the safety
monitor is scored by a simulator signal the verifier never sees). Every table is in
[`online_appendix.md`](online_appendix.md).

---

SAGE is a single-LLM embodied task planner built from two lightweight mechanisms:

1. a **rule-based symbolic precondition verifier** (~250 lines of Python, zero
   tokens, `O(|π|)`) that **gates** precondition-violating actions before
   execution and returns *typed* violation reasons;
2. **hierarchical decomposition with sub-goal-local editing** — on a violation,
   only the failed sub-goal’s suffix is regenerated, keeping completed work and
   untouched future sub-goals intact;

supported by a **hybrid memory** (curated seed plans + successful runtime
episodes). The verifier never calls an LLM.

> **Naming note.** The method is **SAGE** (Symbolic Action-Gating and Editing).
> It is registered as `pyplanner.REGISTRY["SAGE"]` and implemented by
> `SAGEPlanner` in `pyplanner/sage.py`.

---

## 1. What’s in this bundle

```
online_appendix.md         Paper supplementary material (renders inline; all tables)
pyplanner/                 The pluggable LLM-planner library
  pyplanner/               core package
    sage.py                the SAGE planner (BasePlanner subclass)
    verifier.py            SymbolicState, verify_step, simulate (no LLM)
    memory_retriever.py    hybrid seed+live retrieval (Jaccard / optional Chroma)
    base.py                STEP_SCHEMA, ROBOT_ACTIONS, LLMBackend, baselines
  eval_dataset_gt.json         38-task curated, simulator-verified benchmark
  eval_dataset_expanded.json   75-task expanded benchmark (used in the paper)
  eval_dataset_procthor.json   70-task ProcTHOR out-of-distribution set
  apps/evaluate/           simulator execution + goal_checker + human-eval capture
  apps/make_dataset.py     SAMPLES_RAW → the 38 curated tasks (provenance)
  apps/thor_server.py      AI2-THOR ZMQ server (run on a host with a display)
  apps/procthor_server.py  ProcTHOR scene server (for the OOD set)
  thor_app/sim_client.py   ZMQ client used by the evaluator
paper_sage/
  scripts/                 benchmark driver, analysis, table/figure generators
  dataset_release/         DATASHEET.md, REPRODUCE.md, EDGE_RESULTS.md,
                           JETSON_EDGE_PLAN.md, LICENSE_NOTES.md
README.md  LICENSE
```

Not included (reproducible): `results/`, `figures/`, memory caches. You regenerate
them with the commands below.

---

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e pyplanner/                 # core deps: requests (+ optional chromadb)
python -c "import pyplanner; print('SAGE' in pyplanner.REGISTRY)"   # -> True
```

AI2-THOR execution (optional, for the simulator experiments) needs a machine with
a GPU + display:

```bash
pip install ai2thor==5.0.0 pyzmq
```

Models are served via **Ollama** (install from `https://ollama.com`). The
benchmark talks to it over `--host` (default `http://localhost:11434`; point
this at a remote server if Ollama runs elsewhere). Default models in the paper are open-weight
(`llama3.2`, `qwen2.5:7b`, `mistral-nemo`, `qwen2.5:14b`, `qwen2.5:32b-instruct`).

---

## 3. Reproduce the experiments

All commands run from `paper_sage/`. Replace `--host` with your Ollama endpoint.

| Goal | Command |
|------|---------|
| **Offline sanity (no LLM)** | `python scripts/smoke_test.py` |
| **Build seed memory** | `python scripts/seed_memory.py` |
| **Plan-quality grid** | `python scripts/run_benchmark.py --mode plan --host <ollama> --dataset ../pyplanner/eval_dataset_expanded.json --models qwen2.5:7b --methods-csv "Direct,SAGE" --seed 0 --run-id demo` |
| **Leak-free protocol** | add `MEM_LEAVE_ONE_OUT=1` to the environment |
| **Compound (harder) benchmark** | `python scripts/gen_compound_tasks.py` then run the grid on the generated set |
| **Simulator execution** | start `python ../pyplanner/apps/thor_server.py` (host w/ display), then `--mode sim` |
| **Failure-recovery / safety-gate** | `run_benchmark.py … --inject-fail` / `--verify-gate` (see script `--help`) |
| **Verifier overhead micro-bench** | see `dataset_release/EDGE_RESULTS.md` §5.5 |
| **On-device (Jetson) edge run** | `dataset_release/JETSON_EDGE_PLAN.md` + `scripts/analyze_edge.py` |
| **Human-judged end-task eval** | run `apps/evaluate/evaluate_sim.py` with `SAGE_EVAL_BUNDLE=…` set, then `python scripts/make_human_eval_html.py --bundle results/human_eval` → open `judge.html`; reconcile with `python scripts/auto_partial_score.py` + `scripts/reconcile_auto_human.py` |
| **Regenerate paper tables/figures** | `python scripts/gen_tables.py` · `python scripts/gen_supplementary.py` · `python scripts/gen_figures.py` |

> ⚠️ **One heavy LLM stream per Ollama host.** Running multiple concurrent
> benchmark streams against one endpoint can starve generations into silent empty
> plans. Use one stream per host (set `OLLAMA_TIMEOUT=600`).

Full step-by-step protocol, dataset provenance, and the on-device plan are in
[`paper_sage/dataset_release/`](paper_sage/dataset_release/).

---

## 4. The benchmark / dataset

A 75-task AI2-THOR (iTHOR) household benchmark with simulator-verified reference
plans (38 curated core + 37 object-grounded, generation→sim-verify). See
[`DATASHEET.md`](paper_sage/dataset_release/DATASHEET.md) (Gebru et al. format)
for composition, collection, intended use, and limitations.

---

## 5. How to publish this artifact

1. **Pick a license** (this bundle ships MIT for our code; see
   [`LICENSE`](LICENSE) and
   [`LICENSE_NOTES.md`](paper_sage/dataset_release/LICENSE_NOTES.md) for the
   AI2-THOR / dependency terms). Confirm the dataset terms before redistributing.
2. **Create a public Git repo** (GitHub/GitLab). Push this bundle as-is; do **not**
   commit `results/`, `figures/`, or memory caches (they regenerate).
3. **Mint a DOI** for the frozen release (e.g. Zenodo ↔ GitHub release) so the
   paper can cite a permanent artifact.
4. **Anonymity (only if submitting double-blind):** strip author names/affiliations
   and self-references before pushing the review copy; the camera-ready can
   restore them.
5. **Add a citation block** (below) and link the repo + DOI from the paper.

### Citation

```bibtex
@inproceedings{sage2026,
  title     = {{SAGE}: Symbolic Action-Gating and Editing for {LLM} Task Planners},
  author    = {Anonymous},
  booktitle = {(under review)},
  year      = {2026}
}
```
