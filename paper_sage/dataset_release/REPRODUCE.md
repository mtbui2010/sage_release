# Reproducing the SAGE AI2-THOR Benchmark

This document gives the exact steps to (a) regenerate and verify the dataset,
and (b) re-run the planning benchmark. All paths are relative to the
`paper_sage/` artifact directory unless noted; `pyplanner` lives at
`../pyplanner/`.

---

## 0. Environment (pin these)

- **Python**: 3.10+ (any CPython works; use a virtualenv).
- **Simulator**: `ai2thor==5.0.0` (iTHOR). The dataset's `_meta.verifier` is
  `"ai2thor-5.0.0"`; results are only comparable on this version.
- **Display**: AI2-THOR needs an X display. Use `DISPLAY=:0` (or an Xvfb
  display) for every simulator command below.
- **Packages**:

  ```bash
  pip install -e ../pyplanner/
  pip install "ai2thor==5.0.0"
  ```

  > **Note.** `carerobotagent` (the full LangGraph care-robot app that *uses*
  > pyplanner) is **not** part of this release and is **not** needed to
  > reproduce any benchmark, ablation, simulator, or human-eval result here.
  > Everything in this artifact runs against `pyplanner` alone.

- **LLM backend**: Ollama, non-streaming `/api/chat`. Default host
  `http://localhost:11434` (point `OLLAMA_HOST` at your own server to override).
  Set `OLLAMA_TIMEOUT=300` for slow serial generation.

---

## 1. Offline sanity check (no LLM, no simulator)

Always pass this before launching anything heavier:

```bash
make smoke
python -c "import pyplanner; print('SAGE' in pyplanner.REGISTRY)"   # -> True
ls -1 ../pyplanner/eval_dataset_gt.json                              # must exist
```

---

## 2. Start the AI2-THOR server

The dataset regeneration and any sim-execution run require a live server
listening on port 5555:

```bash
DISPLAY=:0 python ../pyplanner/apps/thor_server.py
```

Leave this running in its own terminal. (For ProcTHOR use the
ProcTHOR-capable server, e.g. `apps/procthor_server.py` on port 5558.)

---

## 3. Regenerate the curated 38 tasks

The curated tasks are defined in `../pyplanner/apps/make_dataset.py`
(`SAMPLES_RAW`) and grounded by simulator execution. The released, grounded
file is `../pyplanner/eval_dataset_gt.json` (`_meta.gt_source =
"simulator_execution"`, `candidate_source = "manual"`).

---

## 4. Regenerate + verify the 37 generated tasks

With `thor_server` up (step 2):

```bash
DISPLAY=:0 python scripts/expand_dataset.py \
    --sim-host localhost --sim-port 5555 \
    --target-new 40 \
    --out ../pyplanner/eval_dataset_expanded.json
```

This: (1) inspects each scene's live affordances, (2) instantiates templated
candidates, (3) **executes every candidate plan and keeps only those whose
interaction steps all return `success=True`**, (4) de-duplicates and selects a
difficulty/room-balanced subset.

Outputs:
- `results/expansion/verified_pool.json` — full verified candidate pool.
- `results/expansion/new_tasks.json` — selected verified new tasks.
- `results/expansion/report.json` — per-candidate verification report.
- `../pyplanner/eval_dataset_expanded.json` — merged **curated 38 + generated**
  (75 tasks).

Offline, deterministic re-selection from the saved pool (no simulator):

```bash
python scripts/expand_dataset.py --reselect --target-new 40 \
    --out ../pyplanner/eval_dataset_expanded.json
```

---

## 5. Auxiliary / OOD datasets (optional)

- **COMPOUND** (74 tasks; method-agnostic stress test, no LLM, references
  re-checked through the symbolic verifier):

  ```bash
  python scripts/gen_compound_tasks.py
  ```

- **ProcTHOR OOD** (70 tasks; same template-then-verify discipline on
  ProcTHOR-10k val houses; needs a ProcTHOR server on :5558):

  ```bash
  DISPLAY=:0 python scripts/build_procthor_dataset.py \
      --sim-host localhost --sim-port 5558 \
      --split val --num-houses 40 --target-new 70 \
      --out ../pyplanner/eval_dataset_procthor.json \
      --outdir results/procthor_expansion
  ```

The leave-one-out (leak-free) retrieval protocol for memory methods is enforced
at evaluation time; SAGE's seed memory stays on the curated 38, so the 37
generated tasks are held out (no leakage) when running on the 75-task file.

---

## 6. Re-run the planning benchmark

**Plan-quality grid** (baseline set is `Direct, CoT, Few-Shot CoT,
Self-Refine, ReAct, Hierarchical, Hierarchical Few-Shot, SAGE`),
over seeds 0/1/2, on the 75-task expanded set:

```bash
# One host, one or more models:
METHODS="Direct,CoT,Few-Shot CoT,Self-Refine,ReAct,Hierarchical,Hierarchical Few-Shot,SAGE" \
SEEDS="0 1 2" \
DATASET=../pyplanner/eval_dataset_expanded.json \
scripts/run_grid.sh http://localhost:11434 llama3.2 qwen2.5:7b mistral-nemo
```

Each `(model, seed)` runs serially (concurrent heavy generation starves the
host), gets its own copy of the seed live-memory under `data/grid_mem/`, and
writes `results/grid_<model>_s<seed>/results.csv`.

Convenience targets:

```bash
make seed             # build/refresh seeded memory store
make benchmark-quick  # quick sweep over a couple of methods
make benchmark MODELS="llama3.2 qwen2.5:7b mistral-nemo"   # full grid
make ablate           # ablation variants
make plots            # figures from latest results CSV
```

**Live AI2-THOR execution** (with `thor_server` up):

```bash
make sim
```

---

## 7. Human-judged end-task success

The paper reports an **end-to-end task-success** study (because the simulator
goal-checkers are lenient/unreliable) on a 72-item subset (**24 GT tasks × 3
methods** — Direct, Hierarchical, SAGE — a complete paired design; 9 kitchen /
8 living-room / 7 bedroom; difficulty 9 easy / 10 medium / 5 hard). The full
bundle is **shipped** at `results/human_eval/`:

| File | Contents |
|------|----------|
| `frames/` | 72 final-state RGB frames (one per judged item). |
| `items.jsonl` | Per-item task, expected objects, final symbolic states, plan, method. |
| `judge.html` | **Blinded** rater interface (method + auto verdict hidden; shuffled). |
| `auto_partial.csv` | Partial-credit auto-checker output (regenerable). |
| `sage_human_eval_rater1_author.csv`, `…_rater2_external.csv` | The two raters' judgments (rater1 = an author; rater2 = an external annotator; identities anonymized). |
| `reconcile.csv`, `reconcile_summary.json` | Per-method means + agreement stats (regenerable). |

**Reproduce the numbers offline (no LLM, no simulator):**

```bash
make human-eval
# or directly:
python scripts/auto_partial_score.py     # re-scores items.jsonl → auto_partial.csv
python scripts/reconcile_auto_human.py    # rater CSVs + auto → reconcile.csv + summary
```

Expected output: **SAGE 0.53 > Direct 0.40 > Hierarchical 0.28** (human mean),
inter-rater **Cohen's κ = 0.65** (weighted 0.75, exact 77.8%), and the
zero-token partial-credit checker tracking the human consensus at **Pearson
r = 0.82** (MAE 0.16). These match Table (supplement, §Human-Judged End-Task
Success) in the paper.

**Re-capture a fresh bundle from scratch** (needs `thor_server` up, §2): run
`apps/evaluate/evaluate_sim.py` with `SAGE_EVAL_BUNDLE=results/human_eval` set to
capture frames + final states, then `python scripts/make_human_eval_html.py
--bundle results/human_eval` to (re)generate the blinded `judge.html` for raters.

---

## 8. Expected outputs / directories

| Path | Contents |
|------|----------|
| `../pyplanner/eval_dataset_gt.json` | Curated 38 (grounded). |
| `../pyplanner/eval_dataset_expanded.json` | Merged 75 (curated + generated). |
| `../pyplanner/eval_dataset_procthor.json` | ProcTHOR OOD 70. |
| `results/expansion/{verified_pool,new_tasks,report}.json` | Generation/verification artifacts. |
| `results/grid_<model>_s<seed>/results.csv` | Per-run plan-quality results. |
| `results/grid_logs/stream_<host>.log` | Grid run logs. |
| `figures/` | Plots from `make plots`. |
| `results/human_eval/` | **Shipped** end-task bundle (frames + items + rater CSVs). |

`results/` (except `results/human_eval/`), `figures/`, and `data/memory.jsonl`
are reproducible and are not committed. `results/human_eval/` **is** shipped —
it is the human-judged artifact and is preserved by `make clean`.

---

## 9. Reproducibility checklist

- [x] **Data released** — JSON datasets shipped in the artifact.
- [x] **Code released** — all generators (`make_dataset.py`,
      `expand_dataset.py`, `gen_compound_tasks.py`,
      `build_procthor_dataset.py`) and the benchmark driver
      (`run_benchmark.py`, `run_grid.sh`) are included.
- [x] **Versions pinned** — `ai2thor==5.0.0`; Python 3.10+; verifier version
      recorded per task in `_meta.verifier`.
- [x] **Seeds fixed** — benchmark runs use seeds `{0, 1, 2}`.
- [x] **Deterministic re-selection** — `expand_dataset.py --reselect` rebuilds
      the dataset offline from `verified_pool.json`.
- [x] **One-command smoke** — `make smoke` (offline, no LLM, no simulator).
- [x] **No data leakage** — 37 generated tasks held out from SAGE seed memory;
      leave-one-out retrieval protocol for memory methods.
- [x] **Deterministic, LLM-free grounding** — references are simulator-verified;
      the symbolic verifier never calls an LLM.
- [x] **End-task human study shipped** — 72-item bundle (frames + rater CSVs) at
      `results/human_eval/`; `make human-eval` reproduces 0.53/0.40/0.28, κ=0.65,
      r=0.82 offline.
