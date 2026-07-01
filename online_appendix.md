# Online Appendix — SAGE: Symbolic Action-Gating and Editing for LLM Task Planners

Supplementary material for the RA-L submission (anonymized for double-anonymous
review). All numbers match the main paper. This appendix is the referenced
"online appendix"; the paper is self-contained, and this document provides the
detailed per-model tables and analysis.

---

## Roadmap: claims → evidence

| Claim | Comparison & metric | Evidence (this appendix) |
|---|---|---|
| **C1** — better plans where the benchmark discriminates | vs. 7 baselines; goal-completeness, plan validity/length | §4 (full grid), §5 (compound tasks); context in §3 |
| **C2** — higher step-level execution success | vs. Direct/Hierarchical, gate on all; step-success, precondition-violation count, human end-task | §6 (grounded exec.), §11 (safety monitor), §12 (metric validity), §14 (human-judged) |
| **C3** — cheaper recovery | vs. whole-plan replanners; LLM calls, recovery rate, suffix length | §7 (recovery cost) |
| *Generality* (supports all) | OOD, rule induction, edge, published baselines | §8 (ProcTHOR), §9 (portability), §10 (SOTA), §15 (on-device) |

---

## 1. Benchmark details

A 75-task AI2-THOR (iTHOR) household set spanning four room types (22 kitchen, 18
living-room, 15 bedroom, 20 bathroom), stratified by horizon into 28 easy / 32
medium / 15 hard. Built in two stages: a 38-task curated core
(`eval_dataset_gt`), then 37 object-grounded generated tasks — each instantiated
against a scene's concrete object inventory and **simulator-verified** by
executing a reference plan in AI2-THOR (kept only if goal conditions are
reachable). For generalization we additionally use 70 tasks from 40 ProcTHOR
validation houses and the ALFWorld validation games (distinct action vocabulary).

## 2. Data leakage and the leave-one-out (LOO) protocol

The three memory methods (Few-Shot CoT, Hierarchical Few-Shot, SAGE) retrieve
few-shot exemplars from a seed memory that is exactly the 38 GT tasks — a subset
of the 75-task test set. A naive run therefore lets a memory method retrieve the
exact reference plan for the task being scored. We added an opt-in **leave-one-out**
mode (`MEM_LEAVE_ONE_OUT=1`) that drops any exact-task match before answering, and
re-ran the memory methods under LOO. **All plan-quality numbers use LOO.**

**The base-set result is a tie, and we say so.** Under LOO, base-set goal
completeness *saturates*: pooled over the three completed models, **52% (118/225)**
of task-instances are already scored 1.0 by every strong hierarchical baseline, so
the aggregate is a tie — SAGE sits marginally behind Hierarchical Few-Shot on every
base model (e.g. 0.870 vs. 0.878 on Qwen2.5-7B), all within seed variance. Where
headroom remains (hard tier), SAGE's edge survives LOO (hard completeness **0.811
vs. 0.755**, +0.056).

## 3. Full plan-quality grid (leave-one-out)

Per model: **Compl.** = goal completeness, **P-str** = precondition_strict
(diagnostic only), **Exec** = executability, **Tok** = mean total tokens, **Calls**
= mean LLM calls. `--` = not run for that model. **SAGE is our planner.**

### Llama-3.2 3B
| Method | Compl. | P-str | Exec | Tok | Calls |
|---|---|---|---|---|---|
| Direct | 0.530 | 0.511 | 0.582 | 553 | 1.0 |
| CoT | 0.806 | 0.880 | 0.972 | 1355 | 1.0 |
| Few-Shot CoT | 0.571 | 0.593 | 0.987 | 451 | 1.0 |
| Self-Refine | 0.831 | 0.846 | 0.973 | 3753 | 3.7 |
| ReAct | 0.462 | 0.666 | 0.732 | 1132 | 2.9 |
| Hierarchical | 0.760 | 0.873 | 0.925 | 3527 | 4.4 |
| Hierarchical Few-Shot | 0.821 | 0.877 | 0.933 | 7455 | 4.4 |
| **SAGE (ours)** | **0.816** | **0.931** | **0.973** | **8553** | **5.3** |

### Qwen2.5 7B
| Method | Compl. | P-str | Exec | Tok | Calls |
|---|---|---|---|---|---|
| Direct | 0.740 | 0.798 | 0.992 | 976 | 1.0 |
| CoT | 0.832 | 0.903 | 0.996 | 1142 | 1.0 |
| Few-Shot CoT | 0.622 | 0.571 | 0.953 | 424 | 1.0 |
| Self-Refine | 0.824 | 0.902 | 0.992 | 2831 | 3.8 |
| ReAct | 0.733 | 0.669 | 0.946 | 2291 | 5.7 |
| Hierarchical | 0.839 | 0.889 | 0.995 | 3001 | 3.9 |
| Hierarchical Few-Shot | 0.878 | 0.928 | 0.999 | 7075 | 4.9 |
| **SAGE (ours)** | **0.870** | **0.966** | **0.996** | **6939** | **5.6** |

### Mistral-Nemo 12B
| Method | Compl. | P-str | Exec | Tok | Calls |
|---|---|---|---|---|---|
| Direct | 0.687 | 0.677 | 0.812 | 804 | 1.0 |
| CoT | 0.839 | 0.887 | 0.979 | 1156 | 1.0 |
| Few-Shot CoT | 0.653 | 0.608 | 0.967 | 414 | 1.0 |
| Self-Refine | 0.864 | 0.920 | 0.975 | 2788 | 3.6 |
| ReAct | 0.000 | 0.004 | 0.004 | 365 | 1.0 |
| Hierarchical | 0.880 | 0.889 | 0.966 | 3714 | 4.6 |
| Hierarchical Few-Shot | 0.858 | 0.927 | 0.984 | 7000 | 4.8 |
| **SAGE (ours)** | **0.836** | **0.923** | **0.932** | **6902** | **5.5** |

### Qwen2.5 14B
| Method | Compl. | P-str | Exec | Tok | Calls |
|---|---|---|---|---|---|
| Direct | 0.861 | 0.917 | 0.984 | 998 | 1.0 |
| CoT | 0.886 | 0.920 | 0.979 | 1173 | 1.0 |
| Few-Shot CoT | 0.603 | 0.496 | 0.880 | 303 | 1.0 |
| Self-Refine | 0.839 | 0.917 | 0.969 | 1898 | 2.6 |
| ReAct | 0.759 | 0.718 | 0.997 | 4302 | 9.6 |
| Hierarchical | 0.858 | 0.913 | 0.967 | 3072 | 4.0 |
| Hierarchical Few-Shot | 0.896 | 0.920 | 0.990 | 6951 | 4.9 |
| **SAGE (ours)** | **0.883** | **0.961** | **0.998** | **6321** | **5.1** |

### Qwen2.5 32B
| Method | Compl. | P-str | Exec | Tok | Calls |
|---|---|---|---|---|---|
| Direct | 0.852 | 0.931 | 1.000 | 983 | 1.0 |
| CoT | 0.859 | 0.934 | 1.000 | 1180 | 1.0 |
| Self-Refine | 0.832 | 0.914 | 0.996 | 2659 | 3.5 |
| ReAct | 0.753 | 0.685 | 0.999 | 3356 | 7.6 |
| Hierarchical | 0.876 | 0.883 | 1.000 | 3322 | 4.3 |

*(Few-Shot CoT, Hierarchical Few-Shot, and SAGE were not run at 32B.)*

### Completeness by difficulty (pooled over models)
| Method | Easy | Medium | Hard |
|---|---|---|---|
| Direct | 0.775 | 0.711 | 0.707 |
| CoT | 0.881 | 0.843 | 0.778 |
| Few-Shot CoT | 0.616 | 0.621 | 0.587 |
| Self-Refine | 0.870 | 0.839 | 0.776 |
| ReAct | 0.624 | 0.483 | 0.511 |
| Hierarchical | 0.892 | 0.840 | 0.755 |
| Hierarchical Few-Shot | 0.902 | 0.885 | 0.743 |
| **SAGE (ours)** | **0.835** | **0.884** | **0.811** |

SAGE leads where headroom remains (medium/hard); easy is saturated for most methods.

## 4. Harder benchmark: multi-goal compound tasks

Composed *method-agnostically* from same-scene single-goal tasks into 2- and
3-goal compound tasks (74 tasks; mean expected-object count 3.7 vs. 2.0 in the
base set); each reference plan concatenates already simulator-verified sub-plans.
Once the ceiling is removed, SAGE's completeness lead re-emerges — large and
consistent, and it *grows* as the base model weakens — with shorter, more valid
plans. Δ = SAGE − Hier-FS.

| Model | Direct | Hier-FS | **SAGE** | Δ | SAGE steps | SAGE p-valid |
|---|---|---|---|---|---|---|
| qwen2.5:7b | 0.718 | 0.776 | **0.854** | **+0.078** | 12.0 | 0.943 |
| qwen2.5:14b | 0.757 | 0.783 | **0.902** | **+0.119** | 12.0 | 0.937 |
| llama3.2 | 0.629 | 0.499 | **0.728** | **+0.229** | 9.0 | 0.933 |
| mistral-nemo | 0.570 | 0.824 | **0.880** | **+0.057** | 12.7 | 0.952 |

Pooled advantage **+0.121** (95% CI [+0.092, +0.150]); significant on all four
models under Holm–Bonferroni.

## 5. Grounded execution

Primary execution metric: **step-level success** (`exec_step_success`) — the
fraction of a plan's steps AI2-THOR accepts without an environment error, over the
38 GT tasks. Absolute goal-level *task* success is not reliably measurable here
(see §12), so we lead with step-level success + the failure analysis.

| Model | Direct | Hierarchical | SAGE |
|---|---|---|---|
| Llama-3.2 3B | 0.565 | **0.613** | — |
| Qwen2.5 7B | **0.591** | 0.585 | — |
| Mistral-Nemo 12B | 0.547 | **0.622** | — |
| Qwen2.5 14B | 0.597 | **0.672** | — |
| Qwen2.5 32B | **0.708** | 0.652 | — |
| **Pooled** | 0.602 | **0.629** | — |

Pooled over five models, SAGE reaches **0.710** step success (main paper) vs. 0.602
(Direct) and 0.629 (Hierarchical).

## 6. Failure analysis and recovery cost

**Failure taxonomy** (sim, recovery enabled; counts pooled over five models, 375
executions per method). Precondition-violation failures fall to **1** for SAGE vs.
**17** for Direct (≈17× reduction).

| Failure category | Direct | Hierarchical | SAGE |
|---|---|---|---|
| Precondition violation | 17 | 7 | 1 |
| Incomplete plan (goal not met) | 20 | 11 | 5 |
| Plan-construction error | 0 | 5 | 1 |
| Sim/infra error | 0 | 1 | 0 |
| **Total failed (of 375)** | 37 | 24 | 7 |

**Recovery is measured as cost, not rate** (every method recovers 100% of injected
failures at this difficulty). One mid-execution failure injected per task
(qwen2.5:7b, LOO, `max_replan=3`).

| Method | n | Recovery | Latency (s) | LLM calls | vs. SAGE |
|---|---|---|---|---|---|
| **SAGE (ours)** | 33 | **1.00** | **1.13** | **1.18** | **1.0×** |
| Hierarchical Few-Shot | 27 | 1.00 | 1.40 | 2.96 | 2.5× |
| Hierarchical | 31 | 1.00 | 1.18 | 2.87 | 2.4× |
| Self-Refine | 27 | 1.00 | 2.98 | 3.59 | 3.0× |
| ReAct | 31 | 1.00 | 1.69 | 3.71 | 3.1× |

On the harder compound tasks the gap widens to **2.6–3.3× fewer LLM calls**;
combined across regimes, **2.4–3.3× fewer recovery LLM calls at equal recovery**.

## 7. Generalization to ProcTHOR

On 70 tasks from unseen ProcTHOR validation houses, with *no* re-tuning, SAGE
retains the highest strict-precondition satisfaction (**0.985**) of all methods —
plan *validity* transfers to unseen layouts (completeness saturates near 1.0 for
hierarchical methods on these templated rooms, so this speaks to validity, not a
completeness win).

## 8. Auto-verifier transfer (portability study)

The deployed verifier is hand-written (~250 lines); we separately show its rules
are *inducible* from `(pre-state, action, success)` transitions with **zero manual
rules**. The same pipeline runs on three ontologies.

| Action | iTHOR induced precond. | Acc | ProcTHOR induced precond. | Acc |
|---|---|---|---|---|
| Pick | found ∧ ¬holding ∧ pickupable | 1.00 | found ∧ pickupable | 0.63 |
| Open | found ∧ openable | 1.00 | found ∧ openable | 0.77 |
| Close | found ∧ openable | 1.00 | found ∧ openable | 0.77 |
| TurnOn | found ∧ toggleable | 0.83 | found ∧ toggleable ∧ ¬receptacle | 0.72 |
| TurnOff | found ∧ toggleable | 0.99 | found ∧ toggleable ∧ ¬receptacle | 0.73 |
| Place | holding ∧ ¬toggleable ∧ receptacle | 0.66 | holding ∧ ¬toggleable ∧ receptacle | 0.64 |
| PutIn | holding ∧ receptacle | 0.54 | holding ∧ receptacle | 0.46 |
| **Match vs. ref** | **14/16** | | **13/16** | |
| Held-out pred-acc | 0.859 | | 0.683 | |
| #transitions | 702 | | 723 | |

**ALFWorld** (held-out, distinct vocabulary): `open`: ¬recep-open ∧ openable
(0.604); `close`: recep-open (1.000); `put`: ¬recep-open ∧ holding ∧ in-inv ∧
¬openable (0.946); `take`: ¬holding ∧ ¬in-inv (1.000); **overall held-out
pred-acc 0.888**.

**Robustness to rule quality.** Failure-detection F1 on the 702 grounded iTHOR
transitions as the rule set is degraded (random subsets, 200 samples/level) and
when replaced by the auto-induced set. Degradation is smooth; the non-expert
induced set matches the hand-written one.

| Rule set | fail-recall | F1 |
|---|---|---|
| 25% of rules (random) | 0.20 | 0.32 |
| 50% of rules (random) | 0.39 | 0.53 |
| 75% of rules (random) | 0.63 | 0.73 |
| Full hand-written (13) | 0.79 | 0.82 |
| Auto-induced (no expert) | 0.78 | **0.88** |
| Hand rules → unseen ProcTHOR | 0.69 | 0.77 |

## 9. SOTA baselines

Against LLM+P (LLM→PDDL, solved by `pyperplan`) and SayCan (affordance-filtered
decoding), on the same LOO base grid. SayCan's P-strict = 1.0 is *by construction*
(it uses the symbolic verifier as its affordance filter); its weakness is
completeness.

| Method | Metric | Qwen2.5 7B | Qwen2.5 14B | Qwen2.5 32B |
|---|---|---|---|---|
| LLM+P | Compl. | 0.770 | 0.713 | 0.638 |
| LLM+P | P-strict | 0.963 | 0.965 | 0.976 |
| SayCan | Compl. | 0.620 | 0.607 | 0.620 |
| SayCan | P-strict | 1.000 | 1.000 | 1.000 |
| **SAGE** | Compl. | **0.870** | **0.883** | — |
| **SAGE** | P-strict | **0.966** | **0.961** | — |

## 10. Runtime safety monitor (non-circular)

Metric: `exec_step_success`, which is *simulator-reported* and never seen by the
verifier — so an improvement under the gate cannot be a circular artifact. The gate
runs online (verify-before-execute): flagged steps are repaired pre-emptively and
never executed. Enabling the gate raises step-success for *every* planner, with
per-planner deltas scaling with planner weakness: **Direct +0.107**, CoT +0.081,
Hierarchical +0.041, SAGE +0.016. Pooled over Direct and Hierarchical, the gate
prevents **206** precondition-violating actions before actuation.

| Model | Prevented actions | Repairs fired | step (gate-off) | step (gate-on) |
|---|---|---|---|---|
| Llama-3.2 3B | 54 | 26 | 0.589 | 0.605 |
| Qwen2.5 7B | 59 | 37 | 0.588 | 0.664 |
| Mistral-Nemo 12B | 52 | 30 | 0.585 | 0.628 |
| Qwen2.5 14B | 41 | 25 | 0.634 | 0.635 |
| **Total prevented** | **206** | | | |

*(Single-seed on qwen2.5:7b for per-planner deltas; part of the gain is preemptive
repair rather than pure blocking, so we frame it as "monitor + local repair.")*

## 11. Simulator setup and metric validity (a negative-result contribution)

Execution runs use AI2-THOR over a ZMQ socket (`thor_server`); `step_success` is
the fraction of emitted steps the simulator accepts. We are explicit about why we
report step-level, not goal-level, success: **both automated task-success checkers
we implemented are unreliable.** (i) A reward-based proxy (`reward>0`) gives a
falsely high ~94% ceiling (AI2-THOR emits positive shaping reward for many partial
interactions). (ii) A deterministic goal-condition checker is far too strict —
scoring only ~8% (3/38) of the *ground-truth reference plans* as successful. With
one checker too lenient and the other too strict, neither yields a trustworthy
absolute number; we therefore lead with `exec_step_success` and the human study
(§14), and flag absolute task success as an open measurement problem.

## 12. External validation: EAI VirtualHome

Scored by EAI's *own* transition model (not ours), 305 tasks × 3 seeds; cells are
verifier **on/off** (%). The verifier removes redundant steps on every model and
lifts executability for the small models while saturating on 14B; correctness is
essentially unchanged (the gate improves plan *validity*, not goal grounding).

| Model | Exec. | Correct | Afford. err | Redund. |
|---|---|---|---|---|
| qwen2.5:7b | 58.6 / 55.0 | 23.0 / 19.8 | 30.1 / 31.0 | 10.2 / 12.9 |
| llama3.2 (3B) | 37.3 / 33.6 | 8.3 / 8.2 | 51.7 / 53.7 | 9.5 / 13.9 |
| qwen2.5:14b | 70.4 / 71.9 | 33.3 / 33.8 | 17.1 / 16.8 | 9.2 / 16.6 |

## 13. Human-judged end-task success

Because both in-simulator goal checkers are unreliable (§11), we measure end-task
success by human judgement on a **blinded, method-shuffled** subset: {Direct,
Hierarchical, SAGE} on **24 GT tasks** (9 easy / 10 medium / 5 hard; 9 kitchen / 8
living-room / 7 bedroom), qwen2.5:7b, 72 items total. **Two annotators (one author,
one non-author)** independently rate each item on {0, 0.5, 1} from the final frame,
final state, and plan, with method labels hidden and item order shuffled.

| Method | Human end-task | Auto (partial) | n |
|---|---|---|---|
| Direct | 0.40 | 0.41 | 24 |
| Hierarchical | 0.28 | 0.35 | 24 |
| **SAGE (ours)** | **0.53** | **0.58** | 24 |

Inter-rater agreement: 77.8% exact, Cohen's κ = **0.65** (weighted 0.75). The
zero-token partial-credit verifier score tracks the human consensus at Pearson
**r = 0.82** (MAE 0.16), better than the binary verdict (r = 0.75) — supporting the
partial-credit checker as a cheap, non-circular proxy.

## 14. On-device (edge) deployment

Jetson AGX Orin (64 GB, MAXN, `jetson_clocks`); Ollama 0.30.10 (ARM64+CUDA),
Q4_K_M; balanced 20-task subset, seed 0; planner-on-edge, sim-on-host. On-device
cost and edge/server quality parity:

| Model | Method | tok/s | Lat. (s) | Calls | compl. (edge/server) | exec. | p-strict |
|---|---|---|---|---|---|---|---|
| qwen2.5:3b | Direct | 41 | 2.3 | 1.0 | 0.78 / 0.71 | 0.96 / 0.94 | 0.97 / 0.94 |
| qwen2.5:3b | **SAGE** | 33 | 8.3 | 7.3 | **0.90 / 0.88** | 0.99 / 0.99 | 0.96 / 0.96 |
| qwen2.5:7b | Direct | 23 | 3.9 | 1.0 | 0.76 / 0.78 | 0.98 / 0.99 | 0.88 / 0.88 |
| qwen2.5:7b | **SAGE** | 17 | 9.6 | 6.2 | **0.92 / 0.92** | 0.99 / 0.99 | 0.98 / 0.98 |
| qwen2.5:14b | Direct | 13 | 7.0 | 1.0 | 0.81 / 0.79 | 0.97 / 0.97 | 0.96 / 0.95 |
| qwen2.5:14b | **SAGE** | 10 | 20.1 | 5.5 | **0.94 / 0.94** | 0.99 / 0.99 | 0.97 / 0.98 |

**The gate is free on the edge:** the symbolic verifier's per-plan overhead
(on-device micro-benchmark, 200 reps/plan over all 75 reference plans) is **0.008 ms
median, 0.017 ms max** — six orders of magnitude below the multi-second on-device
LLM latency. On-device SAGE quality matches the server within seed variance
(|Δ| ≤ 0.02), so moving decision-making to the edge incurs no quality regression.

## 15. Prompt templates

SAGE issues four LLM roles, each with a fixed system prompt. Placeholders
`{ACTIONS_STR}`, `{STEP_SCHEMA}`, `{JSON_EXAMPLE}` are filled at call time; the
user message carries the task, scene observation, visible objects, top-k retrieved
exemplars, and (for refine/repair) the typed verifier violation block.

**(1) Decompose — high-level:**
```
You are a household task planner (high-level).
Study the retrieved examples to understand sub-goal granularity, then
break the given task into 2-5 ordered sub-goals (short natural-language
phrases). Each sub-goal should be a distinct phase the robot will
execute end-to-end before moving on.

Return ONLY valid JSON -- no markdown, no explanation:
{"subgoals": ["sub-goal 1", "sub-goal 2", ...]}
```

**(2) Expand — low-level sub-goal expansion:**
```
You are a household assistant robot (low-level executor).
You will receive ONE sub-goal and must expand it into concrete robot
action steps. Stay within this sub-goal -- do not anticipate later ones.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

Return ONLY valid JSON:
{JSON_EXAMPLE}
```

**(3) Refine — in-block precondition repair (verifier-gated):**
```
You are repairing a household plan.  The previous attempt violated
preconditions defined by the action schema.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

Return ONLY valid JSON with the corrected steps for this sub-goal:
{JSON_EXAMPLE}
```

**(4) Repair — suffix-only edit of the failed sub-goal:**
```
You are repairing a failed sub-goal in a household plan.
Generate ONLY the remaining steps to finish the CURRENT sub-goal.
Do NOT re-execute already-completed steps. Do NOT plan future sub-goals.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

Return ONLY valid JSON:
{JSON_EXAMPLE}
```

## 16. Hyperparameters, statistics, reproducibility

- **Models:** five open-weight via Ollama — llama3.2 (3B), qwen2.5:7b,
  mistral-nemo (12B), qwen2.5:14b, qwen2.5:32b-instruct. 3 seeds for the three
  smaller models; single seed for the two largest. Native default temperature; a
  warm-up call precedes each run.
- **Planner:** `max_refines=1` (a single local repair per failed sub-goal); the
  symbolic verifier makes zero LLM calls and runs in O(|π|). Memory uses LOO for
  all reported plan-quality numbers.
- **Statistics:** stdlib only (no SciPy). Paired sign-flip permutation test
  (2×10⁴ permutations, two-sided, seeds averaged before pairing); task-level
  bootstrap 95% CIs (10⁴ resamples); RNGs seeded.
- **Released artifacts:** the 75-task benchmark + simulator-verified reference
  plans, the compound-task generator, the LOO retrieval mode, the `--inject-fail`
  recovery harness, the `--verify-gate` runtime-monitor harness, and the
  verifier-portability induction pipeline (iTHOR/ProcTHOR/ALFWorld).

### Limitations
- **Simulation only.** All embodied execution is in AI2-THOR / ProcTHOR /
  ALFWorld; real-robot deployment is future work. The *planner* does run on edge
  hardware (Jetson AGX Orin, §14).
- **ALFWorld is verifier-only.** The portability result stands (0.89 held-out), but
  running SAGE's *planner* end-to-end on ALFWorld is future work (its
  clean/heat/cool vocabulary is outside our action set — a planning gap, not a
  verifier gap).
- **Saturated completeness** on the base set (honest tie); the advantage is shown
  on the harder compound benchmark (§4).
- **Recovery rate saturates** (100% for all), so we report recovery *cost* (§6).
- **Single-seed largest models** (14B/32B); the per-planner safety-gate deltas are
  single-seed on qwen2.5:7b.
- **Diagnostic-only `precondition_strict`** (circular; computed by the same
  verifier SAGE filters with). The independent validity evidence is the
  non-circular safety-monitor result (§10).
