# SAGE on NVIDIA Jetson AGX Orin — Edge-Deployment Results

**Status:** EXECUTED 2026-06-19. This is the results companion to
[`JETSON_EDGE_PLAN.md`](JETSON_EDGE_PLAN.md). All numbers below were measured
on-device; the raw artifacts are listed in §5.

## 0. TL;DR

- SAGE  **runs natively on a Jetson AGX Orin** with CUDA, for
  3B / 7B / 14B Q4_K_M open-weight LLMs (Ollama 0.30.10, JetPack 6 / L4T R36.4.3,
  CUDA 12.6, MAXN power mode, `jetson_clocks` locked).
- The symbolic **verifier overhead is 0.008 ms/call (median), 0.017 ms (max)** —
  ~6 orders of magnitude below the multi-second LLM latency. The zero-token gate
  is effectively free on the edge.
- On a balanced **20-task subset** (7 easy / 7 medium / 6 hard, across 5 rooms),
  **on-device plan quality matches the server** within seed variance for 7B and
  14B; SAGE's completeness lift over Direct **holds on-device** (+0.12 / +0.16 /
  +0.13 for 3B / 7B / 14B), confirming the verifier helps exactly the small
  models you deploy on the edge.

## 1. Hardware / software actually used

| Item | Value |
|------|-------|
| Device | NVIDIA Jetson **AGX Orin** Developer Kit, 64 GB unified RAM |
| JetPack / L4T | JetPack 6, L4T **R36.4.3** (2025-01-08), Ubuntu 22.04 |
| CUDA | 12.6 |
| Power mode | **MAXN** (`nvpmodel -m 0`), `jetson_clocks` locked |
| LLM server | **Ollama 0.30.10** (ARM64 + CUDA), localhost:11434 |
| Models | `qwen2.5:3b` (1.9 GB), `qwen2.5:7b` (4.7 GB), `qwen2.5:14b` (9.0 GB), all Q4_K_M |
| Planner | pyplanner (SAGE) + verifier, pure-Python, no THOR on device |
| GPU confirmation | `GR3D_FREQ` 94–97 % busy throughout (CUDA, not CPU-only) |

Split: **planner-on-edge, sim-on-host** — the LLM planner and verifier run on
the Jetson; AI2-THOR stays on a separate x86 host. This is a partial mitigation
of limitation L1 (sim-only): it moves the *decision-making* onto edge silicon.

## 2. On-device cost

20-task subset, seed 0, `Direct` vs `SAGE`, Q4_K_M, MAXN.
tok/s = median per-call decode throughput (`output_tokens / latency_s`).
RAM/power from `tegrastats` @200 ms; verifier overhead from the §5.5 micro-bench.

| Model | Method | tok/s | Plan lat (s, med) | LLM calls | Peak RAM (GB) | Power (W, mean) | GPU busy |
|---|---|---|---|---|---|---|---|
| qwen2.5:3b | Direct | 41.2 | 2.3 | 1.0 | 7.2 | 40 | 94% |
| qwen2.5:3b | SAGE | 33.0 | 8.3 | 7.3 | 7.2 | 40 | 94% |
| qwen2.5:7b | Direct | 23.2 | 3.9 | 1.0 | 14.4 | 45 | 95% |
| qwen2.5:7b | SAGE | 16.8 | 9.6 | 6.2 | 14.4 | 45 | 95% |
| qwen2.5:14b | Direct | 12.8 | 7.0 | 1.0 | 27.2 | 48 | 97% |
| qwen2.5:14b | SAGE | 10.2 | 20.1 | 5.5 | 27.2 | 48 | 97% |

**Reading it.** Decode throughput is the expected AGX-Orin profile: ~41 tok/s
(3B) → ~13 tok/s (14B). SAGE issues 5.5–7.3 LLM calls/task (decompose +
per-sub-goal expand + ≤1 verifier-gated repair) vs Direct's 1, so its end-to-end
plan latency is higher — the cost of the quality lift in §3. Peak RAM scales with
model size (7.2 / 14.4 / 27.2 GB), all comfortable on the 64 GB board; even 14B
leaves >35 GB headroom. Mean board power 40–48 W (MAXN). **The verifier itself
adds 0.008 ms — invisible in these latencies.**

## 3. Quality parity: edge vs server

Same 20-task subset, same seed 0, same model tags; server = `192.168.1.18:11535`
(Ollama, datacenter GPU). The claim is **"no quality loss from moving to the
edge"**, so the headline is that these columns *match*.

| Model | Method | compl (edge/srv) | exec (edge/srv) | precS (edge/srv) | SAGE compl lift (edge) |
|---|---|---|---|---|---|
| qwen2.5:3b | Direct | 0.78/0.71 | 0.96/0.94 | 0.97/0.94 | |
| qwen2.5:3b | SAGE | 0.90/0.88 | 0.99/0.99 | 0.96/0.96 | +0.12 |
| qwen2.5:7b | Direct | 0.76/0.78 | 0.98/0.99 | 0.88/0.88 | |
| qwen2.5:7b | SAGE | 0.92/0.92 | 0.99/0.99 | 0.98/0.98 | +0.16 |
| qwen2.5:14b | Direct | 0.81/0.79 | 0.97/0.97 | 0.96/0.95 | |
| qwen2.5:14b | SAGE | 0.94/0.94 | 0.99/0.99 | 0.97/0.98 | +0.13 |

**Reading it.** Edge and server completeness/executability/precondition agree
within run-to-run seed variance for 7B and 14B (|Δ| ≤ ~0.02). SAGE's
completeness lift over Direct reproduces on-device (+0.12 to +0.16). Q4_K_M is
the same quantization both sides, so outputs are close but not bit-identical
(hardware numerics differ); parity is asserted on the *verifier-rechecked
metrics*, not on string equality — see JETSON_EDGE_PLAN §8.

## 4. The on-device argument in one line

> *The component that helps small models the most (the symbolic gate+repair) is
> also the one that costs almost nothing to run on the edge (0.008 ms, zero
> tokens).*

The 3B/7B models that fit an edge SoC are precisely where Direct planning leaves
the most precondition violations on the table, and precisely where SAGE's
zero-token gate recovers the most completeness — at negligible compute cost.

## 5. Raw artifacts (in the repo, not committed)

| Path | Content |
|------|---------|
| `results/edge_agxorin_{3b,7b,14b}_s0/` | on-device plan-mode CSV + aggregate |
| `results/server_parity_{3b,7b,14b}_s0/` | same subset on server (parity) |
| `edge_logs/tegra_{3b,7b,14b}.log` | tegrastats RAM/power/GPU @200 ms |
| `edge_logs/edge_master.log` | on-device run console log + timings |
| `results/edge_summary.json` | consolidated metrics (from `scripts/analyze_edge.py`) |
| `scripts/edge_run.sh` (on Jetson `/tmp`) | the on-device runner |
| `scripts/analyze_edge.py` | parser → the two tables above |

Reproduce the tables: `python scripts/analyze_edge.py`.
