# SAGE on NVIDIA Jetson Orin — Edge-Deployment Experiment Plan

**Status:** PLAN ONLY. Hardware not yet on hand. Execute when the Jetson Orin
arrives. Nothing in here has been run.

**Goal of this experiment.** Produce the evidence behind the RA-L paper's
*on-device / edge-deployment* claim: that SAGE's symbolic planning + zero-token
verifier runs on an edge SoC with the small open-weight LLMs you would actually
deploy when a cloud LLM is impractical, **with no measurable plan-quality loss
versus the server runs and with negligible verifier overhead.**

> **Scope, stated honestly.** This tests **on-device PLANNING feasibility +
> verifier overhead**, not a full embodied robot stack. The LLM planner runs
> *on* the Jetson; AI2-THOR execution (if used at all) stays on a separate host
> (planner-on-edge, sim-on-host). We are not claiming a robot ran on the Jetson.
> This is a partial mitigation of limitation **L1 (sim-only evaluation)**: it
> moves the *decision-making* component onto representative edge silicon.

---

## 1. Motivation and the claim

SAGE plans at the symbolic level with small open-weight LLMs served by Ollama,
then passes every plan through a hand-written symbolic gate/verifier
(`pyplanner/verifier.py`, ~250 LOC, pure Python, **O(|π|)**, **zero LLM tokens**).

The verifier's value is **largest exactly for small models.** The EAI
multi-seed finding (see `paper/experiments.tex`) is that the verifier+repair
loop raises step-success / executability most on **3B and 7B** models and
**saturates on 14B** — a 14B model already produces mostly-valid plans, so the
gate has little to catch; a 3B model produces many precondition violations that
the gate catches and repairs.

Those 3B/7B models are **precisely the ones you run on-device** when a cloud LLM
is unavailable, too costly, or not allowed (privacy, connectivity, latency).
And the verifier is the cheap part: it calls no model, so on an edge SoC where
every LLM token is expensive it adds **negligible compute** while supplying a
deterministic quality/safety signal. That is the on-device argument in one line:

> *The component that helps small models the most is also the one that costs
> almost nothing to run on the edge.*

This experiment quantifies (a) that small SAGE-class models are **feasible to
run on Jetson Orin**, (b) the **verifier overhead is negligible** on that
hardware, and (c) **plan quality is on par** with the server runs (no
quality regression from moving on-device).

---

## 2. Hardware matrix

All models pulled at **Q4_K_M** quantization (4-bit, good quality/size balance,
the Ollama default for these tags). "Fits" = model weights + KV cache + runtime
leave comfortable headroom on the shared CPU/GPU memory (Jetson has unified
memory, so the LLM and the OS contend for the same pool — budget for it).

| Device | Unified RAM | Target model (Q4_K_M) | Approx weights | Headroom note |
|--------|------------|------------------------|----------------|---------------|
| **Orin Nano 8 GB** | 8 GB | `qwen2.5:3b` (≈ llama3.2 3B class) | ~2.0–2.3 GB | Tight. Keep ctx small (≤4k), close desktop/GUI, run headless. ~7B is *not* recommended here. |
| **Orin NX 16 GB** | 16 GB | `qwen2.5:7b` | ~4.5–4.7 GB | Comfortable. 7B is the sweet spot. Can also host 3B for the latency-vs-quality sweep. |
| **AGX Orin 32 GB / 64 GB** | 32 / 64 GB | `qwen2.5:14b` | ~9 GB | Comfortable on 32 GB, easy on 64 GB. The "saturation" reference point on-device. |

Notes:
- **Unified memory.** Subtract ~1.5–2.5 GB for JetPack/OS/desktop before
  budgeting the model. Run the device **headless** for the benchmark.
- **Power mode matters.** Each device has selectable power modes
  (`nvpmodel`). Record the mode used (e.g. AGX Orin `MAXN`, Orin Nano `15W`).
  Report all numbers **per power mode** — latency and tok/s swing a lot.
- **Quantization.** Q4_K_M is the default; if 8 GB is too tight for 3B with the
  desired context, fall back to `q4_0` and note it. Keep the quant tag in every
  results row — quant changes outputs (see §8).

---

## 3. Software setup

The split is the key design decision, so state it up front:

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Jetson Orin (EDGE)          │         │  x86 host (SIM, optional)     │
│  ───────────────────         │         │  ───────────────────────     │
│  Ollama (ARM64 + CUDA)       │  plans  │  thor_server.py + AI2-THOR    │
│  qwen2.5:3b / 7b / 14b       │ ──────► │  evaluate_sim.py executes the │
│  run_benchmark.py --mode plan│         │  on-device plans              │
│  verifier.py (zero-token)    │         │                               │
│  tegrastats logging          │         │                               │
└─────────────────────────────┘         └──────────────────────────────┘
   planner-on-edge                          sim-on-host
```

- **JetPack:** target **JetPack 6.x** (L4T r36.x, Ubuntu 22.04, CUDA 12.x).
  Record exact version: `cat /etc/nv_tegra_release`.
- **Ollama on Jetson:** install natively (ARM64 build with CUDA support):
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
  The official installer detects Jetson/ARM64 + CUDA. Confirm GPU is used
  (not CPU) via `ollama ps` (should show a GPU/CUDA backend) and by watching
  GPU load in `tegrastats` during a generation. If the native build does not
  pick up CUDA on the installed JetPack, fall back to **llama.cpp** built with
  `-DGGML_CUDA=on` and serve via its OpenAI-compatible endpoint OR keep using
  Ollama's CPU path and **clearly label runs as CPU-only** (still a valid edge
  data point, just slower). See §8 for build caveats.
- **Serving locally:** start Ollama bound to localhost so the planner talks to
  the on-device server:
  ```bash
  OLLAMA_HOST=127.0.0.1:11434 ollama serve
  ```
- **Pull the quantized models** (only the one(s) that fit the device):
  ```bash
  ollama pull qwen2.5:3b      # Orin Nano 8GB
  ollama pull qwen2.5:7b      # Orin NX 16GB
  ollama pull qwen2.5:14b     # AGX Orin 32/64GB
  ```
- **SAGE code on the Jetson** (planning runs here):
  ```bash
  git clone <repo> && cd <repo>
  pip install -e pyplanner/        # ARM wheels exist for requests/ollama; no THOR needed on edge
  ```
  Note: AI2-THOR / Unity are **not** installed on the Jetson. Only the planner +
  verifier run there, which is the whole point of the split.
- **Point the benchmark at the on-device server** with `--host localhost:11434`.
  (`LLMBackend` normalizes a bare `localhost:11434` to `http://localhost:11434`
  — see `base.py` host-normalization; localhost → http, no TLS.)

---

## 4. What to measure (metrics)

Two buckets: **on-device cost** (new, the point of the experiment) and
**quality parity** (must match the server runs).

### On-device cost (logged on the Jetson)
- **Per-call LLM latency (s)** and **tokens/s** (decode throughput). Source:
  `latency_s`, `output_tokens` from `PlanMetrics`, plus Ollama's
  `eval_count` / `eval_duration` for a clean tok/s. Report median + p90.
- **End-to-end plan-generation time per task (s)** — wall-clock for one
  `generate_plan` call including all LLM calls and the verifier
  (`latency_s` in the results CSV).
- **Verifier overhead (ms)** — time spent in `verifier.simulate` /
  `verify_step` only. **Expectation: negligible (sub-millisecond to a few ms
  for plan lengths ~5–15 steps; O(|π|), no model).** Measure directly (§5,
  `verifier_overhead` micro-bench) and report as a fraction of end-to-end time.
- **Peak RAM (MB)** and **power draw (W)** from `tegrastats` over each run
  (RAM line + `VDD_IN` / board power rail). Report peak + mean. Tie to the
  `nvpmodel` power mode.
- **Cold-load time (s)** — first-token latency after `ollama run` loads weights
  into memory (one-time per model; the benchmark already warms up once per
  model, so capture it separately).

### Quality parity (must hold vs server)
Run the **same** subset on the server (`--host localhost:11434`, the
workspace default) and on the Jetson, **same `--seed`**, same models, same
task-ids, and confirm parity on:
- **step-success** (sim) / **precondition_strict** + **executability** (plan-mode),
- **recovery-cost** (`refines`, `verifier_rejections`, replan calls),
- **completeness**.

Parity criterion (state in paper): on-device means fall within run-to-run
seed variance of the server means (define a tolerance, e.g. |Δ| ≤ 1 seed-stddev,
from the existing multi-seed grid). **The claim is "no quality loss," so the
headline is that these match — the on-device value-add is the *cost* columns.**

---

## 5. Protocol (step-by-step)

All commands run **on the Jetson** unless marked `[HOST]`.

### 5.0 Flash / setup (once)
```bash
# Flash JetPack 6.x via SDK Manager or the SD-card image (Nano).
cat /etc/nv_tegra_release          # record L4T version
sudo nvpmodel -q                   # record current power mode
sudo nvpmodel -m 0                 # set MAXN (AGX) or chosen mode; record it
sudo jetson_clocks                 # lock clocks for repeatable timing (note it)
```

### 5.1 Install Ollama + pull models
```bash
curl -fsSL https://ollama.com/install.sh | sh
OLLAMA_HOST=127.0.0.1:11434 ollama serve &     # background; or systemd unit
ollama pull qwen2.5:7b                          # device-appropriate model(s)
ollama ps                                       # confirm CUDA/GPU backend
```

### 5.2 Install SAGE and run the offline smoke test (no LLM)
```bash
pip install -e pyplanner/
make smoke
python -c "import pyplanner; print('SAGE' in pyplanner.REGISTRY)"   # → True
```
`make smoke` must pass before any LLM run — it proves the verifier + dataset are
intact on the ARM build.

### 5.3 Pick a balanced ~20-task subset
Choose ~20 task-ids spanning difficulty (easy/medium/hard) and rooms so the
latency/quality numbers are representative, not cherry-picked. Capture the IDs
in a shell var for reuse on both edge and server:
```bash
SUBSET="t01 t02 t05 t08 t11 t14 t17 t20 t23 t26 t29 t32 t35 t38 t41 t44 t47 t50 t53 t56"
# (replace with actual task_ids from eval_dataset_gt.json; balance by 'difficulty'/'room')
```

### 5.4 On-device plan-mode run (the core measurement)
Start `tegrastats` logging, then run the benchmark against **localhost**:
```bash
# Logger: timestamped RAM + power, 200 ms cadence
tegrastats --interval 200 --logfile tegrastats_edge_qwen7b.log &
TEGRA_PID=$!

OLLAMA_TIMEOUT=600 python scripts/run_benchmark.py \
    --mode plan \
    --host localhost:11434 \
    --models qwen2.5:7b \
    --methods-csv "Direct,SAGE" \
    --task-ids $SUBSET \
    --seed 0 \
    --run-id edge_orinnx_qwen7b_s0

kill $TEGRA_PID
```
- `--methods-csv "Direct,SAGE"` isolates the verifier/repair effect: Direct is
  the bare small-model baseline; SAGE adds the zero-token gate + local repair.
- Repeat for the device's model (`qwen2.5:3b` on Nano, `qwen2.5:14b` on AGX).
- Repeat across **seeds 0,1,2** to match the server grid for parity testing.
- `OLLAMA_TIMEOUT=600` guards against slow edge generations being truncated to
  empty plans (see `_chat_ollama` timeout handling in `base.py`).

### 5.5 Verifier-overhead micro-benchmark (zero-token, on-device)
The end-to-end `latency_s` is dominated by the LLM; isolate the verifier so the
"negligible overhead" claim is measured, not asserted:
```bash
python - <<'PY'
import time, json, statistics, sys
sys.path.insert(0, "../pyplanner")
from pyplanner.verifier import simulate, normalize_plan
data = json.load(open("../pyplanner/eval_dataset_gt.json"))["samples"]
ts = []
for s in data:
    steps = normalize_plan(s.get("reference_steps") or [])
    vis = s.get("visible_objects") or []
    t0 = time.perf_counter()
    for _ in range(100):                       # amortize timer noise
        simulate(steps, visible_objects=vis, stop_on_error=False)
    ts.append((time.perf_counter() - t0) / 100 * 1000)   # ms/call
print(f"verifier ms/call: median={statistics.median(ts):.3f} "
      f"max={max(ts):.3f} over {len(ts)} plans")
PY
```
Report this median/max (ms) as the **verifier_overhead** column. Expectation:
≈ sub-ms to low-single-digit ms, i.e. << 1% of end-to-end plan time.

### 5.6 Server baseline (same subset, same seeds) `[HOST or Jetson]`
```bash
python scripts/run_benchmark.py \
    --mode plan \
    --host localhost:11434 \
    --models qwen2.5:7b \
    --methods-csv "Direct,SAGE" \
    --task-ids $SUBSET \
    --seed 0 \
    --run-id server_qwen7b_s0
```
Compare `aggregate.json` from `edge_*` vs `server_*` runs — quality columns
must match within seed variance; only the cost columns differ.

### 5.7 (Optional) sim-execute the on-device plans `[HOST]`
To close the loop on executability (planner-on-edge, sim-on-host):
1. On the host, start the simulator: `python ../pyplanner/apps/thor_server.py`.
2. Run `--mode sim` with the same model/subset. Note: in the current
   `run_benchmark.py`, `--mode sim` regenerates plans via `evaluate_sim.py`
   pointing at `--host`; to truly execute the *edge-generated* plans, either
   (a) point `--host localhost:11434` from a host that can reach the Jetson's
   Ollama over the LAN, or (b) record the edge plans and add a small replay path
   to `evaluate_sim.py`. Document whichever route is used. This step is optional
   and supports the step-success parity column, not the core latency claim.

---

## 6. Expected results table skeleton (placeholders)

Fill from `aggregate.json` (quality), the results CSV (`latency_s`, `refines`),
the §5.5 micro-bench (verifier ms), and `tegrastats` (RAM/power).

| Model | Device (power mode) | tok/s | Plan latency (s, med) | Verifier overhead (ms) | Step-success | Recovery calls | Peak RAM (GB) | Power (W) |
|-------|--------------------|-------|----------------------|------------------------|--------------|----------------|---------------|-----------|
| qwen2.5:3b  | Orin Nano 8 GB (15W)   | _TBD_ | _TBD_ | _TBD_ (<1 expected) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| qwen2.5:7b  | Orin NX 16 GB (25W)    | _TBD_ | _TBD_ | _TBD_ (<1 expected) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| qwen2.5:14b | AGX Orin 32 GB (MAXN)  | _TBD_ | _TBD_ | _TBD_ (<1 expected) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| qwen2.5:7b  | **server (A100/cloud)**| _TBD_ | _TBD_ | _TBD_               | _TBD_ | _TBD_ | n/a   | n/a |

Companion parity check (Direct vs SAGE, per device) — shows the verifier helps
the small on-device models, matching the EAI finding:

| Model / Device | Method | precondition_strict | executability | refines | verifier_rejections |
|----------------|--------|---------------------|---------------|---------|---------------------|
| qwen2.5:3b / Nano | Direct | _TBD_ | _TBD_ | 0 | 0 |
| qwen2.5:3b / Nano | SAGE  | _TBD (↑)_ | _TBD (↑)_ | _TBD_ | _TBD_ |
| qwen2.5:14b / AGX | Direct | _TBD_ | _TBD_ | 0 | 0 |
| qwen2.5:14b / AGX | SAGE  | _TBD (≈ saturates)_ | _TBD_ | _TBD_ | _TBD_ |

---

## 7. Paper integration

Add a short **"On-device deployment"** subsection to §4 (Experiments). One
paragraph + one small table (the §6 skeleton, trimmed to the columns that land).

Draft paragraph:

> **On-device deployment.** To show SAGE is deployable where a cloud LLM is
> impractical, we run the planner on an NVIDIA Jetson Orin with Ollama serving
> 4-bit (Q4_K_M) open-weight models — 3B on Orin Nano 8 GB, 7B on Orin NX 16 GB,
> 14B on AGX Orin — while AI2-THOR execution remains on a host
> (planner-on-edge, sim-on-host). On a balanced 20-task subset, on-device plan
> quality matches the server runs within seed variance (Table X), so moving to
> the edge costs no plan quality. The symbolic verifier adds <1 ms per plan
> (O(|π|), no model) — negligible against multi-second LLM latency — yet
> recovers the same executability gains for the 3B/7B models that we observe at
> scale, gains that saturate at 14B. SAGE's gate is thus a lightweight,
> cloud-free quality/safety layer for small on-device planners.

Honest scope sentence to keep in the text: *"This experiment evaluates on-device
**planning** feasibility and verifier overhead; the embodied execution stays in
simulation on a separate host, a partial mitigation of our sim-only limitation
(L1)."*

---

## 8. Risks and caveats

- **Edge LLM latency may be high.** Decode tok/s on Orin is far below a server
  GPU; 14B on AGX or 7B on Orin NX can take several to tens of seconds per plan.
  Mitigate: small context (≤4k), greedy/low-temp decode, `OLLAMA_TIMEOUT=600`,
  `jetson_clocks`. Report latency honestly per power mode — slowness is expected
  and does not undercut the *feasibility + parity* claim.
- **Thermal / power throttling.** Sustained generation heats the SoC; clocks may
  throttle and inflate latency. Use active cooling, log `tegrastats` thermals,
  and pin a power mode. Report mean and peak power; note any throttling.
- **Unified-memory pressure (esp. Nano 8 GB).** Model + KV cache + OS share one
  pool. Run headless, avoid concurrent processes, drop context size, and fall
  back to `q4_0` if Q4_K_M + ctx does not fit. If 7B truly will not fit a given
  device, do not force it — report the 3B result for that device.
- **Quantization may change outputs.** Q4 weights can produce slightly different
  plans than the server's quant/precision. This is why every on-device plan is
  **re-verified on the same verifier** and parity is checked against a server
  run **at the same quant where possible**. Record the exact model tag + quant
  in every row; if server used a different quant, note it as a confound.
- **Ollama-on-Jetson build notes.** The official ARM64 installer usually enables
  CUDA on JetPack 6.x, but on some L4T/CUDA combos it falls back to CPU. Verify
  with `ollama ps` and GPU load in `tegrastats`. If CPU-only, either (a) build
  **llama.cpp** with `-DGGML_CUDA=on` and serve its OpenAI-compatible endpoint,
  or (b) keep CPU and **label runs CPU-only** (still a valid, if slower, edge
  data point). Pin and record the Ollama version (`ollama --version`) — server
  builds may differ and that is a quant/runtime confound to disclose.
- **Reproducibility.** Same `--seed` across edge and server makes Ollama sampling
  reproducible per seed (forwarded to the backend `options.seed`), but hardware
  numerics and quant differences mean outputs are not guaranteed bit-identical;
  rely on the verifier re-check, not on string equality, for parity.
```
