#!/usr/bin/env bash
# sim_host.sh LLM_HOST SIM_PORT MODEL [MODEL ...]
# Run the AI2-THOR sim for a SET of models on ONE LLM host, ONE MODEL AT A
# TIME (mono-model: concurrent *different* models thrash a shared-GPU Ollama).
# LLM generation -> LLM_HOST ; AI2-THOR rendering -> local thor_server SIM_PORT.
set -u
LLM="${1:?usage: sim_host.sh LLM_HOST SIM_PORT MODEL...}"; shift
PORT="${1:?need sim port}"; shift
MODELS=("$@")
ROOT=/path/to/paper_sage
PYP=/path/to/pyplanner
DS=$PYP/eval_dataset_expanded.json
mkdir -p "$ROOT/results/sim" "$ROOT/results/sim_logs"
export PYTHONPATH="$PYP:$PYP/apps" OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-300}"
for M in "${MODELS[@]}"; do
  SAFE=$(echo "$M" | tr ':' '_')
  echo "=== $(date '+%F %T') SIM start model=$M host=$LLM port=$PORT ==="
  python -u "$PYP/apps/evaluate/evaluate_sim.py" \
    --dataset "$DS" --methods Direct Hierarchical SAGE \
    --host "$LLM" --model "$M" \
    --sim-host localhost --sim-port "$PORT" --max-replan 2 \
    --out "$ROOT/results/sim/sim_${SAFE}.csv" \
    >> "$ROOT/results/sim_logs/sim_${SAFE}.log" 2>&1
  echo "=== $(date '+%F %T') SIM done model=$M (exit $?) ==="
done
