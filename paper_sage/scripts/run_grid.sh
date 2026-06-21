#!/usr/bin/env bash
# run_grid.sh HOST MODEL [MODEL ...]
# ------------------------------------------------------------------
# Serial plan-quality grid for a SET of models on ONE Ollama host.
# Run two instances (one per host) for reliable 2-way parallelism
# without saturating a single host:
#   run_grid.sh http://localhost:11434        llama3.2 mistral-nemo  &
#   run_grid.sh http://ollama.aistations.org  qwen2.5:7b            &
# Each (model, seed) run:
#   * is SERIAL (one method/task at a time) so the host is never
#     saturated — concurrent heavy generation was found to starve and
#     time out (empty plans);
#   * gets its OWN copy of the seed live-memory under data/grid_mem/;
#   * runs all 9 methods × 38 tasks with a per-model warmup;
#   * writes results/grid_<model>_s<seed>/results.csv.
set -u
HOST="${1:?usage: run_grid.sh HOST MODEL [MODEL ...]}"; shift
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || { echo "no models given"; exit 1; }
# SAGE-Fixed dropped: empirically identical to SAGE (0/450 cells differ), so
# running both wastes the most expensive method. Paper reports one combined SAGE.
METHODS="${METHODS:-Direct,CoT,Few-Shot CoT,Self-Refine,ReAct,Hierarchical,Hierarchical Few-Shot,SAGE}"
SEEDS="${SEEDS:-0 1 2}"
# Run on the EXPANDED 75-task benchmark; SAGE seed-memory stays on the curated
# 38 (gt-path default) so the 37 new tasks are a held-out, no-leak evaluation.
DATASET="${DATASET:-/media/keti/workdir/remote_dir/pyplanner/eval_dataset_expanded.json}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE" || exit 1
BASE_MEM="data/memory.jsonl"
mkdir -p data/grid_mem results/grid_logs
# Generous per-call timeout so a slow serial generation completes
# rather than silently truncating to an empty plan.
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-300}"
HOSTTAG=$(echo "$HOST" | sed -E 's#https?://##; s#[:/].*##')
LOG="results/grid_logs/stream_${HOSTTAG}.log"

echo "=== $(date '+%F %T') STREAM START host=$HOST models=${MODELS[*]} timeout=$OLLAMA_TIMEOUT ===" | tee -a "$LOG"
for MODEL in "${MODELS[@]}"; do
    SAFE=$(echo "$MODEL" | tr ':' '_')
    for SEED in $SEEDS; do
        MEM="data/grid_mem/${SAFE}_s${SEED}.jsonl"
        cp "$BASE_MEM" "$MEM" 2>/dev/null || : > "$MEM"
        RUN="grid_${SAFE}_s${SEED}"
        echo "--- $(date '+%F %T') START $RUN on $HOST ---" | tee -a "$LOG"
        python3 scripts/run_benchmark.py \
            --mode plan --host "$HOST" \
            --dataset "$DATASET" \
            --models "$MODEL" \
            --methods-csv "$METHODS" \
            --seed "$SEED" \
            --live-path "$MEM" \
            --resume \
            --run-id "$RUN" >> "$LOG" 2>&1
        echo "--- $(date '+%F %T') DONE  $RUN (exit $?) ---" | tee -a "$LOG"
    done
done
echo "=== $(date '+%F %T') STREAM DONE host=$HOST ===" | tee -a "$LOG"
