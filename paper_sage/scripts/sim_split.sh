#!/usr/bin/env bash
# sim_split.sh LLM_HOST MODEL [MODEL ...]
# For each model (SEQUENTIALLY, to avoid different-model GPU thrash), run TWO
# evaluate_sim workers concurrently on dataset halves (same model => safe
# same-model concurrency), each on its own local thor_server, then merge.
set -u
LLM="${1:?usage: sim_split.sh LLM_HOST MODEL...}"; shift
MODELS=("$@")
ROOT=/media/keti/workdir/remote_dir/paper_sage
PYP=/media/keti/workdir/remote_dir/pyplanner
export PYTHONPATH="$PYP:$PYP/apps" OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-300}"
MAXREPLAN="${MAXREPLAN:-2}"   # 0 = execute-as-planned (discriminative); 2 = with recovery
SUFFIX="${SUFFIX:-}"          # output filename suffix, e.g. _r0
mkdir -p "$ROOT/results/sim" "$ROOT/results/sim_logs"
worker() { # $1=half-file $2=port $3=model $4=outcsv
  /home/keti/miniconda3/bin/python -u "$PYP/apps/evaluate/evaluate_sim.py" \
    --dataset "$1" --methods Direct Hierarchical SAGE \
    --host "$LLM" --model "$3" --sim-host localhost --sim-port "$2" \
    --max-replan "$MAXREPLAN" --out "$4"
}
for M in "${MODELS[@]}"; do
  SAFE=$(echo "$M" | tr ':' '_')
  echo "=== $(date '+%F %T') SPLIT-SIM model=$M ==="
  worker /tmp/eval_h1.json 5556 "$M" "$ROOT/results/sim/sim_${SAFE}${SUFFIX}_h1.csv" \
      >> "$ROOT/results/sim_logs/sim_${SAFE}${SUFFIX}_h1.log" 2>&1 &
  P1=$!
  worker /tmp/eval_h2.json 5557 "$M" "$ROOT/results/sim/sim_${SAFE}${SUFFIX}_h2.csv" \
      >> "$ROOT/results/sim_logs/sim_${SAFE}${SUFFIX}_h2.log" 2>&1 &
  P2=$!
  wait $P1 $P2
  # merge halves -> sim_<model><suffix>.csv
  /home/keti/miniconda3/bin/python - "$ROOT/results/sim/sim_${SAFE}${SUFFIX}_h1.csv" \
        "$ROOT/results/sim/sim_${SAFE}${SUFFIX}_h2.csv" "$ROOT/results/sim/sim_${SAFE}${SUFFIX}.csv" <<'PY'
import csv,sys,os
h1,h2,out=sys.argv[1:4]
rows=[]
for f in (h1,h2):
    if os.path.exists(f): rows+=list(csv.DictReader(open(f)))
if rows:
    with open(out,"w",newline="") as o:
        w=csv.DictWriter(o, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("merged",len(rows),"->",out)
PY
  echo "=== $(date '+%F %T') done model=$M ==="
done
