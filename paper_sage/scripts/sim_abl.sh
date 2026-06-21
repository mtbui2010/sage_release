#!/usr/bin/env bash
# sim_abl.sh LLM_HOST PORT1 PORT2 MODEL [MODEL ...]
# Sim ABLATION: the four SAGE configurations (full + drop-one-pillar), executed
# at max-replan=0 (discriminative) so each pillar's effect on EXECUTION success
# is visible. Same base model => safe same-model concurrency on one LLM host.
# Two dataset-half workers on PORT1/PORT2 (use thor ports free of the main run).
set -u
LLM="${1:?usage: sim_abl.sh LLM_HOST PORT1 PORT2 MODEL...}"; shift
P1="${1:?need port1}"; shift
P2="${1:?need port2}"; shift
MODELS=("$@")
ROOT=/media/keti/workdir/remote_dir/paper_sage
PYP=/media/keti/workdir/remote_dir/pyplanner
export PYTHONPATH="$PYP:$PYP/apps" OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-400}"
METHODS=(SAGE SAGE-NoVerifier SAGE-NoRepair SAGE-NoMemory)
mkdir -p "$ROOT/results/sim" "$ROOT/results/sim_logs"
worker() { # $1=half $2=port $3=model $4=out
  /home/keti/miniconda3/bin/python -u "$PYP/apps/evaluate/evaluate_sim.py" \
    --dataset "$1" --methods "${METHODS[@]}" \
    --host "$LLM" --model "$3" --sim-host localhost --sim-port "$2" \
    --max-replan 0 --out "$4"
}
for M in "${MODELS[@]}"; do
  SAFE=$(echo "$M" | tr ':' '_')
  echo "=== $(date '+%F %T') ABLATION-SIM model=$M ==="
  worker /tmp/eval_h1.json "$P1" "$M" "$ROOT/results/sim/sim_${SAFE}_abl_h1.csv" \
      >> "$ROOT/results/sim_logs/sim_${SAFE}_abl_h1.log" 2>&1 &
  A=$!
  worker /tmp/eval_h2.json "$P2" "$M" "$ROOT/results/sim/sim_${SAFE}_abl_h2.csv" \
      >> "$ROOT/results/sim_logs/sim_${SAFE}_abl_h2.log" 2>&1 &
  B=$!
  wait $A $B
  /home/keti/miniconda3/bin/python - "$ROOT/results/sim/sim_${SAFE}_abl_h1.csv" \
        "$ROOT/results/sim/sim_${SAFE}_abl_h2.csv" "$ROOT/results/sim/sim_${SAFE}_abl.csv" <<'PY'
import csv,sys,os
h1,h2,out=sys.argv[1:4]; rows=[]
for f in (h1,h2):
    if os.path.exists(f): rows+=list(csv.DictReader(open(f)))
if rows:
    import csv as c
    with open(out,"w",newline="") as o:
        w=c.DictWriter(o,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("merged",len(rows),"->",out)
PY
  echo "=== $(date '+%F %T') done ablation model=$M ==="
done
