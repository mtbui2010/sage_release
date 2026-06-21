#!/bin/bash
set -u
cd /media/keti/workdir/remote_dir/paper_sage
DS=../pyplanner/eval_dataset_expanded.json
SUBSET="K01 L02 B03 X03 K10 L10 A12 K04 L04 B06 X01 K15 K16 L14 K08 L07 B08 A08 K19 A19"
for tag in 3b 7b 14b; do
  python3 scripts/run_benchmark.py --mode plan --host 192.168.1.18:11535 \
    --dataset "$DS" --models qwen2.5:$tag --methods-csv "Direct,SAGE" \
    --task-ids $SUBSET --seed 0 \
    --live-path data/grid_mem/srvpar_${tag}_s0.jsonl \
    --run-id server_parity_${tag}_s0 >> edge_logs/server_parity.log 2>&1
done
echo "SERVER PARITY DONE $(date +%T)" >> edge_logs/server_parity.log
