#!/usr/bin/env bash
# R1: U-Net baseline for Gate 2. Same data, inputs, surface head, loss and schedule as R3 (v1);
# only the backbone differs. Trains, then scores on the 200 test cases and plots the curves.
#
#   ./run_R1.sh           U-Net with the FNO's channel width, 64 (7.8 M backbone values; the FNO
#                         has 17.9 M); its hidden field has 64 channels, like the FNO's
#   ./run_R1.sh matched   size-matched U-Net, width 96 (17.5 M), run id R1m
#
# Run it detached so it survives closing the terminal:  nohup ./run_R1.sh > runs_R1.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
PY=~/miniconda3/envs/fno/bin/python

if [[ "${1:-}" == "matched" ]]; then RUN=R1m; WIDTH=96; NEED_MB=9000; else RUN=R1; WIDTH=64; NEED_MB=6000; fi

# a shared GPU also makes training several times slower
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
echo "GPU: ${free_mb} MB free, ${others} other process(es)"
if (( free_mb < NEED_MB )); then
  echo "Not enough free GPU memory for $RUN (need about $((NEED_MB / 1000)) GB). Stop the other jobs first." >&2
  exit 1
fi
if [[ -d "runs/$RUN" ]]; then
  echo "runs/$RUN already exists; move or delete it before starting again." >&2
  exit 1
fi

$PY train.py --arch unet --width "$WIDTH" --run-id "$RUN" --epochs 500
$PY evaluate.py "$RUN"
$PY plot_run.py "$RUN"
echo "done: runs/$RUN, results/${RUN}_test_summary.json, figs/curves_$RUN.png"
