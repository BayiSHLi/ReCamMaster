#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-/mnt/hdd/dataset/webvid}"
SAVE_DIR="${SAVE_DIR:-/mnt/hdd/dataset/webvid/outputs}"
TRAJ_ROOT="${TRAJ_ROOT:-results/evaluation_full}"
OUT_JSON="${OUT_JSON:-results/evaluation_full/full_eval_on_existing_samples.json}"
LOG_FILE="${LOG_FILE:-evaluation/logs/eval/full_eval_on_existing_samples.log}"
PID_FILE="${PID_FILE:-evaluation/logs/eval/full_eval_on_existing_samples.pid}"

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")" "$TRAJ_ROOT"

nohup /opt/anaconda3/bin/conda run -p /home/user/.conda/envs/py310 --no-capture-output \
  env PYTHONUNBUFFERED=1 python -u \
  evaluation/evaluation.py \
  --skip_generation \
  --selected_metrics camera,matching,clip,fvd,vbench \
  --data_root "$DATA_ROOT" \
  --save_dir "$SAVE_DIR" \
  --trajectory_save_root "$TRAJ_ROOT" \
  --output_json "$OUT_JSON" \
  --no-eval_show_progress \
  "$@" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Started evaluation in background."
echo "PID: $PID"
echo "PID file: $PID_FILE"
echo "Log file: $LOG_FILE"
echo "Output json: $OUT_JSON"
echo "Watch logs: tail -f $LOG_FILE"
