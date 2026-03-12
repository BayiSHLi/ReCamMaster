#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/evaluation/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_evaluation_${TIMESTAMP}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/run_evaluation_${TIMESTAMP}.pid}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKER_LOG_DIR_DEFAULT="${LOG_DIR}/workers/run_${TIMESTAMP}/"
WORKER_LOG_DIR="${WORKER_LOG_DIR:-$WORKER_LOG_DIR_DEFAULT}"
START_SAMPLE_IDX="${START_SAMPLE_IDX:-0}"

if [[ ! "$WORKER_LOG_DIR" =~ /run_[0-9]{8}_[0-9]{6}/?$ ]]; then
	WORKER_LOG_DIR="${WORKER_LOG_DIR%/}/run_${TIMESTAMP}"
fi

HAS_WORKER_LOG_DIR_ARG=0
HAS_START_SAMPLE_IDX_ARG=0
HAS_SAVE_ORIGINAL_VIDEO_ARG=0
for arg in "$@"; do
	if [[ "$arg" == "--worker-log-dir" ]] || [[ "$arg" == --worker-log-dir=* ]]; then
		HAS_WORKER_LOG_DIR_ARG=1
	fi
	if [[ "$arg" == "--start_sample_idx" ]] || [[ "$arg" == --start_sample_idx=* ]]; then
		HAS_START_SAMPLE_IDX_ARG=1
	fi
	if [[ "$arg" == "--save_original_video" ]]; then
		HAS_SAVE_ORIGINAL_VIDEO_ARG=1
	fi
done

EXTRA_ARGS=()
if [[ "$HAS_WORKER_LOG_DIR_ARG" -eq 0 ]]; then
	EXTRA_ARGS+=("--worker-log-dir" "$WORKER_LOG_DIR")
fi
if [[ "$HAS_START_SAMPLE_IDX_ARG" -eq 0 ]]; then
	EXTRA_ARGS+=("--start_sample_idx" "$START_SAMPLE_IDX")
fi
if [[ "$HAS_SAVE_ORIGINAL_VIDEO_ARG" -eq 0 ]]; then
	EXTRA_ARGS+=("--save_original_video")
fi

mkdir -p "$WORKER_LOG_DIR"

CMD=("$PYTHON_BIN" "-u" "${REPO_ROOT}/evaluation/inference_webvid.py" "${EXTRA_ARGS[@]}" "$@")

echo "Starting in background: ${CMD[*]}"
nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

echo "Started inference_webvid.py in background"
echo "PID: $PID"
echo "PID file: $PID_FILE"
echo "Log file: $LOG_FILE"
echo "Worker logs dir: $WORKER_LOG_DIR"
echo "Start sample idx: ${START_SAMPLE_IDX}"
echo "Save original video: enabled"
echo "Tail logs with: tail -f $LOG_FILE"
echo "Tail worker logs with: tail -f ${WORKER_LOG_DIR}/run_*_worker*_gpu*.log"
