#!/usr/bin/env bash

# Wrapper around launch_gemma4_gui_lora_refine_0826_seed43_44.sh that saves
# the full stdout/stderr to /data/uber_train/gemma4-training-code/logs, using
# the same naming convention as the existing logs in that directory:
#   uber_{lora|full}_{label}_{YYYYMMDD}_{HHMMSS}.log
#
# Usage:
#   RUN_MODE=full CUDA_VISIBLE_DEVICES=1,7 bash run_gemma4_gui_lora_refine_0826_seed43_44_with_log.sh
#
# Runs in the background via nohup; prints the log path and exits immediately.
# Overridable: LOG_DIR, LOG_LABEL (default: refine_0826_seed43_44_gpu<CUDA_VISIBLE_DEVICES with commas replaced by _>)

set -euo pipefail

GEMMA4_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${GEMMA4_SCRIPT_DIR}/launch_gemma4_gui_lora_refine_0826_seed43_44.sh"

LOG_DIR="${LOG_DIR:-/data/uber_train/gemma4-training-code/logs}"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LOG_LABEL="${LOG_LABEL:-refine_0826_seed43_44_gpu${CUDA_VISIBLE_DEVICES//,/_}}"
LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/uber_lora_${LOG_LABEL}_${LOG_TIMESTAMP}.log"

echo "Logging to: ${LOG_FILE}"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" bash "${LAUNCH_SCRIPT}" \
    > "${LOG_FILE}" 2>&1 &

echo "Started in background, PID: $!"
echo "Tail with: tail -f ${LOG_FILE}"
