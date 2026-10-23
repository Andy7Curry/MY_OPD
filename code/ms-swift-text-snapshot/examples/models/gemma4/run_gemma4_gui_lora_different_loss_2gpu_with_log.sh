#!/usr/bin/env bash

# Wrapper around gemma4_gui_lora_different_loss.sh that runs it on two GPUs and
# saves the full stdout/stderr to /data/uber_train/gemma4-training-code/logs,
# using the same naming convention as the existing logs in that directory:
#   uber_lora_{label}_{YYYYMMDD}_{HHMMSS}.log
#
# The underlying script defaults to four GPUs, so three settings must move
# together for a two-GPU run: CUDA_VISIBLE_DEVICES, NPROC_PER_NODE, and
# TENSOR_MODEL_PARALLEL_SIZE (tensor parallel size must equal the GPU count).
#
# Usage:
#   bash examples/models/gemma4/run_gemma4_gui_lora_different_loss_2gpu_with_log.sh
#   RUN_MODE=smoke bash examples/models/gemma4/run_gemma4_gui_lora_different_loss_2gpu_with_log.sh
#
# Runs in the background via nohup; prints the log path and exits immediately.
# Overridable: LOG_DIR, LOG_LABEL, CUDA_VISIBLE_DEVICES, RUN_MODE, MASTER_PORT,
#              GEMMA4_OUTPUT_DIR, GEMMA4_TRAIN_DATA, GEMMA4_VAL_DATA,
#              GEMMA4_SWANLAB_EXP_NAME

set -euo pipefail

GEMMA4_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${GEMMA4_SCRIPT_DIR}/gemma4_gui_lora_different_loss.sh"

RUN_MODE="${RUN_MODE:-full}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# 29502 is the underlying script's default; use a different one so this run
# cannot collide with a job already started from that script.
MASTER_PORT="${MASTER_PORT:-29510}"

# Derive the process count and tensor parallel size from the GPU list so the
# three stay consistent even when CUDA_VISIBLE_DEVICES is overridden.
IFS=',' read -r -a gemma4_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
gemma4_gpu_count="${#gemma4_gpu_ids[@]}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${gemma4_gpu_count}}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-${gemma4_gpu_count}}"

LOG_DIR="${LOG_DIR:-/data/uber_train/gemma4-training-code/logs}"
mkdir -p "${LOG_DIR}"

LOG_LABEL="${LOG_LABEL:-different_loss_gpu${CUDA_VISIBLE_DEVICES//,/_}}"
LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/uber_lora_${LOG_LABEL}_${LOG_TIMESTAMP}.log"

echo "Run mode:        ${RUN_MODE}"
echo "GPUs:            ${CUDA_VISIBLE_DEVICES} (${gemma4_gpu_count} devices)"
echo "Processes:       ${NPROC_PER_NODE}"
echo "Tensor parallel: ${TENSOR_MODEL_PARALLEL_SIZE}"
echo "Master port:     ${MASTER_PORT}"
echo "Logging to:      ${LOG_FILE}"

nohup env \
    RUN_MODE="${RUN_MODE}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE}" \
    MASTER_PORT="${MASTER_PORT}" \
    bash "${RUN_SCRIPT}" \
    > "${LOG_FILE}" 2>&1 &

echo "Started in background, PID: $!"
echo "Tail with: tail -f ${LOG_FILE}"
