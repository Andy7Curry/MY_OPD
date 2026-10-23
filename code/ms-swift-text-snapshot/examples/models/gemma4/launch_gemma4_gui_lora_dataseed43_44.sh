#!/usr/bin/env bash

# Launch a paired data-order experiment from the same Gemma 4 base model:
#   data_seed=43 on GPUs 0,1,2,3
#   data_seed=44 on GPUs 4,5,6,7
#
# Only data_seed changes between the two runs. The global training seed remains
# 42 so the comparison primarily measures sensitivity to train-data ordering.

set -euo pipefail

PAIR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_TRAIN_SCRIPT="${PAIR_SCRIPT_DIR}/gemma4_gui_lora.sh"

RUN_MODE="${RUN_MODE:-smoke}"
PRINT_STEPS="${PRINT_STEPS:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
MIN_FREE_MIB="${MIN_FREE_MIB:-50000}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_PREFLIGHT_REPORT="${COMMON_PREFLIGHT_REPORT:-/data/lcx/foxx/train/output/gemma4-12b-gui-lora-0731/preflight.json}"
DATASEED43_OUTPUT_DIR="${DATASEED43_OUTPUT_DIR:-/data/lcx/foxx/train/output/gemma4-12b-gui-lora-0731-dataseed43}"
DATASEED44_OUTPUT_DIR="${DATASEED44_OUTPUT_DIR:-/data/lcx/foxx/train/output/gemma4-12b-gui-lora-0731-dataseed44}"
DATASEED43_MASTER_PORT="${DATASEED43_MASTER_PORT:-29543}"
DATASEED44_MASTER_PORT="${DATASEED44_MASTER_PORT:-29544}"
PAIR_LOG_ROOT="${PAIR_LOG_ROOT:-/data/lcx/foxx/train/logs/gemma4_lora_dataseed_pair}"
PAIR_RUN_TAG="${PAIR_RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
PAIR_LOG_DIR="${PAIR_LOG_ROOT}/${PAIR_RUN_TAG}"

DATASEED43_LOG="${PAIR_LOG_DIR}/dataseed43_gpu0123.log"
DATASEED44_LOG="${PAIR_LOG_DIR}/dataseed44_gpu4567.log"
PAIR_PID_FILE="${PAIR_LOG_DIR}/pids.txt"
PAIR_STATUS_FILE="${PAIR_LOG_DIR}/status.txt"

if [[ "${RUN_MODE}" != "smoke" && "${RUN_MODE}" != "full" ]]; then
    echo "RUN_MODE must be 'smoke' or 'full'; received: ${RUN_MODE}" >&2
    exit 2
fi
if [[ ! "${PRINT_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "PRINT_STEPS must be a non-negative integer; received: ${PRINT_STEPS}" >&2
    exit 2
fi
if [[ ! "${NUM_TRAIN_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_TRAIN_EPOCHS must be a positive integer; received: ${NUM_TRAIN_EPOCHS}" >&2
    exit 2
fi
if [[ ! "${MIN_FREE_MIB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MIN_FREE_MIB must be a positive integer; received: ${MIN_FREE_MIB}" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1; received: ${DRY_RUN}" >&2
    exit 2
fi
for pair_port_name in DATASEED43_MASTER_PORT DATASEED44_MASTER_PORT; do
    pair_port_value="${!pair_port_name}"
    if [[ ! "${pair_port_value}" =~ ^[0-9]+$ ]] || (( pair_port_value < 1024 || pair_port_value > 65535 )); then
        echo "${pair_port_name} must be an integer in [1024, 65535]; received: ${pair_port_value}" >&2
        exit 2
    fi
done
if [[ "${DATASEED43_MASTER_PORT}" == "${DATASEED44_MASTER_PORT}" ]]; then
    echo "The paired experiments must use different MASTER_PORT values." >&2
    exit 2
fi
if [[ ! -x "${BASE_TRAIN_SCRIPT}" ]]; then
    echo "Base training script is not executable: ${BASE_TRAIN_SCRIPT}" >&2
    exit 3
fi
if [[ ! -r "${COMMON_PREFLIGHT_REPORT}" ]]; then
    echo "Missing common preflight report: ${COMMON_PREFLIGHT_REPORT}" >&2
    exit 3
fi

mkdir -p "${PAIR_LOG_DIR}"

if [[ "${DRY_RUN}" != "1" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is unavailable; run this launcher on the GPU host." >&2
        exit 4
    fi
    for pair_gpu_id in 0 1 2 3 4 5 6 7; do
        if ! pair_free_mib="$(
            nvidia-smi \
                --id="${pair_gpu_id}" \
                --query-gpu=memory.free \
                --format=csv,noheader,nounits
        )"; then
            echo "Unable to query GPU ${pair_gpu_id}." >&2
            exit 4
        fi
        pair_free_mib="${pair_free_mib//[[:space:]]/}"
        if [[ ! "${pair_free_mib}" =~ ^[0-9]+$ ]]; then
            echo "Invalid free-memory value for GPU ${pair_gpu_id}: ${pair_free_mib}" >&2
            exit 4
        fi
        if (( pair_free_mib < MIN_FREE_MIB )); then
            echo \
                "GPU ${pair_gpu_id} has ${pair_free_mib} MiB free; " \
                "${MIN_FREE_MIB} MiB is required. Neither experiment was started." >&2
            exit 4
        fi
    done
fi

# A paired seed experiment must always start from the same base model. Prevent
# inherited resume variables from silently loading an existing adapter.
unset RESUME_ADAPTER RESUME_LR RESUME_MIN_LR RESUME_LR_DECAY_ITERS

echo "Paired data-order experiment"
echo "Run mode:          ${RUN_MODE}"
echo "Global seed:       42 (fixed for both runs)"
echo "Data seed 43:      GPUs 0,1,2,3; MASTER_PORT=${DATASEED43_MASTER_PORT}"
echo "Data seed 44:      GPUs 4,5,6,7; MASTER_PORT=${DATASEED44_MASTER_PORT}"
echo "Validation:        fixed; val_dataset_shuffle=false"
echo "Preflight report:  ${COMMON_PREFLIGHT_REPORT}"
echo "Log directory:     ${PAIR_LOG_DIR}"
echo "Dry run:           ${DRY_RUN}"

launch_experiment() {
    local pair_data_seed="$1"
    local pair_gpu_ids="$2"
    local pair_master_port="$3"
    local pair_output_dir="$4"
    local pair_log_file="$5"
    local pair_exp_name="gemma4-12b-gui-lora-0731-dataseed${pair_data_seed}-${RUN_MODE}"
    local pair_skip_gpu_check="0"
    local pair_megatron_runner="megatron"

    if [[ "${DRY_RUN}" == "1" ]]; then
        pair_skip_gpu_check="1"
        pair_megatron_runner="echo"
    fi

    env \
        RUN_MODE="${RUN_MODE}" \
        PRINT_STEPS="${PRINT_STEPS}" \
        NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
        CUDA_VISIBLE_DEVICES="${pair_gpu_ids}" \
        MASTER_PORT="${pair_master_port}" \
        MIN_FREE_MIB="${MIN_FREE_MIB}" \
        SKIP_GPU_CHECK="${pair_skip_gpu_check}" \
        MEGATRON_RUNNER="${pair_megatron_runner}" \
        GEMMA4_OUTPUT_DIR="${pair_output_dir}" \
        PREFLIGHT_REPORT="${COMMON_PREFLIGHT_REPORT}" \
        GEMMA4_SWANLAB_EXP_NAME="${pair_exp_name}" \
        bash "${BASE_TRAIN_SCRIPT}" \
            --data_seed "${pair_data_seed}" \
            --seed 42 \
            --dataset_shuffle true \
            --val_dataset_shuffle false \
        >"${pair_log_file}" 2>&1 &

    PAIR_LAUNCHED_PID="$!"
}

pair_pid43=""
pair_pid44=""

terminate_children() {
    echo "Launcher received a termination signal; forwarding SIGTERM to active child processes." >&2
    for pair_pid in "${pair_pid43}" "${pair_pid44}"; do
        if [[ -n "${pair_pid}" ]] && kill -0 "${pair_pid}" 2>/dev/null; then
            kill -TERM "${pair_pid}" 2>/dev/null || true
        fi
    done
}
trap terminate_children INT TERM

launch_experiment \
    43 "0,1,2,3" "${DATASEED43_MASTER_PORT}" \
    "${DATASEED43_OUTPUT_DIR}" "${DATASEED43_LOG}"
pair_pid43="${PAIR_LAUNCHED_PID}"
launch_experiment \
    44 "4,5,6,7" "${DATASEED44_MASTER_PORT}" \
    "${DATASEED44_OUTPUT_DIR}" "${DATASEED44_LOG}"
pair_pid44="${PAIR_LAUNCHED_PID}"

printf \
    'launcher_pid=%s\ndataseed43_pid=%s\ndataseed44_pid=%s\n' \
    "$$" "${pair_pid43}" "${pair_pid44}" \
    >"${PAIR_PID_FILE}"

echo "data_seed=43 PID: ${pair_pid43}; log: ${DATASEED43_LOG}"
echo "data_seed=44 PID: ${pair_pid44}; log: ${DATASEED44_LOG}"
echo "PID file:          ${PAIR_PID_FILE}"

set +e
wait "${pair_pid43}"
pair_status43="$?"
wait "${pair_pid44}"
pair_status44="$?"
set -e

printf \
    'dataseed43_exit_status=%s\ndataseed44_exit_status=%s\n' \
    "${pair_status43}" "${pair_status44}" \
    >"${PAIR_STATUS_FILE}"

echo "data_seed=43 exit status: ${pair_status43}"
echo "data_seed=44 exit status: ${pair_status44}"
echo "Status file:              ${PAIR_STATUS_FILE}"

if (( pair_status43 != 0 || pair_status44 != 0 )); then
    exit 1
fi
