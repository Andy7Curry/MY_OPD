#!/usr/bin/env bash

# Launch two Gemma 4 GUI LoRA experiments from physically pre-shuffled JSONL files:
#   offline shuffle seed=1 on GPUs 0,1,2,3
#   offline shuffle seed=2 on GPUs 4,5,6,7
#
# The runtime dataset/dataloader shuffle and group-by-length sampler are disabled,
# so each epoch consumes the selected JSONL in its stored order. The model seed is
# fixed at 42 in both runs; only the physical training-row order differs.

set -euo pipefail

PAIR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_TRAIN_SCRIPT="${PAIR_SCRIPT_DIR}/gemma4_gui_lora.sh"

RUN_MODE="${RUN_MODE:-smoke}"
PRINT_STEPS="${PRINT_STEPS:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
MIN_FREE_MIB="${MIN_FREE_MIB:-50000}"
DRY_RUN="${DRY_RUN:-0}"

TRAIN_SEED1="${TRAIN_SEED1:-/data/lcx/foxx/train/dataset/0731/train_ms_swift_text_shuffle_seed1.jsonl}"
TRAIN_SEED2="${TRAIN_SEED2:-/data/lcx/foxx/train/dataset/0731/train_ms_swift_text_shuffle_seed2.jsonl}"
VAL_DATA="${VAL_DATA:-/data/lcx/foxx/train/dataset/0731/val_ms_swift_text.jsonl}"
COMMON_PREFLIGHT_REPORT="${COMMON_PREFLIGHT_REPORT:-/data/lcx/foxx/train/output/gemma4-12b-gui-lora-0731/preflight.json}"

SEED1_OUTPUT_DIR="${SEED1_OUTPUT_DIR:-/data/lcx/foxx/train/output/gemma4-12b-gui-lora-0731-preshuffle-seed1}"
SEED2_OUTPUT_DIR="${SEED2_OUTPUT_DIR:-/data/lcx/foxx/train/output/gemma4-12b-gui-lora-0731-preshuffle-seed2}"
SEED1_MASTER_PORT="${SEED1_MASTER_PORT:-29551}"
SEED2_MASTER_PORT="${SEED2_MASTER_PORT:-29552}"

PAIR_LOG_ROOT="${PAIR_LOG_ROOT:-/data/lcx/foxx/train/logs/gemma4_lora_preshuffle_seed1_2}"
PAIR_RUN_TAG="${PAIR_RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
PAIR_LOG_DIR="${PAIR_LOG_ROOT}/${PAIR_RUN_TAG}"
SEED1_LOG="${PAIR_LOG_DIR}/preshuffle_seed1_gpu0123.log"
SEED2_LOG="${PAIR_LOG_DIR}/preshuffle_seed2_gpu4567.log"
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

for pair_port_name in SEED1_MASTER_PORT SEED2_MASTER_PORT; do
    pair_port_value="${!pair_port_name}"
    if [[ ! "${pair_port_value}" =~ ^[0-9]+$ ]] || (( pair_port_value < 1024 || pair_port_value > 65535 )); then
        echo "${pair_port_name} must be an integer in [1024, 65535]; received: ${pair_port_value}" >&2
        exit 2
    fi
done
if [[ "${SEED1_MASTER_PORT}" == "${SEED2_MASTER_PORT}" ]]; then
    echo "The paired experiments must use different MASTER_PORT values." >&2
    exit 2
fi

if [[ ! -x "${BASE_TRAIN_SCRIPT}" ]]; then
    echo "Base training script is not executable: ${BASE_TRAIN_SCRIPT}" >&2
    exit 3
fi
for pair_required_file in "${TRAIN_SEED1}" "${TRAIN_SEED2}" "${VAL_DATA}" "${COMMON_PREFLIGHT_REPORT}"; do
    if [[ ! -s "${pair_required_file}" ]]; then
        echo "Missing or empty required file: ${pair_required_file}" >&2
        exit 3
    fi
done

seed1_rows="$(wc -l < "${TRAIN_SEED1}")"
seed2_rows="$(wc -l < "${TRAIN_SEED2}")"
seed1_rows="${seed1_rows//[[:space:]]/}"
seed2_rows="${seed2_rows//[[:space:]]/}"
if [[ ! "${seed1_rows}" =~ ^[1-9][0-9]*$ ]] || [[ "${seed1_rows}" != "${seed2_rows}" ]]; then
    echo "Offline-shuffled datasets must have the same positive row count; seed1=${seed1_rows}, seed2=${seed2_rows}." >&2
    exit 3
fi
if cmp -s "${TRAIN_SEED1}" "${TRAIN_SEED2}"; then
    echo "The seed1 and seed2 training files are byte-identical; refusing to start." >&2
    exit 3
fi

for pair_output_dir in "${SEED1_OUTPUT_DIR}" "${SEED2_OUTPUT_DIR}"; do
    if [[ "${RUN_MODE}" == "smoke" ]]; then
        pair_mode_output="${pair_output_dir}/smoke"
        if [[ -e "${pair_mode_output}" ]]; then
            echo "Smoke output already exists; refusing to overwrite: ${pair_mode_output}" >&2
            exit 3
        fi
    elif [[ -e "${pair_output_dir}" ]]; then
        if [[ ! -d "${pair_output_dir}" ]]; then
            echo "Full-run output path exists and is not a directory: ${pair_output_dir}" >&2
            exit 3
        fi
        pair_existing_full_entry="$(
            find "${pair_output_dir}" -mindepth 1 -maxdepth 1 ! -name smoke -print -quit
        )"
        if [[ -n "${pair_existing_full_entry}" ]]; then
            echo "Full-run output already contains results; refusing to overwrite or resume: ${pair_existing_full_entry}" >&2
            exit 3
        fi
    fi
done

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
            echo "GPU ${pair_gpu_id} has ${pair_free_mib} MiB free; ${MIN_FREE_MIB} MiB is required. Neither experiment was started." >&2
            exit 4
        fi
    done

    for pair_port in "${SEED1_MASTER_PORT}" "${SEED2_MASTER_PORT}"; do
        if ! /data/lcx/miniconda3/envs/ms-swift-gemma4/bin/python -c \
            'import socket, sys; s=socket.socket(); s.bind(("0.0.0.0", int(sys.argv[1]))); s.close()' \
            "${pair_port}"; then
            echo "MASTER_PORT ${pair_port} is unavailable. Neither experiment was started." >&2
            exit 4
        fi
    done
fi

mkdir -p "${PAIR_LOG_DIR}"

# Always start from the base model; inherited resume settings must not affect
# this controlled data-order comparison.
unset RESUME_ADAPTER RESUME_LR RESUME_MIN_LR RESUME_LR_DECAY_ITERS

echo "Paired offline-shuffle experiment"
echo "Run mode:          ${RUN_MODE}"
echo "Model seed:        42 (fixed for both runs)"
echo "Offline seed 1:    GPUs 0,1,2,3; MASTER_PORT=${SEED1_MASTER_PORT}"
echo "Offline seed 2:    GPUs 4,5,6,7; MASTER_PORT=${SEED2_MASTER_PORT}"
echo "Training rows:     ${seed1_rows} per run"
echo "Runtime shuffle:   disabled"
echo "Group by length:   disabled"
echo "Validation:        fixed and unshuffled"
echo "Preflight report:  ${COMMON_PREFLIGHT_REPORT}"
echo "Log directory:     ${PAIR_LOG_DIR}"
echo "Dry run:           ${DRY_RUN}"

launch_experiment() {
    local pair_offline_seed="$1"
    local pair_train_data="$2"
    local pair_gpu_ids="$3"
    local pair_master_port="$4"
    local pair_output_dir="$5"
    local pair_log_file="$6"
    local pair_exp_name="gemma4-12b-gui-lora-0731-preshuffle-seed${pair_offline_seed}-${RUN_MODE}"
    local pair_skip_gpu_check="0"
    local pair_megatron_runner="megatron"

    if [[ "${DRY_RUN}" == "1" ]]; then
        pair_skip_gpu_check="1"
        pair_megatron_runner="echo"
    fi

    setsid env \
        RUN_MODE="${RUN_MODE}" \
        PRINT_STEPS="${PRINT_STEPS}" \
        NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
        CUDA_VISIBLE_DEVICES="${pair_gpu_ids}" \
        MASTER_PORT="${pair_master_port}" \
        MIN_FREE_MIB="${MIN_FREE_MIB}" \
        SKIP_GPU_CHECK="${pair_skip_gpu_check}" \
        MEGATRON_RUNNER="${pair_megatron_runner}" \
        GEMMA4_TRAIN_DATA="${pair_train_data}" \
        GEMMA4_VAL_DATA="${VAL_DATA}" \
        GEMMA4_OUTPUT_DIR="${pair_output_dir}" \
        PREFLIGHT_REPORT="${COMMON_PREFLIGHT_REPORT}" \
        GEMMA4_SWANLAB_EXP_NAME="${pair_exp_name}" \
        bash "${BASE_TRAIN_SCRIPT}" \
            --data_seed "${pair_offline_seed}" \
            --seed 42 \
            --dataset_shuffle false \
            --train_dataloader_shuffle false \
            --val_dataset_shuffle false \
            --group_by_length false \
        >"${pair_log_file}" 2>&1 &

    PAIR_LAUNCHED_PID="$!"
}

pair_pid1=""
pair_pid2=""

terminate_children() {
    echo "Launcher received a termination signal; forwarding SIGTERM to active child process groups." >&2
    for pair_pid in "${pair_pid1}" "${pair_pid2}"; do
        if [[ -n "${pair_pid}" ]] && kill -0 "${pair_pid}" 2>/dev/null; then
            kill -TERM -- "-${pair_pid}" 2>/dev/null || true
        fi
    done
}
trap terminate_children INT TERM

launch_experiment 1 "${TRAIN_SEED1}" "0,1,2,3" "${SEED1_MASTER_PORT}" "${SEED1_OUTPUT_DIR}" "${SEED1_LOG}"
pair_pid1="${PAIR_LAUNCHED_PID}"
launch_experiment 2 "${TRAIN_SEED2}" "4,5,6,7" "${SEED2_MASTER_PORT}" "${SEED2_OUTPUT_DIR}" "${SEED2_LOG}"
pair_pid2="${PAIR_LAUNCHED_PID}"

printf \
    'launcher_pid=%s\npreshuffle_seed1_pid=%s\npreshuffle_seed2_pid=%s\n' \
    "$$" "${pair_pid1}" "${pair_pid2}" \
    >"${PAIR_PID_FILE}"

echo "preshuffle seed=1 PID: ${pair_pid1}; log: ${SEED1_LOG}"
echo "preshuffle seed=2 PID: ${pair_pid2}; log: ${SEED2_LOG}"
echo "PID file:               ${PAIR_PID_FILE}"

set +e
wait "${pair_pid1}"
pair_status1="$?"
wait "${pair_pid2}"
pair_status2="$?"
set -e

printf \
    'preshuffle_seed1_exit_status=%s\npreshuffle_seed2_exit_status=%s\n' \
    "${pair_status1}" "${pair_status2}" \
    >"${PAIR_STATUS_FILE}"

echo "preshuffle seed=1 exit status: ${pair_status1}"
echo "preshuffle seed=2 exit status: ${pair_status2}"
echo "Status file:                    ${PAIR_STATUS_FILE}"

if (( pair_status1 != 0 || pair_status2 != 0 )); then
    exit 1
fi
