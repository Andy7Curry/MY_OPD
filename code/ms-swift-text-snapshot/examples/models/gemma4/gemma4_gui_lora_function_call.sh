#!/usr/bin/env bash

# Gemma 4 12B Unified GUI function-calling SFT on opd_data/sft, adapted from
# gemma4_unified.sh and the Megatron-SWIFT multimodal LoRA example.
#
# Safe default:
#   ./examples/models/gemma4/gemma4_gui_lora_function_call.sh
# runs only a 3-iteration smoke test.
#
# Full four-epoch run (the default for RUN_MODE=full):
#   RUN_MODE=full ./examples/models/gemma4/gemma4_gui_lora_function_call.sh
#
# Dump the first 10 template-encoded SFT samples as aligned token lists:
#   PRINT_STEPS=10 RUN_MODE=full ./examples/models/gemma4/gemma4_gui_lora_function_call.sh
# The report is written to the versioned output directory as sft_token_debug.txt.

set -euo pipefail

GEMMA4_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep ROOT_IMAGE_DIR at the dataset root because the JSONL image paths are
# relative to /data/lcx/foxx/train/dataset, not to opd_data/sft.
GEMMA4_DATA_ROOT="${GEMMA4_DATA_ROOT:-/data/lcx/foxx/train/dataset}"
GEMMA4_TRAIN_DATA="${GEMMA4_TRAIN_DATA:-${GEMMA4_DATA_ROOT}/opd_data/sft/train.jsonl}"
GEMMA4_VAL_DATA="${GEMMA4_VAL_DATA:-${GEMMA4_DATA_ROOT}/opd_data/sft/val.jsonl}"
GEMMA4_OUTPUT_DIR="${GEMMA4_OUTPUT_DIR:-/data/lcx/foxx/train/output/gemma4-12b-gui-lora-function-call}"
source "${GEMMA4_SCRIPT_DIR}/gemma4_gui_env.sh"

RUN_MODE="${RUN_MODE:-smoke}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-50000}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-/data/lcx/foxx/train/output/gemma4-12b-gui-function-call-preflight.json}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/data/lcx/foxx/train/cache/modelscope}"
MEGATRON_RUNNER="${MEGATRON_RUNNER:-megatron}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-4}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
PRINT_STEPS="${PRINT_STEPS:-0}"
PRESERVE_THINKING="${PRESERVE_THINKING:-true}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-false}"
GEMMA4_SWANLAB_EXP_NAME="${GEMMA4_SWANLAB_EXP_NAME:-gemma4-12b-gui-lora-function-call-${RUN_MODE}}"

if [[ "${RUN_MODE}" != "smoke" && "${RUN_MODE}" != "full" ]]; then
    echo "RUN_MODE must be 'smoke' or 'full'; received: ${RUN_MODE}" >&2
    exit 2
fi

for gemma4_integer_name in NUM_TRAIN_EPOCHS GLOBAL_BATCH_SIZE EVALS_PER_EPOCH SAVE_TOTAL_LIMIT; do
    gemma4_integer_value="${!gemma4_integer_name}"
    if [[ ! "${gemma4_integer_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${gemma4_integer_name} must be a positive integer; received: ${gemma4_integer_value}" >&2
        exit 2
    fi
done

if [[ ! "${PRINT_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "PRINT_STEPS must be a non-negative integer; received: ${PRINT_STEPS}" >&2
    exit 2
fi

if [[ "${PRESERVE_THINKING}" != "true" && "${PRESERVE_THINKING}" != "false" ]]; then
    echo "PRESERVE_THINKING must be 'true' or 'false'; received: ${PRESERVE_THINKING}" >&2
    exit 2
fi

if [[ "${LOAD_FROM_CACHE_FILE}" != "true" && "${LOAD_FROM_CACHE_FILE}" != "false" ]]; then
    echo "LOAD_FROM_CACHE_FILE must be 'true' or 'false'; received: ${LOAD_FROM_CACHE_FILE}" >&2
    exit 2
fi

IFS=',' read -r -a gemma4_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#gemma4_gpu_ids[@]}" -ne 4 ]]; then
    echo "This configuration requires exactly four GPU ids; received: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

if [[ "${SKIP_GPU_CHECK}" != "1" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is unavailable. Run on the GPU host or set SKIP_GPU_CHECK=1." >&2
        exit 3
    fi
    for gemma4_gpu_id in "${gemma4_gpu_ids[@]}"; do
        if ! gemma4_free_mib="$(
            nvidia-smi \
                --id="${gemma4_gpu_id}" \
                --query-gpu=memory.free \
                --format=csv,noheader,nounits
        )"; then
            echo \
                "nvidia-smi could not query GPU ${gemma4_gpu_id}. " \
                "Run this launcher from a host session with GPU devices attached." >&2
            exit 3
        fi
        gemma4_free_mib="${gemma4_free_mib//[[:space:]]/}"
        if [[ ! "${gemma4_free_mib}" =~ ^[0-9]+$ ]]; then
            echo "Could not read free memory for GPU ${gemma4_gpu_id}." >&2
            exit 3
        fi
        if (( gemma4_free_mib < MIN_FREE_MIB )); then
            echo \
                "GPU ${gemma4_gpu_id} has ${gemma4_free_mib} MiB free; " \
                "${MIN_FREE_MIB} MiB is required. Refusing to disturb an active job." >&2
            exit 3
        fi
    done
fi

if [[ -z "${MAX_LENGTH:-}" ]]; then
    if [[ ! -r "${PREFLIGHT_REPORT}" ]]; then
        echo "Missing preflight report: ${PREFLIGHT_REPORT}" >&2
        echo "Run ${GEMMA4_SCRIPT_DIR}/gemma4_gui_preflight.sh first." >&2
        exit 4
    fi
    MAX_LENGTH="$(
        "${GEMMA4_CONDA_ENV}/bin/python" -c \
            'import json, sys; print(json.load(open(sys.argv[1]))["length_recommendation"]["recommended_max_length"])' \
            "${PREFLIGHT_REPORT}"
    )"
fi
if [[ ! "${MAX_LENGTH}" =~ ^[0-9]+$ ]] || (( MAX_LENGTH < 4096 || MAX_LENGTH > 16384 )); then
    echo "MAX_LENGTH must be an integer in [4096, 16384]; received: ${MAX_LENGTH}" >&2
    exit 4
fi

mkdir -p "${MODELSCOPE_CACHE}"

gemma4_train_data="${GEMMA4_TRAIN_DATA}"
gemma4_val_data="${GEMMA4_VAL_DATA}"
gemma4_output_dir="${GEMMA4_OUTPUT_DIR}"
gemma4_load_cache="${LOAD_FROM_CACHE_FILE}"
gemma4_logging_steps="5"
gemma4_save_total_limit="${SAVE_TOTAL_LIMIT}"

# Mirror ms-swift Megatron's epoch calculation: floor(dataset_size / global_batch_size).
# Keep the schedule step-based so evaluation can run four times per epoch while
# checkpointing runs exactly once per epoch.
gemma4_train_rows="$(wc -l < "${gemma4_train_data}")"
gemma4_train_rows="${gemma4_train_rows//[[:space:]]/}"
if [[ ! "${gemma4_train_rows}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Could not determine the training row count from ${gemma4_train_data}." >&2
    exit 4
fi
gemma4_steps_per_epoch="$((gemma4_train_rows / GLOBAL_BATCH_SIZE))"
if (( gemma4_steps_per_epoch < 1 )); then
    echo \
        "Training rows (${gemma4_train_rows}) must be at least GLOBAL_BATCH_SIZE " \
        "(${GLOBAL_BATCH_SIZE})." >&2
    exit 4
fi
gemma4_eval_steps="$((gemma4_steps_per_epoch / EVALS_PER_EPOCH))"
if (( gemma4_eval_steps < 1 )); then
    gemma4_eval_steps=1
fi
gemma4_save_steps="${gemma4_steps_per_epoch}"
gemma4_estimated_train_iters="$((gemma4_train_rows * NUM_TRAIN_EPOCHS / GLOBAL_BATCH_SIZE))"
gemma4_schedule_args=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")

if [[ "${RUN_MODE}" == "smoke" ]]; then
    gemma4_train_data="${GEMMA4_TRAIN_DATA}#64"
    gemma4_val_data="${GEMMA4_VAL_DATA}#32"
    gemma4_output_dir="${GEMMA4_OUTPUT_DIR}/smoke"
    gemma4_load_cache="false"
    gemma4_eval_steps="1"
    gemma4_save_steps="2"
    gemma4_logging_steps="1"
    gemma4_save_total_limit="2"
    gemma4_train_rows="64"
    gemma4_steps_per_epoch="2"
    gemma4_estimated_train_iters="3"
    gemma4_schedule_args=(--train_iters 3)
fi

echo "Run mode:             ${RUN_MODE}"
echo "Conda environment:    ${GEMMA4_CONDA_ENV}"
echo "Model:                ${GEMMA4_MODEL}"
echo "Train dataset:        ${gemma4_train_data}"
echo "Validation dataset:   ${gemma4_val_data}"
echo "Image root:           ${ROOT_IMAGE_DIR}"
echo "GPUs:                 ${CUDA_VISIBLE_DEVICES}"
echo "Maximum length:       ${MAX_LENGTH}"
if [[ "${RUN_MODE}" == "full" ]]; then
    echo "Training epochs:      ${NUM_TRAIN_EPOCHS}"
    echo "Training rows:        ${gemma4_train_rows}"
    echo "Global batch size:    ${GLOBAL_BATCH_SIZE}"
    echo "Steps per epoch:      ${gemma4_steps_per_epoch}"
    echo "Estimated train iters: ${gemma4_estimated_train_iters}"
fi
echo "Evaluation interval:  ${gemma4_eval_steps} steps"
echo "Checkpoint interval:  ${gemma4_save_steps} steps"
echo "Checkpoint limit:     ${gemma4_save_total_limit}"
echo "Token debug samples:  ${PRINT_STEPS}"
echo "Preserve thinking:     ${PRESERVE_THINKING}"
echo "Load dataset cache:    ${gemma4_load_cache}"
echo "SwanLab project:      ${GEMMA4_SWANLAB_PROJECT}"
echo "SwanLab experiment:   ${GEMMA4_SWANLAB_EXP_NAME}"
echo "Output directory:     ${gemma4_output_dir}"

cd "${SWIFT_ROOT}"
export CUDA_VISIBLE_DEVICES
export MODELSCOPE_CACHE
export OMP_NUM_THREADS
export NPROC_PER_NODE=4
export PRINT_STEPS

exec "${MEGATRON_RUNNER}" sft \
    --model "${GEMMA4_MODEL}" \
    --template gemma4 \
    --agent_template gemma4 \
    --save_safetensors true \
    --dataset "${gemma4_train_data}" \
    --val_dataset "${gemma4_val_data}" \
    --load_from_cache_file "${gemma4_load_cache}" \
    --add_non_thinking_prefix true \
    --preserve_thinking "${PRESERVE_THINKING}" \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --tensor_model_parallel_size 4 \
    --micro_batch_size 1 \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --gradient_accumulation_fusion false \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir "${gemma4_output_dir}" \
    --eval_steps "${gemma4_eval_steps}" \
    --save_steps "${gemma4_save_steps}" \
    --save_total_limit "${gemma4_save_total_limit}" \
    --logging_steps "${gemma4_logging_steps}" \
    --report_to tensorboard swanlab \
    --swanlab_project "${GEMMA4_SWANLAB_PROJECT}" \
    --swanlab_exp_name "${GEMMA4_SWANLAB_EXP_NAME}" \
    --max_length "${MAX_LENGTH}" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend unfused \
    --group_by_length true \
    --padding_free false \
    "${gemma4_schedule_args[@]}" \
    "$@"
