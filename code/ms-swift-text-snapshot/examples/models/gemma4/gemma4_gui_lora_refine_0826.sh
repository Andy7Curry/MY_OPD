#!/usr/bin/env bash

# Gemma 4 12B Unified GUI SFT — 0826 uber_gjr_refine last-step-only run.
# Forked from gemma4_gui_lora_different_loss.sh with defaults pinned to:
#   - base model:  checkpoint-225-merged (0817 max-soft-token-560 LoRA run, merged)
#   - dataset:     annotation_export_uber_gjr_refine_8_18_last_step_only
#   - GPUs:        2,3 (TP=2, NPROC_PER_NODE=2)
#   - max soft tokens: 560
#   - epochs:      4
#   - global batch size: 4 (35 train rows; the default 32 would leave ~1 step/epoch)
#
# Safe default:
#   ./examples/models/gemma4/gemma4_gui_lora_refine_0826.sh
# runs only a 3-iteration smoke test.
#
# Full four-epoch run (the default for RUN_MODE=full):
#   RUN_MODE=full ./examples/models/gemma4/gemma4_gui_lora_refine_0826.sh

set -euo pipefail
export GEMMA4_MAX_SOFT_TOKENS="${GEMMA4_MAX_SOFT_TOKENS:-560}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT="${MASTER_PORT:-29503}"


GEMMA4_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GEMMA4_TRAIN_DATA="${GEMMA4_TRAIN_DATA:-/data/uber_train/gemma4-training-code/uber_data/uber_data_0804/sft_refine/annotation_export_uber_gjr_refine_8_18_last_step_only/train_ms_swift_sysfix.jsonl}"
GEMMA4_VAL_DATA="${GEMMA4_VAL_DATA:-/data/uber_train/gemma4-training-code/uber_data/uber_data_0804/sft_refine/annotation_export_uber_gjr_refine_8_18_last_step_only/val_ms_swift_sysfix.jsonl}"
GEMMA4_MODEL="${GEMMA4_MODEL:-/data/uber_train/gemma4-training-code/textoutput/gemma4-12b-gui-lora-0817-max-soft-token-560/v3-20260817-111434/checkpoint-225-merged}"
GEMMA4_OUTPUT_DIR="${GEMMA4_OUTPUT_DIR:-/data/uber_train/gemma4-training-code/textoutput/gemma4-12b-gui-lora-0826-refine-last-step-560-4epoch}"
source "${GEMMA4_SCRIPT_DIR}/gemma4_gui_env_refine_0826.sh"

RUN_MODE="${RUN_MODE:-full}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-50000}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${GEMMA4_OUTPUT_DIR}/preflight.json}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/data/uber_train/cache/modelscope}"
MEGATRON_RUNNER="${MEGATRON_RUNNER:-megatron}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-2}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
PRINT_STEPS="${PRINT_STEPS:-0}"
GEMMA4_SWANLAB_EXP_NAME="${GEMMA4_SWANLAB_EXP_NAME:-gemma4-12b-refine-last-step-560-4epoch}"
RESUME_ADAPTER="${RESUME_ADAPTER:-}"
RESUME_LR="${RESUME_LR:-1e-5}"
RESUME_MIN_LR="${RESUME_MIN_LR:-5e-6}"
RESUME_LR_DECAY_ITERS="${RESUME_LR_DECAY_ITERS:-}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

MAX_LENGTH="${MAX_LENGTH:-16384}"
mkdir -p "${MODELSCOPE_CACHE}"

gemma4_train_data="${GEMMA4_TRAIN_DATA}"
gemma4_val_data="${GEMMA4_VAL_DATA}"
gemma4_output_dir="${GEMMA4_OUTPUT_DIR}"
gemma4_load_cache="true"
gemma4_logging_steps="5"
gemma4_save_total_limit="${SAVE_TOTAL_LIMIT}"

# Mirror ms-swift Megatron's epoch calculation: floor(dataset_size / global_batch_size).
# Keep the schedule step-based so evaluation can run twice per epoch while
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
gemma4_resume_args=()

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

if [[ -n "${RESUME_ADAPTER}" ]]; then
    if [[ "${RUN_MODE}" != "full" ]]; then
        echo "RESUME_ADAPTER is only supported with RUN_MODE=full." >&2
        exit 5
    fi
    if [[ "${RESUME_ADAPTER}" == *-merged ]]; then
        echo "RESUME_ADAPTER must point to an unmerged LoRA checkpoint: ${RESUME_ADAPTER}" >&2
        exit 5
    fi
    if [[ ! -r "${RESUME_ADAPTER}/adapter_model.safetensors" ]]; then
        echo "Missing LoRA weights: ${RESUME_ADAPTER}/adapter_model.safetensors" >&2
        exit 5
    fi
    if [[ ! -r "${RESUME_ADAPTER}/latest_checkpointed_iteration.txt" ]]; then
        echo "Missing iteration tracker: ${RESUME_ADAPTER}/latest_checkpointed_iteration.txt" >&2
        exit 5
    fi
    gemma4_resume_iteration="$(<"${RESUME_ADAPTER}/latest_checkpointed_iteration.txt")"
    gemma4_resume_iteration="${gemma4_resume_iteration//[[:space:]]/}"
    if [[ ! "${gemma4_resume_iteration}" =~ ^[0-9]+$ ]]; then
        echo "Invalid checkpoint iteration: ${gemma4_resume_iteration}" >&2
        exit 5
    fi
    if (( gemma4_resume_iteration >= gemma4_estimated_train_iters )); then
        echo \
            "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS} yields ${gemma4_estimated_train_iters} total steps, " \
            "which is not greater than checkpoint iteration ${gemma4_resume_iteration}." >&2
        exit 5
    fi
    if [[ -z "${RESUME_LR_DECAY_ITERS}" ]]; then
        RESUME_LR_DECAY_ITERS="$((gemma4_estimated_train_iters - gemma4_resume_iteration))"
    fi
    if [[ ! "${RESUME_LR_DECAY_ITERS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "RESUME_LR_DECAY_ITERS must be a positive integer; received: ${RESUME_LR_DECAY_ITERS}" >&2
        exit 5
    fi
    gemma4_resume_args=(
        --adapters "${RESUME_ADAPTER}"
        --finetune false
        --no_load_optim true
        --no_load_rng true
        --lr "${RESUME_LR}"
        --min_lr "${RESUME_MIN_LR}"
        --lr_warmup_fraction 0
        --lr_decay_iters "${RESUME_LR_DECAY_ITERS}"
    )
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
echo "SwanLab project:      ${GEMMA4_SWANLAB_PROJECT}"
echo "SwanLab experiment:   ${GEMMA4_SWANLAB_EXP_NAME}"
echo "Output directory:     ${gemma4_output_dir}"
echo "Seed:                 ${SEED}"
echo "Data seed:            ${DATA_SEED}"
if [[ -n "${RESUME_ADAPTER}" ]]; then
    echo "Resume adapter:       ${RESUME_ADAPTER}"
    echo "Resume iteration:     ${gemma4_resume_iteration}"
    echo "Resume learning rate: ${RESUME_LR} -> ${RESUME_MIN_LR}"
    echo "Resume decay steps:   ${RESUME_LR_DECAY_ITERS}"
fi

cd "${SWIFT_ROOT}"
export CUDA_VISIBLE_DEVICES
export MODELSCOPE_CACHE
export OMP_NUM_THREADS
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export PRINT_STEPS

exec "${MEGATRON_RUNNER}" sft \
    --model "${GEMMA4_MODEL}" \
    --template gemma4 \
    --enable_thinking false \
    --save_safetensors true \
    --dataset "${gemma4_train_data}" \
    --val_dataset "${gemma4_val_data}" \
    --load_from_cache_file "${gemma4_load_cache}" \
    --add_non_thinking_prefix true \
    --preserve_thinking false \
    --loss_scale ignore_empty_think  \
    --external_plugins /data/uber_train/gemma4-training-code/ms-swift-text/examples/models/gemma4/gemma4_max_soft_tokens_plugin.py \
    --split_dataset_ratio 0 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE:-2}" \
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
    --seed "${SEED}" \
    --data_seed "${DATA_SEED}" \
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
    --attention_backend auto \
    --group_by_length true \
    --padding_free false \
    "${gemma4_schedule_args[@]}" \
    "${gemma4_resume_args[@]}" \
    "$@"
