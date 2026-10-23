#!/usr/bin/env bash

# Run gemma4_gui_lora_refine_0826.sh twice, sequentially, on the same GPUs (2,3),
# once with seed/data_seed=43 and once with seed/data_seed=44. Each run gets its
# own output directory and SwanLab experiment name so the two are never mixed up.
#
# This is 2 of a 3-way seed comparison (42, 43, 44). The seed=42 run already
# completed at:
#   /data/uber_train/gemma4-training-code/textoutput/gemma4-12b-gui-lora-0826-refine-last-step-560-4epoch/v0-20260826-051419
# so it is not re-run here.
#
# Usage:
#   RUN_MODE=full bash examples/models/gemma4/launch_gemma4_gui_lora_refine_0826_seed43_44.sh
#
# Overridable:
#   SEED43_OUTPUT_DIR, SEED44_OUTPUT_DIR
#   GEMMA4_MODEL, GEMMA4_TRAIN_DATA, GEMMA4_VAL_DATA, ROOT_IMAGE_DIR
#   CUDA_VISIBLE_DEVICES (defaults to 2,3 for both runs, run sequentially)

set -euo pipefail

GEMMA4_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${GEMMA4_SCRIPT_DIR}/gemma4_gui_lora_refine_0826.sh"

RUN_MODE="${RUN_MODE:-full}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

SEED43_OUTPUT_DIR="${SEED43_OUTPUT_DIR:-/data/uber_train/gemma4-training-code/textoutput/gemma4-12b-gui-lora-0826-refine-last-step-560-4epoch-seed43}"
SEED44_OUTPUT_DIR="${SEED44_OUTPUT_DIR:-/data/uber_train/gemma4-training-code/textoutput/gemma4-12b-gui-lora-0826-refine-last-step-560-4epoch-seed44}"

echo "=== Run 1/2: seed=43, data_seed=43 ==="
echo "Output: ${SEED43_OUTPUT_DIR}"
RUN_MODE="${RUN_MODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
SEED=43 \
DATA_SEED=43 \
GEMMA4_OUTPUT_DIR="${SEED43_OUTPUT_DIR}" \
GEMMA4_SWANLAB_EXP_NAME="${GEMMA4_SWANLAB_EXP_NAME:-gemma4-12b-refine-last-step-560-4epoch-seed43}" \
bash "${RUN_SCRIPT}"

echo "=== Run 2/2: seed=44, data_seed=44 ==="
echo "Output: ${SEED44_OUTPUT_DIR}"
RUN_MODE="${RUN_MODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
SEED=44 \
DATA_SEED=44 \
GEMMA4_OUTPUT_DIR="${SEED44_OUTPUT_DIR}" \
GEMMA4_SWANLAB_EXP_NAME="${GEMMA4_SWANLAB_EXP_NAME:-gemma4-12b-refine-last-step-560-4epoch-seed44}" \
bash "${RUN_SCRIPT}"

echo "=== Both runs complete ==="
echo "seed=43 output: ${SEED43_OUTPUT_DIR}"
echo "seed=44 output: ${SEED44_OUTPUT_DIR}"
