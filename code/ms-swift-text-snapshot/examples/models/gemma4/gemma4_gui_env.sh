#!/usr/bin/env bash

# Shared runtime configuration for the local Gemma 4 GUI SFT workflow.
# Source this file instead of relying on the caller's active Conda environment.

set -euo pipefail

GEMMA4_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWIFT_ROOT="${SWIFT_ROOT:-$(cd -- "${GEMMA4_SCRIPT_DIR}/../../.." && pwd)}"
GEMMA4_CONDA_ENV="${GEMMA4_CONDA_ENV:-/data/lcx/miniconda3/envs/ms-swift-gemma4}"
GEMMA4_MODEL="${GEMMA4_MODEL:-/data/lcx/foxx/train/models/gemma-4-12B-it}"
GEMMA4_DATA_ROOT="${GEMMA4_DATA_ROOT:-/data/uber_train/gemma4-training-code/uber_data/uber_data_0804}"
GEMMA4_TRAIN_DATA="${GEMMA4_TRAIN_DATA:-${GEMMA4_DATA_ROOT}/sft_thinkanswer/train_ms_swift.jsonl}"
GEMMA4_VAL_DATA="${GEMMA4_VAL_DATA:-${GEMMA4_DATA_ROOT}/sft_thinkanswer/val_ms_swift.jsonl}"
GEMMA4_OUTPUT_DIR="${GEMMA4_OUTPUT_DIR:-/data/uber_train/gemma4-training-code/output_text/gemma4-12b-gui-lora}"
GEMMA4_FULL_OUTPUT_DIR="${GEMMA4_FULL_OUTPUT_DIR:-/data/uber_train/gemma4-training-code/output_text/gemma4-12b-gui-full}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-PuboCscoF0iv6SXLQqbUw}"
GEMMA4_SWANLAB_PROJECT="${GEMMA4_SWANLAB_PROJECT:-uber}"

for required_path in \
    "${GEMMA4_CONDA_ENV}/bin/python" \
    "${GEMMA4_MODEL}/config.json" \
    "${GEMMA4_TRAIN_DATA}" \
    "${GEMMA4_VAL_DATA}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        return 1 2>/dev/null || exit 1
    fi
done

if [[ ! -e "${GEMMA4_MODEL}/model.safetensors" && ! -e "${GEMMA4_MODEL}/model.safetensors.index.json" ]]; then
    echo "Required model weights not found (model.safetensors or model.safetensors.index.json) in: ${GEMMA4_MODEL}" >&2
    return 1 2>/dev/null || exit 1
fi
export SWIFT_ROOT
export GEMMA4_CONDA_ENV
export GEMMA4_MODEL
export GEMMA4_DATA_ROOT
export GEMMA4_TRAIN_DATA
export GEMMA4_VAL_DATA
export GEMMA4_OUTPUT_DIR
export GEMMA4_FULL_OUTPUT_DIR
export SWANLAB_API_KEY
export GEMMA4_SWANLAB_PROJECT

export PATH="${GEMMA4_CONDA_ENV}/bin:${PATH}"
export CUDA_HOME="${GEMMA4_CONDA_ENV}"
export ROOT_IMAGE_DIR="/data/dataset/Uber/"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

gemma4_nvidia_libs=""
shopt -s nullglob
for gemma4_lib_dir in "${GEMMA4_CONDA_ENV}"/lib/python3.12/site-packages/nvidia/*/lib; do
    gemma4_nvidia_libs="${gemma4_nvidia_libs:+${gemma4_nvidia_libs}:}${gemma4_lib_dir}"
done
shopt -u nullglob
if [[ -n "${gemma4_nvidia_libs}" ]]; then
    export LD_LIBRARY_PATH="${gemma4_nvidia_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

unset gemma4_lib_dir
unset gemma4_nvidia_libs
