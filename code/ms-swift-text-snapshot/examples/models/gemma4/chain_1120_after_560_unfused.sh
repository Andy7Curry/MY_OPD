#!/usr/bin/env bash
# 等 560-unfused 两个训练（EN PID 1665311 / ZH PID 1665312）结束后，
# 自动用 GEMMA4_MAX_SOFT_TOKENS=1120 在相同卡位、相同参数下各再训一版。
set -uo pipefail

EN_PID=1665311
ZH_PID=1665312
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=/data/uber_train/gemma4-training-code/logs
DATA_ROOT=/data/uber_train/gemma4-training-code/uber_data/uber_data_0804
OUT_ROOT=/data/uber_train/gemma4-training-code/textoutput

echo "[chain] waiting for PID ${EN_PID} (en) and ${ZH_PID} (zh) to exit..."
while kill -0 "$EN_PID" 2>/dev/null || kill -0 "$ZH_PID" 2>/dev/null; do
    sleep 60
done
echo "[chain] both 560 runs exited at $(date)"

# 确认 560 版产物存在（merge 完成），最多再等 20 分钟
for name in en zh; do
    dir="${OUT_ROOT}/gemma4-12b-gui-lora-0901-${name}-max-soft-token-560-4epoch-unfused"
    deadline=$(( $(date +%s) + 1200 ))
    until ls -d "${dir}"/v*/checkpoint-225-merged >/dev/null 2>&1; do
        if (( $(date +%s) > deadline )); then
            echo "[chain] WARNING: ${name} 560 merged checkpoint not found after wait; continuing anyway"
            break
        fi
        sleep 30
    done
done

# 等显存释放
sleep 120

TS=$(date +%Y%m%d_%H%M%S)
cd "$SCRIPT_DIR"

launch() {
    local name=$1 gpus=$2 port=$3 gpu_tag=$4
    CUDA_VISIBLE_DEVICES="$gpus" NPROC_PER_NODE=2 TENSOR_MODEL_PARALLEL_SIZE=1 MASTER_PORT="$port" \
    GEMMA4_CONDA_ENV=/data/miniconda3/envs/ms-swift-gemma4 \
    GEMMA4_MODEL=/data/model/gemma-4-12B-it GEMMA4_MAX_SOFT_TOKENS=1120 NUM_TRAIN_EPOCHS=4 \
    GEMMA4_TRAIN_DATA="${DATA_ROOT}/sft_thinkanswer_${name}/train_ms_swift.jsonl" \
    GEMMA4_VAL_DATA="${DATA_ROOT}/sft_thinkanswer_${name}/val_ms_swift.jsonl" \
    GEMMA4_OUTPUT_DIR="${OUT_ROOT}/gemma4-12b-gui-lora-0901-${name}-max-soft-token-1120-4epoch-unfused" \
    GEMMA4_SWANLAB_EXP_NAME="gemma4-12b-gui-lora-0901-${name}-max-soft-token-1120-4epoch-unfused" \
    nohup bash gemma4_gui_lora_different_loss_unfused.sh \
        > "${LOG_DIR}/uber_lora_${name}_soft1120_4epoch_unfused_gpu${gpu_tag}_${TS}.log" 2>&1 &
    echo "[chain] launched ${name} 1120 run: PID $! log=${LOG_DIR}/uber_lora_${name}_soft1120_4epoch_unfused_gpu${gpu_tag}_${TS}.log"
}

launch en 4,5 29513 45
launch zh 6,7 29514 67
echo "[chain] done at $(date)"
