#!/usr/bin/env bash
# 旧机器（NFS 服务器）一键启动：4 个 unfused LoRA 训练并行，占满 8 卡。
#   GPU 0,1 -> EN  560   GPU 2,3 -> ZH  560
#   GPU 4,5 -> EN 1120   GPU 6,7 -> ZH 1120
# 拓扑与 0817 一致（每任务 2 卡、TP=1、DP=2），除 attention_backend=unfused 和
# max_soft_tokens 外，其余参数与 0831 EN/ZH 训练完全相同。
# 用法：bash launch_all4_unfused_8gpu.sh
set -euo pipefail

cd /data/uber_train/gemma4-training-code/ms-swift-text/examples/models/gemma4

# 1) 生成 unfused 版训练脚本（若不存在）
if [ ! -f gemma4_gui_lora_different_loss_unfused.sh ]; then
    sed 's/--attention_backend auto/--attention_backend unfused/' \
        gemma4_gui_lora_different_loss.sh > gemma4_gui_lora_different_loss_unfused.sh
    chmod +x gemma4_gui_lora_different_loss_unfused.sh
fi
grep -q "attention_backend unfused" gemma4_gui_lora_different_loss_unfused.sh || {
    echo "ERROR: unfused 脚本生成失败"; exit 1; }

# 2) 确认 EN/ZH 数据存在（不存在先跑 replace_system_zh_en.py）
DATA_ROOT=/data/uber_train/gemma4-training-code/uber_data/uber_data_0804
for n in en zh; do
    [ -f "${DATA_ROOT}/sft_thinkanswer_${n}/train_ms_swift.jsonl" ] || {
        echo "ERROR: 缺少 ${DATA_ROOT}/sft_thinkanswer_${n}/train_ms_swift.jsonl"
        echo "先运行: python ${DATA_ROOT}/replace_system_zh_en.py"; exit 1; }
done

MODEL=/data/lcx/foxx/train/models/gemma-4-12B-it   # 0817 同款模型路径
OUT_ROOT=/data/uber_train/gemma4-training-code/textoutput
LOG_DIR=/data/uber_train/gemma4-training-code/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

launch() {  # launch <en|zh> <soft_tokens> <gpus> <port>
    local name=$1 soft=$2 gpus=$3 port=$4
    local tag="${name}-max-soft-token-${soft}-4epoch-unfused"
    CUDA_VISIBLE_DEVICES="$gpus" NPROC_PER_NODE=2 TENSOR_MODEL_PARALLEL_SIZE=1 MASTER_PORT="$port" \
    GEMMA4_MODEL="$MODEL" GEMMA4_MAX_SOFT_TOKENS="$soft" NUM_TRAIN_EPOCHS=4 \
    GEMMA4_TRAIN_DATA="${DATA_ROOT}/sft_thinkanswer_${name}/train_ms_swift.jsonl" \
    GEMMA4_VAL_DATA="${DATA_ROOT}/sft_thinkanswer_${name}/val_ms_swift.jsonl" \
    GEMMA4_OUTPUT_DIR="${OUT_ROOT}/gemma4-12b-gui-lora-${tag}" \
    GEMMA4_SWANLAB_EXP_NAME="gemma4-12b-gui-lora-${tag}" \
    nohup bash gemma4_gui_lora_different_loss_unfused.sh \
        > "${LOG_DIR}/uber_lora_${name}_soft${soft}_unfused_gpu${gpus//,/}_${TS}.log" 2>&1 &
    echo "launched ${tag}: GPU ${gpus}, PID $!, log ${LOG_DIR}/uber_lora_${name}_soft${soft}_unfused_gpu${gpus//,/}_${TS}.log"
}

launch en  560  0,1 29511
launch zh  560  2,3 29512
launch en  1120 4,5 29513
launch zh  1120 6,7 29514

echo "全部启动。约 2 分钟后检查各日志确认 'attention_backend unfused' 和 '[Gemma4 plugin] ... max_soft_tokens=' 行。"
