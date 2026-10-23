#!/bin/bash
# ============================================================================
# 31B 教师 vLLM 服务部署 (方案B): swift deploy, 供 OPD 训练经 --teacher_model_server 取 logprobs
# 背景: 31B 是普通 gemma4 架构(要求 input_ids 预展开图像token), swift 本地教师路径只产
#       unified 紧凑编码 → 架构不匹配(见 v7 判决)。vLLM 引擎内部自行做图像token展开,
#       服务化后教师用自己的 processor+引擎, 绕开该限制。
#
# 用法 (gpu-117):
#   cd /data/liurongbo/tasks/opd_uber_31to12
#   setsid nohup bash code/deploy_teacher_31b.sh > logs/deploy_teacher_1009.log 2>&1 < /dev/null &
# 就绪判据: curl -s http://127.0.0.1:8021/health 返回 200
# ============================================================================
set -euo pipefail

TASK_DIR=/data/chenzhaoqi/opd_starbucks_31to12
MODEL="${MODEL:-/data/luchengxi/train/output/gemma4-31b-starbucks-v5-gui-lora-en/v1-20261012-141651/checkpoint-702-merged}"   # 他人目录, 只读
PORT="${PORT:-8021}"
GPU="${GPU:-6,7}"                       # TP2, 31B bf16 ≈62G → 每卡31G权重
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-16384}"             # 与训练 max_length 一致
VLLM_UTIL="${VLLM_UTIL:-0.85}"

ENV_DIR=/data/miniconda3/envs/swift-opd-gemma4
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$TASK_DIR/code/ms-swift-text-snapshot${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$TASK_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export MODELSCOPE_CACHE="$TASK_DIR/hf_cache/modelscope"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

for p in "$MODEL/config.json" "$TASK_DIR/code/ms-swift-text-snapshot/swift/cli/deploy.py"; do
    [ -e "$p" ] || { echo "[预检失败] $p 不存在"; exit 1; }
done

echo "=== 部署 31B 教师 $(date '+%F %T') ==="
echo "model=$MODEL GPU=$GPU TP=$TP port=$PORT max_len=$MAX_LEN util=$VLLM_UTIL"

cd "$TASK_DIR"
CUDA_VISIBLE_DEVICES="$GPU" "$ENV_DIR/bin/python" -c "from swift.cli.main import cli_main; cli_main()" deploy \
    --model "$MODEL" \
    --template gemma4 \
    --served_model_name teacher-31b \
    --host 127.0.0.1 \
    --port "$PORT" \
    --infer_backend vllm \
    --vllm_tensor_parallel_size "$TP" \
    --vllm_max_model_len "$MAX_LEN" \
    --vllm_gpu_memory_utilization "$VLLM_UTIL" \
    --vllm_limit_mm_per_prompt '{"image":10}' \
    --template_backend swift \
    --max_logprobs 64 \
    --torch_dtype bfloat16 \
    --verbose
# 建议运行指令
# setsid nohup bash code/deploy_teacher_31b.sh \
# > logs/deploy_teacher_$(date +%m%d_%H%M%S).log 2>&1 < /dev/null &
# 检查服务，返回200即为部署成功
# curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8021/health