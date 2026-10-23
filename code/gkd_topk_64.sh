#!/bin/bash
# ============================================================================
# 每组: 教师健康检查 -> 训练 -> verify_completions.py 自检 -> 记录摘要
# 用法: setsid nohup bash run_formal_ablation.sh > /dev/null 2>&1 < /dev/null &
# 日志: logs/formal_ablation_MMDD.log (进度看这个文件)
# ============================================================================
set -u
TASK=/data/chenzhaoqi/opd_starbucks_31to12
STUDENT="/data/zangqian/code/gemma4-training-code/sft_outputs/gemma4-12b-starbucks-v5-gui-lora-merged-clean-newdata/v0-20261009-212658/checkpoint-702-merged"
LOG=$TASK/logs/reverseKL_supervise_PSt_$(date +%m%d).log
cd "$TASK" || exit 1
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

run_one() {
  name=$1; steps=$2; lr=$3; temp=$4
  say "=== START $name (steps=$steps lr=$lr temp=$temp) ==="
  #-s (或 --silent)：静默模式。这会告诉 curl 不要显示进度条、错误信息或其他无关的运行状态，保持输出干净。
  #-o /dev/null (或 --output /dev/null)：这会将服务器返回的实际响应体（比如 HTML 页面或 JSON 数据）重定向，即直接丢弃。
  #-w '%{http_code}' (或 --write-out '%{http_code}')：这是这行命令的核心。它告诉 curl 在请求完成后，按照指定的格式输出额外的信息。%{http_code} 表示提取并输出本次请求的 HTTP 状态码
  #-m 5 (或 --max-time 5)：设置整个请求的最大超时时间为 5 秒。如果服务卡死或网络不通，curl 不会无限期等待，而是会在 5 秒后强制放弃，防止脚本一直挂起。
  health=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8021/health)
  if [ "$health" != "200" ]; then
    say "[abort] 教师服务不可用(health=$health), 序列终止 —— 已完成的组不受影响"
    exit 1
  fi
  cd "$TASK"
  STUDENT="$STUDENT" \
  TEACHER_SERVER=http://127.0.0.1:8022 \
  DATA="$TASK/data/opd_prompts_train.jsonl" \
  TAG="$name" MAX_STEPS="$steps" SAVE_STEPS=20 \
  OUTPUT_DIR="$TASK/output/$name" \
  MAX_LENGTH=8192 MAX_COMPLETION_LENGTH=1024 \
  ATTN_IMPL=eager GAS=8 NUM_GEN=1 DEEPSPEED=zero2 \
  CUDA_VISIBLE_DEVICES=4,5 NPROC_PER_NODE=2 \
  LR="$lr" \
  LORA_DROPOUT=0.0 \
  EXTRA_ARGS="--log_entropy true --temperature $temp --add_version false" \
    bash code/run_opd_smoke.sh >> "$LOG" 2>&1
  rc=$?
  say "=== END $name rc=$rc ==="
  if [ "$rc" -ne 0 ]; then
    say "[abort] $name 训练失败，停止后续流程"
    exit "$rc"
  fi
}


run_one reverseKL_supervise_PSt     100 5e-5 0.9

# setsid nohup bash gkd_topk_64.sh \
# > logs/formal_launcher_$(date +%m%d_%H%M%S).log 2>&1 < /dev/null &