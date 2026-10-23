#!/bin/bash
# ============================================================================
# OPD 冒烟训练: 12B base 学生 + 31B base 教师 (Uber 多模态, gemma4)
# 目的: 只验证链路 (数据加载/rollout/教师打分/loss/checkpoint/completions落盘),
#       不追求训练效果 — 教师/学生换 base 模型的信号无参考意义, 属预期。
#
# 用法 (gpu-182, 卡 0-3):
#   cd /data/liurongbo/tasks/opd_uber_31to12
#   setsid nohup bash code/run_opd_smoke.sh > logs/smoke_v1_$(date +%m%d_%H%M).log 2>&1 < /dev/null &
#   tail -f logs/smoke_v1_*.log
#
# 可覆盖变量 (正式训练时只换这两个模型路径即可):
#   STUDENT / TEACHER / DATA / TAG / MAX_STEPS / SAVE_STEPS / LR / BS / GAS /
#   MAX_LENGTH / MAX_COMPLETION_LENGTH / DEEPSPEED / VLLM_UTIL / ATTN_IMPL /
#   CUDA_VISIBLE_DEVICES / NPROC_PER_NODE / OUTPUT_DIR / EXTRA_ARGS
#   显存不够时: EXTRA_ARGS="--offload_teacher_model true"
#
# 已知风险备注:
#  - 12B 学生是 gemma4_unified (hybrid 注意力)。zangqian 在他的 TRL 栈实证过
#    sdpa(mem_efficient) 对该结构静默算错 (eager 则长序列 OOM)。本脚本 ATTN_IMPL
#    默认 sdpa 仅用于链路冒烟; 正式训练前必须单独验证注意力正确性
#    (用他的 /data/zangqian/code/gemma4_rl/attn_probe 或接入其 triton 内核)。
#    31B 教师是普通 gemma4 结构, 不受此影响。
#  - vllm colocate 退出阶段可能 SIGABRT (已知无害, 产物完好, 见 opd HANDOFF)。
# ============================================================================
set -euo pipefail

TASK_DIR=/data/chenzhaoqi/opd_starbucks_31to12
STUDENT="${STUDENT:-/data/model/gemma-4-12B-it}"
TEACHER="${TEACHER:-/data/luchengxi/train/models/gemma-4-31B-it}"   # 他人目录, 只读
# 方案B: 设 TEACHER_SERVER=http://127.0.0.1:8021 则教师走远程服务(推荐, 31B普通gemma4架构
# 与 swift 本地教师路径不兼容, 见 logs/smoke_v7 判决), 此时 TEACHER 本地路径不生效
TEACHER_SERVER="${TEACHER_SERVER:-}"
DATA="${DATA:-$TASK_DIR/data/opd_prompts_smoke.jsonl}"
TAG="${TAG:-smoke-v1}"
MAX_STEPS="${MAX_STEPS:-12}"
SAVE_STEPS="${SAVE_STEPS:-6}"
MAX_LENGTH="${MAX_LENGTH:-16384}"          # 对齐 Uber SFT 线
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"  # SFT 软上限560, 翻倍防截断
LR="${LR:-5e-5}"                           # 对齐 opd_teacher_distill 配方
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
NUM_GEN="${NUM_GEN:-1}"
BS="${BS:-1}"                              # 多模态长序列, micro batch 压 1
GAS="${GAS:-2}"
DEEPSPEED="${DEEPSPEED:-zero3}"            # 31B 教师需 zero3 分片
VLLM_UTIL="${VLLM_UTIL:-0.4}"              # 12B 权重24G + KV, 0.4*80G=32G
VLLM_TP="${VLLM_TP:-1}"
ATTN_IMPL="${ATTN_IMPL:-eager}"             # 见上方风险备注; 可改 eager/flashinfer
OUTPUT_DIR="${OUTPUT_DIR:-$TASK_DIR/output/$TAG}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

ENV_DIR=/data/miniconda3/envs/swift-opd-gemma4
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$TASK_DIR/code/ms-swift-text-snapshot${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$TASK_DIR/hf_cache"        # 任务自包含, 不碰别人的缓存
export HF_HUB_OFFLINE=1                    # 强制 Hugging Face Hub 进入“离线模式”
export MODELSCOPE_CACHE="$TASK_DIR/hf_cache/modelscope"
export WANDB_MODE=disabled                 # 禁用 Weights & Biases (W&B) 的云端同步
export TOKENIZERS_PARALLELISM=false        # 禁用 Hugging Face tokenizers 库的 Rust 底层并行化机制
export MASTER_PORT="${MASTER_PORT:-29631}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"




# ---------------- 预检 ----------------
for p in "$STUDENT/config.json" "$DATA" \
         "$TASK_DIR/code/ms-swift-text-snapshot/swift/__init__.py" \
         "$ENV_DIR/bin/python"; do
    [ -e "$p" ] || { echo "[预检失败] $p 不存在"; exit 1; }
done
python3 - "$STUDENT" "$TEACHER" <<'PYEOF'
import json, sys
for name, path in [("学生", sys.argv[1]), ("教师", sys.argv[2])]:
    c = json.load(open(path + "/config.json"))
    print(f"[预检] {name}: {path} model_type={c.get('model_type')} arch={c.get('architectures', [None])[0]}")
PYEOF
"$ENV_DIR/bin/python" -c "import deepspeed, vllm, swift.rlhf_trainers.grpo_trainer" 2>/dev/null \
    || { echo "[预检失败] 环境导入异常"; exit 1; }
n_data=$(wc -l < "$DATA")
echo "=== OPD 冒烟配置 $(date '+%F %T') ==="
echo "学生: $STUDENT"
echo "教师: $TEACHER (只读)"
echo "数据: $DATA ($n_data 行)"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC=$NPROC_PER_NODE | deepspeed=$DEEPSPEED vllm_util=$VLLM_UTIL tp=$VLLM_TP attn=$ATTN_IMPL"
echo "训练: steps=$MAX_STEPS save=$SAVE_STEPS lr=$LR bs=$BS×$GAS×$NPROC_PER_NODE gen=$NUM_GEN max_len=$MAX_LENGTH completion=$MAX_COMPLETION_LENGTH"
echo "输出: $OUTPUT_DIR"
echo "--- 目标卡现状 ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# ---------------- 训练 ----------------
if [ -n "$TEACHER_SERVER" ]; then
    TEACHER_ARGS="--teacher_model_server $TEACHER_SERVER"
else
    TEACHER_ARGS="--teacher_model $TEACHER"
fi
cd "$TASK_DIR"
# --teacher_kl_coef控制教师信号在 Advantage 中的权重
# --rlhf_type gkd配合--gkd_logits_topk参数使用
"$ENV_DIR/bin/python" -c "from swift.cli.main import cli_main; cli_main()" rlhf \
    --rlhf_type gkd \
    --gkd_logits_topk 64 \
    --model "$STUDENT" \
    --template gemma4 \
    $TEACHER_ARGS \
    --teacher_kl_coef 1.0 \
    --tuner_type lora \
    --lora_rank 8 --lora_alpha 32 --lora_dropout "$LORA_DROPOUT" --target_modules all-linear \
    --use_vllm true --vllm_mode colocate \
    --vllm_gpu_memory_utilization "$VLLM_UTIL" \
    --vllm_tensor_parallel_size "$VLLM_TP" \
    --vllm_max_model_len "$MAX_LENGTH" \
    --sleep_level 1 \
    --dataset "$DATA" \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_generations "$NUM_GEN" \
    --per_device_train_batch_size "$BS" \
    --gradient_accumulation_steps "$GAS" \
    --learning_rate "$LR" \
    --max_steps "$MAX_STEPS" \
    --logging_steps 1 \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --max_length "$MAX_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --deepspeed "$DEEPSPEED" \
    --warmup_ratio 0.1 \
    --save_only_model true \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --attn_impl "$ATTN_IMPL" \
    --log_completions true \
    --output_dir "$OUTPUT_DIR" \
    --report_to tensorboard \
    --seed 42 \
    --beta 0.0 \
    --lmbda 1.0 \
    $EXTRA_ARGS

# ---------------- 收尾 ----------------
echo "=== 冒烟结束 $(date '+%F %T'), 产物清单 ==="
find "$OUTPUT_DIR" -maxdepth 3 \( -name "checkpoint-*" -o -name "completions.jsonl" \
      -o -name "logging.jsonl" -o -name "args.json" \) | sort
echo "--- 验证要点 ---"
echo "1) completions.jsonl 有内容且为 think/answer 文本:  head -c 500 $OUTPUT_DIR/v*/completions.jsonl"
echo "2) 日志含 teacher_kl 指标且非全零; 3) loss 无 NaN; 4) checkpoint 目录结构完整"
