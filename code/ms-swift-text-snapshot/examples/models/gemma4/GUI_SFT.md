# ms-swift Gemma 4 脚本说明与 GUI SFT 使用指南

本文档说明目录
`/data/lcx/foxx/train/ms-swift/examples/models/gemma4`
中各脚本的用途、执行引擎、模型类型、关键参数和适用场景，并给出本地
Gemma 4 12B GUI 能力微调的完整执行流程。

## 1. 目录概览

目录中的文件可以分为两组：

| 文件                        | 来源/性质         | 用途                                                   |
| --------------------------- | ----------------- | ------------------------------------------------------ |
| `train.sh`                | ms-swift 官方示例 | 使用 Transformers/DeepSpeed 对 Gemma 4 E2B 做 LoRA SFT |
| `gemma4_unified.sh`       | ms-swift 官方示例 | 使用 Megatron 对 Gemma 4 12B Unified 做全参数 SFT      |
| `mcore.sh`                | ms-swift 官方示例 | 使用 Megatron 对 Gemma 4 26B-A4B MoE 做全参数 SFT      |
| `diffusion_gemma.sh`      | ms-swift 官方示例 | 对 DiffusionGemma 26B-A4B 做 LoRA SFT 和推理           |
| `gemma4_gui_env.sh`       | 本项目新增        | 统一设置模型、数据、Conda、CUDA 和 SwanLab 环境        |
| `gemma4_gui_preflight.sh` | 本项目新增        | GUI 数据预检的 Shell 入口                              |
| `gemma4_gui_preflight.py` | 本项目新增        | 执行数据完整性、模板编码、loss mask 和长度检查         |
| `gemma4_gui_lora.sh`      | 本项目新增        | Gemma 4 12B GUI LoRA 的冒烟和正式训练入口              |
| `gemma4_gui_full.sh`      | 本项目新增        | Gemma 4 12B GUI 全参数 LLM 的冒烟和正式训练入口        |
| `GUI_SFT.md`              | 本项目新增        | 当前说明文档                                           |

各官方脚本针对的模型架构和训练方式不同，不能仅通过修改
`--model` 和 `--dataset` 就相互替换。

## 2. 官方示例脚本

### 2.1 `train.sh`

该脚本用于 Gemma 4 E2B-it：

```text
模型：google/gemma-4-E2B-it
训练入口：swift sft
微调方式：LoRA
并行方式：2 卡数据并行 + DeepSpeed ZeRO-2
示例数据：LaTeX OCR
```

`swift sft` 使用 Transformers 训练体系，主要参数如下：

- `NPROC_PER_NODE=2`：启动两个训练进程。
- `per_device_train_batch_size=1`：每张卡每步处理一个样本。
- `gradient_accumulation_steps=16`：累计 16 次梯度后更新参数。
- `tuner_type=lora`：只训练 LoRA 增量参数。
- `lora_rank=8`、`lora_alpha=32`：LoRA 的秩和缩放系数。
- `target_modules=all-linear`：在允许训练的线性层上添加 LoRA。
- `freeze_vit=true`、`freeze_aligner=true`：不更新视觉塔和视觉投影模块。
- `deepspeed=zero2`：使用 DeepSpeed ZeRO-2 切分优化器状态和梯度。
- `max_length=4096`：超过 4096 tokens 的样本默认会被删除。

脚本末尾给出了 LoRA 推理方式：

```bash
swift infer --adapters /path/to/checkpoint
```

它适合 E2B 模型，不是当前 12B Unified GUI 任务的直接入口。

### 2.2 `gemma4_unified.sh`

这是和当前基础模型架构最接近的官方示例：

```text
模型：google/gemma-4-12B-it
训练入口：megatron sft
微调方式：全参数
并行方式：4 卡 Tensor Parallel
官方显存标注：4 × 65 GiB
示例数据：LaTeX OCR
```

主要参数：

- `tuner_type=full`：更新语言模型的全部非冻结参数。
- `tensor_model_parallel_size=4`：单个模型被切分到四张卡。
- `micro_batch_size=16`：每个数据并行组每步处理 16 个样本。
- `global_batch_size=16`：全局 batch 为 16。
- `recompute_*`：使用完整 activation recomputation 降低显存占用。
- `freeze_llm=false`：更新语言模型参数。
- `freeze_vit=true`、`freeze_aligner=true`：冻结视觉相关模块。
- `sequence_parallel=true`：在 TP 组内对部分激活做序列并行。
- `attention_backend=unfused`：使用非融合注意力实现。
- `padding_free=false`：关闭 padding-free；这是官方 Gemma4 示例的选择。
- `group_by_length=true`：把长度相近的样本分到同一 batch，减少 padding。
- `no_save_optim=true`、`no_save_rng=true`：不保存优化器和 RNG 状态。

该脚本不能只改路径就用于当前 GUI 数据，原因包括：

1. 它是全参数训练，而当前方案优先使用 LoRA。
2. `micro_batch_size=16` 按 OCR 短样本设计，不适合 9K 左右的 GUI 上下文。
3. 它从训练集按 1% 临时拆验证集，而当前已有 episode 隔离的独立验证集。
4. `max_length=4096` 会删除大量 GUI 轨迹。
5. GUI 图片使用相对路径，必须配置 `ROOT_IMAGE_DIR`。
6. 它没有训练前的数据、图片、loss mask 和显存安全检查。

`gemma4_gui_lora.sh` 保留了该脚本对 Gemma4 有意义的 Megatron 参数，同时替换了
数据、长度、batch、LoRA、验证集和安全控制。

### 2.3 `mcore.sh`

该脚本面向 Gemma 4 26B-A4B-it MoE：

```text
模型：google/gemma-4-26B-A4B-it
训练入口：megatron sft
微调方式：全参数
硬件：8 × 80 GiB
```

除常规 Tensor Parallel 外，它还使用：

- `tensor_model_parallel_size=2`：TP=2。
- `expert_model_parallel_size=4`：将 MoE experts 分布到四个 rank。
- `pipeline_model_parallel_size=2`：模型按层分为两个流水线 stage。
- `moe_permute_fusion=true`：融合 MoE token 排列操作。
- `moe_grouped_gemm=true`：通过 grouped GEMM 提升 expert 计算效率。
- `moe_shared_expert_overlap=true`：重叠 shared expert 与其他计算。
- `moe_aux_loss_coeff=1e-6`：使用较小的 expert 负载均衡辅助损失。

当前本地模型是 Dense 12B Unified，不是 26B-A4B MoE，因此不应复制这些
EP、PP 或 MoE 参数。

### 2.4 `diffusion_gemma.sh`

该脚本是 DiffusionGemma 训练演示：

```text
模型：google/diffusiongemma-26B-A4B-it
训练入口：swift sft
微调方式：LoRA
硬件标注：2 × 60 GiB
示例任务：Sudoku
```

脚本顶部强调了三个特殊限制：

1. 当前只支持 `per_device_train_batch_size=1`。
2. 单个回答必须短于模型配置中的 `canvas_length`。
3. 必须设置 `gradient_checkpointing=false`，否则 decoder 使用
   `DynamicCache` 时会在反向重计算阶段出错。

DiffusionGemma 的生成和 loss 机制与普通 Gemma 4 自回归模型不同，该脚本不能用于
当前 GUI SFT。

## 3. GUI 专用环境脚本

### 3.1 `gemma4_gui_env.sh`

这是其他 GUI 脚本共用的环境初始化文件。它通过 `source` 被加载，不单独启动训练。

默认路径：

| 变量                       | 默认值                                             |
| -------------------------- | -------------------------------------------------- |
| `GEMMA4_CONDA_ENV`       | `/data/lcx/miniconda3/envs/ms-swift-gemma4`       |
| `GEMMA4_MODEL`           | `/data/lcx/foxx/train/models/gemma-4-12B-it`      |
| `GEMMA4_DATA_ROOT`       | `/data/lcx/foxx/train/dataset`                    |
| `GEMMA4_TRAIN_DATA`      | `${GEMMA4_DATA_ROOT}/0725/train_ms_swift.jsonl`  |
| `GEMMA4_VAL_DATA`        | `${GEMMA4_DATA_ROOT}/0725/val_ms_swift.jsonl`    |
| `GEMMA4_OUTPUT_DIR`      | `/data/lcx/foxx/train/output/gemma4-12b-gui-lora` |
| `GEMMA4_FULL_OUTPUT_DIR` | `/data/lcx/foxx/train/output/gemma4-12b-gui-full` |
| `GEMMA4_SWANLAB_PROJECT` | `gemma4-gui-sft`                                  |
| `SWANLAB_API_KEY`        | 已内置默认值，也可通过同名环境变量覆盖             |

脚本的工作：

1. 检查 Python、模型配置、模型权重、训练集和验证集是否存在。
2. 将指定 Conda 环境的 `bin` 放到 `PATH` 最前面。
3. 设置 `CUDA_HOME`。
4. 设置 `ROOT_IMAGE_DIR=/data/lcx/foxx/train/dataset`。
5. 收集 pip 安装的 NVIDIA 动态库目录并写入 `LD_LIBRARY_PATH`。
6. 默认启用 PyTorch expandable segments，降低显存碎片风险。
7. 设置 SwanLab API Key 和默认项目名。

`ROOT_IMAGE_DIR` 非常重要。JSONL 中的图片类似：

```json
{"images": ["images/claude-opus-4-8-max/.../steps/001.png"]}
```

这是相对路径。ms-swift 会把它和 `ROOT_IMAGE_DIR` 组合成：

```text
/data/lcx/foxx/train/dataset/images/.../steps/001.png
```

如需临时覆盖路径，可以在命令前设置环境变量：

```bash
GEMMA4_MODEL=/new/model/path \
GEMMA4_OUTPUT_DIR=/new/output/path \
GEMMA4_FULL_OUTPUT_DIR=/new/full-output/path \
GEMMA4_SWANLAB_PROJECT=/new/swanlab-project \
source examples/models/gemma4/gemma4_gui_env.sh
```

## 4. GUI 数据预检脚本

### 4.1 `gemma4_gui_preflight.sh`

这是预检的 Shell 包装器。它依次：

1. 加载 `gemma4_gui_env.sh`。
2. 切换到 ms-swift 仓库根目录。
3. 使用指定 Conda 环境执行 `gemma4_gui_preflight.py`。
4. 将用户传入的额外命令行参数原样转发给 Python。

推荐命令：

```bash
cd /data/lcx/foxx/train/ms-swift

./examples/models/gemma4/gemma4_gui_preflight.sh \
  --report /data/lcx/foxx/train/output/gemma4-12b-gui-lora/preflight.json
```

### 4.2 `gemma4_gui_preflight.py`

该程序分为结构检查和真实模板编码检查。

#### 结构检查

逐条验证：

- JSONL 能否正常解析。
- `messages` 和 `images` 是否为非空列表。
- 是否存在 `id` 和 `episode_id`。
- message role 是否属于 ms-swift 支持的角色。
- content 是否为字符串。
- 是否只有最后一个 assistant message 设置了 `loss=true`。
- 图片相对路径解析后是否真实存在。
- 每个 split 内是否存在重复 ID。
- 训练集与验证集之间是否存在 ID、episode 或图片泄漏。

这保证同一 GUI 轨迹不会同时出现在训练集和验证集中。

#### 模板编码检查

程序使用当前 ms-swift 的官方接口：

```python
processor = get_processor(model_path, download_model=False)
template = get_template(processor, ...)
template.set_mode("train")
encoded = template.encode(row)
```

这会真实执行 Gemma4 tokenizer、图片预处理和聊天模板，而不是按字符数估算。
每个样本必须满足：

- `input_ids` 与 `labels` 长度一致。
- 至少存在一个受监督 token。
- 至少存在一个多模态 token。
- 图片可以被 Gemma4 processor 正常处理。

#### 长度推荐

默认目标是覆盖 99% 样本：

1. 汇总训练集和验证集的真实 token 长度。
2. 计算指定 coverage 对应的分位点。
3. 向上取整到 1024 的整数倍。
4. 最低使用 4096，默认上限为 16384。
5. 如果上限仍达不到 coverage，则返回错误并要求压缩上下文。

已完成的全量预检结果为：

| 指标        | 训练集 | 验证集 |
| ----------- | -----: | -----: |
| 样本数      |   2188 |    498 |
| P50 tokens  |   5680 |   5699 |
| P95 tokens  |   7934 |   8174 |
| P99 tokens  |   8763 |   9002 |
| 最大 tokens |   9575 |   9823 |

最终推荐：

```text
recommended_max_length = 9216
实际覆盖率 = 99.8138%
needs_context_compaction = false
```

报告保存于：

```text
/data/lcx/foxx/train/output/gemma4-12b-gui-lora/preflight.json
```

可用参数：

| 参数                   | 含义                            | 默认值                 |
| ---------------------- | ------------------------------- | ---------------------- |
| `--model`            | 模型目录                        | `$GEMMA4_MODEL`      |
| `--train-data`       | 训练 JSONL                      | `$GEMMA4_TRAIN_DATA` |
| `--val-data`         | 验证 JSONL                      | `$GEMMA4_VAL_DATA`   |
| `--image-root`       | 图片根目录                      | `$ROOT_IMAGE_DIR`    |
| `--template-samples` | 每个 split 编码多少条；0 为全部 | `0`                  |
| `--coverage`         | 目标长度覆盖率                  | `0.99`               |
| `--length-cap`       | 推荐长度上限                    | `16384`              |
| `--report`           | JSON 报告输出路径               | 不保存文件             |

快速抽检可以执行：

```bash
./examples/models/gemma4/gemma4_gui_preflight.sh \
  --template-samples 10
```

快速抽检只能验证管线，正式训练仍应使用全量预检报告。

## 5. GUI LoRA 训练脚本

### 5.1 `gemma4_gui_lora.sh` 的执行流程

脚本启动时依次执行：

1. 加载统一环境。
2. 验证 `RUN_MODE`。
3. 验证必须提供四个 GPU ID。
4. 通过 `nvidia-smi` 检查每张卡的空闲显存。
5. 读取预检报告中的 `recommended_max_length`。
6. 根据 smoke/full 模式设置数据量和训练周期。
7. 打印最终模型、数据、GPU、长度和输出目录。
8. 使用 `megatron sft` 启动四进程训练。

### 5.2 显存安全检查

默认要求每张被选中的 GPU 至少空闲：

```text
50000 MiB
```

如果 GPU 正在运行其他任务，脚本会拒绝启动，不会抢占或终止已有进程。

可以调整阈值：

```bash
MIN_FREE_MIB=60000 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./examples/models/gemma4/gemma4_gui_lora.sh
```

`SKIP_GPU_CHECK=1` 只应用于明确知道运行环境状况的调试场景，不建议在共享机器上使用。

### 5.3 smoke 与 full 模式

| 配置         |   smoke，默认 |       full |
| ------------ | ------------: | ---------: |
| 训练样本     |            64 |       2188 |
| 验证样本     |            32 |        498 |
| 训练长度     |  3 iterations |    1 epoch |
| 数据缓存     |          关闭 |       开启 |
| eval 间隔    |             1 |         50 |
| save 间隔    |             2 |         50 |
| logging 间隔 |             1 |          5 |
| 输出目录     | `.../smoke` | 主输出目录 |

smoke 用于检查：

- 四卡和 NCCL 是否正常。
- 模型能否由 MCore-Bridge 转换并加载。
- 数据能否进入训练循环。
- loss 是否为有限值。
- 反向传播是否 OOM。
- checkpoint 是否可以保存和重新加载。

默认执行 smoke：

```bash
cd /data/lcx/foxx/train/ms-swift

CUDA_VISIBLE_DEVICES=0,1,2,3 \
./examples/models/gemma4/gemma4_gui_lora.sh
```

smoke 通过后执行完整一轮：

```bash
RUN_MODE=full \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./examples/models/gemma4/gemma4_gui_lora.sh
```

也可以选择另外四张空闲 GPU：

```bash
RUN_MODE=full \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
./examples/models/gemma4/gemma4_gui_lora.sh
```

### 5.4 LoRA 参数

```text
tuner_type       = lora
lora_rank        = 8
lora_alpha       = 32
lora_dropout     = 0.05
target_modules   = all-linear
freeze_llm       = false
freeze_vit       = true
freeze_aligner   = true
```

含义：

- 在语言模型的线性层添加 LoRA。
- 不冻结 LLM，否则语言模型不会添加 LoRA。
- Gemma 4 12B Unified 是 encoder-free 架构，没有常规独立视觉塔。
- `freeze_aligner=true` 会冻结 `embed_vision/embed_audio` 投影。
- 第一阶段主要让语言模型学习“截图状态 + 历史操作 → 下一动作”的映射。

如果后续确认视觉定位能力受限，可以单独设计解冻视觉投影的对照实验，不应直接在首轮
修改当前脚本。

### 5.5 并行和 batch

```text
NPROC_PER_NODE            = 4
tensor_model_parallel_size = 4
micro_batch_size           = 1
global_batch_size          = 16
sequence_parallel          = true
```

四个进程全部属于同一个 TP 组，因此数据并行大小为 1。Megatron 会通过梯度累积把
micro batch 1 扩展到 global batch 16。

`micro_batch_size=1` 是为了适应最长约 9K tokens、最多三张截图的 GUI 样本。

### 5.6 显存相关参数

- `recompute_granularity=full`：对 Transformer 层做完整重计算。
- `recompute_method=uniform`：均匀选择重计算层。
- `recompute_num_layers=1`：以一层为重计算单元。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：减少显存碎片。
- `gradient_accumulation_fusion=false`：当前环境没有安装可选 Apex，按官方文档关闭对应融合。
- `cross_entropy_loss_fusion=true`：保留官方 Gemma4 脚本的交叉熵融合。

### 5.7 数据和模板参数

- `dataset`：训练 JSONL。
- `val_dataset`：使用已有独立验证集。
- `split_dataset_ratio=0`：不再从训练集随机拆验证数据。
- `ROOT_IMAGE_DIR`：解析相对图片路径。
- `add_non_thinking_prefix=true`：使用 Gemma4 非原生 thinking 前缀路径。
- `loss_scale=ignore_empty_think`：忽略空 thinking 部分的损失。
- `max_length=9216`：从预检报告自动读取，而非硬编码。
- `group_by_length=true`：减少长短样本混合造成的 padding。
- `padding_free=false`、`attention_backend=unfused`：与官方 12B Unified 示例保持一致。

当前数据保留原始文本输出协议：

```text
<think>...</think><answer>do(name="...", arguments={...})</answer>
```

它没有被转换成 Gemma 原生 tool call。训练目标是复现现有 GUI action 协议。

### 5.8 优化器和 checkpoint 参数

```text
lr                    = 1e-4
lr_warmup_fraction    = 0.05
min_lr                = 1e-5
save_safetensors      = true
save_total_limit      = 2
no_save_optim         = true
no_save_rng           = true
```

LoRA 使用比全参数训练更高的学习率。`no_save_optim=true` 和 `no_save_rng=true`
可以减少 checkpoint 大小，但意味着 checkpoint 主要用于加载 adapter、推理或重新开始
finetune，不能恢复完全一致的优化器/RNG 训练状态。

## 6. GUI 全参数训练脚本

### 6.1 `gemma4_gui_full.sh` 的含义

该脚本以官方 `gemma4_unified.sh` 为基线，使用：

```text
tuner_type      = full
freeze_llm      = false
freeze_vit      = true
freeze_aligner  = true
```

这里的 full 与官方定义一致：LLM 不使用 LoRA，所有未冻结的 LLM 参数都参与训练；
ViT 和多模态 aligner 默认仍被冻结。因此它不是默认解冻模型内每一个组件。

如确实要训练视觉组件，可以显式设置：

```bash
FREEZE_VIT=false \
FREEZE_ALIGNER=false \
bash examples/models/gemma4/gemma4_gui_full.sh
```

当前数据只有 174 条训练轨迹，直接解冻视觉模块更容易过拟合，也会增加显存压力。
第一轮全参数实验建议保持官方默认值。

### 6.2 与 LoRA 脚本的主要区别

| 配置             |                    LoRA |                  全参数 |
| ---------------- | ----------------------: | ----------------------: |
| `tuner_type`   |                `lora` |                `full` |
| LLM 更新范围     |            LoRA adapter |     全部未冻结 LLM 参数 |
| 学习率           |                `1e-4` |                `1e-5` |
| 最低学习率       |                `1e-5` |                `1e-6` |
| 默认最低空闲显存 |            50000 MiB/卡 |            70000 MiB/卡 |
| 输出目录         | `gemma4-12b-gui-lora` | `gemma4-12b-gui-full` |
| checkpoint       |          较小的 adapter |   约 12B 参数的完整权重 |

两者共同使用：

```text
4 GPUs
tensor_model_parallel_size = 4
micro_batch_size = 1
global_batch_size = 16
max_length = 9216
full activation recomputation
sequence_parallel = true
attention_backend = unfused
```

官方示例在 4096 长度下使用 `micro_batch_size=16`。当前 GUI 数据最长接近 10K，
因此全参数脚本将 micro batch 降为 1，通过 16 次梯度累积保持 global batch 16。

### 6.3 smoke 与正式全参数训练

默认先运行 3 iterations 的全参数 smoke：

```bash
cd /data/lcx/foxx/train/ms-swift

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/models/gemma4/gemma4_gui_full.sh
```

smoke 仍会在结束时保存完整模型 checkpoint，磁盘占用和保存时间会明显大于 LoRA。
确认 loss、显存峰值、NCCL 和 checkpoint 均正常后，再运行一轮完整数据：

```bash
RUN_MODE=full \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/models/gemma4/gemma4_gui_full.sh
```

正式模式使用全部 2188 条训练样本和 498 条验证样本，输出到：

```text
/data/lcx/foxx/train/output/gemma4-12b-gui-full
```

脚本与官方示例一致，设置 `no_save_optim=true` 和 `no_save_rng=true`，保存完整
safetensors 权重但不保存优化器和 RNG 状态。这可以显著减少 checkpoint 体积，但不能
从中恢复完全一致的优化器训练状态。

### 6.4 显存策略

官方 `gemma4_unified.sh` 标注 12B 全参数训练需要 4 × 65 GiB，但其上下文长度为
4096。GUI 脚本默认要求每张卡至少空闲 70000 MiB，并使用 micro batch 1、完整
activation recomputation、TP=4 和 sequence parallel。

这是一套合理的 A800 80GB 起始配置，但真实峰值仍需通过 smoke 实测。发生 OOM 时，
不要直接开始完整训练；应先检查实际样本长度、其他进程占用和显存碎片。

## 7. SwanLab 训练监控

### 7.1 启用方式

LoRA 和全参数脚本都启用了：

```text
report_to = tensorboard swanlab
```

训练指标会同时写入本地 TensorBoard 和 SwanLab 云端。SwanLab 的本地运行文件保存在
对应训练输出目录的 `swanlab` 子目录。

当前环境安装：

```text
swanlab       = 0.9.1
nvidia-ml-py  = 13.610.43
```

认证使用 ms-swift 官方支持的 `SWANLAB_API_KEY` 环境变量。公共环境脚本已经设置默认
API Key；如需切换账户，可以在启动前覆盖：

```bash
SWANLAB_API_KEY='new-api-key' \
bash examples/models/gemma4/gemma4_gui_lora.sh
```

脚本不会把 API Key 作为命令行参数，也不会将其打印到训练日志。

### 7.2 项目和实验命名

默认 SwanLab 项目是 `gemma4-gui-sft`。实验名根据训练方式和运行模式自动区分：

| 脚本 | 运行模式 | 实验名 |
| --- | --- | --- |
| `gemma4_gui_lora.sh` | `smoke` | `gemma4-12b-gui-lora-smoke` |
| `gemma4_gui_lora.sh` | `full` | `gemma4-12b-gui-lora-full` |
| `gemma4_gui_full.sh` | `smoke` | `gemma4-12b-gui-full-smoke` |
| `gemma4_gui_full.sh` | `full` | `gemma4-12b-gui-full-full` |

可以覆盖项目名或实验名：

```bash
GEMMA4_SWANLAB_PROJECT=my-project \
GEMMA4_SWANLAB_EXP_NAME=my-experiment \
bash examples/models/gemma4/gemma4_gui_lora.sh
```

不要用普通字符串设置 `SWANLAB_PROJECT`。SwanLab 0.9.1 自己使用该环境变量作为
Pydantic JSON 配置项，普通字符串会导致导入失败。本项目使用
`GEMMA4_SWANLAB_PROJECT` 避免命名冲突，再通过 `--swanlab_project` 传给
ms-swift。

### 7.3 网络要求

SwanLab 云端模式需要训练节点能够访问 SwanLab 服务。如果外网连接失败，SwanLab
回调可能在训练初始化阶段报错。可以临时只保留 TensorBoard：

```bash
bash examples/models/gemma4/gemma4_gui_lora.sh \
  --report_to tensorboard
```

附加参数位于脚本参数末尾，会覆盖脚本内的默认 `report_to`。

## 8. Conda 环境

环境位置：

```text
/data/lcx/miniconda3/envs/ms-swift-gemma4
```

已安装的核心版本：

| 组件               | 版本                 |
| ------------------ | -------------------- |
| Python             | 3.12                 |
| PyTorch            | 2.8.0+cu128          |
| Transformers       | 5.12.1               |
| ms-swift           | 4.5.0.dev0，editable |
| Megatron-Core      | 0.18.0               |
| MCore-Bridge       | 1.6.0                |
| Transformer Engine | 2.16.0               |
| FlashAttention     | 2.8.3                |
| SwanLab            | 0.9.1                |

可以显式激活：

```bash
conda activate /data/lcx/miniconda3/envs/ms-swift-gemma4
```

训练脚本本身会把该环境加入 `PATH`，通常不依赖调用前已经激活哪个 Conda 环境。

## 9. 推荐执行顺序

### 第一步：确认 GPU

```bash
nvidia-smi
```

LoRA 选择四张空闲显存不少于 50GB 的 A800 80GB；全参数训练默认要求每张卡至少
空闲 70GB。

### 第二步：运行或复核预检

```bash
cd /data/lcx/foxx/train/ms-swift

./examples/models/gemma4/gemma4_gui_preflight.sh \
  --report /data/lcx/foxx/train/output/gemma4-12b-gui-lora/preflight.json
```

如果模型、数据或模板发生改变，必须重新生成预检报告。

### 第三步：选择训练方式并运行 smoke

推荐先用 LoRA 验证数据和任务是否有效：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./examples/models/gemma4/gemma4_gui_lora.sh
```

全参数 smoke 使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./examples/models/gemma4/gemma4_gui_full.sh
```

检查日志中的模型转换、可训练参数量、第一步 loss、峰值显存和 checkpoint。

### 第四步：完整训练

LoRA：

```bash
RUN_MODE=full \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./examples/models/gemma4/gemma4_gui_lora.sh
```

全参数：

```bash
RUN_MODE=full \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./examples/models/gemma4/gemma4_gui_full.sh
```

### 第五步：推理验证

LoRA checkpoint 应通过 `--adapters` 加载，基础模型仍传给 `--model`：

```bash
source examples/models/gemma4/gemma4_gui_env.sh

CUDA_VISIBLE_DEVICES=0 swift infer \
  --model "${GEMMA4_MODEL}" \
  --adapters /path/to/checkpoint \
  --dataset "${GEMMA4_VAL_DATA}" \
  --load_data_args true \
  --temperature 0 \
  --max_new_tokens 2048 \
  --result_path \
    /data/lcx/foxx/train/output/gemma4-12b-gui-lora/validation_predictions.jsonl
```

全参数 checkpoint 不使用 `--adapters`，直接作为模型加载：

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
  --model /data/lcx/foxx/train/output/gemma4-12b-gui-full/vx-xxx/checkpoint-xxx \
  --dataset "${GEMMA4_VAL_DATA}" \
  --load_data_args true \
  --temperature 0 \
  --max_new_tokens 2048
```

建议比较基础模型与 LoRA checkpoint 的：

- 输出协议可解析率。
- action name 准确率。
- 参数 exact match。
- 坐标动作容差准确率。
- 完整动作 exact match。
- validation loss。

## 10. 常见问题

### 报错“Missing preflight report”

先执行全量预检，或通过 `PREFLIGHT_REPORT` 指向已有报告：

```bash
PREFLIGHT_REPORT=/path/to/preflight.json \
./examples/models/gemma4/gemma4_gui_lora.sh
```

全参数脚本同样支持 `PREFLIGHT_REPORT`。

### 报错 GPU 空闲显存不足

说明目标 GPU 上存在其他任务。应等待任务结束或选择另外四张卡，不要直接跳过检查。

### `nvidia-smi` 在某个环境不可用

必须从实际挂载 GPU 设备的宿主机终端执行训练。仅存在 CUDA 动态库并不代表当前容器或
沙箱拥有 GPU 设备访问权限。

### 图片找不到

确认：

```text
ROOT_IMAGE_DIR=/data/lcx/foxx/train/dataset
```

并检查 JSONL 中的相对路径在该目录下是否存在。

### 为什么不是 `max_length=4096`

全量 Gemma4 模板编码显示 P50 已约为 5.7K tokens，4096 会删除大部分样本。
当前使用 9216，可覆盖 99.81% 数据。

### 为什么先用 LoRA

训练集只有 174 条训练轨迹、2188 个动作样本。LoRA 更适合先验证数据质量、输出协议和
GUI 能力增益，同时显著降低显存、checkpoint 和破坏基础能力的风险。

这不代表不能进行全参数训练。`gemma4_gui_full.sh` 已提供对应入口；建议把 LoRA
作为数据与训练链路基线，再比较全参数训练的增益、遗忘和成本。

### 这是否属于 GKD

不是在线 teacher-logit GKD。Claude 已经提前生成了目标回答，当前训练直接对这些回答做
SFT，因此属于离线 response distillation。
