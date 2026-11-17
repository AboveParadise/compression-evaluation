# 端侧大模型压缩评测工具

本项目用于在统一的单卡 RTX 4090 环境下，对 LLaMA3-8B 与 Qwen3-8B 的稀疏化、量化、精度、部署能力和推理资源消耗进行可复现评测。

项目包含三类工具：

1. **精度评测**：输出 MMLU、ARC、HellaSwag、PIQA、WinoGrande 等任务的准确率。
2. **端侧部署能力评测**：输出推理峰值显存和生成速度（tokens/s）。
3. **推理资源评测**：输出 checkpoint 物理存储、参数量、稀疏率以及 FLOPs/MACs。

> 当前维护分支为 `main`/`master` 的最新内容。实验必须使用本 README 中的仓库本地 `.llmcbench/` 工作目录，不再依赖 `/data`。

## 目录

- [1. 支持范围](#1-支持范围)
- [2. 标准硬件与软件环境](#2-标准硬件与软件环境)
- [3. 仓库本地工作目录](#3-仓库本地工作目录)
- [4. 从零开始：完整流程](#4-从零开始完整流程)
- [5. 模型来源与压缩方式](#5-模型来源与压缩方式)
- [6. 量化 bit 口径](#6-量化-bit-口径)
- [7. 数据集说明](#7-数据集说明)
- [8. 运行完整评测矩阵](#8-运行完整评测矩阵)
- [9. 单独运行某个评测工具](#9-单独运行某个评测工具)
- [10. 输出文件](#10-输出文件)
- [11. 常用控制参数](#11-常用控制参数)
- [12. 常见问题](#12-常见问题)
- [13. 结果使用边界](#13-结果使用边界)

## 1. 支持范围

### 1.1 模型

- `LLaMA3-8B`
- `Qwen3-8B`

### 1.2 压缩方法

- 剪枝：Wanda、SparseGPT
- 量化：GPTQ、AWQ、SmoothQuant、OmniQuant
- 对照组：Dense FP16

每个模型对应 1 个 Dense 基线和 6 个压缩模型，共 14 个评测槽位。

### 1.3 数据集

- MMLU：57 个子科目
- ARC-Challenge（同时缓存 ARC-Easy）
- HellaSwag
- PIQA
- WinoGrande XL

完整数据准备共生成 62 个可离线重载的数据配置。

## 2. 标准硬件与软件环境

### 2.1 推荐硬件

| 项目 | 要求 |
| --- | --- |
| GPU | 1 张 NVIDIA RTX 4090 24GB |
| CPU | 至少 8 核，推荐多核工作站 CPU |
| 内存 | 256GB |
| 存储 | NVMe SSD，建议至少 1TB 可用空间 |
| 供电/散热 | 稳定供电和良好散热，评测期间 GPU 独占 |

### 2.2 推荐系统

- Ubuntu 22.04 LTS 64-bit
- NVIDIA 驱动 550 或更高
- Python 3.10
- PyTorch 2.6.0 + CUDA 12.4 wheel
- cuDNN（随 PyTorch CUDA wheel 安装并在脚本中验证）
- Transformers 4.57.6
- bitsandbytes 0.50.1

无需额外安装完整 CUDA Toolkit；PyTorch cu124 wheel 已包含运行所需 CUDA Runtime 和 cuDNN。宿主机必须先安装可识别 RTX 4090 的 NVIDIA 驱动。

## 3. 仓库本地工作目录

环境、模型、数据和结果全部默认保存在当前代码仓库内：

```text
compression-evaluation/
├── .llmcbench/
│   ├── venv/                         # Python 3.10 虚拟环境
│   ├── models/
│   │   ├── Meta-Llama-3-8B/
│   │   ├── Qwen3-8B/
│   │   └── precompressed/            # 12 个压缩 checkpoint
│   ├── hf_cache/                     # Hugging Face 缓存
│   ├── datasets/
│   │   ├── snapshots/                # 五类数据集的本地快照
│   │   └── llmcbench_datasets.json   # 数据完整性清单
│   ├── calibration/
│   ├── prompts/
│   └── outputs/
│       └── llmcbench_edge_4090/
├── edge_benchmark/
├── scripts/
├── tools/
└── configs/
```

`.llmcbench/` 已加入 `.gitignore`，不会误提交模型、数据或结果。

如确实需要使用外置数据盘，可显式设置：

```bash
export LLMCBENCH_ROOT=/mnt/nvme/llmcbench
```

不设置时始终使用当前仓库的 `.llmcbench/`。

## 4. 从零开始：完整流程

### 4.1 安装系统依赖

```bash
sudo apt update
sudo apt install -y git git-lfs build-essential python3.10 python3.10-venv tmux
```

检查 GPU：

```bash
nvidia-smi
```

应看到 NVIDIA GeForce RTX 4090 和约 24GB 显存。

### 4.2 拉取代码

实验室网络推荐使用 Gitee：

```bash
git clone https://gitee.com/aboveparadise/compression-evaluation.git
cd compression-evaluation
```

GitHub 地址：

```text
https://github.com/AboveParadise/compression-evaluation
```

### 4.3 配置国内镜像

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
export PIP_DEFAULT_TIMEOUT=600
export PIP_RETRIES=20

# 已确认包含 torch/torchvision/torchaudio 的 Python 3.10 + cu124 wheel
export PYTORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu124
```

这些变量只对当前终端有效，不会永久修改系统 pip 配置。

### 4.4 安装并验证软件环境

先查看安装计划：

```bash
DRY_RUN=1 bash scripts/install_4090_environment.sh
```

正式安装：

```bash
bash scripts/install_4090_environment.sh
```

脚本默认创建仓库本地 venv：

```bash
source .llmcbench/venv/bin/activate
```

验证环境：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.get_device_name())"
```

预期类似：

```text
2.6.0+cu124 12.4 9xxxx NVIDIA GeForce RTX 4090
```

环境报告保存在：

```text
.llmcbench/outputs/llmcbench_edge_4090/edge_env_report.json
```

安装器对每个 pip 阶段强制使用 `constraints-cu124.txt`，防止可选依赖把 Torch 2.6.0 升级成未经验证的新版本。

### 4.5 准备数据集

```bash
source .llmcbench/venv/bin/activate
python scripts/prepare_4090_datasets.py
```

脚本行为：

1. 每个数据集仓库只下载一次完整 snapshot。
2. 从本地 snapshot 构建全部 MMLU 子科目和其他任务。
3. 开启离线模式重新加载所有配置。
4. 写入机器可读 manifest。

成功输出：

```text
{
  "ok": true,
  "entries": 62
}
```

数据 manifest：

```text
.llmcbench/datasets/llmcbench_datasets.json
```

只准备某一类数据时：

```bash
python scripts/prepare_4090_datasets.py --family piqa
python scripts/prepare_4090_datasets.py --family mmlu
```

支持的 `--family`：`mmlu`、`arc`、`hellaswag`、`piqa`、`winogrande`。

### 4.6 准备 Base 模型和压缩模型

先确保 GPU 空闲：

```bash
nvidia-smi
```

建议在 tmux 中运行：

```bash
tmux new -s llmcbench
source .llmcbench/venv/bin/activate
bash scripts/prepare_4090_models.sh 2>&1 | tee .llmcbench/outputs/prepare_models.log
```

模型准备可能持续数小时。下载失败时可重跑，完整 checkpoint 会自动跳过；不完整下载会移动到带时间戳的隔离目录，避免污染本地压缩输出。

成功结束时应看到：

```text
Quantization alignment verified: GPTQ/AWQ/OmniQuant=W4G128; SmoothQuant=W8A8
Prepared model config: .../edge_4090_prepared_models.json
```

生成的最终矩阵配置：

```text
.llmcbench/outputs/llmcbench_edge_4090/edge_4090_prepared_models.json
```

该配置已经写入所有 Base、压缩 checkpoint、数据、缓存、prompt、calibration 和输出的仓库内绝对路径。之后的评测不会再次压缩。

只打印模型准备计划：

```bash
DRY_RUN=1 bash scripts/prepare_4090_models.sh
```

只验证已有资产：

```bash
VERIFY_ONLY=1 bash scripts/prepare_4090_models.sh
```

强制重新生成某个已有输出前，建议先备份；全局强制模式为：

```bash
FORCE=1 bash scripts/prepare_4090_models.sh
```

## 5. 模型来源与压缩方式

### 5.1 Base 模型

| 模型 | 来源 | 获取方式 |
| --- | --- | --- |
| LLaMA3-8B | ModelScope `LLM-Research/Meta-Llama-3-8B` | 下载 |
| Qwen3-8B | ModelScope `Qwen/Qwen3-8B` | 下载 |

下载完成后会记录来源、revision 以及关键配置文件 SHA256。

### 5.2 压缩模型

| 模型 | 方法 | 获取方式 | 默认配置 |
| --- | --- | --- | --- |
| LLaMA3-8B | Wanda | 本地压缩 | 50% 非结构化、C4 128×2048、seed 0 |
| LLaMA3-8B | SparseGPT | 本地压缩 | 50%、完整二阶、block 128、damp 0.01 |
| LLaMA3-8B | GPTQ | 优先下载，失败后本地压缩 | W4G128 |
| LLaMA3-8B | AWQ | 优先下载，失败后本地压缩 | W4G128 |
| LLaMA3-8B | SmoothQuant | 本地压缩 | W8A8 工程适配 |
| LLaMA3-8B | OmniQuant | 本地压缩 | W4A16G128 工程适配 |
| Qwen3-8B | Wanda | 本地压缩 | 50% 非结构化、C4 128×2048、seed 0 |
| Qwen3-8B | SparseGPT | 本地压缩 | 50%、完整二阶、block 128、damp 0.01 |
| Qwen3-8B | GPTQ | 优先下载，失败后本地压缩 | W4G128 |
| Qwen3-8B | AWQ | 优先下载，失败后本地压缩 | W4G128 |
| Qwen3-8B | SmoothQuant | 本地压缩 | W8A8 工程适配 |
| Qwen3-8B | OmniQuant | 本地压缩 | W4A16G128 工程适配 |

默认下载源已固定 revision：

- LLaMA3 GPTQ：`Ksgk-fy/Meta-Llama-3-8B-4bit-GPTQ`
- LLaMA3 AWQ：`solidrust/Meta-Llama-3-8B-hf-AWQ`
- Qwen3 GPTQ：`kaitchup/Qwen3-8B-autoround-4bit-gptq`
- Qwen3 AWQ：`Qwen/Qwen3-8B-AWQ`

Wanda、SparseGPT 必须从当前统一 Base 生成。SmoothQuant、OmniQuant 不使用之前的 fake-quant 或错误底座公共 checkpoint，而是从同一 Base 生成可追溯工程 checkpoint。

### 5.3 为什么剪枝 buffer 放在 CPU

Wanda/SparseGPT 默认使用：

```text
pruning_buffer_device=cpu
```

这是 24GB RTX 4090 的显存安全配置。对于 8B 模型，`128 × 2048 × 4096`
的 FP16 校准隐藏状态单份约 2GiB，输入/输出双缓冲约 4GiB；FP16 模型权重
约 15–16GiB，SparseGPT 每层还需要约 0.9–1.1GiB Hessian 和临时矩阵。
若全部放在 GPU，峰值很容易超过 24GB。

CPU buffer 会增加 PCIe 传输并延长一次性的剪枝生成时间，但不会影响生成后
checkpoint 的精度评测、推理显存或 tokens/s。正式 4090 实验应保留 CPU
buffer；只有在 80GB 以上 GPU 上使用自定义压缩命令时，才建议将
`--pruning-buffer-device` 改为 `cuda:0`。

## 6. 量化 bit 口径

主矩阵采用：

| 方法 | 权重 bit | 激活 bit | group size |
| --- | ---: | ---: | ---: |
| GPTQ | 4 | 16 | 128 |
| AWQ | 4 | 16 | 128 |
| OmniQuant | 4 | 16 | 128 |
| SmoothQuant | 8 | 8 | per-channel/per-token |

SmoothQuant 保持标准 W8A8。把它强行改为 W4 weight-only 会变成“SmoothQuant 平滑 + 额外 INT4 权重量化器”的混合方法，不能再作为标准 SmoothQuant 报告。

因此本项目保证：

- GPTQ、AWQ、OmniQuant 在两个 Base 模型上统一 W4G128。
- SmoothQuant 在两个 Base 模型上统一 W8A8。
- 输出表同时记录真实 weight/activation bit、物理存储、显存和精度，避免把不同格式误标为相同 bit。
- checkpoint 若为 fake quant、bit 不符或 group size 不符，模型准备脚本直接失败。

## 7. 数据集说明

| 评测任务 | 本地 snapshot | 评测名称 |
| --- | --- | --- |
| MMLU | `cais/mmlu` | `mmlu` 及 57 个 `mmlu_*` 子任务 |
| ARC | `allenai/ai2_arc` | `arc_challenge`；同时缓存 ARC-Easy |
| HellaSwag | `Rowan/hellaswag` | `hellaswag` |
| PIQA | `baber/piqa` | `piqa` |
| WinoGrande | `allenai/winogrande` | `winogrande` / `winogrande_xl` |

最终精度矩阵默认任务：

```text
mmlu,arc_challenge,hellaswag,piqa,winogrande
```

MMLU 最终结果要求完整子科目覆盖，不能用单个 smoke 子任务代替。

## 8. 运行完整评测矩阵

### 8.1 设置配置路径

```bash
source .llmcbench/venv/bin/activate

export CONFIG="$PWD/.llmcbench/outputs/llmcbench_edge_4090/edge_4090_prepared_models.json"
```

### 8.2 运行非破坏性 dry-run

```bash
CONFIG="$CONFIG" \
DRY_RUN=1 \
RUN_PREWARM=0 \
CHECK_MODEL_FORWARD=0 \
bash scripts/run_edge_4090_full.sh
```

dry-run 不运行长任务，用于检查命令展开、路径和最终矩阵覆盖。

### 8.3 正式预检

```bash
python tools/run_edge_preflight.py \
  --config "$CONFIG" \
  --cuda-visible-devices 0 \
  --check-model-forward \
  --check-optional-backend-resolution \
  --check-quant-runtime \
  --quant-runtime-generate \
  --fail-on-blocked-methods \
  --fail-on-optional-backend-resolution \
  --fail-on-quant-runtime \
  --finalize-dry-run \
  --output "$PWD/.llmcbench/outputs/llmcbench_edge_4090/edge_preflight_report.json"
```

确认预检报告中：

```json
{
  "ok": true
}
```

### 8.4 正式运行

建议新开 tmux：

```bash
tmux new -s llmcbench-eval
source .llmcbench/venv/bin/activate
export CONFIG="$PWD/.llmcbench/outputs/llmcbench_edge_4090/edge_4090_prepared_models.json"
```

启动完整矩阵：

```bash
CONFIG="$CONFIG" \
CHECK_OPTIONAL_BACKEND_RESOLUTION=1 \
FAIL_ON_OPTIONAL_BACKEND_RESOLUTION=1 \
CHECK_QUANT_RUNTIME=1 \
QUANT_RUNTIME_GENERATE=1 \
FAIL_ON_QUANT_RUNTIME=1 \
bash scripts/run_edge_4090_full.sh 2>&1 | \
tee .llmcbench/outputs/llmcbench_edge_4090/full_run.log
```

脚本将自动执行：

1. 资产审计
2. 环境和模型前向预检
3. 数据集预热
4. 两模型 × Dense/六方法矩阵
5. 精度、部署和资源评测
6. JSON/CSV 收集与一致性验证
7. Markdown 报告和 scorecard
8. 完成审计
9. 结果 bundle 导出与校验

### 8.5 查看运行状态

另开终端：

```bash
tail -f .llmcbench/outputs/llmcbench_edge_4090/full_run.log
nvidia-smi
```

若 tmux 断开：

```bash
tmux attach -t llmcbench-eval
```

## 9. 单独运行某个评测工具

以下示例使用已准备的 Qwen3 AWQ checkpoint。

```bash
source .llmcbench/venv/bin/activate

export MODEL_PATH="$PWD/.llmcbench/models/precompressed/Qwen3-8B-awq"
export MODEL_NAME=Qwen3-8B
export COMPRESSION_METHOD=awq
export HF_CACHE_DIR="$PWD/.llmcbench/hf_cache"
export DATA_DIR="$PWD/.llmcbench/datasets"
export CUDA_VISIBLE_DEVICES=0
```

### 9.1 精度评测

```bash
export TASKS=mmlu,arc_challenge,hellaswag,piqa,winogrande
export BATCH_SIZE=1
bash scripts/run_edge_accuracy.sh
```

### 9.2 部署能力评测

```bash
export INPUT_LENGTH=2048
export MAX_NEW_TOKENS=128
export DEPLOYMENT_BATCH_SIZE=1
export REPEAT=3
bash scripts/run_edge_deploy.sh
```

### 9.3 推理资源评测

```bash
export SEQLEN=2048
export QUANT_BITS=4
export GROUP_SIZE=128
bash scripts/run_edge_resource.sh
```

三个工具也可直接调用：

```bash
python -m edge_benchmark accuracy --help
python -m edge_benchmark deploy --help
python -m edge_benchmark resource --help
```

## 10. 输出文件

默认结果目录：

```text
.llmcbench/outputs/llmcbench_edge_4090/
```

关键文件：

| 文件 | 作用 |
| --- | --- |
| `edge_env_report.json` | Python/CUDA/cuDNN/GPU 环境报告 |
| `edge_4090_prepared_models.json` | Base 和 12 个压缩 checkpoint 的最终矩阵配置 |
| `edge_handoff_manifest.json` | 代码版本、配置和运行命令交接清单 |
| `edge_target_paths_report.json` | 模型、数据、prompt、calibration 路径准备报告 |
| `edge_optional_backends_report.json` | 可选压缩后端解析报告 |
| `edge_quant_runtime_report.json` | GPTQ/AWQ 等量化 checkpoint 加载与生成 probe |
| `edge_preflight_report.json` | 正式运行前预检 |
| `edge_prewarm_report.json` | 数据集和任务预热结果 |
| `matrix_run_report.json` | 每个模型/方法任务状态 |
| `matrix_finalize_report.json` | 收集、验证和报告渲染步骤状态 |
| `matrix_results.json` | 完整机器可读结果 |
| `matrix_results.csv` | 表格结果 |
| `edge_report.md` | 可读实验报告 |
| `edge_scorecard.json` | 综合评分和 Pareto 信息 |
| `edge_scorecard.md` | 可读 scorecard |
| `completion_audit.json` | 完成性与一致性审计 |
| `target_readiness.json` | 目标机就绪状态 |
| `edge_results_bundle.tar.gz` | 可搬运结果包 |
| `edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json` | 结果包文件大小与 SHA256 清单 |
| `edge_results_bundle_verify.json` | 结果包 hash/结构验证 |
| `edge_delivery_verify.json` | 根目录、bundle 和压缩包的最终一致性裁决 |

每个模型/方法子目录包含：

```text
compression.json
accuracy.json
deployment.json
resource.json
summary.json
job.log
```

### 10.1 可选：导出代码便携包

如目标机器不能访问 Git 仓库，可导出包含代码、配置和运行说明的便携包：

```bash
python tools/export_edge_portable_bundle.py \
  --config .llmcbench/outputs/llmcbench_edge_4090/edge_4090_prepared_models.json \
  --output-root .llmcbench/outputs/llmcbench_edge_4090 \
  --output-dir .llmcbench/portable/edge_portable_bundle \
  --archive .llmcbench/portable/llmcbench_edge_portable.tar.gz \
  --force

python tools/verify_edge_portable_bundle.py \
  --bundle-root .llmcbench/portable/edge_portable_bundle \
  --archive .llmcbench/portable/llmcbench_edge_portable.tar.gz \
  --require-clean-code \
  --require-commit "$(git rev-parse HEAD)" \
  --output .llmcbench/portable/portable_bundle_verify.json
```

验证器会拒绝脏工作区、commit 不匹配、文件缺失、大小不符或 SHA256 不一致的便携包。

## 11. 常用控制参数

### 11.1 环境安装

```bash
DRY_RUN=1 bash scripts/install_4090_environment.sh
VERIFY_ONLY=1 bash scripts/install_4090_environment.sh
```

默认使用本地 venv。若明确希望使用 Conda：

```bash
ENV_MANAGER=conda ENV_NAME=llmcbench-edge bash scripts/install_4090_environment.sh
```

### 11.2 模型准备

```bash
DRY_RUN=1 bash scripts/prepare_4090_models.sh
VERIFY_ONLY=1 bash scripts/prepare_4090_models.sh
FORCE=1 bash scripts/prepare_4090_models.sh
```

常用资源限制：

```bash
HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0
MAX_MEMORY='0:22GiB,cpu:120GiB'
MIN_FREE_GPU_MEMORY_GB=20
```

### 11.3 数据准备

```bash
python scripts/prepare_4090_datasets.py --dry-run
python scripts/prepare_4090_datasets.py --family mmlu
python scripts/prepare_4090_datasets.py --retries 8 --retry-delay 30
```

### 11.4 评测矩阵

```bash
DRY_RUN=1                 # 只展开命令
STOP_ON_ERROR=1           # 任一任务失败即停止
RUN_PREWARM=1             # 运行数据集预热
CHECK_MODEL_FORWARD=1     # 检查 Base 模型前向
CHECK_QUANT_RUNTIME=1     # 检查量化 checkpoint 运行时
QUANT_RUNTIME_GENERATE=1  # 额外执行一次生成 smoke
```

## 12. 常见问题

### 12.1 `/data` 权限不足

当前版本不再使用 `/data`。执行：

```bash
git pull
```

确认脚本输出路径位于：

```text
当前仓库/.llmcbench/
```

### 12.2 pip 下载了 `torch-2.14.1`

说明使用了旧版本脚本或没有应用约束文件。停止安装并执行：

```bash
git pull
export PYTORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu124
bash scripts/install_4090_environment.sh
```

当前安装器会在所有 pip 阶段应用 `constraints-cu124.txt`，固定：

```text
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
auto-gptq==0.7.1
autoawq==0.2.9
compressed-tensors==0.15.0.1
bitsandbytes==0.50.1
```

### 12.3 阿里云 PyTorch 镜像提示 `ResolutionImpossible`

阿里云对应索引没有所需 Python 3.10/cu124 wheel。使用已验证的上海交大镜像：

```bash
export PYTORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu124
```

### 12.4 `pip: command not found`

安装环境后先激活本地 venv，并始终使用 `python -m pip`：

```bash
source .llmcbench/venv/bin/activate
python -m pip --version
```

### 12.5 Hugging Face 镜像返回 429

数据脚本会按 30、60、120、240、300 秒指数退避，并从本地 snapshot 继续。直接重跑即可：

```bash
python scripts/prepare_4090_datasets.py
```

模型下载也支持重跑；完整 checkpoint 自动跳过。

### 12.6 Qwen3 GPTQ 报 `qwen3 isn't supported yet`

这是旧脚本在 GPTQ checkpoint 下载失败后错误回退到 AutoGPTQ 0.7.1
导致的；AutoGPTQ 0.7.1 不支持本地量化 Qwen3。更新代码后重新运行：

```bash
git pull
source .llmcbench/venv/bin/activate
bash scripts/prepare_4090_models.sh
```

新脚本会对固定的 Qwen3 W4G128 checkpoint 执行可续传指数退避下载。最终
下载失败时会明确停止，不再进入错误的 AutoGPTQ 回退路径。开头出现的
`CUDA extension not installed` 只是 AutoGPTQ 警告，不是本次错误根因。

### 12.7 GPU 空闲显存不足

默认要求至少 20GB 空闲显存。检查：

```bash
nvidia-smi
```

关闭桌面占用、推理服务或其他 CUDA 任务后重试。不要为了绕过问题直接设置 `MIN_FREE_GPU_MEMORY_GB=0` 运行正式实验。

### 12.8 24GB 上 OOM

默认配置已经使用：

```text
FP16
batch size 1
deployment batch size 1
max_memory=0:22GiB,cpu:120GiB
SparseGPT/Wanda calibration buffer on CPU
```

诊断时可临时降低 `SEQLEN` 或 `INPUT_LENGTH` 到 1024，但正式结果必须恢复锁定配置。

### 12.9 AutoAWQ 弃用警告

AutoAWQ 0.2.9 会打印项目弃用警告，但已锁定在当前兼容栈中。最终是否可用由一 token 生成 probe 和完整量化 runtime preflight 决定，不以 import 成功代替运行验证。

## 13. 结果使用边界

### 13.1 标准矩阵与工程适配

- Wanda、SparseGPT 使用当前仓库实现和锁定校准协议。
- GPTQ、AWQ 优先使用固定 revision 的完整公开 checkpoint；下载失败时从同一 Base 本地生成。
- SmoothQuant、OmniQuant 对 LLaMA3/Qwen3 使用明确标记的 target-model engineering adaptation。
- 工程适配结果可用于项目工具评测，但不能表述为 OpenGVLab/原论文针对该目标模型发布的官方复现。

### 13.2 公平比较

同一模型的所有方法必须使用相同 Base 模型。报告中必须保留：

- Base 仓库与 revision
- checkpoint 来源
- 压缩方法和参数
- calibration 数据、样本数、长度和 seed
- weight/activation bit 与 group size
- 真实 packed/integer 或 fake quant 判定
- runtime profile
- GPU、CUDA、cuDNN、Torch、Transformers 版本

### 13.3 稀疏模型存储说明

Wanda/SparseGPT 输出仍是 Hugging Face dense tensor 文件，其中被剪枝权重置零。物理 checkpoint 文件不一定缩小；报告同时给出真实目录大小和按非零率估算的有效稀疏存储/FLOPs，不能把理论稀疏收益当成实际压缩文件大小。

### 13.4 历史结果

`docs/` 中保留 H20 诊断、严格协议状态和完成审计说明。历史 H20 数字不是最终 RTX 4090 验收结果，最终交付必须在目标 4090 机器上重新运行完整矩阵。

## 14. 旧版 LLMCBench 脚本

仓库根目录中的 `evaluate_mmlu.py`、`evaluate_ppl.py`、`evaluate_advglue.py` 等为原始 LLMCBench 兼容脚本。当前端侧项目应优先使用：

```text
python -m edge_benchmark accuracy
python -m edge_benchmark deploy
python -m edge_benchmark resource
python -m edge_benchmark pipeline
```

以及本 README 中的三套 4090 bootstrap 脚本。

## 15. 许可证与原始项目

本仓库保留原 LLMCBench 许可证，详见 [LICENSE](LICENSE)。

原论文：

```bibtex
@article{llmcbench2024,
  title={LLMCBench: Benchmarking Large Language Model Compression for Efficient Deployment},
  year={2024}
}
```
