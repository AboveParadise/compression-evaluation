# LLMCBench: Benchmarking Large Language Model Compression for Efficient Deployment

![image-20241026195404186](./figs/f1.png)

## Introduction

[LLMCBench: Benchmarking Large Language Model Compression for Efficient Deployment [arXiv]](https://arxiv.org/abs/2410.21352)

 The **L**arge **L**anguage **M**odel **C**ompression **Bench**mark (LLMCBench) is a rigorously designed benchmark with an in-depth analysis for LLM compression algorithms. 

## Branch Status

`master` is the current maintained branch. The `edge-pipeline-updates` branch is
retained as a historical edge-pipeline snapshot ending at commit `857f060`; it
is an ancestor of `master`, not a separate line of development. At the time of
this update, `edge-pipeline-updates` has no commits that are not already present
in `master`.

Compared with that snapshot, `master` adds:

- layerwise Wanda and full second-order SparseGPT implementations, including
  unstructured and 2:4 pruning;
- pinned calibration settings, base-model/source attestation, protocol
  validation and more complete result provenance;
- explicit target-model engineering adaptations for SmoothQuant and OmniQuant,
  kept separate from strict official-recipe results;
- standalone compression launchers and expanded 4090/H20 experiment matrices.

Use `master` for current development and new experiments. Use
`edge-pipeline-updates` only when reproducing the older pipeline behavior. In
particular, `master` changes the default pruning calibration from 64 samples
with bootstrap text fallback to 128 fixed-length C4 samples, and strict runs may
reject older checkpoints that do not contain the required manifest and
attestation metadata.

Inspect the current branch delta with:

```bash
git log --left-right --oneline master...origin/edge-pipeline-updates
git diff --stat origin/edge-pipeline-updates..master
```

## Installation

```
git clone https://github.com/AboveParadise/LLMCBench.git
cd LLMCBench

conda create -n llmcbench python=3.9
conda activate llmcbench
pip install -r requirements.txt
```

The installation block above is kept for the original LLMCBench scripts. The
edge-side LLaMA3-8B/Qwen3-8B compression benchmark tools below use the separate
Python 3.10 / CUDA 12.x environment in the `Environment Setup` section.

## Usage

This repo contains codes for testing MMLU, MNLI, QNLI, Wikitext2, advGLUE, TruthfulQA datasets and FLOPs.

#### Testing MMLU

```
bash scripts/run_mmlu.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--data_dir` : Dataset location.
- `--ntrain` : number of shots.
- `--seqlen` : Denotes the maximum input sequence length for LLM.

#### Testing MNLI

```
bash scripts/run_mnli.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--data_dir` : Dataset location.
- `--ntrain` : number of shots.
- `--seqlen` : Denotes the maximum input sequence length for LLM.

#### Testing QNLI

```
bash scripts/run_qnli.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--data_dir` : Dataset location.
- `--ntrain` : number of shots.
- `--seqlen` : Denotes the maximum input sequence length for LLM.

#### Testing Wikitext2

```
bash scripts/run_wikitext2.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--device` : Denotes which device to place the model onto.
- `--seqlen` : Denotes the maximum input sequence length for the model.

#### Testing advGLUE

```
bash scripts/run_advglue.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--data_file` : Dataset file location.
- `--ntrain` : number of shots.
- `--test_origin` : Denotes whether to test on the original GLUE data.

#### Testing TruthfulQA

```
bash scripts/run_tqa.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--presets` : Preset to use for prompt generation. Please see tqa_presets.py for options.
- `--input_path` : Dataset file location.
- `--device` : Denotes which device to place the model onto.

#### Testing FLOPs (floating point operations)

```
bash scripts/run_flops.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--seqlen` : Denotes the input sequence length for the model.

## Edge-side Compressed LLM Benchmark Tools

This repository also provides a unified edge-side benchmark CLI for compressed
LLaMA3-8B and Qwen3-8B checkpoints. The tools treat compression algorithms as
already exported HuggingFace-compatible checkpoints, so the same entrypoints can
evaluate models compressed by Wanda, SparseGPT, GPTQ, AWQ, SmoothQuant and
OmniQuant.

The three tools share the same standard platform:

- Hardware: one workstation with one NVIDIA RTX 4090 24GB GPU, multi-core CPU,
  256GB RAM and NVMe SSD. The target 4090 workflow defaults to
  `CUDA_VISIBLE_DEVICES=0`; on a shared H20 diagnostic machine, override it with
  `CUDA_VISIBLE_DEVICES=1`, and the scripts still address the selected GPU as
  `cuda:0`.
- Dataset storage: all datasets are stored on local disks and passed by path or
  HuggingFace datasets cache.
- Software: Python 3.10, PyTorch 2.x, CUDA 12.x, cuDNN, transformers,
  bitsandbytes, accelerate and compression-checkpoint loading libraries.

### Environment Setup

Create the Python environment:

```bash
conda create -n llmcbench-edge python=3.10 -y
conda activate llmcbench-edge

# Install a CUDA 12.x PyTorch build that matches the host driver.
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
```

The recommended single software profile is Python 3.10 with PyTorch 2.6.0
CUDA 12.4 wheels and Transformers 4.57.6. This combination has been verified to
load/generate GPTQ, AWQ, SmoothQuant W8A8 compressed-tensors and OmniQuant
precompressed checkpoints on the H20 diagnostic host. The compatibility shim in
`edge_benchmark.runtime_compat` restores the `PytorchGELUTanh` symbol expected
by older AutoGPTQ/AutoAWQ imports when running on newer Transformers.

### Reproducible 4090 Bootstrap

Three scripts prepare a clean target workstation in dependency order:

```bash
# 1. Python 3.10, PyTorch 2.6/cu124, cuDNN, bitsandbytes, Transformers,
#    compression runtimes and ModelScope.
bash scripts/install_4090_environment.sh

# 2. MMLU, ARC-Challenge/Easy, HellaSwag, PIQA and WinoGrande caches,
#    followed by an offline reload check.
python scripts/prepare_4090_datasets.py

# 3. ModelScope base models, downloaded GPTQ/AWQ checkpoints and locally
#    generated Wanda/SparseGPT/SmoothQuant/OmniQuant checkpoints.
bash scripts/prepare_4090_models.sh
```

The installer applies `constraints-cu124.txt` to every pip phase. This prevents
unbounded optional dependencies from upgrading the verified Torch 2.6/cu124
stack (for example, resolving a newer `torch` while installing
`compressed-tensors`). If an earlier install was interrupted during such an
upgrade, pull the latest `master` and rerun the installer; use `python -m pip`
inside the activated environment when inspecting packages.

Use `DRY_RUN=1` with either shell script and `--dry-run` with the dataset
script to print the complete plan without downloading or compressing. The model
script writes `.llmcbench/outputs/llmcbench_edge_4090/edge_4090_prepared_models.json`;
use that config for matrix evaluation after all asset checks pass.

All bootstrap artifacts default to the repository-local `.llmcbench/`
directory: the Python venv, base and compressed models, Hugging Face cache,
dataset snapshots, prompts, calibration inputs, reports and final outputs. Set
`LLMCBENCH_ROOT` only when an explicit external data disk is desired.

The aligned weight-only profile is W4 group 128 for GPTQ, AWQ and OmniQuant on
both base models. SmoothQuant remains W8A8 on both models because W8A8 is the
method's standard weight-and-activation setting; labelling a W4 weight-only
adaptation as SmoothQuant would not be a like-for-like implementation. Wanda
and SparseGPT use 50% unstructured sparsity with pinned C4 calibration.

If one method needs a different interpreter on a target host, the matrix runner
also supports per-method `python_bin` overrides:

```bash
ENV_NAME=llmcbench-edge-smoothquant EDGE_RUNTIME_PROFILE=smoothquant \
  bash scripts/setup_edge_env.sh
LLMCBENCH_SMOOTHQUANT_PYTHON=/path/to/llmcbench-edge-smoothquant/bin/python \
  python tools/run_edge_matrix.py --config configs/edge_4090_matrix.json --only-method smoothquant
```

`configs/edge_4090_matrix.json` sets `method_overrides.smoothquant.quant_bits`
to 8 for W8A8 checkpoints and lets `method_overrides.smoothquant.python_bin`
select a SmoothQuant-compatible Python through `LLMCBENCH_SMOOTHQUANT_PYTHON`
when a split profile is needed. The matrix runner still provides one overall
entry point; per-method Python selection is recorded in each row's runtime
metadata through `EDGE_RUNTIME_PROFILE`.

Use hf-mirror for model and dataset downloads:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache/transformers
export HF_DATASETS_CACHE=/data/llm_eval_datasets
```

Download or place models locally. Examples:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B \
  --local-dir /data/models/Meta-Llama-3-8B \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen3-8B \
  --local-dir /data/models/Qwen3-8B \
  --local-dir-use-symlinks False
```

For compressed checkpoints, set `MODEL_PATH` to the exported checkpoint
directory, for example `/data/models/llama3-8b-gptq` or
`/data/models/qwen3-8b-wanda`. The checkpoint directory should contain the
standard tokenizer/config/model files used by `transformers.from_pretrained`.

### 1. Edge Accuracy Evaluation

This tool evaluates compressed checkpoints on MMLU, ARC-Challenge, HellaSwag,
PIQA and WinoGrande through EleutherAI lm-evaluation-harness. It outputs a
normalized JSON summary plus the raw lm-eval output directory.
For final 4090 results, MMLU is not accepted as a single smoke subtask or only
an aggregate score: `collect_edge_results.py` preserves every configured
`mmlu_*` subtask score, and `validate_edge_results.py` requires the full
`DEFAULT_MMLU_SUBTASKS` coverage for every model/method row.

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/data/models/llama3-8b-gptq
export MODEL_NAME=LLaMA3-8B
export COMPRESSION_METHOD=gptq
export HF_CACHE_DIR=/data/hf_cache
export DATA_DIR=/data/llm_eval_datasets
export TASKS=mmlu,arc_challenge,hellaswag,piqa,winogrande
export BATCH_SIZE=auto

bash scripts/run_edge_accuracy.sh
```

Direct CLI form:

```bash
python -m edge_benchmark accuracy \
  --model-path /data/models/qwen3-8b-awq \
  --model-name Qwen3-8B \
  --compression-method awq \
  --cache-dir /data/hf_cache \
  --data-dir /data/llm_eval_datasets \
  --hf-endpoint https://hf-mirror.com \
  --tasks mmlu,arc_challenge,hellaswag,piqa,winogrande \
  --batch-size 1 \
  --device cuda:0 \
  --lm-eval-output outputs/edge_accuracy/lm_eval_raw \
  --output outputs/edge_accuracy/qwen3-8b-awq.json
```

Output fields include model metadata, compression method, evaluated tasks,
elapsed time and per-task accuracy metrics from lm-eval.

### 2. Edge Deployment Capability Evaluation

This tool measures inference-time GPU memory usage and generation speed
(`tokens/s`) for a compressed checkpoint and a fixed inference sequence.

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/data/models/llama3-8b-wanda
export MODEL_NAME=LLaMA3-8B
export COMPRESSION_METHOD=wanda
export PROMPT_FILE=/data/prompts/edge_prompts.txt
export INPUT_LENGTH=2048
export MAX_NEW_TOKENS=128
export DEPLOYMENT_BATCH_SIZE=1
export REPEAT=3

bash scripts/run_edge_deploy.sh
```

Direct CLI form:

```bash
python -m edge_benchmark deploy \
  --model-path /data/models/qwen3-8b-sparsegpt \
  --model-name Qwen3-8B \
  --compression-method sparsegpt \
  --cache-dir /data/hf_cache \
  --prompt-file /data/prompts/edge_prompts.txt \
  --device cuda:0 \
  --device-map auto \
  --input-length 2048 \
  --max-new-tokens 128 \
  --deployment-batch-size 1 \
  --warmup 1 \
  --repeat 3 \
  --output outputs/edge_deploy/qwen3-8b-sparsegpt.json
```

Output fields include peak allocated/reserved GPU memory, generated tokens,
elapsed seconds and tokens/s.

### 3. Edge Inference Resource Evaluation

This tool measures model storage size, parameter count, sparsity/non-zero ratio
and FLOPs/MACs for the configured inference sequence length.

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/data/models/llama3-8b-omniquant
export MODEL_NAME=LLaMA3-8B
export COMPRESSION_METHOD=omniquant
export SEQLEN=2048
export QUANT_BITS=4
export GROUP_SIZE=128

bash scripts/run_edge_resource.sh
```

Direct CLI form:

```bash
python -m edge_benchmark resource \
  --model-path /data/models/qwen3-8b-smoothquant \
  --model-name Qwen3-8B \
  --compression-method smoothquant \
  --cache-dir /data/hf_cache \
  --device cuda:0 \
  --device-map auto \
  --seqlen 2048 \
  --quant-bits 4 \
  --group-size 128 \
  --output outputs/edge_resource/qwen3-8b-smoothquant.json
```

Output fields include on-disk storage size, approximate sparse effective storage
computed as `storage * nonzero_ratio` for Wanda/SparseGPT, inspected
quantization storage format and the physical checkpoint directory size for
real packed/integer quantized methods, total parameters, sampled sparsity, calflops
FLOPs/MACs/parameter strings and an approximate sparse effective FLOPs value
computed as `dense_flops * nonzero_ratio`. The final matrix table keeps both
human-readable FLOPs/MACs strings and numeric fields such as `flops_num`,
`effective_flops_num_sparse_approx`, `effective_storage_gb_quantized_approx`,
`effective_storage_quantized_bits`, `effective_storage_quantized_backend`,
`storage_bytes` and `nonzero_ratio`. For quantization methods, final validation
requires those fields only when tensor inspection confirms real packed/integer
storage. `effective_storage_quantized_backend=checkpoint_directory_size` means
the reported value includes the actual checkpoint files and quantization
metadata; it is not reconstructed from packed tensor element counts.
Matrix defaults and overrides can set `sparsity_sample_tensors` to control how
many tensors are sampled for sparsity measurement; the resolved value is passed
to resource jobs and checked in final validation.
If `calflops` is unavailable or cannot trace a compressed model, the resource
tool falls back to an analytical config-based estimate and records
`backend=analytical_config_estimate` plus a warning in `resource.json`.
Accordingly, `verify_edge_env.py --check-resource-backend` reports which backend
will be used; missing `calflops` is a warning-level fallback rather than a hard
environment failure.

The standalone shell wrappers also accept `OUTPUT_FILE`, `DEVICE`,
`DEVICE_MAP`, `ATTN_IMPLEMENTATION`, `MAX_MEMORY` and
`MIN_FREE_GPU_MEMORY_GB`. `run_edge_accuracy.sh` additionally accepts
`NUM_FEWSHOT`, `LIMIT`, `STREAM_OUTPUT` and `LM_EVAL_OUTPUT`; `run_edge_deploy.sh`
accepts `PROMPT`, `INPUT_LENGTH`, `MAX_NEW_TOKENS`, `WARMUP`, `REPEAT` and
`DEPLOYMENT_MEMORY_LIMIT_GB`; `run_edge_resource.sh` accepts
`SPARSITY_SAMPLE_TENSORS`.
The standalone accuracy, deployment and resource wrappers default to
`MAX_MEMORY=0:22GiB,cpu:120GiB` and `MIN_FREE_GPU_MEMORY_GB=20`; the deployment
wrapper also defaults to `DEPLOYMENT_MEMORY_LIMIT_GB=24`. They also default to
`CUDA_VISIBLE_DEVICES=0`, matching the single-card 4090 target; set
`CUDA_VISIBLE_DEVICES=1` only for the shared H20 diagnostics described above.
Direct `python -m edge_benchmark ...` invocations default to
`HF_ENDPOINT=https://hf-mirror.com`, `MIN_FREE_GPU_MEMORY_GB=20` and a
24GB deployment memory limit. On CUDA hosts, direct CLI model-loading commands
also apply `max_memory=0:22GiB,cpu:120GiB` unless `--max-memory` is set
explicitly.
The end-to-end `run_edge_pipeline.sh` wrapper defaults to the 4090-safe values
`MAX_MEMORY=0:22GiB,cpu:120GiB`, `MIN_FREE_GPU_MEMORY_GB=20` and
`DEPLOYMENT_MEMORY_LIMIT_GB=24`; override them only for a deliberately different
target or for bounded diagnostics.
The 4090 matrix uses `batch_size=1` for lm-eval by default. Avoid
`batch_size=auto` for final 24GB runs because the auto probe can temporarily
push memory far beyond the intended edge-device budget.

### Compression Method Labels

Use one of these labels for `--compression-method`:

```text
dense, wanda, sparsegpt, gptq, awq, smoothquant, omniquant
```

The label is recorded in the result JSON. Loading behavior is determined by the
checkpoint files and installed libraries. GPTQ/AWQ/bitsandbytes-style
checkpoints may require the optional packages listed in
`requirements-optional.txt`.

### Standalone Compression Directory

The real compression implementations are shared by the benchmark pipeline in
`edge_benchmark/compression.py`. The user-facing standalone interface is now
in `compression/`:

```text
compression/
  run_compression.sh          # shell entry point
  run.py                      # Python launcher
  profiles.py                 # method-specific presets
```

Users normally need only a local HuggingFace base-model directory and a method:

```bash
# Equivalent forms; run from the repository root.
CUDA_VISIBLE_DEVICES=1 bash compression/run_compression.sh \
  --base-model /data/models/Qwen3-8B \
  --method awq \
  --output-dir /data/models/Qwen3-8B-AWQ-W4G128

CUDA_VISIBLE_DEVICES=1 python -m compression \
  --base-model /data/models/Meta-Llama-3-8B \
  --method wanda
```

The launcher applies a reproducible preset automatically:

| Method | Default launcher setting |
| --- | --- |
| Wanda | C4 seed 0, 128 × 2048, 50% unstructured |
| SparseGPT | C4 seed 0, 128 × 2048, 50% unstructured, blocksize 128, damp 0.01 |
| GPTQ | C4 seed 0, 128 × 2048, W4 per-channel, asymmetric, act-order, true-sequential |
| AWQ | Pile validation seed 42, 128 × 512, W4g128, zero-point, 20-point grid |
| SmoothQuant | Target-model adaptation: Pile validation 512 texts, real INT8 storage |
| OmniQuant | Target-model adaptation: WikiText2 128 × 2048, real packed UINT4 storage |

`smoothquant` and `omniquant` default to `--setting adapted`, because their
upstream projects have no LLaMA3-8B/Qwen3-8B same-model recipe. Their exported
manifest is marked `target_model_adaptation=true`; it is suitable for the
engineering matrix and is reported separately from native upstream recipes.
Inspect the resolved command without writing weights:

```bash
python -m compression \
  --base-model /data/models/Qwen3-8B \
  --method gptq \
  --dry-run
```

Wanda, SparseGPT, GPTQ and AWQ use strict presets by default.
SmoothQuant/OmniQuant use the explicit engineering-adaptation path when an
upstream target-model recipe is unavailable.

### Result Settings

Compare numerical results only when the model, compression configuration,
calibration corpus/size, evaluation task suite, runtime and hardware all
match. Otherwise report the result as this repository's own edge benchmark and
state the setting difference.

The existing historical H20 rows use LLaMA3-8B/Qwen3-8B and
MMLU/ARC-Challenge/HellaSwag/PIQA/WinoGrande; their Wanda/SparseGPT rows also
used only five calibration texts. Treat them as diagnostics rather than as
directly comparable results. The strict matrix records its complete profile in
`configs/edge_paper_aligned_matrix.json` and
`configs/paper_protocols.json`.

### End-to-end Compression and Benchmark Pipeline

Current progress in this repository:

- Implemented the three benchmark tools requested for compressed checkpoints:
  accuracy, deployment capability and resource consumption.
- Added a unified compression/evaluation entrypoint:
  `python -m edge_benchmark pipeline`.
- Added selectable compression methods through `--compression-method`:
  `dense`, `wanda`, `sparsegpt`, `gptq`, `awq`, `smoothquant`, `omniquant`.
- Added a 4090-oriented default configuration: single visible GPU, `device_map=auto`,
  `fp16`, calibration batch size 1, 2048-token sequences, `max_memory` bounded
  to a 4090-sized GPU budget and a deployment peak-memory validation limit.
  Matrix defaults and overrides can set `device_map` for compression,
  deployment and resource loading; the default remains `auto`.
  Matrix defaults and overrides can also set `revision` to pin the HuggingFace
  model revision used by compression, accuracy, deployment and resource runs.
- Added final runtime gates that reject rows with GPU metadata unless CUDA was
  actually available, so CPU-only or misreported runs cannot be accepted as
  final 4090 evidence.
- Tightened final result validation for portable handoff: aggregate accuracy
  scores, selected metric keys, integer-valued counters, compression/resource
  backend labels, materialized/precompressed provenance, deployment token counts,
  few-shot settings, prompt/cache/attention provenance and
  root/bundle/archive delivery consistency are all checked before final results
  are accepted.
  storage/FLOPs/quantized-storage estimates must be internally consistent.
- Added JSON outputs for every step so results can be moved to another machine
  without parsing console logs.

Run the full flow, including compression and all three evaluations:

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export BASE_MODEL_PATH=/data/models/Meta-Llama-3-8B
export MODEL_NAME=LLaMA3-8B
export COMPRESSION_METHOD=gptq
export HF_CACHE_DIR=/data/hf_cache
export DATA_DIR=/data/llm_eval_datasets
export CALIBRATION_FILE=/data/calibration/wiki_samples.txt
export PROMPT_FILE=/data/prompts/edge_prompts.txt
export OUTPUT_ROOT=/data/outputs/llama3-8b-gptq

bash scripts/run_edge_pipeline.sh
```

Equivalent direct command:

```bash
python -m edge_benchmark pipeline \
  --model-path /data/models/Qwen3-8B \
  --model-name Qwen3-8B \
  --compression-method awq \
  --cache-dir /data/hf_cache \
  --hf-endpoint https://hf-mirror.com \
  --device cuda:0 \
  --device-map auto \
  --dtype fp16 \
  --seqlen 2048 \
  --calibration-file /data/calibration/wiki_samples.txt \
  --calibration-samples 128 \
  --calibration-batch-size 1 \
  --quant-bits 4 \
  --group-size 128 \
  --data-dir /data/llm_eval_datasets \
  --tasks mmlu,arc_challenge,hellaswag,piqa,winogrande \
  --batch-size 1 \
  --prompt-file /data/prompts/edge_prompts.txt \
  --input-length 2048 \
  --max-new-tokens 128 \
  --deployment-batch-size 1 \
  --warmup 1 \
  --repeat 3 \
  --output-root /data/outputs/qwen3-8b-awq
```

If the checkpoint is already compressed/exported, skip recompression and run the
three evaluations directly:

```bash
python -m edge_benchmark pipeline \
  --model-path /data/models/qwen3-8b-gptq \
  --model-name Qwen3-8B \
  --compression-method gptq \
  --use-precompressed \
  --cache-dir /data/hf_cache \
  --data-dir /data/llm_eval_datasets \
  --tasks mmlu,arc_challenge,hellaswag,piqa,winogrande \
  --prompt-file /data/prompts/edge_prompts.txt \
  --output-root /data/outputs/qwen3-8b-gptq-eval
```

For matrix runs, put exported compressed checkpoints in `precompressed_paths`:

```json
{
  "precompressed_paths": {
    "Qwen3-8B": {
      "gptq": "/data/models/qwen3-8b-gptq",
      "awq": "/data/models/qwen3-8b-awq"
    }
  }
}
```

For Qwen3-8B GPTQ/AWQ/SmoothQuant/OmniQuant, the repository includes a helper that can stage known
public precompressed checkpoints from hf-mirror and write a config copy with
`precompressed_paths` populated:

```bash
python tools/prepare_precompressed_assets.py \
  --config configs/edge_4090_matrix.json \
  --local-root /data/models/precompressed \
  --only-model Qwen3-8B \
  --only-method gptq \
  --only-method awq \
  --only-method smoothquant \
  --only-method omniquant \
  --output-config /data/outputs/llmcbench_edge_4090/edge_matrix_with_precompressed.json \
  --output /data/outputs/llmcbench_edge_4090/edge_prepare_precompressed_report.json \
  --fail-on-error
```

Use `--metadata-only` first to check access and tokenizer/index files without
pulling weight shards, or `--dry-run` to render the planned downloads only. Add
`--write-dry-run-config` with `--dry-run` when you also want a temporary config
copy with the planned `precompressed_paths` entries for matrix/preflight
dry-runs. Successful staging writes `.llmcbench_precompressed_source.json` in
each checkpoint directory. If a complete directory has a marker for a different
repo, or if the default Qwen3-8B GPTQ directory still contains old GPTQModel
metadata, the helper reports a blocker instead of silently reusing that
checkpoint; rerun with `--force` or use a clean `--local-root` to replace it.
The default repo mapping includes Qwen3-8B
GPTQ/AWQ/SmoothQuant/OmniQuant and
LLaMA3-8B GPTQ/AWQ/SmoothQuant public candidates. LLaMA3-8B OmniQuant has no
base-model default because the public candidates found during development were
Instruct or derivative checkpoints; the preparation report marks that row with
`repo_status=no_verified_public_default_repo`, adds it to `blockers`, and sets
the report `ok=false`. The wrapper and handoff command use `--fail-on-error` so
this gap fails before a long matrix run. Override any mapping with
`--repo <MODEL>:<method>=<repo_id>` if a different checkpoint is required.

When a model/method pair is present in this map, `tools/run_edge_matrix.py`
passes `USE_PRECOMPRESSED=1` to `scripts/run_edge_pipeline.sh` and evaluates the
given checkpoint directly. For SmoothQuant/OmniQuant pairs with a
`precompressed_paths` entry, no external command template is required because
the exported checkpoint is already present. The precompressed manifest still
records the configured bits, group size, target sparsity, calibration file,
calibration sample count, calibration batch size, `max_memory` and attention
backend so reports keep a consistent schema.
`tools/audit_edge_assets.py --only-method <method>` and
`tools/run_edge_preflight.py --only-method <method>` are method-aware: for a
selected model/method pair backed by `precompressed_paths`, they require the
exported compressed checkpoint instead of the original dense checkpoint.
`tools/check_method_readiness.py` uses the same rule: evaluating an already
exported GPTQ/AWQ/SmoothQuant/OmniQuant checkpoint does not require the
corresponding compression backend or external command to be installed. It still
checks the runtime libraries needed to load the exported checkpoint for
accuracy/deployment. The stable AutoGPTQ path uses `optimum<2` plus
`auto-gptq`; Optimum 2.x routes GPTQ loading through GPTQModel symbols. Keep
the stable pair installed on the target 4090 environment. Some public GPTQ
checkpoints advertise GPTQModel metadata and fail under `auto-gptq` alone.
`scripts/setup_optional_backends.sh` installs
`gptqmodel==4.0.0` separately with `--no-build-isolation` on Python 3.10+ to
match the public Qwen GPTQ checkpoint metadata and avoid pip build isolation
hiding the active Torch install. AWQ loading requires `autoawq`.
Before running a long matrix with precompressed GPTQ/AWQ checkpoints, probe the
runtime loader explicitly:

```bash
python tools/probe_quant_runtime.py \
  --config /data/outputs/llmcbench_edge_4090/edge_matrix_with_precompressed.json \
  --only-model Qwen3-8B \
  --only-method awq \
  --generate \
  --cuda-visible-devices 0 \
  --output /data/outputs/llmcbench_edge_4090/qwen3_awq_runtime_probe.json
```

The probe records import status, quantization metadata, config/tokenizer/model-load
steps, optional one-token generation, package versions and the exact error if a
quantized kernel or runtime is incompatible. For checkpoints whose metadata
requires a specific runtime, such as GPTQModel or compressed-tensors, missing
imports are reported before model load.
`tools/run_edge_preflight.py --check-quant-runtime` runs the same probe and
writes `edge_quant_runtime_report.json` beside the preflight report. Add
`--quant-runtime-generate` to exercise one-token generation and
`--fail-on-quant-runtime` when a failed probe should block the preflight. The
one-command 4090 wrapper exposes the same controls with
`CHECK_QUANT_RUNTIME=1`, `QUANT_RUNTIME_GENERATE=1` and
`FAIL_ON_QUANT_RUNTIME=1`.

Pipeline outputs:

```text
<OUTPUT_ROOT>/compressed/<MODEL_NAME>_<METHOD>/   # compressed checkpoint
<OUTPUT_ROOT>/compression.json                    # compression metadata
<OUTPUT_ROOT>/accuracy.json                       # MMLU/ARC/HellaSwag/PIQA/WinoGrande metrics
<OUTPUT_ROOT>/deployment.json                     # peak memory and tokens/s
<OUTPUT_ROOT>/resource.json                       # storage, sparsity, params and FLOPs
<OUTPUT_ROOT>/summary.json                        # paths and high-level pipeline summary
<OUTPUT_ROOT>/lm_eval_raw/                        # raw lm-eval output
```

`resource` first tries to load the model and run the normal parameter/FLOPs
path. If a quantized precompressed checkpoint cannot be loaded because the
runtime backend is missing, it falls back to a config-only resource estimate:
directory size is still measured from the checkpoint files, parameter count and
FLOPs are estimated from `config.json`, and `model_load_error` records the
missing runtime. This fallback is only for resource accounting; accuracy and
deployment still require the quantized checkpoint to load successfully. Final
matrix validation rejects `resource_backend=config_only_fallback`, so fallback
rows remain diagnostic rather than complete resource results.

`compression.json`, `accuracy.json`, `deployment.json`, `resource.json` and
`summary.json` include lightweight runtime metadata such as Python, PyTorch,
Transformers, CUDA availability/version and visible GPU information. The final
matrix table carries these fields forward so results are tied to the software
and hardware environment that produced them.

### Compression Backend Notes

The unified CLI keeps one interface for all methods, but the backend differs by
method:

- `dense`: baseline method. It reuses the source checkpoint and writes only a
  manifest, avoiding an extra full 8B checkpoint copy.
- `wanda`: built-in activation-aware pruning. The checked-in matrix uses
  `prune_pattern=unstructured` and `sparsity=0.5`. Set
  `--prune-pattern 2:4` to prune the two lowest-scoring weights in every
  consecutive group of four weights; 2:4 always implies 50% element sparsity.
  For LLaMA3/Qwen3 it follows the official
  [Wanda](https://github.com/locuslab/wanda) layerwise procedure: it captures
  first-layer calibration states, collects each decoder block's input
  activation norms, prunes by `abs(W) * sqrt(mean(x^2))`, then propagates the
  sparse block output to the next decoder layer. KV cache is disabled and
  `lm_head` is excluded.
- `sparsegpt`: built-in official SparseGPT-style layerwise second-order pruning
  for LLaMA3/Qwen3. For each decoder Linear module, it collects the full input
  Hessian, applies Cholesky inverse factorization and blockwise error
  compensation. The default is `--sparsegpt-blocksize 128` and
  `--sparsegpt-percdamp 0.01`, matching the public `locuslab/wanda` SparseGPT
  path. Activation buffers use `--pruning-buffer-device cpu` by default so the
  `128 x 2048` layerwise calibration states do not consume the 24GB GPU budget.
  Set `--prune-pattern 2:4` for exact 2:4 masks. `lm_head` is excluded.

For the checked-in pruning matrices, this is the exact reproducible
calibration protocol:

```text
Dataset: allenai/c4, config allenai--c4
Revision: 1588ec454efa1a09f29cd18ddd04fe05fc8653a2
Training shard: en/c4-train.00000-of-01024.json.gz
Seed: 0
Samples: 128 independently sampled C4 documents
Sequence length: 2048 tokens
Sparsity: 50% unstructured by default; exact 2:4 in the structured config
SparseGPT: blocksize=128, percdamp=0.01
```

The code downloads that pinned shard through `HF_ENDPOINT=https://hf-mirror.com`
with an explicit `hf_hub_download` endpoint, then reads it locally before
sampling. Every pruning manifest records dataset, revision, shard, seed,
sample count and token length. This avoids treating a short hand-written text
file as a C4 calibration substitute.
- `gptq`: uses `auto-gptq`; configure `--quant-bits`, `--group-size` and
  optionally `--desc-act`. The matrix `max_memory` and `attn_implementation`
  settings are passed through to the GPTQ model loader for 4090-sized runs.
  Matrix defaults and overrides can set `desc_act`, and the resolved value is
  recorded in compression manifests and final results.
- `awq`: uses `autoawq`; configure `--quant-bits`, `--group-size` and
  `--awq-version`. The same matrix `max_memory` and `attn_implementation`
  settings are passed through to the AWQ model loader. Matrix defaults and
  overrides can set `awq_version`, and the resolved value is recorded in
  compression manifests and final results.
- `smoothquant`: adapter for an external SmoothQuant implementation. Provide
  `--external-command` or `LLMCBENCH_SMOOTHQUANT_CMD`.
- `omniquant`: adapter for an external OmniQuant implementation. Provide
  `--external-command` or `LLMCBENCH_OMNIQUANT_CMD`.
  External-command manifests record the configured bits, group size, sparsity,
  calibration sample count, `max_memory` and attention backend for reporting.

External command templates support these placeholders:

```text
{model_path} {output_dir} {calibration_file} {seqlen}
{calibration_samples} {calibration_batch_size}
{quant_bits} {group_size} {sparsity} {prune_pattern}
{max_memory} {attn_implementation} {cache_dir}
```

Example:

```bash
export LLMCBENCH_SMOOTHQUANT_CMD='python /opt/SmoothQuant/export.py --model {model_path} --output {output_dir} --calib {calibration_file} --seq-len {seqlen}'
python -m edge_benchmark compress \
  --model-path /data/models/Meta-Llama-3-8B \
  --model-name LLaMA3-8B \
  --compression-method smoothquant \
  --compressed-output-dir /data/models/llama3-8b-smoothquant \
  --external-command "${LLMCBENCH_SMOOTHQUANT_CMD}"
```

Quantization semantics in the checked-in matrix:

- GPTQ and AWQ checkpoints contain packed INT4 `qweight/qzeros/scales` tensors;
  they are real weight-only quantized checkpoints, not fake-quantized FP16
  weights.
- Qwen3 SmoothQuant contains INT8 Linear weights plus scales and uses W8A8
  compressed-tensors metadata. LLaMA3 SmoothQuant's currently staged checkpoint
  stores FP16 weights and is therefore a fake-quantized/smoothed diagnostic
  checkpoint rather than packed INT8 storage.
- LLaMA3 OmniQuant is a packed GPTQ-compatible INT4 checkpoint. The currently
  staged Qwen3 OmniQuant checkpoint stores FP16 weights plus quantizer
  scale/zero tensors; treat it as fake-quantized metadata unless replaced by a
  packed export.

> **Engineering-only warning:** The following table describes
> **historical/staged engineering checkpoints**. Its public checkpoint
> parameters can differ from the strict profiles, so keep it separate from
> strict-profile results.

The historical/staged compression settings are:

| Method | Model | Pattern / Precision | Ratio / Group | Stored checkpoint |
| --- | --- | --- | --- | --- |
| Wanda default | Both | Unstructured pruning | 50% element zeros | FP16 sparse tensors |
| SparseGPT default | Both | Unstructured pruning | 50% element zeros | FP16 sparse tensors |
| Wanda 2:4 | Both | Semi-structured 2:4 | Exactly 2 zeros per 4 weights, 50% | FP16 sparse tensors |
| SparseGPT 2:4 | Both | Semi-structured 2:4 | Exactly 2 zeros per 4 weights, 50% | FP16 sparse tensors |
| GPTQ | LLaMA3-8B | W4A16, symmetric, `desc_act=false` | Checkpoint group 64 | Real packed INT4 |
| GPTQ | Qwen3-8B | W4A16, `desc_act=false` | Checkpoint group 128 | Real packed INT4 |
| AWQ | Both | W4A16, zero-point, GEMM | Group 128 | Real packed INT4 |
| SmoothQuant | LLaMA3-8B | Configured W8A8 | Per-channel weights / per-token activations | Current export is FP16 fake-quant/smoothed |
| SmoothQuant | Qwen3-8B | W8A8 | Per-channel weights / per-token activations | Real INT8 weights with BF16 scales |
| OmniQuant | LLaMA3-8B | W4A16, symmetric | Group 128 | Real packed INT4, GPTQ-compatible export |
| OmniQuant | Qwen3-8B | Configured W4A16 | Group 128 | Current export is FP16 plus scale/zero fake-quant metadata |

`configs/edge_4090_matrix.json` requests `quant_bits=4` and `group_size=128`
for newly generated GPTQ/AWQ/OmniQuant assets, with SmoothQuant overriding
`quant_bits=8`. When a precompressed checkpoint is used, its own metadata is
authoritative. Result rows therefore keep the requested values in
`compression_quant_bits/compression_group_size` and report the inspected asset
as `checkpoint_weight_bits/checkpoint_activation_bits/checkpoint_group_size`,
along with `quantization_storage_real`.

### Recommended 4090 Experiment Matrix

Run these combinations for full results:

```text
Default matrix for both models: dense, wanda, sparsegpt, gptq, awq,
smoothquant, omniquant.

Semi-structured matrix (`configs/edge_4090_structured_pruning.json`) for both
models: Wanda 2:4 and SparseGPT 2:4.
```

### Strict Reproducible Matrix

`configs/edge_4090_matrix.json` is the general engineering matrix. Use
`configs/edge_paper_aligned_matrix.json` for strict reproducible settings.
Each row maps to a named profile in
`configs/paper_protocols.json`. Its validator runs before every strict matrix
job and again during finalization.

| Method | Strict profile | Required setting |
| --- | --- | --- |
| Dense | `dense_fp16_same_checkpoint_baseline` | Same released source checkpoint/revision as the compressed comparison |
| Wanda | `wanda_iclr2024_c4_128x2048_seed0_unstructured50` | C4 seed 0, 128 random 2048-token windows, 50% unstructured |
| SparseGPT | `sparsegpt_icml2023_c4_128x2048_seed0_unstructured50_block128_damp001` | C4 seed 0, 128 × 2048, 50% unstructured, blocksize 128, percdamp 0.01 |
| GPTQ | `gptq_for_llama_c4_w4_asymmetric_actorder_true_sequential` | C4 seed 0, 128 × 2048, W4A16 with the official asymmetric default, per-channel `group_size=-1`, act-order and true-sequential |
| AWQ | `awq_mlsys2024_pile_w4_g128_grid20` | Official LLM-AWQ LLaMA3-8B W4A16g128 recipe; Qwen3 setting transfer. Pile validation (`pileval`) 128 × 512, seed 42, zero-point, verified 20-point grid |
| SmoothQuant | `smoothquant_w8a8_pile_validation_512_sentences_alpha_model_specific` | W8A8/Pile/512 profile; **blocked for both targets** because the pinned upstream source has no LLaMA3-8B/Qwen3-8B target-model recipe/alpha |
| OmniQuant | `omniquant_w4a16_wikitext2_128x2048_epochs20_lwc` | WikiText2/seed-2/W4A16/20-epoch LWC profile; **blocked for both targets** because the pinned upstream source has no LLaMA3-8B/Qwen3-8B target-model script/export |

The upstream implementations did not all cover both LLaMA3-8B and Qwen3-8B.
AWQ has an upstream LLaMA3-8B W4g128 command; the Qwen3 AWQ row and the other
unsupported target rows remain explicitly labelled profile transfers. The
strict config never replaces settings with a generic text
file, fewer samples, a different group size, or a guessed SmoothQuant alpha.
SmoothQuant and OmniQuant are deliberately blocked for both requested targets:
the pinned upstream sources do not provide LLaMA3-8B/Qwen3-8B model recipes,
and Qwen3 SmoothQuant also lacks an official alpha. Consequently, the strict
matrix currently runs only the dense/Wanda/SparseGPT/GPTQ/AWQ slices. Do not
turn a third-party checkpoint, a guessed alpha, or a custom adaptation into a
strict-profile external-method row. Add an explicit target-model recipe with
source evidence before lifting a block.

### Complete H20 Target-Model Adaptation Matrix

When complete results are needed before upstream SmoothQuant or OmniQuant
provides a target-model recipe, use
`configs/edge_h20_adapted_full_matrix.json`. This is a separate
**engineering-adaptation** matrix and must be reported separately from native
upstream recipes:

- `smoothquant` uses Pile validation activation calibration, applies
  SmoothQuant-style input scaling to standard `nn.Linear` layers, stores real
  INT8 weights, and dequantizes to FP16 for the Transformers runtime.
- `omniquant` uses WikiText2 activation calibration and groupwise
  activation-aware clipping, stores real packed UINT4 weights with
  scales/zeros, and dequantizes to FP16 for the Transformers runtime.
- Every result includes the standard MMLU/ARC/HellaSwag/PIQA/WinoGrande
  accuracy, physical storage, FLOPs/MACs, measured peak memory and tokens/s
  fields, plus `target_model_adaptation=true` and a runtime disclaimer.

Run it on the assigned H20:

```bash
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com \
python tools/run_edge_matrix.py \
  --config configs/edge_h20_adapted_full_matrix.json \
  --stop-on-error --finalize
```

The adaptation exporter fails if it cannot obtain the configured calibration
coverage; it never silently substitutes the historical five-line file.

On a shared machine, use the waiting launcher instead of competing with an
active process. It starts only after physical GPU `1` has low utilization and
at least the configured free-memory budget:

```bash
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_h20_adapted_full.sh
```

Strict rows also require the **base model** to be the pinned official release:
`meta-llama/Meta-Llama-3-8B@8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920` and
`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`. After preparing
the models, create local attestations before running the strict matrix:

```bash
python tools/attest_base_models.py \
  --config configs/edge_paper_aligned_matrix.json \
  --output /data/outputs/llmcbench_strict_4090/base_model_attestation.json
```

The attestor compares local `config.json` and `model.safetensors.index.json`
against those pinned upstream revisions and rejects any
`.llmcbench_source_override.json` marker. In particular, a NousResearch
fallback must not be labelled as the official Meta-Llama-3-8B strict baseline.

See `docs/STRICT_PROTOCOL_STATUS.md` for the current per-model/per-method
strict admissibility table, source-evidence limitations and the exact blockers
on the development host.

```bash
python tools/validate_paper_protocol.py \
  --config configs/edge_paper_aligned_matrix.json

CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com \
python tools/run_edge_matrix.py \
  --config configs/edge_paper_aligned_matrix.json \
  --stop-on-error --finalize
```

For scoped work, repeat `--only-model` / `--only-method`; strict validation
checks only the selected slice. Strict finalization writes
`completion_paper_protocol_validation.json`, which the completion audit
requires to pass.

SmoothQuant and OmniQuant external exporters receive profile placeholders for
future officially supported target-model recipes:
as `{paper_protocol}`, `{smoothquant_alpha}`,
`{smoothquant_alpha_selection}`, `{smoothquant_calibration_source}`,
`{smoothquant_calibration_sentences}`, `{omniquant_abits}`,
`{omniquant_epochs}`, `{omniquant_nsamples}`, `{omniquant_seqlen}`,
`{omniquant_calibration_source}`, `{omniquant_lwc}`, `{omniquant_let}` and
`{omniquant_aug_loss}`. In strict mode they must write
`<output_dir>/llmcbench_external_protocol.json` containing
`paper_protocol`, `paper_protocol_attested: true`, and the actual relevant
profile fields. A SmoothQuant evidence file must include a numeric
`smoothquant_alpha`.

Strict external rows also bind the exporter to the official source checkout:
set `LLMCBENCH_SMOOTHQUANT_DIR` / `LLMCBENCH_OMNIQUANT_DIR` to Git checkouts
whose `origin` and `HEAD` exactly match the pinned repository and commit in
`configs/paper_protocols.json`. The command must reference
`{smoothquant_official_code_dir}` or `{omniquant_official_code_dir}`, and the
evidence JSON must include matching `official_repository` and
`official_commit`. A wrapper or arbitrary conversion script outside that
checkout is rejected in strict mode.

Provision the pinned checkouts before configuring strict external commands:

```bash
python tools/setup_official_compression_sources.py \
  --root /opt/llmcbench_official \
  --output /data/outputs/llmcbench_strict_4090/official_sources.json

export LLMCBENCH_SMOOTHQUANT_DIR=/opt/llmcbench_official/smoothquant
export LLMCBENCH_OMNIQUANT_DIR=/opt/llmcbench_official/omniquant
```

The strict command must directly invoke an exporter file inside the matching
checkout, for example
`python {smoothquant_official_code_dir}/<official-export-script>.py ...`.
The current upstream repositories do not ship a universal LLaMA3/Qwen3 export
command, so the exact official entrypoint must be selected from the pinned
checkout and must write the required evidence JSON. Do not replace it with an
unrelated conversion wrapper.

For 24GB cards, start with:

```text
SEQLEN=2048
CALIBRATION_SAMPLES=64
CALIBRATION_BATCH_SIZE=1
BATCH_SIZE=auto
MAX_NEW_TOKENS=128
DEPLOYMENT_BATCH_SIZE=1
SPARSITY=0.5
QUANT_BITS=4
GROUP_SIZE=128
DTYPE=fp16
MIN_FREE_GPU_MEMORY_GB=20
MAX_MEMORY=0:22GiB,cpu:120GiB
DEPLOYMENT_MEMORY_LIMIT_GB=24
```

If memory is close to 24GB, reduce `SEQLEN` to 1024, lower
`CALIBRATION_SAMPLES`, or run `SKIP_ACCURACY=1` while debugging compression and
deployment first. The default full-matrix config requires 20GB free GPU memory
before long jobs start; set `MIN_FREE_GPU_MEMORY_GB=0` or
`--min-free-gpu-memory-gb 0` to disable this guard.
Keep `DTYPE=fp16` on the 4090 target unless you intentionally switch to bf16
after preflight; fp32 dense 8B deployment exceeds the 24GB target in the H20
diagnostic record.
Set `MAX_MEMORY=0:22GiB,cpu:120GiB` or `--max-memory 0:22GiB,cpu:120GiB` to
make Transformers respect a 4090-sized GPU memory budget during model loading;
the value is passed through compression, deployment and resource model loaders,
and translated to lm-eval's `max_memory_per_gpu` / `max_cpu_memory` arguments
for accuracy and prewarm.
`configs/edge_4090_matrix.json` also sets `deployment_memory_limit_gb=24`, so
pipeline deployment writes `deployment_memory_limit_exceeded` and fails the job
if the measured peak allocated/reserved GPU memory is above the 24GB target.
The measured peak is `max(peak_memory_allocated_gb, peak_memory_reserved_gb)`,
so allocator reservation spikes are treated as deployment memory usage.
Deployment generation uses `deployment_batch_size=1` by default. The prompt file
may contain multiple inference sequences, but they are generated in micro-batches
and aggregated for tokens/s; this keeps peak-memory measurement aligned with
edge-side single-request or small-batch serving instead of multiplying memory by
the number of prompt-file lines.
`tools/validate_edge_results.py` enforces the same limit on the collected final
matrix table.

### Full Matrix Runner and Result Collection

The repository includes automation for the complete two-model matrix. Edit
`configs/edge_4090_matrix.json` if your model, dataset or output paths differ:
For local development on this H20 host, `configs/edge_local_h20_matrix.json`
uses `/opt/tiger/...` paths instead of `/data/...`.

On the target 4090 machine, the full prepared workflow can be launched through
one wrapper:

```bash
DRY_RUN=1 CHECK_MODEL_FORWARD=0 RUN_PREWARM=0 bash scripts/run_edge_4090_full.sh
bash scripts/run_edge_4090_full.sh
```

To have the wrapper stage public precompressed checkpoints first and use the
generated config copy for the rest of the workflow, enable:

```bash
RUN_PRECOMPRESSED_PREPARE=1 \
bash scripts/run_edge_4090_full.sh
```

Set `PRECOMPRESSED_METADATA_ONLY=1` for an access check without weight shards,
or point `PRECOMPRESSED_LOCAL_ROOT` at a shared model disk. When
`PRECOMPRESSED_MODELS`, `PRECOMPRESSED_METHODS` and `PRECOMPRESSED_GROUPS` are
all unset, the wrapper prepares two safe groups:
`LLaMA3-8B:gptq,awq,smoothquant` and
`Qwen3-8B:gptq,awq,smoothquant,omniquant`. This avoids treating the missing
official LLaMA3-8B OmniQuant default as a blocker for the available public
LLaMA3-8B GPTQ/AWQ/SmoothQuant checkpoints. Override those groups with
`PRECOMPRESSED_GROUPS='MODEL:method,method;MODEL:method'` if needed. Override
checkpoint repos with comma-separated `PRECOMPRESSED_REPOS`, for example
`PRECOMPRESSED_REPOS='LLaMA3-8B:omniquant=<repo_id>'`, and set
`PRECOMPRESSED_FORCE=1` to redownload an existing staged checkpoint. Use that
when a previous Qwen3-8B GPTQ directory was staged from a GPTQModel-format repo
and the default stable AutoGPTQ checkpoint should replace it. Set
`PRECOMPRESSED_PREPARE_ONLY=1` to stop after writing the precompressed report
and generated config. The wrapper
writes `edge_prepare_precompressed_report.json` and
`edge_matrix_with_precompressed.json` beside the other run artifacts, including
during `DRY_RUN=1`, then uses that generated config for asset audit, preflight
and matrix execution.

The first command is a non-destructive dry-run path. The second command audits
assets, runs preflight, prewarms lm-eval datasets and runs the full matrix with
finalization, then reruns `tools/audit_edge_completion.py --require-finalize-report`
with runtime and code-match gates as the final delivery gate. The wrapper also
writes `edge_handoff_manifest.json` beside the run artifacts, or under
`DRY_RUN_OUTPUT_ROOT` during dry runs, so every run directory carries a
machine-readable setup and audit command manifest. It writes
`target_readiness.json` before starting the matrix, after asset audit, preflight
and optional prewarm have finished; non-dry-run launches stop there by default
when `ready_for_matrix=false`. Set `FAIL_ON_TARGET_READINESS_BLOCKERS=0` only
for diagnostic runs where you want the matrix command to enforce the downstream
gates itself. In non-dry-run
mode the wrapper also passes `--fail-on-blocked-methods` to preflight, so a
missing compression backend, missing external command or incomplete
precompressed checkpoint fails before any long matrix job starts. Set
`FAIL_ON_BLOCKED_METHODS=0` only for diagnostic runs where you intentionally
want preflight to record blockers without stopping. Set
`EXPORT_RESULTS_BUNDLE=0` to skip the final `edge_results_bundle` export, or
`VERIFY_RESULTS_BUNDLE=0` to skip verifying that bundle after export. By
default, successful non-dry-run wrapper executions export
`edge_results_bundle/`, `edge_results_bundle.tar.gz` and
`edge_results_bundle_verify.json` under the output root. When verification is
enabled, the wrapper also writes `edge_delivery_verify.json`, which combines
the root `completion_audit.json`, `target_readiness.json` and the results
bundle verifier into one final delivery verdict.
Set
`RUN_PREPARE=1` or `RUN_METADATA_PREPARE=1` if this wrapper should also call
`tools/prepare_edge_assets.py`. Those prepare steps write
`edge_prepare_assets_report.json` and `edge_prepare_metadata_report.json` under
the final output root, or under `DRY_RUN_OUTPUT_ROOT` during dry runs. If full
checkpoints already exist on the target machine or a shared disk, set
`LINK_EXISTING_MODELS=1 MODEL_SOURCE_ROOT=/path/to/models` so the wrapper
symlinks complete checkpoints into the paths configured by
`configs/edge_4090_matrix.json` before falling back to hf-mirror downloads.

```bash
python tools/prepare_edge_assets.py --config configs/edge_4090_matrix.json --dry-run --output /tmp/edge_prepare_dry_run.json
python tools/prepare_edge_assets.py --config configs/edge_4090_matrix.json --skip-lm-eval-probe --output /data/outputs/llmcbench_edge_4090/edge_prepare_assets_report.json
python tools/prepare_edge_assets.py --config configs/edge_4090_matrix.json --model-source-root /opt/tiger/models --link-existing-models --skip-lm-eval-probe --output /data/outputs/llmcbench_edge_4090/edge_prepare_assets_report.json
python tools/audit_edge_assets.py --config configs/edge_4090_matrix.json --min-free-disk-gb 80
python tools/build_edge_handoff_manifest.py --config configs/edge_4090_matrix.json --output-root /data/outputs/llmcbench_edge_4090 --output /data/outputs/llmcbench_edge_4090/edge_handoff_manifest.json
python tools/run_edge_preflight.py --config configs/edge_4090_matrix.json --check-hf-access --check-model-forward --finalize-dry-run --cuda-visible-devices 0 --check-optional-backend-resolution --check-quant-runtime --output /data/outputs/llmcbench_edge_4090/edge_preflight_report.json
python tools/run_smoke_checks.py
python tools/prewarm_lm_eval_tasks.py --config configs/edge_4090_matrix.json --only-model Qwen3-8B --dry-run
python tools/run_edge_matrix.py --config configs/edge_4090_matrix.json --dry-run
python tools/prewarm_lm_eval_tasks.py --config configs/edge_4090_matrix.json --only-model Qwen3-8B --expand-mmlu --sleep-seconds 60 --report-output /data/outputs/llmcbench_edge_4090/edge_prewarm_report.json
python tools/run_edge_matrix.py --config configs/edge_4090_matrix.json --stop-on-error --finalize --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report --require-optional-backends-report --require-quant-runtime-report --require-target-paths-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree
```

#### Calibration Input Contract

`calibration_samples` is a strict input contract for compression, not only
metadata. Text-file mode requires at least the requested number of non-empty
lines and fails before model loading if the file is too short. The strict
strict matrix uses pinned C4 windows for Wanda, SparseGPT and GPTQ, so a
configuration such as `calibration_samples=128` cannot silently compress with
five examples.

The checked-in Wanda/SparseGPT configs use the pinned official-style C4
protocol above, not `calibration_file`. The text file is only used by
`--pruning-calibration-source text_file` and by in-tool GPTQ/AWQ paths.
`prepare_target_paths.py` and `prepare_edge_assets.py` create a 128-line
bootstrap text file when it does not exist. It is useful for checking the
end-to-end workflow, but it repeats a small set of generic prompts and is
**not** a strict-profile calibration corpus. Existing short text files are
preserved and reported as readiness failures when that source is selected.

To run exactly the C4 pruning settings for one model/method:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export CALIBRATION_SAMPLES=128
export SEQLEN=2048
export PRUNING_CALIBRATION_SOURCE=c4
export CALIBRATION_SEED=0
export CALIBRATION_DATASET=allenai/c4
export CALIBRATION_DATASET_CONFIG=allenai--c4
export CALIBRATION_C4_DATA_FILE=en/c4-train.00000-of-01024.json.gz
export CALIBRATION_C4_REVISION=1588ec454efa1a09f29cd18ddd04fe05fc8653a2
export CALIBRATION_DATASET_CACHE=/data/llm_eval_datasets
export SPARSEGPT_BLOCKSIZE=128
export SPARSEGPT_PERCDAMP=0.01
export PRUNING_BUFFER_DEVICE=cpu
```

Run both strict-profile pruning methods on both target models with:

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_ENDPOINT=https://hf-mirror.com \
python tools/run_edge_matrix.py \
  --config configs/edge_paper_aligned_matrix.json \
  --only-method wanda \
  --only-method sparsegpt \
  --output-root /data/outputs/llmcbench_strict_pruning_4090 \
  --stop-on-error
```

Run `--finalize` after all four jobs complete to produce the normalized CSV,
JSON, report and scorecard. Do not reuse the historical H20 `wanda` or
`sparsegpt` checkpoints for this run: they used five text lines and the older
approximation backend.

Run only one model or method while debugging:

```bash
python tools/run_edge_matrix.py \
  --config configs/edge_4090_matrix.json \
  --only-model LLaMA3-8B \
  --only-method gptq \
  --stop-on-error
```

The runner skips jobs that already have `summary.json`, so interrupted
experiments can be resumed with the same command. To rebuild a job, add
`--force`; this also sets `OVERWRITE_COMPRESSED=1` so the old compressed
checkpoint directory is removed before recompression.
To inspect an interrupted output root and generate scoped resume commands, run:

```bash
python tools/plan_edge_matrix_resume.py \
  --config configs/edge_4090_matrix.json \
  --output-root /data/outputs/llmcbench_edge_4090 \
  --output /data/outputs/llmcbench_edge_4090/matrix_resume_plan.json
```

The plan classifies every selected model/method pair as `complete`,
`not_started`, `incomplete_existing`, `failed`, `dry_run` or
`skipped_external_not_configured`. It emits `run_missing`,
`rerun_incomplete_or_failed`, `resolve_external` and finalization command
groups so a long 4090 matrix can be resumed without hand-editing the full
matrix command.
The matrix config supports scoped overrides for job-specific settings:
`method_overrides`, `model_overrides` and `model_method_overrides` keyed as
`"<model>/<method>"`. Use these for values such as `seqlen`,
`calibration_samples`, `batch_size`, `max_memory`, `attn_implementation`,
`python_bin` or `runtime_profile` when one method needs a different 4090 memory
or dependency profile. The checked-in 4090 config uses this for SmoothQuant:
`quant_bits=8` for W8A8 checkpoints and
`python_bin=${LLMCBENCH_SMOOTHQUANT_PYTHON:-python3}` so the same matrix command
can dispatch SmoothQuant rows through a Transformers 4.57.x environment while
other rows keep the default Python. If you run this way, set:

```bash
export LLMCBENCH_SMOOTHQUANT_PYTHON=/path/to/llmcbench-edge-smoothquant/bin/python
```

The final rows record their actual `runtime_profile`, Python, Torch,
Transformers and CUDA metadata in `matrix_results.json`.
Each job writes `<output_root>/<model>/<method>/job.log`. The matrix run report
and rendered Markdown report include the log path, and failed jobs include a
`log_tail` excerpt for quick diagnosis.
Add `--finalize` to automatically run result collection, validation and report
rendering after the matrix completes. If the matrix already exists and you only
need to regenerate final artifacts, use `--finalize-only`; it refreshes
`matrix_config_resolved.json` from the current matrix config before collecting
results.

For `smoothquant` and `omniquant`, fill `external_commands` in
`configs/edge_4090_matrix.json` or export `LLMCBENCH_SMOOTHQUANT_CMD` /
`LLMCBENCH_OMNIQUANT_CMD`. If neither is configured, the matrix runner records
the job under `skipped_external_not_configured` instead of failing late.
Final audit accepts this status only when `--allow-skipped-external` is passed;
otherwise skipped external methods remain an explicit completion blocker.
Use `tools/configure_external_methods.py` to validate the template placeholders
before the full matrix:

```bash
python tools/configure_external_methods.py \
  --config configs/edge_4090_matrix.json \
  --smoothquant-command 'python /opt/SmoothQuant/export.py --model {model_path} --output {output_dir} --calib {calibration_file} --seq-len {seqlen}' \
  --omniquant-command 'python /opt/OmniQuant/export.py --model {model_path} --output {output_dir} --calib {calibration_file} --wbits {quant_bits} --group-size {group_size}' \
  --dry-run \
  --fail-on-missing
```

Remove `--dry-run` to write the rendered templates into the matrix config.
`configs/edge_4090_matrix.json` sets `split_accuracy_tasks=true` so lm-eval
tasks run one at a time and merge into `accuracy.json`. This is useful when a
large task group such as MMLU hits dataset resolver rate limits and you want the
other task results to survive.
It also sets `continue_on_accuracy_error=true` so completed task outputs are
kept and failed tasks are recorded instead of discarding the whole accuracy
step. Final validation still rejects rows with nonzero failed-task counts.
`tools/prewarm_lm_eval_tasks.py --config configs/edge_4090_matrix.json` reads
the model, dtype, attention backend, cache and task settings from the matrix
config. By default it uses the first selected/available model to warm shared
lm-eval datasets; add `--all-models` only when you want to verify every model
path during prewarm.
Use `--report-output <path>` to write the expanded task list, target model paths,
attempted task/model pairs and failures as JSON. The one-command 4090 wrapper
writes this report to `<output_root>/edge_prewarm_report.json`, where
`output_root` comes from the matrix config unless `PREWARM_REPORT_OUTPUT` is
set. During `DRY_RUN=1`, the default report path follows `DRY_RUN_OUTPUT_ROOT`
instead of the final matrix output root. When `RUN_PREWARM=1`, the final
completion audit requires this report and checks that it has `"ok": true`,
non-empty expanded tasks/attempts, enough attempted entries to cover every
expanded task for every prewarm target, no missing model/task pairs when target
names are recorded, matching succeeded model/task coverage, coverage of all
configured matrix tasks after MMLU expansion, and no failed prewarm tasks. Set
`REQUIRE_PREWARM_ALL_MODELS=1` in the wrapper, or pass
`--require-prewarm-all-models` manually, only when the prewarm report must cover
every selected matrix model rather than one shared dataset-cache warmup target.
The one-command wrapper writes asset and preflight reports beside the prewarm
report by default: `<output_root>/edge_asset_audit.json` and
`<output_root>/edge_preflight_report.json`, or under `DRY_RUN_OUTPUT_ROOT` during
dry-runs. Override them with `ASSET_AUDIT_OUTPUT` and `PREFLIGHT_OUTPUT`. In
non-dry-run mode the wrapper requires both reports in the final completion
audit. When `RUN_TARGET_PATH_PREPARE=1`, it requires
`edge_target_paths_report.json` in the final completion audit. When
`RUN_PREPARE=1` or `RUN_METADATA_PREPARE=1`, it also requires the corresponding
prepare report in the final completion audit.

After all jobs finish, collect the final result tables. The shortest path is:

```bash
python tools/run_edge_matrix.py \
  --config configs/edge_4090_matrix.json \
  --finalize-only
```

This runs the same three commands explicitly shown below:

```bash
python tools/collect_edge_results.py \
  --root /data/outputs/llmcbench_edge_4090
python tools/validate_edge_results.py \
  --config configs/edge_4090_matrix.json \
  --results /data/outputs/llmcbench_edge_4090/matrix_results.json
python tools/render_edge_report.py \
  --results /data/outputs/llmcbench_edge_4090/matrix_results.json \
  --run-report /data/outputs/llmcbench_edge_4090/matrix_run_report.json \
  --prewarm-report /data/outputs/llmcbench_edge_4090/edge_prewarm_report.json \
  --output /data/outputs/llmcbench_edge_4090/edge_report.md
```

This writes:

```text
/data/outputs/llmcbench_edge_4090/matrix_config_resolved.json
/data/outputs/llmcbench_edge_4090/edge_asset_audit.json
/data/outputs/llmcbench_edge_4090/edge_preflight_report.json
/data/outputs/llmcbench_edge_4090/edge_prewarm_report.json
/data/outputs/llmcbench_edge_4090/edge_handoff_manifest.json
/data/outputs/llmcbench_edge_4090/edge_target_paths_report.json
/data/outputs/llmcbench_edge_4090/edge_optional_backends_report.json
/data/outputs/llmcbench_edge_4090/edge_quant_runtime_report.json
/data/outputs/llmcbench_edge_4090/edge_prepare_assets_report.json
/data/outputs/llmcbench_edge_4090/edge_prepare_metadata_report.json
/data/outputs/llmcbench_edge_4090/matrix_run_report.json
/data/outputs/llmcbench_edge_4090/matrix_finalize_report.json
/data/outputs/llmcbench_edge_4090/matrix_results.json
/data/outputs/llmcbench_edge_4090/matrix_results.csv
/data/outputs/llmcbench_edge_4090/edge_report.md
/data/outputs/llmcbench_edge_4090/edge_scorecard.json
/data/outputs/llmcbench_edge_4090/edge_scorecard.md
/data/outputs/llmcbench_edge_4090/completion_audit.json
/data/outputs/llmcbench_edge_4090/target_readiness.json
/data/outputs/llmcbench_edge_4090/edge_results_bundle.tar.gz
/data/outputs/llmcbench_edge_4090/edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json
/data/outputs/llmcbench_edge_4090/edge_results_bundle_verify.json
/data/outputs/llmcbench_edge_4090/edge_delivery_verify.json
```

Generate the scorecard independently when needed:

```bash
python tools/summarize_edge_results.py \
  --results /data/outputs/llmcbench_edge_4090/matrix_results.json \
  --output-json /data/outputs/llmcbench_edge_4090/edge_scorecard.json \
  --output-md /data/outputs/llmcbench_edge_4090/edge_scorecard.md \
  --min-accuracy-retention 0.9 \
  --max-peak-memory-gb 24
```

The scorecard ranks methods per model using normalized accuracy, tokens/s,
peak memory, effective storage and effective FLOPs, and marks a simple Pareto
set for method selection. Real quantized checkpoints use their physical
checkpoint size. Wanda/SparseGPT use the explicitly labeled theoretical sparse
storage/FLOPs estimates. Dense and fake-quantized checkpoints use physical
storage and unadjusted FLOPs, so incidental zero-valued tensors are not counted
as compression savings.
Use `--weights` and constraint flags such as
`--min-accuracy-retention`, `--max-peak-memory-gb`, `--max-storage-gb`,
`--min-tokens-per-second` and `--max-flops-num` to generate scorecards for
different edge-device priorities.

After `completion_audit.json` reports `"ok": true`, export a portable results
bundle for delivery or migration:

```bash
python tools/export_edge_results_bundle.py \
  --output-root /data/outputs/llmcbench_edge_4090 \
  --bundle-dir /data/outputs/llmcbench_edge_4090/edge_results_bundle \
  --archive /data/outputs/llmcbench_edge_4090/edge_results_bundle.tar.gz \
  --include-job-logs \
  --require-complete \
  --force
```

The bundle includes the final JSON/CSV/Markdown results, scorecard,
runtime/readiness reports, target path preparation report, completion audit,
current-goal audit when present, optional prepare reports when present, copied job logs,
`tools/verify_edge_results_bundle.py`,
`tools/verify_edge_delivery.py`, `RESULTS_BUNDLE_MANIFEST.json` with checksums,
and `RESULTS_BUNDLE_README.md`. `--require-complete` prevents exporting a
final delivery bundle unless every required artifact exists and the completion
audit has `ok=true` with `validator.ok=true` and target readiness has `ok=true`
with `completion_ok=true` and no `before_matrix_blockers`.
After moving the bundle, verify it before using the numbers:

```bash
python /data/outputs/llmcbench_edge_4090/edge_results_bundle/tools/verify_edge_results_bundle.py \
  --bundle-root /data/outputs/llmcbench_edge_4090/edge_results_bundle \
  --require-complete \
  --require-target-readiness-ok \
  --require-target-runtime
python tools/verify_edge_delivery.py \
  --root /data/outputs/llmcbench_edge_4090 \
  --results-bundle /data/outputs/llmcbench_edge_4090/edge_results_bundle \
  --results-archive /data/outputs/llmcbench_edge_4090/edge_results_bundle.tar.gz \
  --output /data/outputs/llmcbench_edge_4090/edge_delivery_verify.json
```
By default, the verifier also checks `matrix_results.json` model/method
coverage and target-task score columns against the bundled
`matrix_config_resolved.json`; pass `--config` to compare against another
matrix config. It also checks the deployment and resource metric fields needed
by the requested deliverables, including tokens/s, peak memory, storage, params
and FLOPs, and rejects rows that exceed the recorded deployment memory limit.
With `--require-target-readiness-ok`, it requires bundled `target_readiness.json`
to report `ok=true`, `completion_ok=true` and no `before_matrix_blockers`.
With `--require-target-runtime`, it also checks the result rows were
recorded on the required Python 3.10 / CUDA 12.x / single RTX 4090-class
environment using `HF_ENDPOINT=https://hf-mirror.com`.
The delivery verifier also compares the output root, `edge_results_bundle/`
directory and `edge_results_bundle.tar.gz` archive by size and SHA256 for the
required final result files, including `edge_target_paths_report.json`.
It also checks the root `completion_audit.json`
with the same `validator.ok=true` gate as the results-bundle verifier, so stale
directories are rejected, stale tarballs are rejected, and hand-edited
completion-audit files are rejected before the bundle is moved to another host.

`matrix_config_resolved.json` snapshots the selected models/methods, effective
defaults, overrides, external command templates and precompressed paths used for
the run, plus the git branch/commit/remote, so result tables remain reproducible
after the source config or code changes.
`tools/audit_edge_completion.py` uses this resolved snapshot for final
validation when it is present, so historical results can be re-audited even if
`configs/edge_4090_matrix.json` is later edited.
The summary table and report include model name, compression method,
compression backend, compression dtype, compression parameters such as bits,
group size, target sparsity, calibration file, calibration sample count and
calibration batch size, accuracy metrics,
accuracy dtype/batch/split settings, deployment prompt/source and generation settings, peak GPU memory,
tokens/s, storage size, effective sparse and quantized storage estimates,
sparsity, resource prompt/source and sequence-length settings, FLOPs and MACs,
plus Python/Torch/Transformers/CUDA/GPU runtime
metadata. They also
include deployment memory-limit fields so 4090-incompatible results are visible
in the final artifacts.
For each model, the collector also computes average target-task accuracy,
absolute delta from that model's dense row and retention ratio versus dense, so
accuracy preservation is visible directly in `matrix_results.json` and
`edge_report.md`. ARC-Challenge, HellaSwag and PIQA prefer `acc_norm` when
lm-eval provides it; each task score also records the selected metric key in the
JSON/CSV output and `edge_report.md`. Split accuracy runs also carry
`accuracy_failed_task_count` and `accuracy_failed_tasks`; final validation
rejects rows with any failed target task or non-empty `accuracy_limit`, so
`limit=1` smoke runs cannot be accepted as final results. The collected
`accuracy_num_fewshot` value must also match the matrix configuration, including
`None` when no few-shot override is configured. Accuracy artifacts
record `data_dir` and `HF_DATASETS_CACHE`, and those local dataset/cache paths
are carried into `matrix_results.json` so archived results preserve the dataset
path provenance.
`completion_audit.json` is the final machine-readable gate: it checks the
expected model/method coverage, compression backend provenance, duplicate result
rows, required metric fields, matrix result format errors, matrix run failures,
skipped external methods, non-numeric or non-finite quantitative metrics,
non-integer counters in integer-valued fields, required deployment memory-gate
fields, out-of-range quantitative metrics, inconsistent deployment memory-gate
fields, deployment token-count upper bounds, tokens/s-to-elapsed-time consistency,
and the current-goal audit's final matrix completion gate also requires every
row to carry method-specific allowed compression backend provenance plus
`compression_materialized` and `compression_uses_precompressed` booleans,
and `external_command` rows must record a non-empty
`compression_external_command` with `compression_external_returncode=0`,
so manually labeled or ambiguous compressed checkpoints cannot count as final
matrix completion.
prompt source and HF cache
provenance, memory-limit violations, resource unit conversions and report
artifacts, plus exact CSV/JSON value consistency for every matrix result column.
`matrix_run_report.json` jobs must be final statuses such as `completed` or
validated `skipped_existing`; dry-run jobs and top-level dry-run run reports
are rejected as final evidence.
It also re-renders `edge_report.md` from the current result JSON and run report,
then compares it with the archived Markdown so stale table content is rejected.
The report must name the current result JSON path and row count, and
`matrix_finalize_report.json` must point at the standard final artifacts under
the current output root. Finalize dry-runs also write
`matrix_finalize_report.json` with `status=dry_run` steps so target hosts can
audit paths and gates before launching the long run; final completion audit
rejects those dry-run steps as non-final evidence. In the strict 4090 workflow,
the final audit also requires `target_readiness.json` to exist, report
`ok=true` and `completion_ok=true`, reference the same output root and
completion audit path,
and contain no `before_matrix_blockers`. `matrix_config_resolved.json` must
preserve the same
source config path, selected model entries including paths, methods, source
defaults, current output root, external command templates, precompressed paths
and override maps implied by the source config and audit filters, so a stale,
shrunk or weakened resolved snapshot cannot reduce final coverage.
Required auxiliary reports such as asset preparation, asset audit, preflight
and quant runtime reports cannot be dry-run reports in final completion audit.
The optional backend dependency-resolution report is allowed to be a pip
dry-run report because that is its expected scope, but it must include
`mode=dry_run`, non-empty requested package/import summaries and
`pip_returncode=0`. `gptqmodel_install` is a conditional gate: it is required
only when the optional backend request or checkpoint metadata actually selects a
GPTQModel-format runtime, not for the default AutoGPTQ Qwen3-8B GPTQ checkpoint.
The helper may still record a Python-version-marked GPTQModel capability entry
for override checkpoints; that capability record is not a blocker for the
default stable path.
The metadata preparation report is allowed to be
`metadata_only=true`; full asset preparation reports are not. Required full
`edge_prepare_assets_report.json` must contain non-empty model and file
preparation items, and every item must pass. Required
`edge_prepare_metadata_report.json` must be `metadata_only=true`, contain
non-empty model and file metadata-preparation items, and every item must pass.
Required `edge_asset_audit.json` must include model,
precompressed-checkpoint and path sections covering `hf_cache_dir`, `data_dir`,
`output_root`, `calibration_file` and `prompt_file`, including directory
writability/create checks via `unwritable_paths` and per-path `can_create`.
Required
`edge_quant_runtime_report.json` must contain non-empty probe
rows and every row must pass. Required `edge_prepare_precompressed_report.json`
must contain non-empty successful preparation items, no blockers, and an
existing valid `updated_config` JSON path that matches the config passed to
readiness/final audit. `edge_preflight_report.json` must
include the expected `asset_audit`, `verify_env`, `method_readiness` and `matrix_dry_run`
steps, and all required steps must pass. When optional backend or quant runtime
final reports are required, the preflight report must also include
`optional_backend_resolution` and `quant_runtime` steps respectively. Final
`edge_prewarm_report.json` cannot be a
dry-run report; preflight dry-runs are written separately as
`edge_prewarm_dry_run_report.json`. Matrix run job summary/log paths must stay
under the output root, unexpected run jobs are rejected unless explicitly allowed,
`matrix_run_report.json` top-level status
lists must match the scoped job details without missing or stale extra entries,
job details must not contain duplicate model/method records, each job summary's
model/method identity must match the matrix job, and
`matrix_results.json` must record the same `compressed_model_path` as the
corresponding job summary.
Compression source path, backend, materialization flag, precompressed-input
flag and compression parameter columns are compared back to each job's
`compression.json` manifest and are rendered in `edge_report.md` so external
precompressed checkpoints can be distinguished from checkpoints exported by this
toolkit. Method-specific compression parameters must be present: pruning rows
keep `compression_sparsity` and calibration sample counts; GPTQ/AWQ rows keep
quant bits, group size and calibration sample counts; SmoothQuant/OmniQuant rows
keep quant bits, group size, target sparsity and calibration sample counts.
Accuracy scores, selected metric keys, MMLU
aggregation, full MMLU subtask coverage, average score and failed-task fields
are recomputed from each job's `accuracy.json`. Final validation also checks
MMLU and target-task aggregate score consistency and selected metric-key
provenance. Key deployment and resource
columns in `matrix_results.json`
are also compared back to each job's `deployment.json` and `resource.json`,
including tokens/s, peak memory, storage, sparsity,
FLOPs/MACs and compute backend fields. Storage GB/bytes, sparse effective
storage/FLOPs, quantized effective storage, `sparsity`/`nonzero_ratio`, resource
backend labels, deployment token counts, tokens/s from generated tokens and
elapsed seconds, prompt-source labels, HF cache paths and attention backend
fields must be self-consistent.
Dense-relative accuracy delta and retention must match each model's dense row.
Provenance, score-metric and runtime version fields must be strings. It also
records SHA256 checksums for the final result and report artifacts so archived
outputs can be checked for later drift.
The one-command 4090 wrapper passes runtime gates to the final audit in real
mode: Python must start with `3.10` and the recorded GPU name must contain
`4090`; recorded PyTorch must start with `2.`, Transformers must be recorded,
CPU count must be at least 8, system memory must be at least 240GB by default,
the recorded CUDA runtime version must start with `12.`, and recorded GPU total
memory must be at least 23GB by default, matching a 24GB 4090-class device
while allowing small driver/reporting differences. cuDNN must also be available
and recorded in the final runtime metadata. The same strict gate requires
`runtime_cuda_available=true` whenever GPU/CUDA/cuDNN requirements are enabled,
and `validate_edge_results.py` rejects final rows that report GPU metadata while
CUDA was unavailable. `HF_ENDPOINT` must be
`https://hf-mirror.com`.
Override this only if your standardized target environment intentionally uses a
different CUDA major version by setting `REQUIRE_RUNTIME_CUDA_PREFIX` or a
different memory floor by setting `REQUIRE_RUNTIME_GPU_TOTAL_MEMORY_MIN_GB`.
Set `REQUIRE_RUNTIME_TORCH_PREFIX`, `REQUIRE_RUNTIME_TRANSFORMERS=0`, or
`REQUIRE_RUNTIME_CUDNN=0` only for a deliberately different target stack. Use
`REQUIRE_RUNTIME_CPU_COUNT_MIN` and `REQUIRE_RUNTIME_SYSTEM_MEMORY_MIN_GB` only
when your standardized host hardware intentionally differs; use
`REQUIRE_RUNTIME_HF_ENDPOINT` only for an intentionally different mirror. It also
passes `--require-code-match`, so the resolved config code
commit, branch and origin must match the checkout running the final audit, and
`--require-clean-worktree`, so final delivery runs must come from a clean git
worktree. Override `REQUIRE_RUNTIME_PYTHON_PREFIX` or
`REQUIRE_RUNTIME_GPU_NAME_CONTAINS` only when intentionally running a different
standard platform.

### Portable 4090 Setup Checklist

On a fresh 4090 machine:

Generate a machine-readable handoff manifest before making target-specific
changes:

```bash
python tools/build_edge_handoff_manifest.py \
  --config configs/edge_4090_matrix.json \
  --output /data/outputs/llmcbench_edge_4090/edge_handoff_manifest.json
```

The manifest records the git commit, configured model/data/output paths, target
4090 runtime gates, memory controls, final artifact paths and the exact setup,
prepare, optional precompressed-checkpoint preparation, preflight, prewarm,
readiness, run and audit commands. Run the manifest `prewarm` command before the
`readiness` command when using the strict handoff, because readiness requires
the generated `edge_prewarm_report.json`. The `prepare_precompressed` command
stages known public Qwen GPTQ/AWQ/SmoothQuant/OmniQuant checkpoints and writes
`edge_matrix_with_precompressed.json`; use that generated config for a matrix
run that should avoid recompressing those quantized checkpoints. It is safe to
generate before model downloads because it only reads the matrix config and
existing artifact paths.

To move the runnable toolkit to another 4090 host without relying on the
current checkout, export a portable bundle:

```bash
python tools/export_edge_portable_bundle.py \
  --config configs/edge_4090_matrix.json \
  --output-root /data/outputs/llmcbench_edge_4090 \
  --output-dir /tmp/llmcbench_edge_portable \
  --archive /tmp/llmcbench_edge_portable.tar.gz \
  --force
```

The bundle contains `README.md`, configs, `edge_benchmark`, `scripts`, `tools`,
requirements, completion docs, `edge_handoff_manifest.json`,
`PORTABLE_BUNDLE_MANIFEST.json` with file checksums, and `RUN_ON_TARGET.md`
with the target-side setup and run commands. It does not include model weights,
datasets or final benchmark outputs; those remain under the configured target
paths and are validated by the asset/readiness/completion audits.
The portable verifier treats the migration-critical scripts as required files,
including environment validation, SmoothQuant/OmniQuant external-method
configuration, GPTQ/AWQ/SmoothQuant runtime probing, goal audit, result
collection/rendering, final validation and delivery verification tools.
After copying or extracting the bundle on the target host, verify the portable
bundle before installing dependencies. Include `--archive` when the tarball is
available so the verifier also checks that the copied archive matches the
extracted bundle directory, including the size and SHA256 of every archived
file listed in `PORTABLE_BUNDLE_MANIFEST.json` and the archived
`PORTABLE_BUNDLE_MANIFEST.json` itself. It also rejects duplicate or unsafe
archive paths, non-file/non-directory archive members, and tarballs whose
single top-level directory does not match the extracted bundle directory:

```bash
python tools/verify_edge_portable_bundle.py \
  --bundle-root /tmp/llmcbench_edge_portable \
  --archive /tmp/llmcbench_edge_portable.tar.gz \
  --require-clean-code \
  --require-commit $(git rev-parse HEAD) \
  --output /tmp/llmcbench_edge_portable/portable_bundle_verify.json
```

For a delivery bundle, keep `--require-clean-code` enabled. It rejects bundles
exported from a dirty worktree, and `--require-commit` ensures the bundle and
handoff manifest match the exact code revision you intend to run on the target
4090 host.

1. Clone the repository and install dependencies.

```bash
git clone git@code.byted.org:data/llmcbench.git
cd llmcbench
git checkout master

conda create -n llmcbench-edge python=3.10 -y
conda activate llmcbench-edge
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
```

Or use the helper:

```bash
bash scripts/setup_edge_env.sh
```

The helper prefers `mamba`/`conda`. If neither is available, it falls back to a
local `.venv` when `python3.10` is installed. It defaults to
`EDGE_RUNTIME_PROFILE=stable`, which installs the Torch 2.6 / CUDA 12.4 stack
used for the AWQ-oriented diagnostics. To test Qwen GPTQModel checkpoints, use
the separate target profile so GPTQModel's Torch/NumPy/Transformers constraints
are installed after the base requirements:

```bash
EDGE_RUNTIME_PROFILE=gptqmodel bash scripts/setup_edge_env.sh
```

Final rows may use different runtime profiles only when the matrix config
declares the per-method or per-model override. The validator checks each row
against that effective config and every row records its actual runtime metadata.
If you change a profile, rerun preflight, runtime probes, target readiness and
the affected matrix rows before exporting the final bundle.
`requirements.txt` contains the base benchmark stack only: Transformers,
lm-eval, datasets, calflops and shared utilities. Optional compression backend
packages are kept in `requirements-optional.txt` so a GPTQ/AWQ install issue
does not break the base accuracy/deployment/resource tools.
Install GPTQ/AWQ/SmoothQuant loading backends (`optimum<2`, `auto-gptq`,
`gptqmodel`, `autoawq`, `compressed-tensors`) inside this isolated
environment. Avoid adding them to a shared system Python because they can pull
CUDA/PyTorch-adjacent dependencies. Use the optional backend helper after the
base environment is active:

```bash
OPTIONAL_BACKEND_DRY_RUN=1 REPORT_OUTPUT=/data/outputs/llmcbench_edge_4090/edge_optional_backends_report.json bash scripts/setup_optional_backends.sh
bash scripts/setup_optional_backends.sh
python tools/check_method_readiness.py --config configs/edge_4090_matrix.json
```

To install only part of the optional stack, set `INSTALL_BITSANDBYTES=0`,
`INSTALL_GPTQ=0`, `INSTALL_GPTQMODEL=0`, or `INSTALL_AWQ=0` before running the
helper.
`OPTIONAL_BACKEND_DRY_RUN=1` runs `pip install --dry-run --report` and writes a
JSON summary with the resolved packages and current import status; it does not
install or upgrade anything. Use it first on the target 4090 environment if you
need an audit trail before pulling GPTQ/AWQ dependencies. The helper keeps
`gptqmodel==4.0.0` out of the shared `requirements-optional.txt` install and
handles it as a separate Python 3.10+ step with `--no-build-isolation`; the dry
run records this under `gptqmodel_install`, including the GPTQModel requirement
check for Python `>=3.10`, Torch `>=2.7.1`, NumPy `>=2.2.6` and Transformers
`>=4.55.2`. The default Qwen3-8B GPTQ precompressed repo is
`kaitchup/Qwen3-8B-autoround-4bit-gptq`, a public GPTQ 4-bit checkpoint whose
metadata does not advertise a GPTQModel-only quantizer, so it can use the
default GPTQ runtime path with `optimum<2` plus `auto-gptq`; the final audit
does not require `gptqmodel_install.requirements_ok=true` for that default path.
Optimum 2.x moved GPTQ loading onto GPTQModel symbols, so the stable optional
requirements intentionally pin `optimum<2`. If you override the repo to a
GPTQModel-format checkpoint, use
`EDGE_RUNTIME_PROFILE=gptqmodel` and rerun the runtime probes before including
those rows in a final bundle.
`optimum` is included because Transformers requires it when loading GPTQ
checkpoints. `compressed-tensors` is included because several SmoothQuant W8A8
exports use the Transformers `compressed-tensors` quantization path. These
optional backends are required for compressing/exporting GPTQ/AWQ inside this
toolkit and for loading quantized checkpoints during accuracy and deployment.
When `precompressed_paths` points to an already exported HuggingFace checkpoint,
the compression/export backend is skipped, but the loading runtime is still
checked.
The goal-audit snapshot uses the same convention: GPTQ/AWQ compression backends
and SmoothQuant/OmniQuant external commands are considered readiness blockers
only for methods that are not covered by complete `precompressed_paths` entries;
GPTQ/AWQ runtime libraries remain blockers until the checkpoint can be loaded.
Completeness means the configured checkpoint directory contains `config.json`
and at least one weight file such as `.safetensors`, `.bin` or `.pt`; a path
string alone is not treated as ready.

2. Prepare local model/data/cache paths.

```bash
sudo DATA_ROOT=/data bash scripts/init_edge_storage.sh
export HF_ENDPOINT=https://hf-mirror.com
python tools/prepare_target_paths.py --config configs/edge_4090_matrix.json --output /data/outputs/llmcbench_edge_4090/edge_target_paths_report.json --fail-on-error
python tools/prepare_edge_assets.py --config configs/edge_4090_matrix.json --metadata-only --skip-lm-eval-probe --continue-on-error --output /data/outputs/llmcbench_edge_4090/edge_prepare_metadata_report.json
python tools/prepare_edge_assets.py --config configs/edge_4090_matrix.json --skip-lm-eval-probe --continue-on-error --min-free-disk-gb 80 --output /data/outputs/llmcbench_edge_4090/edge_prepare_assets_report.json
python tools/prepare_precompressed_assets.py --config configs/edge_4090_matrix.json --local-root /data/models/precompressed --only-model Qwen3-8B --only-method gptq --only-method awq --only-method smoothquant --only-method omniquant --output-config /data/outputs/llmcbench_edge_4090/edge_matrix_with_precompressed.json --output /data/outputs/llmcbench_edge_4090/edge_prepare_precompressed_report.json --fail-on-error
python tools/audit_edge_assets.py --config configs/edge_4090_matrix.json --min-free-disk-gb 80
```

`scripts/init_edge_storage.sh` creates the standard `/data` directory tree and
hands ownership to the target user before Python tooling runs. Override
`DATA_ROOT`, `TARGET_USER`, `TARGET_GROUP` or `CHOWN_DATA_ROOT=0` only when the
target filesystem is already mounted and owned correctly.
`tools/prepare_target_paths.py` creates the configured cache, dataset, output,
model-parent and precompressed-model directories, writes default calibration and
prompt seed files when they are absent, and records the result in
`edge_target_paths_report.json`. The calibration seed contains the configured
number of non-empty lines; replace it with representative C4/Pile or in-domain
samples before a final quality run. The tool does not download model weights or
datasets.

When checkpoints have already been staged outside `/data/models`, reuse them
without another full download:

```bash
python tools/prepare_edge_assets.py \
  --config configs/edge_4090_matrix.json \
  --model-source-root /opt/tiger/models \
  --link-existing-models \
  --skip-lm-eval-probe \
  --continue-on-error \
  --output /data/outputs/llmcbench_edge_4090/edge_prepare_assets_report.json
```

The linker looks for complete checkpoints named by the model display name
(`Qwen3-8B`) or configured path basename (`Meta-Llama-3-8B`, `Qwen3-8B`). It
creates a symlink only when the target path is absent; an existing incomplete
target path is reported as a preparation error instead of being overwritten.
If no complete source checkpoint is found, the normal hf-mirror download path is
used.

If LLaMA3 access is gated in your HuggingFace account, login first:

```bash
huggingface-cli login
```

In the development environment, `Qwen/Qwen3-8B` config access through
`hf-mirror.com` was verified, while `meta-llama/Meta-Llama-3-8B` returned a
gated-repository 403 without an authorized token. Use `huggingface-cli login` or
place an already authorized LLaMA3 checkpoint at `/data/models/Meta-Llama-3-8B`.
The lm-eval datasets are downloaded into `HF_DATASETS_CACHE` during the first
accuracy run; `prepare_edge_assets.py` only checks the command path and creates
local calibration/prompt files.
For Wanda, SparseGPT, GPTQ and AWQ, the supplied calibration file must contain
at least the configured `calibration_samples` non-empty lines. This is checked
again by the compression entrypoint, so it cannot silently fall back to fewer
samples.
Use `--metadata-only` first to verify gated-model access and tokenizer/config
paths before downloading full 8B weight shards.
If `/data` is not writable, either create/chown it as shown above or override
the generated text files with `--calibration-file` and `--prompt-file`.
If `/data/hf_cache` is not writable, pass `--hf-home /path/to/writable/cache`.
For the final matrix, every configured directory path under `/data` must either
already exist and be writable or have a writable nearest existing parent.
`tools/audit_edge_assets.py` records this in `unwritable_paths` and per-path
`can_create` fields, and target readiness/final audits treat non-creatable
cache, dataset or output paths as before-matrix blockers.
`check_method_readiness.py` requires both `config.json` and model weight files;
metadata-only assets are not considered runnable for compression/evaluation.
`Qwen/Qwen3-8B` weight shards are about 16.4GB; reserve substantially more space
for dense checkpoints, compressed checkpoints and outputs. The examples use an
80GB free-disk preflight for asset preparation.
`tools/audit_edge_assets.py` is read-only and reports whether each configured
model has both `config.json` and weight shards for the selected methods that
need a base checkpoint, whether calibration/prompt files exist, and whether the
configured cache/data/output filesystems have enough free space. It also records
best-effort mount, filesystem, writability and block-device rotational metadata
for those paths, so the target run artifacts can show whether `/data` is
creatable and on SSD/NVMe-like storage. If
`precompressed_paths` is populated, those exported checkpoint directories are
audited with the same config-plus-weights rule and can satisfy a scoped
`--only-method` preflight without requiring the dense base checkpoint.

3. Verify before starting long jobs.

```bash
python tools/build_edge_handoff_manifest.py --config configs/edge_4090_matrix.json --output-root /data/outputs/llmcbench_edge_4090 --output /data/outputs/llmcbench_edge_4090/edge_handoff_manifest.json
python tools/run_edge_preflight.py --config configs/edge_4090_matrix.json --check-hf-access --check-model-forward --finalize-dry-run --cuda-visible-devices 0 --check-optional-backend-resolution --check-quant-runtime --output /data/outputs/llmcbench_edge_4090/edge_preflight_report.json
python tools/prewarm_lm_eval_tasks.py --config configs/edge_4090_matrix.json --only-model Qwen3-8B --attn-implementation eager --expand-mmlu --sleep-seconds 60
python tools/run_edge_matrix.py --config configs/edge_4090_matrix.json --dry-run
```

`tools/run_edge_preflight.py` runs the non-destructive checks needed before a
long matrix: environment verification, method readiness, optional backend
resolution, quantized runtime probes, prewarm dry-run and matrix dry-run. The
strict 4090 example also uses `--check-model-forward` to load the full local
checkpoints and run one forward pass before starting the matrix.
`--cuda-visible-devices 0` injects the same GPU binding into the environment
verification subprocess and requires it to match `defaults.cuda_visible_devices`
from the matrix config. Use
`--allow-blocked` only for diagnostics where preflight should record blockers
without failing. Use `--only-model` and `--only-method` to scope preflight while
debugging one model/method; the filters are passed through to environment
verification, method readiness and matrix dry-run.

`tools/check_method_readiness.py` emits per-model/per-method blockers plus
remediation commands. For GPTQ/AWQ it points to `scripts/setup_optional_backends.sh`.
For SmoothQuant/OmniQuant it prints the required `LLMCBENCH_*_CMD` external
command template.
`tools/run_edge_preflight.py --check-optional-backend-resolution` invokes the
optional backend helper in dry-run mode and writes
`edge_optional_backends_report.json` next to the preflight report. Add
`--fail-on-optional-backend-resolution` when dependency resolution should block
the preflight. The one-command wrapper exposes the same behavior through
`CHECK_OPTIONAL_BACKEND_RESOLUTION=1` and
`FAIL_ON_OPTIONAL_BACKEND_RESOLUTION=1`; when enabled for a non-dry-run matrix,
the wrapper also passes `--require-optional-backends-report` into finalization
and the final completion audit.
`CHECK_QUANT_RUNTIME=1` similarly writes `edge_quant_runtime_report.json`
through the quantized runtime probe and makes
`--require-quant-runtime-report` part of the final gate.

Model-forward preflight uses `defaults.dtype` from the matrix config unless
`--forward-dtype` is provided. If model forward fails, retry preflight with
`--forward-dtype bf16` or `--forward-attn-implementation eager` to distinguish
environment/backend issues.
For actual pipeline runs, set `ATTN_IMPLEMENTATION=eager` or configure
`attn_implementation` in the matrix JSON if the target environment needs a
specific Transformers attention backend.

4. Run and collect.

```bash
python tools/build_edge_handoff_manifest.py --config configs/edge_4090_matrix.json --output-root /data/outputs/llmcbench_edge_4090 --output /data/outputs/llmcbench_edge_4090/edge_handoff_manifest.json
python tools/run_edge_preflight.py --config configs/edge_4090_matrix.json --check-hf-access --check-model-forward --finalize-dry-run --cuda-visible-devices 0 --check-optional-backend-resolution --check-quant-runtime --output /data/outputs/llmcbench_edge_4090/edge_preflight_report.json
python tools/prewarm_lm_eval_tasks.py --config configs/edge_4090_matrix.json --only-model Qwen3-8B --expand-mmlu --sleep-seconds 60 --report-output /data/outputs/llmcbench_edge_4090/edge_prewarm_report.json
python tools/run_edge_matrix.py --config configs/edge_4090_matrix.json --stop-on-error --finalize --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report --require-optional-backends-report --require-quant-runtime-report --require-target-paths-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree
python tools/audit_edge_completion.py --config configs/edge_4090_matrix.json --root /data/outputs/llmcbench_edge_4090 --require-finalize-report --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report --require-optional-backends-report --require-quant-runtime-report --require-target-paths-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree
```

The strict manual commands above enable optional backend resolution and
quantized runtime probes in preflight, then require the corresponding reports in
both finalization commands. If you intentionally skip either probe for a
diagnostic run, remove the matching `--require-*` gate as well, or use
`scripts/run_edge_4090_full.sh` with the matching `CHECK_*` variables so the
wrapper keeps probe generation and final gates aligned.

### Current Result Status

The codebase and runner are ready for the requested two-model edge benchmark.
The latest completed run is a Python 3.10 / CUDA 12.4 diagnostic matrix on the
available H20, not final RTX 4090 acceptance evidence. It used
`CUDA_VISIBLE_DEVICES=1`, `HF_ENDPOINT=https://hf-mirror.com`,
`max_memory=0:22GiB,cpu:120GiB`, deployment batch size 1 and a 24GB deployment
memory gate.

Completed H20 diagnostic artifacts:

- `/data/outputs/llmcbench_h20_full_matrix_py310/matrix_results.json`
- `/data/outputs/llmcbench_h20_full_matrix_py310/matrix_results.csv`
- `/data/outputs/llmcbench_h20_full_matrix_py310/edge_report.md`
- `/data/outputs/llmcbench_h20_full_matrix_py310/completion_audit.json`
- `/data/outputs/llmcbench_h20_full_matrix_py310/smoothquant_runtime_probe.json`

The full H20 diagnostic matrix has 14 completed rows:
`LLaMA3-8B,Qwen3-8B` x
`dense,wanda,sparsegpt,gptq,awq,smoothquant,omniquant`. Its historical
validator reported `"ok": true` with 14 expected and 14 actual rows, no missing
task or MMLU-subtask metrics, and no deployment-memory violations. The pruning
rows, however, used a historical five-line calibration file despite requesting
64 samples; current validation deliberately rejects those four rows as
five-versus-64 calibration mismatches. The current official-aligned rerun
configuration uses 128 samples, so those historical pruning values are also not
comparable to the new benchmark protocol. See
`docs/H20_DIAGNOSTIC_RESULTS.md`; do not use the H20 pruning accuracies as
64-sample final comparison results.

SmoothQuant uses a separate compatible runtime profile:

- LLaMA3-8B and Qwen3-8B SmoothQuant both load and generate under a
  SmoothQuant-compatible Python 3.10 profile with Torch 2.6.0+cu124,
  Transformers 4.57.6 and compressed-tensors 0.15.0.1.
- The older stable profile with Transformers 4.55.2 fails Qwen3-8B SmoothQuant
  at model load with `Only Tensors of floating point and complex dtype can
  require gradients`, so SmoothQuant W8A8 rows use
  `LLMCBENCH_SMOOTHQUANT_PYTHON`.
- LLaMA3 SmoothQuant retains `99.60%` average accuracy and runs at
  `26.56 tokens/s`, but the staged checkpoint stores FP16 weights.
- Qwen3 SmoothQuant is real W8A8 storage and retains `99.86%` average accuracy,
  with `17.17GB` peak memory and `8.81GB` storage. Its compatibility path is
  slow at `0.276 tokens/s`.

The H20 results are engineering evidence only. They do not satisfy the final
4090 requirement because final acceptance must be produced on one RTX 4090
24GB-class GPU with the strict runtime gates enabled. The final 4090 run should
use an authorized official `meta-llama/Meta-Llama-3-8B` checkpoint if the
official repo is required; this H20 diagnostic used the public
`NousResearch/Meta-Llama-3-8B` checkpoint because the official Meta repo
returned 403 without an authorized token.

The current H20 diagnostic run is summarized in
`docs/H20_DIAGNOSTIC_RESULTS.md`. Re-check the scoped H20 artifacts with:

```bash
python tools/validate_edge_results.py \
  --config /data/outputs/llmcbench_h20_full_matrix_py310/h20_matrix_full_smoothquant_tf4576.json \
  --results /data/outputs/llmcbench_h20_full_matrix_py310/matrix_results.json \
  --output /data/outputs/llmcbench_h20_full_matrix_py310/validation_full_current.json

python tools/audit_edge_completion.py \
  --config /data/outputs/llmcbench_h20_full_matrix_py310/h20_matrix_full_smoothquant_tf4576.json \
  --root /data/outputs/llmcbench_h20_full_matrix_py310 \
  --output /data/outputs/llmcbench_h20_full_matrix_py310/completion_audit.json \
  --require-handoff-manifest \
  --require-runtime-python-prefix 3.10 \
  --require-runtime-cuda-prefix 12. \
  --require-runtime-gpu-total-memory-min-gb 23 \
  --require-runtime-torch-prefix 2. \
  --require-runtime-transformers \
  --require-runtime-bitsandbytes \
  --require-runtime-cpu-count-min 8 \
  --require-runtime-system-memory-min-gb 240 \
  --require-runtime-hf-endpoint https://hf-mirror.com \
  --require-code-match \
  --require-clean-worktree
```

Older Python 3.9 smoke notes and failed early probes are retained in
`docs/RUN_STATUS.md` as history, but the Python 3.10 H20 matrix above is the
current diagnostic evidence.

Regenerate the machine-readable current-goal audit with:

```bash
python tools/build_goal_audit.py
```

The audit writes `/opt/tiger/llmcbench_outputs/current_goal_audit.json` and
records checkpoint readiness, Qwen smoke metrics, method readiness blockers,
the remaining gates for the final two-model matrix and a
`requirement_checklist` mapping the requested accuracy, deployment and resource
tools to current evidence and blockers. It also records `final_artifacts` for
the expected 4090 output root, including whether `completion_audit.json` exists
and has `"ok": true`, plus final matrix coverage and completion-audit failure
summaries when those artifacts exist. The final artifact summary also includes
optional backend and quantized runtime reports, and mirrors their completion
audit gate status when those checks are enabled. It also surfaces the final
completion audit's MMLU subtask coverage status, including required subtask
count, scoped row count and violations when final artifacts exist. The audit
includes a `prompt_to_artifact_checklist` that maps the original requested
accuracy, deployment, resource, method coverage, target-platform and
final-result requirements to concrete repo files, output artifacts, gates,
evidence and remaining gaps. The top-level `completion_state` is the stable
machine-readable summary for handoff automation: it reports `status`,
`complete`, incomplete prompt sections, remaining blockers, missing final
artifact count, final output root, `results_bundle_verify_ok`,
`results_bundle_verify_complete`, `delivery_verify_ok`,
`delivery_verify_substatus_ok`, `target_completion_ok` and target-readiness
command groups. The
top-level `target_path_access` section checks whether the configured cache,
dataset, output, calibration and prompt paths are creatable or writable by the
current user; `completion_state.target_path_access_ok` mirrors that result so
path blockers are visible before any target-side asset report exists. Treat
`completion_state.complete=true` as the only goal-level completion signal in
this audit; it is not allowed unless `completion_audit.json` reports
`"ok": true` with `validator.ok=true`, `edge_results_bundle_verify.json`
reports `"ok": true` and `"complete": true`, and `edge_delivery_verify.json`
reports `"ok": true` with all root, bundle-directory and archive substatus
checks passing.
`tools/verify_edge_portable_bundle.py` also treats the portable-bundle manifest
as an integrity gate: every required copied file must exist and be listed in
`PORTABLE_BUNDLE_MANIFEST.json` with a verifiable size and SHA256 entry, except
for the manifest file itself. The required-file set covers the target-side
environment checker, external-method configurator, quantized-runtime probe,
goal audit, result collector/renderer, final validator and delivery verifiers,
so a bundle missing a migration or completion-gate tool is rejected before use.
For final handoff, use `--require-clean-code` and `--require-commit` so stale
or dirty-worktree portable bundles are rejected before they are moved.
`tools/build_goal_audit.py` reads the generated
`portable_bundle_verify.json` and reports its `ok`, archive, file-count and
code-provenance fields in the portable handoff evidence, including tarball
verified-file counts, top-level directory names, duplicate path counts, unsafe
archive member counts and archive size/SHA256 mismatch counts. This records code
package readiness without treating it as a substitute for final 4090 matrix
results.

For a target-machine readiness summary that does not run the matrix, use:

```bash
python tools/audit_target_readiness.py \
  --config configs/edge_4090_matrix.json \
  --root /data/outputs/llmcbench_edge_4090 \
  --check-method-readiness \
  --require-non-rotational-storage \
  --require-prewarm \
  --require-handoff-manifest \
  --require-optional-backends-report \
  --require-quant-runtime-report \
  --output /data/outputs/llmcbench_edge_4090/target_readiness.json
```

This report lists expected artifacts, whether existing prepare/preflight/prewarm
reports are present and OK, whether the handoff manifest exists when required,
optional backend and quant runtime reports when those strict gates are required,
and concrete next commands to produce the final 4090 matrix artifacts. Those
next commands include the same grouped `RUN_PRECOMPRESSED_PREPARE=1 ... bash
scripts/run_edge_4090_full.sh` wrapper used by the portable handoff manifest.
That wrapper stages the configured public precompressed groups, writes
`edge_prepare_precompressed_report.json` and
`edge_matrix_with_precompressed.json`, then exits before preflight or matrix
execution. Follow-up preflight, prewarm, readiness, matrix and final-audit
commands use that generated config. This prevents preparing
`precompressed_paths` without using them and avoids omitting the staged LLaMA3
GPTQ/AWQ/SmoothQuant checkpoints. The same commands are also grouped in
`next_command_groups` so automation can choose the baseline path or the
generated precompressed-config path, including the readiness refresh for each
path, without parsing the flat command list. The generated precompressed-config
readiness, matrix and final-audit commands require
`edge_prepare_precompressed_report.json`, so the exported-checkpoint setup
cannot be skipped accidentally.
After a matrix run finishes, use the matching post-run audit group from the
handoff manifest or `target_readiness.json`: `post_run_audit` /
`baseline_post_run_audit` for `configs/edge_4090_matrix.json`, and
`precompressed_config_post_run_audit` for
`edge_matrix_with_precompressed.json`. This keeps `target_readiness.json` tied
to the same config that produced the final matrix and avoids overwriting a
generated precompressed-config run with a baseline-config readiness summary.
It separates `ready_for_matrix` from final `completion_ok`: `ready_for_matrix`
means the target machine has satisfied the required prepare/preflight/method
readiness gates for launching the long matrix run, while `completion_ok` still
requires a finished `completion_audit.json` with `"ok": true`.
The one-command wrapper writes a pre-matrix readiness summary to
`target_readiness.json` under the final output root, or under
`DRY_RUN_OUTPUT_ROOT` during dry runs, then refreshes it after the matrix/final
audit path has produced final artifacts.

Full benchmark numbers are not checked into the repository yet because the
complete LLaMA3/Qwen3 x method x task matrix has not been run on the target 4090
Python 3.10 environment. Once the models and datasets are available on the 4090
machine, run:

```bash
python tools/build_edge_handoff_manifest.py --config configs/edge_4090_matrix.json --output-root /data/outputs/llmcbench_edge_4090 --output /data/outputs/llmcbench_edge_4090/edge_handoff_manifest.json
python tools/run_edge_preflight.py --config configs/edge_4090_matrix.json --check-hf-access --check-model-forward --finalize-dry-run --cuda-visible-devices 0 --check-optional-backend-resolution --check-quant-runtime --output /data/outputs/llmcbench_edge_4090/edge_preflight_report.json
python tools/prewarm_lm_eval_tasks.py --config configs/edge_4090_matrix.json --only-model Qwen3-8B --expand-mmlu --sleep-seconds 60 --report-output /data/outputs/llmcbench_edge_4090/edge_prewarm_report.json
python tools/run_edge_matrix.py --config configs/edge_4090_matrix.json --stop-on-error --finalize --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report --require-optional-backends-report --require-quant-runtime-report --require-target-paths-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree
python tools/audit_edge_completion.py --config configs/edge_4090_matrix.json --root /data/outputs/llmcbench_edge_4090 --require-finalize-report --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report --require-optional-backends-report --require-quant-runtime-report --require-target-paths-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree
```

The generated `matrix_results.csv` and `matrix_results.json` are the expected
final table artifacts for the two-model complete benchmark result. When prewarm
is enabled, `edge_prewarm_report.json` is also part of the output bundle. The
wrapper also writes `edge_asset_audit.json` and `edge_preflight_report.json`
beside the final results, writes optional backend and quantized runtime reports
when those strict probes are enabled, and writes prepare reports when prepare
steps are enabled. The generated `completion_audit.json` must have `"ok": true`
before treating the run as a complete delivery.

For a requirement-to-artifact checklist, see
`docs/COMPLETION_AUDIT.md`. Run `python tools/run_smoke_checks.py` before long
experiments to verify CLI wiring, scripts and a tiny local-model pipeline
without loading 8B checkpoints.
For real development-environment progress such as partially completed model
downloads or Qwen smoke runs, see `docs/RUN_STATUS.md`.

## Acknowledgements

In addition to the code in this repo, we also use [EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models. (github.com)](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation.

## Citation

If you find our project useful or relevant to your research, please kindly cite our paper:

```
@inproceedings{yang2024llmcbench,
  title={LLMCBench: Benchmarking Large Language Model Compression for Efficient Deployment},
  author={Yang, Ge and He, Changyi and Guo, Jinyang and Wu, Jianyu and Ding, Yifu and Liu, Aishan and Qin, Haotong and Ji, Pengliang and Liu, Xianglong},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024}
}
```
