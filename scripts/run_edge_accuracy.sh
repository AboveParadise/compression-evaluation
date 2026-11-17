#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-/path/to/compressed/model}"
MODEL_NAME="${MODEL_NAME:-LLaMA3-8B}"
COMPRESSION_METHOD="${COMPRESSION_METHOD:-gptq}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/hf_cache}"
DATA_DIR="${DATA_DIR:-/data/llm_eval_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/edge_accuracy}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/${MODEL_NAME}_${COMPRESSION_METHOD}.json}"
LM_EVAL_OUTPUT="${LM_EVAL_OUTPUT:-${OUTPUT_DIR}/lm_eval_raw}"
MAX_MEMORY="${MAX_MEMORY:-0:22GiB,cpu:120GiB}"
MIN_FREE_GPU_MEMORY_GB="${MIN_FREE_GPU_MEMORY_GB:-20}"

ARGS=(
  --model-path "${MODEL_PATH}"
  --model-name "${MODEL_NAME}"
  --compression-method "${COMPRESSION_METHOD}"
  --cache-dir "${HF_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --hf-endpoint "${HF_ENDPOINT}"
  --min-free-gpu-memory-gb "${MIN_FREE_GPU_MEMORY_GB}"
  --dtype "${DTYPE:-fp16}"
  --tasks "${TASKS:-mmlu,arc_challenge,hellaswag,piqa,winogrande}"
  --batch-size "${BATCH_SIZE:-auto}"
  --device "${DEVICE:-cuda:0}"
  --lm-eval-output "${LM_EVAL_OUTPUT}"
  --output "${OUTPUT_FILE}"
)

if [[ -n "${NUM_FEWSHOT:-}" ]]; then
  ARGS+=(--num-fewshot "${NUM_FEWSHOT}")
fi

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

if [[ "${STREAM_OUTPUT:-0}" == "1" ]]; then
  ARGS+=(--stream-output)
fi

if [[ -n "${ATTN_IMPLEMENTATION:-}" ]]; then
  ARGS+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi

ARGS+=(--max-memory "${MAX_MEMORY}")

"${PYTHON_BIN}" -m edge_benchmark accuracy "${ARGS[@]}"
