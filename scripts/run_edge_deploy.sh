#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-/path/to/compressed/model}"
MODEL_NAME="${MODEL_NAME:-LLaMA3-8B}"
COMPRESSION_METHOD="${COMPRESSION_METHOD:-gptq}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/hf_cache}"
PROMPT_FILE="${PROMPT_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/edge_deploy}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/${MODEL_NAME}_${COMPRESSION_METHOD}.json}"
MAX_MEMORY="${MAX_MEMORY:-0:22GiB,cpu:120GiB}"
MIN_FREE_GPU_MEMORY_GB="${MIN_FREE_GPU_MEMORY_GB:-20}"
DEPLOYMENT_MEMORY_LIMIT_GB="${DEPLOYMENT_MEMORY_LIMIT_GB:-24}"
DEPLOYMENT_BATCH_SIZE="${DEPLOYMENT_BATCH_SIZE:-1}"

ARGS=(
  --model-path "${MODEL_PATH}"
  --model-name "${MODEL_NAME}"
  --compression-method "${COMPRESSION_METHOD}"
  --cache-dir "${HF_CACHE_DIR}"
  --hf-endpoint "${HF_ENDPOINT}"
  --min-free-gpu-memory-gb "${MIN_FREE_GPU_MEMORY_GB}"
  --dtype "${DTYPE:-fp16}"
  --device "${DEVICE:-cuda:0}"
  --device-map "${DEVICE_MAP:-auto}"
  --input-length "${INPUT_LENGTH:-2048}"
  --max-new-tokens "${MAX_NEW_TOKENS:-128}"
  --deployment-batch-size "${DEPLOYMENT_BATCH_SIZE}"
  --warmup "${WARMUP:-1}"
  --repeat "${REPEAT:-3}"
  --deployment-memory-limit-gb "${DEPLOYMENT_MEMORY_LIMIT_GB}"
  --output "${OUTPUT_FILE}"
)

if [[ -n "${PROMPT:-}" ]]; then
  ARGS+=(--prompt "${PROMPT}")
fi

if [[ -n "${PROMPT_FILE}" ]]; then
  ARGS+=(--prompt-file "${PROMPT_FILE}")
fi

if [[ -n "${ATTN_IMPLEMENTATION:-}" ]]; then
  ARGS+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi

ARGS+=(--max-memory "${MAX_MEMORY}")

"${PYTHON_BIN}" -m edge_benchmark deploy "${ARGS[@]}"
