#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-/path/to/compressed/model}"
MODEL_NAME="${MODEL_NAME:-LLaMA3-8B}"
COMPRESSION_METHOD="${COMPRESSION_METHOD:-gptq}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/hf_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/edge_resource}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/${MODEL_NAME}_${COMPRESSION_METHOD}.json}"
MAX_MEMORY="${MAX_MEMORY:-0:22GiB,cpu:120GiB}"
MIN_FREE_GPU_MEMORY_GB="${MIN_FREE_GPU_MEMORY_GB:-20}"

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
  --seqlen "${SEQLEN:-2048}"
  --input-length "${INPUT_LENGTH:-2048}"
  --max-new-tokens "${MAX_NEW_TOKENS:-128}"
  --sparsity-sample-tensors "${SPARSITY_SAMPLE_TENSORS:--1}"
  --quant-bits "${QUANT_BITS:-4}"
  --group-size "${GROUP_SIZE:-128}"
  --output "${OUTPUT_FILE}"
)

if [[ -n "${ATTN_IMPLEMENTATION:-}" ]]; then
  ARGS+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi

ARGS+=(--max-memory "${MAX_MEMORY}")

if [[ -n "${PROMPT_FILE:-}" ]]; then
  ARGS+=(--prompt-file "${PROMPT_FILE}")
fi

"${PYTHON_BIN}" -m edge_benchmark resource "${ARGS[@]}"
