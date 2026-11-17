#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data/models/Meta-Llama-3-8B}"
MODEL_NAME="${MODEL_NAME:-LLaMA3-8B}"
COMPRESSION_METHOD="${COMPRESSION_METHOD:-gptq}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/hf_cache}"
DATA_DIR="${DATA_DIR:-/data/llm_eval_datasets}"
CALIBRATION_FILE="${CALIBRATION_FILE:-}"
PROMPT_FILE="${PROMPT_FILE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pipeline/${MODEL_NAME}_${COMPRESSION_METHOD}}"
MAX_MEMORY="${MAX_MEMORY:-0:22GiB,cpu:120GiB}"
MIN_FREE_GPU_MEMORY_GB="${MIN_FREE_GPU_MEMORY_GB:-20}"
DEPLOYMENT_MEMORY_LIMIT_GB="${DEPLOYMENT_MEMORY_LIMIT_GB:-24}"
DEPLOYMENT_BATCH_SIZE="${DEPLOYMENT_BATCH_SIZE:-1}"

ARGS=(
  --model-path "${BASE_MODEL_PATH}"
  --model-name "${MODEL_NAME}"
  --base-model-repository "${BASE_MODEL_REPOSITORY:-}"
  --base-model-revision "${BASE_MODEL_REVISION:-}"
  --base-model-attestation-path "${BASE_MODEL_ATTESTATION_PATH:-}"
  --compression-method "${COMPRESSION_METHOD}"
  --cache-dir "${HF_CACHE_DIR}"
  --hf-endpoint "${HF_ENDPOINT}"
  --device cuda:0
  --device-map "${DEVICE_MAP:-auto}"
  --dtype "${DTYPE:-fp16}"
  --seqlen "${SEQLEN:-2048}"
  --calibration-samples "${CALIBRATION_SAMPLES:-128}"
  --calibration-batch-size "${CALIBRATION_BATCH_SIZE:-1}"
  --pruning-calibration-source "${PRUNING_CALIBRATION_SOURCE:-c4}"
  --calibration-seed "${CALIBRATION_SEED:-0}"
  --calibration-dataset "${CALIBRATION_DATASET:-allenai/c4}"
  --calibration-dataset-config "${CALIBRATION_DATASET_CONFIG:-allenai--c4}"
  --calibration-c4-data-file "${CALIBRATION_C4_DATA_FILE:-en/c4-train.00000-of-01024.json.gz}"
  --calibration-c4-revision "${CALIBRATION_C4_REVISION:-1588ec454efa1a09f29cd18ddd04fe05fc8653a2}"
  --sparsity "${SPARSITY:-0.5}"
  --prune-pattern "${PRUNE_PATTERN:-unstructured}"
  --sparsegpt-blocksize "${SPARSEGPT_BLOCKSIZE:-128}"
  --sparsegpt-percdamp "${SPARSEGPT_PERCDAMP:-0.01}"
  --pruning-buffer-device "${PRUNING_BUFFER_DEVICE:-cpu}"
  --quant-bits "${QUANT_BITS:-4}"
  --group-size "${GROUP_SIZE:-128}"
  --gptq-calibration-source "${GPTQ_CALIBRATION_SOURCE:-c4}"
  --gptq-damp-percent "${GPTQ_DAMP_PERCENT:-0.01}"
  --awq-calibration-source "${AWQ_CALIBRATION_SOURCE:-pileval}"
  --awq-calibration-seed "${AWQ_CALIBRATION_SEED:-42}"
  --awq-calibration-samples "${AWQ_CALIBRATION_SAMPLES:-128}"
  --awq-calibration-seqlen "${AWQ_CALIBRATION_SEQLEN:-512}"
  --awq-grid-size "${AWQ_GRID_SIZE:-20}"
  --smoothquant-calibration-source "${SMOOTHQUANT_CALIBRATION_SOURCE:-pile_validation}"
  --smoothquant-calibration-sentences "${SMOOTHQUANT_CALIBRATION_SENTENCES:-512}"
  --smoothquant-alpha-selection "${SMOOTHQUANT_ALPHA_SELECTION:-official_grid_search}"
  --smoothquant-official-code-dir "${SMOOTHQUANT_OFFICIAL_CODE_DIR:-}"
  --smoothquant-official-repository "${SMOOTHQUANT_OFFICIAL_REPOSITORY:-}"
  --smoothquant-official-commit "${SMOOTHQUANT_OFFICIAL_COMMIT:-}"
  --omniquant-abits "${OMNIQUANT_ABITS:-16}"
  --omniquant-epochs "${OMNIQUANT_EPOCHS:-20}"
  --omniquant-nsamples "${OMNIQUANT_NSAMPLES:-128}"
  --omniquant-seqlen "${OMNIQUANT_SEQLEN:-2048}"
  --omniquant-seed "${OMNIQUANT_SEED:-2}"
  --omniquant-calibration-source "${OMNIQUANT_CALIBRATION_SOURCE:-wikitext2}"
  --omniquant-official-code-dir "${OMNIQUANT_OFFICIAL_CODE_DIR:-}"
  --omniquant-official-repository "${OMNIQUANT_OFFICIAL_REPOSITORY:-}"
  --omniquant-official-commit "${OMNIQUANT_OFFICIAL_COMMIT:-}"
  --min-free-gpu-memory-gb "${MIN_FREE_GPU_MEMORY_GB}"
  --data-dir "${DATA_DIR}"
  --tasks "${TASKS:-mmlu,arc_challenge,hellaswag,piqa,winogrande}"
  --batch-size "${BATCH_SIZE:-auto}"
  --input-length "${INPUT_LENGTH:-2048}"
  --max-new-tokens "${MAX_NEW_TOKENS:-128}"
  --deployment-batch-size "${DEPLOYMENT_BATCH_SIZE}"
  --sparsity-sample-tensors "${SPARSITY_SAMPLE_TENSORS:--1}"
  --warmup "${WARMUP:-1}"
  --repeat "${REPEAT:-3}"
  --deployment-memory-limit-gb "${DEPLOYMENT_MEMORY_LIMIT_GB}"
  --output-root "${OUTPUT_ROOT}"
)

if [[ -n "${CALIBRATION_FILE}" ]]; then
  ARGS+=(--calibration-file "${CALIBRATION_FILE}")
fi

if [[ -n "${CALIBRATION_DATASET_CACHE:-}" ]]; then
  ARGS+=(--calibration-dataset-cache "${CALIBRATION_DATASET_CACHE}")
fi

if [[ -n "${REVISION:-}" ]]; then
  ARGS+=(--revision "${REVISION}")
fi

if [[ -n "${PROMPT_FILE}" ]]; then
  ARGS+=(--prompt-file "${PROMPT_FILE}")
fi

if [[ -n "${EXTERNAL_COMMAND:-}" ]]; then
  ARGS+=(--external-command "${EXTERNAL_COMMAND}")
fi

if [[ "${PRUNE_LM_HEAD:-0}" == "1" ]]; then
  ARGS+=(--prune-lm-head)
fi

if [[ "${DESC_ACT:-0}" == "1" ]]; then
  ARGS+=(--desc-act)
fi

if [[ "${GPTQ_TRUE_SEQUENTIAL:-0}" == "1" ]]; then
  ARGS+=(--gptq-true-sequential)
fi

if [[ "${GPTQ_STATIC_GROUPS:-0}" == "1" ]]; then
  ARGS+=(--gptq-static-groups)
fi

if [[ "${GPTQ_ASYMMETRIC:-0}" == "1" ]]; then
  ARGS+=(--gptq-asymmetric)
fi

if [[ -n "${SMOOTHQUANT_ALPHA:-}" ]]; then
  ARGS+=(--smoothquant-alpha "${SMOOTHQUANT_ALPHA}")
fi

if [[ "${OMNIQUANT_LWC:-0}" == "1" ]]; then
  ARGS+=(--omniquant-lwc)
fi

if [[ "${OMNIQUANT_LET:-0}" == "1" ]]; then
  ARGS+=(--omniquant-let)
fi

if [[ "${OMNIQUANT_AUG_LOSS:-0}" == "1" ]]; then
  ARGS+=(--omniquant-aug-loss)
fi

if [[ -n "${AWQ_VERSION:-}" ]]; then
  ARGS+=(--awq-version "${AWQ_VERSION}")
fi

if [[ -n "${PAPER_PROTOCOL:-}" ]]; then
  ARGS+=(--paper-protocol "${PAPER_PROTOCOL}")
fi

if [[ "${TARGET_MODEL_ADAPTATION:-0}" == "1" ]]; then
  ARGS+=(--target-model-adaptation)
fi

if [[ "${REQUIRE_PAPER_PROTOCOL_ATTESTATION:-0}" == "1" ]]; then
  ARGS+=(--require-paper-protocol-attestation)
fi

if [[ -n "${ATTN_IMPLEMENTATION:-}" ]]; then
  ARGS+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi

ARGS+=(--max-memory "${MAX_MEMORY}")

if [[ "${OVERWRITE_COMPRESSED:-0}" == "1" ]]; then
  ARGS+=(--overwrite-compressed)
fi

if [[ "${USE_PRECOMPRESSED:-0}" == "1" ]]; then
  ARGS+=(--use-precompressed)
fi

if [[ "${SKIP_ACCURACY:-0}" == "1" ]]; then
  ARGS+=(--skip-accuracy)
fi

if [[ "${SPLIT_ACCURACY_TASKS:-0}" == "1" ]]; then
  ARGS+=(--split-accuracy-tasks)
fi

if [[ "${CONTINUE_ON_ACCURACY_ERROR:-0}" == "1" ]]; then
  ARGS+=(--continue-on-accuracy-error)
fi

if [[ -n "${NUM_FEWSHOT:-}" ]]; then
  ARGS+=(--num-fewshot "${NUM_FEWSHOT}")
fi

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

if [[ "${SKIP_DEPLOY:-0}" == "1" ]]; then
  ARGS+=(--skip-deploy)
fi

if [[ "${SKIP_RESOURCE:-0}" == "1" ]]; then
  ARGS+=(--skip-resource)
fi

"${PYTHON_BIN}" -m edge_benchmark pipeline "${ARGS[@]}"
