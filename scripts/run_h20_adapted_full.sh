#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/edge_h20_adapted_full_matrix.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_PHYSICAL_INDEX="${GPU_PHYSICAL_INDEX:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/outputs/llmcbench_h20_adapted_full}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"
MAX_GPU_UTILIZATION_PERCENT="${MAX_GPU_UTILIZATION_PERCENT:-5}"
MAX_USED_MEMORY_MIB="${MAX_USED_MEMORY_MIB:-8000}"
LOG_PATH="${LOG_PATH:-${OUTPUT_ROOT}/matrix_launcher.log}"

export CUDA_VISIBLE_DEVICES
export HF_ENDPOINT

mkdir -p "${OUTPUT_ROOT}"
started_at="$(date +%s)"

while true; do
  line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "${GPU_PHYSICAL_INDEX}")"
  utilization="$(awk -F',' '{gsub(/ /, "", $1); print $1}' <<<"${line}")"
  memory_mib="$(awk -F',' '{gsub(/ /, "", $2); print $2}' <<<"${line}")"
  if [[ "${utilization}" =~ ^[0-9]+$ && "${memory_mib}" =~ ^[0-9]+$ ]] \
    && (( utilization <= MAX_GPU_UTILIZATION_PERCENT && memory_mib <= MAX_USED_MEMORY_MIB )); then
    break
  fi
  now="$(date +%s)"
  if (( WAIT_TIMEOUT_SECONDS > 0 && now - started_at >= WAIT_TIMEOUT_SECONDS )); then
    echo "Timed out waiting for GPU ${GPU_PHYSICAL_INDEX}: utilization=${utilization}%, memory=${memory_mib}MiB" >&2
    exit 1
  fi
  echo "Waiting for GPU ${GPU_PHYSICAL_INDEX}: utilization=${utilization}%, memory=${memory_mib}MiB" | tee -a "${LOG_PATH}"
  sleep "${POLL_SECONDS}"
done

echo "Launching adapted full matrix on GPU ${GPU_PHYSICAL_INDEX}" | tee -a "${LOG_PATH}"
"${PYTHON_BIN}" tools/run_edge_matrix.py \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}" \
  --stop-on-error \
  --finalize 2>&1 | tee -a "${LOG_PATH}"
