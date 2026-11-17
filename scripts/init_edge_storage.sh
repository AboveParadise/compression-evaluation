#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
TARGET_USER="${TARGET_USER:-${SUDO_USER:-$(id -un)}}"
TARGET_GROUP="${TARGET_GROUP:-}"
CHOWN_DATA_ROOT="${CHOWN_DATA_ROOT:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${TARGET_GROUP}" ]]; then
  if id -gn "${TARGET_USER}" >/dev/null 2>&1; then
    TARGET_GROUP="$(id -gn "${TARGET_USER}")"
  else
    TARGET_GROUP="${TARGET_USER}"
  fi
fi

DIRECTORIES=(
  "${DATA_ROOT}/models"
  "${DATA_ROOT}/models/precompressed"
  "${DATA_ROOT}/hf_cache"
  "${DATA_ROOT}/llm_eval_datasets"
  "${DATA_ROOT}/calibration"
  "${DATA_ROOT}/prompts"
  "${DATA_ROOT}/outputs"
)

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

echo "Preparing LLMCBench edge storage under ${DATA_ROOT} for ${TARGET_USER}:${TARGET_GROUP}"
for directory in "${DIRECTORIES[@]}"; do
  run_cmd mkdir -p "${directory}"
done

if [[ "${CHOWN_DATA_ROOT}" == "1" ]]; then
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "CHOWN_DATA_ROOT=1 requires root. Re-run with sudo or set CHOWN_DATA_ROOT=0 after ownership is already correct." >&2
    exit 1
  fi
  run_cmd chown -R "${TARGET_USER}:${TARGET_GROUP}" "${DATA_ROOT}"
fi

echo "Storage initialization finished. Next run:"
echo "  python tools/prepare_target_paths.py --config configs/edge_4090_matrix.json --output ${DATA_ROOT}/outputs/llmcbench_edge_4090/edge_target_paths_report.json --fail-on-error"
