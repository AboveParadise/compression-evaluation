#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/edge_4090_matrix.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TARGET_PATH_PREPARE="${RUN_TARGET_PATH_PREPARE:-1}"
RUN_PREPARE="${RUN_PREPARE:-0}"
RUN_METADATA_PREPARE="${RUN_METADATA_PREPARE:-0}"
RUN_PRECOMPRESSED_PREPARE="${RUN_PRECOMPRESSED_PREPARE:-0}"
PRECOMPRESSED_MODELS_SET="${PRECOMPRESSED_MODELS+x}"
PRECOMPRESSED_METHODS_SET="${PRECOMPRESSED_METHODS+x}"
PRECOMPRESSED_GROUPS_SET="${PRECOMPRESSED_GROUPS+x}"
PRECOMPRESSED_MODELS="${PRECOMPRESSED_MODELS:-Qwen3-8B}"
PRECOMPRESSED_METHODS="${PRECOMPRESSED_METHODS:-gptq,awq,smoothquant,omniquant}"
if [[ -z "${PRECOMPRESSED_GROUPS_SET}" && -z "${PRECOMPRESSED_MODELS_SET}" && -z "${PRECOMPRESSED_METHODS_SET}" ]]; then
  PRECOMPRESSED_GROUPS="LLaMA3-8B:gptq,awq,smoothquant;Qwen3-8B:gptq,awq,smoothquant,omniquant"
else
  PRECOMPRESSED_GROUPS="${PRECOMPRESSED_GROUPS:-}"
fi
PRECOMPRESSED_PREPARE_ONLY="${PRECOMPRESSED_PREPARE_ONLY:-0}"
PRECOMPRESSED_LOCAL_ROOT="${PRECOMPRESSED_LOCAL_ROOT:-/data/models/precompressed}"
PRECOMPRESSED_METADATA_ONLY="${PRECOMPRESSED_METADATA_ONLY:-0}"
PRECOMPRESSED_REPOS="${PRECOMPRESSED_REPOS:-}"
PRECOMPRESSED_FORCE="${PRECOMPRESSED_FORCE:-0}"
RUN_PREWARM="${RUN_PREWARM:-1}"
CHECK_HF_ACCESS="${CHECK_HF_ACCESS:-1}"
CHECK_MODEL_FORWARD="${CHECK_MODEL_FORWARD:-1}"
CHECK_OPTIONAL_BACKEND_RESOLUTION="${CHECK_OPTIONAL_BACKEND_RESOLUTION:-0}"
FAIL_ON_OPTIONAL_BACKEND_RESOLUTION="${FAIL_ON_OPTIONAL_BACKEND_RESOLUTION:-0}"
CHECK_QUANT_RUNTIME="${CHECK_QUANT_RUNTIME:-0}"
FAIL_ON_QUANT_RUNTIME="${FAIL_ON_QUANT_RUNTIME:-0}"
QUANT_RUNTIME_GENERATE="${QUANT_RUNTIME_GENERATE:-0}"
FINALIZE_DRY_RUN="${FINALIZE_DRY_RUN:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-80}"
FAIL_ON_BLOCKED_METHODS="${FAIL_ON_BLOCKED_METHODS:-1}"
LINK_EXISTING_MODELS="${LINK_EXISTING_MODELS:-0}"
MODEL_SOURCE_ROOT="${MODEL_SOURCE_ROOT:-}"
PREWARM_MODEL="${PREWARM_MODEL:-Qwen3-8B}"
PREWARM_SLEEP_SECONDS="${PREWARM_SLEEP_SECONDS:-60}"
REQUIRE_PREWARM_ALL_MODELS="${REQUIRE_PREWARM_ALL_MODELS:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN_OUTPUT_ROOT="${DRY_RUN_OUTPUT_ROOT:-/tmp/llmcbench_edge_4090_dryrun}"
REQUIRE_RUNTIME_PYTHON_PREFIX="${REQUIRE_RUNTIME_PYTHON_PREFIX:-3.10}"
REQUIRE_RUNTIME_GPU_NAME_CONTAINS="${REQUIRE_RUNTIME_GPU_NAME_CONTAINS:-4090}"
REQUIRE_RUNTIME_CUDA_PREFIX="${REQUIRE_RUNTIME_CUDA_PREFIX:-12.}"
REQUIRE_RUNTIME_GPU_TOTAL_MEMORY_MIN_GB="${REQUIRE_RUNTIME_GPU_TOTAL_MEMORY_MIN_GB:-23}"
REQUIRE_RUNTIME_GPU_COUNT="${REQUIRE_RUNTIME_GPU_COUNT:-1}"
REQUIRE_RUNTIME_CUDNN="${REQUIRE_RUNTIME_CUDNN:-1}"
REQUIRE_RUNTIME_TORCH_PREFIX="${REQUIRE_RUNTIME_TORCH_PREFIX:-2.}"
REQUIRE_RUNTIME_TRANSFORMERS="${REQUIRE_RUNTIME_TRANSFORMERS:-1}"
REQUIRE_RUNTIME_BITSANDBYTES="${REQUIRE_RUNTIME_BITSANDBYTES:-1}"
REQUIRE_RUNTIME_CPU_COUNT_MIN="${REQUIRE_RUNTIME_CPU_COUNT_MIN:-8}"
REQUIRE_RUNTIME_SYSTEM_MEMORY_MIN_GB="${REQUIRE_RUNTIME_SYSTEM_MEMORY_MIN_GB:-240}"
REQUIRE_RUNTIME_HF_ENDPOINT="${REQUIRE_RUNTIME_HF_ENDPOINT:-https://hf-mirror.com}"
REQUIRE_NON_ROTATIONAL_STORAGE="${REQUIRE_NON_ROTATIONAL_STORAGE:-1}"
RUN_HANDOFF_MANIFEST="${RUN_HANDOFF_MANIFEST:-1}"
REQUIRE_TARGET_READINESS_REPORT="${REQUIRE_TARGET_READINESS_REPORT:-1}"
FAIL_ON_TARGET_READINESS_BLOCKERS="${FAIL_ON_TARGET_READINESS_BLOCKERS:-1}"
EXPORT_RESULTS_BUNDLE="${EXPORT_RESULTS_BUNDLE:-1}"
VERIFY_RESULTS_BUNDLE="${VERIFY_RESULTS_BUNDLE:-1}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CONFIG
export CUDA_VISIBLE_DEVICES

CONFIG_OUTPUT_ROOT="$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
config = json.loads(Path(__import__("os").environ["CONFIG"]).read_text())
print(config["defaults"]["output_root"])
PY
)"
DEFAULT_PREWARM_REPORT_ROOT="${CONFIG_OUTPUT_ROOT}"
if [[ "${DRY_RUN}" == "1" ]]; then
  DEFAULT_PREWARM_REPORT_ROOT="${DRY_RUN_OUTPUT_ROOT}"
fi
PREWARM_REPORT_OUTPUT="${PREWARM_REPORT_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_prewarm_report.json}"
ASSET_AUDIT_OUTPUT="${ASSET_AUDIT_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_asset_audit.json}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_preflight_report.json}"
PREPARE_METADATA_OUTPUT="${PREPARE_METADATA_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_prepare_metadata_report.json}"
TARGET_PATH_PREPARE_OUTPUT="${TARGET_PATH_PREPARE_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_target_paths_report.json}"
PREPARE_OUTPUT="${PREPARE_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_prepare_assets_report.json}"
PRECOMPRESSED_PREPARE_OUTPUT="${PRECOMPRESSED_PREPARE_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_prepare_precompressed_report.json}"
PRECOMPRESSED_CONFIG_OUTPUT="${PRECOMPRESSED_CONFIG_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_matrix_with_precompressed.json}"
TARGET_READINESS_OUTPUT="${TARGET_READINESS_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/target_readiness.json}"
HANDOFF_MANIFEST_OUTPUT="${HANDOFF_MANIFEST_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_handoff_manifest.json}"
RESULTS_BUNDLE_DIR="${RESULTS_BUNDLE_DIR:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_results_bundle}"
RESULTS_BUNDLE_ARCHIVE="${RESULTS_BUNDLE_ARCHIVE:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_results_bundle.tar.gz}"
RESULTS_BUNDLE_VERIFY_OUTPUT="${RESULTS_BUNDLE_VERIFY_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_results_bundle_verify.json}"
DELIVERY_VERIFY_OUTPUT="${DELIVERY_VERIFY_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_delivery_verify.json}"
SCORECARD_JSON_OUTPUT="${SCORECARD_JSON_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_scorecard.json}"
SCORECARD_MD_OUTPUT="${SCORECARD_MD_OUTPUT:-${DEFAULT_PREWARM_REPORT_ROOT}/edge_scorecard.md}"
SCORECARD_EXTRA_ARGS="${SCORECARD_EXTRA_ARGS:-}"

run_cmd() {
  echo "+ $*"
  "$@"
}

build_target_readiness_args() {
  TARGET_READINESS_ARGS=(
    --config "${CONFIG}"
    --root "${DEFAULT_PREWARM_REPORT_ROOT}"
    --output "${TARGET_READINESS_OUTPUT}"
    --check-method-readiness
  )
  if [[ "${REQUIRE_NON_ROTATIONAL_STORAGE}" == "1" ]]; then
    TARGET_READINESS_ARGS+=(--require-non-rotational-storage)
  fi
  if [[ "${RUN_PREWARM}" == "1" ]]; then
    TARGET_READINESS_ARGS+=(--require-prewarm)
  fi
  if [[ "${RUN_HANDOFF_MANIFEST}" == "1" ]]; then
    TARGET_READINESS_ARGS+=(--require-handoff-manifest)
  fi
  if [[ "${CHECK_OPTIONAL_BACKEND_RESOLUTION}" == "1" ]]; then
    TARGET_READINESS_ARGS+=(--require-optional-backends-report)
  fi
  if [[ "${CHECK_QUANT_RUNTIME}" == "1" ]]; then
    TARGET_READINESS_ARGS+=(--require-quant-runtime-report)
  fi
  if [[ "${RUN_PRECOMPRESSED_PREPARE}" == "1" ]]; then
    TARGET_READINESS_ARGS+=(--require-prepare-precompressed-report)
  fi
}

check_target_ready_for_matrix() {
  "${PYTHON_BIN}" - "${TARGET_READINESS_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("ready_for_matrix") is not True:
    blockers = payload.get("before_matrix_blockers")
    print(
        json.dumps(
            {
                "ok": False,
                "path": str(path),
                "ready_for_matrix": payload.get("ready_for_matrix"),
                "missing_before_matrix": payload.get("missing_before_matrix"),
                "before_matrix_blockers": blockers,
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)
print(json.dumps({"ok": True, "path": str(path), "ready_for_matrix": True}, indent=2))
PY
}

run_target_readiness() {
  local stage="$1"
  build_target_readiness_args
  run_cmd "${PYTHON_BIN}" tools/audit_target_readiness.py "${TARGET_READINESS_ARGS[@]}"
  if [[ "${stage}" == "pre_matrix" && "${DRY_RUN}" != "1" && "${FAIL_ON_TARGET_READINESS_BLOCKERS}" == "1" ]]; then
    check_target_ready_for_matrix
  fi
}

PREPARE_MODEL_ARGS=()
if [[ "${LINK_EXISTING_MODELS}" == "1" ]]; then
  PREPARE_MODEL_ARGS+=(--link-existing-models)
fi
if [[ -n "${MODEL_SOURCE_ROOT}" ]]; then
  PREPARE_MODEL_ARGS+=(--model-source-root "${MODEL_SOURCE_ROOT}")
fi

if [[ "${RUN_TARGET_PATH_PREPARE}" == "1" ]]; then
  TARGET_PATH_ARGS=(
    --config "${CONFIG}"
    --output "${TARGET_PATH_PREPARE_OUTPUT}"
    --fail-on-error
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    TARGET_PATH_ARGS+=(--dry-run)
  fi
  run_cmd "${PYTHON_BIN}" tools/prepare_target_paths.py "${TARGET_PATH_ARGS[@]}"
fi

PREPARE_COMMON_ARGS=(
  --config "${CONFIG}"
  --skip-lm-eval-probe
  --continue-on-error
)
if [[ "${DRY_RUN}" == "1" ]]; then
  PREPARE_COMMON_ARGS+=(--dry-run)
fi

if [[ "${RUN_METADATA_PREPARE}" == "1" ]]; then
  run_cmd "${PYTHON_BIN}" tools/prepare_edge_assets.py \
    "${PREPARE_COMMON_ARGS[@]}" \
    --metadata-only \
    --output "${PREPARE_METADATA_OUTPUT}" \
    "${PREPARE_MODEL_ARGS[@]}"
fi

if [[ "${RUN_PREPARE}" == "1" ]]; then
  run_cmd "${PYTHON_BIN}" tools/prepare_edge_assets.py \
    "${PREPARE_COMMON_ARGS[@]}" \
    --min-free-disk-gb "${MIN_FREE_DISK_GB}" \
    --output "${PREPARE_OUTPUT}" \
    "${PREPARE_MODEL_ARGS[@]}"
fi

if [[ "${RUN_PRECOMPRESSED_PREPARE}" == "1" ]]; then
  PRECOMPRESSED_BASE_ARGS=(
    --local-root "${PRECOMPRESSED_LOCAL_ROOT}"
    --fail-on-error
  )
  if [[ -n "${PRECOMPRESSED_REPOS}" ]]; then
    IFS=',' read -ra PRECOMPRESSED_REPO_LIST <<< "${PRECOMPRESSED_REPOS}"
    for repo_override in "${PRECOMPRESSED_REPO_LIST[@]}"; do
      if [[ -n "${repo_override}" ]]; then
        PRECOMPRESSED_BASE_ARGS+=(--repo "${repo_override}")
      fi
    done
  fi
  if [[ "${PRECOMPRESSED_METADATA_ONLY}" == "1" ]]; then
    PRECOMPRESSED_BASE_ARGS+=(--metadata-only)
  fi
  if [[ "${PRECOMPRESSED_FORCE}" == "1" ]]; then
    PRECOMPRESSED_BASE_ARGS+=(--force)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    PRECOMPRESSED_BASE_ARGS+=(--dry-run --write-dry-run-config)
  fi

  if [[ -n "${PRECOMPRESSED_GROUPS}" ]]; then
    IFS=';' read -ra PRECOMPRESSED_GROUP_LIST <<< "${PRECOMPRESSED_GROUPS}"
    PRECOMPRESSED_INPUT_CONFIG="${CONFIG}"
    PRECOMPRESSED_GROUP_INDEX=0
    PRECOMPRESSED_REPORTS=()
    for group in "${PRECOMPRESSED_GROUP_LIST[@]}"; do
      if [[ -z "${group}" ]]; then
        continue
      fi
      group_model="${group%%:*}"
      group_methods="${group#*:}"
      if [[ -z "${group_model}" || "${group_model}" == "${group_methods}" ]]; then
        echo "Invalid PRECOMPRESSED_GROUPS item: ${group}. Expected MODEL:method,method." >&2
        exit 1
      fi
      PRECOMPRESSED_GROUP_INDEX=$((PRECOMPRESSED_GROUP_INDEX + 1))
      group_report="${PRECOMPRESSED_PREPARE_OUTPUT%.json}.${PRECOMPRESSED_GROUP_INDEX}.${group_model}.json"
      PRECOMPRESSED_ARGS=(
        --config "${PRECOMPRESSED_INPUT_CONFIG}"
        "${PRECOMPRESSED_BASE_ARGS[@]}"
        --output "${group_report}"
        --output-config "${PRECOMPRESSED_CONFIG_OUTPUT}"
        --only-model "${group_model}"
      )
      IFS=',' read -ra PRECOMPRESSED_GROUP_METHOD_LIST <<< "${group_methods}"
      for method_name in "${PRECOMPRESSED_GROUP_METHOD_LIST[@]}"; do
        if [[ -n "${method_name}" ]]; then
          PRECOMPRESSED_ARGS+=(--only-method "${method_name}")
        fi
      done
      run_cmd "${PYTHON_BIN}" tools/prepare_precompressed_assets.py "${PRECOMPRESSED_ARGS[@]}"
      PRECOMPRESSED_REPORTS+=("${group_report}")
      PRECOMPRESSED_INPUT_CONFIG="${PRECOMPRESSED_CONFIG_OUTPUT}"
    done
    "${PYTHON_BIN}" - "${PRECOMPRESSED_PREPARE_OUTPUT}" "${PRECOMPRESSED_REPORTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
reports = []
for value in sys.argv[2:]:
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports.append({"path": str(path), "payload": payload})
items = []
blockers = []
for report in reports:
    payload = report["payload"]
    for item in payload.get("items", []):
        if isinstance(item, dict):
            items.append({**item, "source_report": report["path"]})
    for item in payload.get("blockers", []):
        if isinstance(item, dict):
            blockers.append({**item, "source_report": report["path"]})
summary = {
    "ok": all(report["payload"].get("ok") is True for report in reports) and not blockers,
    "combined": True,
    "reports": [report["path"] for report in reports],
    "items": items,
    "blockers": blockers,
    "updated_config": reports[-1]["payload"].get("updated_config") if reports else None,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if summary["ok"] else 1)
PY
  else
    PRECOMPRESSED_ARGS=(
      --config "${CONFIG}"
      "${PRECOMPRESSED_BASE_ARGS[@]}"
      --output "${PRECOMPRESSED_PREPARE_OUTPUT}"
      --output-config "${PRECOMPRESSED_CONFIG_OUTPUT}"
    )
    IFS=',' read -ra PRECOMPRESSED_MODEL_LIST <<< "${PRECOMPRESSED_MODELS}"
    for model_name in "${PRECOMPRESSED_MODEL_LIST[@]}"; do
      if [[ -n "${model_name}" ]]; then
        PRECOMPRESSED_ARGS+=(--only-model "${model_name}")
      fi
    done
    IFS=',' read -ra PRECOMPRESSED_METHOD_LIST <<< "${PRECOMPRESSED_METHODS}"
    for method_name in "${PRECOMPRESSED_METHOD_LIST[@]}"; do
      if [[ -n "${method_name}" ]]; then
        PRECOMPRESSED_ARGS+=(--only-method "${method_name}")
      fi
    done
    run_cmd "${PYTHON_BIN}" tools/prepare_precompressed_assets.py "${PRECOMPRESSED_ARGS[@]}"
  fi
  if [[ -f "${PRECOMPRESSED_CONFIG_OUTPUT}" ]]; then
    CONFIG="${PRECOMPRESSED_CONFIG_OUTPUT}"
    export CONFIG
  fi
  if [[ "${PRECOMPRESSED_PREPARE_ONLY}" == "1" ]]; then
    exit 0
  fi
fi

if [[ "${RUN_HANDOFF_MANIFEST}" == "1" ]]; then
  run_cmd "${PYTHON_BIN}" tools/build_edge_handoff_manifest.py \
    --config "${CONFIG}" \
    --output-root "${DEFAULT_PREWARM_REPORT_ROOT}" \
    --output "${HANDOFF_MANIFEST_OUTPUT}"
fi

ASSET_AUDIT_ARGS=(
  --config "${CONFIG}"
  --min-free-disk-gb "${MIN_FREE_DISK_GB}"
  --output "${ASSET_AUDIT_OUTPUT}"
)
if [[ "${DRY_RUN}" != "1" ]]; then
  ASSET_AUDIT_ARGS+=(--fail-on-missing)
fi
run_cmd "${PYTHON_BIN}" tools/audit_edge_assets.py "${ASSET_AUDIT_ARGS[@]}"

PREFLIGHT_ARGS=(
  --config "${CONFIG}"
  --output "${PREFLIGHT_OUTPUT}"
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}"
)
if [[ "${CHECK_HF_ACCESS}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--check-hf-access)
fi
if [[ "${CHECK_MODEL_FORWARD}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--check-model-forward)
fi
if [[ "${CHECK_OPTIONAL_BACKEND_RESOLUTION}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--check-optional-backend-resolution)
  if [[ "${FAIL_ON_OPTIONAL_BACKEND_RESOLUTION}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--fail-on-optional-backend-resolution)
  fi
fi
if [[ "${CHECK_QUANT_RUNTIME}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--check-quant-runtime)
  if [[ "${FAIL_ON_QUANT_RUNTIME}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--fail-on-quant-runtime)
  fi
  if [[ "${QUANT_RUNTIME_GENERATE}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--quant-runtime-generate)
  fi
fi
if [[ "${FINALIZE_DRY_RUN}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--finalize-dry-run)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--allow-blocked)
elif [[ "${FAIL_ON_BLOCKED_METHODS}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--fail-on-blocked-methods)
fi
run_cmd "${PYTHON_BIN}" tools/run_edge_preflight.py "${PREFLIGHT_ARGS[@]}"

if [[ "${RUN_PREWARM}" == "1" ]]; then
  PREWARM_ARGS=(
    --config "${CONFIG}"
    --expand-mmlu
    --sleep-seconds "${PREWARM_SLEEP_SECONDS}"
    --report-output "${PREWARM_REPORT_OUTPUT}"
  )
  if [[ "${REQUIRE_PREWARM_ALL_MODELS}" == "1" ]]; then
    PREWARM_ARGS+=(--all-models)
  else
    PREWARM_ARGS+=(--only-model "${PREWARM_MODEL}")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    PREWARM_ARGS+=(--dry-run --sleep-seconds 0)
  fi
  run_cmd "${PYTHON_BIN}" tools/prewarm_lm_eval_tasks.py "${PREWARM_ARGS[@]}"
fi

run_target_readiness pre_matrix

MATRIX_ARGS=(
  --config "${CONFIG}"
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}"
)
if [[ "${STOP_ON_ERROR}" == "1" ]]; then
  MATRIX_ARGS+=(--stop-on-error)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  MATRIX_ARGS+=(--dry-run --finalize --output-root "${DRY_RUN_OUTPUT_ROOT}")
else
  MATRIX_ARGS+=(
    --finalize
    --require-runtime-python-prefix "${REQUIRE_RUNTIME_PYTHON_PREFIX}"
    --require-runtime-gpu-name-contains "${REQUIRE_RUNTIME_GPU_NAME_CONTAINS}"
    --require-runtime-cuda-prefix "${REQUIRE_RUNTIME_CUDA_PREFIX}"
    --require-runtime-gpu-total-memory-min-gb "${REQUIRE_RUNTIME_GPU_TOTAL_MEMORY_MIN_GB}"
    --require-runtime-gpu-count "${REQUIRE_RUNTIME_GPU_COUNT}"
    --require-code-match
    --require-clean-worktree
    --require-asset-audit-report
    --require-preflight-report
    --require-handoff-manifest
  )
  if [[ "${REQUIRE_TARGET_READINESS_REPORT}" == "1" ]]; then
    MATRIX_ARGS+=(--require-target-readiness-report)
  fi
  if [[ "${REQUIRE_NON_ROTATIONAL_STORAGE}" == "1" ]]; then
    MATRIX_ARGS+=(--require-non-rotational-storage)
  fi
  if [[ "${RUN_PREWARM}" == "1" ]]; then
    MATRIX_ARGS+=(--require-prewarm-report)
    if [[ "${REQUIRE_PREWARM_ALL_MODELS}" == "1" ]]; then
      MATRIX_ARGS+=(--require-prewarm-all-models)
    fi
  fi
  if [[ "${RUN_PREPARE}" == "1" ]]; then
    MATRIX_ARGS+=(--require-prepare-report)
  fi
  if [[ "${RUN_METADATA_PREPARE}" == "1" ]]; then
    MATRIX_ARGS+=(--require-prepare-metadata-report)
  fi
  if [[ "${RUN_PRECOMPRESSED_PREPARE}" == "1" ]]; then
    MATRIX_ARGS+=(--require-prepare-precompressed-report)
  fi
  if [[ "${CHECK_OPTIONAL_BACKEND_RESOLUTION}" == "1" ]]; then
    MATRIX_ARGS+=(--require-optional-backends-report)
  fi
  if [[ "${CHECK_QUANT_RUNTIME}" == "1" ]]; then
    MATRIX_ARGS+=(--require-quant-runtime-report)
  fi
  if [[ "${RUN_TARGET_PATH_PREPARE}" == "1" ]]; then
    MATRIX_ARGS+=(--require-target-paths-report)
  fi
  if [[ "${REQUIRE_RUNTIME_CUDNN}" == "1" ]]; then
    MATRIX_ARGS+=(--require-runtime-cudnn)
  fi
  if [[ -n "${REQUIRE_RUNTIME_TORCH_PREFIX}" ]]; then
    MATRIX_ARGS+=(--require-runtime-torch-prefix "${REQUIRE_RUNTIME_TORCH_PREFIX}")
  fi
  if [[ "${REQUIRE_RUNTIME_TRANSFORMERS}" == "1" ]]; then
    MATRIX_ARGS+=(--require-runtime-transformers)
  fi
  if [[ "${REQUIRE_RUNTIME_BITSANDBYTES}" == "1" ]]; then
    MATRIX_ARGS+=(--require-runtime-bitsandbytes)
  fi
  if [[ "${REQUIRE_RUNTIME_CPU_COUNT_MIN}" != "0" ]]; then
    MATRIX_ARGS+=(--require-runtime-cpu-count-min "${REQUIRE_RUNTIME_CPU_COUNT_MIN}")
  fi
  if [[ "${REQUIRE_RUNTIME_SYSTEM_MEMORY_MIN_GB}" != "0" ]]; then
    MATRIX_ARGS+=(--require-runtime-system-memory-min-gb "${REQUIRE_RUNTIME_SYSTEM_MEMORY_MIN_GB}")
  fi
  if [[ -n "${REQUIRE_RUNTIME_HF_ENDPOINT}" ]]; then
    MATRIX_ARGS+=(--require-runtime-hf-endpoint "${REQUIRE_RUNTIME_HF_ENDPOINT}")
  fi
fi

run_cmd "${PYTHON_BIN}" tools/run_edge_matrix.py "${MATRIX_ARGS[@]}"

if [[ "${DRY_RUN}" != "1" ]]; then
  FINAL_AUDIT_ARGS=(
    --config "${CONFIG}"
    --root "${CONFIG_OUTPUT_ROOT}"
    --require-finalize-report
    --require-runtime-python-prefix "${REQUIRE_RUNTIME_PYTHON_PREFIX}"
    --require-runtime-gpu-name-contains "${REQUIRE_RUNTIME_GPU_NAME_CONTAINS}"
    --require-runtime-cuda-prefix "${REQUIRE_RUNTIME_CUDA_PREFIX}"
    --require-runtime-gpu-total-memory-min-gb "${REQUIRE_RUNTIME_GPU_TOTAL_MEMORY_MIN_GB}"
    --require-runtime-gpu-count "${REQUIRE_RUNTIME_GPU_COUNT}"
    --require-code-match
    --require-clean-worktree
    --require-asset-audit-report
    --require-preflight-report
    --require-handoff-manifest
  )
  if [[ "${REQUIRE_TARGET_READINESS_REPORT}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-target-readiness-report)
  fi
  if [[ "${REQUIRE_NON_ROTATIONAL_STORAGE}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-non-rotational-storage)
  fi
  if [[ "${RUN_PREWARM}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-prewarm-report)
    if [[ "${REQUIRE_PREWARM_ALL_MODELS}" == "1" ]]; then
      FINAL_AUDIT_ARGS+=(--require-prewarm-all-models)
    fi
  fi
  if [[ "${RUN_PREPARE}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-prepare-report)
  fi
  if [[ "${RUN_METADATA_PREPARE}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-prepare-metadata-report)
  fi
  if [[ "${RUN_PRECOMPRESSED_PREPARE}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-prepare-precompressed-report)
  fi
  if [[ "${CHECK_OPTIONAL_BACKEND_RESOLUTION}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-optional-backends-report)
  fi
  if [[ "${CHECK_QUANT_RUNTIME}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-quant-runtime-report)
  fi
  if [[ "${RUN_TARGET_PATH_PREPARE}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-target-paths-report)
  fi
  if [[ "${REQUIRE_RUNTIME_CUDNN}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-runtime-cudnn)
  fi
  if [[ -n "${REQUIRE_RUNTIME_TORCH_PREFIX}" ]]; then
    FINAL_AUDIT_ARGS+=(--require-runtime-torch-prefix "${REQUIRE_RUNTIME_TORCH_PREFIX}")
  fi
  if [[ "${REQUIRE_RUNTIME_TRANSFORMERS}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-runtime-transformers)
  fi
  if [[ "${REQUIRE_RUNTIME_BITSANDBYTES}" == "1" ]]; then
    FINAL_AUDIT_ARGS+=(--require-runtime-bitsandbytes)
  fi
  if [[ "${REQUIRE_RUNTIME_CPU_COUNT_MIN}" != "0" ]]; then
    FINAL_AUDIT_ARGS+=(--require-runtime-cpu-count-min "${REQUIRE_RUNTIME_CPU_COUNT_MIN}")
  fi
  if [[ "${REQUIRE_RUNTIME_SYSTEM_MEMORY_MIN_GB}" != "0" ]]; then
    FINAL_AUDIT_ARGS+=(--require-runtime-system-memory-min-gb "${REQUIRE_RUNTIME_SYSTEM_MEMORY_MIN_GB}")
  fi
  if [[ -n "${REQUIRE_RUNTIME_HF_ENDPOINT}" ]]; then
    FINAL_AUDIT_ARGS+=(--require-runtime-hf-endpoint "${REQUIRE_RUNTIME_HF_ENDPOINT}")
  fi
  run_cmd "${PYTHON_BIN}" tools/audit_edge_completion.py \
    "${FINAL_AUDIT_ARGS[@]}"
fi

run_target_readiness post_matrix

if [[ "${DRY_RUN}" != "1" && "${EXPORT_RESULTS_BUNDLE}" == "1" ]]; then
  run_cmd "${PYTHON_BIN}" tools/summarize_edge_results.py \
    --results "${CONFIG_OUTPUT_ROOT}/matrix_results.json" \
    --output-json "${SCORECARD_JSON_OUTPUT}" \
    --output-md "${SCORECARD_MD_OUTPUT}" \
    ${SCORECARD_EXTRA_ARGS}
  run_cmd "${PYTHON_BIN}" tools/export_edge_results_bundle.py \
    --output-root "${CONFIG_OUTPUT_ROOT}" \
    --bundle-dir "${RESULTS_BUNDLE_DIR}" \
    --archive "${RESULTS_BUNDLE_ARCHIVE}" \
    --include-job-logs \
    --require-complete \
    --force
  if [[ "${VERIFY_RESULTS_BUNDLE}" == "1" ]]; then
    run_cmd "${PYTHON_BIN}" "${RESULTS_BUNDLE_DIR}/tools/verify_edge_results_bundle.py" \
      --bundle-root "${RESULTS_BUNDLE_DIR}" \
      --require-complete \
      --require-target-readiness-ok \
      --require-target-runtime \
      --output "${RESULTS_BUNDLE_VERIFY_OUTPUT}"
    run_cmd "${PYTHON_BIN}" tools/verify_edge_delivery.py \
      --root "${CONFIG_OUTPUT_ROOT}" \
      --results-bundle "${RESULTS_BUNDLE_DIR}" \
      --results-archive "${RESULTS_BUNDLE_ARCHIVE}" \
      --output "${DELIVERY_VERIFY_OUTPUT}"
  fi
fi
