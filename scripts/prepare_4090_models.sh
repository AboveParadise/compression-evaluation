#!/usr/bin/env bash
set -euo pipefail

# Prepare the two base models and all compressed checkpoints required by the
# single-RTX-4090 evaluation matrix. Downloads are preferred when a complete,
# same-base checkpoint is available; otherwise the checkpoint is generated
# locally from the downloaded base model.

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"
COMPRESSED_ROOT="${COMPRESSED_ROOT:-${MODEL_ROOT}/precompressed}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/hf_cache}"
DATASET_CACHE="${DATASET_CACHE:-/data/llm_eval_datasets}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/outputs/llmcbench_edge_4090}"
OUTPUT_CONFIG="${OUTPUT_CONFIG:-${OUTPUT_ROOT}/edge_4090_prepared_models.json}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_MEMORY="${MAX_MEMORY:-0:22GiB,cpu:120GiB}"
MIN_FREE_GPU_MEMORY_GB="${MIN_FREE_GPU_MEMORY_GB:-20}"
TARGET_WEIGHT_BITS="${TARGET_WEIGHT_BITS:-4}"
QUANT_GROUP_SIZE="${QUANT_GROUP_SIZE:-128}"
SMOOTHQUANT_WEIGHT_BITS="${SMOOTHQUANT_WEIGHT_BITS:-8}"
DRY_RUN="${DRY_RUN:-0}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
FORCE="${FORCE:-0}"
ALLOW_LOCAL_QUANT_FALLBACK="${ALLOW_LOCAL_QUANT_FALLBACK:-1}"

LLAMA_BASE_REPO="${LLAMA_BASE_REPO:-LLM-Research/Meta-Llama-3-8B}"
LLAMA_BASE_REVISION="${LLAMA_BASE_REVISION:-master}"
QWEN_BASE_REPO="${QWEN_BASE_REPO:-Qwen/Qwen3-8B}"
QWEN_BASE_REVISION="${QWEN_BASE_REVISION:-master}"

LLAMA_GPTQ_REPO="${LLAMA_GPTQ_REPO:-Ksgk-fy/Meta-Llama-3-8B-4bit-GPTQ}"
LLAMA_GPTQ_REVISION="${LLAMA_GPTQ_REVISION:-0f30b0e053ba5d0b38b6bc2eb729dc48ab8edc5c}"
LLAMA_AWQ_REPO="${LLAMA_AWQ_REPO:-solidrust/Meta-Llama-3-8B-hf-AWQ}"
LLAMA_AWQ_REVISION="${LLAMA_AWQ_REVISION:-64bee1e93799fbfa1337fc8f824b0cc6569a7919}"
QWEN_GPTQ_REPO="${QWEN_GPTQ_REPO:-kaitchup/Qwen3-8B-autoround-4bit-gptq}"
QWEN_GPTQ_REVISION="${QWEN_GPTQ_REVISION:-b7e026b92d0019c20745eae7f843fac019c9c6e0}"
QWEN_AWQ_REPO="${QWEN_AWQ_REPO:-Qwen/Qwen3-8B-AWQ}"
QWEN_AWQ_REVISION="${QWEN_AWQ_REVISION:-4da05a8edb55c6046cce958586c33b61da07bb79}"

LLAMA_BASE="${MODEL_ROOT}/Meta-Llama-3-8B"
QWEN_BASE="${MODEL_ROOT}/Qwen3-8B"

declare -A CHECKPOINTS=(
  ["LLaMA3-8B/wanda"]="${COMPRESSED_ROOT}/LLaMA3-8B-wanda"
  ["LLaMA3-8B/sparsegpt"]="${COMPRESSED_ROOT}/LLaMA3-8B-sparsegpt"
  ["LLaMA3-8B/gptq"]="${COMPRESSED_ROOT}/LLaMA3-8B-gptq-g128"
  ["LLaMA3-8B/awq"]="${COMPRESSED_ROOT}/LLaMA3-8B-awq"
  ["LLaMA3-8B/smoothquant"]="${COMPRESSED_ROOT}/LLaMA3-8B-smoothquant-w8a8-adapted"
  ["LLaMA3-8B/omniquant"]="${COMPRESSED_ROOT}/LLaMA3-8B-omniquant-w4g128-adapted"
  ["Qwen3-8B/wanda"]="${COMPRESSED_ROOT}/Qwen3-8B-wanda"
  ["Qwen3-8B/sparsegpt"]="${COMPRESSED_ROOT}/Qwen3-8B-sparsegpt"
  ["Qwen3-8B/gptq"]="${COMPRESSED_ROOT}/Qwen3-8B-gptq-g128"
  ["Qwen3-8B/awq"]="${COMPRESSED_ROOT}/Qwen3-8B-awq"
  ["Qwen3-8B/smoothquant"]="${COMPRESSED_ROOT}/Qwen3-8B-smoothquant-w8a8-adapted"
  ["Qwen3-8B/omniquant"]="${COMPRESSED_ROOT}/Qwen3-8B-omniquant-w4g128-adapted"
)

if [[ "${TARGET_WEIGHT_BITS}" != "4" ]]; then
  echo "TARGET_WEIGHT_BITS must be 4 for the aligned GPTQ/AWQ/OmniQuant profile." >&2
  exit 2
fi
if [[ "${QUANT_GROUP_SIZE}" != "128" ]]; then
  echo "QUANT_GROUP_SIZE must be 128 for the aligned quantization profile." >&2
  exit 2
fi
if [[ "${SMOOTHQUANT_WEIGHT_BITS}" != "8" ]]; then
  echo "SmoothQuant is evaluated as its standard W8A8 method; SMOOTHQUANT_WEIGHT_BITS must be 8." >&2
  exit 2
fi

export HF_ENDPOINT CUDA_VISIBLE_DEVICES
export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${DATASET_CACHE}"

quote_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  quote_cmd "$@"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

checkpoint_ready() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
config = root / "config.json"
if not config.is_file():
    raise SystemExit(1)
expected = []
for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
    index = root / index_name
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        expected.extend(sorted(set((payload.get("weight_map") or {}).values())))
if expected:
    ok = all((root / name).is_file() and (root / name).stat().st_size > 0 for name in expected)
else:
    ok = any(path.is_file() and path.stat().st_size > 0 for pattern in ("*.safetensors", "pytorch_model*.bin") for path in root.glob(pattern))
raise SystemExit(0 if ok else 1)
PY
}

write_source_marker() {
  local output_dir="$1"
  local provider="$2"
  local repo="$3"
  local revision="$4"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "${PYTHON_BIN}" - "${output_dir}" "${provider}" "${repo}" "${revision}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
hashes = {}
for name in ("config.json", "model.safetensors.index.json"):
    path = root / name
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
payload = {
    "schema_version": 1,
    "provider": sys.argv[2],
    "repository": sys.argv[3],
    "revision": sys.argv[4],
    "local_path": str(root),
    "files_sha256": hashes,
}
(root / ".llmcbench_source.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

write_precompressed_marker() {
  local output_dir="$1"
  local model_name="$2"
  local method="$3"
  local repo="$4"
  local revision="$5"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "${PYTHON_BIN}" - "${output_dir}" "${model_name}" "${method}" "${repo}" "${revision}" "${HF_ENDPOINT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "model": sys.argv[2],
    "method": sys.argv[3],
    "repo": sys.argv[4],
    "revision": sys.argv[5],
    "hf_endpoint": sys.argv[6],
    "metadata_only": False,
}
(root / ".llmcbench_precompressed_source.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

download_modelscope() {
  local repo="$1"
  local revision="$2"
  local output_dir="$3"
  if checkpoint_ready "${output_dir}" 2>/dev/null && [[ "${FORCE}" != "1" ]]; then
    echo "[ready] ${output_dir}"
    return
  fi
  run_cmd modelscope download --model "${repo}" --revision "${revision}" --local_dir "${output_dir}"
  write_source_marker "${output_dir}" modelscope "${repo}" "${revision}"
}

hf_download_command() {
  if command -v hf >/dev/null 2>&1; then
    printf '%s\n' hf
  elif command -v huggingface-cli >/dev/null 2>&1; then
    printf '%s\n' huggingface-cli
  else
    echo "Neither hf nor huggingface-cli is installed." >&2
    return 1
  fi
}

download_hf() {
  local repo="$1"
  local revision="$2"
  local output_dir="$3"
  local model_name="$4"
  local method="$5"
  if checkpoint_ready "${output_dir}" 2>/dev/null && [[ "${FORCE}" != "1" ]]; then
    echo "[ready] ${output_dir}"
    return
  fi
  local hf_bin
  hf_bin="$(hf_download_command)"
  if [[ "${hf_bin}" == "hf" ]]; then
    if ! run_cmd hf download "${repo}" --revision "${revision}" --local-dir "${output_dir}" --max-workers 2; then
      return 1
    fi
  else
    if ! run_cmd huggingface-cli download "${repo}" --revision "${revision}" --local-dir "${output_dir}" --max-workers 2; then
      return 1
    fi
  fi
  write_precompressed_marker "${output_dir}" "${model_name}" "${method}" "${repo}" "${revision}"
}

compression_args() {
  local base_model="$1"
  local model_name="$2"
  local method="$3"
  local output_dir="$4"
  local setting="$5"
  local args=(
    "${PYTHON_BIN}" -m compression
    --base-model "${base_model}"
    --model-name "${model_name}"
    --method "${method}"
    --setting "${setting}"
    --output-dir "${output_dir}"
    --cache-dir "${HF_CACHE_DIR}"
    --dataset-cache "${DATASET_CACHE}"
    --hf-endpoint "${HF_ENDPOINT}"
    --device cuda:0
    --device-map auto
    --dtype fp16
    --max-memory "${MAX_MEMORY}"
    --min-free-gpu-memory-gb "${MIN_FREE_GPU_MEMORY_GB}"
  )
  if [[ "${FORCE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    args+=(--dry-run)
  fi
  printf '%s\0' "${args[@]}"
}

compress_local() {
  local base_model="$1"
  local model_name="$2"
  local method="$3"
  local output_dir="$4"
  local setting="$5"
  if checkpoint_ready "${output_dir}" 2>/dev/null && [[ "${FORCE}" != "1" ]]; then
    echo "[ready] ${output_dir}"
    return
  fi
  local args=()
  while IFS= read -r -d '' item; do args+=("${item}"); done < <(compression_args "${base_model}" "${model_name}" "${method}" "${output_dir}" "${setting}")
  quote_cmd "${args[@]}"
  "${args[@]}"
}

compress_download_fallback() {
  local base_model="$1"
  local model_name="$2"
  local method="$3"
  local output_dir="$4"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[fallback] local ${method} compression would run if the download fails"
    return
  fi
  if [[ "${ALLOW_LOCAL_QUANT_FALLBACK}" != "1" ]]; then
    return 1
  fi
  local extra=()
  if [[ "${method}" == "gptq" ]]; then
    extra=(
      --gptq-calibration-source c4 --calibration-seed 0 --calibration-samples 128 --seqlen 2048
      --quant-bits 4 --group-size 128 --gptq-damp-percent 0.01
    )
  else
    extra=(
      --quant-bits 4 --group-size 128 --awq-calibration-source pileval
      --awq-calibration-seed 42 --awq-calibration-samples 128
      --awq-calibration-seqlen 512 --awq-grid-size 20 --awq-version GEMM
    )
  fi
  local cmd=(
    "${PYTHON_BIN}" -m edge_benchmark compress
    --model-path "${base_model}" --model-name "${model_name}"
    --compression-method "${method}" --compressed-output-dir "${output_dir}"
    --cache-dir "${HF_CACHE_DIR}" --hf-endpoint "${HF_ENDPOINT}"
    --device cuda:0 --device-map auto --dtype fp16 --max-memory "${MAX_MEMORY}"
    --min-free-gpu-memory-gb "${MIN_FREE_GPU_MEMORY_GB}" --calibration-batch-size 1
    "${extra[@]}"
  )
  if [[ "${FORCE}" == "1" ]]; then cmd+=(--overwrite-compressed); fi
  run_cmd "${cmd[@]}"
}

download_or_compress() {
  local repo="$1"
  local revision="$2"
  local base_model="$3"
  local model_name="$4"
  local method="$5"
  local output_dir="$6"
  if checkpoint_ready "${output_dir}" 2>/dev/null && [[ "${FORCE}" != "1" ]]; then
    echo "[ready] ${output_dir}"
    return
  fi
  if ! download_hf "${repo}" "${revision}" "${output_dir}" "${model_name}" "${method}"; then
    if checkpoint_ready "${output_dir}" 2>/dev/null; then
      echo "[ready after downloader warning] ${output_dir}"
      return
    fi
    if [[ -d "${output_dir}" ]]; then
      local quarantine="${output_dir}.download-failed-$(date -u +%Y%m%dT%H%M%SZ)"
      run_cmd mv -- "${output_dir}" "${quarantine}"
      echo "Incomplete download moved to ${quarantine}" >&2
    fi
    echo "Download failed for ${model_name}/${method}; trying local compression." >&2
    compress_download_fallback "${base_model}" "${model_name}" "${method}" "${output_dir}"
  fi
}

write_matrix_config() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] would write ${OUTPUT_CONFIG}"
    return
  fi
  mkdir -p "$(dirname "${OUTPUT_CONFIG}")"
  "${PYTHON_BIN}" - \
    configs/edge_4090_matrix.json "${OUTPUT_CONFIG}" "${OUTPUT_ROOT}" \
    "${LLAMA_BASE}" "${QWEN_BASE}" \
    "${CHECKPOINTS[LLaMA3-8B/wanda]}" "${CHECKPOINTS[LLaMA3-8B/sparsegpt]}" \
    "${CHECKPOINTS[LLaMA3-8B/gptq]}" "${CHECKPOINTS[LLaMA3-8B/awq]}" \
    "${CHECKPOINTS[LLaMA3-8B/smoothquant]}" "${CHECKPOINTS[LLaMA3-8B/omniquant]}" \
    "${CHECKPOINTS[Qwen3-8B/wanda]}" "${CHECKPOINTS[Qwen3-8B/sparsegpt]}" \
    "${CHECKPOINTS[Qwen3-8B/gptq]}" "${CHECKPOINTS[Qwen3-8B/awq]}" \
    "${CHECKPOINTS[Qwen3-8B/smoothquant]}" "${CHECKPOINTS[Qwen3-8B/omniquant]}" <<'PY'
import json
import sys
from pathlib import Path

template, output, output_root, llama_base, qwen_base, *paths = sys.argv[1:]
payload = json.loads(Path(template).read_text(encoding="utf-8"))
payload["models"] = [
    {"name": "LLaMA3-8B", "path": llama_base},
    {"name": "Qwen3-8B", "path": qwen_base},
]
methods = ["wanda", "sparsegpt", "gptq", "awq", "smoothquant", "omniquant"]
payload["precompressed_paths"] = {
    "LLaMA3-8B": dict(zip(methods, paths[:6])),
    "Qwen3-8B": dict(zip(methods, paths[6:])),
}
payload["defaults"]["output_root"] = output_root
payload["defaults"]["cuda_visible_devices"] = "0"
payload["defaults"]["quant_bits"] = 4
payload["defaults"]["group_size"] = 128
payload.setdefault("method_overrides", {}).setdefault("smoothquant", {})["quant_bits"] = 8
Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(output)
PY
}

verify_all() {
  if [[ "${DRY_RUN}" == "1" ]]; then return; fi
  local failed=0
  for path in "${LLAMA_BASE}" "${QWEN_BASE}" "${CHECKPOINTS[@]}"; do
    if checkpoint_ready "${path}"; then
      echo "[verified] ${path}"
    else
      echo "[missing or incomplete] ${path}" >&2
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then return 1; fi
  "${PYTHON_BIN}" tools/check_method_readiness.py --config "${OUTPUT_CONFIG}"
  "${PYTHON_BIN}" - \
    "${CHECKPOINTS[LLaMA3-8B/gptq]}" "${CHECKPOINTS[LLaMA3-8B/awq]}" \
    "${CHECKPOINTS[LLaMA3-8B/smoothquant]}" "${CHECKPOINTS[LLaMA3-8B/omniquant]}" \
    "${CHECKPOINTS[Qwen3-8B/gptq]}" "${CHECKPOINTS[Qwen3-8B/awq]}" \
    "${CHECKPOINTS[Qwen3-8B/smoothquant]}" "${CHECKPOINTS[Qwen3-8B/omniquant]}" <<'PY'
import sys
from pathlib import Path

from edge_benchmark.checkpoint_inspection import inspect_quantization_storage

methods = ["gptq", "awq", "smoothquant", "omniquant"] * 2
expected = {
    "gptq": {"weight_bits": 4, "group_size": 128},
    "awq": {"weight_bits": 4, "group_size": 128},
    "smoothquant": {"weight_bits": 8, "activation_bits": 8},
    "omniquant": {"weight_bits": 4, "activation_bits": 16, "group_size": 128},
}
failures = []
for method, path_value in zip(methods, sys.argv[1:]):
    status = inspect_quantization_storage(Path(path_value))
    mismatches = {
        key: {"expected": value, "actual": status.get(key)}
        for key, value in expected[method].items()
        if status.get(key) != value
    }
    if status.get("is_real_quantized") is not True:
        mismatches["is_real_quantized"] = {"expected": True, "actual": status.get("is_real_quantized")}
    if mismatches:
        failures.append({"method": method, "path": path_value, "mismatches": mismatches, "status": status})
if failures:
    raise SystemExit(f"Quantization alignment check failed: {failures}")
print("Quantization alignment verified: GPTQ/AWQ/OmniQuant=W4G128; SmoothQuant=W8A8")
PY
}

mkdir -p "${MODEL_ROOT}" "${COMPRESSED_ROOT}" "${HF_CACHE_DIR}" "${DATASET_CACHE}"

if [[ "${VERIFY_ONLY}" != "1" ]]; then
  download_modelscope "${LLAMA_BASE_REPO}" "${LLAMA_BASE_REVISION}" "${LLAMA_BASE}"
  download_modelscope "${QWEN_BASE_REPO}" "${QWEN_BASE_REVISION}" "${QWEN_BASE}"

  compress_local "${LLAMA_BASE}" LLaMA3-8B wanda "${CHECKPOINTS[LLaMA3-8B/wanda]}" paper
  compress_local "${LLAMA_BASE}" LLaMA3-8B sparsegpt "${CHECKPOINTS[LLaMA3-8B/sparsegpt]}" paper
  download_or_compress "${LLAMA_GPTQ_REPO}" "${LLAMA_GPTQ_REVISION}" "${LLAMA_BASE}" LLaMA3-8B gptq "${CHECKPOINTS[LLaMA3-8B/gptq]}"
  download_or_compress "${LLAMA_AWQ_REPO}" "${LLAMA_AWQ_REVISION}" "${LLAMA_BASE}" LLaMA3-8B awq "${CHECKPOINTS[LLaMA3-8B/awq]}"
  compress_local "${LLAMA_BASE}" LLaMA3-8B smoothquant "${CHECKPOINTS[LLaMA3-8B/smoothquant]}" adapted
  compress_local "${LLAMA_BASE}" LLaMA3-8B omniquant "${CHECKPOINTS[LLaMA3-8B/omniquant]}" adapted

  compress_local "${QWEN_BASE}" Qwen3-8B wanda "${CHECKPOINTS[Qwen3-8B/wanda]}" paper
  compress_local "${QWEN_BASE}" Qwen3-8B sparsegpt "${CHECKPOINTS[Qwen3-8B/sparsegpt]}" paper
  download_or_compress "${QWEN_GPTQ_REPO}" "${QWEN_GPTQ_REVISION}" "${QWEN_BASE}" Qwen3-8B gptq "${CHECKPOINTS[Qwen3-8B/gptq]}"
  download_or_compress "${QWEN_AWQ_REPO}" "${QWEN_AWQ_REVISION}" "${QWEN_BASE}" Qwen3-8B awq "${CHECKPOINTS[Qwen3-8B/awq]}"
  compress_local "${QWEN_BASE}" Qwen3-8B smoothquant "${CHECKPOINTS[Qwen3-8B/smoothquant]}" adapted
  compress_local "${QWEN_BASE}" Qwen3-8B omniquant "${CHECKPOINTS[Qwen3-8B/omniquant]}" adapted
fi

write_matrix_config
verify_all

echo "Prepared model config: ${OUTPUT_CONFIG}"
