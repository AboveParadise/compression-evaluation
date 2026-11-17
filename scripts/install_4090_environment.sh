#!/usr/bin/env bash
set -euo pipefail

# Install the reproducible single-RTX-4090 LLMCBench runtime and verify the
# requested Python/PyTorch/CUDA/cuDNN/bitsandbytes/Transformers stack.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_ROOT="${LLMCBENCH_ROOT:-${PROJECT_ROOT}/.llmcbench}"
cd "${PROJECT_ROOT}"

ENV_NAME="${ENV_NAME:-llmcbench-edge}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-${RUNTIME_ROOT}/venv}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
PYTORCH_PACKAGES="${PYTORCH_PACKAGES:-torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0}"
MODELSCOPE_PACKAGE="${MODELSCOPE_PACKAGE:-modelscope>=1.20,<2}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-constraints-cu124.txt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REQUIRE_GPU_NAME="${REQUIRE_GPU_NAME:-4090}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-23}"
ALLOW_NON_4090="${ALLOW_NON_4090:-0}"
DRY_RUN="${DRY_RUN:-0}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
REPORT_OUTPUT="${REPORT_OUTPUT:-${RUNTIME_ROOT}/outputs/llmcbench_edge_4090/edge_env_report.json}"

export CUDA_VISIBLE_DEVICES

quote_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
  quote_cmd python -m pip install --constraint "${CONSTRAINTS_FILE}" --index-url "${PYTORCH_INDEX_URL}" ${PYTORCH_PACKAGES}
  quote_cmd python -m pip install --constraint "${CONSTRAINTS_FILE}" -r requirements.txt "${MODELSCOPE_PACKAGE}"
  echo "+ PIP_CONSTRAINT=${CONSTRAINTS_FILE} INSTALL_GPTQMODEL=0 bash scripts/setup_optional_backends.sh"
  echo "+ verify Python/PyTorch/CUDA/cuDNN/bitsandbytes/Transformers/GPU/CLI and write ${REPORT_OUTPUT}"
  exit 0
fi

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  CONDA_BIN=""
elif command -v mamba >/dev/null 2>&1; then
  CONDA_BIN=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN=conda
else
  CONDA_BIN=""
fi

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  echo "[verify-only] using $(command -v python)"
elif [[ -n "${CONDA_BIN}" ]]; then
  if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    "${CONDA_BIN}" create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
  fi
  set +u
  source "$("${CONDA_BIN}" info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  set -u
else
  if command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
    PYTHON_BOOTSTRAP="python${PYTHON_VERSION}"
  elif python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)'; then
    PYTHON_BOOTSTRAP=python3
  else
    echo "Python ${PYTHON_VERSION} or conda/mamba is required." >&2
    exit 1
  fi
  "${PYTHON_BOOTSTRAP}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

if [[ "${VERIFY_ONLY}" != "1" ]]; then
  if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
    echo "Constraint file does not exist: ${CONSTRAINTS_FILE}" >&2
    exit 1
  fi
  CONSTRAINTS_FILE="$(realpath "${CONSTRAINTS_FILE}")"
  python -m pip install --upgrade pip setuptools wheel
  # shellcheck disable=SC2086
  python -m pip install --constraint "${CONSTRAINTS_FILE}" --index-url "${PYTORCH_INDEX_URL}" ${PYTORCH_PACKAGES}
  python -m pip install --constraint "${CONSTRAINTS_FILE}" -r requirements.txt "${MODELSCOPE_PACKAGE}"
  PIP_CONSTRAINT="${CONSTRAINTS_FILE}" INSTALL_GPTQMODEL=0 PYTHON_BIN=python bash scripts/setup_optional_backends.sh
fi

mkdir -p "$(dirname "${REPORT_OUTPUT}")"
python - "${REQUIRE_GPU_NAME}" "${MIN_GPU_MEMORY_GB}" "${ALLOW_NON_4090}" "${REPORT_OUTPUT}" <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path

import bitsandbytes
import modelscope
import torch
import transformers

required_name = sys.argv[1]
minimum_gb = float(sys.argv[2])
allow_other = sys.argv[3] == "1"
report_output = Path(sys.argv[4])
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split("+", 1)[0].startswith("2."), torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("12."), torch.version.cuda
assert torch.backends.cudnn.version() is not None
assert torch.cuda.is_available(), "CUDA is not available"
gpu = torch.cuda.get_device_properties(0)
total_gb = gpu.total_memory / 1024**3
if not allow_other:
    assert required_name.lower() in gpu.name.lower(), (required_name, gpu.name)
assert total_gb >= minimum_gb, (minimum_gb, total_gb)
cli = subprocess.run([sys.executable, "-m", "edge_benchmark", "--help"], text=True, capture_output=True, check=False)
modelscope = subprocess.run(["modelscope", "--help"], text=True, capture_output=True, check=False)
assert cli.returncode == 0, cli.stderr
assert modelscope.returncode == 0, modelscope.stderr
payload = {
    "ok": True,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "bitsandbytes": getattr(bitsandbytes, "__version__", None),
    "modelscope": getattr(modelscope, "__version__", None),
    "transformers": transformers.__version__,
    "gpu": gpu.name,
    "gpu_total_gb": total_gb,
    "edge_benchmark_cli": True,
    "modelscope_cli": True,
}
report_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "Environment verification report: ${REPORT_OUTPUT}"
