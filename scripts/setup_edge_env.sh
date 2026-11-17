#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-llmcbench-edge}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
EDGE_RUNTIME_PROFILE="${EDGE_RUNTIME_PROFILE:-stable}"
VENV_DIR="${VENV_DIR:-.venv}"

case "${EDGE_RUNTIME_PROFILE}" in
  stable)
    DEFAULT_PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    DEFAULT_PYTORCH_PACKAGES="torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0"
    DEFAULT_PROFILE_EXTRA_PACKAGES=""
    ;;
  smoothquant)
    DEFAULT_PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    DEFAULT_PYTORCH_PACKAGES="torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0"
    DEFAULT_PROFILE_EXTRA_PACKAGES="transformers==4.57.6 compressed-tensors==0.15.0.1 bitsandbytes peft==0.10.0"
    ;;
  gptqmodel)
    DEFAULT_PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    DEFAULT_PYTORCH_PACKAGES="torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0"
    DEFAULT_PROFILE_EXTRA_PACKAGES="numpy>=2.2.6 transformers>=4.55.2"
    ;;
  *)
    echo "Unsupported EDGE_RUNTIME_PROFILE=${EDGE_RUNTIME_PROFILE}. Use stable, smoothquant or gptqmodel." >&2
    exit 1
    ;;
esac

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-${DEFAULT_PYTORCH_INDEX_URL}}"
PYTORCH_PACKAGES="${PYTORCH_PACKAGES:-${DEFAULT_PYTORCH_PACKAGES}}"
PROFILE_EXTRA_PACKAGES="${PROFILE_EXTRA_PACKAGES:-${DEFAULT_PROFILE_EXTRA_PACKAGES}}"

if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN=conda
else
  CONDA_BIN=""
fi

if [[ -n "${CONDA_BIN}" ]]; then
  if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    "${CONDA_BIN}" create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
  fi

  set +u
  source "$("${CONDA_BIN}" info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  set -u
else
  if command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
    PYTHON_BIN="python${PYTHON_VERSION}"
  elif command -v python3 >/dev/null 2>&1 && python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
  then
    PYTHON_BIN=python3
  else
    echo "conda/mamba was not found and python${PYTHON_VERSION} is unavailable." >&2
    echo "Install Miniconda/Mambaforge or system Python ${PYTHON_VERSION} with venv support." >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install --index-url "${PYTORCH_INDEX_URL}" ${PYTORCH_PACKAGES}
python -m pip install -r requirements.txt
if [[ -n "${PROFILE_EXTRA_PACKAGES}" ]]; then
  python -m pip install ${PROFILE_EXTRA_PACKAGES}
fi

python tools/verify_edge_env.py --config configs/edge_4090_matrix.json --check-cli || true
