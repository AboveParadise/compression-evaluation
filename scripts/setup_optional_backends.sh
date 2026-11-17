#!/usr/bin/env bash
set -euo pipefail

# Install optional compression backends after the base Python 3.10 edge
# environment is active. Keep this separate from setup_edge_env.sh so users can
# retry backend-specific installs without rebuilding the whole environment.

INSTALL_BITSANDBYTES="${INSTALL_BITSANDBYTES:-1}"
INSTALL_OPTIMUM="${INSTALL_OPTIMUM:-1}"
INSTALL_GPTQ="${INSTALL_GPTQ:-1}"
INSTALL_GPTQMODEL="${INSTALL_GPTQMODEL:-1}"
INSTALL_AWQ="${INSTALL_AWQ:-1}"
INSTALL_COMPRESSED_TENSORS="${INSTALL_COMPRESSED_TENSORS:-1}"
OPTIONAL_REQUIREMENTS="${OPTIONAL_REQUIREMENTS:-requirements-optional.txt}"
OPTIONAL_BACKEND_DRY_RUN="${OPTIONAL_BACKEND_DRY_RUN:-0}"
REPORT_OUTPUT="${REPORT_OUTPUT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OPTIMUM_PACKAGE="${OPTIMUM_PACKAGE:-optimum<2}"
GPTQMODEL_PACKAGE="gptqmodel==4.0.0"

REQUESTED_PACKAGES=()
REQUESTED_GPTQMODEL=()
if [[ "${INSTALL_BITSANDBYTES}" == "1" ]]; then
  REQUESTED_PACKAGES+=(bitsandbytes)
fi
if [[ "${INSTALL_OPTIMUM}" == "1" ]]; then
  REQUESTED_PACKAGES+=("${OPTIMUM_PACKAGE}")
fi
if [[ "${INSTALL_GPTQ}" == "1" ]]; then
  REQUESTED_PACKAGES+=(auto-gptq)
fi
if [[ "${INSTALL_GPTQMODEL}" == "1" ]]; then
  REQUESTED_GPTQMODEL+=("${GPTQMODEL_PACKAGE}")
fi
if [[ "${INSTALL_AWQ}" == "1" ]]; then
  REQUESTED_PACKAGES+=(autoawq)
fi
if [[ "${INSTALL_COMPRESSED_TENSORS}" == "1" ]]; then
  REQUESTED_PACKAGES+=(compressed-tensors)
fi

INSTALL_ARGS=()
if [[ "${INSTALL_BITSANDBYTES}" == "1" && "${INSTALL_OPTIMUM}" == "1" && "${INSTALL_GPTQ}" == "1" && "${INSTALL_AWQ}" == "1" && "${INSTALL_COMPRESSED_TENSORS}" == "1" && -f "${OPTIONAL_REQUIREMENTS}" ]]; then
  INSTALL_ARGS=(-r "${OPTIONAL_REQUIREMENTS}")
else
  INSTALL_ARGS=("${REQUESTED_PACKAGES[@]}")
fi

if [[ "${#REQUESTED_PACKAGES[@]}" == "0" && "${#REQUESTED_GPTQMODEL[@]}" == "0" ]]; then
  echo "No optional backend packages selected."
  exit 0
fi

if [[ "${OPTIONAL_BACKEND_DRY_RUN}" == "1" ]]; then
  PIP_REPORT="$(mktemp /tmp/llmcbench_optional_pip_report.XXXXXX.json)"
  GPTQMODEL_PIP_REPORT="$(mktemp /tmp/llmcbench_gptqmodel_pip_report.XXXXXX.json)"
  STDOUT_LOG="$(mktemp /tmp/llmcbench_optional_pip_stdout.XXXXXX.log)"
  STDERR_LOG="$(mktemp /tmp/llmcbench_optional_pip_stderr.XXXXXX.log)"
  GPTQMODEL_STDOUT_LOG="$(mktemp /tmp/llmcbench_gptqmodel_pip_stdout.XXXXXX.log)"
  GPTQMODEL_STDERR_LOG="$(mktemp /tmp/llmcbench_gptqmodel_pip_stderr.XXXXXX.log)"
  set +e
  if [[ "${#INSTALL_ARGS[@]}" == "0" ]]; then
    echo "No standard optional backend packages selected." >"${STDOUT_LOG}"
    : >"${STDERR_LOG}"
    printf '{"install":[]}\n' >"${PIP_REPORT}"
    PIP_RETURN_CODE=0
  else
    "${PYTHON_BIN}" -m pip install --dry-run --report "${PIP_REPORT}" "${INSTALL_ARGS[@]}" >"${STDOUT_LOG}" 2>"${STDERR_LOG}"
    PIP_RETURN_CODE="$?"
  fi
  GPTQMODEL_PIP_RETURN_CODE=0
  if [[ "${INSTALL_GPTQMODEL}" == "1" ]]; then
    if "${PYTHON_BIN}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
    then
      "${PYTHON_BIN}" -m pip install --dry-run --no-build-isolation --report "${GPTQMODEL_PIP_REPORT}" "${GPTQMODEL_PACKAGE}" >"${GPTQMODEL_STDOUT_LOG}" 2>"${GPTQMODEL_STDERR_LOG}"
      GPTQMODEL_PIP_RETURN_CODE="$?"
    else
      echo "Skipping ${GPTQMODEL_PACKAGE}: target Python 3.10+ dependency." >"${GPTQMODEL_STDOUT_LOG}"
      : >"${GPTQMODEL_STDERR_LOG}"
      printf '{"install":[]}\n' >"${GPTQMODEL_PIP_REPORT}"
      GPTQMODEL_PIP_RETURN_CODE=0
    fi
  else
    echo "GPTQModel install disabled." >"${GPTQMODEL_STDOUT_LOG}"
    : >"${GPTQMODEL_STDERR_LOG}"
    printf '{"install":[]}\n' >"${GPTQMODEL_PIP_REPORT}"
  fi
  set -e
  PIP_REPORT="${PIP_REPORT}" \
  GPTQMODEL_PIP_REPORT="${GPTQMODEL_PIP_REPORT}" \
  STDOUT_LOG="${STDOUT_LOG}" \
  STDERR_LOG="${STDERR_LOG}" \
  GPTQMODEL_STDOUT_LOG="${GPTQMODEL_STDOUT_LOG}" \
  GPTQMODEL_STDERR_LOG="${GPTQMODEL_STDERR_LOG}" \
  PIP_RETURN_CODE="${PIP_RETURN_CODE}" \
  GPTQMODEL_PIP_RETURN_CODE="${GPTQMODEL_PIP_RETURN_CODE}" \
  REPORT_OUTPUT="${REPORT_OUTPUT}" \
  OPTIONAL_REQUIREMENTS="${OPTIONAL_REQUIREMENTS}" \
  GPTQMODEL_PACKAGE="${GPTQMODEL_PACKAGE}" \
  INSTALL_BITSANDBYTES="${INSTALL_BITSANDBYTES}" \
  INSTALL_OPTIMUM="${INSTALL_OPTIMUM}" \
  INSTALL_GPTQ="${INSTALL_GPTQ}" \
  INSTALL_GPTQMODEL="${INSTALL_GPTQMODEL}" \
  INSTALL_AWQ="${INSTALL_AWQ}" \
  INSTALL_COMPRESSED_TENSORS="${INSTALL_COMPRESSED_TENSORS}" \
  OPTIMUM_PACKAGE="${OPTIMUM_PACKAGE}" \
  "${PYTHON_BIN}" - <<'PY'
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

try:
    from edge_benchmark.runtime_compat import apply_runtime_compatibility_patches
    apply_runtime_compatibility_patches()
except Exception:
    pass

pip_report_path = Path(os.environ["PIP_REPORT"])
gptqmodel_pip_report_path = Path(os.environ["GPTQMODEL_PIP_REPORT"])
stdout_path = Path(os.environ["STDOUT_LOG"])
stderr_path = Path(os.environ["STDERR_LOG"])
gptqmodel_stdout_path = Path(os.environ["GPTQMODEL_STDOUT_LOG"])
gptqmodel_stderr_path = Path(os.environ["GPTQMODEL_STDERR_LOG"])
try:
    pip_report = json.loads(pip_report_path.read_text(encoding="utf-8"))
except Exception as exc:
    pip_report = {"error": f"failed_to_read_pip_report:{exc}"}
try:
    gptqmodel_pip_report = json.loads(gptqmodel_pip_report_path.read_text(encoding="utf-8"))
except Exception as exc:
    gptqmodel_pip_report = {"error": f"failed_to_read_gptqmodel_pip_report:{exc}"}

modules = []
requested_packages = []
if os.environ.get("INSTALL_BITSANDBYTES") == "1":
    modules.append("bitsandbytes")
    requested_packages.append("bitsandbytes")
if os.environ.get("INSTALL_OPTIMUM") == "1":
    modules.append("optimum")
    requested_packages.append(os.environ.get("OPTIMUM_PACKAGE", "optimum<2"))
if os.environ.get("INSTALL_GPTQ") == "1":
    modules.append("auto_gptq")
    requested_packages.append("auto-gptq")
if os.environ.get("INSTALL_GPTQMODEL") == "1" and sys.version_info >= (3, 10):
    modules.append("gptqmodel")
    requested_packages.append(os.environ.get("GPTQMODEL_PACKAGE", "gptqmodel==4.0.0"))
elif os.environ.get("INSTALL_GPTQMODEL") == "1":
    requested_packages.append(f"{os.environ.get('GPTQMODEL_PACKAGE', 'gptqmodel==4.0.0')}; python_version >= '3.10'")
if os.environ.get("INSTALL_AWQ") == "1":
    modules.append("awq")
    requested_packages.append("autoawq")
if os.environ.get("INSTALL_COMPRESSED_TENSORS") == "1":
    modules.append("compressed_tensors")
    requested_packages.append("compressed-tensors")

imports = {}
for name in modules:
    try:
        module = importlib.import_module(name)
        imports[name] = {"ok": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        imports[name] = {"ok": False, "error": str(exc)}
if os.environ.get("INSTALL_GPTQMODEL") == "1" and sys.version_info < (3, 10):
    imports["gptqmodel"] = {
        "ok": True,
        "skipped": True,
        "reason": "gptqmodel is a target Python 3.10+ runtime dependency",
    }

resolved = []
for item in pip_report.get("install", []) if isinstance(pip_report, dict) else []:
    metadata = item.get("metadata") if isinstance(item, dict) else {}
    if isinstance(metadata, dict):
        resolved.append(
            {
                "name": metadata.get("name"),
                "version": metadata.get("version"),
                "download_info": item.get("download_info"),
            }
        )
gptqmodel_resolved = []
for item in gptqmodel_pip_report.get("install", []) if isinstance(gptqmodel_pip_report, dict) else []:
    metadata = item.get("metadata") if isinstance(item, dict) else {}
    if isinstance(metadata, dict):
        gptqmodel_resolved.append(
            {
                "name": metadata.get("name"),
                "version": metadata.get("version"),
                "download_info": item.get("download_info"),
            }
        )

def _version_tuple(value):
    parts = []
    for chunk in str(value).replace("+", ".").split("."):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            break
    return tuple(parts)

def _installed_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None

gptqmodel_requirements = {
    "python": ">=3.10",
    "torch": ">=2.7.1",
    "numpy": ">=2.2.6",
    "transformers": ">=4.55.2",
}
gptqmodel_requirement_status = {
    "python": {
        "installed": ".".join(str(item) for item in sys.version_info[:3]),
        "required": gptqmodel_requirements["python"],
        "ok": sys.version_info >= (3, 10),
    }
}
for package, required_version in (("torch", "2.7.1"), ("numpy", "2.2.6"), ("transformers", "4.55.2")):
    installed = _installed_version(package)
    gptqmodel_requirement_status[package] = {
        "installed": installed,
        "required": f">={required_version}",
        "ok": installed is not None and _version_tuple(installed) >= _version_tuple(required_version),
    }
gptqmodel_requirements_ok = all(item["ok"] for item in gptqmodel_requirement_status.values())
gptqmodel_requested = os.environ.get("INSTALL_GPTQMODEL") == "1"

pip_version = subprocess.run(
    [sys.executable, "-m", "pip", "--version"],
    text=True,
    capture_output=True,
    check=False,
).stdout.strip()
return_code = int(os.environ["PIP_RETURN_CODE"])
gptqmodel_return_code = int(os.environ["GPTQMODEL_PIP_RETURN_CODE"])
gptqmodel_gate_ok = (
    not gptqmodel_requested
    or (gptqmodel_return_code == 0 and (sys.version_info < (3, 10) or gptqmodel_requirements_ok))
)
summary = {
    "ok": return_code == 0 and gptqmodel_gate_ok,
    "mode": "dry_run",
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "pip_version": pip_version,
    "optional_requirements": os.environ.get("OPTIONAL_REQUIREMENTS"),
    "requested_packages": requested_packages,
    "resolved_packages": resolved,
    "gptqmodel_install": {
        "requested": gptqmodel_requested,
        "package": os.environ.get("GPTQMODEL_PACKAGE"),
        "uses_no_build_isolation": True,
        "requires_python_3_10": True,
        "requirements": gptqmodel_requirements,
        "requirement_status": gptqmodel_requirement_status,
        "requirements_ok": gptqmodel_requirements_ok,
        "pip_returncode": gptqmodel_return_code,
        "resolved_packages": gptqmodel_resolved,
        "pip_stdout_tail": gptqmodel_stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:],
        "pip_stderr_tail": gptqmodel_stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:],
    },
    "current_imports": imports,
    "pip_returncode": return_code,
    "pip_stdout_tail": stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:],
    "pip_stderr_tail": stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:],
}
text = json.dumps(summary, ensure_ascii=False, indent=2)
output = os.environ.get("REPORT_OUTPUT")
if output:
    Path(output).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(output).expanduser().write_text(text + "\n", encoding="utf-8")
print(text)
raise SystemExit(0 if return_code == 0 and gptqmodel_gate_ok else return_code or gptqmodel_return_code or 1)
PY
  exit "$?"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel packaging
if [[ "${#INSTALL_ARGS[@]}" != "0" ]]; then
  "${PYTHON_BIN}" -m pip install "${INSTALL_ARGS[@]}"
fi
if [[ "${INSTALL_GPTQMODEL}" == "1" ]]; then
  if "${PYTHON_BIN}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
  then
    "${PYTHON_BIN}" -m pip install --no-build-isolation "${GPTQMODEL_PACKAGE}"
  else
    echo "Skipping ${GPTQMODEL_PACKAGE}: target Python 3.10+ dependency."
  fi
fi

INSTALL_BITSANDBYTES="${INSTALL_BITSANDBYTES}" INSTALL_OPTIMUM="${INSTALL_OPTIMUM}" INSTALL_GPTQ="${INSTALL_GPTQ}" INSTALL_GPTQMODEL="${INSTALL_GPTQMODEL}" INSTALL_AWQ="${INSTALL_AWQ}" INSTALL_COMPRESSED_TENSORS="${INSTALL_COMPRESSED_TENSORS}" OPTIMUM_PACKAGE="${OPTIMUM_PACKAGE}" REPORT_OUTPUT="${REPORT_OUTPUT}" "${PYTHON_BIN}" - <<'PY'
import importlib
import json
import os
import sys
from pathlib import Path

try:
    from edge_benchmark.runtime_compat import apply_runtime_compatibility_patches
    apply_runtime_compatibility_patches()
except Exception:
    pass

modules = []
if os.environ.get("INSTALL_BITSANDBYTES") == "1":
    modules.append("bitsandbytes")
if os.environ.get("INSTALL_OPTIMUM") == "1":
    modules.append("optimum")
if os.environ.get("INSTALL_GPTQ") == "1":
    modules.append("auto_gptq")
if os.environ.get("INSTALL_GPTQMODEL") == "1" and sys.version_info >= (3, 10):
    modules.append("gptqmodel")
if os.environ.get("INSTALL_AWQ") == "1":
    modules.append("awq")
if os.environ.get("INSTALL_COMPRESSED_TENSORS") == "1":
    modules.append("compressed_tensors")
report = {}
for name in modules:
    try:
        module = importlib.import_module(name)
        report[name] = {"ok": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        report[name] = {"ok": False, "error": str(exc)}
if os.environ.get("INSTALL_GPTQMODEL") == "1" and sys.version_info < (3, 10):
    report["gptqmodel"] = {
        "ok": True,
        "skipped": True,
        "reason": "gptqmodel is a target Python 3.10+ runtime dependency",
    }
text = json.dumps(report, indent=2)
output = os.environ.get("REPORT_OUTPUT")
if output:
    Path(output).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(output).expanduser().write_text(text + "\n", encoding="utf-8")
print(text)
raise SystemExit(0 if all(item["ok"] for item in report.values()) else 1)
PY

cat <<'EOF'

External adapters are configured separately because SmoothQuant/OmniQuant
layouts differ across checkouts. Set one or both before running the matrix:

export LLMCBENCH_SMOOTHQUANT_CMD='python /path/to/SmoothQuant/export.py --model {model_path} --output {output_dir} --calib {calibration_file} --seq-len {seqlen}'
export LLMCBENCH_OMNIQUANT_CMD='python /path/to/OmniQuant/export.py --model {model_path} --output {output_dir} --calib {calibration_file} --wbits {quant_bits} --group-size {group_size}'

Then verify:

python tools/check_method_readiness.py --config configs/edge_4090_matrix.json
EOF
