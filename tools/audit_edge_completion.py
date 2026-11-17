import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS
except ModuleNotFoundError:
    from prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS


TARGET_SCORE_FIELDS = [
    "mmlu_score",
    "arc_challenge_score",
    "hellaswag_score",
    "piqa_score",
    "winogrande_score",
]

TASK_METRIC_PREFERENCES = {
    "arc_challenge": ("acc_norm,none", "acc,none"),
    "hellaswag": ("acc_norm,none", "acc,none"),
    "piqa": ("acc_norm,none", "acc,none"),
}


def _split_tasks(tasks: str) -> List[str]:
    return [task.strip() for task in tasks.split(",") if task.strip()]


def _expand_prewarm_tasks(tasks: List[str]) -> List[str]:
    expanded = []
    for task in tasks:
        if task == "mmlu":
            expanded.extend(DEFAULT_MMLU_SUBTASKS)
        else:
            expanded.append(task)
    return expanded


def _load_json(path: Path, parse_errors: Optional[List[Dict]] = None, label: str = ""):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if parse_errors is not None:
            parse_errors.append(
                {
                    "artifact": label or path.name,
                    "path": str(path),
                    "error": "invalid JSON",
                    "message": str(exc),
                }
            )
            return None
        raise


def _sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_section_status(
    path: Path,
    results_path: Optional[Path] = None,
    matrix_results=None,
    prewarm_report_path: Optional[Path] = None,
    asset_audit_report_path: Optional[Path] = None,
    preflight_report_path: Optional[Path] = None,
    prepare_report_path: Optional[Path] = None,
    prepare_metadata_report_path: Optional[Path] = None,
    prepare_precompressed_report_path: Optional[Path] = None,
) -> Dict:
    required_sections = [
        "## Accuracy",
        "## Accuracy Metric Keys",
        "## Accuracy Settings",
        "## Deployment",
        "## Compression",
        "## Runtime",
        "## Resources",
    ]
    if prewarm_report_path is not None and prewarm_report_path.exists():
        required_sections.append("## Prewarm")
    if (
        asset_audit_report_path is not None
        and asset_audit_report_path.exists()
        or preflight_report_path is not None
        and preflight_report_path.exists()
    ):
        required_sections.append("## Preflight Checks")
    asset_audit_payload = _load_json(asset_audit_report_path) if asset_audit_report_path is not None else None
    if isinstance(asset_audit_payload, dict) and isinstance(asset_audit_payload.get("paths"), dict):
        required_sections.append("## Asset Paths")
    if (
        prepare_report_path is not None
        and prepare_report_path.exists()
        or prepare_metadata_report_path is not None
        and prepare_metadata_report_path.exists()
        or prepare_precompressed_report_path is not None
        and prepare_precompressed_report_path.exists()
    ):
        required_sections.append("## Asset Preparation")
    if not path.exists():
        return {
            "ok": False,
            "required_sections": required_sections,
            "missing_sections": required_sections,
            "metadata_mismatches": [],
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    missing_sections = [section for section in required_sections if section not in text]
    metadata_mismatches = []
    if results_path is not None:
        expected_source = f"- Source results: `{results_path}`"
        if expected_source not in text:
            metadata_mismatches.append(
                {
                    "field": "source_results",
                    "expected": expected_source,
                }
            )
    if isinstance(matrix_results, list):
        expected_rows = f"- Rows: {len(matrix_results)}"
        if expected_rows not in text:
            metadata_mismatches.append(
                {
                    "field": "rows",
                    "expected": expected_rows,
                }
            )
    return {
        "ok": not missing_sections and not metadata_mismatches,
        "required_sections": required_sections,
        "missing_sections": missing_sections,
        "metadata_mismatches": metadata_mismatches,
    }


def _csv_status(path: Path, matrix_results) -> Dict:
    if not path.exists():
        return {"ok": False, "error": "csv file not found", "header": [], "data_rows": None}
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            csv_pairs = []
            csv_rows_by_pair = {}
            data_rows = 0
            for row in reader:
                data_rows += 1
                pair = (row.get("model_name"), row.get("compression_method"))
                csv_pairs.append(pair)
                csv_rows_by_pair[pair] = row
    except Exception as exc:
        return {
            "ok": False,
            "error": "csv read failed",
            "message": str(exc),
            "header": [],
            "data_rows": None,
        }
    required_columns = ["model_name", "compression_method"]
    json_columns = sorted(
        {
            key
            for row in matrix_results
            if isinstance(row, dict)
            for key in row.keys()
        }
    ) if isinstance(matrix_results, list) else []
    required_columns = sorted(set(required_columns) | set(json_columns))
    missing_columns = [column for column in required_columns if column not in header]
    expected_rows = len(matrix_results) if isinstance(matrix_results, list) else None
    row_count_matches = expected_rows is not None and data_rows == expected_rows
    json_pairs = [
        (row.get("model_name"), row.get("compression_method"))
        for row in matrix_results
        if isinstance(row, dict)
    ] if isinstance(matrix_results, list) else []
    pair_set_matches = sorted(csv_pairs) == sorted(json_pairs) if not missing_columns else False
    value_mismatches = []
    if isinstance(matrix_results, list) and not missing_columns:
        for row in matrix_results:
            if not isinstance(row, dict):
                continue
            pair = (row.get("model_name"), row.get("compression_method"))
            csv_row = csv_rows_by_pair.get(pair)
            if not isinstance(csv_row, dict):
                continue
            for column in json_columns:
                json_value = row.get(column)
                csv_value = csv_row.get(column)
                expected_value = "" if json_value is None else str(json_value)
                if csv_value != expected_value:
                    value_mismatches.append(
                        {
                            "model": pair[0],
                            "method": pair[1],
                            "column": column,
                            "csv_value": csv_value,
                            "json_value": json_value,
                            "expected_csv_value": expected_value,
                        }
                    )
    return {
        "ok": not missing_columns and row_count_matches and pair_set_matches and not value_mismatches,
        "header": header,
        "missing_columns": missing_columns,
        "expected_columns": required_columns,
        "data_rows": data_rows,
        "expected_rows": expected_rows,
        "row_count_matches": row_count_matches,
        "pair_set_matches": pair_set_matches,
        "value_mismatches": value_mismatches,
        "csv_pairs": [{"model": model, "method": method} for model, method in sorted(csv_pairs)],
        "json_pairs": [{"model": model, "method": method} for model, method in sorted(json_pairs)],
    }


def _run_validator(args: argparse.Namespace, config_path: str, results_path: Path, output_path: Path) -> Dict:
    cmd = [
        sys.executable,
        "tools/validate_edge_results.py",
        "--config",
        config_path,
        "--results",
        str(results_path),
        "--output",
        str(output_path),
    ]
    for model_name in args.only_model:
        cmd.extend(["--only-model", model_name])
    for method in args.only_method:
        cmd.extend(["--only-method", method])
    if args.allow_unexpected:
        cmd.append("--allow-unexpected")
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    payload = _load_json(output_path)
    payload_ok = isinstance(payload, dict) and payload.get("ok") is True
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0 and payload_ok,
        "payload_ok": payload_ok,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "payload": payload,
    }


def _run_paper_protocol_validator(
    args: argparse.Namespace,
    config_path: str,
    results_path: Path,
    output_path: Path,
) -> Dict:
    cmd = [
        sys.executable,
        "tools/validate_paper_protocol.py",
        "--config",
        config_path,
        "--results",
        str(results_path),
        "--output",
        str(output_path),
        "--require-official-checkouts",
        "--require-base-models",
    ]
    for model_name in args.only_model:
        cmd.extend(["--only-model", model_name])
    for method in args.only_method:
        cmd.extend(["--only-method", method])
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    payload = _load_json(output_path)
    payload_ok = isinstance(payload, dict) and payload.get("ok") is True
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0 and payload_ok,
        "payload_ok": payload_ok,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "payload": payload,
    }


def _render_report_status(results_path: Path, run_report_path: Path, edge_report_path: Path, root: Path) -> Dict:
    expected_path = root / ".completion_expected_edge_report.md"
    cmd = [
        sys.executable,
        "tools/render_edge_report.py",
        "--results",
        str(results_path),
        "--output",
        str(expected_path),
    ]
    if run_report_path.exists():
        cmd.extend(["--run-report", str(run_report_path)])
    asset_audit_report_path = root / "edge_asset_audit.json"
    if asset_audit_report_path.exists():
        cmd.extend(["--asset-audit-report", str(asset_audit_report_path)])
    preflight_report_path = root / "edge_preflight_report.json"
    if preflight_report_path.exists():
        cmd.extend(["--preflight-report", str(preflight_report_path)])
    prewarm_report_path = root / "edge_prewarm_report.json"
    if prewarm_report_path.exists():
        cmd.extend(["--prewarm-report", str(prewarm_report_path)])
    prepare_report_path = root / "edge_prepare_assets_report.json"
    if prepare_report_path.exists():
        cmd.extend(["--prepare-report", str(prepare_report_path)])
    prepare_metadata_report_path = root / "edge_prepare_metadata_report.json"
    if prepare_metadata_report_path.exists():
        cmd.extend(["--prepare-metadata-report", str(prepare_metadata_report_path)])
    prepare_precompressed_report_path = root / "edge_prepare_precompressed_report.json"
    if prepare_precompressed_report_path.exists():
        cmd.extend(["--prepare-precompressed-report", str(prepare_precompressed_report_path)])
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    expected_sha256 = _sha256(expected_path)
    actual_sha256 = _sha256(edge_report_path)
    matches = bool(expected_sha256 and actual_sha256 and expected_sha256 == actual_sha256)
    try:
        if expected_path.exists():
            expected_path.unlink()
    except OSError:
        pass
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0 and matches,
        "matches_rendered_report": matches,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _git_value(args: List[str]) -> str:
    proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git_dirty() -> Optional[bool]:
    proc = subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=False)
    return bool(proc.stdout.strip()) if proc.returncode == 0 else None


def _code_status(resolved_config) -> Dict:
    resolved_code = resolved_config.get("code") if isinstance(resolved_config, dict) else {}
    if not isinstance(resolved_code, dict):
        resolved_code = {}
    current = {
        "commit": _git_value(["rev-parse", "HEAD"]),
        "branch": _git_value(["branch", "--show-current"]),
        "origin": _git_value(["remote", "get-url", "origin"]),
        "dirty": _git_dirty(),
    }
    comparisons = {
        key: bool(resolved_code.get(key) and current.get(key) and resolved_code.get(key) == current.get(key))
        for key in ("commit", "branch", "origin")
    }
    comparisons["dirty"] = (
        isinstance(resolved_code.get("dirty"), bool)
        and isinstance(current.get("dirty"), bool)
        and resolved_code.get("dirty") == current.get("dirty")
    )
    return {
        "resolved": resolved_code,
        "current": current,
        "matches_current": comparisons,
        "all_present": all(resolved_code.get(key) for key in ("commit", "branch", "origin")),
        "all_match_current": all(comparisons[key] for key in ("commit", "branch", "origin")),
    }


def _expected_pairs(config: Dict, only_model: List[str], only_method: List[str]) -> List[Dict]:
    wanted_models = set(only_model or [])
    wanted_methods = set(only_method or [])
    models = [model for model in config["models"] if not wanted_models or model["name"] in wanted_models]
    methods = [method for method in config["methods"] if not wanted_methods or method in wanted_methods]
    return [{"model": model["name"], "method": method} for model in models for method in methods]


def _selection_status(source_config, resolved_config, config_path: str, root: Path, only_model: List[str], only_method: List[str]) -> Dict:
    if not isinstance(source_config, dict):
        return {"ok": False, "required": True, "violations": [{"error": "source config must be a JSON object"}]}
    if not isinstance(resolved_config, dict):
        return {"ok": True, "required": False, "violations": []}
    wanted_models = set(only_model or [])
    wanted_methods = set(only_method or [])
    expected_models = [
        model.get("name")
        for model in source_config.get("models", [])
        if isinstance(model, dict) and (not wanted_models or model.get("name") in wanted_models)
    ]
    expected_model_entries = [
        model
        for model in source_config.get("models", [])
        if isinstance(model, dict) and (not wanted_models or model.get("name") in wanted_models)
    ]
    expected_methods = [
        method
        for method in source_config.get("methods", [])
        if not wanted_methods or method in wanted_methods
    ]
    resolved_models = [
        model.get("name")
        for model in resolved_config.get("models", [])
        if isinstance(model, dict)
    ] if isinstance(resolved_config.get("models"), list) else []
    resolved_model_entries = [
        model
        for model in resolved_config.get("models", [])
        if isinstance(model, dict)
    ] if isinstance(resolved_config.get("models"), list) else []
    resolved_selected_models = resolved_config.get("selected_models")
    resolved_methods = resolved_config.get("methods") if isinstance(resolved_config.get("methods"), list) else []
    resolved_selected_methods = resolved_config.get("selected_methods")
    checks = {
        "source_config": (resolved_config.get("source_config"), config_path),
        "model_entries": (resolved_model_entries, expected_model_entries),
        "models": (resolved_models, expected_models),
        "selected_models": (resolved_selected_models, expected_models),
        "methods": (resolved_methods, expected_methods),
        "selected_methods": (resolved_selected_methods, expected_methods),
    }
    violations = []
    for field, (actual, expected) in checks.items():
        if actual != expected:
            violations.append({"field": field, "expected": expected, "actual": actual})
    for field in (
        "external_commands",
        "precompressed_paths",
        "method_overrides",
        "model_overrides",
        "model_method_overrides",
    ):
        expected_value = source_config.get(field, {})
        actual_value = resolved_config.get(field, {})
        if actual_value != expected_value:
            violations.append({"field": field, "expected": expected_value, "actual": actual_value})
    source_defaults = source_config.get("defaults")
    resolved_defaults = resolved_config.get("defaults")
    ignored_defaults = {"cuda_visible_devices"}
    if isinstance(source_defaults, dict) and isinstance(resolved_defaults, dict):
        for key in sorted(set(source_defaults) | set(resolved_defaults)):
            if key in ignored_defaults:
                continue
            expected_value = source_defaults.get(key)
            if key == "output_root":
                expected_value = str(root)
            actual_value = resolved_defaults.get(key)
            if actual_value != expected_value:
                violations.append(
                    {
                        "field": f"defaults.{key}",
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
    else:
        violations.append(
            {
                "field": "defaults",
                "expected_type": "object",
                "source_type": type(source_defaults).__name__,
                "resolved_type": type(resolved_defaults).__name__,
            }
        )
    return {
        "ok": not violations,
        "required": True,
        "expected_models": expected_models,
        "expected_methods": expected_methods,
        "violations": violations,
    }


def _runtime_violations(
    rows,
    python_prefix: str,
    gpu_name_contains: str,
    cuda_prefix: str,
    cuda_visible_devices: str,
    gpu_total_memory_min_gb: float,
    gpu_count: int,
    require_cudnn: bool,
    torch_prefix: str,
    require_transformers: bool,
    require_bitsandbytes: bool,
    cpu_count_min: int,
    system_memory_min_gb: float,
    hf_endpoint: str,
) -> List[Dict]:
    violations = []
    if not isinstance(rows, list):
        return violations
    for row in rows:
        identity = {"model": row.get("model_name"), "method": row.get("compression_method")}
        runtime_python = row.get("runtime_python")
        runtime_cpu_count = row.get("runtime_cpu_count")
        runtime_system_memory_total_gb = row.get("runtime_system_memory_total_gb")
        runtime_hf_endpoint = row.get("runtime_hf_endpoint")
        runtime_torch = row.get("runtime_torch")
        runtime_transformers = row.get("runtime_transformers")
        runtime_bitsandbytes_available = row.get("runtime_bitsandbytes_available")
        runtime_bitsandbytes_version = row.get("runtime_bitsandbytes_version")
        runtime_bitsandbytes_error = row.get("runtime_bitsandbytes_error")
        runtime_gpu_name = row.get("runtime_gpu_name")
        runtime_gpu_count = row.get("runtime_gpu_count")
        runtime_cuda_visible_devices = row.get("runtime_cuda_visible_devices")
        runtime_cuda_available = row.get("runtime_cuda_available")
        runtime_cuda_version = row.get("runtime_cuda_version")
        runtime_cudnn_version = row.get("runtime_cudnn_version")
        runtime_gpu_total_memory_gb = row.get("runtime_gpu_total_memory_gb")
        if python_prefix and (not isinstance(runtime_python, str) or not runtime_python.startswith(python_prefix)):
            violations.append({**identity, "field": "runtime_python", "value": runtime_python, "required_prefix": python_prefix})
        if torch_prefix and (not isinstance(runtime_torch, str) or not runtime_torch.startswith(torch_prefix)):
            violations.append({**identity, "field": "runtime_torch", "value": runtime_torch, "required_prefix": torch_prefix})
        if require_transformers and not (isinstance(runtime_transformers, str) and runtime_transformers):
            violations.append(
                {**identity, "field": "runtime_transformers", "value": runtime_transformers, "required": "non-empty transformers version"}
            )
        if require_bitsandbytes and runtime_bitsandbytes_available is not True:
            violations.append(
                {
                    **identity,
                    "field": "runtime_bitsandbytes_available",
                    "value": runtime_bitsandbytes_available,
                    "version": runtime_bitsandbytes_version,
                    "error": runtime_bitsandbytes_error,
                    "required": True,
                }
            )
        if cpu_count_min > 0 and (
            not isinstance(runtime_cpu_count, (int, float))
            or isinstance(runtime_cpu_count, bool)
            or runtime_cpu_count < cpu_count_min
        ):
            violations.append(
                {**identity, "field": "runtime_cpu_count", "value": runtime_cpu_count, "required_min": cpu_count_min}
            )
        if system_memory_min_gb > 0 and (
            not isinstance(runtime_system_memory_total_gb, (int, float))
            or isinstance(runtime_system_memory_total_gb, bool)
            or runtime_system_memory_total_gb < system_memory_min_gb
        ):
            violations.append(
                {
                    **identity,
                    "field": "runtime_system_memory_total_gb",
                    "value": runtime_system_memory_total_gb,
                    "required_min_gb": system_memory_min_gb,
                }
            )
        if hf_endpoint and runtime_hf_endpoint != hf_endpoint:
            violations.append(
                {**identity, "field": "runtime_hf_endpoint", "value": runtime_hf_endpoint, "required": hf_endpoint}
            )
        if (
            (gpu_name_contains or cuda_prefix or gpu_total_memory_min_gb > 0 or gpu_count > 0 or require_cudnn)
            and runtime_cuda_available is not True
        ):
            violations.append(
                {**identity, "field": "runtime_cuda_available", "value": runtime_cuda_available, "required": True}
            )
        if gpu_name_contains and (
            not isinstance(runtime_gpu_name, str) or gpu_name_contains.lower() not in runtime_gpu_name.lower()
        ):
            violations.append(
                {**identity, "field": "runtime_gpu_name", "value": runtime_gpu_name, "required_contains": gpu_name_contains}
            )
        if gpu_count > 0 and (
            not isinstance(runtime_gpu_count, (int, float))
            or isinstance(runtime_gpu_count, bool)
            or int(runtime_gpu_count) != gpu_count
        ):
            violations.append(
                {**identity, "field": "runtime_gpu_count", "value": runtime_gpu_count, "required": gpu_count}
            )
        if cuda_visible_devices and runtime_cuda_visible_devices != cuda_visible_devices:
            violations.append(
                {
                    **identity,
                    "field": "runtime_cuda_visible_devices",
                    "value": runtime_cuda_visible_devices,
                    "required": cuda_visible_devices,
                }
            )
        if cuda_prefix and (not isinstance(runtime_cuda_version, str) or not runtime_cuda_version.startswith(cuda_prefix)):
            violations.append(
                {**identity, "field": "runtime_cuda_version", "value": runtime_cuda_version, "required_prefix": cuda_prefix}
            )
        if gpu_total_memory_min_gb > 0 and (
            not isinstance(runtime_gpu_total_memory_gb, (int, float))
            or isinstance(runtime_gpu_total_memory_gb, bool)
            or runtime_gpu_total_memory_gb < gpu_total_memory_min_gb
        ):
            violations.append(
                {
                    **identity,
                    "field": "runtime_gpu_total_memory_gb",
                    "value": runtime_gpu_total_memory_gb,
                    "required_min_gb": gpu_total_memory_min_gb,
                }
            )
        if require_cudnn and not (isinstance(runtime_cudnn_version, str) and runtime_cudnn_version):
            violations.append(
                {**identity, "field": "runtime_cudnn_version", "value": runtime_cudnn_version, "required": "non-empty cuDNN version"}
            )
    return violations


def _resolve_artifact_path(root: Path, value) -> Optional[Path]:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _matrix_rows_by_pair(matrix_results) -> Dict:
    if not isinstance(matrix_results, list):
        return {}
    return {
        (row.get("model_name"), row.get("compression_method")): row
        for row in matrix_results
        if isinstance(row, dict)
    }


def _mmlu_subtask_coverage_status(matrix_results, expected_pairs: List[Dict], config: Dict) -> Dict:
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    require_mmlu = "mmlu" in set(_split_tasks(defaults.get("tasks", "")))
    if not require_mmlu:
        return {
            "ok": True,
            "required": False,
            "required_subtask_count": 0,
            "required_subtasks": [],
            "rows": [],
            "violations": [],
        }
    rows_by_pair = _matrix_rows_by_pair(matrix_results)
    row_status = []
    violations = []
    for item in expected_pairs:
        pair = (item.get("model"), item.get("method"))
        row = rows_by_pair.get(pair)
        if not isinstance(row, dict):
            continue
        present = [
            task
            for task in DEFAULT_MMLU_SUBTASKS
            if isinstance(row.get(f"{task}_score"), (int, float)) and not isinstance(row.get(f"{task}_score"), bool)
        ]
        missing = [task for task in DEFAULT_MMLU_SUBTASKS if task not in present]
        status = {
            "model": pair[0],
            "method": pair[1],
            "present_count": len(present),
            "required_count": len(DEFAULT_MMLU_SUBTASKS),
            "complete": not missing,
            "missing_subtasks": missing,
        }
        row_status.append(status)
        if missing:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "present_count": len(present),
                    "missing_count": len(missing),
                    "missing_subtasks": missing,
                    "error": "matrix_results row is missing required MMLU subtask score fields",
                }
            )
    return {
        "ok": not violations,
        "required": True,
        "required_subtask_count": len(DEFAULT_MMLU_SUBTASKS),
        "required_subtasks": list(DEFAULT_MMLU_SUBTASKS),
        "rows": row_status,
        "violations": violations,
    }


def _nested_get(payload: Dict, path: List[str]):
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _artifact_value_violations(pair, matrix_row: Dict, artifact_path: Path, artifact_name: str, field_map: Dict[str, List[str]]) -> List[Dict]:
    if not artifact_path.exists():
        return []
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            {
                "model": pair[0],
                "method": pair[1],
                "artifact": artifact_name,
                "path": str(artifact_path),
                "error": "invalid JSON",
                "message": str(exc),
            }
        ]
    if not isinstance(artifact, dict):
        return [
            {
                "model": pair[0],
                "method": pair[1],
                "artifact": artifact_name,
                "path": str(artifact_path),
                "error": "artifact must be a JSON object",
                "actual_type": type(artifact).__name__,
            }
        ]
    violations = []
    for matrix_field, artifact_field_path in field_map.items():
        matrix_value = matrix_row.get(matrix_field)
        artifact_value = _nested_get(artifact, artifact_field_path)
        if matrix_value != artifact_value:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": matrix_field,
                    "artifact": artifact_name,
                    "artifact_field_path": ".".join(artifact_field_path),
                    "path": str(artifact_path),
                    "matrix_results_value": matrix_value,
                    "artifact_value": artifact_value,
                    "error": "matrix_results row does not match source artifact",
                }
            )
    return violations


def _runtime_artifact_violations(pair, matrix_row: Dict, artifact_path: Path, artifact_name: str) -> List[Dict]:
    if not artifact_path.exists():
        return []
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            {
                "model": pair[0],
                "method": pair[1],
                "artifact": artifact_name,
                "path": str(artifact_path),
                "error": "invalid JSON",
                "message": str(exc),
            }
        ]
    if not isinstance(artifact, dict):
        return []
    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        return []
    gpus = runtime.get("gpus") if isinstance(runtime.get("gpus"), list) else []
    first_gpu = gpus[0] if gpus and isinstance(gpus[0], dict) else {}
    optional_imports = runtime.get("optional_imports") if isinstance(runtime.get("optional_imports"), dict) else {}
    bitsandbytes = optional_imports.get("bitsandbytes") if isinstance(optional_imports.get("bitsandbytes"), dict) else {}
    expected = {
        "runtime_python": runtime.get("python"),
        "runtime_cpu_count": runtime.get("cpu_count"),
        "runtime_system_memory_total_gb": runtime.get("system_memory_total_gb"),
        "runtime_hf_endpoint": runtime.get("hf_endpoint"),
        "runtime_hf_home": runtime.get("hf_home"),
        "runtime_transformers_cache": runtime.get("transformers_cache"),
        "runtime_huggingface_hub_cache": runtime.get("huggingface_hub_cache"),
        "runtime_hf_datasets_cache": runtime.get("hf_datasets_cache"),
        "runtime_torch": runtime.get("torch"),
        "runtime_transformers": runtime.get("transformers"),
        "runtime_cuda_available": runtime.get("cuda_available"),
        "runtime_cuda_version": runtime.get("cuda_version"),
        "runtime_cudnn_version": runtime.get("cudnn_version"),
        "runtime_cuda_visible_devices": runtime.get("cuda_visible_devices"),
        "runtime_gpu_count": len(gpus),
        "runtime_gpu_name": first_gpu.get("name"),
        "runtime_gpu_total_memory_gb": first_gpu.get("total_memory_gb"),
        "runtime_bitsandbytes_available": bitsandbytes.get("ok"),
        "runtime_bitsandbytes_version": bitsandbytes.get("version"),
        "runtime_bitsandbytes_error": bitsandbytes.get("error"),
    }
    violations = []
    for field, expected_value in expected.items():
        if expected_value is None:
            continue
        matrix_value = matrix_row.get(field)
        if field in {"runtime_transformers_cache", "runtime_huggingface_hub_cache"} and matrix_value in (None, ""):
            continue
        if matrix_value != expected_value:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "artifact": artifact_name,
                    "path": str(artifact_path),
                    "matrix_results_value": matrix_value,
                    "artifact_value": expected_value,
                    "error": "matrix_results row does not match source artifact runtime",
                }
            )
    return violations


def _best_accuracy_metric(metrics: Dict, task: str = ""):
    for key in TASK_METRIC_PREFERENCES.get(task, ("acc,none", "acc_norm,none", "exact_match,none")):
        if key in metrics:
            return metrics[key], key
    for key, value in metrics.items():
        if key.startswith("acc") and isinstance(value, (int, float)):
            return value, key
    return None, None


def _accuracy_value(accuracy: Dict, key: str):
    value = accuracy.get(key)
    if value not in (None, ""):
        return value
    task_outputs = accuracy.get("task_outputs")
    if not isinstance(task_outputs, dict):
        return value
    for path_value in task_outputs.values():
        if not isinstance(path_value, str) or not path_value:
            continue
        try:
            task_payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
        except Exception:
            continue
        task_value = task_payload.get(key) if isinstance(task_payload, dict) else None
        if task_value not in (None, ""):
            return task_value
    return value


def _accuracy_results_with_split_outputs(accuracy: Dict) -> Dict:
    merged = {}
    results = accuracy.get("results")
    if isinstance(results, dict):
        merged.update(results)
    task_outputs = accuracy.get("task_outputs")
    if not isinstance(task_outputs, dict):
        return merged
    for path_value in task_outputs.values():
        if not isinstance(path_value, str) or not path_value:
            continue
        try:
            task_payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
        except Exception:
            continue
        task_results = task_payload.get("results") if isinstance(task_payload, dict) else None
        if isinstance(task_results, dict):
            merged.update(task_results)
    return merged


def _expected_accuracy_fields(accuracy: Dict) -> Dict:
    expected = {}
    results = _accuracy_results_with_split_outputs(accuracy)
    if isinstance(results, dict):
        for task, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            value, metric_key = _best_accuracy_metric(metrics, task)
            expected[f"{task}_score"] = value
            if metric_key:
                expected[f"{task}_score_metric"] = metric_key
    values = []
    metric_keys = []
    for task in DEFAULT_MMLU_SUBTASKS:
        key = f"{task}_score"
        value = expected.get(key)
        if isinstance(value, (int, float)):
            values.append(value)
            metric_key = expected.get(f"{key}_metric")
            if isinstance(metric_key, str) and metric_key:
                metric_keys.append(metric_key)
    if values:
        expected["mmlu_score"] = sum(values) / len(values)
        if metric_keys:
            unique_metrics = sorted(set(metric_keys))
            expected["mmlu_score_metric"] = f"aggregate:{unique_metrics[0]}" if len(unique_metrics) == 1 else "aggregate:mixed"
    values = [expected.get(field) for field in TARGET_SCORE_FIELDS if isinstance(expected.get(field), (int, float))]
    if values:
        expected["accuracy_avg_score"] = sum(values) / len(values)
    failed_tasks = accuracy.get("failed_tasks")
    if not isinstance(failed_tasks, dict):
        failed_tasks = {}
    expected["accuracy_failed_task_count"] = len(failed_tasks)
    expected["accuracy_failed_tasks"] = ",".join(sorted(failed_tasks))
    return expected


def _accuracy_artifact_violations(pair, matrix_row: Dict, accuracy_path: Path) -> List[Dict]:
    if not accuracy_path.exists():
        return []
    try:
        accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            {
                "model": pair[0],
                "method": pair[1],
                "artifact": "accuracy.json",
                "path": str(accuracy_path),
                "error": "invalid JSON",
                "message": str(exc),
            }
        ]
    if not isinstance(accuracy, dict):
        return [
            {
                "model": pair[0],
                "method": pair[1],
                "artifact": "accuracy.json",
                "path": str(accuracy_path),
                "error": "artifact must be a JSON object",
                "actual_type": type(accuracy).__name__,
            }
        ]
    violations = []
    for field, expected_value in {
        "accuracy_dtype": accuracy.get("dtype"),
        "accuracy_lm_eval_dtype": accuracy.get("lm_eval_dtype"),
        "accuracy_batch_size": accuracy.get("batch_size"),
        "accuracy_num_fewshot": accuracy.get("num_fewshot"),
        "accuracy_limit": accuracy.get("limit"),
        "accuracy_device": accuracy.get("device"),
        "accuracy_data_dir": _accuracy_value(accuracy, "data_dir"),
        "accuracy_hf_datasets_cache": _accuracy_value(accuracy, "hf_datasets_cache"),
        "accuracy_max_memory": accuracy.get("max_memory"),
        "accuracy_attn_implementation": accuracy.get("attn_implementation"),
        "accuracy_split_tasks": accuracy.get("split_accuracy_tasks", False),
        "accuracy_continue_on_error": accuracy.get("continue_on_accuracy_error", False),
    }.items():
        matrix_value = matrix_row.get(field)
        if matrix_value != expected_value:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "artifact": "accuracy.json",
                    "path": str(accuracy_path),
                    "matrix_results_value": matrix_value,
                    "artifact_value": expected_value,
                    "error": "matrix_results row does not match source artifact",
                }
            )
    for field, expected_value in _expected_accuracy_fields(accuracy).items():
        matrix_value = matrix_row.get(field)
        if matrix_value != expected_value:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "artifact": "accuracy.json",
                    "path": str(accuracy_path),
                    "matrix_results_value": matrix_value,
                    "artifact_value": expected_value,
                    "error": "matrix_results row does not match source artifact",
                }
            )
    return violations


def _effective_defaults(config: Dict, model_name: str, method: str) -> Dict:
    ignored_keys = {"output_root", "cuda_visible_devices", "python_bin", "runtime_profile"}
    defaults = config.get("defaults", {})
    merged = {name: value for name, value in defaults.items() if name not in ignored_keys} if isinstance(defaults, dict) else {}
    for section_name, key in (
        ("method_overrides", method),
        ("model_overrides", model_name),
        ("model_method_overrides", f"{model_name}/{method}"),
    ):
        section = config.get(section_name, {})
        override = section.get(key) if isinstance(section, dict) else None
        if isinstance(override, dict):
            merged.update({name: value for name, value in override.items() if name not in ignored_keys})
    return merged


def _run_report_status(run_report, expected_pairs: List[Dict], root: Path, matrix_results=None, config=None, allow_unexpected: bool = False, allow_skipped_external: bool = False) -> Dict:
    allowed_statuses = {"completed", "skipped_existing"}
    if allow_skipped_external:
        allowed_statuses.add("skipped_external_not_configured")
    expected_pair_set = {(item["model"], item["method"]) for item in expected_pairs}
    matrix_rows = _matrix_rows_by_pair(matrix_results)
    job_pairs = {}
    missing_artifacts = []
    summary_violations = []
    result_violations = []
    list_violations = []
    duplicate_jobs = []
    override_violations = []
    report_violations = []
    expected_completed = []
    expected_skipped = []
    expected_failed = []
    expected_skipped_external = []
    unexpected_jobs = []
    if isinstance(run_report, dict) and run_report.get("dry_run") is True:
        report_violations.append({"field": "dry_run", "actual": True, "error": "dry_run_report_not_final"})
    if isinstance(run_report, dict) and isinstance(run_report.get("jobs"), list):
        for job in run_report["jobs"]:
            if not isinstance(job, dict):
                continue
            pair = (job.get("model"), job.get("method"))
            if pair not in expected_pair_set:
                unexpected_jobs.append({"model": pair[0], "method": pair[1], "status": job.get("status")})
                continue
            if pair in expected_pair_set:
                if pair in job_pairs:
                    duplicate_jobs.append({"model": pair[0], "method": pair[1]})
                job_pairs[pair] = job.get("status")
                summary_path = _resolve_artifact_path(root, job.get("summary"))
                log_path = _resolve_artifact_path(root, job.get("log"))
                summary_value = str(summary_path) if summary_path is not None else job.get("summary")
                if job.get("status") == "completed":
                    expected_completed.append(summary_value)
                elif job.get("status") == "skipped_existing":
                    expected_skipped.append(summary_value)
                elif job.get("status") == "failed":
                    expected_failed.append(
                        {
                            key: job.get(key)
                            for key in ("model", "method", "reason", "summary", "log", "returncode")
                            if key in job
                        }
                    )
                if isinstance(config, dict) and job.get("status") in {"completed", "skipped_existing"}:
                    expected_overrides = _effective_defaults(config, pair[0], pair[1])
                    raw_actual_overrides = job.get("job_overrides")
                    ignored_override_keys = {"output_root", "cuda_visible_devices", "python_bin", "runtime_profile"}
                    actual_overrides = (
                        {
                            name: value
                            for name, value in raw_actual_overrides.items()
                            if name not in ignored_override_keys
                        }
                        if isinstance(raw_actual_overrides, dict)
                        else raw_actual_overrides
                    )
                    if actual_overrides != expected_overrides:
                        override_violations.append(
                            {
                                "model": pair[0],
                                "method": pair[1],
                                "expected": expected_overrides,
                                "actual": actual_overrides,
                            }
                        )
                if job.get("status") == "skipped_external_not_configured":
                    expected_skipped_external.append(
                        {
                            key: job.get(key)
                            for key in ("model", "method", "reason")
                            if key in job
                        }
                    )
                if job.get("status") == "skipped_external_not_configured":
                    continue
                if summary_path is None or not summary_path.exists():
                    missing_artifacts.append(
                        {"model": pair[0], "method": pair[1], "field": "summary", "path": str(summary_path) if summary_path else None}
                    )
                elif not _is_under_root(summary_path, root):
                    missing_artifacts.append(
                        {"model": pair[0], "method": pair[1], "field": "summary", "path": str(summary_path), "error": "path outside output root"}
                    )
                else:
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        summary_violations.append(
                            {
                                "model": pair[0],
                                "method": pair[1],
                                "field": "summary",
                                "path": str(summary_path),
                                "error": "invalid JSON",
                                "message": str(exc),
                            }
                        )
                    else:
                        if not isinstance(summary, dict):
                            summary_violations.append(
                                {
                                    "model": pair[0],
                                    "method": pair[1],
                                    "field": "summary",
                                    "path": str(summary_path),
                                    "error": "summary must be a JSON object",
                                    "actual_type": type(summary).__name__,
                                }
                            )
                        else:
                            summary_model = summary.get("model_name")
                            summary_method = summary.get("compression_method")
                            if summary_model != pair[0] or summary_method != pair[1]:
                                summary_violations.append(
                                    {
                                        "model": pair[0],
                                        "method": pair[1],
                                        "field": "summary",
                                        "path": str(summary_path),
                                        "error": "summary identity does not match matrix job",
                                        "summary_model_name": summary_model,
                                        "summary_compression_method": summary_method,
                                    }
                                )
                            matrix_row = matrix_rows.get(pair)
                            if isinstance(matrix_row, dict):
                                matrix_path = matrix_row.get("compressed_model_path")
                                summary_compressed_path = summary.get("compressed_model_path")
                                if matrix_path != summary_compressed_path:
                                    result_violations.append(
                                        {
                                            "model": pair[0],
                                            "method": pair[1],
                                            "field": "compressed_model_path",
                                            "summary": str(summary_path),
                                            "matrix_results_value": matrix_path,
                                            "summary_value": summary_compressed_path,
                                            "error": "matrix_results row does not match job summary",
                                        }
                                    )
                                job_dir = summary_path.parent
                                result_violations.extend(
                                    _runtime_artifact_violations(pair, matrix_row, summary_path, "summary.json")
                                )
                                result_violations.extend(
                                    _artifact_value_violations(
                                        pair,
                                        matrix_row,
                                        job_dir / "compression.json",
                                        "compression.json",
                                        {
                                            "source_model_path": ["manifest", "source_model_path"],
                                            "model_revision": ["manifest", "revision"],
                                            "compression_base_model_repository": ["manifest", "base_model_repository"],
                                            "compression_base_model_revision": ["manifest", "base_model_revision"],
                                            "compression_base_model_attestation_path": ["manifest", "base_model_attestation_path"],
                                            "compression_base_model_attested": ["manifest", "base_model_attested"],
                                            "compression_backend": ["manifest", "backend"],
                                            "compression_quant_bits": ["manifest", "quant_bits"],
                                            "compression_group_size": ["manifest", "group_size"],
                                            "compression_target_model_adaptation": ["manifest", "target_model_adaptation"],
                                            "compression_target_model_adaptation_disclaimer": ["manifest", "target_model_adaptation_disclaimer"],
                                            "compression_runtime_loading": ["manifest", "runtime_loading"],
                                            "compression_storage_quantization": ["manifest", "storage_quantization"],
                                            "compression_adaptation_calibration_source": ["manifest", "adaptation_calibration_source"],
                                            "compression_adaptation_calibration_seed": ["manifest", "adaptation_calibration_seed"],
                                            "compression_adaptation_calibration_samples": ["manifest", "adaptation_calibration_samples"],
                                            "compression_adaptation_calibration_sequence_length": ["manifest", "adaptation_calibration_sequence_length"],
                                            "compression_adaptation_calibration_processed_batches": ["manifest", "adaptation_calibration_processed_batches"],
                                            "compression_desc_act": ["manifest", "desc_act"],
                                            "compression_awq_version": ["manifest", "awq_version"],
                                            "compression_sparsity": ["manifest", "sparsity"],
                                            "compression_prune_lm_head": ["manifest", "prune_lm_head"],
                                            "compression_pruning_algorithm": ["manifest", "pruning_algorithm"],
                                            "compression_pruning_buffer_device": ["manifest", "pruning_buffer_device"],
                                            "compression_sparsegpt_blocksize": ["manifest", "sparsegpt_blocksize"],
                                            "compression_sparsegpt_percdamp": ["manifest", "sparsegpt_percdamp"],
                                            "compression_calibration_samples": ["manifest", "calibration_samples"],
                                            "compression_calibration_file": ["manifest", "calibration_file"],
                                            "compression_calibration_batch_size": ["manifest", "calibration_batch_size"],
                                            "compression_calibration_protocol": ["manifest", "calibration_protocol"],
                                            "compression_calibration_source": ["manifest", "calibration_source"],
                                            "compression_calibration_sampling_algorithm": ["manifest", "calibration_sampling_algorithm"],
                                            "compression_calibration_seed": ["manifest", "calibration_seed"],
                                            "compression_calibration_dataset": ["manifest", "calibration_dataset"],
                                            "compression_calibration_data_file": ["manifest", "calibration_data_file"],
                                            "compression_calibration_dataset_revision": ["manifest", "calibration_dataset_revision"],
                                            "compression_calibration_token_length": ["manifest", "calibration_token_length"],
                                            "compression_calibration_token_sha256": ["manifest", "calibration_token_sha256"],
                                            "compression_paper_protocol": ["manifest", "paper_protocol"],
                                            "compression_paper_protocol_attested": ["manifest", "paper_protocol_attested"],
                                            "compression_gptq_damp_percent": ["manifest", "gptq_damp_percent"],
                                            "compression_gptq_static_groups": ["manifest", "gptq_static_groups"],
                                            "compression_gptq_symmetric": ["manifest", "gptq_symmetric"],
                                            "compression_gptq_true_sequential": ["manifest", "gptq_true_sequential"],
                                            "compression_awq_zero_point": ["manifest", "awq_zero_point"],
                                            "compression_awq_calibration_source": ["manifest", "awq_calibration_source"],
                                            "compression_awq_calibration_seed": ["manifest", "awq_calibration_seed"],
                                            "compression_awq_grid_size": ["manifest", "awq_grid_size"],
                                            "compression_awq_runtime_scale_search_grid": ["manifest", "awq_runtime_scale_search_grid"],
                                            "compression_awq_runtime_calibration_seed": ["manifest", "awq_runtime_calibration_seed"],
                                            "compression_quantization_protocol": ["manifest", "quantization_protocol"],
                                            "compression_smoothquant_alpha": ["manifest", "smoothquant_alpha"],
                                            "compression_smoothquant_alpha_selection": ["manifest", "smoothquant_alpha_selection"],
                                            "compression_official_code_dir": ["manifest", "official_implementation", "official_code_dir"],
                                            "compression_official_repository": ["manifest", "official_implementation", "official_repository"],
                                            "compression_official_commit": ["manifest", "official_implementation", "official_commit"],
                                            "compression_official_implementation_verified": ["manifest", "official_implementation", "official_implementation_verified"],
                                            "compression_omniquant_abits": ["manifest", "omniquant_abits"],
                                            "compression_omniquant_epochs": ["manifest", "omniquant_epochs"],
                                            "compression_omniquant_nsamples": ["manifest", "omniquant_nsamples"],
                                            "compression_omniquant_seqlen": ["manifest", "omniquant_seqlen"],
                                            "compression_omniquant_seed": ["manifest", "omniquant_seed"],
                                            "compression_omniquant_lwc": ["manifest", "omniquant_lwc"],
                                            "compression_omniquant_let": ["manifest", "omniquant_let"],
                                            "compression_omniquant_aug_loss": ["manifest", "omniquant_aug_loss"],
                                            "compression_dtype": ["manifest", "dtype"],
                                            "compression_seqlen": ["manifest", "seqlen"],
                                            "compression_max_memory": ["manifest", "max_memory"],
                                            "compression_attn_implementation": ["manifest", "attn_implementation"],
                                            "compression_materialized": ["manifest", "materialized"],
                                            "compression_note": ["manifest", "note"],
                                            "compression_external_command": ["manifest", "external_command"],
                                            "compression_external_returncode": ["manifest", "external_returncode"],
                                            "compression_stdout_tail": ["manifest", "stdout_tail"],
                                            "compression_stderr_tail": ["manifest", "stderr_tail"],
                                        },
                                    )
                                )
                                result_violations.extend(
                                    _runtime_artifact_violations(pair, matrix_row, job_dir / "compression.json", "compression.json")
                                )
                                result_violations.extend(
                                    _accuracy_artifact_violations(pair, matrix_row, job_dir / "accuracy.json")
                                )
                                result_violations.extend(
                                    _runtime_artifact_violations(pair, matrix_row, job_dir / "accuracy.json", "accuracy.json")
                                )
                                result_violations.extend(
                                    _artifact_value_violations(
                                        pair,
                                        matrix_row,
                                        job_dir / "deployment.json",
                                        "deployment.json",
                                        {
                                            "tokens_per_second": ["tokens_per_second"],
                                            "deployment_dtype": ["dtype"],
                                            "deployment_device_map": ["device_map"],
                                            "deployment_max_memory": ["max_memory"],
                                            "deployment_attn_implementation": ["attn_implementation"],
                                            "deployment_prompt_source": ["prompt_source"],
                                            "deployment_prompt_path": ["prompt_path"],
                                            "deployment_prompt_count": ["prompt_count"],
                                            "deployment_input_length": ["input_length"],
                                            "deployment_max_new_tokens": ["max_new_tokens"],
                                            "deployment_warmup": ["warmup"],
                                            "deployment_repeat": ["repeat"],
                                            "deployment_input_tokens": ["input_tokens"],
                                            "deployment_generated_tokens": ["generated_tokens"],
                                            "deployment_elapsed_seconds": ["elapsed_seconds"],
                                            "peak_memory_allocated_gb": ["peak_memory_allocated_gb"],
                                            "peak_memory_reserved_gb": ["peak_memory_reserved_gb"],
                                            "deployment_memory_limit_gb": ["deployment_memory_limit_gb"],
                                            "deployment_memory_measured_peak_gb": ["deployment_memory_measured_peak_gb"],
                                            "deployment_memory_limit_exceeded": ["deployment_memory_limit_exceeded"],
                                        },
                                    )
                                )
                                result_violations.extend(
                                    _runtime_artifact_violations(pair, matrix_row, job_dir / "deployment.json", "deployment.json")
                                )
                                result_violations.extend(
                                    _artifact_value_violations(
                                        pair,
                                        matrix_row,
                                        job_dir / "resource.json",
                                        "resource.json",
                                        {
                                            "resource_backend": ["resource_backend"],
                                            "resource_model_load_error": ["model_load_error"],
                                            "resource_seqlen": ["seqlen"],
                                            "resource_prompt_source": ["prompt_source"],
                                            "resource_prompt_path": ["prompt_path"],
                                            "resource_prompt_count": ["prompt_count"],
                                            "resource_input_length": ["input_length"],
                                            "resource_max_new_tokens": ["max_new_tokens"],
                                            "resource_sparsity_sample_tensors": ["sparsity_sample_tensors"],
                                            "storage_bytes": ["storage_bytes"],
                                            "storage_gb": ["storage_gb"],
                                            "effective_storage_bytes_sparse_approx": ["effective_storage_bytes_sparse_approx"],
                                            "effective_storage_gb_sparse_approx": ["effective_storage_gb_sparse_approx"],
                                            "total_params": ["parameter_stats", "total_params"],
                                            "sparsity": ["parameter_stats", "sparsity"],
                                            "nonzero_ratio": ["parameter_stats", "nonzero_ratio"],
                                            "flops": ["compute_complexity", "flops"],
                                            "macs": ["compute_complexity", "macs"],
                                            "flops_num": ["compute_complexity", "flops_num"],
                                            "effective_flops_num_sparse_approx": ["compute_complexity", "effective_flops_num_sparse_approx"],
                                            "compute_backend": ["compute_complexity", "backend"],
                                            "compute_warning": ["compute_complexity", "warning"],
                                        },
                                    )
                                )
                                result_violations.extend(
                                    _runtime_artifact_violations(pair, matrix_row, job_dir / "resource.json", "resource.json")
                                )
                if job.get("status") == "completed" and (log_path is None or not log_path.exists()):
                    missing_artifacts.append(
                        {"model": pair[0], "method": pair[1], "field": "log", "path": str(log_path) if log_path else None}
                    )
                elif job.get("status") == "completed" and not _is_under_root(log_path, root):
                    missing_artifacts.append(
                        {"model": pair[0], "method": pair[1], "field": "log", "path": str(log_path), "error": "path outside output root"}
                    )
    missing_jobs = [
        {"model": model, "method": method}
        for model, method in sorted(expected_pair_set - set(job_pairs))
    ]
    bad_statuses = [
        {"model": model, "method": method, "status": status}
        for (model, method), status in sorted(job_pairs.items())
        if status not in allowed_statuses
    ]
    top_level_checks = {
        "completed": sorted(expected_completed),
        "skipped": sorted(expected_skipped),
        "failed": sorted(expected_failed, key=lambda item: (item.get("model", ""), item.get("method", ""))),
        "skipped_external_not_configured": sorted(
            expected_skipped_external,
            key=lambda item: (item.get("model", ""), item.get("method", "")),
        ),
    }
    if isinstance(run_report, dict):
        for field, expected_value in top_level_checks.items():
            actual_value = run_report.get(field)
            normalized_actual = actual_value
            if field in {"completed", "skipped"} and isinstance(actual_value, list):
                normalized_actual = sorted(actual_value)
            elif field in {"failed", "skipped_external_not_configured"} and isinstance(actual_value, list):
                normalized_actual = sorted(
                    [
                        {
                            key: item.get(key)
                            for key in ("model", "method", "reason", "summary", "log", "returncode")
                            if isinstance(item, dict) and key in item
                        }
                        for item in actual_value
                        if isinstance(item, dict)
                    ],
                    key=lambda item: (item.get("model", ""), item.get("method", "")),
                )
            if not isinstance(normalized_actual, list):
                list_violations.append({"field": field, "expected": expected_value, "actual": normalized_actual})
                continue
            missing_items = [item for item in expected_value if item not in normalized_actual]
            if missing_items:
                list_violations.append(
                    {
                        "field": field,
                        "expected": expected_value,
                        "actual": normalized_actual,
                        "missing_items": missing_items,
                    }
                )
            extra_items = [item for item in normalized_actual if item not in expected_value]
            if extra_items and not allow_unexpected:
                list_violations.append(
                    {
                        "field": field,
                        "expected": expected_value,
                        "actual": normalized_actual,
                        "extra_items": extra_items,
                    }
                )
    return {
        "ok": not report_violations and not missing_jobs and not duplicate_jobs and not bad_statuses and not missing_artifacts and not summary_violations and not result_violations and not list_violations and not override_violations and (allow_unexpected or not unexpected_jobs),
        "allowed_statuses": sorted(allowed_statuses),
        "report_violations": report_violations,
        "missing_jobs": missing_jobs,
        "duplicate_jobs": duplicate_jobs,
        "unexpected_jobs": unexpected_jobs,
        "bad_statuses": bad_statuses,
        "missing_artifacts": missing_artifacts,
        "summary_violations": summary_violations,
        "result_violations": result_violations,
        "override_violations": override_violations,
        "list_violations": list_violations,
        "scoped_jobs": [
            {"model": model, "method": method, "status": status}
            for (model, method), status in sorted(job_pairs.items())
        ],
    }


def _finalize_report_status(
    finalize_report,
    root: Optional[Path] = None,
    require_prewarm_report: bool = False,
    require_asset_audit_report: bool = False,
    require_preflight_report: bool = False,
    require_handoff_manifest: bool = False,
    require_target_readiness_report: bool = False,
    require_target_paths_report: bool = False,
    require_prepare_report: bool = False,
    require_prepare_metadata_report: bool = False,
    require_prepare_precompressed_report: bool = False,
) -> Dict:
    expected_tools = [
        "tools/collect_edge_results.py",
        "tools/validate_edge_results.py",
        "tools/render_edge_report.py",
        "tools/audit_edge_completion.py",
    ]
    if require_target_readiness_report:
        expected_tools.append("tools/audit_target_readiness.py")
    if finalize_report is None:
        return {
            "ok": True,
            "required": False,
            "expected_tools": expected_tools,
            "missing_tools": [],
            "failed_steps": [],
            "path_violations": [],
        }
    if not isinstance(finalize_report, dict):
        return {
            "ok": False,
            "required": True,
            "expected_tools": expected_tools,
            "missing_tools": expected_tools,
            "failed_steps": [],
            "path_violations": [],
            "error": "finalize report must be a JSON object",
        }
    steps = finalize_report.get("steps")
    if not isinstance(steps, list):
        return {
            "ok": False,
            "required": True,
            "expected_tools": expected_tools,
            "missing_tools": expected_tools,
            "failed_steps": [],
            "path_violations": [],
            "error": "finalize report steps must be a list",
        }
    seen_tools = set()
    failed_steps = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            failed_steps.append({"index": index, "error": "step must be a JSON object"})
            continue
        cmd = step.get("cmd") if isinstance(step.get("cmd"), list) else []
        for tool in expected_tools:
            if tool in cmd:
                seen_tools.add(tool)
        if step.get("status") == "dry_run":
            failed_steps.append(
                {
                    "index": index,
                    "cmd": cmd,
                    "status": step.get("status"),
                    "error": "dry_run_step_not_final",
                }
            )
            continue
        if step.get("ok") is not True:
            failed_steps.append(
                {
                    "index": index,
                    "cmd": cmd,
                    "returncode": step.get("returncode"),
                    "ok": step.get("ok"),
                    "status": step.get("status"),
                }
            )
    missing_tools = [tool for tool in expected_tools if tool not in seen_tools]
    path_violations = []
    if root is not None:
        expected_paths = {
            "output_root": str(root),
            "results_json": str(root / "matrix_results.json"),
            "results_csv": str(root / "matrix_results.csv"),
            "edge_report": str(root / "edge_report.md"),
            "completion_audit": str(root / "completion_audit.json"),
            "resolved_config": str(root / "matrix_config_resolved.json"),
        }
        if require_prewarm_report or "edge_prewarm_report" in finalize_report:
            expected_paths["edge_prewarm_report"] = str(root / "edge_prewarm_report.json")
        if require_asset_audit_report or "edge_asset_audit" in finalize_report:
            expected_paths["edge_asset_audit"] = str(root / "edge_asset_audit.json")
        if require_preflight_report or "edge_preflight_report" in finalize_report:
            expected_paths["edge_preflight_report"] = str(root / "edge_preflight_report.json")
        if require_handoff_manifest or "edge_handoff_manifest" in finalize_report:
            expected_paths["edge_handoff_manifest"] = str(root / "edge_handoff_manifest.json")
        if require_target_readiness_report:
            expected_paths["target_readiness"] = str(root / "target_readiness.json")
        if require_target_paths_report or "edge_target_paths_report" in finalize_report:
            expected_paths["edge_target_paths_report"] = str(root / "edge_target_paths_report.json")
        if require_prepare_report or "edge_prepare_assets_report" in finalize_report:
            expected_paths["edge_prepare_assets_report"] = str(root / "edge_prepare_assets_report.json")
        if require_prepare_metadata_report or "edge_prepare_metadata_report" in finalize_report:
            expected_paths["edge_prepare_metadata_report"] = str(root / "edge_prepare_metadata_report.json")
        if require_prepare_precompressed_report or "edge_prepare_precompressed_report" in finalize_report:
            expected_paths["edge_prepare_precompressed_report"] = str(root / "edge_prepare_precompressed_report.json")
        for field, expected_value in expected_paths.items():
            actual_value = finalize_report.get(field)
            if actual_value != expected_value:
                path_violations.append(
                    {
                        "field": field,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
            resolved_path = _resolve_artifact_path(root, actual_value)
            if resolved_path is not None and not _is_under_root(resolved_path, root):
                path_violations.append(
                    {
                        "field": field,
                        "path": str(resolved_path),
                        "error": "path outside output root",
                    }
                )
    return {
        "ok": not missing_tools and not failed_steps and not path_violations,
        "required": True,
        "expected_tools": expected_tools,
        "missing_tools": missing_tools,
        "failed_steps": failed_steps,
        "path_violations": path_violations,
    }


def _prewarm_report_status(
    path: Path,
    required: bool,
    required_tasks: Optional[List[str]] = None,
    required_models: Optional[List[str]] = None,
    expected_dtype: str = "",
    expected_attn_implementation: str = "",
) -> Dict:
    if not required:
        return {"ok": True, "required": False, "path": str(path), "exists": path.exists()}
    if not path.exists():
        return {
            "ok": False,
            "required": True,
            "path": str(path),
            "exists": False,
            "error": "required prewarm report is missing",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "required": True,
            "path": str(path),
            "exists": True,
            "error": "invalid JSON",
            "message": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "required": True,
            "path": str(path),
            "exists": True,
            "error": "prewarm report must be a JSON object",
            "actual_type": type(payload).__name__,
        }
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    attempted = payload.get("attempted") if isinstance(payload.get("attempted"), list) else []
    succeeded = payload.get("succeeded") if isinstance(payload.get("succeeded"), list) else []
    failed = payload.get("failed") if isinstance(payload.get("failed"), list) else []
    targets = payload.get("targets") if isinstance(payload.get("targets"), list) else []
    target_count = payload.get("target_count")
    if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count <= 0:
        target_count = len(targets) if targets else 1
    expected_attempted_count = len(tasks) * max(target_count, 1)
    missing_required_tasks = []
    if required_tasks:
        missing_required_tasks = sorted(set(required_tasks) - set(tasks))
    target_names = [
        item.get("name")
        for item in targets
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name")
    ]
    missing_required_models = []
    if required_models:
        missing_required_models = sorted(set(required_models) - set(target_names))
    attempted_pairs = {
        (item.get("model"), item.get("task"))
        for item in attempted
        if isinstance(item, dict)
    }
    succeeded_pairs = {
        (item.get("model"), item.get("task"))
        for item in succeeded
        if isinstance(item, dict)
    }
    missing_attempted_pairs = []
    missing_succeeded_pairs = []
    if target_names and tasks:
        expected_pairs = {(target, task) for target in target_names for task in tasks}
        missing_attempted_pairs = [
            {"model": model, "task": task}
            for model, task in sorted(expected_pairs - attempted_pairs)
        ]
        missing_succeeded_pairs = [
            {"model": model, "task": task}
            for model, task in sorted(expected_pairs - succeeded_pairs)
        ]
    violations = []
    if payload.get("dry_run") is True:
        violations.append("prewarm dry-run report is not final evidence")
    if not tasks:
        violations.append("tasks is empty or missing")
    if not attempted:
        violations.append("attempted is empty or missing")
    if not succeeded:
        violations.append("succeeded is empty or missing")
    if missing_required_tasks:
        violations.append("prewarm tasks do not cover configured matrix tasks")
    if missing_required_models:
        violations.append("prewarm targets do not cover configured matrix models")
    if tasks and attempted and len(attempted) < expected_attempted_count:
        violations.append("attempted count is smaller than expanded task count times target count")
    if missing_attempted_pairs:
        violations.append("attempted is missing model/task pairs")
    if missing_succeeded_pairs:
        violations.append("succeeded is missing model/task pairs")
    report_dtype = payload.get("dtype")
    dtype_mismatch = None
    if expected_dtype and report_dtype != expected_dtype:
        dtype_mismatch = {"expected": expected_dtype, "actual": report_dtype}
        violations.append("prewarm dtype does not match matrix config")
    report_attn = payload.get("attn_implementation")
    expected_attn = expected_attn_implementation or None
    attn_mismatch = None
    if expected_attn is not None and report_attn != expected_attn:
        attn_mismatch = {"expected": expected_attn, "actual": report_attn}
        violations.append("prewarm attention implementation does not match matrix config")
    ok = payload.get("ok") is True and not failed and not violations
    return {
        "ok": ok,
        "required": True,
        "path": str(path),
        "exists": True,
        "report_ok": payload.get("ok"),
        "dry_run": payload.get("dry_run"),
        "task_count": len(tasks),
        "target_count": target_count,
        "expected_attempted_count": expected_attempted_count,
        "required_task_count": len(required_tasks or []),
        "missing_required_tasks": missing_required_tasks,
        "required_models": required_models or [],
        "missing_required_models": missing_required_models,
        "attempted_count": len(attempted),
        "succeeded_count": len(succeeded),
        "dtype": report_dtype,
        "expected_dtype": expected_dtype or None,
        "dtype_mismatch": dtype_mismatch,
        "attn_implementation": report_attn,
        "expected_attn_implementation": expected_attn,
        "attn_implementation_mismatch": attn_mismatch,
        "missing_attempted_pairs": missing_attempted_pairs,
        "missing_succeeded_pairs": missing_succeeded_pairs,
        "failed": failed,
        "violations": violations,
    }


def _simple_ok_report_status(path: Path, required: bool, label: str) -> Dict:
    if not required and not path.exists():
        return {"ok": True, "required": False, "path": str(path), "exists": path.exists()}
    if not path.exists():
        return {
            "ok": False,
            "required": True,
            "path": str(path),
            "exists": False,
            "error": f"required {label} report is missing",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "required": True,
            "path": str(path),
            "exists": True,
            "error": "invalid JSON",
            "message": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "required": True,
            "path": str(path),
            "exists": True,
            "error": f"{label} report must be a JSON object",
            "actual_type": type(payload).__name__,
        }
    status = {
        "ok": payload.get("ok") is True,
        "required": required,
        "path": str(path),
        "exists": True,
        "report_ok": payload.get("ok"),
    }
    violations = []
    if "dry_run" in payload:
        status["dry_run"] = payload.get("dry_run")
        if required and label != "optional backends" and payload.get("dry_run") is True:
            violations.append("dry-run report is not final evidence")
    if "metadata_only" in payload:
        status["metadata_only"] = payload.get("metadata_only")
        if required and label != "metadata prepare" and payload.get("metadata_only") is True:
            violations.append("metadata-only report is not final evidence")
    for key in ("blockers", "violations", "error"):
        if payload.get(key):
            status[key] = payload.get(key)
    if violations:
        existing_violations = status.get("violations") if isinstance(status.get("violations"), list) else []
        status["violations"] = [*existing_violations, *violations]
        status["ok"] = False
    if isinstance(payload.get("items"), list):
        status["item_count"] = len(payload["items"])
        status["failed_items"] = [
            item
            for item in payload["items"]
            if isinstance(item, dict) and item.get("ok") is not True
        ]
    return status


def _asset_audit_report_status(path: Path, required: bool) -> Dict:
    status = _simple_ok_report_status(path, required, "asset audit")
    if not required or not status.get("exists") or status.get("report_ok") is not True:
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return status
    violations = status.get("violations") if isinstance(status.get("violations"), list) else []
    models = payload.get("models") if isinstance(payload, dict) else None
    precompressed = payload.get("precompressed") if isinstance(payload, dict) else None
    paths = payload.get("paths") if isinstance(payload, dict) else None
    missing_files = payload.get("missing_files") if isinstance(payload, dict) else None
    low_disk = payload.get("low_disk") if isinstance(payload, dict) else None
    unwritable_paths = payload.get("unwritable_paths") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        violations.append("asset audit models must be a list")
    if not isinstance(precompressed, list):
        violations.append("asset audit precompressed must be a list")
    if not isinstance(missing_files, list):
        violations.append("asset audit missing_files must be a list")
    if not isinstance(low_disk, list):
        violations.append("asset audit low_disk must be a list")
    if not isinstance(unwritable_paths, list):
        violations.append("asset audit unwritable_paths must be a list")
    expected_path_keys = {"hf_cache_dir", "data_dir", "output_root", "calibration_file", "prompt_file"}
    path_keys = set(paths) if isinstance(paths, dict) else set()
    missing_path_keys = sorted(expected_path_keys - path_keys)
    if not isinstance(paths, dict):
        violations.append("asset audit paths must be a JSON object")
    elif missing_path_keys:
        violations.append("asset audit paths do not cover required config paths")
    status.update(
        {
            "model_count": len(models) if isinstance(models, list) else None,
            "precompressed_count": len(precompressed) if isinstance(precompressed, list) else None,
            "path_keys": sorted(path_keys),
            "missing_path_keys": missing_path_keys,
        }
    )
    if violations:
        status["violations"] = violations
        status["ok"] = False
    return status


def _prepare_report_status(path: Path, required: bool) -> Dict:
    status = _simple_ok_report_status(path, required, "asset prepare")
    if not required or not status.get("exists") or status.get("report_ok") is not True:
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return status
    models = payload.get("models") if isinstance(payload, dict) else None
    files = payload.get("files") if isinstance(payload, dict) else None
    violations = status.get("violations") if isinstance(status.get("violations"), list) else []
    if not isinstance(models, list) or not models:
        violations.append("asset prepare models are empty or missing")
        models = [] if models is None else models
    if not isinstance(files, list) or not files:
        violations.append("asset prepare files are empty or missing")
        files = [] if files is None else files
    failed_models = [
        {
            "name": item.get("name"),
            "model_id": item.get("model_id"),
            "local_dir": item.get("local_dir"),
            "error": item.get("error"),
        }
        for item in models
        if isinstance(item, dict) and item.get("ok") is not True
    ]
    failed_files = [
        {
            "path": item.get("path"),
            "error": item.get("error"),
        }
        for item in files
        if isinstance(item, dict) and item.get("ok") is not True
    ]
    status.update(
        {
            "model_count": len(models) if isinstance(models, list) else None,
            "file_count": len(files) if isinstance(files, list) else None,
            "failed_models": failed_models,
            "failed_files": failed_files,
        }
    )
    if failed_models:
        violations.append("asset prepare model items failed")
    if failed_files:
        violations.append("asset prepare file items failed")
    if violations:
        status["violations"] = violations
        status["ok"] = False
    return status


def _target_paths_report_status(path: Path, required: bool) -> Dict:
    status = _simple_ok_report_status(path, required, "target paths")
    if not required or not status.get("exists") or status.get("report_ok") is not True:
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return status
    directories = payload.get("directories") if isinstance(payload, dict) else None
    files = payload.get("files") if isinstance(payload, dict) else None
    violations = status.get("violations") if isinstance(status.get("violations"), list) else []
    if not isinstance(directories, dict) or not directories:
        violations.append("target paths directories are empty or missing")
        directories = {} if directories is None else directories
    if not isinstance(files, dict) or not files:
        violations.append("target paths files are empty or missing")
        files = {} if files is None else files
    expected_file_keys = {"calibration_file", "prompt_file"}
    missing_file_keys = sorted(expected_file_keys - set(files)) if isinstance(files, dict) else sorted(expected_file_keys)
    if missing_file_keys:
        violations.append("target paths files do not cover required seed files")
    failed_directories = [
        {"name": name, "path": item.get("path"), "error": item.get("error")}
        for name, item in directories.items()
        if isinstance(item, dict) and item.get("ok") is not True
    ] if isinstance(directories, dict) else []
    failed_files = [
        {"name": name, "path": item.get("path"), "error": item.get("error")}
        for name, item in files.items()
        if isinstance(item, dict) and item.get("ok") is not True
    ] if isinstance(files, dict) else []
    status.update(
        {
            "directory_count": len(directories) if isinstance(directories, dict) else None,
            "file_count": len(files) if isinstance(files, dict) else None,
            "missing_file_keys": missing_file_keys,
            "failed_directories": failed_directories,
            "failed_files": failed_files,
        }
    )
    if failed_directories:
        violations.append("target path directory items failed")
    if failed_files:
        violations.append("target path file items failed")
    if violations:
        status["violations"] = violations
        status["ok"] = False
    return status


def _prepare_metadata_report_status(path: Path, required: bool) -> Dict:
    status = _simple_ok_report_status(path, required, "metadata prepare")
    if not required or not status.get("exists") or status.get("report_ok") is not True:
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return status
    models = payload.get("models") if isinstance(payload, dict) else None
    files = payload.get("files") if isinstance(payload, dict) else None
    violations = status.get("violations") if isinstance(status.get("violations"), list) else []
    if payload.get("metadata_only") is not True:
        violations.append("metadata prepare report must be metadata_only")
    if not isinstance(models, list) or not models:
        violations.append("metadata prepare models are empty or missing")
        models = [] if models is None else models
    if not isinstance(files, list) or not files:
        violations.append("metadata prepare files are empty or missing")
        files = [] if files is None else files
    failed_models = [
        {
            "name": item.get("name"),
            "model_id": item.get("model_id"),
            "local_dir": item.get("local_dir"),
            "error": item.get("error"),
        }
        for item in models
        if isinstance(item, dict) and item.get("ok") is not True
    ]
    failed_files = [
        {
            "path": item.get("path"),
            "error": item.get("error"),
        }
        for item in files
        if isinstance(item, dict) and item.get("ok") is not True
    ]
    status.update(
        {
            "model_count": len(models) if isinstance(models, list) else None,
            "file_count": len(files) if isinstance(files, list) else None,
            "failed_models": failed_models,
            "failed_files": failed_files,
        }
    )
    if failed_models:
        violations.append("metadata prepare model items failed")
    if failed_files:
        violations.append("metadata prepare file items failed")
    if violations:
        status["violations"] = violations
        status["ok"] = False
    return status


def _optional_backends_report_status(path: Path, required: bool) -> Dict:
    status = _simple_ok_report_status(path, required, "optional backends")
    if not required or not status.get("exists") or status.get("report_ok") is not True:
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return status
    violations = status.get("violations") if isinstance(status.get("violations"), list) else []
    mode = payload.get("mode") if isinstance(payload, dict) else None
    requested_packages = payload.get("requested_packages") if isinstance(payload, dict) else None
    current_imports = payload.get("current_imports") if isinstance(payload, dict) else None
    gptqmodel_install = payload.get("gptqmodel_install") if isinstance(payload.get("gptqmodel_install"), dict) else None
    pip_returncode = payload.get("pip_returncode") if isinstance(payload, dict) else None
    status.update(
        {
            "mode": mode,
            "requested_package_count": len(requested_packages) if isinstance(requested_packages, list) else None,
            "current_import_count": len(current_imports) if isinstance(current_imports, dict) else None,
            "gptqmodel_install": gptqmodel_install,
            "pip_returncode": pip_returncode,
        }
    )
    if mode != "dry_run":
        violations.append("optional backend report mode must be dry_run")
    if not isinstance(requested_packages, list) or not requested_packages:
        violations.append("optional backend requested_packages are empty or missing")
    if not isinstance(current_imports, dict) or not current_imports:
        violations.append("optional backend current_imports are empty or missing")
    if pip_returncode != 0:
        violations.append("optional backend pip dry-run failed")
    needs_gptqmodel_runtime = isinstance(requested_packages, list) and any(
        ";" not in str(item) and (str(item) == "gptqmodel" or str(item).startswith("gptqmodel"))
        for item in requested_packages
    )
    status["needs_gptqmodel_runtime"] = needs_gptqmodel_runtime
    if needs_gptqmodel_runtime:
        if not isinstance(gptqmodel_install, dict):
            violations.append("optional backend gptqmodel_install is missing")
        else:
            if gptqmodel_install.get("package") != "gptqmodel==4.0.0":
                violations.append("optional backend gptqmodel_install package must be gptqmodel==4.0.0")
            if gptqmodel_install.get("uses_no_build_isolation") is not True:
                violations.append("optional backend gptqmodel_install must use --no-build-isolation")
            if gptqmodel_install.get("requires_python_3_10") is not True:
                violations.append("optional backend gptqmodel_install must require Python 3.10+")
            if gptqmodel_install.get("requirements_ok") is not True:
                violations.append("optional backend gptqmodel requirements are not satisfied")
            if gptqmodel_install.get("pip_returncode") != 0:
                violations.append("optional backend gptqmodel dry-run failed")
    if violations:
        status["violations"] = violations
        status["ok"] = False
    return status


def _preflight_report_status(
    path: Path,
    required: bool,
    require_optional_backends_report: bool = False,
    require_quant_runtime_report: bool = False,
) -> Dict:
    status = _simple_ok_report_status(path, required, "preflight")
    if not required or not status.get("exists") or status.get("report_ok") is not True:
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return status
    steps = payload.get("steps") if isinstance(payload, dict) else None
    violations = status.get("violations") if isinstance(status.get("violations"), list) else []
    if not isinstance(steps, list) or not steps:
        violations.append("preflight steps are empty or missing")
        status["steps"] = [] if steps is None else steps
    else:
        expected_step_names = {"asset_audit", "verify_env", "method_readiness", "matrix_dry_run"}
        if require_optional_backends_report:
            expected_step_names.add("optional_backend_resolution")
        if require_quant_runtime_report:
            expected_step_names.add("quant_runtime")
        step_names = [
            step.get("name")
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("name"), str)
        ]
        missing_steps = sorted(expected_step_names - set(step_names))
        failed_required_steps = [
            {
                "name": step.get("name"),
                "returncode": step.get("returncode"),
                "ok": step.get("ok"),
            }
            for step in steps
            if isinstance(step, dict) and step.get("required") is True and step.get("ok") is not True
        ]
        verify_env_steps = [
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == "verify_env"
        ]
        verify_env_payload = (
            verify_env_steps[0].get("payload")
            if verify_env_steps and isinstance(verify_env_steps[0].get("payload"), dict)
            else None
        )
        cuda_visible_status = (
            verify_env_payload.get("cuda_visible_devices")
            if isinstance(verify_env_payload, dict) and isinstance(verify_env_payload.get("cuda_visible_devices"), dict)
            else None
        )
        cuda_visible_matches = (
            cuda_visible_status.get("matches_config") is True
            if isinstance(cuda_visible_status, dict)
            else False
        )
        status.update(
            {
                "step_count": len(steps),
                "step_names": step_names,
                "missing_steps": missing_steps,
                "failed_required_steps": failed_required_steps,
                "verify_env_cuda_visible_devices": cuda_visible_status,
            }
        )
        if missing_steps:
            violations.append("preflight required steps are missing")
        if failed_required_steps:
            violations.append("preflight required steps failed")
        if "verify_env" in step_names and not cuda_visible_matches:
            violations.append("preflight verify_env CUDA_VISIBLE_DEVICES did not match config")
    if violations:
        status["violations"] = violations
        status["ok"] = False
    return status


def _quant_runtime_report_status(path: Path, required: bool) -> Dict:
    status = _simple_ok_report_status(path, required, "quant runtime")
    if not required or not status.get("exists") or status.get("report_ok") is not True:
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return status
    rows = payload.get("rows") if isinstance(payload, dict) else None
    row_source_field = "rows" if isinstance(rows, list) else None
    if not isinstance(rows, list) and isinstance(payload, dict):
        rows = payload.get("probes")
        row_source_field = "probes" if isinstance(rows, list) else None
    violations = status.get("violations") if isinstance(status.get("violations"), list) else []
    if not isinstance(rows, list) or not rows:
        violations.append("quant runtime rows are empty or missing")
        status["rows"] = [] if rows is None else rows
    else:
        failed_rows = [
            {
                "model": row.get("model"),
                "method": row.get("method"),
                "path": row.get("model_path"),
                "reason": row.get("reason"),
                "returncode": row.get("returncode"),
            }
            for row in rows
            if isinstance(row, dict) and row.get("ok") is not True
        ]
        status.update(
            {
                "entry_source_field": row_source_field,
                "row_count": len(rows),
                "failed_rows": failed_rows,
            }
        )
        if failed_rows:
            violations.append("quant runtime rows failed")
    if violations:
        status["violations"] = violations
        status["ok"] = False
    return status


def _prepare_precompressed_report_status(path: Path, required: bool, expected_config_path: str = "") -> Dict:
    status = _simple_ok_report_status(path, required, "precompressed prepare")
    if not required or not status.get("exists") or status.get("report_ok") is not True:
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return status
    items = payload.get("items") if isinstance(payload, dict) else None
    blockers = payload.get("blockers") if isinstance(payload, dict) and isinstance(payload.get("blockers"), list) else []
    violations = status.get("violations") if isinstance(status.get("violations"), list) else []
    if not isinstance(items, list) or not items:
        violations.append("precompressed preparation items are empty or missing")
        status["items"] = [] if items is None else items
    else:
        failed_items = [
            {
                "model": item.get("model"),
                "method": item.get("method"),
                "repo": item.get("repo"),
                "local_dir": item.get("local_dir"),
                "reason": item.get("reason"),
            }
            for item in items
            if isinstance(item, dict) and item.get("ok") is not True
        ]
        status.update(
            {
                "item_count": len(items),
                "failed_items": failed_items,
            }
        )
        if failed_items:
            violations.append("precompressed preparation items failed")
    status["blockers"] = blockers
    if blockers:
        violations.append("precompressed preparation blockers are present")
    updated_config = payload.get("updated_config") if isinstance(payload, dict) else None
    status["updated_config"] = updated_config
    if not updated_config:
        violations.append("precompressed updated_config is missing")
    else:
        updated_config_path = Path(str(updated_config)).expanduser()
        status["updated_config_exists"] = updated_config_path.exists()
        if not updated_config_path.exists():
            violations.append("precompressed updated_config file is missing")
        else:
            try:
                updated_config_payload = json.loads(updated_config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                status["updated_config_parse_error"] = str(exc)
                violations.append("precompressed updated_config is not valid JSON")
            else:
                if not isinstance(updated_config_payload, dict):
                    violations.append("precompressed updated_config must be a JSON object")
        if expected_config_path:
            expected_resolved = str(Path(expected_config_path).expanduser())
            actual_resolved = str(updated_config_path)
            status["expected_config_path"] = expected_resolved
            if actual_resolved != expected_resolved:
                violations.append("precompressed updated_config does not match audit config")
    if violations:
        status["violations"] = violations
        status["ok"] = False
    return status


def _target_readiness_status(
    path: Path,
    required: bool,
    root: Path,
    config_path: str,
    completion_audit_path: Path,
) -> Dict:
    if not required and not path.exists():
        return {"ok": True, "required": False, "path": str(path), "exists": False}
    if not path.exists():
        return {
            "ok": False,
            "required": required,
            "path": str(path),
            "exists": False,
            "error": "required target readiness report is missing",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "required": required,
            "path": str(path),
            "exists": True,
            "error": "invalid JSON",
            "message": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "required": required,
            "path": str(path),
            "exists": True,
            "error": "target readiness report must be a JSON object",
            "actual_type": type(payload).__name__,
        }
    violations = []
    if payload.get("root") != str(root):
        violations.append(
            {
                "field": "root",
                "expected": str(root),
                "actual": payload.get("root"),
            }
        )
    config_value = payload.get("config")
    if config_value != config_path and str(Path(config_value or "").expanduser()) != str(Path(config_path).expanduser()):
        violations.append(
            {
                "field": "config",
                "expected": config_path,
                "actual": config_value,
            }
        )
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    completion_artifact = artifacts.get("completion_audit") if isinstance(artifacts.get("completion_audit"), dict) else {}
    if completion_artifact.get("path") != str(completion_audit_path):
        violations.append(
            {
                "field": "artifacts.completion_audit.path",
                "expected": str(completion_audit_path),
                "actual": completion_artifact.get("path"),
            }
        )
    if completion_artifact.get("exists") is not True:
        violations.append({"field": "artifacts.completion_audit.exists", "expected": True, "actual": completion_artifact.get("exists")})
    if completion_artifact.get("ok") is not True:
        violations.append({"field": "artifacts.completion_audit.ok", "expected": True, "actual": completion_artifact.get("ok")})
    if payload.get("completion_ok") is not True:
        violations.append({"field": "completion_ok", "expected": True, "actual": payload.get("completion_ok")})
    if payload.get("ok") is not True:
        violations.append({"field": "ok", "expected": True, "actual": payload.get("ok")})
    blockers = payload.get("before_matrix_blockers")
    if not isinstance(blockers, list):
        violations.append(
            {
                "field": "before_matrix_blockers",
                "expected": "list",
                "actual_type": type(blockers).__name__,
            }
        )
        blockers = []
    if blockers:
        violations.append(
            {
                "field": "before_matrix_blockers",
                "expected": [],
                "actual": blockers,
            }
        )
    if payload.get("ready_for_matrix") is not True:
        violations.append({"field": "ready_for_matrix", "expected": True, "actual": payload.get("ready_for_matrix")})
    return {
        "ok": not violations,
        "required": required,
        "path": str(path),
        "exists": True,
        "report_ok": payload.get("ok"),
        "ready_for_matrix": payload.get("ready_for_matrix"),
        "completion_ok": payload.get("completion_ok"),
        "before_matrix_blockers": blockers,
        "violations": violations,
    }


def _asset_storage_status(path: Path, required: bool) -> Dict:
    if not required and not path.exists():
        return {"ok": True, "required": False, "path": str(path), "exists": False}
    if not path.exists():
        return {
            "ok": False,
            "required": required,
            "path": str(path),
            "exists": False,
            "error": "required asset audit report is missing",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "required": required,
            "path": str(path),
            "exists": True,
            "error": "invalid JSON",
            "message": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "required": required,
            "path": str(path),
            "exists": True,
            "error": "asset audit report must be a JSON object",
            "actual_type": type(payload).__name__,
        }
    paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
    required_paths = ("hf_cache_dir", "data_dir", "output_root")
    missing = [name for name in required_paths if name not in paths]
    violations = []
    media = {}
    for name in required_paths:
        item = paths.get(name) if isinstance(paths.get(name), dict) else {}
        block = item.get("block_device") if isinstance(item.get("block_device"), dict) else {}
        mount = item.get("mount") if isinstance(item.get("mount"), dict) else {}
        media[name] = {
            "path": item.get("path"),
            "probe_path": item.get("probe_path"),
            "mount_source": mount.get("source"),
            "mount_fstype": mount.get("fstype"),
            "block_device": block.get("device"),
            "is_non_rotational": block.get("is_non_rotational"),
            "is_writable": item.get("is_writable"),
            "can_create": item.get("can_create"),
        }
        if name in paths and block.get("is_non_rotational") is not True:
            violations.append(
                {
                    "name": name,
                    "path": item.get("path"),
                    "probe_path": item.get("probe_path"),
                    "block_device": block,
                    "error": "path is not proven to be on non-rotational storage",
                }
            )
        if name in paths and item.get("can_create") is False:
            violations.append(
                {
                    "name": name,
                    "path": item.get("path"),
                    "probe_path": item.get("writable_probe_path") or item.get("probe_path"),
                    "error": "path does not exist with a writable parent, or exists but is not writable",
                }
            )
    return {
        "ok": (not missing and not violations) if required else True,
        "required": required,
        "path": str(path),
        "exists": True,
        "missing_paths": missing,
        "media": media,
        "violations": violations,
    }


def _handoff_manifest_status(
    path: Path,
    required: bool,
    current_code: Optional[Dict] = None,
    require_code_match: bool = False,
    require_clean_worktree: bool = False,
) -> Dict:
    if not required and not path.exists():
        return {"ok": True, "required": False, "path": str(path), "exists": False}
    if not path.exists():
        return {
            "ok": False,
            "required": required,
            "path": str(path),
            "exists": False,
            "error": "required handoff manifest is missing",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "required": required,
            "path": str(path),
            "exists": True,
            "error": "invalid JSON",
            "message": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "required": required,
            "path": str(path),
            "exists": True,
            "error": "handoff manifest must be a JSON object",
            "actual_type": type(payload).__name__,
        }
    violations = []
    if payload.get("schema_version") != 1:
        violations.append("schema_version must be 1")
    for field in ("code", "config", "target_environment", "paths", "commands", "required_artifacts", "completion_gates"):
        if not isinstance(payload.get(field), dict):
            violations.append(f"{field} must be a JSON object")
    commands = payload.get("commands") if isinstance(payload.get("commands"), dict) else {}
    for group in (
        "setup",
        "prepare",
        "prepare_precompressed",
        "preflight",
        "prewarm",
        "readiness",
        "precompressed_config_followup",
        "run",
        "manual_finalize",
        "post_run_audit",
        "baseline_post_run_audit",
        "precompressed_config_post_run_audit",
    ):
        if not isinstance(commands.get(group), list) or not commands.get(group):
            violations.append(f"commands.{group} must be a non-empty list")
    prepare_precompressed_commands = (
        commands.get("prepare_precompressed")
        if isinstance(commands.get("prepare_precompressed"), list)
        else []
    )
    prepare_precompressed_cmd = str(prepare_precompressed_commands[0]) if prepare_precompressed_commands else ""
    direct_prepare_tokens = (
        "tools/prepare_precompressed_assets.py",
        "edge_prepare_precompressed_report.json",
        "edge_matrix_with_precompressed.json",
        "--fail-on-error",
    )
    grouped_wrapper_tokens = (
        "RUN_PRECOMPRESSED_PREPARE=1",
        "PRECOMPRESSED_PREPARE_ONLY=1",
        "PRECOMPRESSED_GROUPS=",
        "scripts/run_edge_4090_full.sh",
        "edge_prepare_precompressed_report.json",
        "edge_matrix_with_precompressed.json",
    )
    direct_prepare_ok = all(token in prepare_precompressed_cmd for token in direct_prepare_tokens)
    grouped_wrapper_ok = all(token in prepare_precompressed_cmd for token in grouped_wrapper_tokens)
    if prepare_precompressed_cmd and not direct_prepare_ok and not grouped_wrapper_ok:
        missing_direct = [token for token in direct_prepare_tokens if token not in prepare_precompressed_cmd]
        missing_grouped = [token for token in grouped_wrapper_tokens if token not in prepare_precompressed_cmd]
        violations.append(
            "commands.prepare_precompressed[0] must be either a direct "
            f"prepare_precompressed_assets.py command missing {missing_direct} "
            f"or a grouped run_edge_4090_full.sh wrapper missing {missing_grouped}"
        )
    preflight_commands = commands.get("preflight") if isinstance(commands.get("preflight"), list) else []
    preflight_cmd = str(preflight_commands[0]) if preflight_commands else ""
    for token in ("--check-optional-backend-resolution", "--check-quant-runtime"):
        if token not in preflight_cmd:
            violations.append(f"commands.preflight[0] missing {token}")
    prewarm_commands = commands.get("prewarm") if isinstance(commands.get("prewarm"), list) else []
    prewarm_cmd = str(prewarm_commands[0]) if prewarm_commands else ""
    for token in ("tools/prewarm_lm_eval_tasks.py", "--expand-mmlu", "edge_prewarm_report.json"):
        if token not in prewarm_cmd:
            violations.append(f"commands.prewarm[0] missing {token}")
    readiness_commands = commands.get("readiness") if isinstance(commands.get("readiness"), list) else []
    readiness_cmd = str(readiness_commands[0]) if readiness_commands else ""
    readiness_required_tokens = [
        "tools/audit_target_readiness.py",
        "--check-method-readiness",
        "--require-non-rotational-storage",
        "--require-prewarm",
        "--require-handoff-manifest",
        "--require-optional-backends-report",
        "--require-quant-runtime-report",
    ]
    for token in readiness_required_tokens:
        if token not in readiness_cmd:
            violations.append(f"commands.readiness[0] missing {token}")
    precompressed_followup = (
        commands.get("precompressed_config_followup")
        if isinstance(commands.get("precompressed_config_followup"), list)
        else []
    )
    if not any("edge_matrix_with_precompressed.json" in str(command) for command in precompressed_followup):
        violations.append("commands.precompressed_config_followup must use edge_matrix_with_precompressed.json")
    if not any("tools/audit_target_readiness.py" in str(command) for command in precompressed_followup):
        violations.append("commands.precompressed_config_followup must include tools/audit_target_readiness.py")
    if not any(
        "tools/audit_target_readiness.py" in str(command)
        and "--require-prepare-precompressed-report" in str(command)
        for command in precompressed_followup
    ):
        violations.append("commands.precompressed_config_followup readiness missing --require-prepare-precompressed-report")
    if not any("tools/run_edge_matrix.py" in str(command) and "edge_matrix_with_precompressed.json" in str(command) for command in precompressed_followup):
        violations.append("commands.precompressed_config_followup must include matrix run with edge_matrix_with_precompressed.json")
    if not any("tools/audit_edge_completion.py" in str(command) and "edge_matrix_with_precompressed.json" in str(command) for command in precompressed_followup):
        violations.append("commands.precompressed_config_followup must include completion audit with edge_matrix_with_precompressed.json")
    if not any(
        "tools/run_edge_matrix.py" in str(command)
        and "--require-prepare-precompressed-report" in str(command)
        for command in precompressed_followup
    ):
        violations.append("commands.precompressed_config_followup matrix run missing --require-prepare-precompressed-report")
    if not any(
        "tools/audit_edge_completion.py" in str(command)
        and "--require-prepare-precompressed-report" in str(command)
        for command in precompressed_followup
    ):
        violations.append("commands.precompressed_config_followup completion audit missing --require-prepare-precompressed-report")
    post_run_commands = commands.get("post_run_audit") if isinstance(commands.get("post_run_audit"), list) else []
    if not any("tools/build_goal_audit.py" in str(command) for command in post_run_commands):
        violations.append("commands.post_run_audit must include tools/build_goal_audit.py")
    post_run_readiness_cmd = next(
        (str(command) for command in post_run_commands if "tools/audit_target_readiness.py" in str(command)),
        "",
    )
    for token in readiness_required_tokens:
        if token not in post_run_readiness_cmd:
            violations.append(f"commands.post_run_audit readiness command missing {token}")
    baseline_post_run_commands = (
        commands.get("baseline_post_run_audit")
        if isinstance(commands.get("baseline_post_run_audit"), list)
        else []
    )
    if not any("tools/build_goal_audit.py" in str(command) for command in baseline_post_run_commands):
        violations.append("commands.baseline_post_run_audit must include tools/build_goal_audit.py")
    baseline_post_readiness_cmd = next(
        (str(command) for command in baseline_post_run_commands if "tools/audit_target_readiness.py" in str(command)),
        "",
    )
    for token in readiness_required_tokens:
        if token not in baseline_post_readiness_cmd:
            violations.append(f"commands.baseline_post_run_audit readiness command missing {token}")
    if "edge_matrix_with_precompressed.json" in baseline_post_readiness_cmd:
        violations.append("commands.baseline_post_run_audit readiness command must use the baseline config")
    precompressed_post_run_commands = (
        commands.get("precompressed_config_post_run_audit")
        if isinstance(commands.get("precompressed_config_post_run_audit"), list)
        else []
    )
    precompressed_post_readiness_cmd = next(
        (str(command) for command in precompressed_post_run_commands if "tools/audit_target_readiness.py" in str(command)),
        "",
    )
    for token in (*readiness_required_tokens, "edge_matrix_with_precompressed.json", "--require-prepare-precompressed-report"):
        if token not in precompressed_post_readiness_cmd:
            violations.append(f"commands.precompressed_config_post_run_audit readiness command missing {token}")
    if not any("tools/build_goal_audit.py" in str(command) for command in precompressed_post_run_commands):
        violations.append("commands.precompressed_config_post_run_audit must include tools/build_goal_audit.py")
    manual_finalize = commands.get("manual_finalize") if isinstance(commands.get("manual_finalize"), list) else []
    if len(manual_finalize) < 2:
        violations.append("commands.manual_finalize must include matrix finalize and completion audit commands")
    else:
        matrix_finalize_cmd = str(manual_finalize[0])
        completion_audit_cmd = str(manual_finalize[1])
        matrix_required_tokens = [
            "tools/run_edge_matrix.py",
            "--finalize",
            "--require-prewarm-report",
            "--require-asset-audit-report",
            "--require-non-rotational-storage",
            "--require-preflight-report",
            "--require-handoff-manifest",
            "--require-target-readiness-report",
            "--require-optional-backends-report",
            "--require-quant-runtime-report",
            "--require-runtime-python-prefix 3.10",
            "--require-runtime-gpu-name-contains 4090",
            "--require-runtime-cuda-prefix 12.",
            "--require-runtime-gpu-total-memory-min-gb 23",
            "--require-runtime-gpu-count 1",
            "--require-runtime-cudnn",
            "--require-runtime-torch-prefix 2.",
            "--require-runtime-transformers",
            "--require-runtime-bitsandbytes",
            "--require-runtime-cpu-count-min 8",
            "--require-runtime-system-memory-min-gb 240",
            "--require-runtime-hf-endpoint https://hf-mirror.com",
            "--require-code-match",
            "--require-clean-worktree",
        ]
        audit_required_tokens = [
            "tools/audit_edge_completion.py",
            "--require-finalize-report",
            *matrix_required_tokens[2:],
        ]
        for token in matrix_required_tokens:
            if token not in matrix_finalize_cmd:
                violations.append(f"commands.manual_finalize[0] missing {token}")
        for token in audit_required_tokens:
            if token not in completion_audit_cmd:
                violations.append(f"commands.manual_finalize[1] missing {token}")
    required_artifacts = payload.get("required_artifacts") if isinstance(payload.get("required_artifacts"), dict) else {}
    for name in (
        "matrix_config_resolved.json",
        "edge_asset_audit.json",
        "edge_preflight_report.json",
        "edge_prewarm_report.json",
        "edge_handoff_manifest.json",
        "edge_optional_backends_report.json",
        "edge_quant_runtime_report.json",
        "matrix_run_report.json",
        "matrix_finalize_report.json",
        "matrix_results.json",
        "matrix_results.csv",
        "edge_report.md",
        "completion_audit.json",
        "target_readiness.json",
    ):
        if name not in required_artifacts:
            violations.append(f"required_artifacts missing {name}")
    completion_gates = payload.get("completion_gates") if isinstance(payload.get("completion_gates"), dict) else {}
    coverage_gates = completion_gates.get("coverage") if isinstance(completion_gates.get("coverage"), dict) else {}
    if coverage_gates:
        if coverage_gates.get("required_mmlu_subtask_count") != len(DEFAULT_MMLU_SUBTASKS):
            violations.append("completion_gates.coverage.required_mmlu_subtask_count must match DEFAULT_MMLU_SUBTASKS")
        required_mmlu_subtasks = coverage_gates.get("required_mmlu_subtasks")
        if required_mmlu_subtasks != list(DEFAULT_MMLU_SUBTASKS):
            violations.append("completion_gates.coverage.required_mmlu_subtasks must match DEFAULT_MMLU_SUBTASKS")
    reports = completion_gates.get("reports") if isinstance(completion_gates.get("reports"), list) else []
    if not any("target_readiness.json" in str(item) and "ok=true" in str(item) for item in reports):
        violations.append("completion_gates.reports must include target_readiness.json with ok=true")
    if not any("completion_audit.json" in str(item) and "ok=true" in str(item) for item in reports):
        violations.append("completion_gates.reports must include completion_audit.json with ok=true")
    manifest_code = payload.get("code") if isinstance(payload.get("code"), dict) else {}
    code_matches_current = {}
    if current_code:
        for key in ("commit", "branch", "origin"):
            code_matches_current[key] = bool(
                manifest_code.get(key)
                and current_code.get(key)
                and manifest_code.get(key) == current_code.get(key)
            )
        code_matches_current["dirty"] = (
            isinstance(manifest_code.get("dirty"), bool)
            and isinstance(current_code.get("dirty"), bool)
            and manifest_code.get("dirty") == current_code.get("dirty")
        )
        if require_code_match:
            for key in ("commit", "branch", "origin"):
                if not code_matches_current.get(key):
                    violations.append(f"code.{key} does not match current checkout")
        if require_clean_worktree and manifest_code.get("dirty") is not False:
            violations.append("code.dirty must be false")
    return {
        "ok": not violations,
        "required": required,
        "path": str(path),
        "exists": True,
        "schema_version": payload.get("schema_version"),
        "code_commit": manifest_code.get("commit"),
        "code_branch": manifest_code.get("branch"),
        "code_origin": manifest_code.get("origin"),
        "code_dirty": manifest_code.get("dirty"),
        "code_matches_current": code_matches_current,
        "output_root": payload.get("paths", {}).get("output_root") if isinstance(payload.get("paths"), dict) else None,
        "violations": violations,
    }


def audit(args: argparse.Namespace) -> Dict:
    root = Path(args.root).expanduser()
    source_config = _load_json(Path(args.config))
    if not isinstance(source_config, dict):
        raise SystemExit(f"Config not found or invalid: {args.config}")

    results_path = root / "matrix_results.json"
    results_csv_path = root / "matrix_results.csv"
    edge_report_path = root / "edge_report.md"
    run_report_path = root / "matrix_run_report.json"
    finalize_report_path = root / "matrix_finalize_report.json"
    resolved_config_path = root / "matrix_config_resolved.json"
    validation_path = root / "completion_validation.json"
    paper_protocol_validation_path = root / "completion_paper_protocol_validation.json"
    prewarm_report_path = root / "edge_prewarm_report.json"
    asset_audit_report_path = root / "edge_asset_audit.json"
    preflight_report_path = root / "edge_preflight_report.json"
    handoff_manifest_path = root / "edge_handoff_manifest.json"
    target_readiness_path = root / "target_readiness.json"
    optional_backends_report_path = root / "edge_optional_backends_report.json"
    quant_runtime_report_path = root / "edge_quant_runtime_report.json"
    target_paths_report_path = root / "edge_target_paths_report.json"
    prepare_report_path = root / "edge_prepare_assets_report.json"
    prepare_metadata_report_path = root / "edge_prepare_metadata_report.json"
    prepare_precompressed_report_path = root / "edge_prepare_precompressed_report.json"

    artifact_parse_errors = []
    matrix_results = _load_json(results_path, artifact_parse_errors, "matrix_results_json")
    run_report = _load_json(run_report_path, artifact_parse_errors, "matrix_run_report")
    finalize_report = _load_json(finalize_report_path, artifact_parse_errors, "matrix_finalize_report")
    resolved_config = _load_json(resolved_config_path, artifact_parse_errors, "matrix_config_resolved")
    resolved_config_usable = (
        isinstance(resolved_config, dict)
        and isinstance(resolved_config.get("models"), list)
        and isinstance(resolved_config.get("methods"), list)
        and isinstance(resolved_config.get("defaults"), dict)
    )
    config = resolved_config if resolved_config_usable else source_config
    code_status = _code_status(resolved_config)
    resolved_selection_status = _selection_status(source_config, resolved_config, args.config, root, args.only_model, args.only_method)
    validator_config_path = str(resolved_config_path) if resolved_config_usable else args.config
    expected_pairs = _expected_pairs(config, args.only_model, args.only_method)
    expected_pair_set = {(item["model"], item["method"]) for item in expected_pairs}
    scoped_matrix_results = [
        row
        for row in matrix_results
        if isinstance(row, dict) and (row.get("model_name"), row.get("compression_method")) in expected_pair_set
    ] if isinstance(matrix_results, list) else matrix_results
    runtime_cuda_visible_devices = (
        config.get("defaults", {}).get("cuda_visible_devices")
        if isinstance(config.get("defaults"), dict)
        else ""
    )
    runtime_violations = _runtime_violations(
        scoped_matrix_results,
        args.require_runtime_python_prefix,
        args.require_runtime_gpu_name_contains,
        args.require_runtime_cuda_prefix,
        runtime_cuda_visible_devices,
        args.require_runtime_gpu_total_memory_min_gb,
        args.require_runtime_gpu_count,
        args.require_runtime_cudnn,
        args.require_runtime_torch_prefix,
        args.require_runtime_transformers,
        args.require_runtime_bitsandbytes,
        args.require_runtime_cpu_count_min,
        args.require_runtime_system_memory_min_gb,
        args.require_runtime_hf_endpoint,
    )
    mmlu_subtask_coverage_status = _mmlu_subtask_coverage_status(
        scoped_matrix_results,
        expected_pairs,
        config,
    )

    validator = None
    paper_protocol_validator = None
    if results_path.exists():
        validator = _run_validator(args, validator_config_path, results_path, validation_path)
        if config.get("require_paper_protocol_validation") is True:
            paper_protocol_validator = _run_paper_protocol_validator(
                args,
                validator_config_path,
                results_path,
                paper_protocol_validation_path,
            )

    missing_artifacts = [
        str(path)
        for path in [results_path, results_csv_path, edge_report_path, run_report_path, resolved_config_path]
        if not path.exists()
    ]
    if args.require_finalize_report and not finalize_report_path.exists():
        missing_artifacts.append(str(finalize_report_path))
    if args.require_prewarm_report and not prewarm_report_path.exists():
        missing_artifacts.append(str(prewarm_report_path))
    if args.require_asset_audit_report and not asset_audit_report_path.exists():
        missing_artifacts.append(str(asset_audit_report_path))
    if args.require_preflight_report and not preflight_report_path.exists():
        missing_artifacts.append(str(preflight_report_path))
    if args.require_handoff_manifest and not handoff_manifest_path.exists():
        missing_artifacts.append(str(handoff_manifest_path))
    if args.require_target_readiness_report and not target_readiness_path.exists():
        missing_artifacts.append(str(target_readiness_path))
    if args.require_optional_backends_report and not optional_backends_report_path.exists():
        missing_artifacts.append(str(optional_backends_report_path))
    if args.require_quant_runtime_report and not quant_runtime_report_path.exists():
        missing_artifacts.append(str(quant_runtime_report_path))
    if args.require_target_paths_report and not target_paths_report_path.exists():
        missing_artifacts.append(str(target_paths_report_path))
    if args.require_prepare_report and not prepare_report_path.exists():
        missing_artifacts.append(str(prepare_report_path))
    if args.require_prepare_metadata_report and not prepare_metadata_report_path.exists():
        missing_artifacts.append(str(prepare_metadata_report_path))
    if args.require_prepare_precompressed_report and not prepare_precompressed_report_path.exists():
        missing_artifacts.append(str(prepare_precompressed_report_path))

    run_failures = []
    skipped_external = []
    skipped_existing = []
    if isinstance(run_report, dict):
        run_failures = run_report.get("failed", []) or []
        skipped_external = run_report.get("skipped_external_not_configured", []) or []
        skipped_existing = run_report.get("skipped", []) or []
    run_report_status = _run_report_status(
        run_report,
        expected_pairs,
        root,
        matrix_results,
        config=config,
        allow_unexpected=args.allow_unexpected,
        allow_skipped_external=args.allow_skipped_external,
    )

    finalize_ok = None
    if isinstance(finalize_report, dict):
        finalize_ok = bool(finalize_report.get("ok"))
    finalize_report_status = _finalize_report_status(
        finalize_report,
        root,
        args.require_prewarm_report,
        args.require_asset_audit_report,
        args.require_preflight_report,
        args.require_handoff_manifest,
        args.require_target_readiness_report,
        args.require_target_paths_report,
        args.require_prepare_report,
        args.require_prepare_metadata_report,
        args.require_prepare_precompressed_report,
    )
    edge_report_sections = _report_section_status(
        edge_report_path,
        results_path,
        matrix_results,
        prewarm_report_path,
        asset_audit_report_path,
        preflight_report_path,
        prepare_report_path,
        prepare_metadata_report_path,
        prepare_precompressed_report_path,
    )
    edge_report_render_status = _render_report_status(results_path, run_report_path, edge_report_path, root)
    matrix_results_csv_status = _csv_status(results_csv_path, matrix_results)
    config_defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    required_prewarm_tasks = _expand_prewarm_tasks(_split_tasks(config_defaults.get("tasks", "")))
    required_prewarm_models = (
        sorted({item["model"] for item in expected_pairs if item.get("model")})
        if args.require_prewarm_all_models
        else []
    )
    prewarm_report_status = _prewarm_report_status(
        prewarm_report_path,
        args.require_prewarm_report,
        required_prewarm_tasks,
        required_prewarm_models,
        config_defaults.get("dtype", "") or "",
        config_defaults.get("attn_implementation", "") or "",
    )
    asset_audit_report_status = _asset_audit_report_status(
        asset_audit_report_path,
        args.require_asset_audit_report,
    )
    asset_storage_status = _asset_storage_status(
        asset_audit_report_path,
        args.require_non_rotational_storage,
    )
    preflight_report_status = _preflight_report_status(
        preflight_report_path,
        args.require_preflight_report,
        args.require_optional_backends_report,
        args.require_quant_runtime_report,
    )
    handoff_manifest_status = _handoff_manifest_status(
        handoff_manifest_path,
        args.require_handoff_manifest,
        code_status["current"],
        args.require_code_match,
        args.require_clean_worktree,
    )
    target_readiness_status = _target_readiness_status(
        target_readiness_path,
        args.require_target_readiness_report,
        root,
        args.config,
        root / "completion_audit.json",
    )
    optional_backends_report_status = _optional_backends_report_status(
        optional_backends_report_path,
        args.require_optional_backends_report,
    )
    quant_runtime_report_status = _quant_runtime_report_status(
        quant_runtime_report_path,
        args.require_quant_runtime_report,
    )
    target_paths_report_status = _target_paths_report_status(
        target_paths_report_path,
        args.require_target_paths_report,
    )
    prepare_report_status = _prepare_report_status(
        prepare_report_path,
        args.require_prepare_report,
    )
    prepare_metadata_report_status = _prepare_metadata_report_status(
        prepare_metadata_report_path,
        args.require_prepare_metadata_report,
    )
    prepare_precompressed_report_status = _prepare_precompressed_report_status(
        prepare_precompressed_report_path,
        args.require_prepare_precompressed_report,
        args.config,
    )
    artifact_paths = {
        "matrix_results_json": results_path,
        "matrix_results_csv": results_csv_path,
        "edge_report": edge_report_path,
        "matrix_run_report": run_report_path,
        "matrix_finalize_report": finalize_report_path,
        "matrix_config_resolved": resolved_config_path,
        "completion_validation": validation_path,
        "completion_paper_protocol_validation": paper_protocol_validation_path,
        "edge_prewarm_report": prewarm_report_path,
        "edge_asset_audit": asset_audit_report_path,
        "edge_preflight_report": preflight_report_path,
        "edge_handoff_manifest": handoff_manifest_path,
        "target_readiness": target_readiness_path,
        "edge_optional_backends_report": optional_backends_report_path,
        "edge_quant_runtime_report": quant_runtime_report_path,
        "edge_target_paths_report": target_paths_report_path,
        "edge_prepare_assets_report": prepare_report_path,
        "edge_prepare_metadata_report": prepare_metadata_report_path,
        "edge_prepare_precompressed_report": prepare_precompressed_report_path,
    }

    report = {
        "ok": (
            not missing_artifacts
            and bool(validator and validator["ok"])
            and (paper_protocol_validator is None or paper_protocol_validator["ok"])
            and not run_failures
            and (args.allow_skipped_external or not skipped_external)
            and (finalize_ok in (None, True))
            and not runtime_violations
            and resolved_selection_status["ok"]
            and edge_report_sections["ok"]
            and edge_report_render_status["ok"]
            and mmlu_subtask_coverage_status["ok"]
            and not artifact_parse_errors
            and matrix_results_csv_status["ok"]
            and run_report_status["ok"]
            and finalize_report_status["ok"]
            and prewarm_report_status["ok"]
            and asset_audit_report_status["ok"]
            and asset_storage_status["ok"]
            and preflight_report_status["ok"]
            and handoff_manifest_status["ok"]
            and target_readiness_status["ok"]
            and (optional_backends_report_status["ok"] or not args.require_optional_backends_report)
            and (quant_runtime_report_status["ok"] or not args.require_quant_runtime_report)
            and target_paths_report_status["ok"]
            and prepare_report_status["ok"]
            and prepare_metadata_report_status["ok"]
            and prepare_precompressed_report_status["ok"]
            and (not args.require_code_match or (code_status["all_present"] and code_status["all_match_current"]))
            and (
                not args.require_clean_worktree
                or (code_status["resolved"].get("dirty") is False and code_status["current"].get("dirty") is False)
            )
        ),
        "root": str(root),
        "config": args.config,
        "validator_config": validator_config_path,
        "uses_resolved_config": resolved_config_usable,
        "code_status": code_status,
        "resolved_selection_status": resolved_selection_status,
        "final_gate": "completion_audit.json ok=true",
        "expected_pairs": expected_pairs,
        "artifact_status": {
            name: path.exists() for name, path in artifact_paths.items()
        },
        "artifact_sha256": {name: _sha256(path) for name, path in artifact_paths.items() if path.exists()},
        "missing_artifacts": missing_artifacts,
        "artifact_parse_errors": artifact_parse_errors,
        "matrix_results_csv_status": matrix_results_csv_status,
        "matrix_results_rows": len(matrix_results) if isinstance(matrix_results, list) else None,
        "runtime_requirements": {
            "python_prefix": args.require_runtime_python_prefix,
            "gpu_name_contains": args.require_runtime_gpu_name_contains,
            "cuda_prefix": args.require_runtime_cuda_prefix,
            "cuda_visible_devices": runtime_cuda_visible_devices,
            "gpu_total_memory_min_gb": args.require_runtime_gpu_total_memory_min_gb,
            "gpu_count": args.require_runtime_gpu_count,
            "cudnn_required": args.require_runtime_cudnn,
            "torch_prefix": args.require_runtime_torch_prefix,
            "transformers_required": args.require_runtime_transformers,
            "bitsandbytes_required": args.require_runtime_bitsandbytes,
            "cpu_count_min": args.require_runtime_cpu_count_min,
            "system_memory_min_gb": args.require_runtime_system_memory_min_gb,
            "hf_endpoint": args.require_runtime_hf_endpoint,
        },
        "runtime_violations": runtime_violations,
        "edge_report_sections": edge_report_sections,
        "edge_report_render_status": edge_report_render_status,
        "mmlu_subtask_coverage_status": mmlu_subtask_coverage_status,
        "validator": validator,
        "paper_protocol_validator": paper_protocol_validator,
        "matrix_run_status": {
            "failed": run_failures,
            "skipped_external_not_configured": skipped_external,
            "skipped_existing": skipped_existing,
        },
        "matrix_run_report_status": run_report_status,
        "matrix_finalize_report_status": finalize_report_status,
        "prewarm_report_status": prewarm_report_status,
        "asset_audit_report_status": asset_audit_report_status,
        "asset_storage_status": asset_storage_status,
        "preflight_report_status": preflight_report_status,
        "handoff_manifest_status": handoff_manifest_status,
        "target_readiness_status": target_readiness_status,
        "optional_backends_report_status": optional_backends_report_status,
        "quant_runtime_report_status": quant_runtime_report_status,
        "target_paths_report_status": target_paths_report_status,
        "prepare_report_status": prepare_report_status,
        "prepare_metadata_report_status": prepare_metadata_report_status,
        "prepare_precompressed_report_status": prepare_precompressed_report_status,
        "finalize_ok": finalize_ok,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "output": str(output)}, indent=2))
    if not report["ok"]:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit final LLMCBench edge completion artifacts.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--root", default="/data/outputs/llmcbench_edge_4090")
    parser.add_argument("--output", default=None)
    parser.add_argument("--only-model", action="append", default=[], help="Repeatable model filter.")
    parser.add_argument("--only-method", action="append", default=[], help="Repeatable method filter.")
    parser.add_argument("--allow-unexpected", action="store_true")
    parser.add_argument("--allow-skipped-external", action="store_true")
    parser.add_argument("--require-finalize-report", action="store_true")
    parser.add_argument("--require-prewarm-report", action="store_true")
    parser.add_argument("--require-prewarm-all-models", action="store_true")
    parser.add_argument("--require-asset-audit-report", action="store_true")
    parser.add_argument("--require-non-rotational-storage", action="store_true")
    parser.add_argument("--require-preflight-report", action="store_true")
    parser.add_argument("--require-handoff-manifest", action="store_true")
    parser.add_argument("--require-target-readiness-report", action="store_true")
    parser.add_argument("--require-optional-backends-report", action="store_true")
    parser.add_argument("--require-quant-runtime-report", action="store_true")
    parser.add_argument("--require-target-paths-report", action="store_true")
    parser.add_argument("--require-prepare-report", action="store_true")
    parser.add_argument("--require-prepare-metadata-report", action="store_true")
    parser.add_argument("--require-prepare-precompressed-report", action="store_true")
    parser.add_argument("--require-runtime-python-prefix", default="")
    parser.add_argument("--require-runtime-gpu-name-contains", default="")
    parser.add_argument("--require-runtime-cuda-prefix", default="")
    parser.add_argument("--require-runtime-gpu-total-memory-min-gb", type=float, default=0.0)
    parser.add_argument("--require-runtime-gpu-count", type=int, default=0)
    parser.add_argument("--require-runtime-cudnn", action="store_true")
    parser.add_argument("--require-runtime-torch-prefix", default="")
    parser.add_argument("--require-runtime-transformers", action="store_true")
    parser.add_argument("--require-runtime-bitsandbytes", action="store_true")
    parser.add_argument("--require-runtime-cpu-count-min", type=int, default=0)
    parser.add_argument("--require-runtime-system-memory-min-gb", type=float, default=0.0)
    parser.add_argument("--require-runtime-hf-endpoint", default="")
    parser.add_argument("--require-code-match", action="store_true", help="Require resolved config code commit/branch/origin to match the current checkout.")
    parser.add_argument("--require-clean-worktree", action="store_true", help="Require resolved and current git dirty state to be clean.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output is None:
        args.output = str(Path(args.root).expanduser() / "completion_audit.json")
    audit(args)


if __name__ == "__main__":
    main()
