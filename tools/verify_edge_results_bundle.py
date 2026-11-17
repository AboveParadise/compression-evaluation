import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

EXPECTED_SCORE_METRICS = {
    "mmlu": "aggregate:acc,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
}

EXPECTED_COMPRESSION_BACKENDS = {
    "dense": {"source_checkpoint_reference", "precompressed_checkpoint"},
    "wanda": {"built_in_activation_pruning", "precompressed_checkpoint"},
    "sparsegpt": {"built_in_activation_pruning", "precompressed_checkpoint"},
    "gptq": {"auto_gptq", "precompressed_checkpoint"},
    "awq": {"autoawq", "precompressed_checkpoint"},
    "smoothquant": {"external_command", "precompressed_checkpoint"},
    "omniquant": {"external_command", "precompressed_checkpoint"},
}
QUANTIZED_METHODS = {"gptq", "awq", "smoothquant", "omniquant"}

try:
    from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS
except ModuleNotFoundError:
    try:
        from prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS
    except ModuleNotFoundError:
        DEFAULT_MMLU_SUBTASKS = [
            "mmlu_abstract_algebra",
            "mmlu_anatomy",
            "mmlu_astronomy",
            "mmlu_business_ethics",
            "mmlu_clinical_knowledge",
            "mmlu_college_biology",
            "mmlu_college_chemistry",
            "mmlu_college_computer_science",
            "mmlu_college_mathematics",
            "mmlu_college_medicine",
            "mmlu_college_physics",
            "mmlu_computer_security",
            "mmlu_conceptual_physics",
            "mmlu_econometrics",
            "mmlu_electrical_engineering",
            "mmlu_elementary_mathematics",
            "mmlu_formal_logic",
            "mmlu_global_facts",
            "mmlu_high_school_biology",
            "mmlu_high_school_chemistry",
            "mmlu_high_school_computer_science",
            "mmlu_high_school_european_history",
            "mmlu_high_school_geography",
            "mmlu_high_school_government_and_politics",
            "mmlu_high_school_macroeconomics",
            "mmlu_high_school_mathematics",
            "mmlu_high_school_microeconomics",
            "mmlu_high_school_physics",
            "mmlu_high_school_psychology",
            "mmlu_high_school_statistics",
            "mmlu_high_school_us_history",
            "mmlu_high_school_world_history",
            "mmlu_human_aging",
            "mmlu_human_sexuality",
            "mmlu_international_law",
            "mmlu_jurisprudence",
            "mmlu_logical_fallacies",
            "mmlu_machine_learning",
            "mmlu_management",
            "mmlu_marketing",
            "mmlu_medical_genetics",
            "mmlu_miscellaneous",
            "mmlu_moral_disputes",
            "mmlu_moral_scenarios",
            "mmlu_nutrition",
            "mmlu_philosophy",
            "mmlu_prehistory",
            "mmlu_professional_accounting",
            "mmlu_professional_law",
            "mmlu_professional_medicine",
            "mmlu_professional_psychology",
            "mmlu_public_relations",
            "mmlu_security_studies",
            "mmlu_sociology",
            "mmlu_us_foreign_policy",
            "mmlu_virology",
            "mmlu_world_religions",
        ]


REQUIRED_BUNDLE_FILES = [
    "RESULTS_BUNDLE_MANIFEST.json",
    "RESULTS_BUNDLE_README.md",
    "matrix_config_resolved.json",
    "edge_asset_audit.json",
    "edge_preflight_report.json",
    "edge_prewarm_report.json",
    "edge_handoff_manifest.json",
    "edge_target_paths_report.json",
    "edge_optional_backends_report.json",
    "edge_quant_runtime_report.json",
    "matrix_run_report.json",
    "matrix_finalize_report.json",
    "matrix_results.json",
    "matrix_results.csv",
    "edge_report.md",
    "edge_scorecard.json",
    "edge_scorecard.md",
    "completion_audit.json",
    "target_readiness.json",
    "tools/verify_edge_results_bundle.py",
    "tools/verify_edge_delivery.py",
    "tools/prewarm_lm_eval_tasks.py",
]
SELF_DESCRIBING_BUNDLE_FILES = {"RESULTS_BUNDLE_MANIFEST.json"}


def _effective_runtime_profile(config: Dict[str, Any], model_name: str, method: str):
    merged = dict(config.get("defaults", {})) if isinstance(config.get("defaults"), dict) else {}
    for section_name, key in (
        ("method_overrides", method),
        ("model_overrides", model_name),
        ("model_method_overrides", f"{model_name}/{method}"),
    ):
        section = config.get(section_name, {})
        override = section.get(key) if isinstance(section, dict) else None
        if isinstance(override, dict):
            merged.update(override)
    return merged.get("runtime_profile")


DEPLOYMENT_METRIC_FIELDS = [
    "tokens_per_second",
    "peak_memory_allocated_gb",
    "peak_memory_reserved_gb",
    "deployment_batch_size",
    "deployment_memory_limit_gb",
    "deployment_memory_measured_peak_gb",
]

RESOURCE_METRIC_FIELDS = [
    "storage_gb",
    "total_params",
    "sparsity",
    "flops_num",
]

TARGET_RUNTIME_REQUIREMENTS = {
    "python_prefix": "3.10",
    "cuda_prefix": "12.",
    "gpu_name_contains": "4090",
    "gpu_total_memory_min_gb": 23.0,
    "gpu_count": 1,
    "torch_prefix": "2.",
    "cpu_count_min": 8,
    "system_memory_min_gb": 240.0,
    "hf_endpoint": "https://hf-mirror.com",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _file_status(bundle_root: Path, item: Dict[str, Any]) -> Dict[str, Any]:
    rel = item.get("path")
    status = {
        "path": rel,
        "ok": False,
        "exists": False,
        "size_ok": False,
        "sha256_ok": False,
        "violations": [],
    }
    if not _safe_relative_path(rel):
        status["violations"].append("path is not a safe relative path")
        return status
    path = bundle_root / rel
    if not path.is_file():
        status["violations"].append("file is missing")
        return status
    status["exists"] = True
    expected_size = item.get("size_bytes")
    actual_size = path.stat().st_size
    status["expected_size_bytes"] = expected_size
    status["actual_size_bytes"] = actual_size
    if expected_size == actual_size:
        status["size_ok"] = True
    else:
        status["violations"].append("size mismatch")
    expected_sha = item.get("sha256")
    actual_sha = _sha256(path)
    status["expected_sha256"] = expected_sha
    status["actual_sha256"] = actual_sha
    if expected_sha == actual_sha:
        status["sha256_ok"] = True
    else:
        status["violations"].append("sha256 mismatch")
    status["ok"] = status["exists"] and status["size_ok"] and status["sha256_ok"]
    return status


def _json_ok(path: Path) -> Dict[str, Any]:
    status = {"path": str(path), "exists": path.is_file(), "ok": False, "violations": []}
    if not path.is_file():
        status["violations"].append("file is missing")
        return status
    try:
        payload = _load_json(path)
    except Exception as exc:
        status["violations"].append(f"invalid JSON: {type(exc).__name__}: {exc}")
        return status
    if not isinstance(payload, dict):
        status["violations"].append("JSON root is not an object")
        return status
    status["value"] = payload.get("ok")
    status["ok"] = payload.get("ok") is True
    if not status["ok"]:
        status["violations"].append("ok is not true")
    return status


def _completion_audit_status(path: Path) -> Dict[str, Any]:
    status = _json_ok(path)
    status.update(
        {
            "validator_exists": False,
            "validator_ok": False,
            "validator_payload_ok": None,
        }
    )
    if not path.is_file():
        return status
    try:
        payload = _load_json(path)
    except Exception:
        return status
    if not isinstance(payload, dict):
        return status
    validator = payload.get("validator")
    if isinstance(validator, dict):
        status["validator_exists"] = True
        status["validator_ok"] = validator.get("ok") is True
        status["validator_payload_ok"] = validator.get("payload_ok")
    if status["ok"] and not status["validator_ok"]:
        status["violations"].append("validator.ok is not true")
        status["ok"] = False
    return status


def _target_readiness_status(path: Path) -> Dict[str, Any]:
    status = _json_ok(path)
    status.update(
        {
            "ready_for_matrix": None,
            "completion_ok": None,
            "before_matrix_blocker_count": None,
        }
    )
    if not path.is_file():
        return status
    try:
        payload = _load_json(path)
    except Exception:
        return status
    if not isinstance(payload, dict):
        return status
    blockers = payload.get("before_matrix_blockers")
    status["ready_for_matrix"] = payload.get("ready_for_matrix")
    status["completion_ok"] = payload.get("completion_ok")
    status["before_matrix_blocker_count"] = len(blockers) if isinstance(blockers, list) else None
    if status["ok"] and payload.get("completion_ok") is not True:
        status["violations"].append("completion_ok is not true")
        status["ok"] = False
    if status["ok"] and isinstance(blockers, list) and blockers:
        status["violations"].append("before_matrix_blockers is not empty")
        status["ok"] = False
    return status


def _target_paths_report_status(path: Path) -> Dict[str, Any]:
    status = _json_ok(path)
    status.update(
        {
            "report_ok": status.get("value"),
            "dry_run": None,
            "directory_count": None,
            "file_count": None,
            "missing_file_keys": [],
            "failed_directories": [],
            "failed_files": [],
        }
    )
    if not path.is_file():
        return status
    try:
        payload = _load_json(path)
    except Exception:
        return status
    if not isinstance(payload, dict):
        return status
    status["dry_run"] = payload.get("dry_run")
    if payload.get("dry_run") is True:
        status["violations"].append("dry-run report is not final evidence")
        status["ok"] = False
    directories = payload.get("directories")
    files = payload.get("files")
    if not isinstance(directories, dict) or not directories:
        status["violations"].append("target paths directories are empty or missing")
        directories = {} if directories is None else directories
    if not isinstance(files, dict) or not files:
        status["violations"].append("target paths files are empty or missing")
        files = {} if files is None else files
    expected_file_keys = {"calibration_file", "prompt_file"}
    missing_file_keys = sorted(expected_file_keys - set(files)) if isinstance(files, dict) else sorted(expected_file_keys)
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
    if missing_file_keys:
        status["violations"].append("target paths files do not cover required seed files")
    if failed_directories:
        status["violations"].append("target path directory items failed")
    if failed_files:
        status["violations"].append("target path file items failed")
    if status["violations"]:
        status["ok"] = False
    return status


def _resource_metric_value(row: Dict[str, Any], field: str) -> Any:
    method = str(row.get("compression_method") or "").lower()
    if field == "storage_gb":
        if row.get("quantization_storage_real") is True:
            value = row.get("effective_storage_gb_quantized_approx")
            if _finite_number(value):
                return value
        if method in {"wanda", "sparsegpt"}:
            value = row.get("effective_storage_gb_sparse_approx")
            if _finite_number(value):
                return value
        return row.get("storage_gb")
    if field == "flops_num":
        if method in {"wanda", "sparsegpt"}:
            value = row.get("effective_flops_num_sparse_approx")
            if _finite_number(value):
                return value
        return row.get("flops_num")
    return row.get(field)


def _values_match(left: Any, right: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    if _finite_number(left) and _finite_number(right):
        return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)
    return left == right


def _scorecard_status(bundle_root: Path) -> Dict[str, Any]:
    scorecard_json = bundle_root / "edge_scorecard.json"
    scorecard_md = bundle_root / "edge_scorecard.md"
    results_path = bundle_root / "matrix_results.json"
    status = {
        "ok": False,
        "json_path": str(scorecard_json),
        "markdown_path": str(scorecard_md),
        "results_path": str(results_path),
        "model_status": [],
        "violations": [],
    }
    if not scorecard_json.is_file():
        status["violations"].append("edge_scorecard.json is missing")
        return status
    if not scorecard_md.is_file():
        status["violations"].append("edge_scorecard.md is missing")
        return status
    try:
        payload = _load_json(scorecard_json)
    except Exception as exc:
        status["violations"].append(f"edge_scorecard.json is not valid JSON: {type(exc).__name__}: {exc}")
        return status
    try:
        results = _load_json(results_path)
    except Exception as exc:
        status["violations"].append(f"matrix_results.json is not valid JSON: {type(exc).__name__}: {exc}")
        return status
    if not isinstance(payload, dict):
        status["violations"].append("edge_scorecard.json root is not an object")
        return status
    if not isinstance(results, list):
        status["violations"].append("matrix_results.json root is not a list")
        return status
    result_methods: Dict[str, Set[str]] = {}
    results_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        model_name = row.get("model_name")
        method = row.get("compression_method")
        if isinstance(model_name, str) and isinstance(method, str):
            result_methods.setdefault(model_name, set()).add(method)
            results_by_pair[(model_name, method)] = row
    if payload.get("schema_version") != 1:
        status["violations"].append("edge_scorecard.json schema_version must be 1")
    scoring = payload.get("scoring")
    if not isinstance(scoring, dict):
        status["violations"].append("edge_scorecard.json scoring metadata is missing")
    else:
        preference = scoring.get("resource_metric_preference")
        if not isinstance(preference, dict):
            status["violations"].append("edge_scorecard.json resource metric preference is missing")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        status["violations"].append("edge_scorecard.json models must be a non-empty list")
    else:
        for model in models:
            model_status = {
                "model": model.get("model_name") if isinstance(model, dict) else None,
                "ok": False,
                "expected_methods": [],
                "ranking_methods": [],
                "metric_mismatches": [],
                "violations": [],
            }
            if not isinstance(model, dict):
                status["violations"].append("edge_scorecard.json contains a non-object model entry")
                model_status["violations"].append("model entry is not an object")
                status["model_status"].append(model_status)
                continue
            model_name = model.get("model_name")
            ranking = model.get("ranking")
            if not isinstance(model_name, str) or not model_name:
                model_status["violations"].append("model_name is missing")
            if not model.get("recommended_method"):
                model_status["violations"].append("recommended_method is missing")
            if not isinstance(ranking, list) or not ranking:
                model_status["violations"].append("ranking is empty")
                status["violations"].append("edge_scorecard.json model ranking is empty")
            else:
                ranking_methods = [
                    item.get("compression_method")
                    for item in ranking
                    if isinstance(item, dict) and isinstance(item.get("compression_method"), str)
                ]
                expected = sorted(result_methods.get(model_name, set()))
                model_status["expected_methods"] = expected
                model_status["ranking_methods"] = sorted(ranking_methods)
                if sorted(ranking_methods) != expected:
                    model_status["violations"].append("ranking methods do not match matrix_results methods")
                recommended = model.get("recommended_method")
                if recommended and recommended not in ranking_methods:
                    model_status["violations"].append("recommended_method is not present in ranking")
                for item in ranking:
                    if not isinstance(item, dict):
                        model_status["violations"].append("ranking entry is not an object")
                        continue
                    method = item.get("compression_method")
                    if not isinstance(model_name, str) or not isinstance(method, str):
                        model_status["violations"].append("ranking entry is missing model or method")
                        continue
                    source_row = results_by_pair.get((model_name, method))
                    if source_row is None:
                        continue
                    comparisons = {
                        "accuracy_avg_score": source_row.get("accuracy_avg_score"),
                        "tokens_per_second": source_row.get("tokens_per_second"),
                        "deployment_memory_measured_peak_gb": source_row.get("deployment_memory_measured_peak_gb"),
                        "storage_gb": source_row.get("storage_gb"),
                        "score_storage_gb": _resource_metric_value(source_row, "storage_gb"),
                        "flops_num": source_row.get("flops_num"),
                        "score_flops_num": _resource_metric_value(source_row, "flops_num"),
                        "accuracy_retention_vs_dense": source_row.get("accuracy_retention_vs_dense"),
                        "resource_backend": source_row.get("resource_backend"),
                        "compute_backend": source_row.get("compute_backend"),
                    }
                    for field, expected_value in comparisons.items():
                        if field in item and not _values_match(item.get(field), expected_value):
                            model_status["metric_mismatches"].append(
                                {
                                    "method": method,
                                    "field": field,
                                    "scorecard_value": item.get(field),
                                    "matrix_results_value": expected_value,
                                }
                            )
                if model_status["metric_mismatches"]:
                    model_status["violations"].append("ranking metrics do not match matrix_results")
            if model_status["violations"]:
                status["violations"].append("edge_scorecard.json model does not match matrix_results")
            model_status["ok"] = not model_status["violations"]
            status["model_status"].append(model_status)
    text = scorecard_md.read_text(encoding="utf-8", errors="replace")
    if "LLMCBench Edge Result Scorecard" not in text:
        status["violations"].append("edge_scorecard.md is missing scorecard title")
    status["ok"] = not status["violations"]
    return status


def _matrix_coverage_status(bundle_root: Path, config_path: Path) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "ok": False,
        "config": str(config_path),
        "results": str(bundle_root / "matrix_results.json"),
        "expected_pair_count": None,
        "actual_pair_count": None,
        "missing_pairs": [],
        "unexpected_pairs": [],
        "duplicate_pairs": [],
        "violations": [],
    }
    try:
        config = _load_json(config_path)
    except Exception as exc:
        status["violations"].append(f"config is not valid JSON: {type(exc).__name__}: {exc}")
        return status
    try:
        rows = _load_json(bundle_root / "matrix_results.json")
    except Exception as exc:
        status["violations"].append(f"matrix_results.json is not valid JSON: {type(exc).__name__}: {exc}")
        return status
    if not isinstance(config, dict):
        status["violations"].append("config root is not an object")
        return status
    if not isinstance(rows, list):
        status["violations"].append("matrix_results.json root is not a list")
        return status
    if isinstance(config.get("selected_models"), list):
        models = [str(model) for model in config.get("selected_models", []) if model]
    else:
        models = [str(item.get("name")) for item in config.get("models", []) if isinstance(item, dict) and item.get("name")]
    if isinstance(config.get("selected_methods"), list):
        methods = [str(method) for method in config.get("selected_methods", []) if method]
    else:
        methods = [str(method) for method in config.get("methods", []) if method]
    expected: Set[Tuple[str, str]] = {(model, method) for model in models for method in methods}
    actual_pairs: List[Tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            status["violations"].append("matrix_results.json contains a non-object row")
            continue
        model = row.get("model_name")
        method = row.get("compression_method")
        if isinstance(model, str) and isinstance(method, str):
            actual_pairs.append((model, method))
        else:
            status["violations"].append("matrix_results.json row is missing model_name or compression_method")
    actual = set(actual_pairs)
    duplicates = sorted({pair for pair in actual_pairs if actual_pairs.count(pair) > 1})
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    status.update(
        {
            "expected_pair_count": len(expected),
            "actual_pair_count": len(actual_pairs),
            "missing_pairs": [{"model": model, "method": method} for model, method in missing],
            "unexpected_pairs": [{"model": model, "method": method} for model, method in unexpected],
            "duplicate_pairs": [{"model": model, "method": method} for model, method in duplicates],
        }
    )
    if missing:
        status["violations"].append("matrix_results.json is missing expected model/method pairs")
    if unexpected:
        status["violations"].append("matrix_results.json contains unexpected model/method pairs")
    if duplicates:
        status["violations"].append("matrix_results.json contains duplicate model/method pairs")
    status["ok"] = not status["violations"]
    return status


def _configured_tasks(config: Dict[str, Any]) -> List[str]:
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    tasks = defaults.get("tasks", "")
    if not isinstance(tasks, str):
        return []
    return [task.strip() for task in tasks.split(",") if task.strip()]


def _score_coverage_status(bundle_root: Path, config_path: Path) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "ok": False,
        "config": str(config_path),
        "results": str(bundle_root / "matrix_results.json"),
        "tasks": [],
        "required_score_fields": [],
        "required_mmlu_subtask_count": 0,
        "rows": [],
        "violations": [],
    }
    try:
        config = _load_json(config_path)
    except Exception as exc:
        status["violations"].append(f"config is not valid JSON: {type(exc).__name__}: {exc}")
        return status
    try:
        rows = _load_json(bundle_root / "matrix_results.json")
    except Exception as exc:
        status["violations"].append(f"matrix_results.json is not valid JSON: {type(exc).__name__}: {exc}")
        return status
    if not isinstance(config, dict):
        status["violations"].append("config root is not an object")
        return status
    if not isinstance(rows, list):
        status["violations"].append("matrix_results.json root is not a list")
        return status
    tasks = _configured_tasks(config)
    score_fields = [f"{task}_score" for task in tasks]
    score_metric_fields = [f"{task}_score_metric" for task in tasks]
    requires_mmlu_subtasks = "mmlu" in tasks
    mmlu_fields = [f"{task}_score" for task in DEFAULT_MMLU_SUBTASKS] if requires_mmlu_subtasks else []
    mmlu_metric_fields = [f"{task}_score_metric" for task in DEFAULT_MMLU_SUBTASKS] if requires_mmlu_subtasks else []
    status["tasks"] = tasks
    status["required_score_fields"] = score_fields
    status["required_score_metric_fields"] = score_metric_fields
    status["required_mmlu_subtask_count"] = len(mmlu_fields)
    for index, row in enumerate(rows):
        row_status: Dict[str, Any] = {
            "index": index,
            "model": row.get("model_name") if isinstance(row, dict) else None,
            "method": row.get("compression_method") if isinstance(row, dict) else None,
            "missing_score_fields": [],
            "missing_score_metric_fields": [],
            "score_metric_mismatches": [],
            "score_aggregate_mismatches": [],
            "missing_mmlu_subtasks": [],
            "missing_mmlu_subtask_metric_fields": [],
            "mmlu_subtask_metric_mismatches": [],
            "accuracy_provenance_violations": [],
            "ok": False,
        }
        if not isinstance(row, dict):
            row_status["missing_score_fields"] = score_fields
            row_status["missing_score_metric_fields"] = score_metric_fields
            row_status["missing_mmlu_subtasks"] = mmlu_fields
            row_status["missing_mmlu_subtask_metric_fields"] = mmlu_metric_fields
            row_status["accuracy_provenance_violations"] = ["row"]
            status["rows"].append(row_status)
            status["violations"].append("matrix_results.json contains a non-object row")
            continue
        row_status["missing_score_fields"] = [
            field
            for field in score_fields
            if not isinstance(row.get(field), (int, float)) or isinstance(row.get(field), bool)
        ]
        row_status["missing_score_metric_fields"] = [
            field
            for field in score_metric_fields
            if not isinstance(row.get(field), str) or not row.get(field)
        ]
        row_status["score_metric_mismatches"] = [
            {"field": f"{task}_score_metric", "value": row.get(f"{task}_score_metric"), "expected": expected}
            for task, expected in EXPECTED_SCORE_METRICS.items()
            if task in tasks
            and isinstance(row.get(f"{task}_score_metric"), str)
            and row.get(f"{task}_score_metric") != expected
        ]
        row_status["missing_mmlu_subtasks"] = [
            field
            for field in mmlu_fields
            if not isinstance(row.get(field), (int, float)) or isinstance(row.get(field), bool)
        ]
        row_status["missing_mmlu_subtask_metric_fields"] = [
            field
            for field in mmlu_metric_fields
            if not isinstance(row.get(field), str) or not row.get(field)
        ]
        row_status["mmlu_subtask_metric_mismatches"] = [
            {"field": field, "value": row.get(field), "expected": "acc,none"}
            for field in mmlu_metric_fields
            if isinstance(row.get(field), str) and row.get(field) != "acc,none"
        ]
        if requires_mmlu_subtasks:
            mmlu_values = [row.get(field) for field in mmlu_fields if _finite_number(row.get(field))]
            mmlu_score = row.get("mmlu_score")
            if len(mmlu_values) == len(mmlu_fields) and _finite_number(mmlu_score):
                expected_mmlu = sum(float(value) for value in mmlu_values) / len(mmlu_values)
                if abs(float(mmlu_score) - expected_mmlu) > 1e-8 * max(1.0, abs(expected_mmlu), abs(float(mmlu_score))):
                    row_status["score_aggregate_mismatches"].append(
                        {"field": "mmlu_score", "value": mmlu_score, "expected": expected_mmlu}
                    )
        task_values = [row.get(field) for field in score_fields if _finite_number(row.get(field))]
        accuracy_avg = row.get("accuracy_avg_score")
        if len(task_values) == len(score_fields) and _finite_number(accuracy_avg):
            expected_avg = sum(float(value) for value in task_values) / len(task_values)
            if abs(float(accuracy_avg) - expected_avg) > 1e-8 * max(1.0, abs(expected_avg), abs(float(accuracy_avg))):
                row_status["score_aggregate_mismatches"].append(
                    {"field": "accuracy_avg_score", "value": accuracy_avg, "expected": expected_avg}
                )
        if row.get("accuracy_limit") not in (None, ""):
            row_status["accuracy_provenance_violations"].append("accuracy_limit")
        if row.get("accuracy_failed_task_count") != 0:
            row_status["accuracy_provenance_violations"].append("accuracy_failed_task_count")
        if row.get("accuracy_failed_tasks") not in ("", None):
            row_status["accuracy_provenance_violations"].append("accuracy_failed_tasks")
        row_status["ok"] = (
            not row_status["missing_score_fields"]
            and not row_status["missing_score_metric_fields"]
            and not row_status["score_metric_mismatches"]
            and not row_status["score_aggregate_mismatches"]
            and not row_status["missing_mmlu_subtasks"]
            and not row_status["missing_mmlu_subtask_metric_fields"]
            and not row_status["mmlu_subtask_metric_mismatches"]
            and not row_status["accuracy_provenance_violations"]
        )
        if row_status["missing_score_fields"]:
            status["violations"].append("matrix_results.json row is missing target task score fields")
        if row_status["missing_score_metric_fields"]:
            status["violations"].append("matrix_results.json row is missing target task score metric fields")
        if row_status["score_metric_mismatches"]:
            status["violations"].append("matrix_results.json row has unexpected target task score metric keys")
        if row_status["score_aggregate_mismatches"]:
            status["violations"].append("matrix_results.json row has inconsistent aggregate accuracy scores")
        if row_status["missing_mmlu_subtasks"]:
            status["violations"].append("matrix_results.json row is missing MMLU subtask score fields")
        if row_status["missing_mmlu_subtask_metric_fields"]:
            status["violations"].append("matrix_results.json row is missing MMLU subtask score metric fields")
        if row_status["mmlu_subtask_metric_mismatches"]:
            status["violations"].append("matrix_results.json row has unexpected MMLU subtask score metric keys")
        if row_status["accuracy_provenance_violations"]:
            status["violations"].append("matrix_results.json row has non-final accuracy provenance")
        status["rows"].append(row_status)
    status["ok"] = not status["violations"]
    return status


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _metric_coverage_status(bundle_root: Path) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "ok": False,
        "results": str(bundle_root / "matrix_results.json"),
        "deployment_metric_fields": DEPLOYMENT_METRIC_FIELDS,
        "resource_metric_fields": RESOURCE_METRIC_FIELDS,
        "rows": [],
        "violations": [],
    }
    try:
        rows = _load_json(bundle_root / "matrix_results.json")
    except Exception as exc:
        status["violations"].append(f"matrix_results.json is not valid JSON: {type(exc).__name__}: {exc}")
        return status
    if not isinstance(rows, list):
        status["violations"].append("matrix_results.json root is not a list")
        return status
    for index, row in enumerate(rows):
        row_status: Dict[str, Any] = {
            "index": index,
            "model": row.get("model_name") if isinstance(row, dict) else None,
            "method": row.get("compression_method") if isinstance(row, dict) else None,
            "missing_or_invalid_deployment_metrics": [],
            "missing_or_invalid_resource_metrics": [],
            "memory_limit_violations": [],
            "deployment_consistency_violations": [],
            "invalid_backend_fields": [],
            "resource_consistency_violations": [],
            "quantized_storage_violations": [],
            "ok": False,
        }
        if not isinstance(row, dict):
            row_status["missing_or_invalid_deployment_metrics"] = list(DEPLOYMENT_METRIC_FIELDS)
            row_status["missing_or_invalid_resource_metrics"] = list(RESOURCE_METRIC_FIELDS)
            row_status["memory_limit_violations"] = ["deployment_memory_limit_gb"]
            row_status["deployment_consistency_violations"] = ["row"]
            row_status["invalid_backend_fields"] = ["resource_backend", "compute_backend"]
            row_status["resource_consistency_violations"] = ["row"]
            row_status["quantized_storage_violations"] = ["row"]
            status["rows"].append(row_status)
            status["violations"].append("matrix_results.json contains a non-object row")
            continue
        row_status["missing_or_invalid_deployment_metrics"] = [
            field for field in DEPLOYMENT_METRIC_FIELDS if not _finite_number(row.get(field))
        ]
        row_status["missing_or_invalid_resource_metrics"] = [
            field for field in RESOURCE_METRIC_FIELDS if not _finite_number(row.get(field))
        ]
        if row.get("tokens_per_second") is not None and _finite_number(row.get("tokens_per_second")) and row.get("tokens_per_second") <= 0:
            row_status["missing_or_invalid_deployment_metrics"].append("tokens_per_second")
        if row.get("deployment_memory_measured_peak_gb") is not None and _finite_number(row.get("deployment_memory_measured_peak_gb")) and row.get("deployment_memory_measured_peak_gb") < 0:
            row_status["missing_or_invalid_deployment_metrics"].append("deployment_memory_measured_peak_gb")
        generated_tokens = row.get("deployment_generated_tokens")
        elapsed_seconds = row.get("deployment_elapsed_seconds")
        tokens_per_second = row.get("tokens_per_second")
        if _finite_number(generated_tokens) and _finite_number(elapsed_seconds) and _finite_number(tokens_per_second) and elapsed_seconds > 0:
            expected_tps = float(generated_tokens) / float(elapsed_seconds)
            if abs(float(tokens_per_second) - expected_tps) > 1e-8 * max(1.0, abs(expected_tps), abs(float(tokens_per_second))):
                row_status["deployment_consistency_violations"].append("tokens_per_second")
        allocated = row.get("peak_memory_allocated_gb")
        reserved = row.get("peak_memory_reserved_gb")
        measured_peak = row.get("deployment_memory_measured_peak_gb")
        if _finite_number(allocated) and _finite_number(reserved) and _finite_number(measured_peak):
            expected_peak = max(float(allocated), float(reserved))
            if abs(float(measured_peak) - expected_peak) > 1e-8 * max(1.0, abs(expected_peak), abs(float(measured_peak))):
                row_status["deployment_consistency_violations"].append("deployment_memory_measured_peak_gb")
        limit = row.get("deployment_memory_limit_gb")
        limit_exceeded = row.get("deployment_memory_limit_exceeded")
        if _finite_number(limit) and limit > 0 and _finite_number(measured_peak):
            expected_exceeded = measured_peak > limit
            if expected_exceeded:
                row_status["memory_limit_violations"].append("deployment_memory_measured_peak_gb")
            if limit_exceeded is not expected_exceeded:
                row_status["memory_limit_violations"].append("deployment_memory_limit_exceeded")
        elif limit not in (None, 0):
            row_status["memory_limit_violations"].append("deployment_memory_limit_gb")
        if row.get("storage_gb") is not None and _finite_number(row.get("storage_gb")) and row.get("storage_gb") <= 0:
            row_status["missing_or_invalid_resource_metrics"].append("storage_gb")
        if row.get("total_params") is not None and _finite_number(row.get("total_params")) and row.get("total_params") <= 0:
            row_status["missing_or_invalid_resource_metrics"].append("total_params")
        if row.get("flops_num") is not None and _finite_number(row.get("flops_num")) and row.get("flops_num") <= 0:
            row_status["missing_or_invalid_resource_metrics"].append("flops_num")
        if not isinstance(row.get("resource_backend"), str) or row.get("resource_backend") == "config_only_fallback":
            row_status["invalid_backend_fields"].append("resource_backend")
        if not isinstance(row.get("compute_backend"), str) or not row.get("compute_backend"):
            row_status["invalid_backend_fields"].append("compute_backend")
        storage_bytes = row.get("storage_bytes")
        storage_gb = row.get("storage_gb")
        if _finite_number(storage_bytes) and _finite_number(storage_gb):
            expected_storage_gb = float(storage_bytes) / (1024**3)
            if abs(float(storage_gb) - expected_storage_gb) > 1e-8 * max(1.0, abs(expected_storage_gb), abs(float(storage_gb))):
                row_status["resource_consistency_violations"].append("storage_gb")
        nonzero_ratio = row.get("nonzero_ratio")
        sparse_storage_bytes = row.get("effective_storage_bytes_sparse_approx")
        if _finite_number(storage_bytes) and _finite_number(nonzero_ratio) and _finite_number(sparse_storage_bytes):
            expected_sparse_storage = float(storage_bytes) * float(nonzero_ratio)
            if abs(float(sparse_storage_bytes) - expected_sparse_storage) > 1e-8 * max(1.0, abs(expected_sparse_storage), abs(float(sparse_storage_bytes))):
                row_status["resource_consistency_violations"].append("effective_storage_bytes_sparse_approx")
        sparse_storage_gb = row.get("effective_storage_gb_sparse_approx")
        if _finite_number(sparse_storage_bytes) and _finite_number(sparse_storage_gb):
            expected_sparse_storage_gb = float(sparse_storage_bytes) / (1024**3)
            if abs(float(sparse_storage_gb) - expected_sparse_storage_gb) > 1e-8 * max(1.0, abs(expected_sparse_storage_gb), abs(float(sparse_storage_gb))):
                row_status["resource_consistency_violations"].append("effective_storage_gb_sparse_approx")
        flops_num = row.get("flops_num")
        sparse_flops = row.get("effective_flops_num_sparse_approx")
        if _finite_number(flops_num) and _finite_number(nonzero_ratio) and _finite_number(sparse_flops):
            expected_sparse_flops = float(flops_num) * float(nonzero_ratio)
            if abs(float(sparse_flops) - expected_sparse_flops) > 1e-8 * max(1.0, abs(expected_sparse_flops), abs(float(sparse_flops))):
                row_status["resource_consistency_violations"].append("effective_flops_num_sparse_approx")
        method = str(row.get("compression_method", "")).lower()
        quantized_fields = [
            "effective_storage_bytes_quantized_approx",
            "effective_storage_gb_quantized_approx",
            "effective_storage_quantized_bits",
            "effective_storage_quantized_backend",
        ]
        if method in QUANTIZED_METHODS:
            storage_real = row.get("quantization_storage_real")
            storage_format = row.get("quantization_storage_format")
            if not isinstance(storage_real, bool):
                row_status["quantized_storage_violations"].append("quantization_storage_real")
            if not isinstance(storage_format, str) or not storage_format:
                row_status["quantized_storage_violations"].append("quantization_storage_format")
            for field in quantized_fields:
                value = row.get(field)
                if storage_real:
                    if field == "effective_storage_quantized_backend":
                        if value not in {"checkpoint_directory_size", "param_count_times_quant_bits"}:
                            row_status["quantized_storage_violations"].append(field)
                    elif not _finite_number(value) or value < 0:
                        row_status["quantized_storage_violations"].append(field)
                elif value not in (None, ""):
                    row_status["quantized_storage_violations"].append(field)
            quant_bits = row.get("compression_quant_bits")
            quantized_bytes = row.get("effective_storage_bytes_quantized_approx")
            storage_backend = row.get("effective_storage_quantized_backend")
            if storage_real and storage_backend == "checkpoint_directory_size":
                checkpoint_bytes = row.get("storage_bytes")
                if _finite_number(checkpoint_bytes) and _finite_number(quantized_bytes):
                    if abs(float(quantized_bytes) - float(checkpoint_bytes)) > 1e-8 * max(1.0, abs(float(checkpoint_bytes)), abs(float(quantized_bytes))):
                        row_status["quantized_storage_violations"].append("effective_storage_bytes_quantized_approx")
            elif storage_real and storage_backend == "param_count_times_quant_bits":
                total_params = row.get("total_params")
                if _finite_number(total_params) and _finite_number(quant_bits) and _finite_number(quantized_bytes):
                    expected_quantized_bytes = float(total_params) * float(quant_bits) / 8.0
                    if abs(float(quantized_bytes) - expected_quantized_bytes) > 1e-8 * max(1.0, abs(expected_quantized_bytes), abs(float(quantized_bytes))):
                        row_status["quantized_storage_violations"].append("effective_storage_bytes_quantized_approx")
            quantized_gb = row.get("effective_storage_gb_quantized_approx")
            if storage_real and _finite_number(quantized_bytes) and _finite_number(quantized_gb):
                expected_quantized_gb = float(quantized_bytes) / (1024**3)
                if abs(float(quantized_gb) - expected_quantized_gb) > 1e-8 * max(1.0, abs(expected_quantized_gb), abs(float(quantized_gb))):
                    row_status["quantized_storage_violations"].append("effective_storage_gb_quantized_approx")
            storage_bits = row.get("effective_storage_quantized_bits")
            if storage_real and _finite_number(quant_bits) and _finite_number(storage_bits) and storage_bits != quant_bits:
                row_status["quantized_storage_violations"].append("effective_storage_quantized_bits")
        else:
            for field in quantized_fields:
                if row.get(field) not in (None, ""):
                    row_status["quantized_storage_violations"].append(field)
        row_status["missing_or_invalid_deployment_metrics"] = sorted(set(row_status["missing_or_invalid_deployment_metrics"]))
        row_status["missing_or_invalid_resource_metrics"] = sorted(set(row_status["missing_or_invalid_resource_metrics"]))
        row_status["memory_limit_violations"] = sorted(set(row_status["memory_limit_violations"]))
        row_status["deployment_consistency_violations"] = sorted(set(row_status["deployment_consistency_violations"]))
        row_status["invalid_backend_fields"] = sorted(set(row_status["invalid_backend_fields"]))
        row_status["resource_consistency_violations"] = sorted(set(row_status["resource_consistency_violations"]))
        row_status["quantized_storage_violations"] = sorted(set(row_status["quantized_storage_violations"]))
        row_status["ok"] = (
            not row_status["missing_or_invalid_deployment_metrics"]
            and not row_status["missing_or_invalid_resource_metrics"]
            and not row_status["memory_limit_violations"]
            and not row_status["deployment_consistency_violations"]
            and not row_status["invalid_backend_fields"]
            and not row_status["resource_consistency_violations"]
            and not row_status["quantized_storage_violations"]
        )
        if row_status["missing_or_invalid_deployment_metrics"]:
            status["violations"].append("matrix_results.json row is missing or has invalid deployment metrics")
        if row_status["missing_or_invalid_resource_metrics"]:
            status["violations"].append("matrix_results.json row is missing or has invalid resource metrics")
        if row_status["memory_limit_violations"]:
            status["violations"].append("matrix_results.json row violates deployment memory limit")
        if row_status["deployment_consistency_violations"]:
            status["violations"].append("matrix_results.json row has inconsistent deployment metrics")
        if row_status["invalid_backend_fields"]:
            status["violations"].append("matrix_results.json row has invalid backend fields")
        if row_status["resource_consistency_violations"]:
            status["violations"].append("matrix_results.json row has inconsistent resource estimates")
        if row_status["quantized_storage_violations"]:
            status["violations"].append("matrix_results.json row has invalid quantized storage estimates")
        status["rows"].append(row_status)
    status["ok"] = not status["violations"]
    return status


def _compression_provenance_status(bundle_root: Path) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "ok": False,
        "results": str(bundle_root / "matrix_results.json"),
        "rows": [],
        "violations": [],
    }
    try:
        rows = _load_json(bundle_root / "matrix_results.json")
    except Exception as exc:
        status["violations"].append(f"matrix_results.json is not valid JSON: {type(exc).__name__}: {exc}")
        return status
    if not isinstance(rows, list):
        status["violations"].append("matrix_results.json root is not a list")
        return status
    for index, row in enumerate(rows):
        row_status: Dict[str, Any] = {
            "index": index,
            "model": row.get("model_name") if isinstance(row, dict) else None,
            "method": row.get("compression_method") if isinstance(row, dict) else None,
            "violations": [],
            "ok": False,
        }
        if not isinstance(row, dict):
            row_status["violations"].append("row is not an object")
            status["rows"].append(row_status)
            status["violations"].append("matrix_results.json contains a non-object row")
            continue
        method = str(row.get("compression_method", "")).lower()
        backend = row.get("compression_backend")
        allowed = EXPECTED_COMPRESSION_BACKENDS.get(method)
        if not isinstance(backend, str) or not backend:
            row_status["violations"].append("compression_backend is missing")
        elif allowed and backend not in allowed:
            row_status["violations"].append("compression_backend is not allowed for method")
        uses_precompressed = row.get("compression_uses_precompressed")
        if not isinstance(uses_precompressed, bool) or uses_precompressed != (backend == "precompressed_checkpoint"):
            row_status["violations"].append("compression_uses_precompressed is inconsistent")
        materialized = row.get("compression_materialized")
        expected_materialized = backend not in {"source_checkpoint_reference", "precompressed_checkpoint"}
        if not isinstance(materialized, bool) or materialized != expected_materialized:
            row_status["violations"].append("compression_materialized is inconsistent")
        if backend == "external_command":
            if not isinstance(row.get("compression_external_command"), str) or not row.get("compression_external_command"):
                row_status["violations"].append("compression_external_command is missing for external backend")
            if row.get("compression_external_returncode") != 0:
                row_status["violations"].append("compression_external_returncode is not zero for external backend")
        row_status["ok"] = not row_status["violations"]
        if row_status["violations"]:
            status["violations"].append("matrix_results.json row has invalid compression provenance")
        status["rows"].append(row_status)
    status["ok"] = not status["violations"]
    return status


def _csv_consistency_status(bundle_root: Path) -> Dict[str, Any]:
    json_path = bundle_root / "matrix_results.json"
    csv_path = bundle_root / "matrix_results.csv"
    status: Dict[str, Any] = {
        "ok": False,
        "json": str(json_path),
        "csv": str(csv_path),
        "header": [],
        "expected_columns": [],
        "missing_columns": [],
        "row_count_matches": False,
        "pair_set_matches": False,
        "value_mismatches": [],
        "violations": [],
    }
    try:
        rows = _load_json(json_path)
    except Exception as exc:
        status["violations"].append(f"matrix_results.json is not valid JSON: {type(exc).__name__}: {exc}")
        return status
    if not isinstance(rows, list):
        status["violations"].append("matrix_results.json root is not a list")
        return status
    if not csv_path.is_file():
        status["violations"].append("matrix_results.csv is missing")
        return status
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            csv_rows = list(reader)
    except Exception as exc:
        status["violations"].append(f"matrix_results.csv could not be read: {type(exc).__name__}: {exc}")
        return status
    json_columns = sorted({key for row in rows if isinstance(row, dict) for key in row.keys()})
    required_columns = sorted(set(json_columns) | {"model_name", "compression_method"})
    missing_columns = [column for column in required_columns if column not in header]
    json_pairs = [
        (row.get("model_name"), row.get("compression_method"))
        for row in rows
        if isinstance(row, dict)
    ]
    csv_pairs = [(row.get("model_name"), row.get("compression_method")) for row in csv_rows]
    csv_rows_by_pair = {pair: row for pair, row in zip(csv_pairs, csv_rows)}
    value_mismatches = []
    if not missing_columns:
        for row in rows:
            if not isinstance(row, dict):
                continue
            pair = (row.get("model_name"), row.get("compression_method"))
            csv_row = csv_rows_by_pair.get(pair)
            if not isinstance(csv_row, dict):
                continue
            for column in json_columns:
                expected = "" if row.get(column) is None else str(row.get(column))
                actual = csv_row.get(column)
                if actual != expected:
                    value_mismatches.append(
                        {
                            "model": pair[0],
                            "method": pair[1],
                            "column": column,
                            "csv_value": actual,
                            "json_value": row.get(column),
                            "expected_csv_value": expected,
                        }
                    )
    status.update(
        {
            "header": header,
            "expected_columns": required_columns,
            "missing_columns": missing_columns,
            "json_row_count": len(rows),
            "csv_row_count": len(csv_rows),
            "row_count_matches": len(rows) == len(csv_rows),
            "pair_set_matches": sorted(json_pairs) == sorted(csv_pairs),
            "value_mismatches": value_mismatches,
        }
    )
    if missing_columns:
        status["violations"].append("matrix_results.csv is missing JSON result columns")
    if not status["row_count_matches"]:
        status["violations"].append("matrix_results.csv row count does not match JSON")
    if not status["pair_set_matches"]:
        status["violations"].append("matrix_results.csv model/method pairs do not match JSON")
    if value_mismatches:
        status["violations"].append("matrix_results.csv values do not match JSON")
    status["ok"] = not status["violations"]
    return status


def _runtime_environment_status(bundle_root: Path) -> Dict[str, Any]:
    requirements = dict(TARGET_RUNTIME_REQUIREMENTS)
    status: Dict[str, Any] = {
        "ok": False,
        "requirements": requirements,
        "runtime_profiles": [],
        "rows": [],
        "violations": [],
    }
    try:
        rows = _load_json(bundle_root / "matrix_results.json")
    except Exception as exc:
        status["violations"].append(f"matrix_results.json is not valid JSON: {type(exc).__name__}: {exc}")
        return status
    if not isinstance(rows, list):
        status["violations"].append("matrix_results.json root is not a list")
        return status
    expected_cuda_visible_devices = ""
    config = {}
    try:
        config = _load_json(bundle_root / "matrix_config_resolved.json")
        if isinstance(config, dict) and isinstance(config.get("defaults"), dict):
            expected_cuda_visible_devices = config["defaults"].get("cuda_visible_devices") or ""
    except Exception:
        expected_cuda_visible_devices = ""
    requirements["cuda_visible_devices"] = expected_cuda_visible_devices
    configured_runtime_profiles = []
    if isinstance(config, dict):
        configured_runtime_profiles = [
            _effective_runtime_profile(config, row.get("model_name"), row.get("compression_method"))
            for row in rows
            if isinstance(row, dict)
        ]
    runtime_profiles = sorted(
        {
            row.get("runtime_profile")
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("runtime_profile"), str) and row.get("runtime_profile")
        }
    )
    status["runtime_profiles"] = runtime_profiles
    if len(runtime_profiles) > 1 and not any(profile not in (None, "") for profile in configured_runtime_profiles):
        status["violations"].append("matrix_results.json mixes runtime_profile values")
    for index, row in enumerate(rows):
        row_status: Dict[str, Any] = {
            "index": index,
            "model": row.get("model_name") if isinstance(row, dict) else None,
            "method": row.get("compression_method") if isinstance(row, dict) else None,
            "runtime_profile": row.get("runtime_profile") if isinstance(row, dict) else None,
            "violations": [],
            "ok": False,
        }
        if not isinstance(row, dict):
            row_status["violations"].append("row is not an object")
            status["rows"].append(row_status)
            status["violations"].append("matrix_results.json contains a non-object row")
            continue
        expected_runtime_profile = _effective_runtime_profile(
            config if isinstance(config, dict) else {},
            row.get("model_name"),
            row.get("compression_method"),
        )
        if expected_runtime_profile not in (None, "") and row.get("runtime_profile") != expected_runtime_profile:
            row_status["violations"].append("runtime_profile does not match configured profile")
        runtime_python = row.get("runtime_python")
        if not isinstance(runtime_python, str) or not runtime_python.startswith(requirements["python_prefix"]):
            row_status["violations"].append("runtime_python does not match required prefix")
        cuda_available = row.get("runtime_cuda_available")
        if cuda_available is not True:
            row_status["violations"].append("runtime_cuda_available is not true")
        cuda_version = row.get("runtime_cuda_version")
        if not isinstance(cuda_version, str) or not cuda_version.startswith(requirements["cuda_prefix"]):
            row_status["violations"].append("runtime_cuda_version does not match required prefix")
        gpu_name = row.get("runtime_gpu_name")
        if not isinstance(gpu_name, str) or requirements["gpu_name_contains"] not in gpu_name:
            row_status["violations"].append("runtime_gpu_name does not contain required text")
        gpu_total = row.get("runtime_gpu_total_memory_gb")
        if not _finite_number(gpu_total) or gpu_total < requirements["gpu_total_memory_min_gb"]:
            row_status["violations"].append("runtime_gpu_total_memory_gb is below required minimum")
        gpu_count = row.get("runtime_gpu_count")
        if gpu_count != requirements["gpu_count"]:
            row_status["violations"].append("runtime_gpu_count does not match required value")
        if expected_cuda_visible_devices and row.get("runtime_cuda_visible_devices") != expected_cuda_visible_devices:
            row_status["violations"].append("runtime_cuda_visible_devices does not match config default")
        torch_version = row.get("runtime_torch")
        if not isinstance(torch_version, str) or not torch_version.startswith(requirements["torch_prefix"]):
            row_status["violations"].append("runtime_torch does not match required prefix")
        if not isinstance(row.get("runtime_transformers"), str) or not row.get("runtime_transformers"):
            row_status["violations"].append("runtime_transformers is missing")
        if row.get("runtime_bitsandbytes_available") is not True:
            row_status["violations"].append("runtime_bitsandbytes_available is not true")
        cpu_count = row.get("runtime_cpu_count")
        if not isinstance(cpu_count, int) or isinstance(cpu_count, bool) or cpu_count < requirements["cpu_count_min"]:
            row_status["violations"].append("runtime_cpu_count is below required minimum")
        system_memory = row.get("runtime_system_memory_total_gb")
        if not _finite_number(system_memory) or system_memory < requirements["system_memory_min_gb"]:
            row_status["violations"].append("runtime_system_memory_total_gb is below required minimum")
        if row.get("runtime_hf_endpoint") != requirements["hf_endpoint"]:
            row_status["violations"].append("runtime_hf_endpoint does not match required value")
        row_status["ok"] = not row_status["violations"]
        if row_status["violations"]:
            status["violations"].append("matrix_results.json row does not satisfy target runtime requirements")
        status["rows"].append(row_status)
    status["ok"] = not status["violations"]
    return status


def _readme_status(bundle_root: Path) -> Dict[str, Any]:
    path = bundle_root / "RESULTS_BUNDLE_README.md"
    required_text = [
        "LLMCBench Edge Results Bundle",
        "tools/verify_edge_results_bundle.py",
        "tools/verify_edge_delivery.py",
        "validator.ok=true",
        "current_goal_audit.json",
        "edge_target_paths_report.json",
        "before_matrix_blockers",
        "JSON/CSV matrix consistency",
        "score metric keys",
        "compression provenance",
        "quantized storage estimates",
        "target runtime",
    ]
    status: Dict[str, Any] = {
        "ok": False,
        "path": str(path),
        "exists": path.is_file(),
        "missing_required_text": [],
        "violations": [],
    }
    if not path.is_file():
        status["violations"].append("RESULTS_BUNDLE_README.md is missing")
        return status
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [needle for needle in required_text if needle not in text]
    status["missing_required_text"] = missing
    if missing:
        status["violations"].append("RESULTS_BUNDLE_README.md is missing required verification text")
    status["ok"] = not status["violations"]
    return status


def verify_results_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else bundle_root / "RESULTS_BUNDLE_MANIFEST.json"
    report: Dict[str, Any] = {
        "ok": False,
        "complete": False,
        "bundle_root": str(bundle_root),
        "manifest_path": str(manifest_path),
        "violations": [],
        "file_status": [],
        "required_path_status": [],
        "completion_audit_status": {},
        "target_readiness_status": {},
        "scorecard_status": {},
        "matrix_coverage_status": {},
        "score_coverage_status": {},
        "metric_coverage_status": {},
        "compression_provenance_status": {},
        "csv_consistency_status": {},
        "runtime_environment_status": {},
        "target_paths_report_status": {},
        "readme_status": {},
    }
    if not bundle_root.is_dir():
        report["violations"].append("bundle root is missing or not a directory")
        return report
    if not manifest_path.is_file():
        report["violations"].append("RESULTS_BUNDLE_MANIFEST.json is missing")
        return report
    try:
        manifest = _load_json(manifest_path)
    except Exception as exc:
        report["violations"].append(f"manifest is not valid JSON: {type(exc).__name__}: {exc}")
        return report
    if not isinstance(manifest, dict):
        report["violations"].append("manifest must be a JSON object")
        return report
    report["manifest_schema_version"] = manifest.get("schema_version")
    report["complete"] = manifest.get("complete") is True
    if manifest.get("schema_version") != 1:
        report["violations"].append("manifest schema_version must be 1")
    if args.require_complete and not report["complete"]:
        report["violations"].append("manifest complete is not true")
    if args.require_complete and manifest.get("missing_required_artifacts"):
        report["violations"].append("manifest reports missing required artifacts")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        report["violations"].append("manifest files must be a non-empty list")
        files = []
    manifest_file_paths = [item.get("path") for item in files if isinstance(item, dict)]
    duplicate_manifest_paths = sorted({path for path in manifest_file_paths if manifest_file_paths.count(path) > 1})
    report["duplicate_manifest_paths"] = duplicate_manifest_paths
    if duplicate_manifest_paths:
        report["violations"].append("manifest files contain duplicate paths")
    file_status = [_file_status(bundle_root, item if isinstance(item, dict) else {"path": None}) for item in files]
    bad_files = [item for item in file_status if not item["ok"]]
    if bad_files:
        report["violations"].append(f"{len(bad_files)} manifest file entries failed verification")
    report["file_status"] = file_status if args.include_file_status else [
        {key: item[key] for key in ("path", "ok", "exists", "violations")} for item in bad_files
    ]

    manifest_paths = {item.get("path") for item in files if isinstance(item, dict)}
    required_status = []
    for rel in REQUIRED_BUNDLE_FILES:
        path = bundle_root / rel
        ok = path.is_file()
        in_manifest_files = rel in manifest_paths
        required_status.append({"path": rel, "ok": ok, "exists": path.exists(), "in_manifest_files": in_manifest_files})
        if not ok:
            report["violations"].append(f"required bundle file is missing: {rel}")
        elif rel not in SELF_DESCRIBING_BUNDLE_FILES and not in_manifest_files:
            report["violations"].append(f"required bundle file is missing from manifest files: {rel}")
    report["required_path_status"] = required_status

    completion_status = _completion_audit_status(bundle_root / "completion_audit.json")
    readiness_status = _target_readiness_status(bundle_root / "target_readiness.json")
    report["completion_audit_status"] = completion_status
    report["target_readiness_status"] = readiness_status
    if args.require_complete and not completion_status["ok"]:
        report["violations"].append("completion_audit.json ok and validator.ok are not true")
    if args.require_target_readiness_ok and not readiness_status["ok"]:
        report["violations"].append("target_readiness.json ok and completion_ok are not true")
    target_paths_status = _target_paths_report_status(bundle_root / "edge_target_paths_report.json")
    report["target_paths_report_status"] = target_paths_status
    if not target_paths_status["ok"]:
        report["violations"].append("edge_target_paths_report.json is not valid")
    scorecard_status = _scorecard_status(bundle_root)
    report["scorecard_status"] = scorecard_status
    if not scorecard_status["ok"]:
        report["violations"].append("scorecard verification failed")
    readme_status = _readme_status(bundle_root)
    report["readme_status"] = readme_status
    if not readme_status["ok"]:
        report["violations"].append("results bundle README verification failed")

    coverage_config = Path(args.config).expanduser().resolve() if args.config else bundle_root / "matrix_config_resolved.json"
    coverage_status = _matrix_coverage_status(bundle_root, coverage_config)
    report["matrix_coverage_status"] = coverage_status
    if not coverage_status["ok"]:
        report["violations"].append("matrix results coverage verification failed")
    score_status = _score_coverage_status(bundle_root, coverage_config)
    report["score_coverage_status"] = score_status
    if not score_status["ok"]:
        report["violations"].append("matrix results score coverage verification failed")
    metric_status = _metric_coverage_status(bundle_root)
    report["metric_coverage_status"] = metric_status
    if not metric_status["ok"]:
        report["violations"].append("matrix results deployment/resource metric verification failed")
    compression_status = _compression_provenance_status(bundle_root)
    report["compression_provenance_status"] = compression_status
    if not compression_status["ok"]:
        report["violations"].append("matrix results compression provenance verification failed")
    csv_status = _csv_consistency_status(bundle_root)
    report["csv_consistency_status"] = csv_status
    if not csv_status["ok"]:
        report["violations"].append("matrix results CSV consistency verification failed")
    runtime_status = _runtime_environment_status(bundle_root)
    report["runtime_environment_status"] = runtime_status
    if args.require_target_runtime and not runtime_status["ok"]:
        report["violations"].append("matrix results target runtime verification failed")

    report["file_count"] = len(files)
    report["verified_file_count"] = sum(1 for item in file_status if item["ok"])
    report["ok"] = not report["violations"]
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify an exported LLMCBench edge results bundle.")
    parser.add_argument("--bundle-root", required=True, help="Root directory of the exported results bundle.")
    parser.add_argument("--manifest", default="", help="Override path to RESULTS_BUNDLE_MANIFEST.json.")
    parser.add_argument("--output", default="", help="Optional JSON report output path.")
    parser.add_argument("--config", default="", help="Optional matrix config used to verify model/method coverage. Defaults to bundle matrix_config_resolved.json.")
    parser.add_argument("--require-complete", action="store_true", help="Require complete=true and completion_audit.json ok=true with validator.ok=true.")
    parser.add_argument("--require-target-readiness-ok", action="store_true", help="Require target_readiness.json ok=true and completion_ok=true.")
    parser.add_argument("--require-target-runtime", action="store_true", help="Require Python 3.10, CUDA 12.x, single 4090-class GPU, PyTorch 2.x, bitsandbytes, >=8 CPUs, >=240GB RAM and hf-mirror runtime metadata.")
    parser.add_argument("--include-file-status", action="store_true", help="Include every manifest file entry in the report.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_results_bundle(args)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
