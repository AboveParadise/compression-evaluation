import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS
except ModuleNotFoundError:
    from prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS


EXPECTED_COMPRESSION_BACKENDS = {
    "dense": {"source_checkpoint_reference", "precompressed_checkpoint"},
    "wanda": {"built_in_activation_pruning", "precompressed_checkpoint"},
    "sparsegpt": {"built_in_activation_pruning", "precompressed_checkpoint"},
    "gptq": {"auto_gptq", "precompressed_checkpoint"},
    "awq": {"autoawq", "precompressed_checkpoint"},
    "smoothquant": {"external_command", "precompressed_checkpoint", "target_model_adaptation"},
    "omniquant": {"external_command", "precompressed_checkpoint", "target_model_adaptation"},
}


def _exists(path: str) -> bool:
    return Path(path).expanduser().exists()


def _load_json(path: str) -> Optional[Any]:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _quant_runtime_entries(payload: Any):
    if not isinstance(payload, dict):
        return [], None
    rows = payload.get("rows")
    if isinstance(rows, list):
        return rows, "rows"
    probes = payload.get("probes")
    if isinstance(probes, list):
        return probes, "probes"
    return [], None


def _file_summary(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def _nearest_existing_parent(path: Path) -> Path:
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def _path_access_summary(path_value: Any, expected_kind: str) -> Dict[str, Any]:
    path = Path(str(path_value or "")).expanduser()
    exists = path.exists()
    nearest_existing_parent = _nearest_existing_parent(path)
    is_dir = path.is_dir() if exists else None
    is_file = path.is_file() if exists else None
    path_writable = os.access(path, os.W_OK) if exists else None
    parent_is_dir = nearest_existing_parent.is_dir()
    parent_writable = parent_is_dir and os.access(nearest_existing_parent, os.W_OK)
    if expected_kind == "directory":
        can_create = bool(exists and is_dir and path_writable) if exists else parent_writable
        ok = can_create
    else:
        can_create = bool(is_file and path_writable) if exists else parent_writable
        ok = bool(can_create)
    return {
        "path": str(path),
        "expected_kind": expected_kind,
        "exists": exists,
        "is_dir": is_dir,
        "is_file": is_file,
        "is_writable": path_writable,
        "nearest_existing_parent": str(nearest_existing_parent),
        "nearest_existing_parent_is_dir": parent_is_dir,
        "nearest_existing_parent_writable": parent_writable,
        "can_create": can_create,
        "ok": ok,
    }


def _target_path_access(defaults: Dict[str, Any]) -> Dict[str, Any]:
    required = {
        "hf_cache_dir": ("directory", defaults.get("hf_cache_dir")),
        "data_dir": ("directory", defaults.get("data_dir")),
        "output_root": ("directory", defaults.get("output_root")),
        "calibration_file": ("file", defaults.get("calibration_file")),
        "prompt_file": ("file", defaults.get("prompt_file")),
    }
    paths = {
        name: _path_access_summary(path_value, expected_kind)
        for name, (expected_kind, path_value) in required.items()
    }
    blockers = [
        {
            "name": name,
            "path": item["path"],
            "expected_kind": item["expected_kind"],
            "nearest_existing_parent": item["nearest_existing_parent"],
            "reason": "path is not creatable or writable by the current user",
        }
        for name, item in paths.items()
        if item.get("ok") is not True
    ]
    return {
        "ok": not blockers,
        "paths": paths,
        "blockers": blockers,
    }


def _json_ok(path: str, key: str = "ok") -> bool:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return False
    return bool(payload.get(key))


def _score(task_payload: Dict[str, Any]) -> Optional[float]:
    for key in ("acc,none", "acc_norm,none", "exact_match,strict-match"):
        value = task_payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _score_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def _method_compression_params_present(row: Dict[str, Any]) -> bool:
    method = row.get("compression_method")
    if method in {"wanda", "sparsegpt"}:
        legacy_fields_ok = _positive_number(row.get("compression_sparsity")) and isinstance(
            row.get("compression_prune_lm_head"), bool
        )
        provenance_present = row.get("compression_prune_pattern") is not None or row.get(
            "compression_prune_structure"
        ) is not None
        return legacy_fields_ok and (
            not provenance_present
            or (
                isinstance(row.get("compression_prune_pattern"), str)
                and bool(row.get("compression_prune_pattern"))
                and isinstance(row.get("compression_prune_structure"), str)
                and bool(row.get("compression_prune_structure"))
            )
        )
    if method in {"gptq", "awq", "smoothquant", "omniquant"}:
        group_size = row.get("compression_group_size")
        per_channel_protocols = {
            "gptq_for_llama_c4_w4_asymmetric_actorder_true_sequential": "gptq",
            "omniquant_w4a16_wikitext2_128x2048_epochs20_lwc": "omniquant",
        }
        group_size_valid = method == "smoothquant" or (
            isinstance(group_size, int)
            and not isinstance(group_size, bool)
            and (
                group_size > 0
                or (
                    group_size == -1
                    and per_channel_protocols.get(row.get("compression_paper_protocol")) == method
                )
            )
        )
        common = (
            isinstance(row.get("compression_quant_bits"), int)
            and not isinstance(row.get("compression_quant_bits"), bool)
            and row.get("compression_quant_bits") > 0
            and group_size_valid
            and isinstance(row.get("compression_calibration_samples"), int)
            and not isinstance(row.get("compression_calibration_samples"), bool)
            and row.get("compression_calibration_samples") > 0
        )
        if method == "gptq":
            return common and isinstance(row.get("compression_desc_act"), bool)
        if method == "awq":
            return common and isinstance(row.get("compression_awq_version"), str) and bool(row.get("compression_awq_version"))
        return common
    return method == "dense"


def _accuracy_summary(path: str) -> Dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {"exists": False}
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    target_tasks = ["mmlu", "arc_challenge", "hellaswag", "piqa", "winogrande"]
    scores = {
        task: _score(results.get(task, {})) if isinstance(results.get(task), dict) else None
        for task in target_tasks
    }
    failed_tasks = payload.get("failed_tasks") if isinstance(payload.get("failed_tasks"), dict) else {}
    return {
        "exists": True,
        "path": str(Path(path).expanduser()),
        "tasks": payload.get("tasks", []),
        "split_accuracy_tasks": bool(payload.get("split_accuracy_tasks")),
        "scores": scores,
        "failed_tasks": failed_tasks,
        "all_target_tasks_present": all(task in results for task in target_tasks),
        "all_target_tasks_succeeded": all(task in results for task in target_tasks) and not failed_tasks,
        "mmlu_subtask_count": len([name for name in results if name.startswith("mmlu_")]),
    }


def _deployment_summary(path: str) -> Dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {"exists": False}
    peak_allocated = payload.get("peak_memory_allocated_gb")
    peak_reserved = payload.get("peak_memory_reserved_gb")
    exceeds_24gb = bool(
        isinstance(peak_allocated, (int, float)) and peak_allocated > 24
        or isinstance(peak_reserved, (int, float)) and peak_reserved > 24
    )
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    gpus = runtime.get("gpus") if isinstance(runtime.get("gpus"), list) else []
    gpu_name = gpus[0].get("name") if gpus and isinstance(gpus[0], dict) else None
    runtime_python = runtime.get("python")
    target_runtime_valid = (
        isinstance(runtime_python, str)
        and runtime_python.startswith("3.10")
        and isinstance(gpu_name, str)
        and "4090" in gpu_name
    )
    return {
        "exists": True,
        "path": str(Path(path).expanduser()),
        "tokens_per_second": payload.get("tokens_per_second"),
        "dtype": payload.get("dtype"),
        "device_map": payload.get("device_map"),
        "max_memory": payload.get("max_memory"),
        "attn_implementation": payload.get("attn_implementation"),
        "peak_memory_allocated_gb": peak_allocated,
        "peak_memory_reserved_gb": peak_reserved,
        "runtime_python": runtime_python,
        "runtime_gpu_name": gpu_name,
        "exceeds_24gb_target": exceeds_24gb,
        "within_24gb_target": not exceeds_24gb,
        "target_runtime_valid": target_runtime_valid,
        "diagnostic_only": not target_runtime_valid,
    }


def _resource_summary(path: str) -> Dict[str, Any]:
    file_path = Path(path).expanduser()
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {"exists": False}
    parameter_stats = payload.get("parameter_stats") if isinstance(payload.get("parameter_stats"), dict) else {}
    compute_complexity = payload.get("compute_complexity") if isinstance(payload.get("compute_complexity"), dict) else {}
    summary = _load_json(str(file_path.parent / "summary.json"))
    compression = _load_json(str(file_path.parent / "compression.json"))
    manifest = compression.get("manifest") if isinstance(compression, dict) and isinstance(compression.get("manifest"), dict) else {}
    compressed_path = None
    if isinstance(summary, dict):
        compressed_path = summary.get("compressed_model_path")
    if not compressed_path:
        compressed_path = manifest.get("compressed_model_path")
    return {
        "exists": True,
        "path": str(file_path),
        "compressed_model_path": compressed_path,
        "compressed_checkpoint_ready": _checkpoint_ready_path(compressed_path) if compressed_path else None,
        "storage_gb": payload.get("storage_gb"),
        "resource_backend": payload.get("resource_backend"),
        "model_load_error": payload.get("model_load_error"),
        "total_params": parameter_stats.get("total_params"),
        "sparsity": parameter_stats.get("sparsity"),
        "flops": compute_complexity.get("flops"),
        "macs": compute_complexity.get("macs"),
        "flops_num": compute_complexity.get("flops_num"),
        "compute_backend": compute_complexity.get("backend"),
        "compute_warning": compute_complexity.get("warning"),
    }


def _validation_summary(path: str) -> Dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {"exists": False, "ok": False}
    return {
        "exists": True,
        "path": str(Path(path).expanduser()),
        "ok": bool(payload.get("ok")),
        "expected_rows": payload.get("expected_rows"),
        "actual_rows": payload.get("actual_rows"),
        "missing_pairs": payload.get("missing_pairs"),
        "unexpected_pairs": payload.get("unexpected_pairs"),
        "duplicate_pairs": payload.get("duplicate_pairs"),
        "missing_fields": payload.get("missing_fields"),
        "type_violations": payload.get("type_violations"),
        "range_violations": payload.get("range_violations"),
        "consistency_violations": payload.get("consistency_violations"),
        "accuracy_violations": payload.get("accuracy_violations"),
        "memory_violations": payload.get("memory_violations"),
        "format_errors": payload.get("format_errors"),
        "required_score_fields": payload.get("required_score_fields"),
        "required_score_metric_fields": payload.get("required_score_metric_fields"),
    }


def _runtime_probe_summary(path: str) -> Dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {"exists": False}
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    first = rows[0] if rows and isinstance(rows[0], dict) else {}
    probe = first.get("payload") if isinstance(first.get("payload"), dict) else {}
    steps = probe.get("steps") if isinstance(probe.get("steps"), list) else []
    step_ok = {
        item.get("name"): item.get("ok")
        for item in steps
        if isinstance(item, dict) and item.get("name")
    }
    return {
        "exists": True,
        "path": str(Path(path).expanduser()),
        "ok": bool(payload.get("ok")),
        "method": first.get("method") or probe.get("method"),
        "model_path": first.get("model_path") or probe.get("model_path"),
        "returncode": first.get("returncode"),
        "runtime_imports_ok": step_ok.get("runtime_imports"),
        "config_ok": step_ok.get("config"),
        "tokenizer_ok": step_ok.get("tokenizer"),
        "model_load_ok": step_ok.get("model_load"),
        "generate_ok": step_ok.get("generate"),
        "error": probe.get("error") or first.get("stderr_tail") or first.get("stdout_tail"),
        "torch": probe.get("torch"),
        "transformers": probe.get("transformers"),
        "cuda_visible_devices": probe.get("cuda_visible_devices"),
        "imports": probe.get("imports"),
    }


def _portable_bundle_verify_summary(path: str) -> Dict[str, Any]:
    payload = _load_json(path)
    path_obj = Path(path).expanduser()
    if not isinstance(payload, dict):
        return {
            "exists": path_obj.exists(),
            "path": str(path_obj),
            "ok": False,
            "code_ok": None,
            "archive_ok": None,
        }
    code_status = payload.get("code_status") if isinstance(payload.get("code_status"), dict) else {}
    archive_status = payload.get("archive_status") if isinstance(payload.get("archive_status"), dict) else {}
    return {
        "exists": True,
        "path": str(path_obj),
        "ok": payload.get("ok"),
        "file_count": payload.get("file_count"),
        "verified_file_count": payload.get("verified_file_count"),
        "archive_ok": archive_status.get("ok"),
        "archive_checked": archive_status.get("checked"),
        "archive_expected_top_level_dir": archive_status.get("expected_top_level_dir"),
        "archive_top_level_dirs": archive_status.get("archive_top_level_dirs", []),
        "archive_verified_file_count": archive_status.get("verified_archive_file_count"),
        "archive_duplicate_file_count": len(archive_status.get("duplicate_archive_files", []))
        if isinstance(archive_status.get("duplicate_archive_files"), list)
        else None,
        "archive_duplicate_files": archive_status.get("duplicate_archive_files", []),
        "archive_unsafe_member_count": len(archive_status.get("unsafe_archive_members", []))
        if isinstance(archive_status.get("unsafe_archive_members"), list)
        else None,
        "archive_unsafe_members": archive_status.get("unsafe_archive_members", []),
        "archive_unsupported_member_count": len(archive_status.get("unsupported_archive_members", []))
        if isinstance(archive_status.get("unsupported_archive_members"), list)
        else None,
        "archive_unsupported_members": archive_status.get("unsupported_archive_members", []),
        "archive_size_mismatch_count": len(archive_status.get("size_mismatches", []))
        if isinstance(archive_status.get("size_mismatches"), list)
        else None,
        "archive_sha256_mismatch_count": len(archive_status.get("sha256_mismatches", []))
        if isinstance(archive_status.get("sha256_mismatches"), list)
        else None,
        "archive_size_mismatches": archive_status.get("size_mismatches", []),
        "archive_sha256_mismatches": archive_status.get("sha256_mismatches", []),
        "code_ok": code_status.get("ok"),
        "manifest_commit": code_status.get("manifest_commit"),
        "handoff_commit": code_status.get("handoff_commit"),
        "manifest_dirty": code_status.get("manifest_dirty"),
        "handoff_dirty": code_status.get("handoff_dirty"),
        "require_clean_code": code_status.get("require_clean_code"),
        "require_commit": code_status.get("require_commit"),
        "violations": payload.get("violations", []),
        "code_violations": code_status.get("violations", []),
    }


def _matrix_results_summary(path: str) -> Dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        return {
            "exists": False,
            "raw_row_count": 0,
            "invalid_row_count": 0,
            "invalid_rows": [],
            "row_count": 0,
            "models": [],
            "methods": [],
            "rows": [],
        }
    invalid_rows = [
        {"index": index, "type": type(row).__name__}
        for index, row in enumerate(payload)
        if not isinstance(row, dict)
    ]
    mmlu_subtask_score_fields = [f"{task}_score" for task in DEFAULT_MMLU_SUBTASKS]
    target_score_metric_fields = [
        "mmlu_score_metric",
        "arc_challenge_score_metric",
        "hellaswag_score_metric",
        "piqa_score_metric",
        "winogrande_score_metric",
    ]
    mmlu_subtask_score_metric_fields = [f"{task}_score_metric" for task in DEFAULT_MMLU_SUBTASKS]
    rows = [
        {
            "model_name": row.get("model_name"),
            "compression_method": row.get("compression_method"),
            "compression_backend": row.get("compression_backend"),
            "has_method_compression_params": _method_compression_params_present(row),
            "has_compression_provenance": (
                isinstance(row.get("source_model_path"), str)
                and bool(row.get("source_model_path"))
                and isinstance(row.get("compressed_model_path"), str)
                and bool(row.get("compressed_model_path"))
                and isinstance(row.get("compression_backend"), str)
                and bool(row.get("compression_backend"))
                and row.get("compression_backend")
                in EXPECTED_COMPRESSION_BACKENDS.get(str(row.get("compression_method")), set())
                and isinstance(row.get("compression_materialized"), bool)
                and isinstance(row.get("compression_uses_precompressed"), bool)
                and row.get("compression_uses_precompressed") == (row.get("compression_backend") == "precompressed_checkpoint")
                and row.get("compression_materialized")
                == (row.get("compression_backend") not in {"source_checkpoint_reference", "precompressed_checkpoint"})
                and (
                    row.get("compression_backend") != "external_command"
                    or (
                        isinstance(row.get("compression_external_command"), str)
                        and bool(row.get("compression_external_command").strip())
                        and row.get("compression_external_returncode") == 0
                    )
                )
            ),
            "has_accuracy": all(
                _score_value(row.get(field))
                for field in [
                    "mmlu_score",
                    "arc_challenge_score",
                    "hellaswag_score",
                    "piqa_score",
                    "winogrande_score",
                ]
            ),
            "has_accuracy_metric_keys": all(
                isinstance(row.get(field), str) and bool(row.get(field))
                for field in target_score_metric_fields
            ),
            "mmlu_subtask_score_count": sum(
                1
                for field in mmlu_subtask_score_fields
                if _score_value(row.get(field))
            ),
            "mmlu_subtask_metric_count": sum(
                1
                for field in mmlu_subtask_score_metric_fields
                if isinstance(row.get(field), str) and bool(row.get(field))
            ),
            "has_full_mmlu_subtask_accuracy": all(
                _score_value(row.get(field))
                for field in mmlu_subtask_score_fields
            ),
            "has_full_mmlu_subtask_metric_keys": all(
                isinstance(row.get(field), str) and bool(row.get(field))
                for field in mmlu_subtask_score_metric_fields
            ),
            "accuracy_limit": row.get("accuracy_limit"),
            "accuracy_failed_task_count": row.get("accuracy_failed_task_count"),
            "accuracy_failed_tasks": row.get("accuracy_failed_tasks"),
            "has_final_accuracy_run": (
                row.get("accuracy_limit") in (None, "")
                and row.get("accuracy_failed_task_count") in (0, 0.0, None)
                and row.get("accuracy_failed_tasks") in ("", None)
            ),
            "deployment_memory_measured_peak_gb": row.get("deployment_memory_measured_peak_gb"),
            "deployment_memory_limit_gb": row.get("deployment_memory_limit_gb"),
            "deployment_memory_limit_exceeded": row.get("deployment_memory_limit_exceeded"),
            "deployment_batch_size": row.get("deployment_batch_size"),
            "has_deployment": (
                _positive_number(row.get("tokens_per_second"))
                and isinstance(row.get("deployment_memory_measured_peak_gb"), (int, float))
                and not isinstance(row.get("deployment_memory_measured_peak_gb"), bool)
                and row.get("deployment_memory_measured_peak_gb") >= 0
                and _positive_number(row.get("deployment_memory_limit_gb"))
                and row.get("deployment_memory_limit_gb") <= 24
                and row.get("deployment_memory_measured_peak_gb") <= row.get("deployment_memory_limit_gb")
                and row.get("deployment_memory_limit_exceeded") is False
            ),
            "resource_backend": row.get("resource_backend"),
            "compute_backend": row.get("compute_backend"),
            "has_resource": (
                _positive_number(row.get("storage_gb"))
                and _positive_number(row.get("flops_num"))
                and row.get("resource_backend") == "model_load"
                and row.get("compute_backend") in {"calflops", "analytical_config_estimate"}
            ),
            "note": row.get("note"),
        }
        for row in payload
        if isinstance(row, dict)
    ]
    return {
        "exists": True,
        "path": str(Path(path).expanduser()),
        "raw_row_count": len(payload),
        "invalid_row_count": len(invalid_rows),
        "invalid_rows": invalid_rows,
        "row_count": len(rows),
        "models": sorted({row["model_name"] for row in rows if row.get("model_name")}),
        "methods": sorted({row["compression_method"] for row in rows if row.get("compression_method")}),
        "rows": rows,
    }


def _validator_summary(completion_audit: Any) -> Dict[str, Any]:
    if not isinstance(completion_audit, dict):
        return {"exists": False, "ok": False}
    validator = completion_audit.get("validator")
    if not isinstance(validator, dict):
        return {"exists": False, "ok": False}
    payload = validator.get("payload") if isinstance(validator.get("payload"), dict) else {}
    count_fields = [
        "missing_pairs",
        "duplicate_pairs",
        "unexpected_pairs",
        "missing_fields",
        "type_violations",
        "range_violations",
        "consistency_violations",
        "accuracy_violations",
        "memory_violations",
        "format_errors",
    ]
    violation_counts = {
        field: len(payload.get(field)) if isinstance(payload.get(field), list) else None
        for field in count_fields
    }
    total_violation_count = sum(
        value for value in violation_counts.values() if isinstance(value, int)
    )
    return {
        "exists": True,
        "ok": validator.get("ok") is True and validator.get("payload_ok") is True,
        "returncode": validator.get("returncode"),
        "payload_ok": validator.get("payload_ok"),
        "cmd": validator.get("cmd"),
        "payload_reported_ok": payload.get("ok") if isinstance(payload, dict) else None,
        "expected_rows": payload.get("expected_rows") if isinstance(payload, dict) else None,
        "actual_rows": payload.get("actual_rows") if isinstance(payload, dict) else None,
        "violation_counts": violation_counts,
        "total_violation_count": total_violation_count,
    }


def _final_artifacts_summary(output_root: str) -> Dict[str, Any]:
    root = Path(output_root).expanduser()
    artifact_names = [
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
        "edge_results_bundle.tar.gz",
        "edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json",
        "edge_results_bundle_verify.json",
        "edge_delivery_verify.json",
    ]
    artifacts = {}
    for name in artifact_names:
        artifacts[name] = _file_summary(root / name)
    missing_required_artifacts = [
        {"name": name, "path": artifact["path"]}
        for name, artifact in artifacts.items()
        if artifact["exists"] is not True
    ]
    optional_artifact_names = [
        "edge_prepare_assets_report.json",
        "edge_prepare_metadata_report.json",
        "edge_prepare_precompressed_report.json",
    ]
    optional_artifacts = {}
    for name in optional_artifact_names:
        optional_artifacts[name] = _file_summary(root / name)
    completion_audit = _load_json(str(root / "completion_audit.json"))
    completion_ok = isinstance(completion_audit, dict) and completion_audit.get("ok") is True
    validator_summary = _validator_summary(completion_audit)
    asset_audit = _load_json(str(root / "edge_asset_audit.json"))
    asset_audit_summary = {
        "exists": isinstance(asset_audit, dict),
        "ok": bool(asset_audit.get("ok")) if isinstance(asset_audit, dict) else False,
    }
    if isinstance(asset_audit, dict):
        models = asset_audit.get("models") if isinstance(asset_audit.get("models"), list) else []
        precompressed = asset_audit.get("precompressed") if isinstance(asset_audit.get("precompressed"), list) else []
        asset_audit_summary.update(
            {
                "missing_files": asset_audit.get("missing_files"),
                "low_disk": asset_audit.get("low_disk"),
                "unwritable_paths": asset_audit.get("unwritable_paths"),
                "paths": asset_audit.get("paths"),
                "required_models_not_ready": [
                    {
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "required_for_methods": item.get("required_for_methods"),
                        "weight_status": item.get("weight_status"),
                    }
                    for item in models
                    if isinstance(item, dict) and item.get("required") and not item.get("checkpoint_ready")
                ],
                "precompressed_not_ready": [
                    {
                        "model": item.get("model"),
                        "method": item.get("method"),
                        "path": item.get("path"),
                        "weight_status": item.get("weight_status"),
                    }
                    for item in precompressed
                    if isinstance(item, dict) and not item.get("checkpoint_ready")
                ],
            }
        )
    preflight_report = _load_json(str(root / "edge_preflight_report.json"))
    preflight_summary = {
        "exists": isinstance(preflight_report, dict),
        "ok": bool(preflight_report.get("ok")) if isinstance(preflight_report, dict) else False,
    }
    if isinstance(preflight_report, dict):
        steps = preflight_report.get("steps") if isinstance(preflight_report.get("steps"), list) else []
        failed_steps = [
            {"name": step.get("name"), "required": step.get("required"), "returncode": step.get("returncode")}
            for step in steps
            if isinstance(step, dict) and step.get("ok") is not True
        ]
        preflight_summary.update(
            {
                "step_count": len(steps),
                "failed_required_steps": [step for step in failed_steps if step.get("required") is True],
                "failed_optional_steps": [step for step in failed_steps if step.get("required") is not True],
            }
        )
    prewarm_report = _load_json(str(root / "edge_prewarm_report.json"))
    prewarm_summary = {"exists": False, "ok": False}
    if isinstance(prewarm_report, dict):
        tasks = prewarm_report.get("tasks") if isinstance(prewarm_report.get("tasks"), list) else []
        attempted = prewarm_report.get("attempted") if isinstance(prewarm_report.get("attempted"), list) else []
        succeeded = prewarm_report.get("succeeded") if isinstance(prewarm_report.get("succeeded"), list) else []
        target_count = prewarm_report.get("target_count")
        if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count <= 0:
            targets = prewarm_report.get("targets") if isinstance(prewarm_report.get("targets"), list) else []
            target_count = len(targets) if targets else None
        prewarm_summary = {
            "exists": True,
            "ok": bool(prewarm_report.get("ok")),
            "dtype": prewarm_report.get("dtype"),
            "attn_implementation": prewarm_report.get("attn_implementation"),
            "task_count": len(tasks),
            "target_count": target_count,
            "expected_attempted_count": len(tasks) * target_count if isinstance(target_count, int) else None,
            "attempted_count": len(attempted),
            "succeeded_count": len(succeeded),
            "failed": prewarm_report.get("failed") if isinstance(prewarm_report.get("failed"), list) else None,
        }
    handoff_manifest = _load_json(str(root / "edge_handoff_manifest.json"))
    handoff_summary = {
        "exists": isinstance(handoff_manifest, dict),
        "ok": isinstance(handoff_manifest, dict) and handoff_manifest.get("schema_version") == 1,
    }
    if isinstance(handoff_manifest, dict):
        handoff_summary.update(
            {
                "schema_version": handoff_manifest.get("schema_version"),
                "code_commit": handoff_manifest.get("code", {}).get("commit")
                if isinstance(handoff_manifest.get("code"), dict)
                else None,
                "code_dirty": handoff_manifest.get("code", {}).get("dirty")
                if isinstance(handoff_manifest.get("code"), dict)
                else None,
                "output_root": handoff_manifest.get("paths", {}).get("output_root")
                if isinstance(handoff_manifest.get("paths"), dict)
                else None,
            }
        )
    optional_backends_report = _load_json(str(root / "edge_optional_backends_report.json"))
    optional_backends_summary = {
        "exists": isinstance(optional_backends_report, dict),
        "ok": bool(optional_backends_report.get("ok")) if isinstance(optional_backends_report, dict) else False,
    }
    if isinstance(optional_backends_report, dict):
        optional_backends_summary.update(
            {
                "dry_run": optional_backends_report.get("dry_run"),
                "selected_packages": optional_backends_report.get("selected_packages"),
                "pip_returncode": optional_backends_report.get("pip_returncode"),
                "imports": optional_backends_report.get("imports"),
            }
        )
    quant_runtime_report = _load_json(str(root / "edge_quant_runtime_report.json"))
    quant_runtime_summary = {
        "exists": isinstance(quant_runtime_report, dict),
        "ok": bool(quant_runtime_report.get("ok")) if isinstance(quant_runtime_report, dict) else False,
    }
    if isinstance(quant_runtime_report, dict):
        probes, probe_source = _quant_runtime_entries(quant_runtime_report)
        quant_runtime_summary.update(
            {
                "entry_source_field": probe_source,
                "row_count": len(probes) if probe_source == "rows" else None,
                "probe_count": len(probes),
                "failed_probes": [
                    {
                        "model": probe.get("model"),
                        "method": probe.get("method"),
                        "path": probe.get("model_path") or probe.get("path"),
                        "ok": probe.get("ok"),
                        "returncode": probe.get("returncode"),
                        "reason": probe.get("reason"),
                    }
                    for probe in probes
                    if isinstance(probe, dict) and probe.get("ok") is not True
                ],
            }
        )
    target_readiness_report = _load_json(str(root / "target_readiness.json"))
    target_readiness_summary = {
        "exists": isinstance(target_readiness_report, dict),
        "ok": bool(target_readiness_report.get("ok")) if isinstance(target_readiness_report, dict) else False,
    }
    if isinstance(target_readiness_report, dict):
        blockers = (
            target_readiness_report.get("before_matrix_blockers")
            if isinstance(target_readiness_report.get("before_matrix_blockers"), list)
            else []
        )
        command_groups = (
            target_readiness_report.get("next_command_groups")
            if isinstance(target_readiness_report.get("next_command_groups"), dict)
            else {}
        )
        command_group_counts = {
            name: len(commands)
            for name, commands in command_groups.items()
            if isinstance(commands, list)
        }
        precompressed_followup = (
            command_groups.get("precompressed_config_followup")
            if isinstance(command_groups.get("precompressed_config_followup"), list)
            else []
        )
        baseline_finalize = (
            command_groups.get("baseline_finalize")
            if isinstance(command_groups.get("baseline_finalize"), list)
            else []
        )
        precompressed_post_run = (
            command_groups.get("precompressed_config_post_run_audit")
            if isinstance(command_groups.get("precompressed_config_post_run_audit"), list)
            else []
        )
        baseline_post_run = (
            command_groups.get("baseline_post_run_audit")
            if isinstance(command_groups.get("baseline_post_run_audit"), list)
            else []
        )
        method_payload = (
            target_readiness_report.get("method_readiness", {}).get("payload")
            if isinstance(target_readiness_report.get("method_readiness"), dict)
            and isinstance(target_readiness_report.get("method_readiness", {}).get("payload"), dict)
            else {}
        )
        target_readiness_summary.update(
            {
                "config": target_readiness_report.get("config"),
                "root": target_readiness_report.get("root"),
                "ready_for_matrix": target_readiness_report.get("ready_for_matrix"),
                "completion_ok": target_readiness_report.get("completion_ok"),
                "required_before_matrix": target_readiness_report.get("required_before_matrix"),
                "missing_before_matrix": target_readiness_report.get("missing_before_matrix"),
                "method_summary": method_payload.get("summary") if isinstance(method_payload.get("summary"), dict) else {},
                "method_rows": method_payload.get("rows") if isinstance(method_payload.get("rows"), list) else [],
                "before_matrix_blocker_count": len(blockers),
                "before_matrix_blockers": blockers,
                "next_command_group_names": sorted(command_group_counts),
                "next_command_group_counts": command_group_counts,
                "has_precompressed_config_followup_matrix": any(
                    "tools/run_edge_matrix.py" in str(command)
                    and "edge_matrix_with_precompressed.json" in str(command)
                    for command in precompressed_followup
                ),
                "has_baseline_finalize_matrix": any(
                    "tools/run_edge_matrix.py" in str(command)
                    and "edge_matrix_with_precompressed.json" not in str(command)
                    for command in baseline_finalize
                ),
                "has_precompressed_config_post_run_readiness": any(
                    "tools/audit_target_readiness.py" in str(command)
                    and "edge_matrix_with_precompressed.json" in str(command)
                    and "--require-prepare-precompressed-report" in str(command)
                    for command in precompressed_post_run
                ),
                "has_baseline_post_run_readiness": any(
                    "tools/audit_target_readiness.py" in str(command)
                    and "edge_matrix_with_precompressed.json" not in str(command)
                    for command in baseline_post_run
                ),
            }
        )
    results_bundle_verify_report = _load_json(str(root / "edge_results_bundle_verify.json"))
    results_bundle_verify_summary = {
        "exists": isinstance(results_bundle_verify_report, dict),
        "ok": bool(results_bundle_verify_report.get("ok")) if isinstance(results_bundle_verify_report, dict) else False,
    }
    if isinstance(results_bundle_verify_report, dict):
        completion_status = (
            results_bundle_verify_report.get("completion_audit_status")
            if isinstance(results_bundle_verify_report.get("completion_audit_status"), dict)
            else {}
        )
        target_status = (
            results_bundle_verify_report.get("target_readiness_status")
            if isinstance(results_bundle_verify_report.get("target_readiness_status"), dict)
            else {}
        )
        results_bundle_verify_summary.update(
            {
                "complete": results_bundle_verify_report.get("complete"),
                "violations": results_bundle_verify_report.get("violations"),
                "file_count": results_bundle_verify_report.get("file_count"),
                "verified_file_count": results_bundle_verify_report.get("verified_file_count"),
                "completion_audit_ok": completion_status.get("ok"),
                "completion_validator_ok": completion_status.get("validator_ok"),
                "completion_validator_payload_ok": completion_status.get("validator_payload_ok"),
                "target_readiness_ok": target_status.get("ok"),
                "target_completion_ok": target_status.get("completion_ok"),
                "target_ready_for_matrix": target_status.get("ready_for_matrix"),
            }
        )
    delivery_verify_report = _load_json(str(root / "edge_delivery_verify.json"))
    delivery_verify_summary = {
        "exists": isinstance(delivery_verify_report, dict),
        "ok": bool(delivery_verify_report.get("ok")) if isinstance(delivery_verify_report, dict) else False,
    }
    if isinstance(delivery_verify_report, dict):
        delivery_verify_summary.update(
            {
                "violations": delivery_verify_report.get("violations"),
                "completion_audit_ok": (
                    delivery_verify_report.get("completion_audit_status", {}).get("ok")
                    if isinstance(delivery_verify_report.get("completion_audit_status"), dict)
                    else None
                ),
                "completion_validator_ok": (
                    delivery_verify_report.get("completion_audit_status", {}).get("validator_ok")
                    if isinstance(delivery_verify_report.get("completion_audit_status"), dict)
                    else None
                ),
                "completion_validator_payload_ok": (
                    delivery_verify_report.get("completion_audit_status", {}).get("validator_payload_ok")
                    if isinstance(delivery_verify_report.get("completion_audit_status"), dict)
                    else None
                ),
                "target_readiness_ok": (
                    delivery_verify_report.get("target_readiness_status", {}).get("ok")
                    if isinstance(delivery_verify_report.get("target_readiness_status"), dict)
                    else None
                ),
                "target_completion_ok": (
                    delivery_verify_report.get("target_readiness_status", {}).get("completion_ok")
                    if isinstance(delivery_verify_report.get("target_readiness_status"), dict)
                    else None
                ),
                "results_bundle_ok": (
                    delivery_verify_report.get("results_bundle_status", {}).get("ok")
                    if isinstance(delivery_verify_report.get("results_bundle_status"), dict)
                    else None
                ),
                "root_bundle_match_ok": (
                    delivery_verify_report.get("root_bundle_match_status", {}).get("ok")
                    if isinstance(delivery_verify_report.get("root_bundle_match_status"), dict)
                    else None
                ),
                "archive_bundle_match_ok": (
                    delivery_verify_report.get("archive_bundle_match_status", {}).get("ok")
                    if isinstance(delivery_verify_report.get("archive_bundle_match_status"), dict)
                    else None
                ),
            }
        )
        delivery_substatus_fields = [
            "completion_audit_ok",
            "completion_validator_ok",
            "target_readiness_ok",
            "target_completion_ok",
            "results_bundle_ok",
            "root_bundle_match_ok",
            "archive_bundle_match_ok",
        ]
        delivery_verify_summary["substatus_ok"] = all(
            delivery_verify_summary.get(field) is True for field in delivery_substatus_fields
        )
    matrix_results = _matrix_results_summary(str(root / "matrix_results.json"))
    prepare_report = _load_json(str(root / "edge_prepare_assets_report.json"))
    prepare_summary = {
        "exists": isinstance(prepare_report, dict),
        "ok": bool(prepare_report.get("ok")) if isinstance(prepare_report, dict) else False,
    }
    if isinstance(prepare_report, dict):
        prepare_summary.update(
            {
                "dry_run": prepare_report.get("dry_run"),
                "metadata_only": prepare_report.get("metadata_only"),
                "model_count": len(prepare_report.get("models", [])) if isinstance(prepare_report.get("models"), list) else None,
                "file_count": len(prepare_report.get("files", [])) if isinstance(prepare_report.get("files"), list) else None,
            }
        )
    prepare_metadata_report = _load_json(str(root / "edge_prepare_metadata_report.json"))
    prepare_metadata_summary = {
        "exists": isinstance(prepare_metadata_report, dict),
        "ok": bool(prepare_metadata_report.get("ok")) if isinstance(prepare_metadata_report, dict) else False,
    }
    if isinstance(prepare_metadata_report, dict):
        prepare_metadata_summary.update(
            {
                "dry_run": prepare_metadata_report.get("dry_run"),
                "metadata_only": prepare_metadata_report.get("metadata_only"),
                "model_count": len(prepare_metadata_report.get("models", [])) if isinstance(prepare_metadata_report.get("models"), list) else None,
                "file_count": len(prepare_metadata_report.get("files", [])) if isinstance(prepare_metadata_report.get("files"), list) else None,
            }
        )
    prepare_precompressed_report = _load_json(str(root / "edge_prepare_precompressed_report.json"))
    prepare_precompressed_summary = {
        "exists": isinstance(prepare_precompressed_report, dict),
        "ok": bool(prepare_precompressed_report.get("ok")) if isinstance(prepare_precompressed_report, dict) else False,
    }
    if isinstance(prepare_precompressed_report, dict):
        precompressed_blockers = (
            prepare_precompressed_report.get("blockers")
            if isinstance(prepare_precompressed_report.get("blockers"), list)
            else []
        )
        precompressed_items = (
            prepare_precompressed_report.get("items")
            if isinstance(prepare_precompressed_report.get("items"), list)
            else []
        )
        prepare_precompressed_summary.update(
            {
                "dry_run": prepare_precompressed_report.get("dry_run"),
                "metadata_only": prepare_precompressed_report.get("metadata_only"),
                "item_count": len(precompressed_items),
                "blocker_count": len(precompressed_blockers),
                "blockers": precompressed_blockers,
                "failed_items": [
                    item
                    for item in precompressed_items
                    if isinstance(item, dict) and item.get("ok") is not True
                ],
                "updated_config": prepare_precompressed_report.get("updated_config"),
            }
        )
    completion_summary = {"exists": False, "ok": False}
    mmlu_subtask_coverage_summary = {"exists": False, "ok": False}
    if isinstance(completion_audit, dict):
        mmlu_subtask_coverage_status = (
            completion_audit.get("mmlu_subtask_coverage_status")
            if isinstance(completion_audit.get("mmlu_subtask_coverage_status"), dict)
            else {}
        )
        if mmlu_subtask_coverage_status:
            rows = mmlu_subtask_coverage_status.get("rows") if isinstance(mmlu_subtask_coverage_status.get("rows"), list) else []
            violations = (
                mmlu_subtask_coverage_status.get("violations")
                if isinstance(mmlu_subtask_coverage_status.get("violations"), list)
                else []
            )
            mmlu_subtask_coverage_summary = {
                "exists": True,
                "ok": bool(mmlu_subtask_coverage_status.get("ok")),
                "required": mmlu_subtask_coverage_status.get("required"),
                "required_subtask_count": mmlu_subtask_coverage_status.get("required_subtask_count"),
                "row_count": len(rows),
                "complete_row_count": sum(1 for row in rows if isinstance(row, dict) and row.get("complete") is True),
                "violation_count": len(violations),
                "violations": violations,
            }
        asset_gate = (
            completion_audit.get("asset_audit_report_status")
            if isinstance(completion_audit.get("asset_audit_report_status"), dict)
            else {}
        )
        if asset_gate:
            asset_audit_summary.update(
                {
                    "gate_ok": asset_gate.get("ok"),
                    "gate_required": asset_gate.get("required"),
                    "gate_report_ok": asset_gate.get("report_ok"),
                    "gate_error": asset_gate.get("error"),
                }
            )
        preflight_gate = (
            completion_audit.get("preflight_report_status")
            if isinstance(completion_audit.get("preflight_report_status"), dict)
            else {}
        )
        if preflight_gate:
            preflight_summary.update(
                {
                    "gate_ok": preflight_gate.get("ok"),
                    "gate_required": preflight_gate.get("required"),
                    "gate_report_ok": preflight_gate.get("report_ok"),
                    "gate_error": preflight_gate.get("error"),
                }
            )
        prewarm_gate = (
            completion_audit.get("prewarm_report_status")
            if isinstance(completion_audit.get("prewarm_report_status"), dict)
            else {}
        )
        if prewarm_gate:
            prewarm_summary.update(
                {
                    "gate_ok": prewarm_gate.get("ok"),
                    "gate_required": prewarm_gate.get("required"),
                    "gate_report_ok": prewarm_gate.get("report_ok"),
                    "gate_dtype": prewarm_gate.get("dtype"),
                    "gate_expected_dtype": prewarm_gate.get("expected_dtype"),
                    "gate_dtype_mismatch": prewarm_gate.get("dtype_mismatch"),
                    "gate_attn_implementation": prewarm_gate.get("attn_implementation"),
                    "gate_expected_attn_implementation": prewarm_gate.get("expected_attn_implementation"),
                    "gate_attn_implementation_mismatch": prewarm_gate.get("attn_implementation_mismatch"),
                    "missing_attempted_pairs": prewarm_gate.get("missing_attempted_pairs"),
                    "missing_succeeded_pairs": prewarm_gate.get("missing_succeeded_pairs"),
                    "violations": prewarm_gate.get("violations"),
                }
            )
        handoff_gate = (
            completion_audit.get("handoff_manifest_status")
            if isinstance(completion_audit.get("handoff_manifest_status"), dict)
            else {}
        )
        if handoff_gate:
            handoff_summary.update(
                {
                    "gate_ok": handoff_gate.get("ok"),
                    "gate_required": handoff_gate.get("required"),
                    "gate_schema_version": handoff_gate.get("schema_version"),
                    "gate_code_commit": handoff_gate.get("code_commit"),
                    "gate_code_dirty": handoff_gate.get("code_dirty"),
                    "gate_output_root": handoff_gate.get("output_root"),
                    "gate_violations": handoff_gate.get("violations"),
                }
            )
        optional_backends_gate = (
            completion_audit.get("optional_backends_report_status")
            if isinstance(completion_audit.get("optional_backends_report_status"), dict)
            else {}
        )
        if optional_backends_gate:
            optional_backends_summary.update(
                {
                    "gate_ok": optional_backends_gate.get("ok"),
                    "gate_required": optional_backends_gate.get("required"),
                    "gate_report_ok": optional_backends_gate.get("report_ok"),
                    "gate_error": optional_backends_gate.get("error"),
                }
            )
        quant_runtime_gate = (
            completion_audit.get("quant_runtime_report_status")
            if isinstance(completion_audit.get("quant_runtime_report_status"), dict)
            else {}
        )
        if quant_runtime_gate:
            quant_runtime_summary.update(
                {
                    "gate_ok": quant_runtime_gate.get("ok"),
                    "gate_required": quant_runtime_gate.get("required"),
                    "gate_report_ok": quant_runtime_gate.get("report_ok"),
                    "gate_error": quant_runtime_gate.get("error"),
                }
            )
        completion_summary = {
            "exists": True,
            "ok": bool(completion_audit.get("ok")),
            "validator_ok": validator_summary.get("ok"),
            "validator": validator_summary,
            "expected_pairs": completion_audit.get("expected_pairs"),
            "missing_artifacts": completion_audit.get("missing_artifacts"),
            "artifact_parse_errors": completion_audit.get("artifact_parse_errors"),
            "runtime_violations": completion_audit.get("runtime_violations"),
            "runtime_requirements": completion_audit.get("runtime_requirements"),
            "matrix_run_status": completion_audit.get("matrix_run_status"),
            "matrix_run_report_status_ok": (
                completion_audit.get("matrix_run_report_status", {}).get("ok")
                if isinstance(completion_audit.get("matrix_run_report_status"), dict)
                else None
            ),
            "matrix_finalize_report_status_ok": (
                completion_audit.get("matrix_finalize_report_status", {}).get("ok")
                if isinstance(completion_audit.get("matrix_finalize_report_status"), dict)
                else None
            ),
            "prewarm_report_status_ok": (
                completion_audit.get("prewarm_report_status", {}).get("ok")
                if isinstance(completion_audit.get("prewarm_report_status"), dict)
                else None
            ),
            "handoff_manifest_status_ok": (
                completion_audit.get("handoff_manifest_status", {}).get("ok")
                if isinstance(completion_audit.get("handoff_manifest_status"), dict)
                else None
            ),
            "optional_backends_report_status_ok": (
                completion_audit.get("optional_backends_report_status", {}).get("ok")
                if isinstance(completion_audit.get("optional_backends_report_status"), dict)
                else None
            ),
            "quant_runtime_report_status_ok": (
                completion_audit.get("quant_runtime_report_status", {}).get("ok")
                if isinstance(completion_audit.get("quant_runtime_report_status"), dict)
                else None
            ),
            "prepare_report_status_ok": (
                completion_audit.get("prepare_report_status", {}).get("ok")
                if isinstance(completion_audit.get("prepare_report_status"), dict)
                else None
            ),
            "prepare_metadata_report_status_ok": (
                completion_audit.get("prepare_metadata_report_status", {}).get("ok")
                if isinstance(completion_audit.get("prepare_metadata_report_status"), dict)
                else None
            ),
            "prewarm_report_status": completion_audit.get("prewarm_report_status"),
            "handoff_manifest_status": completion_audit.get("handoff_manifest_status"),
            "optional_backends_report_status": completion_audit.get("optional_backends_report_status"),
            "quant_runtime_report_status": completion_audit.get("quant_runtime_report_status"),
            "prepare_report_status": completion_audit.get("prepare_report_status"),
            "prepare_metadata_report_status": completion_audit.get("prepare_metadata_report_status"),
            "prepare_precompressed_report_status": completion_audit.get("prepare_precompressed_report_status"),
            "mmlu_subtask_coverage_status": completion_audit.get("mmlu_subtask_coverage_status"),
        }
    optional_required_gate_summaries = [
        summary
        for summary in (optional_backends_summary, quant_runtime_summary)
        if summary.get("gate_required") is True
    ]
    required_report_gates_ok = all(
        summary.get("gate_required") is True and summary.get("gate_ok") is True
        for summary in (asset_audit_summary, preflight_summary, prewarm_summary)
    ) and all(summary.get("gate_ok") is True for summary in optional_required_gate_summaries)
    prepare_reports_ok = all(
        summary.get("ok") is True
        for summary in (prepare_summary, prepare_metadata_summary, prepare_precompressed_summary)
        if summary.get("exists")
    )
    final_rows = matrix_results.get("rows") if isinstance(matrix_results.get("rows"), list) else []
    target_methods = ["dense", "wanda", "sparsegpt", "gptq", "awq", "smoothquant", "omniquant"]
    target_pairs = {(model, method) for model in ("LLaMA3-8B", "Qwen3-8B") for method in target_methods}
    actual_pairs = [
        (row.get("model_name"), row.get("compression_method"))
        for row in final_rows
        if isinstance(row, dict)
    ]
    actual_pair_set = set(actual_pairs)
    duplicate_pairs = sorted({pair for pair in actual_pairs if actual_pairs.count(pair) > 1})
    unexpected_pairs = sorted(actual_pair_set - target_pairs)
    missing_pairs = sorted(target_pairs - actual_pair_set)
    rows_by_pair = {
        (row.get("model_name"), row.get("compression_method")): row
        for row in final_rows
        if isinstance(row, dict)
    }
    target_pair_rows_complete = all(
        pair in rows_by_pair
        and rows_by_pair[pair].get("has_compression_provenance") is True
        and rows_by_pair[pair].get("has_method_compression_params") is True
        and rows_by_pair[pair].get("has_accuracy") is True
        and rows_by_pair[pair].get("has_accuracy_metric_keys") is True
        and rows_by_pair[pair].get("has_final_accuracy_run") is True
        and rows_by_pair[pair].get("has_deployment") is True
        and rows_by_pair[pair].get("has_resource") is True
        and rows_by_pair[pair].get("has_full_mmlu_subtask_accuracy") is True
        and rows_by_pair[pair].get("has_full_mmlu_subtask_metric_keys") is True
        for pair in target_pairs
    )
    final_matrix_run_complete = (
        matrix_results.get("invalid_row_count") == 0
        and len(actual_pairs) == len(target_pairs)
        and not duplicate_pairs
        and not unexpected_pairs
        and not missing_pairs
        and target_pair_rows_complete
    )
    return {
        "output_root": str(root),
        "artifacts": artifacts,
        "optional_artifacts": optional_artifacts,
        "required_artifacts": artifacts,
        "missing_required_artifacts": missing_required_artifacts,
        "missing_required_artifact_count": len(missing_required_artifacts),
        "all_artifacts_present": not missing_required_artifacts,
        "completion_audit_ok": completion_ok,
        "validator_ok": validator_summary.get("ok") is True,
        "validator": validator_summary,
        "required_report_gates_ok": required_report_gates_ok,
        "prepare_reports_ok": prepare_reports_ok,
        "asset_audit": asset_audit_summary,
        "preflight_report": preflight_summary,
        "prewarm_report": prewarm_summary,
        "handoff_manifest": handoff_summary,
        "optional_backends_report": optional_backends_summary,
        "quant_runtime_report": quant_runtime_summary,
        "target_readiness": target_readiness_summary,
        "results_bundle_verify": results_bundle_verify_summary,
        "delivery_verify": delivery_verify_summary,
        "prepare_report": prepare_summary,
        "prepare_metadata_report": prepare_metadata_summary,
        "prepare_precompressed_report": prepare_precompressed_summary,
        "mmlu_subtask_coverage": mmlu_subtask_coverage_summary,
        "matrix_results": matrix_results,
        "final_matrix_expected_pair_count": len(target_pairs),
        "final_matrix_actual_pair_count": len(actual_pairs),
        "final_matrix_missing_pairs": [{"model": model, "method": method} for model, method in missing_pairs],
        "final_matrix_unexpected_pairs": [{"model": model, "method": method} for model, method in unexpected_pairs],
        "final_matrix_duplicate_pairs": [{"model": model, "method": method} for model, method in duplicate_pairs],
        "final_matrix_run_complete": final_matrix_run_complete,
        "matrix_results_csv": artifacts["matrix_results.csv"],
        "edge_report": artifacts["edge_report.md"],
        "completion_audit": completion_summary,
        "complete": not missing_required_artifacts
        and final_matrix_run_complete
        and completion_ok
        and validator_summary.get("ok") is True
        and required_report_gates_ok
        and target_readiness_summary.get("ok") is True
        and target_readiness_summary.get("completion_ok") is True
        and target_readiness_summary.get("before_matrix_blocker_count") == 0
        and results_bundle_verify_summary.get("ok") is True
        and results_bundle_verify_summary.get("complete") is True
        and results_bundle_verify_summary.get("completion_validator_ok") is True
        and results_bundle_verify_summary.get("target_completion_ok") is True
        and delivery_verify_summary.get("ok") is True
        and delivery_verify_summary.get("substatus_ok") is True
        and prepare_reports_ok,
    }


def _checkpoint_summary(model_dir: Path, required_files: list[str]) -> Dict[str, Any]:
    files = []
    for name in required_files:
        path = model_dir / name
        files.append({"name": name, "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None})
    return {
        "path": str(model_dir),
        "checkpoint_ready": all(item["exists"] for item in files),
        "required_files": files,
    }


def _read_config(path: str) -> Dict[str, Any]:
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {}


def _checkpoint_ready_path(path_value: str) -> bool:
    path = Path(path_value).expanduser()
    if not (path / "config.json").exists():
        return False
    patterns = ("*.safetensors", "*.bin", "*.pt")
    return any(any(path.glob(pattern)) for pattern in patterns)


def _merge_nested_dicts(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(fallback)
    for key, value in primary.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dicts(value, merged[key])
        else:
            merged[key] = value
    return merged


def _method_blockers(config_path: str, local_config_path: str, llama_ready: bool, qwen_ready: bool) -> Dict[str, Any]:
    target_config = _read_config(config_path)
    local_config = _read_config(local_config_path)
    config = target_config or local_config
    target_external = target_config.get("external_commands") if isinstance(target_config.get("external_commands"), dict) else {}
    local_external = local_config.get("external_commands") if isinstance(local_config.get("external_commands"), dict) else {}
    external_commands = _merge_nested_dicts(target_external, local_external)
    target_precompressed = target_config.get("precompressed_paths") if isinstance(target_config.get("precompressed_paths"), dict) else {}
    local_precompressed = local_config.get("precompressed_paths") if isinstance(local_config.get("precompressed_paths"), dict) else {}
    precompressed_paths = _merge_nested_dicts(target_precompressed, local_precompressed)
    target_models = ["LLaMA3-8B", "Qwen3-8B"]

    def precompressed_status(method: str) -> Dict[str, Any]:
        entries = {}
        for model in target_models:
            raw_path = ""
            if isinstance(precompressed_paths.get(model), dict):
                raw_path = precompressed_paths[model].get(method) or ""
            entries[model] = {
                "path": raw_path or None,
                "ready": _checkpoint_ready_path(raw_path) if raw_path else False,
            }
        return {
            "ready": all(item["ready"] for item in entries.values()),
            "models": entries,
        }

    def checkpoint_blocker(method: str) -> Optional[str]:
        if precompressed_status(method)["ready"]:
            return None
        return None if llama_ready and qwen_ready else "missing target model checkpoint or precompressed checkpoint"

    def method_payload(method: str, backend_ready: bool = True, backend_blocker: Optional[str] = None) -> Dict[str, Any]:
        precompressed = precompressed_status(method)
        precompressed_ready = bool(precompressed["ready"])
        checkpoint_ok = precompressed_ready or (llama_ready and qwen_ready)
        ready = checkpoint_ok and (precompressed_ready or backend_ready)
        blockers = [
            blocker
            for blocker in [
                None if precompressed_ready or backend_ready else backend_blocker,
                checkpoint_blocker(method),
            ]
            if blocker
        ]
        return {
            "ready": ready,
            "precompressed_ready": precompressed_ready,
            "precompressed_paths": precompressed["models"],
            "blockers": blockers,
        }

    def quant_payload(
        method: str,
        compression_ready: bool,
        runtime_ready: bool,
        compression_blocker: str,
        runtime_blocker: str,
    ) -> Dict[str, Any]:
        precompressed = precompressed_status(method)
        precompressed_ready = bool(precompressed["ready"])
        checkpoint_ok = precompressed_ready or (llama_ready and qwen_ready)
        ready = checkpoint_ok and (precompressed_ready or compression_ready) and runtime_ready
        blockers = [
            blocker
            for blocker in [
                None if precompressed_ready or compression_ready else compression_blocker,
                None if runtime_ready else runtime_blocker,
                checkpoint_blocker(method),
            ]
            if blocker
        ]
        return {
            "ready": ready,
            "compression_backend_ready": compression_ready,
            "runtime_ready": runtime_ready,
            "precompressed_ready": precompressed_ready,
            "precompressed_paths": precompressed["models"],
            "blockers": blockers,
        }

    return {
        "dense": method_payload("dense"),
        "wanda": method_payload("wanda"),
        "sparsegpt": method_payload("sparsegpt"),
        "gptq": quant_payload(
            "gptq",
            importlib.util.find_spec("auto_gptq") is not None,
            importlib.util.find_spec("optimum") is not None
            and (
                importlib.util.find_spec("auto_gptq") is not None
                or importlib.util.find_spec("gptqmodel") is not None
            ),
            "auto-gptq import is unavailable and no complete precompressed GPTQ checkpoints are configured",
            "GPTQ runtime loading requires optimum plus auto-gptq or gptqmodel",
        ),
        "awq": quant_payload(
            "awq",
            importlib.util.find_spec("awq") is not None,
            importlib.util.find_spec("awq") is not None,
            "autoawq/awq import is unavailable and no complete precompressed AWQ checkpoints are configured",
            "AWQ runtime loading requires autoawq/awq",
        ),
        "smoothquant": quant_payload(
            "smoothquant",
            bool(external_commands.get("smoothquant") or os.environ.get("LLMCBENCH_SMOOTHQUANT_CMD")),
            importlib.util.find_spec("compressed_tensors") is not None,
            "SmoothQuant external command is not configured and no complete precompressed SmoothQuant checkpoints are configured",
            "SmoothQuant runtime loading requires compressed-tensors",
        ),
        "omniquant": method_payload(
            "omniquant",
            bool(external_commands.get("omniquant") or os.environ.get("LLMCBENCH_OMNIQUANT_CMD")),
            "OmniQuant external command is not configured and no complete precompressed OmniQuant checkpoints are configured",
        ),
    }


def _target_readiness_scope_matches(config_path: str, final_artifacts: Dict[str, Any]) -> bool:
    target = final_artifacts.get("target_readiness") if isinstance(final_artifacts.get("target_readiness"), dict) else {}
    target_config = target.get("config")
    if not isinstance(target_config, str) or not target_config:
        return False
    config = Path(config_path)
    target_config_path = Path(target_config)
    if target_config == config_path or target_config_path.resolve() == config.expanduser().resolve():
        return True
    return (
        config.as_posix() == "configs/edge_4090_matrix.json"
        and target_config_path.name == "edge_matrix_with_precompressed.json"
        and str(target.get("root") or "") == "/data/outputs/llmcbench_edge_4090"
    )


def _merge_target_method_readiness(method_readiness: Dict[str, Any], final_artifacts: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    if not _target_readiness_scope_matches(config_path, final_artifacts):
        return method_readiness
    target = final_artifacts.get("target_readiness") if isinstance(final_artifacts.get("target_readiness"), dict) else {}
    rows = target.get("method_rows") if isinstance(target.get("method_rows"), list) else []
    if not rows:
        return method_readiness
    merged = json.loads(json.dumps(method_readiness))
    rows_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("method"), str):
            rows_by_method.setdefault(row["method"], []).append(row)
    for method, method_rows in rows_by_method.items():
        payload = merged.setdefault(method, {})
        blocked_rows = [row for row in method_rows if row.get("ok") is not True]
        payload["target_readiness_checked"] = True
        payload["target_ready_count"] = sum(1 for row in method_rows if row.get("ok") is True)
        payload["target_blocked_count"] = len(blocked_rows)
        payload["target_rows"] = [
            {
                "model": row.get("model"),
                "ok": row.get("ok"),
                "uses_precompressed": row.get("uses_precompressed"),
                "precompressed_path": row.get("precompressed_path"),
                "runtime_requirements_reason": row.get("runtime_requirements_reason"),
                "quantization_metadata": row.get("quantization_metadata"),
                "blockers": row.get("blockers"),
            }
            for row in method_rows
        ]
        if blocked_rows:
            blocker_messages = []
            for row in blocked_rows:
                blockers = row.get("blockers") if isinstance(row.get("blockers"), list) else []
                for blocker in blockers:
                    blocker_messages.append(f"{row.get('model')}:{blocker}")
            payload["blockers"] = list(dict.fromkeys(blocker_messages))
            payload["ready"] = False
            payload["runtime_ready"] = False
        elif method_rows:
            payload["blockers"] = []
            payload["ready"] = True
            if "runtime_ready" in payload:
                payload["runtime_ready"] = True
    return merged


def _requirement_checklist(
    qwen_checkpoint: Dict[str, Any],
    llama_checkpoint: Dict[str, Any],
    qwen_accuracy: Dict[str, Any],
    qwen_wanda_accuracy: Dict[str, Any],
    qwen_sparsegpt_accuracy: Dict[str, Any],
    qwen_wanda_mmlu_accuracy: Dict[str, Any],
    qwen_sparsegpt_mmlu_accuracy: Dict[str, Any],
    qwen_deployment: Dict[str, Any],
    qwen_gptq_runtime_probe: Dict[str, Any],
    qwen_awq_runtime_probe: Dict[str, Any],
    qwen_smoothquant_runtime_probe: Dict[str, Any],
    qwen_omniquant_runtime_probe: Dict[str, Any],
    llama_gptq_runtime_probe: Dict[str, Any],
    llama_awq_runtime_probe: Dict[str, Any],
    llama_smoothquant_runtime_probe: Dict[str, Any],
    qwen_dense_resource: Dict[str, Any],
    qwen_wanda_resource: Dict[str, Any],
    qwen_sparsegpt_resource: Dict[str, Any],
    qwen_gptq_resource: Dict[str, Any],
    qwen_awq_resource: Dict[str, Any],
    qwen_smoothquant_resource: Dict[str, Any],
    qwen_omniquant_resource: Dict[str, Any],
    llama_gptq_resource: Dict[str, Any],
    llama_awq_resource: Dict[str, Any],
    llama_smoothquant_resource: Dict[str, Any],
    target_defaults: Dict[str, Any],
    method_readiness: Dict[str, Any],
    qwen_partial_results: Dict[str, Any],
    final_artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    target_tasks = ["mmlu", "arc_challenge", "hellaswag", "piqa", "winogrande"]
    required_mmlu_subtasks = list(DEFAULT_MMLU_SUBTASKS)
    target_models = ["LLaMA3-8B", "Qwen3-8B"]
    target_methods = ["dense", "wanda", "sparsegpt", "gptq", "awq", "smoothquant", "omniquant"]
    llama_ready = bool(llama_checkpoint.get("checkpoint_ready"))
    qwen_ready = bool(qwen_checkpoint.get("checkpoint_ready"))
    ready_methods = [
        method for method, payload in method_readiness.items()
        if isinstance(payload, dict) and payload.get("ready")
    ]
    matrix_rows = qwen_partial_results.get("rows") if isinstance(qwen_partial_results.get("rows"), list) else []
    final_complete = bool(final_artifacts.get("complete"))
    final_matrix = final_artifacts.get("matrix_results") if isinstance(final_artifacts.get("matrix_results"), dict) else {}
    final_rows = final_matrix.get("rows") if isinstance(final_matrix.get("rows"), list) else []
    final_pairs = {
        (row.get("model_name"), row.get("compression_method"))
        for row in final_rows
        if isinstance(row, dict)
    }
    target_pairs = {(model, method) for model in target_models for method in target_methods}
    final_covers_all_pairs = target_pairs <= final_pairs
    rows_by_pair = {
        (row.get("model_name"), row.get("compression_method")): row
        for row in final_rows
        if isinstance(row, dict)
    }
    final_accuracy_rows_complete = all(
        pair in rows_by_pair
        and rows_by_pair[pair].get("has_accuracy") is True
        and rows_by_pair[pair].get("has_accuracy_metric_keys") is True
        and rows_by_pair[pair].get("has_final_accuracy_run") is True
        and rows_by_pair[pair].get("has_full_mmlu_subtask_accuracy") is True
        and rows_by_pair[pair].get("has_full_mmlu_subtask_metric_keys") is True
        for pair in target_pairs
    )
    final_deployment_rows_complete = all(
        pair in rows_by_pair and rows_by_pair[pair].get("has_deployment") is True
        for pair in target_pairs
    )
    final_resource_rows_complete = all(
        pair in rows_by_pair and rows_by_pair[pair].get("has_resource") is True
        for pair in target_pairs
    )
    final_completion_audit = final_artifacts.get("completion_audit") if isinstance(final_artifacts.get("completion_audit"), dict) else {}
    runtime_requirements = (
        final_completion_audit.get("runtime_requirements")
        if isinstance(final_completion_audit.get("runtime_requirements"), dict)
        else {}
    )
    target_platform_complete = (
        final_complete
        and runtime_requirements.get("python_prefix") == "3.10"
        and runtime_requirements.get("gpu_name_contains") == "4090"
        and runtime_requirements.get("cuda_prefix") == "12."
        and runtime_requirements.get("gpu_total_memory_min_gb", 0) >= 23
        and runtime_requirements.get("gpu_count") == 1
        and runtime_requirements.get("cudnn_required") is True
        and runtime_requirements.get("torch_prefix") == "2."
        and runtime_requirements.get("transformers_required") is True
        and runtime_requirements.get("bitsandbytes_required") is True
        and runtime_requirements.get("cpu_count_min", 0) >= 8
        and runtime_requirements.get("system_memory_min_gb", 0) >= 240
        and runtime_requirements.get("hf_endpoint") == "https://hf-mirror.com"
        and not final_completion_audit.get("runtime_violations")
    )
    final_accuracy_complete = (
        final_complete
        and final_covers_all_pairs
        and final_accuracy_rows_complete
    )
    final_deployment_complete = (
        final_complete
        and final_covers_all_pairs
        and final_deployment_rows_complete
    )
    final_resource_complete = (
        final_complete
        and final_covers_all_pairs
        and final_resource_rows_complete
    )
    return {
        "accuracy_tool": {
            "requirement": "Evaluate compressed LLaMA3-8B and Qwen3-8B on MMLU, ARC, HellaSwag, PIQA and WinoGrande for pruning and quantization methods.",
            "implemented_entrypoints": [
                "python -m edge_benchmark accuracy",
                "python -m edge_benchmark pipeline --compression-method <method>",
                "python tools/run_edge_matrix.py --config configs/edge_4090_matrix.json --finalize",
            ],
            "target_models": target_models,
            "target_methods": target_methods,
            "target_tasks": target_tasks,
            "required_mmlu_subtask_count": len(required_mmlu_subtasks),
            "required_mmlu_subtasks": required_mmlu_subtasks,
            "current_evidence": {
                "qwen_dense_five_task_smoke": qwen_accuracy,
                "qwen_wanda_four_task_smoke": qwen_wanda_accuracy,
                "qwen_sparsegpt_four_task_smoke": qwen_sparsegpt_accuracy,
                "qwen_wanda_mmlu_subtask_smoke": qwen_wanda_mmlu_accuracy,
                "qwen_sparsegpt_mmlu_subtask_smoke": qwen_sparsegpt_mmlu_accuracy,
                "partial_matrix_rows": matrix_rows,
                "target_split_accuracy_tasks": target_defaults.get("split_accuracy_tasks"),
                "target_continue_on_accuracy_error": target_defaults.get("continue_on_accuracy_error"),
                "final_accuracy_rows_complete": final_accuracy_rows_complete,
                "final_rows_with_accuracy_metric_keys": sum(
                    1 for row in final_rows if isinstance(row, dict) and row.get("has_accuracy_metric_keys")
                ),
                "final_rows_with_final_accuracy_run": sum(
                    1 for row in final_rows if isinstance(row, dict) and row.get("has_final_accuracy_run")
                ),
                "final_rows_with_full_mmlu_subtask_metric_keys": sum(
                    1 for row in final_rows if isinstance(row, dict) and row.get("has_full_mmlu_subtask_metric_keys")
                ),
            },
            "complete": final_accuracy_complete,
            "blockers": [
                blocker for blocker in [
                    None if llama_ready else "LLaMA3-8B authorized checkpoint is missing",
                    None if qwen_ready else "Qwen3-8B checkpoint is missing",
                    None if final_accuracy_complete else "Full two-model x all-method x all-task accuracy matrix has not been run",
                    None if final_accuracy_complete else "Final MMLU results must include every configured MMLU subtask score, not only an aggregate or smoke subtask",
                    None if final_complete else "Final completion_audit.json ok=true is missing",
                ] if blocker
            ],
        },
        "deployment_tool": {
            "requirement": "Measure inference GPU memory and tokens/s for compressed LLaMA3-8B and Qwen3-8B on the 24GB 4090 target.",
            "implemented_entrypoints": [
                "python -m edge_benchmark deploy",
                "python -m edge_benchmark pipeline --compression-method <method>",
                "scripts/run_edge_deploy.sh",
            ],
            "target_memory_limit_gb": target_defaults.get("deployment_memory_limit_gb"),
            "target_deployment_batch_size": target_defaults.get("deployment_batch_size"),
            "current_evidence": {
                "qwen_dense_deployment_diagnostic": qwen_deployment,
                "qwen_gptq_runtime_probe": qwen_gptq_runtime_probe,
                "qwen_awq_runtime_probe": qwen_awq_runtime_probe,
                "qwen_smoothquant_runtime_probe": qwen_smoothquant_runtime_probe,
                "qwen_omniquant_runtime_probe": qwen_omniquant_runtime_probe,
                "llama_gptq_runtime_probe": llama_gptq_runtime_probe,
                "llama_awq_runtime_probe": llama_awq_runtime_probe,
                "llama_smoothquant_runtime_probe": llama_smoothquant_runtime_probe,
                "final_deployment_rows_complete": final_deployment_rows_complete,
            },
            "complete": final_deployment_complete,
            "blockers": [
                blocker for blocker in [
                    None if llama_ready else "LLaMA3-8B authorized checkpoint is missing",
                    None if qwen_ready else "Qwen3-8B checkpoint is missing",
                    None if not qwen_deployment.get("exceeds_24gb_target") else "Current Qwen dense deployment diagnostic exceeds 24GB and is not final",
                    None if target_platform_complete else "Target Python 3.10 / RTX 4090 deployment environment has not been verified",
                    None if final_deployment_complete else "Full two-model x all-method deployment matrix has not been run",
                ] if blocker
            ],
        },
        "resource_tool": {
            "requirement": "Measure compute complexity and storage footprint for compressed LLaMA3-8B and Qwen3-8B.",
            "implemented_entrypoints": [
                "python -m edge_benchmark resource",
                "python -m edge_benchmark pipeline --compression-method <method>",
                "scripts/run_edge_resource.sh",
            ],
            "current_evidence": {
                "qwen_dense_resource_calflops": qwen_dense_resource,
                "qwen_wanda_resource_smoke": qwen_wanda_resource,
                "qwen_sparsegpt_resource_smoke": qwen_sparsegpt_resource,
                "qwen_gptq_resource_precompressed": qwen_gptq_resource,
                "qwen_awq_resource_precompressed": qwen_awq_resource,
                "qwen_smoothquant_resource_precompressed": qwen_smoothquant_resource,
                "qwen_omniquant_resource_precompressed": qwen_omniquant_resource,
                "llama_gptq_resource_precompressed": llama_gptq_resource,
                "llama_awq_resource_precompressed": llama_awq_resource,
                "llama_smoothquant_resource_precompressed": llama_smoothquant_resource,
                "final_resource_rows_complete": final_resource_rows_complete,
            },
            "complete": final_resource_complete,
            "blockers": [
                blocker for blocker in [
                    None if llama_ready else "LLaMA3-8B authorized checkpoint is missing",
                    None if qwen_ready else "Qwen3-8B checkpoint is missing",
                    None if final_resource_complete else "Full two-model x all-method resource matrix has not been run",
                ] if blocker
            ],
        },
        "method_coverage": {
            "target_methods": target_methods,
            "ready_methods": ready_methods,
            "method_readiness": method_readiness,
            "complete": set(ready_methods) >= set(target_methods),
        },
        "target_platform": {
            "requirement": "Final results must come from a standardized multicore CPU / 256GB RAM / Python 3.10 / PyTorch 2.x / Transformers / RTX 4090 24GB / CUDA 12.x / cuDNN environment using HF_ENDPOINT=https://hf-mirror.com.",
            "configured_controls": {
                "min_free_gpu_memory_gb": target_defaults.get("min_free_gpu_memory_gb"),
                "dtype": target_defaults.get("dtype"),
                "max_memory": target_defaults.get("max_memory"),
                "deployment_memory_limit_gb": target_defaults.get("deployment_memory_limit_gb"),
                "deployment_batch_size": target_defaults.get("deployment_batch_size"),
            },
            "runtime_requirements": runtime_requirements,
            "complete": target_platform_complete,
            "blockers": [
                blocker for blocker in [
                    None if target_platform_complete else "Target multicore CPU / 256GB RAM / Python 3.10 / PyTorch 2.x / Transformers / RTX 4090 24GB / CUDA 12.x / cuDNN / hf-mirror dependency environment verification is missing",
                    None if final_complete else "Final completion_audit.json ok=true is missing",
                ] if blocker
            ],
        },
    }


def _git_value(args: list[str]) -> Optional[str]:
    try:
        proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _prompt_to_artifact_checklist(report: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = report.get("artifacts", {}) if isinstance(report.get("artifacts"), dict) else {}
    checklist = report.get("requirement_checklist", {}) if isinstance(report.get("requirement_checklist"), dict) else {}
    final_artifacts = report.get("final_artifacts", {}) if isinstance(report.get("final_artifacts"), dict) else {}
    code = report.get("code", {}) if isinstance(report.get("code"), dict) else {}
    final_complete = bool(final_artifacts.get("complete"))
    accuracy = checklist.get("accuracy_tool", {}) if isinstance(checklist.get("accuracy_tool"), dict) else {}
    deployment = checklist.get("deployment_tool", {}) if isinstance(checklist.get("deployment_tool"), dict) else {}
    resource = checklist.get("resource_tool", {}) if isinstance(checklist.get("resource_tool"), dict) else {}
    methods = checklist.get("method_coverage", {}) if isinstance(checklist.get("method_coverage"), dict) else {}
    platform = checklist.get("target_platform", {}) if isinstance(checklist.get("target_platform"), dict) else {}
    final_matrix = final_artifacts.get("matrix_results") if isinstance(final_artifacts.get("matrix_results"), dict) else {}
    final_rows = final_matrix.get("rows") if isinstance(final_matrix.get("rows"), list) else []
    target_models = accuracy.get("target_models", ["LLaMA3-8B", "Qwen3-8B"])
    target_methods = accuracy.get("target_methods", ["dense", "wanda", "sparsegpt", "gptq", "awq", "smoothquant", "omniquant"])
    target_tasks = accuracy.get("target_tasks", ["mmlu", "arc_challenge", "hellaswag", "piqa", "winogrande"])
    required_mmlu_subtasks = accuracy.get("required_mmlu_subtasks", DEFAULT_MMLU_SUBTASKS)
    required_mmlu_subtask_count = accuracy.get("required_mmlu_subtask_count", len(DEFAULT_MMLU_SUBTASKS))
    final_pairs = {
        (row.get("model_name"), row.get("compression_method"))
        for row in final_rows
        if isinstance(row, dict)
    }
    expected_pairs = {(model, method) for model in target_models for method in target_methods}
    missing_required_artifacts = (
        final_artifacts.get("missing_required_artifacts")
        if isinstance(final_artifacts.get("missing_required_artifacts"), list)
        else []
    )
    missing_required_artifact_names = [
        item.get("name")
        for item in missing_required_artifacts
        if isinstance(item, dict) and item.get("name")
    ]
    final_missing_or_unverified = []
    if final_artifacts.get("completion_audit_ok") is not True:
        final_missing_or_unverified.append("Final completion_audit.json ok=true for complete matrix outputs")
    if final_artifacts.get("validator_ok") is not True:
        final_missing_or_unverified.append("Final validate_edge_results.py sub-gate ok=true is missing")
    if (
        not isinstance(final_artifacts.get("results_bundle_verify"), dict)
        or final_artifacts.get("results_bundle_verify", {}).get("ok") is not True
        or final_artifacts.get("results_bundle_verify", {}).get("complete") is not True
        or final_artifacts.get("results_bundle_verify", {}).get("completion_validator_ok") is not True
        or final_artifacts.get("results_bundle_verify", {}).get("target_completion_ok") is not True
    ):
        final_missing_or_unverified.append("Final edge_results_bundle_verify.json ok=true and complete=true is missing")
    if (
        not isinstance(final_artifacts.get("delivery_verify"), dict)
        or final_artifacts.get("delivery_verify", {}).get("ok") is not True
        or final_artifacts.get("delivery_verify", {}).get("substatus_ok") is not True
    ):
        final_missing_or_unverified.append("Final edge_delivery_verify.json ok=true with all substatus checks passing is missing")
    if (
        not isinstance(final_artifacts.get("target_readiness"), dict)
        or final_artifacts.get("target_readiness", {}).get("completion_ok") is not True
        or final_artifacts.get("target_readiness", {}).get("before_matrix_blocker_count") != 0
    ):
        final_missing_or_unverified.append("Final target_readiness.json completion_ok=true with no before_matrix_blockers is missing")
    if not final_complete:
        final_missing_or_unverified.append("Final artifact gates are not all complete")
    if expected_pairs > final_pairs:
        final_missing_or_unverified.append("Full two-model x all-method matrix rows are missing")
    if missing_required_artifact_names:
        final_missing_or_unverified.append(
            "Missing required final artifacts: " + ", ".join(missing_required_artifact_names)
        )
    if final_artifacts.get("matrix_results", {}).get("exists") is not True:
        final_missing_or_unverified.append("Final matrix_results.json is missing")
    if final_artifacts.get("matrix_results_csv", {}).get("exists") is not True:
        final_missing_or_unverified.append("Final matrix_results.csv is missing")
    if final_artifacts.get("edge_report", {}).get("exists") is not True:
        final_missing_or_unverified.append("Final edge_report.md is missing")
    return {
        "accuracy_tool": {
            "prompt_requirement": "Output compressed-model accuracy on MMLU, ARC, HellaSwag, PIQA and WinoGrande for LLaMA3-8B and Qwen3-8B across sparse and quantized methods.",
            "artifacts": [
                artifacts.get("cli"),
                artifacts.get("matrix_runner"),
                "matrix_results.json",
                "matrix_results.csv",
                "edge_report.md",
                artifacts.get("scorecard"),
                "edge_scorecard.json",
                "edge_scorecard.md",
                "completion_audit.json",
            ],
            "gates": [
                "tools/validate_edge_results.py",
                "tools/audit_edge_completion.py",
                "split_accuracy_tasks=true",
                "full MMLU subtask coverage",
                "completion_audit.json ok=true",
            ],
            "evidence": {
                "target_models": target_models,
                "target_methods": target_methods,
                "target_tasks": target_tasks,
                "required_mmlu_subtask_count": required_mmlu_subtask_count,
                "required_mmlu_subtasks": required_mmlu_subtasks,
                "final_pair_count": len(final_pairs),
                "expected_pair_count": len(expected_pairs),
                "final_covers_all_pairs": expected_pairs <= final_pairs,
                "final_rows_with_full_mmlu_subtask_accuracy": sum(
                    1 for row in final_rows if isinstance(row, dict) and row.get("has_full_mmlu_subtask_accuracy")
                ),
                "complete": accuracy.get("complete"),
            },
            "missing_or_unverified": accuracy.get("blockers", []),
            "complete": bool(accuracy.get("complete")),
        },
        "deployment_tool": {
            "prompt_requirement": "Output inference memory usage and tokens/s for compressed models on the 24GB edge device target.",
            "artifacts": [
                artifacts.get("cli"),
                "deployment.json",
                "matrix_results.json",
                "edge_report.md",
                "completion_audit.json",
            ],
            "gates": [
                "deployment_memory_limit_gb=24",
                "tools/validate_edge_results.py memory checks",
                "completion_audit runtime/code gates",
            ],
            "evidence": {
                "target_memory_limit_gb": deployment.get("target_memory_limit_gb"),
                "complete": deployment.get("complete"),
            },
            "missing_or_unverified": deployment.get("blockers", []),
            "complete": bool(deployment.get("complete")),
        },
        "resource_tool": {
            "prompt_requirement": "Output compute complexity and storage footprint for compressed models.",
            "artifacts": [
                artifacts.get("cli"),
                "resource.json",
                "matrix_results.json",
                "edge_report.md",
                "completion_audit.json",
            ],
            "gates": [
                "tools/validate_edge_results.py resource backend checks",
                "resource_backend=config_only_fallback rejected in final results",
                "completion_audit source artifact checks",
            ],
            "evidence": {
                "complete": resource.get("complete"),
            },
            "missing_or_unverified": resource.get("blockers", []),
            "complete": bool(resource.get("complete")),
        },
        "method_and_model_coverage": {
            "prompt_requirement": "Cover LLaMA3-8B and Qwen3-8B across Wanda, SparseGPT, GPTQ, AWQ, SmoothQuant and OmniQuant, plus dense baseline.",
            "artifacts": [
                artifacts.get("matrix_config"),
                artifacts.get("compression"),
                artifacts.get("method_readiness"),
                "matrix_config_resolved.json",
                "matrix_results.json",
            ],
            "gates": [
                "tools/check_method_readiness.py",
                "tools/audit_edge_assets.py",
                "tools/audit_edge_completion.py expected pair coverage",
            ],
            "evidence": {
                "target_models": target_models,
                "target_methods": target_methods,
                "ready_methods": methods.get("ready_methods"),
                "complete": methods.get("complete"),
            },
            "missing_or_unverified": [
                payload.get("blockers")
                for payload in (methods.get("method_readiness") or {}).values()
                if isinstance(payload, dict) and payload.get("blockers")
            ],
            "complete": bool(methods.get("complete")),
        },
        "portable_environment_and_handoff": {
            "prompt_requirement": "Use hf-mirror downloads, 4090 24GB memory controls, README setup flow and portable handoff artifacts.",
            "artifacts": [
                artifacts.get("matrix_config"),
                artifacts.get("target_storage_init"),
                artifacts.get("full_4090_workflow"),
                artifacts.get("matrix_resume_plan"),
                artifacts.get("handoff_manifest"),
                artifacts.get("portable_bundle"),
                artifacts.get("portable_bundle_verify"),
                artifacts.get("run_status"),
                artifacts.get("completion_audit"),
                "PORTABLE_BUNDLE_MANIFEST.json",
                "RUN_ON_TARGET.md",
                "edge_handoff_manifest.json",
                "target_readiness.json",
            ],
            "gates": [
                "HF_ENDPOINT=https://hf-mirror.com",
                "scripts/init_edge_storage.sh",
                "max_memory=0:22GiB,cpu:120GiB",
                "deployment_memory_limit_gb=24",
                "audit_target_readiness.py --require-handoff-manifest",
                "audit_target_readiness.py --require-non-rotational-storage",
                "audit_target_readiness.py --require-optional-backends-report",
                "audit_target_readiness.py --require-quant-runtime-report",
                "audit_target_readiness.py --require-prepare-precompressed-report",
                "audit_edge_completion.py --require-handoff-manifest",
                "audit_edge_completion.py --require-target-readiness-report",
                "audit_edge_completion.py --require-non-rotational-storage",
                "audit_edge_completion.py --require-optional-backends-report",
                "audit_edge_completion.py --require-quant-runtime-report",
                "audit_edge_completion.py --require-prepare-precompressed-report",
            ],
            "evidence": {
                "configured_controls": platform.get("configured_controls"),
                "runtime_requirements": platform.get("runtime_requirements"),
                "remote_tracking_in_sync": code.get("remote_tracking_in_sync"),
                "portable_bundle_verify": report.get("portable_bundle_verify"),
                "target_readiness_next_command_groups": final_artifacts.get("target_readiness", {}).get("next_command_group_names"),
                "has_precompressed_config_followup_matrix": final_artifacts.get("target_readiness", {}).get("has_precompressed_config_followup_matrix"),
                "has_baseline_finalize_matrix": final_artifacts.get("target_readiness", {}).get("has_baseline_finalize_matrix"),
                "has_precompressed_config_post_run_readiness": final_artifacts.get("target_readiness", {}).get("has_precompressed_config_post_run_readiness"),
                "has_baseline_post_run_readiness": final_artifacts.get("target_readiness", {}).get("has_baseline_post_run_readiness"),
                "complete": platform.get("complete"),
            },
            "missing_or_unverified": platform.get("blockers", []),
            "complete": bool(platform.get("complete") and code.get("remote_tracking_in_sync")),
        },
        "final_two_model_results": {
            "prompt_requirement": "Provide complete final benchmark results for both models and all methods.",
            "artifacts": [
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
                artifacts.get("scorecard"),
                "edge_scorecard.json",
                "edge_scorecard.md",
                artifacts.get("results_bundle"),
                artifacts.get("results_bundle_verify"),
                artifacts.get("delivery_verify"),
                "RESULTS_BUNDLE_MANIFEST.json",
                "edge_results_bundle.tar.gz",
                "edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json",
                "edge_results_bundle_verify.json",
                "edge_delivery_verify.json",
                "completion_audit.json",
                "target_readiness.json",
                "edge_target_paths_report.json",
            ],
            "gates": [
                "completion_audit.json ok=true",
                "completion_audit validator sub-gate ok=true",
                "target_readiness.json ok=true and completion_ok=true",
                "edge_results_bundle_verify.json ok=true and complete=true",
                "edge_delivery_verify.json ok=true with all substatus checks passing",
                "full two-model x all-method expected coverage",
                "clean worktree and code-match gates",
            ],
            "evidence": {
                "final_artifacts_complete": final_complete,
                "completion_audit_ok": final_artifacts.get("completion_audit_ok"),
                "validator_ok": final_artifacts.get("validator_ok"),
                "validator": final_artifacts.get("validator"),
                "results_bundle_verify_ok": final_artifacts.get("results_bundle_verify", {}).get("ok")
                if isinstance(final_artifacts.get("results_bundle_verify"), dict)
                else None,
                "results_bundle_verify_complete": final_artifacts.get("results_bundle_verify", {}).get("complete")
                if isinstance(final_artifacts.get("results_bundle_verify"), dict)
                else None,
                "results_bundle_completion_validator_ok": final_artifacts.get("results_bundle_verify", {}).get("completion_validator_ok")
                if isinstance(final_artifacts.get("results_bundle_verify"), dict)
                else None,
                "results_bundle_target_completion_ok": final_artifacts.get("results_bundle_verify", {}).get("target_completion_ok")
                if isinstance(final_artifacts.get("results_bundle_verify"), dict)
                else None,
                "delivery_verify_ok": final_artifacts.get("delivery_verify", {}).get("ok")
                if isinstance(final_artifacts.get("delivery_verify"), dict)
                else None,
                "delivery_verify_substatus_ok": final_artifacts.get("delivery_verify", {}).get("substatus_ok")
                if isinstance(final_artifacts.get("delivery_verify"), dict)
                else None,
                "delivery_completion_validator_ok": final_artifacts.get("delivery_verify", {}).get("completion_validator_ok")
                if isinstance(final_artifacts.get("delivery_verify"), dict)
                else None,
                "delivery_target_completion_ok": final_artifacts.get("delivery_verify", {}).get("target_completion_ok")
                if isinstance(final_artifacts.get("delivery_verify"), dict)
                else None,
                "target_readiness_ok": final_artifacts.get("target_readiness", {}).get("ok")
                if isinstance(final_artifacts.get("target_readiness"), dict)
                else None,
                "target_completion_ok": final_artifacts.get("target_readiness", {}).get("completion_ok")
                if isinstance(final_artifacts.get("target_readiness"), dict)
                else None,
                "all_artifacts_present": final_artifacts.get("all_artifacts_present"),
                "missing_required_artifact_count": final_artifacts.get("missing_required_artifact_count"),
                "missing_required_artifacts": final_artifacts.get("missing_required_artifacts"),
                "matrix_results_json_exists": final_artifacts.get("matrix_results", {}).get("exists")
                if isinstance(final_artifacts.get("matrix_results"), dict)
                else None,
                "matrix_results_csv_exists": final_artifacts.get("matrix_results_csv", {}).get("exists")
                if isinstance(final_artifacts.get("matrix_results_csv"), dict)
                else None,
                "edge_report_exists": final_artifacts.get("edge_report", {}).get("exists")
                if isinstance(final_artifacts.get("edge_report"), dict)
                else None,
                "final_pair_count": len(final_pairs),
                "expected_pair_count": len(expected_pairs),
                "final_matrix_expected_pair_count": final_artifacts.get("final_matrix_expected_pair_count"),
                "final_matrix_actual_pair_count": final_artifacts.get("final_matrix_actual_pair_count"),
                "final_matrix_missing_pairs": final_artifacts.get("final_matrix_missing_pairs"),
                "final_matrix_unexpected_pairs": final_artifacts.get("final_matrix_unexpected_pairs"),
                "final_matrix_duplicate_pairs": final_artifacts.get("final_matrix_duplicate_pairs"),
                "final_matrix_invalid_row_count": final_matrix.get("invalid_row_count")
                if isinstance(final_matrix, dict)
                else None,
                "final_matrix_run_complete": final_artifacts.get("final_matrix_run_complete"),
            },
            "missing_or_unverified": final_missing_or_unverified,
            "complete": final_complete and expected_pairs <= final_pairs,
        },
    }


def _completion_state(report: Dict[str, Any]) -> Dict[str, Any]:
    code = report.get("code", {}) if isinstance(report.get("code"), dict) else {}
    final_artifacts = report.get("final_artifacts", {}) if isinstance(report.get("final_artifacts"), dict) else {}
    target_path_access = (
        report.get("target_path_access")
        if isinstance(report.get("target_path_access"), dict)
        else {}
    )
    prompt_checklist = (
        report.get("prompt_to_artifact_checklist")
        if isinstance(report.get("prompt_to_artifact_checklist"), dict)
        else {}
    )
    remaining = report.get("remaining_for_completion")
    if not isinstance(remaining, list):
        remaining = []
    incomplete_sections = [
        name
        for name, section in prompt_checklist.items()
        if isinstance(section, dict) and section.get("complete") is not True
    ]
    all_prompt_sections_complete = bool(prompt_checklist) and not incomplete_sections
    code_remote_tracking_in_sync = code.get("remote_tracking_in_sync") is True
    complete = all_prompt_sections_complete and code_remote_tracking_in_sync and not remaining
    target_readiness = (
        final_artifacts.get("target_readiness")
        if isinstance(final_artifacts.get("target_readiness"), dict)
        else {}
    )
    results_bundle_verify = (
        final_artifacts.get("results_bundle_verify")
        if isinstance(final_artifacts.get("results_bundle_verify"), dict)
        else {}
    )
    delivery_verify = (
        final_artifacts.get("delivery_verify")
        if isinstance(final_artifacts.get("delivery_verify"), dict)
        else {}
    )
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "code_remote_tracking_in_sync": code_remote_tracking_in_sync,
        "all_prompt_sections_complete": all_prompt_sections_complete,
        "incomplete_prompt_sections": incomplete_sections,
        "remaining_for_completion": remaining,
        "final_output_root": final_artifacts.get("output_root"),
        "final_artifacts_complete": bool(final_artifacts.get("complete")),
        "completion_audit_ok": bool(final_artifacts.get("completion_audit_ok")),
        "validator_ok": final_artifacts.get("validator_ok"),
        "results_bundle_verify_ok": results_bundle_verify.get("ok"),
        "results_bundle_verify_complete": results_bundle_verify.get("complete"),
        "results_bundle_completion_validator_ok": results_bundle_verify.get("completion_validator_ok"),
        "results_bundle_target_completion_ok": results_bundle_verify.get("target_completion_ok"),
        "delivery_verify_ok": delivery_verify.get("ok"),
        "delivery_verify_substatus_ok": delivery_verify.get("substatus_ok"),
        "delivery_completion_validator_ok": delivery_verify.get("completion_validator_ok"),
        "delivery_target_completion_ok": delivery_verify.get("target_completion_ok"),
        "target_readiness_ok": target_readiness.get("ok") is True,
        "target_completion_ok": target_readiness.get("completion_ok") is True,
        "target_ready_for_matrix": target_readiness.get("ready_for_matrix"),
        "target_path_access_ok": target_path_access.get("ok"),
        "target_path_access_blockers": target_path_access.get("blockers"),
        "missing_required_artifact_count": final_artifacts.get("missing_required_artifact_count"),
        "missing_required_artifacts": final_artifacts.get("missing_required_artifacts"),
        "next_command_group_names": target_readiness.get("next_command_group_names"),
    }


def build(args: argparse.Namespace) -> dict:
    qwen_dir = Path(args.qwen_path).expanduser()
    llama_dir = Path(args.llama_path).expanduser()
    target_config = _read_config(args.config)
    target_defaults = target_config.get("defaults", {}) if isinstance(target_config.get("defaults"), dict) else {}
    target_path_access = _target_path_access(target_defaults)
    qwen_checkpoint = _checkpoint_summary(
        qwen_dir,
        [
            "config.json",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
        ],
    )
    llama_checkpoint = _checkpoint_summary(llama_dir, ["config.json"])
    qwen_accuracy = _accuracy_summary(args.qwen_five_task_accuracy)
    qwen_wanda_accuracy = _accuracy_summary(args.qwen_wanda_accuracy)
    qwen_sparsegpt_accuracy = _accuracy_summary(args.qwen_sparsegpt_accuracy)
    qwen_wanda_mmlu_accuracy = _accuracy_summary(args.qwen_wanda_mmlu_accuracy)
    qwen_sparsegpt_mmlu_accuracy = _accuracy_summary(args.qwen_sparsegpt_mmlu_accuracy)
    qwen_deployment = _deployment_summary(args.qwen_dense_deployment)
    qwen_gptq_runtime_probe = _runtime_probe_summary(args.qwen_gptq_runtime_probe)
    qwen_awq_runtime_probe = _runtime_probe_summary(args.qwen_awq_runtime_probe)
    qwen_smoothquant_runtime_probe = _runtime_probe_summary(args.qwen_smoothquant_runtime_probe)
    qwen_omniquant_runtime_probe = _runtime_probe_summary(args.qwen_omniquant_runtime_probe)
    llama_gptq_runtime_probe = _runtime_probe_summary(args.llama_gptq_runtime_probe)
    llama_awq_runtime_probe = _runtime_probe_summary(args.llama_awq_runtime_probe)
    llama_smoothquant_runtime_probe = _runtime_probe_summary(args.llama_smoothquant_runtime_probe)
    qwen_dense_resource = _resource_summary(args.qwen_dense_resource)
    qwen_wanda_resource = _resource_summary(args.qwen_wanda_resource)
    qwen_sparsegpt_resource = _resource_summary(args.qwen_sparsegpt_resource)
    qwen_gptq_resource = _resource_summary(args.qwen_gptq_resource)
    qwen_awq_resource = _resource_summary(args.qwen_awq_resource)
    qwen_smoothquant_resource = _resource_summary(args.qwen_smoothquant_resource)
    qwen_omniquant_resource = _resource_summary(args.qwen_omniquant_resource)
    llama_gptq_resource = _resource_summary(args.llama_gptq_resource)
    llama_awq_resource = _resource_summary(args.llama_awq_resource)
    llama_smoothquant_resource = _resource_summary(args.llama_smoothquant_resource)
    qwen_validation = _validation_summary(args.qwen_dense_validation)
    qwen_partial_results = _matrix_results_summary(args.qwen_partial_results)
    portable_bundle_verify = _portable_bundle_verify_summary(args.portable_bundle_verify)
    final_artifacts = _final_artifacts_summary(args.final_output_root)
    method_readiness = _method_blockers(
        args.config,
        args.local_config,
        bool(llama_checkpoint["checkpoint_ready"]),
        bool(qwen_checkpoint["checkpoint_ready"]),
    )
    method_readiness = _merge_target_method_readiness(method_readiness, final_artifacts, args.config)
    requirement_checklist = _requirement_checklist(
        qwen_checkpoint,
        llama_checkpoint,
        qwen_accuracy,
        qwen_wanda_accuracy,
        qwen_sparsegpt_accuracy,
        qwen_wanda_mmlu_accuracy,
        qwen_sparsegpt_mmlu_accuracy,
        qwen_deployment,
        qwen_gptq_runtime_probe,
        qwen_awq_runtime_probe,
        qwen_smoothquant_runtime_probe,
        qwen_omniquant_runtime_probe,
        llama_gptq_runtime_probe,
        llama_awq_runtime_probe,
        llama_smoothquant_runtime_probe,
        qwen_dense_resource,
        qwen_wanda_resource,
        qwen_sparsegpt_resource,
        qwen_gptq_resource,
        qwen_awq_resource,
        qwen_smoothquant_resource,
        qwen_omniquant_resource,
        llama_gptq_resource,
        llama_awq_resource,
        llama_smoothquant_resource,
        target_defaults,
        method_readiness,
        qwen_partial_results,
        final_artifacts,
    )
    branch = _git_value(["branch", "--show-current"])
    head_commit = _git_value(["rev-parse", "HEAD"])
    upstream_ref = _git_value(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    remote_tracking_ref = upstream_ref or (f"origin/{branch}" if branch else None)
    remote_tracking_commit = _git_value(["rev-parse", remote_tracking_ref]) if remote_tracking_ref else None
    report = {
        "code": {
            "expected_branch": args.branch,
            "actual_branch": branch,
            "commit": head_commit,
            "upstream": upstream_ref,
            "remote_tracking_ref": remote_tracking_ref,
            "remote_tracking_commit": remote_tracking_commit,
            "remote_tracking_in_sync": bool(
                head_commit and remote_tracking_commit and head_commit == remote_tracking_commit
            ),
            "remote": args.remote,
            "origin_push_url": _git_value(["remote", "get-url", "--push", "origin"]),
        },
        "artifacts": {
            "cli": "edge_benchmark/cli.py",
            "compression": "edge_benchmark/compression.py",
            "matrix_config": args.config,
            "local_config": args.local_config,
            "run_status": "docs/RUN_STATUS.md",
            "completion_audit": "docs/COMPLETION_AUDIT.md",
            "base_requirements": "requirements.txt",
            "optional_requirements": "requirements-optional.txt",
            "asset_audit": "tools/audit_edge_assets.py",
            "external_method_config": "tools/configure_external_methods.py",
            "target_storage_init": "scripts/init_edge_storage.sh",
            "full_4090_workflow": "scripts/run_edge_4090_full.sh",
            "matrix_runner": "tools/run_edge_matrix.py",
            "matrix_resume_plan": "tools/plan_edge_matrix_resume.py",
            "scorecard": "tools/summarize_edge_results.py",
            "matrix_finalize_modes": ["--finalize", "--finalize-only"],
            "results_bundle": "tools/export_edge_results_bundle.py",
            "results_bundle_verify": "tools/verify_edge_results_bundle.py",
            "delivery_verify": "tools/verify_edge_delivery.py",
            "final_completion_audit": "tools/audit_edge_completion.py",
            "handoff_manifest": "tools/build_edge_handoff_manifest.py",
            "portable_bundle": "tools/export_edge_portable_bundle.py",
            "portable_bundle_verify": "tools/verify_edge_portable_bundle.py",
            "dataset_prewarm": "tools/prewarm_lm_eval_tasks.py --config configs/edge_4090_matrix.json --expand-mmlu",
            "target_preflight": "tools/run_edge_preflight.py --config configs/edge_4090_matrix.json --check-hf-access --check-model-forward --finalize-dry-run --cuda-visible-devices 0 --check-optional-backend-resolution --check-quant-runtime",
            "optional_backend_setup": "scripts/setup_optional_backends.sh",
            "method_readiness": "tools/check_method_readiness.py",
            "qwen_partial_report": args.qwen_partial_report,
            "qwen_partial_results": args.qwen_partial_results,
        },
        "qwen3": {
            **qwen_checkpoint,
            "dense_partial_validation": qwen_validation,
            "partial_report": {
                "markdown_exists": _exists(args.qwen_partial_report),
                "markdown_path": str(Path(args.qwen_partial_report).expanduser()),
                "matrix_results": qwen_partial_results,
            },
            "dense_five_task_split_accuracy_smoke": qwen_accuracy,
            "wanda_four_task_accuracy_smoke": qwen_wanda_accuracy,
            "sparsegpt_four_task_accuracy_smoke": qwen_sparsegpt_accuracy,
            "wanda_mmlu_subtask_accuracy_smoke": qwen_wanda_mmlu_accuracy,
            "sparsegpt_mmlu_subtask_accuracy_smoke": qwen_sparsegpt_mmlu_accuracy,
            "dense_deployment_diagnostic": qwen_deployment,
            "gptq_runtime_probe": qwen_gptq_runtime_probe,
            "awq_runtime_probe": qwen_awq_runtime_probe,
            "smoothquant_runtime_probe": qwen_smoothquant_runtime_probe,
            "omniquant_runtime_probe": qwen_omniquant_runtime_probe,
            "dense_resource_calflops": qwen_dense_resource,
            "wanda_resource_smoke": qwen_wanda_resource,
            "sparsegpt_resource_smoke": qwen_sparsegpt_resource,
            "gptq_resource_precompressed": qwen_gptq_resource,
            "awq_resource_precompressed": qwen_awq_resource,
            "smoothquant_resource_precompressed": qwen_smoothquant_resource,
            "omniquant_resource_precompressed": qwen_omniquant_resource,
            "current_verified_smoke_scope": {
                "dense_accuracy_all_target_tasks_limit_1": bool(qwen_accuracy.get("all_target_tasks_succeeded")),
                "wanda_accuracy_four_target_tasks_limit_1": bool(
                    qwen_wanda_accuracy.get("exists")
                    and all(qwen_wanda_accuracy.get("scores", {}).get(task) is not None for task in ["arc_challenge", "hellaswag", "piqa", "winogrande"])
                ),
                "wanda_accuracy_mmlu_subtask_limit_1": bool(qwen_wanda_mmlu_accuracy.get("mmlu_subtask_count")),
                "sparsegpt_accuracy_four_target_tasks_limit_1": bool(
                    qwen_sparsegpt_accuracy.get("exists")
                    and all(qwen_sparsegpt_accuracy.get("scores", {}).get(task) is not None for task in ["arc_challenge", "hellaswag", "piqa", "winogrande"])
                ),
                "sparsegpt_accuracy_mmlu_subtask_limit_1": bool(qwen_sparsegpt_mmlu_accuracy.get("mmlu_subtask_count")),
                "dense_deployment_code_path": bool(qwen_deployment.get("exists")),
                "dense_deployment_within_24gb": bool(
                    qwen_deployment.get("exists") and not qwen_deployment.get("exceeds_24gb_target")
                ),
                "gptq_runtime_imports_ok": bool(qwen_gptq_runtime_probe.get("runtime_imports_ok")),
                "awq_runtime_model_load_ok": bool(qwen_awq_runtime_probe.get("model_load_ok")),
                "smoothquant_runtime_config_tokenizer_ok": bool(
                    qwen_smoothquant_runtime_probe.get("config_ok")
                    and qwen_smoothquant_runtime_probe.get("tokenizer_ok")
                ),
                "omniquant_runtime_probe_exists": bool(qwen_omniquant_runtime_probe.get("exists")),
                "dense_resource_calflops": qwen_dense_resource.get("compute_backend") == "calflops",
                "wanda_resource_tiny_calibration": bool(qwen_wanda_resource.get("exists")),
                "wanda_compressed_checkpoint_ready": bool(qwen_wanda_resource.get("compressed_checkpoint_ready")),
                "sparsegpt_resource_tiny_calibration": bool(qwen_sparsegpt_resource.get("exists")),
                "sparsegpt_compressed_checkpoint_ready": bool(qwen_sparsegpt_resource.get("compressed_checkpoint_ready")),
                "gptq_precompressed_resource_config_fallback": bool(
                    qwen_gptq_resource.get("exists")
                    and qwen_gptq_resource.get("resource_backend") == "config_only_fallback"
                ),
                "awq_precompressed_resource_config_fallback": bool(
                    qwen_awq_resource.get("exists")
                    and qwen_awq_resource.get("resource_backend") == "config_only_fallback"
                ),
                "smoothquant_precompressed_resource_config_fallback": bool(
                    qwen_smoothquant_resource.get("exists")
                    and qwen_smoothquant_resource.get("resource_backend") == "config_only_fallback"
                ),
                "omniquant_precompressed_resource_model_load": bool(
                    qwen_omniquant_resource.get("exists")
                    and qwen_omniquant_resource.get("resource_backend") == "model_load"
                ),
            },
        },
        "llama3": {
            **llama_checkpoint,
            "blocker": None if llama_checkpoint["checkpoint_ready"] else "gated HuggingFace access or local checkpoint required",
            "gptq_resource_precompressed": llama_gptq_resource,
            "gptq_runtime_probe": llama_gptq_runtime_probe,
            "awq_resource_precompressed": llama_awq_resource,
            "awq_runtime_probe": llama_awq_runtime_probe,
            "smoothquant_resource_precompressed": llama_smoothquant_resource,
            "smoothquant_runtime_probe": llama_smoothquant_runtime_probe,
        },
        "method_readiness_blockers": method_readiness,
        "requirement_checklist": requirement_checklist,
        "final_artifacts": final_artifacts,
        "portable_bundle_verify": portable_bundle_verify,
        "target_path_access": target_path_access,
        "target_4090_controls": {
            "config": args.config,
            "min_free_gpu_memory_gb": target_defaults.get("min_free_gpu_memory_gb"),
            "dtype": target_defaults.get("dtype"),
            "max_memory": target_defaults.get("max_memory"),
            "deployment_memory_limit_gb": target_defaults.get("deployment_memory_limit_gb"),
            "deployment_batch_size": target_defaults.get("deployment_batch_size"),
            "split_accuracy_tasks": target_defaults.get("split_accuracy_tasks"),
            "continue_on_accuracy_error": target_defaults.get("continue_on_accuracy_error"),
            "within_24gb_controls_configured": bool(
                target_defaults.get("dtype") in ("fp16", "bf16", "float16", "bfloat16")
                and target_defaults.get("max_memory")
                and target_defaults.get("deployment_memory_limit_gb")
            ),
        },
        "remaining_for_completion": [
            "LLaMA3-8B authorized checkpoint",
            "Target multicore CPU / 256GB RAM / Python 3.10 / PyTorch 2.x / Transformers / 4090 24GB / CUDA 12.x / cuDNN / hf-mirror dependency environment verification",
            "Final target_readiness.json completion_ok=true with no before_matrix_blockers",
            "GPTQ/AWQ compression backend or complete precompressed checkpoints plus loading runtime",
            "SmoothQuant/OmniQuant external commands or complete precompressed checkpoints",
            "Full two-model x all-method x all-task matrix run",
            "Final completion_audit.json ok=true for complete matrix outputs",
            "Final validate_edge_results.py sub-gate ok=true",
            "Final edge_results_bundle_verify.json ok=true and complete=true",
            "Final edge_delivery_verify.json ok=true with all substatus checks passing",
        ],
    }
    if report["llama3"]["checkpoint_ready"]:
        report["remaining_for_completion"] = [
            item for item in report["remaining_for_completion"] if item != "LLaMA3-8B authorized checkpoint"
        ]
    if method_readiness["gptq"]["ready"] and method_readiness["awq"]["ready"]:
        report["remaining_for_completion"] = [
            item for item in report["remaining_for_completion"] if item != "GPTQ/AWQ compression backend or complete precompressed checkpoints plus loading runtime"
        ]
    if method_readiness["smoothquant"]["ready"] and method_readiness["omniquant"]["ready"]:
        report["remaining_for_completion"] = [
            item for item in report["remaining_for_completion"] if item != "SmoothQuant/OmniQuant external commands or complete precompressed checkpoints"
        ]
    if final_artifacts.get("final_matrix_run_complete") is True:
        report["remaining_for_completion"] = [
            item for item in report["remaining_for_completion"] if item != "Full two-model x all-method x all-task matrix run"
        ]
    if final_artifacts.get("completion_audit_ok") is True:
        report["remaining_for_completion"] = [
            item for item in report["remaining_for_completion"] if item != "Final completion_audit.json ok=true for complete matrix outputs"
        ]
    if final_artifacts.get("validator_ok") is True:
        report["remaining_for_completion"] = [
            item for item in report["remaining_for_completion"] if item != "Final validate_edge_results.py sub-gate ok=true"
        ]
    if (
        isinstance(final_artifacts.get("results_bundle_verify"), dict)
        and final_artifacts["results_bundle_verify"].get("ok") is True
        and final_artifacts["results_bundle_verify"].get("complete") is True
        and final_artifacts["results_bundle_verify"].get("completion_validator_ok") is True
        and final_artifacts["results_bundle_verify"].get("target_completion_ok") is True
    ):
        report["remaining_for_completion"] = [
            item for item in report["remaining_for_completion"] if item != "Final edge_results_bundle_verify.json ok=true and complete=true"
        ]
    if (
        isinstance(final_artifacts.get("delivery_verify"), dict)
        and final_artifacts["delivery_verify"].get("ok") is True
        and final_artifacts["delivery_verify"].get("substatus_ok") is True
    ):
        report["remaining_for_completion"] = [
            item for item in report["remaining_for_completion"] if item != "Final edge_delivery_verify.json ok=true with all substatus checks passing"
        ]
    if requirement_checklist["target_platform"]["complete"]:
        report["remaining_for_completion"] = [
            item for item in report["remaining_for_completion"] if item != "Target multicore CPU / 256GB RAM / Python 3.10 / PyTorch 2.x / Transformers / 4090 24GB / CUDA 12.x / cuDNN / hf-mirror dependency environment verification"
        ]
    if (
        isinstance(final_artifacts.get("target_readiness"), dict)
        and final_artifacts["target_readiness"].get("completion_ok") is True
        and final_artifacts["target_readiness"].get("before_matrix_blocker_count") == 0
    ):
        report["remaining_for_completion"] = [
            item for item in report["remaining_for_completion"] if item != "Final target_readiness.json completion_ok=true with no before_matrix_blockers"
        ]
    report["prompt_to_artifact_checklist"] = _prompt_to_artifact_checklist(report)
    report["completion_state"] = _completion_state(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a machine-readable audit snapshot for the active LLMCBench goal.")
    parser.add_argument("--branch", default="master")
    parser.add_argument("--remote", default="git@code.byted.org:data/llmcbench.git")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--local-config", default="configs/edge_local_h20_matrix.json")
    parser.add_argument("--qwen-path", default="/opt/tiger/models/Qwen3-8B")
    parser.add_argument("--llama-path", default="/opt/tiger/models/Meta-Llama-3-8B")
    parser.add_argument("--qwen-partial-report", default="/opt/tiger/llmcbench_outputs/qwen_smoke_report/edge_report.md")
    parser.add_argument("--qwen-partial-results", default="/opt/tiger/llmcbench_outputs/qwen_smoke_report/matrix_results.json")
    parser.add_argument("--qwen-dense-validation", default="/opt/tiger/llmcbench_outputs/qwen_smoke_report/validation_dense.json")
    parser.add_argument("--qwen-five-task-accuracy", default="/opt/tiger/llmcbench_outputs/qwen3_split_5task_smoke/accuracy.json")
    parser.add_argument("--qwen-wanda-accuracy", default="/opt/tiger/llmcbench_outputs/qwen3_wanda_deployable/accuracy_4task.json")
    parser.add_argument("--qwen-sparsegpt-accuracy", default="/opt/tiger/llmcbench_outputs/qwen3_sparsegpt_deployable/accuracy_4task.json")
    parser.add_argument("--qwen-wanda-mmlu-accuracy", default="/opt/tiger/llmcbench_outputs/qwen3_wanda_deployable/accuracy_mmlu_abstract_algebra.json")
    parser.add_argument("--qwen-sparsegpt-mmlu-accuracy", default="/opt/tiger/llmcbench_outputs/qwen3_sparsegpt_deployable/accuracy_mmlu_abstract_algebra.json")
    parser.add_argument("--qwen-dense-deployment", default="/opt/tiger/llmcbench_outputs/qwen3_dense_deploy_fp16_24gb_gate/deployment.json")
    parser.add_argument("--qwen-gptq-runtime-probe", default="/opt/tiger/llmcbench_outputs/qwen3_gptq_precompressed/runtime_probe.json")
    parser.add_argument("--qwen-awq-runtime-probe", default="/opt/tiger/llmcbench_outputs/qwen3_awq_precompressed/runtime_probe.json")
    parser.add_argument("--qwen-smoothquant-runtime-probe", default="/opt/tiger/llmcbench_outputs/qwen3_smoothquant_precompressed/runtime_probe.json")
    parser.add_argument("--qwen-omniquant-runtime-probe", default="/opt/tiger/llmcbench_outputs/qwen3_omniquant_precompressed/runtime_probe.json")
    parser.add_argument("--llama-gptq-runtime-probe", default="/opt/tiger/llmcbench_outputs/llama3_gptq_precompressed/runtime_probe.json")
    parser.add_argument("--llama-awq-runtime-probe", default="/opt/tiger/llmcbench_outputs/llama3_awq_precompressed/runtime_probe.json")
    parser.add_argument("--llama-smoothquant-runtime-probe", default="/opt/tiger/llmcbench_outputs/llama3_smoothquant_precompressed/runtime_probe.json")
    parser.add_argument("--qwen-dense-resource", default="/opt/tiger/llmcbench_outputs/qwen3_dense_resource_calflops/resource.json")
    parser.add_argument("--qwen-wanda-resource", default="/opt/tiger/llmcbench_outputs/qwen3_wanda_deployable/resource.json")
    parser.add_argument("--qwen-sparsegpt-resource", default="/opt/tiger/llmcbench_outputs/qwen3_sparsegpt_deployable/resource.json")
    parser.add_argument("--qwen-gptq-resource", default="/opt/tiger/llmcbench_outputs/qwen3_gptq_precompressed/resource.json")
    parser.add_argument("--qwen-awq-resource", default="/opt/tiger/llmcbench_outputs/qwen3_awq_precompressed/resource.json")
    parser.add_argument("--qwen-smoothquant-resource", default="/opt/tiger/llmcbench_outputs/qwen3_smoothquant_precompressed/resource.json")
    parser.add_argument("--qwen-omniquant-resource", default="/opt/tiger/llmcbench_outputs/qwen3_omniquant_precompressed/resource.json")
    parser.add_argument("--llama-gptq-resource", default="/opt/tiger/llmcbench_outputs/llama3_gptq_precompressed/resource.json")
    parser.add_argument("--llama-awq-resource", default="/opt/tiger/llmcbench_outputs/llama3_awq_precompressed/resource.json")
    parser.add_argument("--llama-smoothquant-resource", default="/opt/tiger/llmcbench_outputs/llama3_smoothquant_precompressed/resource.json")
    parser.add_argument("--portable-bundle-verify", default="/opt/tiger/llmcbench_outputs/edge_portable_bundle/portable_bundle_verify.json")
    parser.add_argument("--final-output-root", default="/data/outputs/llmcbench_edge_4090")
    parser.add_argument("--output", default="/opt/tiger/llmcbench_outputs/current_goal_audit.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build(args)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
