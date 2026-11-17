import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from edge_benchmark.checkpoint_inspection import inspect_quantization_storage
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from edge_benchmark.checkpoint_inspection import inspect_quantization_storage

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


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _best_accuracy_metric(metrics: Dict, task: str = "") -> tuple[Optional[float], Optional[str]]:
    for key in TASK_METRIC_PREFERENCES.get(task, ("acc,none", "acc_norm,none", "exact_match,none")):
        if key in metrics:
            return metrics[key], key
    for key, value in metrics.items():
        if key.startswith("acc") and isinstance(value, (int, float)):
            return value, key
    return None, None


def _add_aggregate_scores(row: Dict) -> None:
    values = []
    metric_keys = []
    for task in DEFAULT_MMLU_SUBTASKS:
        key = f"{task}_score"
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(value)
            metric_key = row.get(f"{key}_metric")
            if isinstance(metric_key, str) and metric_key:
                metric_keys.append(metric_key)
    if values:
        row["mmlu_score"] = sum(values) / len(values)
        if metric_keys:
            unique_metrics = sorted(set(metric_keys))
            if len(unique_metrics) == 1:
                row["mmlu_score_metric"] = f"aggregate:{unique_metrics[0]}"
            else:
                row["mmlu_score_metric"] = "aggregate:mixed"
    values = [row.get(field) for field in TARGET_SCORE_FIELDS if isinstance(row.get(field), (int, float))]
    if values:
        row["accuracy_avg_score"] = sum(values) / len(values)


def _add_dense_retention(rows: List[Dict]) -> None:
    dense_by_model = {
        row.get("model_name"): row.get("accuracy_avg_score")
        for row in rows
        if row.get("compression_method") == "dense" and isinstance(row.get("accuracy_avg_score"), (int, float))
    }
    for row in rows:
        dense_score = dense_by_model.get(row.get("model_name"))
        score = row.get("accuracy_avg_score")
        if isinstance(dense_score, (int, float)) and isinstance(score, (int, float)):
            row["accuracy_delta_vs_dense"] = score - dense_score
            row["accuracy_retention_vs_dense"] = score / dense_score if dense_score else None


def _compression_params(manifest: Dict) -> Dict:
    fields = [
        "quant_bits",
        "group_size",
        "desc_act",
        "awq_version",
        "sparsity",
        "sparsity_semantics",
        "prune_pattern",
        "prune_structure",
        "prune_n",
        "prune_m",
        "prune_lm_head",
        "pruning_algorithm",
        "pruning_buffer_device",
        "sparsegpt_blocksize",
        "sparsegpt_percdamp",
        "calibration_samples",
        "calibration_file",
        "calibration_batch_size",
        "calibration_protocol",
        "calibration_source",
        "calibration_sampling_algorithm",
        "calibration_seed",
        "calibration_dataset",
        "calibration_dataset_config",
        "calibration_data_file",
        "calibration_dataset_revision",
        "calibration_local_file",
        "calibration_token_length",
        "calibration_token_sha256",
        "pruning_calibration_source",
        "calibration_c4_data_file",
        "calibration_dataset_cache",
        "paper_protocol",
        "paper_protocol_attested",
        "base_model_repository",
        "base_model_revision",
        "base_model_attestation_path",
        "base_model_attested",
        "gptq_damp_percent",
        "gptq_static_groups",
        "gptq_symmetric",
        "gptq_true_sequential",
        "awq_zero_point",
        "awq_calibration_source",
        "awq_calibration_seed",
        "awq_grid_size",
        "awq_runtime_scale_search_grid",
        "awq_runtime_calibration_seed",
        "quantization_protocol",
        "quantization_source",
        "quantization_dataset",
        "quantization_dataset_split",
        "quantization_seed",
        "quantization_samples",
        "quantization_token_length",
        "smoothquant_alpha",
        "smoothquant_alpha_selection",
        "smoothquant_calibration_source",
        "smoothquant_calibration_sentences",
        "omniquant_abits",
        "omniquant_epochs",
        "omniquant_nsamples",
        "omniquant_seqlen",
        "omniquant_seed",
        "omniquant_lwc",
        "omniquant_let",
        "omniquant_aug_loss",
        "dtype",
        "seqlen",
        "max_memory",
        "attn_implementation",
        "materialized",
        "target_model_adaptation",
        "target_model_adaptation_disclaimer",
        "runtime_loading",
        "storage_quantization",
        "adaptation_calibration_source",
        "adaptation_calibration_seed",
        "adaptation_calibration_samples",
        "adaptation_calibration_sequence_length",
        "adaptation_calibration_processed_batches",
        "adaptation_calibration_activation_statistics",
        "note",
        "external_command",
        "external_returncode",
        "stdout_tail",
        "stderr_tail",
    ]
    params = {f"compression_{field}": manifest.get(field) for field in fields if field in manifest}
    official_implementation = manifest.get("official_implementation")
    if isinstance(official_implementation, dict):
        for field in (
            "official_code_dir",
            "official_repository",
            "official_commit",
            "official_implementation_verified",
        ):
            if field in official_implementation:
                params[f"compression_{field}"] = official_implementation[field]
    return params


def _pruning_compat_fields(manifest: Dict, method: str) -> Dict:
    fields = {}
    if method in {"wanda", "sparsegpt"} and manifest.get("backend") == "built_in_activation_pruning":
        pattern = manifest.get("prune_pattern") or "unstructured"
        fields = {
            "compression_prune_pattern": pattern,
            "compression_prune_structure": manifest.get("prune_structure")
            or ("semi_structured" if pattern == "2:4" else "unstructured"),
            "compression_sparsity_semantics": manifest.get("sparsity_semantics") or "element_zero_fraction",
            "compression_prune_n": manifest.get("prune_n") if pattern == "2:4" else None,
            "compression_prune_m": manifest.get("prune_m") if pattern == "2:4" else None,
        }
    return fields


def _runtime_fields(summary: Dict, accuracy: Optional[Dict], deployment: Optional[Dict], resource: Optional[Dict], compression: Optional[Dict]) -> Dict:
    runtime_sources = [
        summary.get("runtime"),
        (accuracy or {}).get("runtime"),
        (deployment or {}).get("runtime"),
        (resource or {}).get("runtime"),
        (compression or {}).get("runtime"),
    ]
    runtime: Dict = {}
    for source in runtime_sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if key not in runtime or runtime.get(key) in (None, "", []):
                runtime[key] = value
    gpus = runtime.get("gpus") if isinstance(runtime.get("gpus"), list) else []
    first_gpu = gpus[0] if gpus else {}
    optional_imports = runtime.get("optional_imports") if isinstance(runtime.get("optional_imports"), dict) else {}
    bitsandbytes = optional_imports.get("bitsandbytes") if isinstance(optional_imports.get("bitsandbytes"), dict) else {}
    accuracy_dataset_cache = (accuracy or {}).get("hf_datasets_cache")
    if accuracy_dataset_cache in (None, ""):
        accuracy_dataset_cache = _accuracy_value(accuracy, "hf_datasets_cache")
    return {
        "runtime_python": runtime.get("python"),
        "runtime_profile": runtime.get("runtime_profile"),
        "runtime_cpu_count": runtime.get("cpu_count"),
        "runtime_system_memory_total_gb": runtime.get("system_memory_total_gb"),
        "runtime_hf_endpoint": runtime.get("hf_endpoint"),
        "runtime_hf_home": runtime.get("hf_home"),
        "runtime_transformers_cache": runtime.get("transformers_cache"),
        "runtime_huggingface_hub_cache": runtime.get("huggingface_hub_cache"),
        "runtime_hf_datasets_cache": runtime.get("hf_datasets_cache") or accuracy_dataset_cache,
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


def _accuracy_failure_fields(accuracy: Optional[Dict]) -> Dict:
    failed_tasks = (accuracy or {}).get("failed_tasks")
    if not isinstance(failed_tasks, dict):
        failed_tasks = {}
    return {
        "accuracy_failed_task_count": len(failed_tasks),
        "accuracy_failed_tasks": ",".join(sorted(failed_tasks)),
    }


def _accuracy_value(accuracy: Optional[Dict], key: str):
    if not accuracy:
        return None
    value = accuracy.get(key)
    if value not in (None, ""):
        return value
    task_outputs = accuracy.get("task_outputs")
    if not isinstance(task_outputs, dict):
        return value
    for path_value in task_outputs.values():
        try:
            task_payload = _load_json(Path(path_value))
        except Exception:
            task_payload = None
        task_value = (task_payload or {}).get(key)
        if task_value not in (None, ""):
            return task_value
    return value


def collect(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for summary_path in sorted(root.glob("**/summary.json")):
        job_dir = summary_path.parent
        compression = _load_json(job_dir / "compression.json")
        accuracy = _load_json(job_dir / "accuracy.json")
        deployment = _load_json(job_dir / "deployment.json")
        resource = _load_json(job_dir / "resource.json")
        summary = _load_json(summary_path) or {}

        model_name = summary.get("model_name") or job_dir.parent.name
        method = summary.get("compression_method") or job_dir.name
        manifest = (compression or {}).get("manifest", {})
        compressed_path = summary.get("compressed_model_path")
        quantized_method = method in {"gptq", "awq", "smoothquant", "omniquant"}
        quantization_storage = (
            inspect_quantization_storage(Path(compressed_path))
            if quantized_method and isinstance(compressed_path, str) and compressed_path
            else None
        )
        real_quantized_storage = bool(
            isinstance(quantization_storage, dict) and quantization_storage.get("is_real_quantized")
        )
        checkpoint_storage_bytes = (resource or {}).get("storage_bytes") if real_quantized_storage else None
        checkpoint_weight_bits = (
            quantization_storage.get("weight_bits")
            if isinstance(quantization_storage, dict)
            else None
        )
        if checkpoint_weight_bits is None:
            checkpoint_weight_bits = manifest.get("quant_bits")
        row = {
            "model_name": model_name,
            "compression_method": method,
            "source_model_path": manifest.get("source_model_path"),
            "model_revision": manifest.get("revision")
            or (compression or {}).get("revision")
            or (accuracy or {}).get("revision")
            or (deployment or {}).get("revision")
            or (resource or {}).get("revision"),
            "compressed_model_path": compressed_path,
            "compression_backend": manifest.get("backend"),
            "compression_uses_precompressed": manifest.get("backend") == "precompressed_checkpoint",
            **_compression_params(manifest),
            **_pruning_compat_fields(manifest, method),
            **_runtime_fields(summary, accuracy, deployment, resource, compression),
            **_accuracy_failure_fields(accuracy),
            "accuracy_dtype": (accuracy or {}).get("dtype"),
            "accuracy_lm_eval_dtype": (accuracy or {}).get("lm_eval_dtype"),
            "accuracy_batch_size": (accuracy or {}).get("batch_size"),
            "accuracy_num_fewshot": (accuracy or {}).get("num_fewshot"),
            "accuracy_limit": (accuracy or {}).get("limit"),
            "accuracy_device": (accuracy or {}).get("device"),
            "accuracy_data_dir": _accuracy_value(accuracy, "data_dir"),
            "accuracy_hf_datasets_cache": _accuracy_value(accuracy, "hf_datasets_cache"),
            "accuracy_max_memory": (accuracy or {}).get("max_memory"),
            "accuracy_attn_implementation": (accuracy or {}).get("attn_implementation"),
            "accuracy_split_tasks": (accuracy or {}).get("split_accuracy_tasks", False),
            "accuracy_continue_on_error": (accuracy or {}).get("continue_on_accuracy_error", False),
            "tokens_per_second": (deployment or {}).get("tokens_per_second"),
            "deployment_dtype": (deployment or {}).get("dtype"),
            "deployment_device_map": (deployment or {}).get("device_map"),
            "deployment_max_memory": (deployment or {}).get("max_memory"),
            "deployment_attn_implementation": (deployment or {}).get("attn_implementation"),
            "deployment_prompt_source": (deployment or {}).get("prompt_source"),
            "deployment_prompt_path": (deployment or {}).get("prompt_path"),
            "deployment_prompt_count": (deployment or {}).get("prompt_count"),
            "deployment_batch_size": (deployment or {}).get("deployment_batch_size"),
            "deployment_input_length": (deployment or {}).get("input_length"),
            "deployment_max_new_tokens": (deployment or {}).get("max_new_tokens"),
            "deployment_warmup": (deployment or {}).get("warmup"),
            "deployment_repeat": (deployment or {}).get("repeat"),
            "deployment_input_tokens": (deployment or {}).get("input_tokens"),
            "deployment_generated_tokens": (deployment or {}).get("generated_tokens"),
            "deployment_elapsed_seconds": (deployment or {}).get("elapsed_seconds"),
            "peak_memory_allocated_gb": (deployment or {}).get("peak_memory_allocated_gb"),
            "peak_memory_reserved_gb": (deployment or {}).get("peak_memory_reserved_gb"),
            "deployment_memory_limit_gb": (deployment or {}).get("deployment_memory_limit_gb"),
            "deployment_memory_measured_peak_gb": (deployment or {}).get("deployment_memory_measured_peak_gb"),
            "deployment_memory_limit_exceeded": (deployment or {}).get("deployment_memory_limit_exceeded"),
            "resource_backend": (resource or {}).get("resource_backend"),
            "resource_model_load_error": (resource or {}).get("model_load_error"),
            "resource_seqlen": (resource or {}).get("seqlen"),
            "resource_prompt_source": (resource or {}).get("prompt_source"),
            "resource_prompt_path": (resource or {}).get("prompt_path"),
            "resource_prompt_count": (resource or {}).get("prompt_count"),
            "resource_input_length": (resource or {}).get("input_length"),
            "resource_max_new_tokens": (resource or {}).get("max_new_tokens"),
            "resource_sparsity_sample_tensors": (resource or {}).get("sparsity_sample_tensors"),
            "storage_bytes": (resource or {}).get("storage_bytes"),
            "storage_gb": (resource or {}).get("storage_gb"),
            "effective_storage_bytes_sparse_approx": (resource or {}).get("effective_storage_bytes_sparse_approx"),
            "effective_storage_gb_sparse_approx": (resource or {}).get("effective_storage_gb_sparse_approx"),
            "effective_storage_bytes_quantized_approx": (
                checkpoint_storage_bytes
            ),
            "effective_storage_gb_quantized_approx": (
                checkpoint_storage_bytes / (1024**3)
                if checkpoint_storage_bytes is not None
                else None
            ),
            "effective_storage_quantized_bits": (
                checkpoint_weight_bits if real_quantized_storage else None
            ),
            "effective_storage_quantized_backend": (
                "checkpoint_directory_size" if real_quantized_storage else None
            ),
            "quantization_storage_format": (
                quantization_storage.get("format") if isinstance(quantization_storage, dict) else None
            ),
            "quantization_storage_real": (
                quantization_storage.get("is_real_quantized")
                if isinstance(quantization_storage, dict)
                else None
            ),
            "quantization_storage_dtypes": (
                ",".join(quantization_storage.get("dtypes") or [])
                if isinstance(quantization_storage, dict)
                else None
            ),
            "checkpoint_quant_method": (
                quantization_storage.get("quant_method") if isinstance(quantization_storage, dict) else None
            ),
            "checkpoint_weight_bits": (
                checkpoint_weight_bits if isinstance(quantization_storage, dict) else None
            ),
            "checkpoint_activation_bits": (
                quantization_storage.get("activation_bits") if isinstance(quantization_storage, dict) else None
            ),
            "checkpoint_group_size": (
                quantization_storage.get("group_size") if isinstance(quantization_storage, dict) else None
            ),
            "total_params": ((resource or {}).get("parameter_stats") or {}).get("total_params"),
            "sparsity": ((resource or {}).get("parameter_stats") or {}).get("sparsity"),
            "nonzero_ratio": ((resource or {}).get("parameter_stats") or {}).get("nonzero_ratio"),
            "flops": ((resource or {}).get("compute_complexity") or {}).get("flops"),
            "macs": ((resource or {}).get("compute_complexity") or {}).get("macs"),
            "flops_num": ((resource or {}).get("compute_complexity") or {}).get("flops_num"),
            "effective_flops_num_sparse_approx": ((resource or {}).get("compute_complexity") or {}).get("effective_flops_num_sparse_approx"),
            "compute_backend": ((resource or {}).get("compute_complexity") or {}).get("backend"),
            "compute_warning": ((resource or {}).get("compute_complexity") or {}).get("warning"),
        }
        if accuracy:
            for task, metrics in accuracy.get("results", {}).items():
                value, metric_key = _best_accuracy_metric(metrics, task)
                row[f"{task}_score"] = value
                if metric_key:
                    row[f"{task}_score_metric"] = metric_key
        _add_aggregate_scores(row)
        rows.append(row)
    _add_dense_retention(rows)
    return rows


def write_outputs(rows: List[Dict], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    if rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect LLMCBench edge matrix results.")
    parser.add_argument("--root", default="/data/outputs/llmcbench_edge_4090")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).expanduser()
    output_json = Path(args.output_json).expanduser() if args.output_json else root / "matrix_results.json"
    output_csv = Path(args.output_csv).expanduser() if args.output_csv else root / "matrix_results.csv"
    rows = collect(root)
    write_outputs(rows, output_json, output_csv)
    print(json.dumps({"rows": len(rows), "output_json": str(output_json), "output_csv": str(output_csv)}, indent=2))


if __name__ == "__main__":
    main()
