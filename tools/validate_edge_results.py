import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS
except ModuleNotFoundError:
    from prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS


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
    "smoothquant": {"external_command", "precompressed_checkpoint", "target_model_adaptation"},
    "omniquant": {"external_command", "precompressed_checkpoint", "target_model_adaptation"},
}

LM_EVAL_DTYPE_BY_DTYPE = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
}


REQUIRED_BASE_FIELDS = [
    "model_name",
    "compression_method",
    "source_model_path",
    "compressed_model_path",
    "compression_backend",
    "compression_dtype",
    "compression_seqlen",
    "compression_calibration_file",
    "compression_calibration_batch_size",
    "compression_materialized",
    "compression_uses_precompressed",
    "tokens_per_second",
    "deployment_prompt_source",
    "deployment_prompt_count",
    "deployment_batch_size",
    "deployment_input_length",
    "deployment_max_new_tokens",
    "deployment_warmup",
    "deployment_repeat",
    "deployment_input_tokens",
    "deployment_generated_tokens",
    "deployment_elapsed_seconds",
    "peak_memory_allocated_gb",
    "peak_memory_reserved_gb",
    "deployment_memory_limit_gb",
    "deployment_memory_measured_peak_gb",
    "deployment_memory_limit_exceeded",
    "resource_backend",
    "resource_seqlen",
    "resource_prompt_source",
    "resource_prompt_count",
    "resource_input_length",
    "resource_max_new_tokens",
    "resource_sparsity_sample_tensors",
    "storage_gb",
    "storage_bytes",
    "effective_storage_gb_sparse_approx",
    "effective_storage_bytes_sparse_approx",
    "total_params",
    "sparsity",
    "nonzero_ratio",
    "flops",
    "flops_num",
    "effective_flops_num_sparse_approx",
    "macs",
    "compute_backend",
    "accuracy_avg_score",
    "accuracy_delta_vs_dense",
    "accuracy_retention_vs_dense",
    "accuracy_failed_task_count",
    "accuracy_failed_tasks",
    "accuracy_dtype",
    "accuracy_lm_eval_dtype",
    "accuracy_batch_size",
    "accuracy_num_fewshot",
    "accuracy_data_dir",
    "accuracy_hf_datasets_cache",
    "accuracy_max_memory",
    "accuracy_split_tasks",
    "accuracy_continue_on_error",
    "deployment_dtype",
    "deployment_device_map",
    "deployment_max_memory",
    "runtime_python",
    "runtime_profile",
    "runtime_cpu_count",
    "runtime_system_memory_total_gb",
    "runtime_hf_endpoint",
    "runtime_hf_home",
    "runtime_hf_datasets_cache",
    "runtime_torch",
    "runtime_transformers",
    "runtime_cuda_available",
    "runtime_cuda_version",
    "runtime_cudnn_version",
    "runtime_cuda_visible_devices",
    "runtime_gpu_count",
    "runtime_gpu_name",
    "runtime_gpu_total_memory_gb",
    "runtime_bitsandbytes_available",
]

REQUIRED_NUMERIC_BASE_FIELDS = [
    "tokens_per_second",
    "deployment_prompt_count",
    "deployment_batch_size",
    "deployment_input_length",
    "deployment_max_new_tokens",
    "deployment_warmup",
    "deployment_repeat",
    "deployment_input_tokens",
    "deployment_generated_tokens",
    "deployment_elapsed_seconds",
    "peak_memory_allocated_gb",
    "peak_memory_reserved_gb",
    "deployment_memory_limit_gb",
    "deployment_memory_measured_peak_gb",
    "resource_seqlen",
    "resource_prompt_count",
    "resource_input_length",
    "resource_max_new_tokens",
    "resource_sparsity_sample_tensors",
    "storage_gb",
    "storage_bytes",
    "effective_storage_gb_sparse_approx",
    "effective_storage_bytes_sparse_approx",
    "total_params",
    "sparsity",
    "nonzero_ratio",
    "flops_num",
    "effective_flops_num_sparse_approx",
    "accuracy_avg_score",
    "accuracy_delta_vs_dense",
    "accuracy_retention_vs_dense",
    "accuracy_failed_task_count",
    "runtime_gpu_total_memory_gb",
    "runtime_gpu_count",
    "runtime_cpu_count",
    "runtime_system_memory_total_gb",
]

INTEGER_NUMERIC_FIELDS = [
    "compression_quant_bits",
    "compression_group_size",
    "compression_calibration_samples",
    "compression_calibration_batch_size",
    "compression_seqlen",
    "compression_external_returncode",
    "deployment_prompt_count",
    "deployment_batch_size",
    "deployment_input_length",
    "deployment_max_new_tokens",
    "deployment_warmup",
    "deployment_repeat",
    "deployment_input_tokens",
    "deployment_generated_tokens",
    "resource_seqlen",
    "resource_prompt_count",
    "resource_input_length",
    "resource_max_new_tokens",
    "resource_sparsity_sample_tensors",
    "storage_bytes",
    "effective_storage_quantized_bits",
    "total_params",
    "accuracy_failed_task_count",
    "accuracy_num_fewshot",
    "runtime_gpu_count",
    "runtime_cpu_count",
]

OPTIONAL_NUMERIC_FIELDS = [
    "effective_storage_bytes_quantized_approx",
    "effective_storage_gb_quantized_approx",
    "effective_storage_quantized_bits",
]

OPTIONAL_STRING_FIELDS = [
    "effective_storage_quantized_backend",
    "compression_awq_version",
]

QUANTIZED_METHODS = {"gptq", "awq", "smoothquant", "omniquant"}
PRUNING_METHODS = {"wanda", "sparsegpt"}
EXTERNAL_QUANTIZED_METHODS = {"smoothquant", "omniquant"}

REQUIRED_BOOL_BASE_FIELDS = [
    "compression_materialized",
    "compression_uses_precompressed",
    "compression_desc_act",
    "compression_prune_lm_head",
    "deployment_memory_limit_exceeded",
    "runtime_cuda_available",
    "runtime_bitsandbytes_available",
    "accuracy_split_tasks",
    "accuracy_continue_on_error",
]

REQUIRED_STRING_BASE_FIELDS = [
    "model_name",
    "compression_method",
    "source_model_path",
    "compressed_model_path",
    "compression_backend",
    "compression_dtype",
    "accuracy_dtype",
    "accuracy_lm_eval_dtype",
    "accuracy_batch_size",
    "accuracy_failed_tasks",
    "accuracy_data_dir",
    "accuracy_hf_datasets_cache",
    "accuracy_max_memory",
    "deployment_dtype",
    "deployment_device_map",
    "deployment_max_memory",
    "deployment_prompt_source",
    "resource_backend",
    "resource_prompt_source",
    "flops",
    "macs",
    "compute_backend",
    "runtime_python",
    "runtime_profile",
    "runtime_hf_endpoint",
    "runtime_hf_home",
    "runtime_hf_datasets_cache",
    "runtime_torch",
    "runtime_transformers",
    "runtime_cuda_version",
    "runtime_cudnn_version",
    "runtime_cuda_visible_devices",
    "runtime_gpu_name",
]

NONNEGATIVE_NUMERIC_BASE_FIELDS = [
    "tokens_per_second",
    "deployment_prompt_count",
    "deployment_batch_size",
    "deployment_input_length",
    "deployment_max_new_tokens",
    "deployment_warmup",
    "deployment_repeat",
    "deployment_input_tokens",
    "deployment_generated_tokens",
    "deployment_elapsed_seconds",
    "peak_memory_allocated_gb",
    "peak_memory_reserved_gb",
    "deployment_memory_limit_gb",
    "deployment_memory_measured_peak_gb",
    "resource_seqlen",
    "storage_gb",
    "storage_bytes",
    "effective_storage_gb_sparse_approx",
    "effective_storage_bytes_sparse_approx",
    "total_params",
    "flops_num",
    "effective_flops_num_sparse_approx",
    "accuracy_failed_task_count",
    "accuracy_retention_vs_dense",
    "runtime_gpu_total_memory_gb",
    "runtime_cpu_count",
    "runtime_system_memory_total_gb",
]

POSITIVE_NUMERIC_BASE_FIELDS = [
    "tokens_per_second",
    "deployment_prompt_count",
    "deployment_batch_size",
    "deployment_input_length",
    "deployment_max_new_tokens",
    "deployment_input_tokens",
    "deployment_generated_tokens",
    "deployment_elapsed_seconds",
    "resource_seqlen",
    "resource_prompt_count",
    "resource_input_length",
    "resource_max_new_tokens",
    "storage_gb",
    "storage_bytes",
    "effective_storage_gb_sparse_approx",
    "effective_storage_bytes_sparse_approx",
    "total_params",
    "flops_num",
    "effective_flops_num_sparse_approx",
    "runtime_gpu_count",
    "runtime_cpu_count",
    "runtime_system_memory_total_gb",
]

OPTIONAL_NONNEGATIVE_NUMERIC_FIELDS = [
    "effective_storage_bytes_quantized_approx",
    "effective_storage_gb_quantized_approx",
    "effective_storage_quantized_bits",
]

OPTIONAL_POSITIVE_NUMERIC_FIELDS = [
    "effective_storage_bytes_quantized_approx",
    "effective_storage_gb_quantized_approx",
    "effective_storage_quantized_bits",
]

RATIO_BASE_FIELDS = [
    "sparsity",
    "nonzero_ratio",
    "accuracy_avg_score",
]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _required_score_fields(tasks: str) -> List[str]:
    return [f"{task.strip()}_score" for task in tasks.split(",") if task.strip()]


def _required_score_metric_fields(tasks: str) -> List[str]:
    return [f"{task.strip()}_score_metric" for task in tasks.split(",") if task.strip()]


def _requires_mmlu_subtasks(tasks: str) -> bool:
    return "mmlu" in {task.strip() for task in tasks.split(",") if task.strip()}


def _required_mmlu_subtask_score_fields(tasks: str) -> List[str]:
    if not _requires_mmlu_subtasks(tasks):
        return []
    return [f"{task}_score" for task in DEFAULT_MMLU_SUBTASKS]


def _required_mmlu_subtask_score_metric_fields(tasks: str) -> List[str]:
    if not _requires_mmlu_subtasks(tasks):
        return []
    return [f"{task}_score_metric" for task in DEFAULT_MMLU_SUBTASKS]


def _required_compression_param_fields(method: str) -> List[str]:
    method = method.lower()
    fields = []
    if method in PRUNING_METHODS or method in EXTERNAL_QUANTIZED_METHODS:
        fields.append("compression_sparsity")
    if method in PRUNING_METHODS:
        fields.extend(
            [
                "compression_sparsity_semantics",
                "compression_prune_lm_head",
                "compression_prune_pattern",
                "compression_prune_structure",
            ]
        )
    if method in QUANTIZED_METHODS:
        fields.append("compression_quant_bits")
    if method in {"gptq", "awq", "omniquant"}:
        fields.append("compression_group_size")
    if method == "gptq":
        fields.append("compression_desc_act")
    if method == "awq":
        fields.append("compression_awq_version")
    if method in PRUNING_METHODS or method in QUANTIZED_METHODS:
        fields.append("compression_calibration_samples")
    return fields


def _paper_group_size_allowed(method: str, value: Any, defaults: Dict) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    if value > 0:
        return True
    per_channel_protocols = {
        "gptq_for_llama_c4_w4_asymmetric_actorder_true_sequential": "gptq",
        "omniquant_w4a16_wikitext2_128x2048_epochs20_lwc": "omniquant",
    }
    return value == -1 and per_channel_protocols.get(defaults.get("paper_protocol")) == method


def _paper_protocol_violations(row: Dict, pair: Tuple[str, str], defaults: Dict) -> List[Dict]:
    expected = defaults.get("paper_protocol")
    if not expected:
        return []
    violations = []
    if row.get("compression_paper_protocol") != expected:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "compression_paper_protocol",
                "value": row.get("compression_paper_protocol"),
                "expected": expected,
                "source": "effective paper_protocol",
            }
        )
    if row.get("compression_paper_protocol_attested") is not True:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "compression_paper_protocol_attested",
                "value": row.get("compression_paper_protocol_attested"),
                "expected": True,
                "source": "paper-aligned matrix gate",
            }
        )
    method = str(row.get("compression_method") or "").lower()
    if method in {"gptq", "awq", "omniquant"} and not _paper_group_size_allowed(
        method, row.get("compression_group_size"), defaults
    ):
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "compression_group_size",
                "value": row.get("compression_group_size"),
                "error": "paper protocol requires a positive group size, except an explicitly selected per-channel paper profile",
            }
        )
    return violations


def _emit_report(report: Dict, output: str = "") -> Dict:
    if output:
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)
    return report


def _type_violations(
    row: Dict,
    pair: Tuple[str, str],
    numeric_fields: List[str],
    integer_fields: List[str],
    bool_fields: List[str],
    string_fields: List[str],
) -> List[Dict]:
    violations = []
    for field in numeric_fields:
        value = row.get(field)
        if value is None or value == "":
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "expected_type": "number",
                    "actual_type": type(value).__name__,
                }
            )
        elif not math.isfinite(value):
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "expected_type": "finite number",
                    "actual_type": "non-finite number",
                }
            )
    for field in integer_fields:
        value = row.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and int(value) != value:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "expected_type": "integer",
                    "actual_type": "non-integer number",
                    "value": value,
                }
            )
    for field in bool_fields:
        value = row.get(field)
        if value is None or value == "":
            continue
        if not isinstance(value, bool):
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "expected_type": "bool",
                    "actual_type": type(value).__name__,
                }
            )
    for field in string_fields:
        value = row.get(field)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "expected_type": "string",
                    "actual_type": type(value).__name__,
                }
            )
    return violations


def _range_violations(
    row: Dict,
    pair: Tuple[str, str],
    nonnegative_fields: List[str],
    positive_fields: List[str],
    ratio_fields: List[str],
) -> List[Dict]:
    violations = []
    for field in nonnegative_fields:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "constraint": ">= 0",
                    "value": value,
                }
            )
    for field in positive_fields:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "constraint": "> 0",
                    "value": value,
                }
            )
    for field in ratio_fields:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not 0 <= value <= 1:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "constraint": "0 <= value <= 1",
                    "value": value,
                }
            )
    return violations


def _deployment_consistency_violations(row: Dict, pair: Tuple[str, str]) -> List[Dict]:
    violations = []
    measured_peak = row.get("deployment_memory_measured_peak_gb")
    allocated_peak = row.get("peak_memory_allocated_gb")
    reserved_peak = row.get("peak_memory_reserved_gb")
    limit = row.get("deployment_memory_limit_gb")
    exceeded = row.get("deployment_memory_limit_exceeded")
    prompt_count = row.get("deployment_prompt_count")
    deployment_batch_size = row.get("deployment_batch_size")
    input_length = row.get("deployment_input_length")
    max_new_tokens = row.get("deployment_max_new_tokens")
    repeat = row.get("deployment_repeat")
    input_tokens = row.get("deployment_input_tokens")
    generated_tokens = row.get("deployment_generated_tokens")
    elapsed_seconds = row.get("deployment_elapsed_seconds")
    tokens_per_second = row.get("tokens_per_second")
    peaks = [
        value
        for value in (allocated_peak, reserved_peak)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if isinstance(measured_peak, (int, float)) and not isinstance(measured_peak, bool) and peaks:
        expected_min = max(peaks)
        if measured_peak < expected_min:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "deployment_memory_measured_peak_gb",
                    "constraint": ">= max(peak_memory_allocated_gb, peak_memory_reserved_gb)",
                    "value": measured_peak,
                    "expected_min": expected_min,
                }
            )
    if (
        isinstance(measured_peak, (int, float))
        and not isinstance(measured_peak, bool)
        and isinstance(limit, (int, float))
        and not isinstance(limit, bool)
        and isinstance(exceeded, bool)
    ):
        expected_exceeded = limit > 0 and measured_peak > limit
        if exceeded != expected_exceeded:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "deployment_memory_limit_exceeded",
                    "constraint": "equals deployment_memory_measured_peak_gb > deployment_memory_limit_gb when limit > 0",
                    "value": exceeded,
                    "expected": expected_exceeded,
                    "measured_peak_gb": measured_peak,
                    "limit_gb": limit,
                }
            )
    if (
        isinstance(deployment_batch_size, (int, float))
        and not isinstance(deployment_batch_size, bool)
        and isinstance(prompt_count, (int, float))
        and not isinstance(prompt_count, bool)
        and deployment_batch_size > prompt_count
    ):
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "deployment_batch_size",
                "constraint": "<= deployment_prompt_count",
                "value": deployment_batch_size,
                "expected_max": prompt_count,
            }
        )
    if (
        isinstance(input_tokens, (int, float))
        and not isinstance(input_tokens, bool)
        and isinstance(prompt_count, (int, float))
        and not isinstance(prompt_count, bool)
        and isinstance(input_length, (int, float))
        and not isinstance(input_length, bool)
    ):
        expected_max = prompt_count * input_length
        if input_tokens > expected_max:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "deployment_input_tokens",
                    "constraint": "<= deployment_prompt_count * deployment_input_length",
                    "value": input_tokens,
                    "expected_max": expected_max,
                }
            )
    if (
        isinstance(generated_tokens, (int, float))
        and not isinstance(generated_tokens, bool)
        and isinstance(prompt_count, (int, float))
        and not isinstance(prompt_count, bool)
        and isinstance(repeat, (int, float))
        and not isinstance(repeat, bool)
        and isinstance(max_new_tokens, (int, float))
        and not isinstance(max_new_tokens, bool)
    ):
        expected_max = prompt_count * repeat * max_new_tokens
        if generated_tokens > expected_max:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "deployment_generated_tokens",
                    "constraint": "<= deployment_prompt_count * deployment_repeat * deployment_max_new_tokens",
                    "value": generated_tokens,
                    "expected_max": expected_max,
                }
            )
    if (
        isinstance(tokens_per_second, (int, float))
        and not isinstance(tokens_per_second, bool)
        and isinstance(generated_tokens, (int, float))
        and not isinstance(generated_tokens, bool)
        and isinstance(elapsed_seconds, (int, float))
        and not isinstance(elapsed_seconds, bool)
        and elapsed_seconds > 0
    ):
        expected = generated_tokens / elapsed_seconds
        allowed_error = 1e-8 * max(1.0, abs(expected), abs(tokens_per_second))
        if abs(tokens_per_second - expected) > allowed_error:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "tokens_per_second",
                    "value": tokens_per_second,
                    "expected": expected,
                    "source": "deployment_generated_tokens / deployment_elapsed_seconds",
                }
            )
    return violations


def _runtime_consistency_violations(row: Dict, pair: Tuple[str, str]) -> List[Dict]:
    violations = []
    cuda_available = row.get("runtime_cuda_available")
    gpu_count = row.get("runtime_gpu_count")
    gpu_name = row.get("runtime_gpu_name")
    gpu_total_memory_gb = row.get("runtime_gpu_total_memory_gb")
    has_gpu_count = (
        isinstance(gpu_count, (int, float))
        and not isinstance(gpu_count, bool)
        and gpu_count > 0
    )
    has_gpu_metadata = has_gpu_count or bool(gpu_name) or (
        isinstance(gpu_total_memory_gb, (int, float))
        and not isinstance(gpu_total_memory_gb, bool)
        and gpu_total_memory_gb > 0
    )
    if has_gpu_metadata and cuda_available is not True:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "runtime_cuda_available",
                "value": cuda_available,
                "expected": True,
                "error": "GPU runtime metadata requires CUDA to be available",
            }
        )
    if cuda_available is True and not has_gpu_count:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "runtime_gpu_count",
                "value": gpu_count,
                "expected_min": 1,
                "error": "CUDA runtime availability requires at least one reported GPU",
            }
        )
    return violations


def _accuracy_retention_violations(rows_by_pair: Dict[Tuple[str, str], Dict], required_pairs: Set[Tuple[str, str]]) -> List[Dict]:
    violations = []
    tolerance = 1e-8
    dense_scores = {
        model: row.get("accuracy_avg_score")
        for (model, method), row in rows_by_pair.items()
        if method == "dense" and isinstance(row.get("accuracy_avg_score"), (int, float)) and not isinstance(row.get("accuracy_avg_score"), bool)
    }
    for pair in sorted(required_pairs & set(rows_by_pair.keys())):
        model, method = pair
        row = rows_by_pair[pair]
        dense_score = dense_scores.get(model)
        score = row.get("accuracy_avg_score")
        delta = row.get("accuracy_delta_vs_dense")
        retention = row.get("accuracy_retention_vs_dense")
        if not (
            isinstance(dense_score, (int, float))
            and not isinstance(dense_score, bool)
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
        ):
            continue
        expected_delta = score - dense_score
        if isinstance(delta, (int, float)) and not isinstance(delta, bool) and abs(delta - expected_delta) > tolerance:
            violations.append(
                {
                    "model": model,
                    "method": method,
                    "field": "accuracy_delta_vs_dense",
                    "value": delta,
                    "expected": expected_delta,
                }
            )
        if dense_score != 0 and isinstance(retention, (int, float)) and not isinstance(retention, bool):
            expected_retention = score / dense_score
            if abs(retention - expected_retention) > tolerance:
                violations.append(
                    {
                        "model": model,
                        "method": method,
                        "field": "accuracy_retention_vs_dense",
                        "value": retention,
                        "expected": expected_retention,
                    }
                )
    return violations


def _runtime_profile_consistency_violations(rows_by_pair: Dict[Tuple[str, str], Dict], required_pairs: Set[Tuple[str, str]]) -> List[Dict]:
    profiles = {
        row.get("runtime_profile")
        for pair, row in rows_by_pair.items()
        if pair in required_pairs and isinstance(row.get("runtime_profile"), str) and row.get("runtime_profile")
    }
    if len(profiles) <= 1:
        return []
    expected = sorted(profiles)[0]
    return [
        {
            "model": pair[0],
            "method": pair[1],
            "field": "runtime_profile",
            "value": row.get("runtime_profile"),
            "expected": expected,
            "all_profiles": sorted(profiles),
            "error": "final results must not mix runtime profiles",
        }
        for pair, row in sorted(rows_by_pair.items())
        if pair in required_pairs
    ]


def _runtime_profile_config_violations(rows_by_pair: Dict[Tuple[str, str], Dict], required_pairs: Set[Tuple[str, str]], config: Dict) -> List[Dict]:
    violations = []
    for pair in sorted(required_pairs & set(rows_by_pair.keys())):
        row = rows_by_pair[pair]
        expected = _effective_runner_setting(config, pair[0], pair[1], "runtime_profile")
        if expected in (None, ""):
            continue
        if row.get("runtime_profile") != expected:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "runtime_profile",
                    "value": row.get("runtime_profile"),
                    "expected": expected,
                    "source": "effective runtime_profile",
                }
            )
    return violations


def _accuracy_aggregate_consistency_violations(row: Dict, pair: Tuple[str, str]) -> List[Dict]:
    violations = []
    tolerance = 1e-8
    mmlu_values = [
        row.get(f"{task}_score")
        for task in DEFAULT_MMLU_SUBTASKS
        if isinstance(row.get(f"{task}_score"), (int, float))
        and not isinstance(row.get(f"{task}_score"), bool)
    ]
    mmlu_score = row.get("mmlu_score")
    if (
        len(mmlu_values) == len(DEFAULT_MMLU_SUBTASKS)
        and isinstance(mmlu_score, (int, float))
        and not isinstance(mmlu_score, bool)
    ):
        expected = sum(mmlu_values) / len(mmlu_values)
        if abs(mmlu_score - expected) > tolerance:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "mmlu_score",
                    "value": mmlu_score,
                    "expected": expected,
                    "source": "mean(DEFAULT_MMLU_SUBTASKS)",
                }
            )
    mmlu_metric_keys = [
        row.get(f"{task}_score_metric")
        for task in DEFAULT_MMLU_SUBTASKS
        if isinstance(row.get(f"{task}_score_metric"), str)
        and row.get(f"{task}_score_metric")
    ]
    mmlu_score_metric = row.get("mmlu_score_metric")
    if len(mmlu_metric_keys) == len(DEFAULT_MMLU_SUBTASKS) and isinstance(mmlu_score_metric, str):
        unique_metrics = sorted(set(mmlu_metric_keys))
        expected_metric = f"aggregate:{unique_metrics[0]}" if len(unique_metrics) == 1 else "aggregate:mixed"
        if mmlu_score_metric != expected_metric:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "mmlu_score_metric",
                    "value": mmlu_score_metric,
                    "expected": expected_metric,
                    "source": "aggregate(DEFAULT_MMLU_SUBTASKS score metrics)",
                }
            )
    task_values = [
        row.get(field)
        for field in ("mmlu_score", "arc_challenge_score", "hellaswag_score", "piqa_score", "winogrande_score")
        if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)
    ]
    accuracy_avg_score = row.get("accuracy_avg_score")
    if len(task_values) == 5 and isinstance(accuracy_avg_score, (int, float)) and not isinstance(accuracy_avg_score, bool):
        expected = sum(task_values) / len(task_values)
        if abs(accuracy_avg_score - expected) > tolerance:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "accuracy_avg_score",
                    "value": accuracy_avg_score,
                    "expected": expected,
                    "source": "mean(target_task_scores)",
                }
            )
    return violations


def _score_metric_consistency_violations(row: Dict, pair: Tuple[str, str], tasks: str) -> List[Dict]:
    violations = []
    for task in (task.strip() for task in tasks.split(",") if task.strip()):
        expected = EXPECTED_SCORE_METRICS.get(task)
        if not expected:
            continue
        field = f"{task}_score_metric"
        value = row.get(field)
        if isinstance(value, str) and value != expected:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "value": value,
                    "expected": expected,
                    "source": "expected target task metric",
                }
            )
    for task in DEFAULT_MMLU_SUBTASKS:
        field = f"{task}_score_metric"
        value = row.get(field)
        if isinstance(value, str) and value != "acc,none":
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "value": value,
                    "expected": "acc,none",
                    "source": "expected MMLU subtask metric",
                }
            )
    return violations


def _quantized_storage_consistency_violations(row: Dict, pair: Tuple[str, str]) -> List[Dict]:
    violations = []
    method = str(row.get("compression_method", "")).lower()
    quantized_fields = [
        "effective_storage_bytes_quantized_approx",
        "effective_storage_gb_quantized_approx",
        "effective_storage_quantized_bits",
        "effective_storage_quantized_backend",
    ]
    if method not in QUANTIZED_METHODS:
        for field in quantized_fields:
            value = row.get(field)
            if value not in (None, ""):
                violations.append(
                    {
                        "model": pair[0],
                        "method": pair[1],
                        "field": field,
                        "value": value,
                        "expected": None,
                        "source": "non-quantized compression method",
                    }
                )
        return violations
    storage_real = row.get("quantization_storage_real")
    storage_format = row.get("quantization_storage_format")
    if not isinstance(storage_real, bool):
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "quantization_storage_real",
                "value": storage_real,
                "expected": "boolean",
            }
        )
        return violations
    if not isinstance(storage_format, str) or not storage_format:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "quantization_storage_format",
                "value": storage_format,
                "expected": "non-empty string",
            }
        )
    if not storage_real:
        for field in quantized_fields:
            if row.get(field) not in (None, ""):
                violations.append(
                    {
                        "model": pair[0],
                        "method": pair[1],
                        "field": field,
                        "value": row.get(field),
                        "expected": None,
                        "source": "floating-weight or fake-quant checkpoint",
                    }
                )
        return violations
    quant_bits = row.get("compression_quant_bits")
    storage_bits = row.get("effective_storage_quantized_bits")
    if (
        isinstance(quant_bits, (int, float))
        and not isinstance(quant_bits, bool)
        and isinstance(storage_bits, (int, float))
        and not isinstance(storage_bits, bool)
        and quant_bits != storage_bits
    ):
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "effective_storage_quantized_bits",
                "value": storage_bits,
                "expected": quant_bits,
            }
        )
    storage_bytes = row.get("effective_storage_bytes_quantized_approx")
    storage_backend = row.get("effective_storage_quantized_backend")
    if storage_backend == "checkpoint_directory_size":
        checkpoint_bytes = row.get("storage_bytes")
        if (
            isinstance(checkpoint_bytes, (int, float))
            and not isinstance(checkpoint_bytes, bool)
            and isinstance(storage_bytes, (int, float))
            and not isinstance(storage_bytes, bool)
        ):
            expected = checkpoint_bytes
            allowed_error = 1e-8 * max(1.0, abs(expected), abs(storage_bytes))
            if abs(storage_bytes - expected) > allowed_error:
                violations.append(
                    {
                        "model": pair[0],
                        "method": pair[1],
                        "field": "effective_storage_bytes_quantized_approx",
                        "value": storage_bytes,
                        "expected": expected,
                        "source": "storage_bytes for real quantized checkpoint",
                    }
                )
    elif storage_backend == "param_count_times_quant_bits":
        total_params = row.get("total_params")
        if (
            isinstance(total_params, (int, float))
            and not isinstance(total_params, bool)
            and isinstance(quant_bits, (int, float))
            and not isinstance(quant_bits, bool)
            and isinstance(storage_bytes, (int, float))
            and not isinstance(storage_bytes, bool)
        ):
            expected = total_params * quant_bits / 8.0
            allowed_error = 1e-8 * max(1.0, abs(expected), abs(storage_bytes))
            if abs(storage_bytes - expected) > allowed_error:
                violations.append(
                    {
                        "model": pair[0],
                        "method": pair[1],
                        "field": "effective_storage_bytes_quantized_approx",
                        "value": storage_bytes,
                        "expected": expected,
                        "source": "total_params * compression_quant_bits / 8",
                    }
                )
    else:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "effective_storage_quantized_backend",
                "value": storage_backend,
                "expected": "checkpoint_directory_size",
                "source": "quantized effective storage backend",
            }
        )
    storage_gb = row.get("effective_storage_gb_quantized_approx")
    if (
        isinstance(storage_bytes, (int, float))
        and not isinstance(storage_bytes, bool)
        and isinstance(storage_gb, (int, float))
        and not isinstance(storage_gb, bool)
    ):
        expected = storage_bytes / (1024**3)
        allowed_error = 1e-8 * max(1.0, abs(expected), abs(storage_gb))
        if abs(storage_gb - expected) > allowed_error:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "effective_storage_gb_quantized_approx",
                    "value": storage_gb,
                    "expected": expected,
                    "source": "effective_storage_bytes_quantized_approx / 1024**3",
                }
            )
    return violations


def _sparse_resource_consistency_violations(row: Dict, pair: Tuple[str, str]) -> List[Dict]:
    violations = []
    tolerance = 1e-8
    storage_bytes = row.get("storage_bytes")
    storage_gb = row.get("storage_gb")
    if (
        isinstance(storage_bytes, (int, float))
        and not isinstance(storage_bytes, bool)
        and isinstance(storage_gb, (int, float))
        and not isinstance(storage_gb, bool)
    ):
        expected = storage_bytes / (1024**3)
        allowed_error = tolerance * max(1.0, abs(expected), abs(storage_gb))
        if abs(storage_gb - expected) > allowed_error:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "storage_gb",
                    "value": storage_gb,
                    "expected": expected,
                    "source": "storage_bytes / 1024**3",
                }
            )
    nonzero_ratio = row.get("nonzero_ratio")
    if not (isinstance(nonzero_ratio, (int, float)) and not isinstance(nonzero_ratio, bool)):
        return violations
    sparsity = row.get("sparsity")
    if isinstance(sparsity, (int, float)) and not isinstance(sparsity, bool):
        expected_nonzero_ratio = 1.0 - sparsity
        allowed_error = tolerance * max(1.0, abs(expected_nonzero_ratio), abs(nonzero_ratio))
        if abs(nonzero_ratio - expected_nonzero_ratio) > allowed_error:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "nonzero_ratio",
                    "value": nonzero_ratio,
                    "expected": expected_nonzero_ratio,
                    "source": "1 - sparsity",
                    "sparsity": sparsity,
                }
            )
    for field, source_field in (
        ("effective_storage_bytes_sparse_approx", "storage_bytes"),
        ("effective_storage_gb_sparse_approx", "storage_gb"),
        ("effective_flops_num_sparse_approx", "flops_num"),
    ):
        value = row.get(field)
        source_value = row.get(source_field)
        if not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(source_value, (int, float))
            and not isinstance(source_value, bool)
        ):
            continue
        expected = source_value * nonzero_ratio
        allowed_error = tolerance * max(1.0, abs(expected), abs(value))
        if abs(value - expected) > allowed_error:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "value": value,
                    "expected": expected,
                    "source": f"{source_field} * nonzero_ratio",
                    "nonzero_ratio": nonzero_ratio,
                }
            )
    return violations


def _compression_provenance_violations(row: Dict, pair: Tuple[str, str]) -> List[Dict]:
    violations = []
    backend = row.get("compression_backend")
    method = str(row.get("compression_method", "")).lower()
    expected_backends = EXPECTED_COMPRESSION_BACKENDS.get(method)
    if isinstance(backend, str) and expected_backends and backend not in expected_backends:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "compression_backend",
                "value": backend,
                "expected": sorted(expected_backends),
            }
        )
    uses_precompressed = row.get("compression_uses_precompressed")
    if isinstance(backend, str) and isinstance(uses_precompressed, bool):
        expected = backend == "precompressed_checkpoint"
        if uses_precompressed != expected:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "compression_uses_precompressed",
                    "value": uses_precompressed,
                    "expected": expected,
                    "compression_backend": backend,
                }
            )
    materialized = row.get("compression_materialized")
    if isinstance(backend, str) and isinstance(materialized, bool):
        expected_materialized = backend not in {"source_checkpoint_reference", "precompressed_checkpoint"}
        if materialized != expected_materialized:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "compression_materialized",
                    "value": materialized,
                    "expected": expected_materialized,
                    "compression_backend": backend,
                }
            )
    external_command = row.get("compression_external_command")
    if backend == "external_command" and (not isinstance(external_command, str) or not external_command.strip()):
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "compression_external_command",
                "value": external_command,
                "expected": "non-empty string when compression_backend is external_command",
                "compression_backend": backend,
            }
        )
    external_returncode = row.get("compression_external_returncode")
    if backend == "external_command" and external_returncode != 0:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "compression_external_returncode",
                "value": external_returncode,
                "expected": 0,
                "compression_backend": backend,
            }
        )
    return violations


def _resource_backend_violations(row: Dict, pair: Tuple[str, str]) -> List[Dict]:
    violations = []
    if row.get("resource_backend") == "config_only_fallback":
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "resource_backend",
                "value": row.get("resource_backend"),
                "model_load_error": row.get("resource_model_load_error"),
                "error": "final resource results must load the checkpoint rather than using config-only fallback",
            }
        )
    elif row.get("resource_backend") != "model_load":
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "resource_backend",
                "value": row.get("resource_backend"),
                "expected": "model_load",
            }
        )
    compute_backend = row.get("compute_backend")
    if isinstance(compute_backend, str) and compute_backend not in {"calflops", "analytical_config_estimate"}:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "compute_backend",
                "value": compute_backend,
                "expected": "calflops or analytical_config_estimate",
            }
        )
    return violations


def _effective_defaults(config: Dict, model_name: str, method: str) -> Dict:
    merged = dict(config.get("defaults", {}))
    ignored_keys = {"output_root", "cuda_visible_devices", "python_bin", "runtime_profile"}
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


def _effective_runner_setting(config: Dict, model_name: str, method: str, field: str):
    merged = dict(config.get("defaults", {}))
    for section_name, key in (
        ("method_overrides", method),
        ("model_overrides", model_name),
        ("model_method_overrides", f"{model_name}/{method}"),
    ):
        section = config.get(section_name, {})
        override = section.get(key) if isinstance(section, dict) else None
        if isinstance(override, dict):
            merged.update(override)
    return merged.get(field)


def _effective_calibration_samples(defaults: Dict) -> int:
    configured = defaults.get("calibration_samples")
    if not isinstance(configured, int) or isinstance(configured, bool) or configured <= 0:
        return configured
    return configured


def _config_consistency_violations(row: Dict, pair: Tuple[str, str], defaults: Dict) -> List[Dict]:
    violations = []
    expected_dtype = defaults.get("dtype")
    method = str(row.get("compression_method", "")).lower()
    prune_pattern = row.get("compression_prune_pattern")
    if method in {"wanda", "sparsegpt"}:
        if row.get("compression_sparsity_semantics") != "element_zero_fraction":
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "compression_sparsity_semantics",
                    "value": row.get("compression_sparsity_semantics"),
                    "expected": "element_zero_fraction",
                }
            )
        expected_structure = "semi_structured" if prune_pattern == "2:4" else "unstructured"
        if row.get("compression_prune_structure") != expected_structure:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "compression_prune_structure",
                    "value": row.get("compression_prune_structure"),
                    "expected": expected_structure,
                    "source": "compression_prune_pattern",
                }
            )
        if prune_pattern == "2:4":
            for field, expected in (("compression_prune_n", 2), ("compression_prune_m", 4)):
                if row.get(field) != expected:
                    violations.append(
                        {
                            "model": pair[0],
                            "method": pair[1],
                            "field": field,
                            "value": row.get(field),
                            "expected": expected,
                            "source": "2:4 pruning pattern",
                        }
                    )
    if expected_dtype and row.get("compression_dtype") != expected_dtype:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "compression_dtype",
                "value": row.get("compression_dtype"),
                "expected": expected_dtype,
                "source": "defaults.dtype",
            }
        )
    expected_lm_eval_dtype = LM_EVAL_DTYPE_BY_DTYPE.get(str(expected_dtype))
    if expected_lm_eval_dtype and row.get("accuracy_lm_eval_dtype") != expected_lm_eval_dtype:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "accuracy_lm_eval_dtype",
                "value": row.get("accuracy_lm_eval_dtype"),
                "expected": expected_lm_eval_dtype,
                "source": "defaults.dtype",
            }
        )
    expected_num_fewshot = defaults.get("num_fewshot")
    if row.get("accuracy_num_fewshot") != expected_num_fewshot:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "accuracy_num_fewshot",
                "value": row.get("accuracy_num_fewshot"),
                "expected": expected_num_fewshot,
                "source": "defaults.num_fewshot",
            }
        )
    expected_revision = defaults.get("revision")
    if row.get("model_revision") != expected_revision:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "model_revision",
                "value": row.get("model_revision"),
                "expected": expected_revision,
                "source": "defaults.revision",
            }
        )
    expected_accuracy_limit = defaults.get("accuracy_limit")
    if row.get("accuracy_limit") != expected_accuracy_limit:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "accuracy_limit",
                "value": row.get("accuracy_limit"),
                "expected": expected_accuracy_limit,
                "source": "defaults.accuracy_limit",
            }
        )
    expected_max_memory = defaults.get("max_memory")
    if expected_max_memory and row.get("compression_max_memory") != expected_max_memory:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "compression_max_memory",
                "value": row.get("compression_max_memory"),
                "expected": expected_max_memory,
                "source": "defaults.max_memory",
            }
        )
    expected_attn = defaults.get("attn_implementation")
    if expected_attn and row.get("compression_attn_implementation") != expected_attn:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "compression_attn_implementation",
                "value": row.get("compression_attn_implementation"),
                "expected": expected_attn,
                "source": "defaults.attn_implementation",
            }
        )
    if expected_attn:
        for field in ("accuracy_attn_implementation", "deployment_attn_implementation"):
            if row.get(field) != expected_attn:
                violations.append(
                    {
                        "model": pair[0],
                        "method": pair[1],
                        "field": field,
                        "value": row.get(field),
                        "expected": expected_attn,
                        "source": "defaults.attn_implementation",
                    }
                )
    expected_device_map = defaults.get("device_map", "auto")
    if expected_device_map and row.get("deployment_device_map") != expected_device_map:
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "deployment_device_map",
                "value": row.get("deployment_device_map"),
                "expected": expected_device_map,
                "source": "defaults.device_map",
            }
        )
    expected_prompt_source = "file" if defaults.get("prompt_file") else "default"
    for field in ("deployment_prompt_source", "resource_prompt_source"):
        if row.get(field) != expected_prompt_source:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "value": row.get(field),
                    "expected": expected_prompt_source,
                    "source": "defaults.prompt_file",
                }
            )
    expected_hf_cache = defaults.get("hf_cache_dir")
    if expected_hf_cache:
        for field, expected in (("runtime_hf_home", expected_hf_cache),):
            if row.get(field) != expected:
                violations.append(
                    {
                        "model": pair[0],
                        "method": pair[1],
                        "field": field,
                        "value": row.get(field),
                        "expected": expected,
                        "source": "defaults.hf_cache_dir",
                    }
                )
        for field, expected in (
            ("runtime_transformers_cache", str(Path(expected_hf_cache) / "transformers")),
            ("runtime_huggingface_hub_cache", str(Path(expected_hf_cache) / "hub")),
        ):
            value = row.get(field)
            if value not in (None, "") and value != expected:
                violations.append(
                    {
                        "model": pair[0],
                        "method": pair[1],
                        "field": field,
                        "value": value,
                        "expected": expected,
                        "source": "defaults.hf_cache_dir",
                    }
                )
    for field, default_key in (
        ("runtime_hf_endpoint", "hf_endpoint"),
        ("runtime_cuda_visible_devices", "cuda_visible_devices"),
        ("runtime_hf_datasets_cache", "data_dir"),
        ("accuracy_dtype", "dtype"),
        ("accuracy_batch_size", "batch_size"),
        ("accuracy_data_dir", "data_dir"),
        ("accuracy_hf_datasets_cache", "data_dir"),
        ("accuracy_max_memory", "max_memory"),
        ("accuracy_split_tasks", "split_accuracy_tasks"),
        ("accuracy_continue_on_error", "continue_on_accuracy_error"),
        ("deployment_dtype", "dtype"),
        ("deployment_max_memory", "max_memory"),
        ("deployment_prompt_path", "prompt_file"),
        ("compression_calibration_batch_size", "calibration_batch_size"),
        ("deployment_batch_size", "deployment_batch_size"),
        ("deployment_input_length", "input_length"),
        ("deployment_max_new_tokens", "max_new_tokens"),
        ("deployment_warmup", "warmup"),
        ("deployment_repeat", "repeat"),
        ("compression_seqlen", "seqlen"),
        ("resource_seqlen", "seqlen"),
        ("resource_prompt_path", "prompt_file"),
        ("resource_input_length", "input_length"),
        ("resource_max_new_tokens", "max_new_tokens"),
        ("resource_sparsity_sample_tensors", "sparsity_sample_tensors"),
    ):
        expected = defaults.get(default_key)
        if expected is not None and row.get(field) != expected:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "value": row.get(field),
                    "expected": expected,
                    "source": f"defaults.{default_key}",
                }
            )
    for field, default_key in (
        ("compression_sparsity", "sparsity"),
        ("compression_prune_pattern", "prune_pattern"),
        ("compression_quant_bits", "quant_bits"),
        ("compression_group_size", "group_size"),
        ("compression_desc_act", "desc_act"),
        ("compression_awq_version", "awq_version"),
        ("compression_prune_lm_head", "prune_lm_head"),
        ("compression_calibration_file", "calibration_file"),
    ):
        expected = defaults.get(default_key)
        value = row.get(field)
        if expected is not None and value is not None and value != expected:
            violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": field,
                    "value": value,
                    "expected": expected,
                    "source": f"defaults.{default_key}",
                }
            )
    expected_calibration_samples = _effective_calibration_samples(defaults)
    if (
        expected_calibration_samples is not None
        and row.get("compression_calibration_samples") is not None
        and not row.get("compression_uses_precompressed")
        and row.get("compression_calibration_samples") != expected_calibration_samples
    ):
        violations.append(
            {
                "model": pair[0],
                "method": pair[1],
                "field": "compression_calibration_samples",
                "value": row.get("compression_calibration_samples"),
                "expected": expected_calibration_samples,
                "source": "defaults.calibration_samples",
            }
        )
    return violations


def validate(args: argparse.Namespace) -> Dict:
    config = _load_json(Path(args.config))
    result_path = Path(args.results).expanduser()
    defaults = config.get("defaults", {})
    deployment_memory_limit_gb = args.deployment_memory_limit_gb
    if deployment_memory_limit_gb is None:
        deployment_memory_limit_gb = defaults.get("deployment_memory_limit_gb", 0) or 0
    models = [
        model for model in config["models"]
        if not args.only_model or model["name"] in set(args.only_model)
    ]
    methods = [
        method for method in config["methods"]
        if not args.only_method or method in set(args.only_method)
    ]
    required_pairs: Set[Tuple[str, str]] = {
        (model["name"], method)
        for model in models
        for method in methods
    }
    try:
        rows = _load_json(result_path)
    except FileNotFoundError:
        return _emit_report(
            {
                "ok": False,
                "results": str(result_path),
                "expected_rows": len(required_pairs),
                "actual_rows": None,
                "format_errors": [
                    {
                        "field": "matrix_results",
                        "error": "results file not found",
                    }
                ],
            },
            args.output,
        )
    except json.JSONDecodeError as exc:
        return _emit_report(
            {
                "ok": False,
                "results": str(result_path),
                "expected_rows": len(required_pairs),
                "actual_rows": None,
                "format_errors": [
                    {
                        "field": "matrix_results",
                        "error": "invalid JSON",
                        "message": str(exc),
                    }
                ],
            },
            args.output,
        )
    if not isinstance(rows, list):
        return _emit_report(
            {
                "ok": False,
                "results": str(result_path),
                "expected_rows": len(required_pairs),
                "actual_rows": None,
                "format_errors": [
                    {
                        "field": "matrix_results",
                        "error": "expected JSON list",
                        "actual_type": type(rows).__name__,
                    }
                ],
            },
            args.output,
        )
    invalid_rows = [
        {"index": index, "actual_type": type(row).__name__}
        for index, row in enumerate(rows)
        if not isinstance(row, dict)
    ]
    if invalid_rows:
        return _emit_report(
            {
                "ok": False,
                "results": str(result_path),
                "expected_rows": len(required_pairs),
                "actual_rows": len(rows),
                "format_errors": [
                    {
                        "field": "matrix_results",
                        "error": "all rows must be JSON objects",
                        "invalid_rows": invalid_rows,
                    }
                ],
            },
            args.output,
        )
    duplicate_pairs = []
    pair_counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        pair = (row.get("model_name"), row.get("compression_method"))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    for pair, count in sorted(pair_counts.items()):
        if count > 1:
            duplicate_pairs.append({"model": pair[0], "method": pair[1], "count": count})
    rows_by_pair = {
        (row.get("model_name"), row.get("compression_method")): row
        for row in rows
    }
    required_score_fields = _required_score_fields(config["defaults"]["tasks"])
    required_score_metric_fields = _required_score_metric_fields(config["defaults"]["tasks"])
    require_mmlu_subtasks = _requires_mmlu_subtasks(config["defaults"]["tasks"])
    required_mmlu_subtask_score_fields = _required_mmlu_subtask_score_fields(config["defaults"]["tasks"])
    required_mmlu_subtask_score_metric_fields = _required_mmlu_subtask_score_metric_fields(config["defaults"]["tasks"])
    required_numeric_fields = (
        REQUIRED_NUMERIC_BASE_FIELDS
        + OPTIONAL_NUMERIC_FIELDS
        + required_score_fields
        + required_mmlu_subtask_score_fields
    )
    required_ratio_fields = RATIO_BASE_FIELDS + required_score_fields + required_mmlu_subtask_score_fields
    missing_pairs = sorted(required_pairs - set(rows_by_pair.keys()))
    unexpected_pairs = sorted(set(rows_by_pair.keys()) - required_pairs)

    missing_fields = []
    type_violations = []
    range_violations = []
    consistency_violations = []
    accuracy_violations = []
    memory_violations = []
    configured_runtime_profiles = [
        _effective_runner_setting(config, model, method, "runtime_profile")
        for model, method in required_pairs
    ]
    if any(profile not in (None, "") for profile in configured_runtime_profiles):
        consistency_violations.extend(_runtime_profile_config_violations(rows_by_pair, required_pairs, config))
    else:
        consistency_violations.extend(_runtime_profile_consistency_violations(rows_by_pair, required_pairs))
    for pair in sorted(required_pairs & set(rows_by_pair.keys())):
        row = rows_by_pair[pair]
        pair_defaults = _effective_defaults(config, pair[0], pair[1])
        required_fields = (
            REQUIRED_BASE_FIELDS
            + required_score_fields
            + required_score_metric_fields
            + required_mmlu_subtask_score_fields
            + required_mmlu_subtask_score_metric_fields
            + _required_compression_param_fields(str(row.get("compression_method", "")))
        )
        quantized_method = str(row.get("compression_method", "")).lower() in QUANTIZED_METHODS
        storage_real = row.get("quantization_storage_real") is True
        if quantized_method:
            required_fields = required_fields + [
                "quantization_storage_format",
                "quantization_storage_real",
                "quantization_storage_dtypes",
            ]
            if storage_real:
                required_fields = required_fields + OPTIONAL_NUMERIC_FIELDS + ["effective_storage_quantized_backend"]
        for field in required_fields:
            value = row.get(field)
            if (
                field not in row
                or (value is None and field != "accuracy_num_fewshot")
                or (value == "" and field != "accuracy_failed_tasks")
            ):
                missing_fields.append({"model": pair[0], "method": pair[1], "field": field})
        type_violations.extend(
            _type_violations(
                row,
                pair,
                required_numeric_fields,
                INTEGER_NUMERIC_FIELDS,
                REQUIRED_BOOL_BASE_FIELDS + (["quantization_storage_real"] if quantized_method else []),
                REQUIRED_STRING_BASE_FIELDS
                + OPTIONAL_STRING_FIELDS
                + (["quantization_storage_format", "quantization_storage_dtypes"] if quantized_method else [])
                + required_score_metric_fields
                + required_mmlu_subtask_score_metric_fields,
            )
        )
        range_violations.extend(
            _range_violations(
                row,
                pair,
                NONNEGATIVE_NUMERIC_BASE_FIELDS + OPTIONAL_NONNEGATIVE_NUMERIC_FIELDS,
                POSITIVE_NUMERIC_BASE_FIELDS + OPTIONAL_POSITIVE_NUMERIC_FIELDS,
                required_ratio_fields,
            )
        )
        consistency_violations.extend(_deployment_consistency_violations(row, pair))
        consistency_violations.extend(_runtime_consistency_violations(row, pair))
        consistency_violations.extend(_accuracy_aggregate_consistency_violations(row, pair))
        consistency_violations.extend(_score_metric_consistency_violations(row, pair, config["defaults"]["tasks"]))
        consistency_violations.extend(_quantized_storage_consistency_violations(row, pair))
        consistency_violations.extend(_sparse_resource_consistency_violations(row, pair))
        consistency_violations.extend(_compression_provenance_violations(row, pair))
        consistency_violations.extend(_resource_backend_violations(row, pair))
        consistency_violations.extend(_config_consistency_violations(row, pair, pair_defaults))
        consistency_violations.extend(_paper_protocol_violations(row, pair, pair_defaults))
        failed_task_count = row.get("accuracy_failed_task_count")
        failed_tasks = row.get("accuracy_failed_tasks")
        accuracy_limit = row.get("accuracy_limit")
        if accuracy_limit not in (None, ""):
            accuracy_violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "field": "accuracy_limit",
                    "value": accuracy_limit,
                    "error": "final accuracy results must not use lm-eval limit/smoke subsets",
                }
            )
        if require_mmlu_subtasks:
            missing_mmlu_subtasks = [
                task
                for task in DEFAULT_MMLU_SUBTASKS
                if not isinstance(row.get(f"{task}_score"), (int, float))
                or isinstance(row.get(f"{task}_score"), bool)
            ]
            if missing_mmlu_subtasks:
                accuracy_violations.append(
                    {
                        "model": pair[0],
                        "method": pair[1],
                        "field": "mmlu_subtask_coverage",
                        "missing_count": len(missing_mmlu_subtasks),
                        "missing_tasks": missing_mmlu_subtasks,
                        "required_count": len(DEFAULT_MMLU_SUBTASKS),
                        "error": "final MMLU results must include all MMLU subtasks",
                    }
                )
        if isinstance(failed_task_count, (int, float)) and not isinstance(failed_task_count, bool) and isinstance(failed_tasks, str):
            failed_task_names = [item for item in failed_tasks.split(",") if item]
            if int(failed_task_count) == failed_task_count and int(failed_task_count) != len(failed_task_names):
                accuracy_violations.append(
                    {
                        "model": pair[0],
                        "method": pair[1],
                        "field": "accuracy_failed_tasks",
                        "failed_task_count": failed_task_count,
                        "failed_tasks": failed_tasks,
                        "expected_list_count": int(failed_task_count),
                        "actual_list_count": len(failed_task_names),
                    }
                )
        if isinstance(failed_task_count, (int, float)) and failed_task_count > 0:
            accuracy_violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "failed_task_count": failed_task_count,
                    "failed_tasks": failed_tasks,
                }
            )
        peaks = [
            value
            for value in (
                row.get("deployment_memory_measured_peak_gb"),
                row.get("peak_memory_allocated_gb"),
                row.get("peak_memory_reserved_gb"),
            )
            if isinstance(value, (int, float))
        ]
        measured_peak = max(peaks) if peaks else None
        row_limit = row.get("deployment_memory_limit_gb")
        pair_deployment_memory_limit_gb = args.deployment_memory_limit_gb
        if pair_deployment_memory_limit_gb is None:
            pair_deployment_memory_limit_gb = pair_defaults.get("deployment_memory_limit_gb", 0) or 0
        if (
            pair_deployment_memory_limit_gb
            and isinstance(row_limit, (int, float))
            and not isinstance(row_limit, bool)
            and row_limit != pair_deployment_memory_limit_gb
        ):
            memory_violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "row_limit_gb": row_limit,
                    "config_limit_gb": pair_deployment_memory_limit_gb,
                    "source": "deployment_memory_limit_gb",
                }
            )
        if pair_deployment_memory_limit_gb and measured_peak is not None and measured_peak > pair_deployment_memory_limit_gb:
            memory_violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "measured_peak_gb": measured_peak,
                    "limit_gb": pair_deployment_memory_limit_gb,
                }
            )
        if row.get("deployment_memory_limit_exceeded") is True:
            memory_violations.append(
                {
                    "model": pair[0],
                    "method": pair[1],
                    "measured_peak_gb": measured_peak,
                    "limit_gb": row.get("deployment_memory_limit_gb"),
                    "source": "deployment_json",
                }
            )
    consistency_violations.extend(_accuracy_retention_violations(rows_by_pair, required_pairs))

    report = {
        "ok": (
            not missing_pairs
            and not duplicate_pairs
            and not missing_fields
            and not type_violations
            and not range_violations
            and not consistency_violations
            and not accuracy_violations
            and not memory_violations
            and (args.allow_unexpected or not unexpected_pairs)
        ),
        "results": str(result_path),
        "expected_rows": len(required_pairs),
        "actual_rows": len(rows),
        "missing_pairs": [{"model": model, "method": method} for model, method in missing_pairs],
        "duplicate_pairs": duplicate_pairs,
        "unexpected_pairs": [{"model": model, "method": method} for model, method in unexpected_pairs],
        "missing_fields": missing_fields,
        "type_violations": type_violations,
        "range_violations": range_violations,
        "consistency_violations": consistency_violations,
        "accuracy_violations": accuracy_violations,
        "memory_violations": memory_violations,
        "deployment_memory_limit_gb": deployment_memory_limit_gb,
        "required_score_fields": required_score_fields,
        "required_score_metric_fields": required_score_metric_fields,
        "required_mmlu_subtask_score_fields": required_mmlu_subtask_score_fields,
        "required_mmlu_subtask_score_metric_fields": required_mmlu_subtask_score_metric_fields,
        "format_errors": [],
    }
    return _emit_report(report, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the final LLMCBench edge result table.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--results", required=True, help="matrix_results.json from collect_edge_results.py")
    parser.add_argument("--output", default=None)
    parser.add_argument("--only-model", action="append", default=[], help="Repeatable model name filter.")
    parser.add_argument("--only-method", action="append", default=[], help="Repeatable method filter.")
    parser.add_argument("--allow-unexpected", action="store_true")
    parser.add_argument("--deployment-memory-limit-gb", type=float, default=None, help="Override deployment memory limit for validation. Defaults to config defaults.deployment_memory_limit_gb.")
    return parser


def main() -> None:
    validate(build_parser().parse_args())


if __name__ == "__main__":
    main()
