import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

try:
    from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS
except ModuleNotFoundError:
    from prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS


def _write_validator_fixture(root: Path) -> Dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "models": [{"name": "TinyModel", "path": "/tmp/tiny"}],
        "methods": ["dense", "wanda", "gptq"],
        "defaults": {
            "tasks": "mmlu,arc_challenge,hellaswag,piqa,winogrande",
            "dtype": "fp16",
            "batch_size": "auto",
            "num_fewshot": None,
            "accuracy_limit": None,
            "split_accuracy_tasks": True,
            "continue_on_accuracy_error": True,
            "calibration_batch_size": 1,
            "max_memory": "0:22GiB,cpu:120GiB",
            "seqlen": 32,
            "input_length": 32,
            "max_new_tokens": 1,
            "sparsity_sample_tensors": -1,
            "warmup": 0,
            "repeat": 1,
            "deployment_memory_limit_gb": 1,
            "cuda_visible_devices": "0",
            "hf_cache_dir": "/tmp/hf-cache",
            "data_dir": "/tmp/datasets",
            "revision": None,
            "device_map": "auto",
            "prompt_file": "/tmp/prompts.txt",
            "calibration_file": "/tmp/calib.txt",
            "calibration_samples": 2,
            "sparsity": 0.5,
            "prune_pattern": "unstructured",
            "prune_lm_head": False,
            "quant_bits": 4,
            "group_size": 128,
            "desc_act": False,
            "awq_version": "GEMM",
        },
    }
    result = [
        {
            "model_name": "TinyModel",
            "compression_method": "dense",
            "source_model_path": "/tmp/tiny-source",
            "model_revision": None,
            "compressed_model_path": "/tmp/tiny-compressed",
            "compression_backend": "source_checkpoint_reference",
            "compression_quant_bits": 4,
            "compression_group_size": 128,
            "compression_desc_act": False,
            "compression_awq_version": "GEMM",
            "compression_sparsity": 0.5,
            "compression_prune_lm_head": False,
            "compression_calibration_samples": 2,
            "compression_calibration_file": "/tmp/calib.txt",
            "compression_dtype": "fp16",
            "compression_seqlen": 32,
            "compression_max_memory": "0:22GiB,cpu:120GiB",
            "compression_attn_implementation": "eager",
            "compression_calibration_batch_size": 1,
            "compression_materialized": False,
            "compression_uses_precompressed": False,
            "tokens_per_second": 1.0,
            "deployment_prompt_source": "file",
            "deployment_prompt_count": 1,
            "deployment_batch_size": 1,
            "deployment_input_length": 32,
            "deployment_max_new_tokens": 1,
            "deployment_warmup": 0,
            "deployment_repeat": 1,
            "deployment_input_tokens": 4,
            "deployment_generated_tokens": 1,
            "deployment_elapsed_seconds": 1.0,
            "resource_seqlen": 32,
            "resource_sparsity_sample_tensors": -1,
            "peak_memory_allocated_gb": 0.1,
            "peak_memory_reserved_gb": 0.1,
            "deployment_memory_limit_gb": 1,
            "deployment_memory_measured_peak_gb": 0.1,
            "deployment_memory_limit_exceeded": False,
            "resource_backend": "model_load",
            "resource_model_load_error": None,
            "storage_bytes": 100,
            "storage_gb": 100 / (1024**3),
            "effective_storage_bytes_sparse_approx": 100,
            "effective_storage_gb_sparse_approx": 100 / (1024**3),
            "total_params": 123,
            "sparsity": 0.0,
            "nonzero_ratio": 1.0,
            "flops": "1 MFLOPS",
            "flops_num": 1000000.0,
            "effective_flops_num_sparse_approx": 1000000.0,
            "macs": "0.5 MMACs",
            "compute_backend": "analytical_config_estimate",
            "mmlu_score": 0.1,
            "mmlu_score_metric": "aggregate:acc,none",
            "arc_challenge_score": 0.1,
            "arc_challenge_score_metric": "acc_norm,none",
            "hellaswag_score": 0.1,
            "hellaswag_score_metric": "acc_norm,none",
            "piqa_score": 0.1,
            "piqa_score_metric": "acc_norm,none",
            "winogrande_score": 0.1,
            "winogrande_score_metric": "acc,none",
            "accuracy_avg_score": 0.1,
            "accuracy_delta_vs_dense": 0.0,
            "accuracy_retention_vs_dense": 1.0,
            "accuracy_failed_task_count": 0,
            "accuracy_failed_tasks": "",
            "accuracy_dtype": "fp16",
            "accuracy_lm_eval_dtype": "float16",
            "accuracy_batch_size": "auto",
            "accuracy_num_fewshot": None,
            "accuracy_limit": None,
            "accuracy_data_dir": "/tmp/datasets",
            "accuracy_hf_datasets_cache": "/tmp/datasets",
            "accuracy_max_memory": "0:22GiB,cpu:120GiB",
            "accuracy_split_tasks": True,
            "accuracy_continue_on_error": True,
            "deployment_dtype": "fp16",
            "deployment_device_map": "auto",
            "deployment_max_memory": "0:22GiB,cpu:120GiB",
            "deployment_prompt_path": "/tmp/prompts.txt",
            "resource_prompt_source": "file",
            "resource_prompt_path": "/tmp/prompts.txt",
            "resource_prompt_count": 1,
            "resource_input_length": 32,
            "resource_max_new_tokens": 1,
            "runtime_python": "3.10.0",
            "runtime_profile": "stable",
            "runtime_cpu_count": 16,
            "runtime_system_memory_total_gb": 256.0,
            "runtime_hf_endpoint": "https://hf-mirror.com",
            "runtime_hf_home": "/tmp/hf-cache",
            "runtime_transformers_cache": "/tmp/hf-cache/transformers",
            "runtime_huggingface_hub_cache": "/tmp/hf-cache/hub",
            "runtime_hf_datasets_cache": "/tmp/datasets",
            "runtime_torch": "2.0.0",
            "runtime_transformers": "4.0.0",
            "runtime_cuda_available": True,
            "runtime_cuda_version": "12.1",
            "runtime_cudnn_version": "8900",
            "runtime_cuda_visible_devices": "0",
            "runtime_gpu_count": 1,
            "runtime_gpu_name": "NVIDIA Test GPU",
            "runtime_gpu_total_memory_gb": 24.0,
            "runtime_bitsandbytes_available": True,
            "runtime_bitsandbytes_version": "0.48.2",
            "runtime_bitsandbytes_error": None,
        }
    ]
    for subtask in DEFAULT_MMLU_SUBTASKS:
        result[0][f"{subtask}_score"] = 0.1
        result[0][f"{subtask}_score_metric"] = "acc,none"
    wanda_result = dict(
        result[0],
        compression_method="wanda",
        compressed_model_path="/tmp/tiny-wanda",
        compression_backend="built_in_activation_pruning",
        compression_materialized=True,
        compression_sparsity_semantics="element_zero_fraction",
        compression_prune_pattern="unstructured",
        compression_prune_structure="unstructured",
        mmlu_score=0.2,
        arc_challenge_score=0.2,
        hellaswag_score=0.2,
        piqa_score=0.2,
        winogrande_score=0.2,
        accuracy_avg_score=0.2,
        accuracy_delta_vs_dense=0.1,
        accuracy_retention_vs_dense=2.0,
    )
    for subtask in DEFAULT_MMLU_SUBTASKS:
        wanda_result[f"{subtask}_score"] = 0.2
    result.append(wanda_result)

    def real_quantized_row(**updates):
        return dict(
            result[0],
            compression_method="gptq",
            compression_backend="auto_gptq",
            compression_materialized=True,
            quantization_storage_real=True,
            quantization_storage_format="packed_integer",
            quantization_storage_dtypes="F16,I32",
            **updates,
        )

    config_path = root / "validator_config.json"
    result_path = root / "validator_results.json"
    run_report_path = root / "matrix_run_report.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    failed_result_path = root / "validator_failed_accuracy_results.json"
    failed_result = [dict(result[0], accuracy_failed_task_count=1, accuracy_failed_tasks="hellaswag")]
    failed_result_path.write_text(json.dumps(failed_result, indent=2) + "\n", encoding="utf-8")
    inconsistent_failed_tasks_result_path = root / "validator_inconsistent_failed_tasks_results.json"
    inconsistent_failed_tasks_result = [dict(result[0], accuracy_failed_task_count=0, accuracy_failed_tasks="hellaswag")]
    inconsistent_failed_tasks_result_path.write_text(
        json.dumps(inconsistent_failed_tasks_result, indent=2) + "\n",
        encoding="utf-8",
    )
    limited_accuracy_result_path = root / "validator_limited_accuracy_results.json"
    limited_accuracy_result = [dict(result[0], accuracy_limit=1)]
    limited_accuracy_result_path.write_text(json.dumps(limited_accuracy_result, indent=2) + "\n", encoding="utf-8")
    missing_mmlu_subtask_result_path = root / "validator_missing_mmlu_subtask_results.json"
    missing_mmlu_subtask_result = [dict(result[0])]
    missing_mmlu_subtask_result[0].pop(f"{DEFAULT_MMLU_SUBTASKS[0]}_score")
    missing_mmlu_subtask_result_path.write_text(json.dumps(missing_mmlu_subtask_result, indent=2) + "\n", encoding="utf-8")
    wrong_mmlu_subtask_score_result_path = root / "validator_wrong_mmlu_subtask_score_results.json"
    wrong_mmlu_subtask_score_result = [dict(result[0], **{f"{DEFAULT_MMLU_SUBTASKS[0]}_score": 1.2})]
    wrong_mmlu_subtask_score_result_path.write_text(
        json.dumps(wrong_mmlu_subtask_score_result, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_mmlu_subtask_metric_result_path = root / "validator_wrong_mmlu_subtask_metric_results.json"
    wrong_mmlu_subtask_metric_result = [dict(result[0], **{f"{DEFAULT_MMLU_SUBTASKS[0]}_score_metric": 123})]
    wrong_mmlu_subtask_metric_result_path.write_text(
        json.dumps(wrong_mmlu_subtask_metric_result, indent=2) + "\n",
        encoding="utf-8",
    )
    duplicate_result_path = root / "validator_duplicate_results.json"
    duplicate_result_path.write_text(json.dumps(result + result, indent=2) + "\n", encoding="utf-8")
    malformed_result_path = root / "validator_malformed_results.json"
    malformed_result_path.write_text(json.dumps({"rows": result}, indent=2) + "\n", encoding="utf-8")
    invalid_json_result_path = root / "validator_invalid_json_results.json"
    invalid_json_result_path.write_text("{not-json\n", encoding="utf-8")
    missing_result_path = root / "validator_missing_results.json"
    invalid_row_result_path = root / "validator_invalid_row_results.json"
    invalid_row_result_path.write_text(json.dumps(result + ["not-a-row"], indent=2) + "\n", encoding="utf-8")
    wrong_type_result_path = root / "validator_wrong_type_results.json"
    wrong_type_result = [dict(result[0], tokens_per_second="1.0")]
    wrong_type_result_path.write_text(json.dumps(wrong_type_result, indent=2) + "\n", encoding="utf-8")
    non_integer_result_path = root / "validator_non_integer_results.json"
    non_integer_result = [dict(result[0], deployment_repeat=1.5)]
    non_integer_result_path.write_text(json.dumps(non_integer_result, indent=2) + "\n", encoding="utf-8")
    wrong_string_type_result_path = root / "validator_wrong_string_type_results.json"
    wrong_string_type_result = [dict(result[0], runtime_python=3.10)]
    wrong_string_type_result_path.write_text(json.dumps(wrong_string_type_result, indent=2) + "\n", encoding="utf-8")
    wrong_runtime_cuda_result_path = root / "validator_wrong_runtime_cuda_results.json"
    wrong_runtime_cuda_result = [dict(result[0], runtime_cuda_available=False)]
    wrong_runtime_cuda_result_path.write_text(json.dumps(wrong_runtime_cuda_result, indent=2) + "\n", encoding="utf-8")
    wrong_runtime_cuda_visible_result_path = root / "validator_wrong_runtime_cuda_visible_results.json"
    wrong_runtime_cuda_visible_result = [dict(result[0], runtime_cuda_visible_devices="1")]
    wrong_runtime_cuda_visible_result_path.write_text(json.dumps(wrong_runtime_cuda_visible_result, indent=2) + "\n", encoding="utf-8")
    wrong_score_metric_type_result_path = root / "validator_wrong_score_metric_type_results.json"
    wrong_score_metric_type_result = [dict(result[0], piqa_score_metric=123)]
    wrong_score_metric_type_result_path.write_text(
        json.dumps(wrong_score_metric_type_result, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_target_metric_key_result_path = root / "validator_wrong_target_metric_key_results.json"
    wrong_target_metric_key_result = [dict(result[0], piqa_score_metric="acc,none")]
    wrong_target_metric_key_result_path.write_text(
        json.dumps(wrong_target_metric_key_result, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_mmlu_subtask_metric_key_result_path = root / "validator_wrong_mmlu_subtask_metric_key_results.json"
    wrong_mmlu_subtask_metric_key_result = [
        dict(result[0], **{f"{DEFAULT_MMLU_SUBTASKS[0]}_score_metric": "acc_norm,none"})
    ]
    wrong_mmlu_subtask_metric_key_result_path.write_text(
        json.dumps(wrong_mmlu_subtask_metric_key_result, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_retention_result_path = root / "validator_wrong_retention_results.json"
    wrong_retention_result = [result[0], dict(wanda_result, accuracy_retention_vs_dense=0.5)]
    wrong_retention_result_path.write_text(json.dumps(wrong_retention_result, indent=2) + "\n", encoding="utf-8")
    wrong_mmlu_aggregate_result_path = root / "validator_wrong_mmlu_aggregate_results.json"
    wrong_mmlu_aggregate_result = [dict(result[0], mmlu_score=0.2)]
    wrong_mmlu_aggregate_result_path.write_text(
        json.dumps(wrong_mmlu_aggregate_result, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_mmlu_metric_aggregate_result_path = root / "validator_wrong_mmlu_metric_aggregate_results.json"
    wrong_mmlu_metric_aggregate_result = [dict(result[0], mmlu_score_metric="acc,none")]
    wrong_mmlu_metric_aggregate_result_path.write_text(
        json.dumps(wrong_mmlu_metric_aggregate_result, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_accuracy_average_result_path = root / "validator_wrong_accuracy_average_results.json"
    wrong_accuracy_average_result = [dict(result[0], accuracy_avg_score=0.2)]
    wrong_accuracy_average_result_path.write_text(
        json.dumps(wrong_accuracy_average_result, indent=2) + "\n",
        encoding="utf-8",
    )
    missing_backend_result_path = root / "validator_missing_backend_results.json"
    missing_backend_result = [dict(result[0])]
    missing_backend_result[0].pop("compression_backend")
    missing_backend_result_path.write_text(json.dumps(missing_backend_result, indent=2) + "\n", encoding="utf-8")
    wrong_compression_backend_result_path = root / "validator_wrong_compression_backend_results.json"
    wrong_compression_backend_result = [dict(result[0], compression_backend="manual_backend")]
    wrong_compression_backend_result_path.write_text(
        json.dumps(wrong_compression_backend_result, indent=2) + "\n",
        encoding="utf-8",
    )
    missing_dtype_result_path = root / "validator_missing_dtype_results.json"
    missing_dtype_result = [dict(result[0])]
    missing_dtype_result[0].pop("compression_dtype")
    missing_dtype_result_path.write_text(json.dumps(missing_dtype_result, indent=2) + "\n", encoding="utf-8")
    missing_quant_param_result_path = root / "validator_missing_quant_param_results.json"
    missing_quant_param_result = [
        real_quantized_row(
            effective_storage_bytes_quantized_approx=61.5,
            effective_storage_gb_quantized_approx=61.5 / (1024**3),
            effective_storage_quantized_bits=4,
            effective_storage_quantized_backend="param_count_times_quant_bits",
        )
    ]
    missing_quant_param_result[0].pop("compression_quant_bits")
    missing_quant_param_result_path.write_text(
        json.dumps(missing_quant_param_result, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_dtype_result_path = root / "validator_wrong_dtype_results.json"
    wrong_dtype_result = [dict(result[0], compression_dtype="fp32")]
    wrong_dtype_result_path.write_text(json.dumps(wrong_dtype_result, indent=2) + "\n", encoding="utf-8")
    wrong_lm_eval_dtype_result_path = root / "validator_wrong_lm_eval_dtype_results.json"
    wrong_lm_eval_dtype_result = [dict(result[0], accuracy_lm_eval_dtype="float32")]
    wrong_lm_eval_dtype_result_path.write_text(
        json.dumps(wrong_lm_eval_dtype_result, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_num_fewshot_result_path = root / "validator_wrong_num_fewshot_results.json"
    wrong_num_fewshot_result = [dict(result[0], accuracy_num_fewshot=5)]
    wrong_num_fewshot_result_path.write_text(json.dumps(wrong_num_fewshot_result, indent=2) + "\n", encoding="utf-8")
    wrong_hf_cache_result_path = root / "validator_wrong_hf_cache_results.json"
    wrong_hf_cache_result = [dict(result[0], runtime_hf_home="/tmp/wrong-hf-cache")]
    wrong_hf_cache_result_path.write_text(json.dumps(wrong_hf_cache_result, indent=2) + "\n", encoding="utf-8")
    wrong_max_memory_result_path = root / "validator_wrong_max_memory_results.json"
    wrong_max_memory_result = [dict(result[0], compression_max_memory="0:24GiB,cpu:120GiB")]
    wrong_max_memory_result_path.write_text(json.dumps(wrong_max_memory_result, indent=2) + "\n", encoding="utf-8")
    wrong_device_map_result_path = root / "validator_wrong_device_map_results.json"
    wrong_device_map_result = [dict(result[0], deployment_device_map="cpu")]
    wrong_device_map_result_path.write_text(json.dumps(wrong_device_map_result, indent=2) + "\n", encoding="utf-8")
    wrong_compression_seqlen_result_path = root / "validator_wrong_compression_seqlen_results.json"
    wrong_compression_seqlen_result = [dict(result[0], compression_seqlen=64)]
    wrong_compression_seqlen_result_path.write_text(json.dumps(wrong_compression_seqlen_result, indent=2) + "\n", encoding="utf-8")
    wrong_calibration_batch_result_path = root / "validator_wrong_calibration_batch_results.json"
    wrong_calibration_batch_result = [dict(result[0], compression_calibration_batch_size=2)]
    wrong_calibration_batch_result_path.write_text(json.dumps(wrong_calibration_batch_result, indent=2) + "\n", encoding="utf-8")
    wrong_calibration_file_result_path = root / "validator_wrong_calibration_file_results.json"
    wrong_calibration_file_result = [dict(result[0], compression_calibration_file="/tmp/wrong-calib.txt")]
    wrong_calibration_file_result_path.write_text(json.dumps(wrong_calibration_file_result, indent=2) + "\n", encoding="utf-8")
    wrong_compression_param_result_path = root / "validator_wrong_compression_param_results.json"
    wrong_compression_param_result = [
        dict(
            result[0],
            compression_quant_bits=8,
            compression_group_size=64,
            compression_desc_act=True,
            compression_awq_version="GEMV",
            compression_sparsity=0.25,
            compression_prune_lm_head=True,
        )
    ]
    wrong_compression_param_result_path.write_text(json.dumps(wrong_compression_param_result, indent=2) + "\n", encoding="utf-8")
    wrong_deployment_setting_result_path = root / "validator_wrong_deployment_setting_results.json"
    wrong_deployment_setting_result = [dict(result[0], deployment_input_length=64)]
    wrong_deployment_setting_result_path.write_text(json.dumps(wrong_deployment_setting_result, indent=2) + "\n", encoding="utf-8")
    wrong_resource_setting_result_path = root / "validator_wrong_resource_setting_results.json"
    wrong_resource_setting_result = [dict(result[0], resource_seqlen=64)]
    wrong_resource_setting_result_path.write_text(json.dumps(wrong_resource_setting_result, indent=2) + "\n", encoding="utf-8")
    wrong_prompt_path_result_path = root / "validator_wrong_prompt_path_results.json"
    wrong_prompt_path_result = [
        dict(result[0], deployment_prompt_path="/tmp/wrong-prompts.txt", resource_prompt_path="/tmp/wrong-prompts.txt")
    ]
    wrong_prompt_path_result_path.write_text(json.dumps(wrong_prompt_path_result, indent=2) + "\n", encoding="utf-8")
    wrong_prompt_source_result_path = root / "validator_wrong_prompt_source_results.json"
    wrong_prompt_source_result = [dict(result[0], deployment_prompt_source="default", resource_prompt_source="default")]
    wrong_prompt_source_result_path.write_text(json.dumps(wrong_prompt_source_result, indent=2) + "\n", encoding="utf-8")
    wrong_resource_backend_result_path = root / "validator_wrong_resource_backend_results.json"
    wrong_resource_backend_result = [dict(result[0], resource_backend="config_only_fallback", resource_model_load_error="ImportError: missing backend")]
    wrong_resource_backend_result_path.write_text(json.dumps(wrong_resource_backend_result, indent=2) + "\n", encoding="utf-8")
    unknown_resource_backend_result_path = root / "validator_unknown_resource_backend_results.json"
    unknown_resource_backend_result = [dict(result[0], resource_backend="metadata_only")]
    unknown_resource_backend_result_path.write_text(
        json.dumps(unknown_resource_backend_result, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_compute_backend_result_path = root / "validator_wrong_compute_backend_results.json"
    wrong_compute_backend_result = [dict(result[0], compute_backend="manual")]
    wrong_compute_backend_result_path.write_text(
        json.dumps(wrong_compute_backend_result, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_accuracy_setting_result_path = root / "validator_wrong_accuracy_setting_results.json"
    wrong_accuracy_setting_result = [dict(result[0], accuracy_batch_size="4")]
    wrong_accuracy_setting_result_path.write_text(json.dumps(wrong_accuracy_setting_result, indent=2) + "\n", encoding="utf-8")
    wrong_dataset_path_result_path = root / "validator_wrong_dataset_path_results.json"
    wrong_dataset_path_result = [
        dict(result[0], accuracy_data_dir="/tmp/wrong-datasets", accuracy_hf_datasets_cache="/tmp/wrong-datasets")
    ]
    wrong_dataset_path_result_path.write_text(json.dumps(wrong_dataset_path_result, indent=2) + "\n", encoding="utf-8")
    non_finite_result_path = root / "validator_non_finite_results.json"
    non_finite_result_path.write_text(json.dumps([dict(result[0], tokens_per_second=float("nan"))]) + "\n", encoding="utf-8")
    missing_deployment_gate_result_path = root / "validator_missing_deployment_gate_results.json"
    missing_deployment_gate_result = [dict(result[0])]
    missing_deployment_gate_result[0].pop("deployment_memory_measured_peak_gb")
    missing_deployment_gate_result_path.write_text(
        json.dumps(missing_deployment_gate_result, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_deployment_limit_result_path = root / "validator_wrong_deployment_limit_results.json"
    wrong_deployment_limit_result = [dict(result[0], deployment_memory_limit_gb=2)]
    wrong_deployment_limit_result_path.write_text(json.dumps(wrong_deployment_limit_result, indent=2) + "\n", encoding="utf-8")
    wrong_quantized_storage_result_path = root / "validator_wrong_quantized_storage_results.json"
    wrong_quantized_storage_result = [
        real_quantized_row(
            effective_storage_bytes_quantized_approx=61.5,
            effective_storage_gb_quantized_approx=-0.1,
            effective_storage_quantized_bits=4,
            effective_storage_quantized_backend="param_count_times_quant_bits",
        )
    ]
    wrong_quantized_storage_result_path.write_text(json.dumps(wrong_quantized_storage_result, indent=2) + "\n", encoding="utf-8")
    inconsistent_quantized_storage_result_path = root / "validator_inconsistent_quantized_storage_results.json"
    inconsistent_quantized_storage_result = [
        real_quantized_row(
            compression_quant_bits=4,
            effective_storage_bytes_quantized_approx=61.5,
            effective_storage_gb_quantized_approx=61.5 / (1024**3),
            effective_storage_quantized_bits=8,
            effective_storage_quantized_backend="param_count_times_quant_bits",
        )
    ]
    inconsistent_quantized_storage_result_path.write_text(
        json.dumps(inconsistent_quantized_storage_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_quantized_storage_bytes_result_path = root / "validator_inconsistent_quantized_storage_bytes_results.json"
    inconsistent_quantized_storage_bytes_result = [
        real_quantized_row(
            total_params=123,
            compression_quant_bits=4,
            effective_storage_bytes_quantized_approx=25,
            effective_storage_gb_quantized_approx=25 / (1024**3),
            effective_storage_quantized_bits=4,
            effective_storage_quantized_backend="param_count_times_quant_bits",
        )
    ]
    inconsistent_quantized_storage_bytes_result_path.write_text(
        json.dumps(inconsistent_quantized_storage_bytes_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_quantized_storage_gb_result_path = root / "validator_inconsistent_quantized_storage_gb_results.json"
    inconsistent_quantized_storage_gb_result = [
        real_quantized_row(
            effective_storage_bytes_quantized_approx=61.5,
            effective_storage_gb_quantized_approx=0.025,
            effective_storage_quantized_bits=4,
            effective_storage_quantized_backend="param_count_times_quant_bits",
        )
    ]
    inconsistent_quantized_storage_gb_result_path.write_text(
        json.dumps(inconsistent_quantized_storage_gb_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_quantized_storage_backend_result_path = root / "validator_inconsistent_quantized_storage_backend_results.json"
    inconsistent_quantized_storage_backend_result = [
        real_quantized_row(
            effective_storage_bytes_quantized_approx=61.5,
            effective_storage_gb_quantized_approx=61.5 / (1024**3),
            effective_storage_quantized_bits=4,
            effective_storage_quantized_backend="manual_estimate",
        )
    ]
    inconsistent_quantized_storage_backend_result_path.write_text(
        json.dumps(inconsistent_quantized_storage_backend_result, indent=2) + "\n",
        encoding="utf-8",
    )
    non_quantized_storage_result_path = root / "validator_non_quantized_storage_results.json"
    non_quantized_storage_result = [
        dict(
            result[0],
            effective_storage_bytes_quantized_approx=61.5,
            effective_storage_gb_quantized_approx=61.5 / (1024**3),
            effective_storage_quantized_bits=4,
            effective_storage_quantized_backend="param_count_times_quant_bits",
        )
    ]
    non_quantized_storage_result_path.write_text(
        json.dumps(non_quantized_storage_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_storage_gb_result_path = root / "validator_inconsistent_storage_gb_results.json"
    inconsistent_storage_gb_result = [dict(result[0], storage_bytes=100, storage_gb=0.01)]
    inconsistent_storage_gb_result_path.write_text(
        json.dumps(inconsistent_storage_gb_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_nonzero_ratio_result_path = root / "validator_inconsistent_nonzero_ratio_results.json"
    inconsistent_nonzero_ratio_result = [dict(result[0], sparsity=0.25, nonzero_ratio=0.5)]
    inconsistent_nonzero_ratio_result_path.write_text(
        json.dumps(inconsistent_nonzero_ratio_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_sparse_storage_result_path = root / "validator_inconsistent_sparse_storage_results.json"
    inconsistent_sparse_storage_result = [
        dict(result[0], nonzero_ratio=0.5, storage_bytes=100, effective_storage_bytes_sparse_approx=75)
    ]
    inconsistent_sparse_storage_result_path.write_text(
        json.dumps(inconsistent_sparse_storage_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_sparse_flops_result_path = root / "validator_inconsistent_sparse_flops_results.json"
    inconsistent_sparse_flops_result = [
        dict(result[0], nonzero_ratio=0.5, flops_num=1000000.0, effective_flops_num_sparse_approx=750000.0)
    ]
    inconsistent_sparse_flops_result_path.write_text(
        json.dumps(inconsistent_sparse_flops_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_precompressed_result_path = root / "validator_inconsistent_precompressed_results.json"
    inconsistent_precompressed_result = [
        dict(result[0], compression_backend="precompressed_checkpoint", compression_uses_precompressed=False)
    ]
    inconsistent_precompressed_result_path.write_text(
        json.dumps(inconsistent_precompressed_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_materialized_result_path = root / "validator_inconsistent_materialized_results.json"
    inconsistent_materialized_result = [dict(result[0], compression_materialized=True)]
    inconsistent_materialized_result_path.write_text(
        json.dumps(inconsistent_materialized_result, indent=2) + "\n",
        encoding="utf-8",
    )
    missing_quantized_storage_result_path = root / "validator_missing_quantized_storage_results.json"
    missing_quantized_storage_result = [
        {
            key: value
            for key, value in real_quantized_row().items()
            if key not in {
                "effective_storage_bytes_quantized_approx",
                "effective_storage_gb_quantized_approx",
                "effective_storage_quantized_bits",
                "effective_storage_quantized_backend",
            }
        }
    ]
    missing_quantized_storage_result_path.write_text(
        json.dumps(missing_quantized_storage_result, indent=2) + "\n",
        encoding="utf-8",
    )
    out_of_range_result_path = root / "validator_out_of_range_results.json"
    out_of_range_result = [dict(result[0], tokens_per_second=-1.0, piqa_score=1.2)]
    out_of_range_result_path.write_text(json.dumps(out_of_range_result, indent=2) + "\n", encoding="utf-8")
    zero_measurement_result_path = root / "validator_zero_measurement_results.json"
    zero_measurement_result = [
        dict(
            result[0],
            tokens_per_second=0.0,
            deployment_generated_tokens=0,
            storage_bytes=0,
            total_params=0,
            flops_num=0.0,
        )
    ]
    zero_measurement_result_path.write_text(
        json.dumps(zero_measurement_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_deployment_result_path = root / "validator_inconsistent_deployment_results.json"
    inconsistent_deployment_result = [
        dict(
            result[0],
            peak_memory_reserved_gb=0.5,
            deployment_memory_measured_peak_gb=0.1,
            deployment_memory_limit_gb=1,
            deployment_memory_limit_exceeded=False,
        )
    ]
    inconsistent_deployment_result_path.write_text(
        json.dumps(inconsistent_deployment_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_deployment_input_tokens_result_path = root / "validator_inconsistent_deployment_input_tokens_results.json"
    inconsistent_deployment_input_tokens_result = [
        dict(result[0], deployment_prompt_count=1, deployment_batch_size=1, deployment_input_length=32, deployment_input_tokens=64)
    ]
    inconsistent_deployment_input_tokens_result_path.write_text(
        json.dumps(inconsistent_deployment_input_tokens_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_deployment_generated_tokens_result_path = root / "validator_inconsistent_deployment_generated_tokens_results.json"
    inconsistent_deployment_generated_tokens_result = [
        dict(
            result[0],
            deployment_prompt_count=1,
            deployment_repeat=1,
            deployment_max_new_tokens=1,
            deployment_generated_tokens=2,
        )
    ]
    inconsistent_deployment_generated_tokens_result_path.write_text(
        json.dumps(inconsistent_deployment_generated_tokens_result, indent=2) + "\n",
        encoding="utf-8",
    )
    inconsistent_deployment_tps_result_path = root / "validator_inconsistent_deployment_tps_results.json"
    inconsistent_deployment_tps_result = [
        dict(result[0], deployment_generated_tokens=1, deployment_elapsed_seconds=2.0, tokens_per_second=1.0)
    ]
    inconsistent_deployment_tps_result_path.write_text(
        json.dumps(inconsistent_deployment_tps_result, indent=2) + "\n",
        encoding="utf-8",
    )
    run_report_path.write_text(
        json.dumps(
            {
                "completed": ["/tmp/tiny/summary.json"],
                "skipped": [],
                "skipped_external_not_configured": [],
                "failed": [],
                "jobs": [
                    {
                        "model": "TinyModel",
                        "method": "dense",
                        "status": "completed",
                        "summary": "/tmp/tiny/summary.json",
                        "log": "/tmp/tiny/job.log",
                        "log_tail": "ok",
                        "elapsed_seconds": 1.0,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "config": str(config_path),
        "results": str(result_path),
        "failed_results": str(failed_result_path),
        "inconsistent_failed_tasks_results": str(inconsistent_failed_tasks_result_path),
        "limited_accuracy_results": str(limited_accuracy_result_path),
        "missing_mmlu_subtask_results": str(missing_mmlu_subtask_result_path),
        "wrong_mmlu_subtask_score_results": str(wrong_mmlu_subtask_score_result_path),
        "wrong_mmlu_subtask_metric_results": str(wrong_mmlu_subtask_metric_result_path),
        "duplicate_results": str(duplicate_result_path),
        "malformed_results": str(malformed_result_path),
        "invalid_json_results": str(invalid_json_result_path),
        "missing_results": str(missing_result_path),
        "invalid_row_results": str(invalid_row_result_path),
        "wrong_type_results": str(wrong_type_result_path),
        "non_integer_results": str(non_integer_result_path),
        "wrong_string_type_results": str(wrong_string_type_result_path),
        "wrong_runtime_cuda_results": str(wrong_runtime_cuda_result_path),
        "wrong_runtime_cuda_visible_results": str(wrong_runtime_cuda_visible_result_path),
        "wrong_score_metric_type_results": str(wrong_score_metric_type_result_path),
        "wrong_target_metric_key_results": str(wrong_target_metric_key_result_path),
        "wrong_mmlu_subtask_metric_key_results": str(wrong_mmlu_subtask_metric_key_result_path),
        "wrong_retention_results": str(wrong_retention_result_path),
        "wrong_mmlu_aggregate_results": str(wrong_mmlu_aggregate_result_path),
        "wrong_mmlu_metric_aggregate_results": str(wrong_mmlu_metric_aggregate_result_path),
        "wrong_accuracy_average_results": str(wrong_accuracy_average_result_path),
        "missing_backend_results": str(missing_backend_result_path),
        "wrong_compression_backend_results": str(wrong_compression_backend_result_path),
        "missing_dtype_results": str(missing_dtype_result_path),
        "missing_quant_param_results": str(missing_quant_param_result_path),
        "wrong_dtype_results": str(wrong_dtype_result_path),
        "wrong_lm_eval_dtype_results": str(wrong_lm_eval_dtype_result_path),
        "wrong_num_fewshot_results": str(wrong_num_fewshot_result_path),
        "wrong_hf_cache_results": str(wrong_hf_cache_result_path),
        "wrong_max_memory_results": str(wrong_max_memory_result_path),
        "wrong_device_map_results": str(wrong_device_map_result_path),
        "wrong_compression_seqlen_results": str(wrong_compression_seqlen_result_path),
        "wrong_calibration_batch_results": str(wrong_calibration_batch_result_path),
        "wrong_calibration_file_results": str(wrong_calibration_file_result_path),
        "wrong_compression_param_results": str(wrong_compression_param_result_path),
        "wrong_deployment_setting_results": str(wrong_deployment_setting_result_path),
        "wrong_resource_setting_results": str(wrong_resource_setting_result_path),
        "wrong_prompt_path_results": str(wrong_prompt_path_result_path),
        "wrong_prompt_source_results": str(wrong_prompt_source_result_path),
        "wrong_resource_backend_results": str(wrong_resource_backend_result_path),
        "unknown_resource_backend_results": str(unknown_resource_backend_result_path),
        "wrong_compute_backend_results": str(wrong_compute_backend_result_path),
        "wrong_accuracy_setting_results": str(wrong_accuracy_setting_result_path),
        "wrong_dataset_path_results": str(wrong_dataset_path_result_path),
        "non_finite_results": str(non_finite_result_path),
        "missing_deployment_gate_results": str(missing_deployment_gate_result_path),
        "wrong_deployment_limit_results": str(wrong_deployment_limit_result_path),
        "wrong_quantized_storage_results": str(wrong_quantized_storage_result_path),
        "inconsistent_quantized_storage_results": str(inconsistent_quantized_storage_result_path),
        "inconsistent_quantized_storage_bytes_results": str(inconsistent_quantized_storage_bytes_result_path),
        "inconsistent_quantized_storage_gb_results": str(inconsistent_quantized_storage_gb_result_path),
        "inconsistent_quantized_storage_backend_results": str(inconsistent_quantized_storage_backend_result_path),
        "non_quantized_storage_results": str(non_quantized_storage_result_path),
        "inconsistent_storage_gb_results": str(inconsistent_storage_gb_result_path),
        "inconsistent_nonzero_ratio_results": str(inconsistent_nonzero_ratio_result_path),
        "inconsistent_sparse_storage_results": str(inconsistent_sparse_storage_result_path),
        "inconsistent_sparse_flops_results": str(inconsistent_sparse_flops_result_path),
        "inconsistent_precompressed_results": str(inconsistent_precompressed_result_path),
        "inconsistent_materialized_results": str(inconsistent_materialized_result_path),
        "missing_quantized_storage_results": str(missing_quantized_storage_result_path),
        "out_of_range_results": str(out_of_range_result_path),
        "zero_measurement_results": str(zero_measurement_result_path),
        "inconsistent_deployment_results": str(inconsistent_deployment_result_path),
        "inconsistent_deployment_input_tokens_results": str(inconsistent_deployment_input_tokens_result_path),
        "inconsistent_deployment_generated_tokens_results": str(inconsistent_deployment_generated_tokens_result_path),
        "inconsistent_deployment_tps_results": str(inconsistent_deployment_tps_result_path),
        "run_report": str(run_report_path),
    }


def _write_collect_fixture(root: Path) -> Dict[str, str]:
    job = root / "TinyModel" / "dense"
    wanda_job = root / "TinyModel" / "wanda"
    job.mkdir(parents=True, exist_ok=True)
    wanda_job.mkdir(parents=True, exist_ok=True)
    (job / "job.log").write_text("ok\n", encoding="utf-8")
    (wanda_job / "job.log").write_text("ok\n", encoding="utf-8")
    job_overrides = {
        "tasks": "mmlu,arc_challenge,hellaswag,piqa,winogrande",
        "dtype": "fp16",
        "batch_size": "auto",
        "num_fewshot": None,
        "accuracy_limit": None,
        "split_accuracy_tasks": True,
        "continue_on_accuracy_error": True,
        "calibration_batch_size": 1,
        "max_memory": "0:22GiB,cpu:120GiB",
        "seqlen": 32,
        "input_length": 32,
        "max_new_tokens": 1,
        "sparsity_sample_tensors": -1,
        "warmup": 0,
        "repeat": 1,
        "deployment_memory_limit_gb": 1,
        "hf_cache_dir": "/tmp/hf-cache",
        "data_dir": "/tmp/datasets",
        "revision": None,
        "device_map": "auto",
        "prompt_file": "/tmp/prompts.txt",
        "calibration_file": "/tmp/calib.txt",
        "calibration_samples": 2,
        "sparsity": 0.5,
        "prune_pattern": "unstructured",
        "prune_lm_head": False,
        "quant_bits": 4,
        "group_size": 128,
        "desc_act": False,
        "awq_version": "GEMM",
        "output_root": str(root),
    }
    (root / "matrix_config_resolved.json").write_text(
        json.dumps(
            {
                "source_config": "/tmp/llmcbench_validator_smoke/validator_config.json",
                "code": {
                    "commit": subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        text=True,
                        capture_output=True,
                        check=False,
                    ).stdout.strip(),
                    "branch": subprocess.run(
                        ["git", "branch", "--show-current"],
                        text=True,
                        capture_output=True,
                        check=False,
                    ).stdout.strip(),
                    "origin": subprocess.run(
                        ["git", "remote", "get-url", "origin"],
                        text=True,
                        capture_output=True,
                        check=False,
                    ).stdout.strip(),
                    "dirty": bool(subprocess.run(
                        ["git", "status", "--porcelain"],
                        text=True,
                        capture_output=True,
                        check=False,
                    ).stdout.strip()),
                },
                "selected_models": ["TinyModel"],
                "selected_methods": ["dense"],
                "models": [{"name": "TinyModel", "path": "/tmp/tiny"}],
                "methods": ["dense"],
                "defaults": {
                    "tasks": "mmlu,arc_challenge,hellaswag,piqa,winogrande",
                    "dtype": "fp16",
                    "batch_size": "auto",
                    "split_accuracy_tasks": True,
                    "continue_on_accuracy_error": True,
                    "calibration_batch_size": 1,
                    "max_memory": "0:22GiB,cpu:120GiB",
                    "seqlen": 32,
                    "input_length": 32,
                    "max_new_tokens": 1,
                    "warmup": 0,
                    "repeat": 1,
                    "deployment_memory_limit_gb": 1,
                    "hf_cache_dir": "/tmp/hf-cache",
                    "output_root": str(root),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "matrix_run_report.json").write_text(
        json.dumps(
            {
                "completed": [str(job / "summary.json"), str(wanda_job / "summary.json")],
                "skipped": [],
                "skipped_external_not_configured": [],
                "failed": [],
                "jobs": [
                    {
                        "model": "TinyModel",
                        "method": "dense",
                        "status": "completed",
                        "summary": str(job / "summary.json"),
                        "log": str(job / "job.log"),
                        "log_tail": "ok",
                        "job_overrides": job_overrides,
                        "elapsed_seconds": 1.0,
                    },
                    {
                        "model": "TinyModel",
                        "method": "wanda",
                        "status": "completed",
                        "summary": str(wanda_job / "summary.json"),
                        "log": str(wanda_job / "job.log"),
                        "log_tail": "ok",
                        "job_overrides": job_overrides,
                        "elapsed_seconds": 1.0,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    asset_paths = {
        name: {
            "path": f"/tmp/{name}",
            "probe_path": "/tmp",
            "exists": True,
            "is_dir": True,
            "is_writable": True,
            "can_create": True,
            "writable_probe_path": "/tmp",
            "writable_probe_path_is_writable": True,
            "free_gb": 100.0,
            "total_gb": 200.0,
            "mount": {"ok": True, "target": "/tmp", "source": "/dev/nvme0n1p1", "fstype": "ext4"},
            "block_device": {"device": "nvme0n1", "rotational": False, "is_non_rotational": True},
        }
        for name in ("hf_cache_dir", "data_dir", "output_root")
    }
    (root / "edge_asset_audit.json").write_text(
        json.dumps({"ok": True, "models": [], "precompressed": [], "missing_files": [], "low_disk": [], "unwritable_paths": [], "paths": asset_paths}) + "\n",
        encoding="utf-8",
    )
    verify_env_payload = {
        "ok": True,
        "cuda_visible_devices": {
            "actual": "0",
            "expected": "0",
            "matches_config": True,
            "required": True,
        },
    }
    preflight_steps = [
        {"name": "asset_audit", "cmd": [sys.executable, "tools/audit_edge_assets.py"], "required": True, "returncode": 0, "ok": True},
        {"name": "verify_env", "cmd": [sys.executable, "tools/verify_edge_env.py"], "required": True, "returncode": 0, "ok": True, "payload": verify_env_payload},
        {"name": "method_readiness", "cmd": [sys.executable, "tools/check_method_readiness.py"], "required": True, "returncode": 0, "ok": True},
        {"name": "matrix_dry_run", "cmd": [sys.executable, "tools/run_edge_matrix.py", "--config", "configs/edge_4090_matrix.json", "--dry-run"], "required": True, "returncode": 0, "ok": True},
    ]
    (root / "edge_preflight_report.json").write_text(
        json.dumps({"ok": True, "steps": preflight_steps}) + "\n",
        encoding="utf-8",
    )
    target_paths_payload = {
        "ok": True,
        "dry_run": False,
        "directories": {
            "hf_cache_dir": {"path": "/tmp/hf_cache", "ok": True, "exists_after": True, "is_dir_after": True},
            "data_dir": {"path": "/tmp/datasets", "ok": True, "exists_after": True, "is_dir_after": True},
            "output_root": {"path": str(root), "ok": True, "exists_after": True, "is_dir_after": True},
        },
        "files": {
            "calibration_file": {"path": "/tmp/calib.txt", "ok": True, "exists_after": True, "is_file_after": True},
            "prompt_file": {"path": "/tmp/prompts.txt", "ok": True, "exists_after": True, "is_file_after": True},
        },
    }
    (root / "edge_target_paths_report.json").write_text(
        json.dumps(target_paths_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "current_goal_audit.json").write_text(
        json.dumps(
            {
                "completion_state": {"status": "complete", "complete": True},
                "prompt_to_artifact_checklist": {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    prepare_payload = {
        "ok": True,
        "dry_run": False,
        "metadata_only": False,
        "models": [{"name": "TinyModel", "ok": True}],
        "files": [{"path": "/tmp/calib.txt", "ok": True}, {"path": "/tmp/prompts.txt", "ok": True}],
        "lm_eval_probe": None,
    }
    (root / "edge_prepare_assets_report.json").write_text(
        json.dumps(prepare_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    prepare_metadata_payload = dict(prepare_payload, metadata_only=True)
    (root / "edge_prepare_metadata_report.json").write_text(
        json.dumps(prepare_metadata_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    prewarm_tasks = DEFAULT_MMLU_SUBTASKS + ["arc_challenge", "hellaswag", "piqa", "winogrande"]
    prewarm_pairs = [{"model": "TinyModel", "task": task, "output_path": f"/tmp/prewarm/{task}"} for task in prewarm_tasks]
    (root / "edge_prewarm_report.json").write_text(
        json.dumps(
            {
                "ok": True,
                "tasks": prewarm_tasks,
                "target_count": 1,
                "targets": [{"name": "TinyModel", "path": "/tmp/tiny"}],
                "attempted": prewarm_pairs,
                "succeeded": [{"model": item["model"], "task": item["task"]} for item in prewarm_pairs],
                "failed": [],
                "dry_run": False,
                "dtype": "fp16",
                "attn_implementation": None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    current_code = json.loads((root / "matrix_config_resolved.json").read_text(encoding="utf-8"))["code"]
    handoff_matrix_finalize_cmd = (
        "python tools/run_edge_matrix.py --config /tmp/llmcbench_validator_smoke/validator_config.json "
        f"--output-root {root} --stop-on-error --finalize "
        "--require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report "
        "--require-handoff-manifest --require-target-readiness-report "
        "--require-optional-backends-report --require-quant-runtime-report "
        "--require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 "
        "--require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 "
        "--require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. "
        "--require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 "
        "--require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com "
        "--require-code-match --require-clean-worktree"
    )
    handoff_completion_audit_cmd = (
        "python tools/audit_edge_completion.py --config /tmp/llmcbench_validator_smoke/validator_config.json "
        f"--root {root} --require-finalize-report "
        "--require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report "
        "--require-handoff-manifest --require-target-readiness-report "
        "--require-optional-backends-report --require-quant-runtime-report "
        "--require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 "
        "--require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 "
        "--require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. "
        "--require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 "
        "--require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com "
        "--require-code-match --require-clean-worktree"
    )
    (root / "edge_handoff_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "code": current_code,
                "config": {"models": [{"name": "TinyModel", "path": "/tmp/tiny"}], "methods": ["dense"]},
                "target_environment": {"hardware": {}, "software": {}, "memory_controls": {}},
                "paths": {"output_root": str(root)},
                "commands": {
                    "setup": ["setup"],
                    "prepare": ["prepare"],
                    "prepare_precompressed": [
                        "python tools/prepare_precompressed_assets.py --output edge_prepare_precompressed_report.json --output-config edge_matrix_with_precompressed.json --fail-on-error",
                    ],
                    "preflight": [
                        "preflight --check-optional-backend-resolution --check-quant-runtime",
                    ],
                    "prewarm": [
                        "python tools/prewarm_lm_eval_tasks.py --expand-mmlu --report-output edge_prewarm_report.json",
                    ],
                    "readiness": [
                        "python tools/audit_target_readiness.py --check-method-readiness --require-non-rotational-storage --require-prewarm --require-handoff-manifest --require-optional-backends-report --require-quant-runtime-report",
                    ],
                    "precompressed_config_followup": [
                        "python tools/audit_edge_assets.py --config edge_matrix_with_precompressed.json",
                        "python tools/audit_target_readiness.py --config edge_matrix_with_precompressed.json --require-prepare-precompressed-report",
                        "python tools/run_edge_matrix.py --config edge_matrix_with_precompressed.json --require-prepare-precompressed-report",
                        "python tools/audit_edge_completion.py --config edge_matrix_with_precompressed.json --require-prepare-precompressed-report",
                    ],
                    "run": ["run"],
                    "manual_finalize": [handoff_matrix_finalize_cmd, handoff_completion_audit_cmd],
                    "post_run_audit": [
                        "python tools/audit_target_readiness.py --check-method-readiness --require-non-rotational-storage --require-prewarm --require-handoff-manifest --require-optional-backends-report --require-quant-runtime-report",
                        "python tools/build_goal_audit.py",
                    ],
                    "baseline_post_run_audit": [
                        "python tools/audit_target_readiness.py --config configs/edge_4090_matrix.json --check-method-readiness --require-non-rotational-storage --require-prewarm --require-handoff-manifest --require-optional-backends-report --require-quant-runtime-report",
                        "python tools/build_goal_audit.py",
                    ],
                    "precompressed_config_post_run_audit": [
                        "python tools/audit_target_readiness.py --config edge_matrix_with_precompressed.json --check-method-readiness --require-non-rotational-storage --require-prewarm --require-handoff-manifest --require-optional-backends-report --require-quant-runtime-report --require-prepare-precompressed-report",
                        "python tools/build_goal_audit.py",
                    ],
                },
                "required_artifacts": {
                    "matrix_config_resolved.json": {"path": str(root / "matrix_config_resolved.json")},
                    "edge_asset_audit.json": {"path": str(root / "edge_asset_audit.json")},
                    "edge_preflight_report.json": {"path": str(root / "edge_preflight_report.json")},
                    "edge_prewarm_report.json": {"path": str(root / "edge_prewarm_report.json")},
                    "edge_handoff_manifest.json": {"path": str(root / "edge_handoff_manifest.json")},
                    "edge_optional_backends_report.json": {"path": str(root / "edge_optional_backends_report.json")},
                    "edge_quant_runtime_report.json": {"path": str(root / "edge_quant_runtime_report.json")},
                    "matrix_run_report.json": {"path": str(root / "matrix_run_report.json")},
                    "matrix_finalize_report.json": {"path": str(root / "matrix_finalize_report.json")},
                    "matrix_results.json": {"path": str(root / "matrix_results.json")},
                    "matrix_results.csv": {"path": str(root / "matrix_results.csv")},
                    "edge_report.md": {"path": str(root / "edge_report.md")},
                    "completion_audit.json": {"path": str(root / "completion_audit.json")},
                    "target_readiness.json": {"path": str(root / "target_readiness.json")},
                },
                "completion_gates": {
                    "coverage": {
                        "required_mmlu_subtask_count": len(DEFAULT_MMLU_SUBTASKS),
                        "required_mmlu_subtasks": list(DEFAULT_MMLU_SUBTASKS),
                    },
                    "reports": [
                        "target_readiness.json with ok=true, completion_ok=true and no before_matrix_blockers",
                        "completion_audit.json with ok=true",
                    ],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (job / "summary.json").write_text(
        json.dumps(
            {
                "model_name": "TinyModel",
                "compression_method": "dense",
                "compressed_model_path": "/tmp/tiny",
                "runtime": {
                    "python": "3.10.0",
                    "runtime_profile": "stable",
                    "cpu_count": 16,
                    "system_memory_total_gb": 256.0,
                    "hf_endpoint": "https://hf-mirror.com",
                    "hf_home": "/tmp/hf-cache",
                    "transformers_cache": "/tmp/hf-cache/transformers",
                    "huggingface_hub_cache": "/tmp/hf-cache/hub",
                    "hf_datasets_cache": "/tmp/datasets",
                    "torch": "2.0.0",
                    "transformers": "4.0.0",
                    "cuda_available": True,
                    "cuda_version": "12.1",
                    "cudnn_version": "8900",
                    "cuda_visible_devices": "0",
                    "gpus": [{"index": 0, "name": "NVIDIA Test GPU", "total_memory_gb": 24.0}],
                    "optional_imports": {"bitsandbytes": {"ok": True, "version": "0.48.2"}},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (job / "compression.json").write_text(
        json.dumps(
            {
                "manifest": {
                    "source_model_path": "/tmp/tiny-source",
                    "revision": None,
                    "backend": "source_checkpoint_reference",
                    "quant_bits": 4,
                    "group_size": 128,
                    "desc_act": False,
                    "awq_version": "GEMM",
                    "sparsity": 0.5,
                    "prune_lm_head": False,
                    "calibration_samples": 2,
                    "calibration_file": "/tmp/calib.txt",
                    "calibration_batch_size": 1,
                    "dtype": "fp16",
                    "seqlen": 32,
                    "max_memory": "0:22GiB,cpu:120GiB",
                    "attn_implementation": "eager",
                    "materialized": False,
                    "note": "collect fixture dense checkpoint",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (job / "accuracy.json").write_text(
        json.dumps(
            {
                "dtype": "fp16",
                "lm_eval_dtype": "float16",
                "batch_size": "auto",
                "num_fewshot": None,
                "limit": None,
                "device": "cuda:0",
                "data_dir": "/tmp/datasets",
                "hf_datasets_cache": "/tmp/datasets",
                "max_memory": "0:22GiB,cpu:120GiB",
                "attn_implementation": None,
                "split_accuracy_tasks": True,
                "continue_on_accuracy_error": True,
                "results": {
                    "mmlu_abstract_algebra": {"acc,none": 0.25},
                    "arc_challenge": {"acc,none": 0.5, "acc_norm,none": 0.55},
                    "hellaswag": {"acc,none": 0.6, "acc_norm,none": 0.65},
                    "piqa": {"acc,none": 0.7, "acc_norm,none": 0.75},
                    "winogrande": {"acc,none": 0.8},
                },
                "failed_tasks": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dense_accuracy_path = job / "accuracy.json"
    dense_accuracy = json.loads(dense_accuracy_path.read_text(encoding="utf-8"))
    for subtask in DEFAULT_MMLU_SUBTASKS:
        dense_accuracy["results"].setdefault(subtask, {"acc,none": 0.25})
    dense_accuracy_path.write_text(json.dumps(dense_accuracy, indent=2) + "\n", encoding="utf-8")
    (job / "deployment.json").write_text(
        json.dumps(
            {
                "tokens_per_second": 1.0,
                "dtype": "fp16",
                "device_map": "auto",
                "max_memory": "0:22GiB,cpu:120GiB",
                "attn_implementation": None,
                "prompt_source": "file",
                "prompt_path": "/tmp/prompts.txt",
                "prompt_count": 1,
                "deployment_batch_size": 1,
                "input_length": 32,
                "max_new_tokens": 1,
                "warmup": 0,
                "repeat": 1,
                "input_tokens": 4,
                "generated_tokens": 1,
                "elapsed_seconds": 1.0,
                "peak_memory_allocated_gb": 0.1,
                "peak_memory_reserved_gb": 0.1,
                "deployment_memory_limit_gb": 1,
                "deployment_memory_measured_peak_gb": 0.1,
                "deployment_memory_limit_exceeded": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (job / "resource.json").write_text(
        json.dumps(
            {
                "seqlen": 32,
                "resource_backend": "model_load",
                "model_load_error": None,
                "prompt_source": "file",
                "prompt_path": "/tmp/prompts.txt",
                "prompt_count": 1,
                "deployment_batch_size": 1,
                "input_length": 32,
                "max_new_tokens": 1,
                "sparsity_sample_tensors": -1,
                "storage_bytes": 100,
                "storage_gb": 100 / (1024**3),
                "effective_storage_bytes_sparse_approx": 100,
                "effective_storage_gb_sparse_approx": 100 / (1024**3),
                "parameter_stats": {"total_params": 1, "sparsity": 0.0, "nonzero_ratio": 1.0},
                "compute_complexity": {
                    "flops": "1 FLOPS",
                    "macs": "0.5 MACs",
                    "flops_num": 1.0,
                    "backend": "analytical_config_estimate",
                    "effective_flops_num_sparse_approx": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (wanda_job / "summary.json").write_text(
        json.dumps(
            {
                "model_name": "TinyModel",
                "compression_method": "wanda",
                "compressed_model_path": "/tmp/tiny-wanda",
                "runtime": {
                    "python": "3.10.0",
                    "runtime_profile": "stable",
                    "cpu_count": 16,
                    "system_memory_total_gb": 256.0,
                    "hf_endpoint": "https://hf-mirror.com",
                    "hf_home": "/tmp/hf-cache",
                    "transformers_cache": "/tmp/hf-cache/transformers",
                    "huggingface_hub_cache": "/tmp/hf-cache/hub",
                    "hf_datasets_cache": "/tmp/datasets",
                    "torch": "2.0.0",
                    "transformers": "4.0.0",
                    "cuda_available": True,
                    "cuda_version": "12.1",
                    "cudnn_version": "8900",
                    "cuda_visible_devices": "0",
                    "gpus": [{"index": 0, "name": "NVIDIA Test GPU", "total_memory_gb": 24.0}],
                    "optional_imports": {"bitsandbytes": {"ok": True, "version": "0.48.2"}},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (wanda_job / "compression.json").write_text(
        json.dumps(
            {
                "manifest": {
                    "source_model_path": "/tmp/tiny-source",
                    "revision": None,
                    "backend": "built_in_activation_pruning",
                    "desc_act": False,
                    "awq_version": "GEMM",
                    "sparsity": 0.5,
                    "prune_lm_head": False,
                    "calibration_samples": 2,
                    "calibration_file": "/tmp/calib.txt",
                    "calibration_batch_size": 1,
                    "dtype": "fp16",
                    "seqlen": 32,
                    "materialized": True,
                    "note": "collect fixture wanda checkpoint",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (wanda_job / "accuracy.json").write_text(
        json.dumps(
            {
                "dtype": "fp16",
                "lm_eval_dtype": "float16",
                "batch_size": "auto",
                "num_fewshot": None,
                "limit": None,
                "device": "cuda:0",
                "data_dir": "/tmp/datasets",
                "hf_datasets_cache": "/tmp/datasets",
                "max_memory": "0:22GiB,cpu:120GiB",
                "attn_implementation": None,
                "split_accuracy_tasks": True,
                "continue_on_accuracy_error": True,
                "results": {
                    "mmlu": {"aggregate:acc,none": 0.2},
                    "arc_challenge": {"acc,none": 0.4, "acc_norm,none": 0.45},
                    "hellaswag": {"acc,none": 0.5, "acc_norm,none": 0.55},
                    "piqa": {"acc,none": 0.6, "acc_norm,none": 0.65},
                    "winogrande": {"acc,none": 0.7},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    wanda_accuracy_path = wanda_job / "accuracy.json"
    wanda_accuracy = json.loads(wanda_accuracy_path.read_text(encoding="utf-8"))
    for subtask in DEFAULT_MMLU_SUBTASKS:
        wanda_accuracy["results"].setdefault(subtask, {"acc,none": 0.2})
    wanda_accuracy_path.write_text(json.dumps(wanda_accuracy, indent=2) + "\n", encoding="utf-8")
    (wanda_job / "deployment.json").write_text(
        json.dumps(
            {
                "tokens_per_second": 1.0,
                "prompt_source": "file",
                "prompt_path": "/tmp/prompts.txt",
                "prompt_count": 1,
                "deployment_batch_size": 1,
                "input_length": 32,
                "max_new_tokens": 1,
                "warmup": 0,
                "repeat": 1,
                "input_tokens": 4,
                "generated_tokens": 1,
                "elapsed_seconds": 1.0,
                "peak_memory_allocated_gb": 0.1,
                "peak_memory_reserved_gb": 0.1,
                "deployment_memory_limit_gb": 1,
                "deployment_memory_measured_peak_gb": 0.1,
                "deployment_memory_limit_exceeded": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (wanda_job / "resource.json").write_text(
        json.dumps(
            {
                "seqlen": 32,
                "resource_backend": "model_load",
                "model_load_error": None,
                "prompt_source": "file",
                "prompt_path": "/tmp/prompts.txt",
                "prompt_count": 1,
                "deployment_batch_size": 1,
                "input_length": 32,
                "max_new_tokens": 1,
                "sparsity_sample_tensors": -1,
                "storage_bytes": 100,
                "storage_gb": 100 / (1024**3),
                "effective_storage_bytes_sparse_approx": 50,
                "effective_storage_gb_sparse_approx": 50 / (1024**3),
                "parameter_stats": {"total_params": 1, "sparsity": 0.5, "nonzero_ratio": 0.5},
                "compute_complexity": {
                    "flops": "1 FLOPS",
                    "macs": "0.5 MACs",
                    "flops_num": 1.0,
                    "backend": "analytical_config_estimate",
                    "effective_flops_num_sparse_approx": 0.5,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {"root": str(root)}


def _write_precompressed_matrix_fixture(root: Path) -> Dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "precompressed_tiny"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "tiny"}) + "\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_text("tiny\n", encoding="utf-8")
    config = {
        "models": [{"name": "TinyModel", "path": "/tmp/base-model"}],
        "methods": ["dense", "smoothquant", "omniquant"],
        "defaults": {
            "hf_endpoint": "https://hf-mirror.com",
            "hf_cache_dir": "/tmp/hf-cache",
            "data_dir": "/tmp/datasets",
            "calibration_file": "/tmp/calibration.txt",
            "prompt_file": "/tmp/prompts.txt",
            "output_root": str(root / "outputs"),
            "cuda_visible_devices": "0",
            "seqlen": 32,
            "calibration_samples": 1,
            "calibration_batch_size": 1,
            "sparsity": 0.5,
            "quant_bits": 4,
            "group_size": 128,
            "min_free_gpu_memory_gb": 0,
            "dtype": "fp16",
            "attn_implementation": "",
            "max_memory": "",
            "tasks": "piqa",
            "split_accuracy_tasks": False,
            "continue_on_accuracy_error": False,
            "batch_size": "1",
            "input_length": 32,
            "max_new_tokens": 1,
            "deployment_memory_limit_gb": 0,
            "warmup": 0,
            "repeat": 1,
        },
        "external_commands": {"omniquant": "python omni.py --model {model_path} --out {output_dir}"},
        "precompressed_paths": {"TinyModel": {"dense": str(checkpoint), "smoothquant": str(checkpoint)}},
        "method_overrides": {"dense": {"seqlen": 16, "batch_size": "2"}},
        "model_overrides": {"TinyModel": {"input_length": 16}},
        "model_method_overrides": {"TinyModel/dense": {"max_new_tokens": 2}},
    }
    config_path = root / "precompressed_matrix_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {"config": str(config_path), "output_root": str(root / "outputs")}


def _write_incomplete_existing_matrix_fixture(root: Path) -> Dict[str, str]:
    output_root = root / "outputs"
    job = output_root / "TinyModel" / "dense"
    job.mkdir(parents=True, exist_ok=True)
    (job / "summary.json").write_text(
        json.dumps(
            {
                "model_name": "TinyModel",
                "compression_method": "dense",
                "compressed_model_path": "/tmp/tiny",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "models": [{"name": "TinyModel", "path": "/tmp/tiny"}],
        "methods": ["dense"],
        "defaults": {
            "hf_endpoint": "https://hf-mirror.com",
            "hf_cache_dir": "/tmp/hf-cache",
            "data_dir": "/tmp/datasets",
            "calibration_file": "/tmp/calibration.txt",
            "prompt_file": "/tmp/prompts.txt",
            "output_root": str(output_root),
            "cuda_visible_devices": "0",
            "seqlen": 32,
            "calibration_samples": 1,
            "calibration_batch_size": 1,
            "sparsity": 0.5,
            "quant_bits": 4,
            "group_size": 128,
            "min_free_gpu_memory_gb": 0,
            "dtype": "fp16",
            "attn_implementation": "",
            "max_memory": "",
            "tasks": "piqa",
            "split_accuracy_tasks": False,
            "continue_on_accuracy_error": False,
            "batch_size": "1",
            "input_length": 32,
            "max_new_tokens": 1,
            "deployment_memory_limit_gb": 0,
            "warmup": 0,
            "repeat": 1,
        },
        "external_commands": {},
        "precompressed_paths": {},
        "method_overrides": {},
        "model_overrides": {},
        "model_method_overrides": {},
    }
    config_path = root / "incomplete_existing_matrix_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {"config": str(config_path), "output_root": str(output_root)}


def _run(name: str, cmd: List[str]) -> Dict:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_checks(args: argparse.Namespace) -> Dict:
    validator_fixture = _write_validator_fixture(Path("/tmp/llmcbench_validator_smoke"))
    collect_fixture = _write_collect_fixture(Path("/tmp/llmcbench_collect_smoke"))
    precompressed_fixture = _write_precompressed_matrix_fixture(Path("/tmp/llmcbench_precompressed_matrix_smoke"))
    incomplete_existing_fixture = _write_incomplete_existing_matrix_fixture(Path("/tmp/llmcbench_incomplete_existing_matrix_smoke"))
    checks = [
        ("compile", [sys.executable, "-m", "compileall", "edge_benchmark", "tools"]),
        ("cli_help", [sys.executable, "-m", "edge_benchmark", "--help"]),
        (
            "cli_defaults_to_hf_mirror",
            [
                sys.executable,
                "-c",
                "import os; os.environ.pop('HF_ENDPOINT', None); "
                "from edge_benchmark.cli import build_parser, _runtime_metadata; "
                "args=build_parser().parse_args(['deploy','--model-path','/tmp/model']); "
                "assert args.hf_endpoint == 'https://hf-mirror.com'; "
                "assert args.min_free_gpu_memory_gb == 20.0; "
                "assert args.deployment_memory_limit_gb == 24.0; "
                "pipe=build_parser().parse_args(['pipeline','--model-path','/tmp/model','--output-root','/tmp/out']); "
                "assert pipe.min_free_gpu_memory_gb == 20.0; "
                "assert pipe.deployment_memory_limit_gb == 24.0; "
                "help_text=build_parser().parse_args(['deploy','--model-path','/tmp/model']).func.__name__; "
                "assert 'run_deploy' == help_text; "
                "assert 'Defaults to {DEFAULT_CUDA_MAX_MEMORY} on CUDA hosts.' in __import__('pathlib').Path('edge_benchmark/cli.py').read_text(); "
                "assert _runtime_metadata()['hf_endpoint'] == 'https://hf-mirror.com'",
            ],
        ),
        (
            "cli_applies_cuda_default_max_memory",
            [
                sys.executable,
                "-c",
                "from argparse import Namespace; import edge_benchmark.cli as c; "
                "old=c.torch.cuda.is_available; c.torch.cuda.is_available=lambda: True; "
                "args=Namespace(max_memory=None); c._apply_cuda_default_max_memory(args); "
                "assert args.max_memory == '0:22GiB,cpu:120GiB'; "
                "args2=Namespace(max_memory='custom'); c._apply_cuda_default_max_memory(args2); "
                "assert args2.max_memory == 'custom'; "
                "c.torch.cuda.is_available=old",
            ],
        ),
        (
            "dtype_mapping",
            [
                sys.executable,
                "-c",
                "from edge_benchmark.cli import _lm_eval_dtype; "
                "assert _lm_eval_dtype('fp16') == 'float16'; "
                "assert _lm_eval_dtype('bf16') == 'bfloat16'; "
                "assert _lm_eval_dtype('fp32') == 'float32'",
            ],
        ),
        (
            "lm_eval_max_memory_format",
            [
                sys.executable,
                "-c",
                "from edge_benchmark.cli import _lm_eval_memory_args; "
                "assert _lm_eval_memory_args('0:22GiB,cpu:120GiB') == ['max_memory_per_gpu=22GiB','max_cpu_memory=120GiB']; "
                "assert _lm_eval_memory_args('22GiB') == ['max_memory_per_gpu=22GiB']",
            ],
        ),
        (
            "quant_backend_memory_kwargs",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('edge_benchmark/compression.py').read_text(); "
                "assert 'load_kwargs[\"max_memory\"] = _parse_max_memory(args.max_memory)' in text; "
                "assert 'load_kwargs[\"attn_implementation\"] = args.attn_implementation' in text; "
                "assert '\"backend\": \"auto_gptq\"' in text and '\"backend\": \"autoawq\"' in text; "
                "assert '\"desc_act\": args.desc_act' in text; "
                "assert '\"awq_version\": args.awq_version' in text; "
                "assert 'args.calibration_batch_size' in text; "
                "assert 'quantize_kwargs[\"calib_data\"] = calibration_data' in text; "
                "assert '_awq_calibration_data(args)' in text; "
                "assert '\"quantization_dataset\": \"mit-han-lab/pile-val-backup\"' in text; "
                "assert '\"max_memory\": args.max_memory' in text; "
                "assert '\"attn_implementation\": args.attn_implementation' in text",
            ],
        ),
        (
            "awq_strict_runtime_scale_grid_matches_paper_profile",
            [
                sys.executable,
                "-c",
                "from edge_benchmark.compression import _awq_runtime_scale_search_grid; "
                "assert _awq_runtime_scale_search_grid() == 20; "
                "from pathlib import Path; text=Path('edge_benchmark/compression.py').read_text(); "
                "assert 'Strict AWQ profile requires the installed AutoAWQ scale-search grid' in text; "
                "assert '\"awq_runtime_scale_search_grid\": runtime_scale_search_grid' in text",
            ],
        ),
        (
            "awq_strict_runtime_calibration_seed_matches_paper_profile",
            [
                sys.executable,
                "-c",
                "from argparse import Namespace; from pathlib import Path; from tempfile import TemporaryDirectory; "
                "from edge_benchmark.compression import _awq_runtime_calibration_seed,_run_awq; "
                "assert _awq_runtime_calibration_seed() == 42; "
                "args=Namespace(require_paper_protocol_attestation=True,awq_grid_size=20,awq_calibration_seed=41,model_path='/not/a/model',cache_dir=None,revision=None); "
                "exec(\"with TemporaryDirectory(prefix='llmcbench-awq-seed-') as root:\\n"
                "    try:\\n"
                "        _run_awq(args, Path(root) / 'out')\\n"
                "    except RuntimeError as exc:\\n"
                "        assert 'pileval shuffle seed' in str(exc)\\n"
                "    else:\\n"
                "        raise AssertionError('strict AWQ seed drift was not rejected')\"); "
                "from pathlib import Path; text=Path('edge_benchmark/compression.py').read_text(); "
                "assert 'Strict AWQ profile requires the installed AutoAWQ pileval shuffle seed' in text; "
                "assert '\"awq_runtime_calibration_seed\": runtime_calibration_seed' in text",
            ],
        ),
        (
            "adapted_quantization_contract",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('edge_benchmark/adapted_quantization.py').read_text(); "
                "assert 'real_int8_weights' in text and 'real_packed_uint4_weights' in text; "
                "assert '_collect_activation_maxima' in text; "
                "assert 'mit-han-lab/pile-val-backup' in text and 'wikitext-2-raw-v1' in text; "
                "assert 'not an upstream same-model recipe or a paper reproduction' in text; "
                "assert 'class _InputScaledLinear' in text; "
                "from edge_benchmark.checkpoint_inspection import inspect_quantization_storage; "
                "assert callable(inspect_quantization_storage)",
            ],
        ),
        (
            "standalone_compression_launcher_profiles",
            [
                sys.executable,
                "-c",
                "from compression.profiles import default_setting,method_arguments; "
                "from compression.run import build_parser,build_command; "
                "assert default_setting('wanda')=='paper' and default_setting('smoothquant')=='adapted'; "
                "assert '--target-model-adaptation' in method_arguments('omniquant','adapted'); "
                "exec(\"try:\\n"
                "    method_arguments('smoothquant','paper')\\n"
                "except ValueError:\\n"
                "    pass\\n"
                "else:\\n"
                "    raise AssertionError('unsupported external paper profile accepted')\"); "
                "args=build_parser().parse_args(['--base-model','/tmp/Qwen3-8B','--method','awq','--dry-run']); "
                "cmd=build_command(args); assert cmd[2:4]==['edge_benchmark','compress']; "
                "assert '--awq-grid-size' in cmd and '20' in cmd",
            ],
        ),
        (
            "paper_protocol_validator_rejects_under_calibration",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench-paper-protocol-')); "
                "good={'model_name':'LLaMA3-8B','compression_method':'gptq',"
                "'compression_paper_protocol':'gptq_for_llama_c4_w4_asymmetric_actorder_true_sequential',"
                "'compression_paper_protocol_attested':True,"
                "'compression_base_model_repository':'meta-llama/Meta-Llama-3-8B',"
                "'compression_base_model_revision':'8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920',"
                "'compression_base_model_attested':True,'compression_quant_bits':4,'compression_group_size':-1,"
                "'compression_calibration_source':'c4','compression_calibration_seed':0,"
                "'compression_calibration_samples':128,'compression_calibration_token_length':2048,"
                "'compression_calibration_dataset':'allenai/c4','compression_calibration_dataset_config':'allenai--c4',"
                "'compression_calibration_data_file':'en/c4-train.00000-of-01024.json.gz',"
                "'compression_calibration_dataset_revision':'1588ec454efa1a09f29cd18ddd04fe05fc8653a2',"
                "'compression_desc_act':True,'compression_gptq_true_sequential':True,"
                "'compression_gptq_symmetric':False,'compression_gptq_damp_percent':0.01,"
                "'compression_gptq_static_groups':False}; "
                "good_path=root/'good.json'; bad_path=root/'bad.json'; "
                "good_path.write_text(json.dumps([good])); good['compression_calibration_samples']=5; "
                "bad_path.write_text(json.dumps([good])); "
                "base=[sys.executable,'tools/validate_paper_protocol.py','--config','configs/edge_paper_aligned_matrix.json',"
                "'--only-model','LLaMA3-8B','--only-method','gptq']; "
                "ok=subprocess.run(base+['--results',str(good_path)],capture_output=True,text=True); "
                "bad=subprocess.run(base+['--results',str(bad_path)],capture_output=True,text=True); "
                "assert ok.returncode == 0, ok.stderr + ok.stdout; "
                "assert bad.returncode != 0 and 'compression_calibration_samples' in bad.stdout",
            ],
        ),
        (
            "paper_protocol_validator_rejects_gptq_config_drift_before_execution",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                "cfg=json.load(open('configs/edge_paper_aligned_matrix.json')); "
                "cfg['defaults']['calibration_samples']=5; "
                "path=Path(tempfile.mkdtemp(prefix='llmcbench-paper-config-'))/'config.json'; "
                "path.write_text(json.dumps(cfg)); "
                "cmd=[sys.executable,'tools/validate_paper_protocol.py','--config',str(path),"
                "'--only-model','Qwen3-8B','--only-method','gptq']; "
                "p=subprocess.run(cmd,capture_output=True,text=True); "
                "assert p.returncode != 0 and 'calibration_samples' in p.stdout; "
                "cfg['defaults']['calibration_samples']=128; cfg['method_overrides']['gptq']['gptq_static_groups']=True; "
                "path.write_text(json.dumps(cfg)); p=subprocess.run(cmd,capture_output=True,text=True); "
                "assert p.returncode != 0 and 'gptq_static_groups' in p.stdout",
            ],
        ),
        (
            "gptq_strict_profile_uses_official_asymmetric_default",
            [
                sys.executable,
                "-c",
                "import json; "
                "cfg=json.load(open('configs/edge_paper_aligned_matrix.json')); "
                "override=cfg['method_overrides']['gptq']; "
                "assert override['gptq_asymmetric'] is True; "
                "assert override['paper_protocol']=='gptq_for_llama_c4_w4_asymmetric_actorder_true_sequential'; "
                "protocol=json.load(open('configs/paper_protocols.json'))['protocols'][override['paper_protocol']]; "
                "assert protocol['compression']['symmetric'] is False",
            ],
        ),
        (
            "omniquant_strict_profile_uses_official_seed_two",
            [
                sys.executable,
                "-c",
                "import json; "
                "cfg=json.load(open('configs/edge_paper_aligned_matrix.json')); "
                "override=cfg['method_overrides']['omniquant']; "
                "assert override['omniquant_seed']==2; "
                "protocol=json.load(open('configs/paper_protocols.json'))['protocols'][override['paper_protocol']]; "
                "assert protocol['calibration']['seed']==2; "
                "assert protocol['compression']['lwc'] is True and protocol['compression']['let'] is False",
            ],
        ),
        (
            "paper_protocol_validator_blocks_missing_smoothquant_target_recipe",
            [
                sys.executable,
                "-c",
                "import subprocess,sys; "
                "cmd=[sys.executable,'tools/validate_paper_protocol.py','--config','configs/edge_paper_aligned_matrix.json',"
                "'--only-model','Qwen3-8B','--only-method','smoothquant']; "
                "p=subprocess.run(cmd,capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "assert 'no Qwen3-8B target-model recipe/export' in p.stdout; "
                "p=subprocess.run([sys.executable,'tools/run_edge_matrix.py','--config','configs/edge_paper_aligned_matrix.json',"
                "'--only-model','Qwen3-8B','--only-method','smoothquant','--dry-run'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "assert 'Paper/official-code protocol preflight failed' in p.stderr",
            ],
        ),
        (
            "paper_protocol_runner_blocks_missing_smoothquant_target_recipe",
            [
                sys.executable,
                "-c",
                "import subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/run_edge_matrix.py','--config','configs/edge_paper_aligned_matrix.json',"
                "'--only-model','LLaMA3-8B','--only-method','smoothquant','--dry-run'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "assert 'no LLaMA3-8B target-model recipe/export' in p.stdout; "
                "assert 'Paper/official-code protocol preflight failed' in p.stderr",
            ],
        ),
        (
            "paper_protocol_runner_allow_skipped_external_cannot_bypass_missing_recipe",
            [
                sys.executable,
                "-c",
                "import subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/run_edge_matrix.py','--config','configs/edge_paper_aligned_matrix.json',"
                "'--only-model','Qwen3-8B','--only-method','smoothquant','--dry-run','--allow-skipped-external'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "assert 'no Qwen3-8B target-model recipe/export' in p.stdout; "
                "assert 'Paper/official-code protocol preflight failed' in p.stderr",
            ],
        ),
        (
            "paper_protocol_validator_blocks_missing_omniquant_target_recipe",
            [
                sys.executable,
                "-c",
                "import subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_paper_protocol.py','--config','configs/edge_paper_aligned_matrix.json',"
                "'--only-model','Qwen3-8B','--only-method','omniquant'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "assert 'no Qwen3-8B target-model script or export recipe' in p.stdout",
            ],
        ),
        (
            "strict_external_command_requires_direct_official_script",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench-official-external-')); "
                "model=root/'model'; official=root/'official'; model.mkdir(); official.mkdir(); (model/'config.json').write_text('{}'); "
                "subprocess.run(['git','-C',str(official),'init','-q'],check=True); "
                "subprocess.run(['git','-C',str(official),'config','user.email','smoke@example.com'],check=True); "
                "subprocess.run(['git','-C',str(official),'config','user.name','smoke'],check=True); "
                "script=official/'export.py'; script.write_text(\"import json,sys\\nfrom pathlib import Path\\nout=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True); (out/'weights.bin').write_bytes(b'x'); (out/'llmcbench_external_protocol.json').write_text(json.dumps({'paper_protocol':'smoothquant_w8a8_pile_validation_512_sentences_alpha_model_specific','paper_protocol_attested':True,'quant_bits':8,'smoothquant_alpha':0.85,'smoothquant_alpha_selection':'official_recipe','smoothquant_calibration_source':'pile_validation','smoothquant_calibration_sentences':512,'official_repository':'https://github.com/mit-han-lab/smoothquant','official_commit':sys.argv[2]}))\\n\"); "
                "subprocess.run(['git','-C',str(official),'add','export.py'],check=True); subprocess.run(['git','-C',str(official),'commit','-qm','init'],check=True); "
                "subprocess.run(['git','-C',str(official),'remote','add','origin','https://github.com/mit-han-lab/smoothquant.git'],check=True); "
                "commit=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip(); "
                "base=[sys.executable,'-m','edge_benchmark','compress','--model-path',str(model),'--model-name','unknown','--compression-method','smoothquant','--min-free-gpu-memory-gb','0','--quant-bits','8','--paper-protocol','smoothquant_w8a8_pile_validation_512_sentences_alpha_model_specific','--require-paper-protocol-attestation','--smoothquant-alpha','0.85','--smoothquant-alpha-selection','official_recipe','--smoothquant-official-code-dir',str(official),'--smoothquant-official-repository','https://github.com/mit-han-lab/smoothquant','--smoothquant-official-commit',commit]; "
                "good=subprocess.run(base+['--compressed-output-dir',str(root/'good'),'--external-command',f'{sys.executable} {script} {{output_dir}} {{smoothquant_official_commit}}'],capture_output=True,text=True); "
                "assert good.returncode==0, good.stderr+good.stdout; "
                "bad=subprocess.run(base+['--compressed-output-dir',str(root/'bad'),'--external-command',f'{sys.executable} {root}/not_official.py {{output_dir}}'],capture_output=True,text=True); "
                "assert bad.returncode!=0 and 'must directly execute a Python or executable file inside' in bad.stderr",
            ],
        ),
        (
            "official_source_setup_dry_run",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench-source-setup-')); out=root/'sources.json'; "
                "p=subprocess.run([sys.executable,'tools/setup_official_compression_sources.py','--root',str(root/'sources'),'--dry-run','--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode==0, p.stderr+p.stdout; report=json.loads(out.read_text()); "
                "assert report['ok'] and {row['method'] for row in report['rows']}=={'smoothquant','omniquant'}; "
                "assert all(row['dry_run'] for row in report['rows'])",
            ],
        ),
        (
            "strict_base_model_attestation_gate",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench-base-attest-')); model=root/'model'; model.mkdir(); "
                "(model/'config.json').write_text('{}'); (model/'model.safetensors.index.json').write_text('{}'); "
                "import hashlib; "
                "sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); "
                "(model/'.llmcbench_base_model_attestation.json').write_text(json.dumps({'repository':'meta-llama/Meta-Llama-3-8B','revision':'8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920','files_sha256':{'config.json':sha(model/'config.json'),'model.safetensors.index.json':sha(model/'model.safetensors.index.json')}})); "
                "cfg={'models':[{'name':'LLaMA3-8B','path':str(model),'repository':'meta-llama/Meta-Llama-3-8B','revision':'8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920','attestation_file':'.llmcbench_base_model_attestation.json'}],'methods':['dense'],'defaults':{'paper_protocol':'dense_fp16_same_checkpoint_baseline','require_paper_protocol_attestation':True},'method_overrides':{'dense':{'paper_protocol':'dense_fp16_same_checkpoint_baseline','require_paper_protocol_attestation':True}}}; "
                "cfg_path=root/'config.json'; proto_path=root/'protocols.json'; cfg_path.write_text(json.dumps(cfg)); proto_path.write_text(json.dumps({'protocols':{'dense_fp16_same_checkpoint_baseline':{'method':'dense'}}})); "
                "p=subprocess.run([sys.executable,'tools/validate_paper_protocol.py','--config',str(cfg_path),'--protocols',str(proto_path),'--require-base-models'],capture_output=True,text=True); "
                "assert p.returncode==0, p.stderr+p.stdout; "
                "(model/'config.json').write_text('{\"changed\":true}'); "
                "p=subprocess.run([sys.executable,'tools/validate_paper_protocol.py','--config',str(cfg_path),'--protocols',str(proto_path),'--require-base-models'],capture_output=True,text=True); "
                "assert p.returncode!=0 and 'no longer match the attestation SHA256 values' in p.stdout; "
                "(model/'config.json').write_text('{}'); "
                "(model/'.llmcbench_source_override.json').write_text('{}'); "
                "p=subprocess.run([sys.executable,'tools/validate_paper_protocol.py','--config',str(cfg_path),'--protocols',str(proto_path),'--require-base-models'],capture_output=True,text=True); "
                "assert p.returncode!=0 and 'source override marker' in p.stdout",
            ],
        ),
        (
            "strict_preflight_records_paper_protocol",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench-strict-preflight-')); model=root/'model'; model.mkdir(); "
                "(model/'config.json').write_text('{}'); (model/'model.safetensors.index.json').write_text('{}'); "
                "import hashlib; sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); "
                "(model/'.llmcbench_base_model_attestation.json').write_text(json.dumps({'repository':'Qwen/Qwen3-8B','revision':'b968826d9c46dd6066d109eabc6255188de91218','files_sha256':{'config.json':sha(model/'config.json'),'model.safetensors.index.json':sha(model/'model.safetensors.index.json')}})); "
                "cfg={'models':[{'name':'Qwen3-8B','path':str(model),'repository':'Qwen/Qwen3-8B','revision':'b968826d9c46dd6066d109eabc6255188de91218','attestation_file':'.llmcbench_base_model_attestation.json'}],'methods':['dense'],'require_paper_protocol_validation':True,'defaults':{'paper_protocol':'dense_fp16_same_checkpoint_baseline','require_paper_protocol_attestation':True,'hf_endpoint':'https://hf-mirror.com','hf_cache_dir':str(root/'cache'),'data_dir':str(root/'data'),'calibration_file':str(root/'calib.txt'),'prompt_file':str(root/'prompts.txt'),'output_root':str(root/'out'),'cuda_visible_devices':'0','tasks':'piqa','dtype':'fp16','batch_size':'1','seqlen':32,'input_length':32,'max_new_tokens':1,'deployment_batch_size':1,'warmup':0,'repeat':1,'sparsity':0.5,'prune_pattern':'unstructured','prune_lm_head':False,'quant_bits':4,'group_size':128,'desc_act':False,'awq_version':'GEMM','calibration_samples':2,'calibration_batch_size':1,'sparsity_sample_tensors':-1,'deployment_memory_limit_gb':1,'min_free_gpu_memory_gb':0,'device_map':'auto','max_memory':'','split_accuracy_tasks':True,'continue_on_accuracy_error':True,'pruning_calibration_source':'text_file','calibration_seed':0,'calibration_dataset':'allenai/c4','calibration_dataset_config':'allenai--c4','calibration_c4_data_file':'en/c4-train.00000-of-01024.json.gz','calibration_c4_revision':'x','calibration_dataset_cache':str(root/'data'),'sparsegpt_blocksize':128,'sparsegpt_percdamp':0.01,'pruning_buffer_device':'cpu','gptq_calibration_source':'c4','gptq_damp_percent':0.01,'gptq_true_sequential':True,'gptq_static_groups':False,'gptq_asymmetric':False,'awq_calibration_source':'pileval','awq_calibration_samples':128,'awq_calibration_seqlen':512,'awq_grid_size':20,'smoothquant_alpha':None,'smoothquant_alpha_selection':'official_grid_search','smoothquant_calibration_source':'pile_validation','smoothquant_calibration_sentences':512,'omniquant_abits':16,'omniquant_epochs':20,'omniquant_nsamples':128,'omniquant_seqlen':2048,'omniquant_calibration_source':'wikitext2','omniquant_lwc':True,'omniquant_let':False,'omniquant_aug_loss':False},'method_overrides':{'dense':{'paper_protocol':'dense_fp16_same_checkpoint_baseline','require_paper_protocol_attestation':True}}}; "
                "cfg_path=root/'config.json'; cfg_path.write_text(json.dumps(cfg)); (root/'calib.txt').write_text('x\\ny\\n'); (root/'prompts.txt').write_text('prompt\\n'); "
                "p=subprocess.run([sys.executable,'tools/run_edge_preflight.py','--config',str(cfg_path),'--only-model','Qwen3-8B','--only-method','dense','--allow-blocked','--skip-prewarm-dry-run','--output-root',str(root/'matrix'),'--output',str(root/'preflight.json')],capture_output=True,text=True); "
                "assert p.returncode==0, p.stderr+p.stdout; report=json.loads((root/'preflight.json').read_text()); steps={x['name']:x for x in report['steps']}; assert steps['paper_protocol']['ok'] and steps['matrix_dry_run']['ok']",
            ],
        ),
        (
            "wanda_unstructured_mask_matches_official_stable_sort",
            [
                sys.executable,
                "-c",
                "import torch; from edge_benchmark.compression import _apply_unstructured_prune; "
                "weight=torch.ones((1,4)); score=torch.tensor([[1.,1.,1.,2.]]); "
                "_apply_unstructured_prune(weight,score,.5); "
                "assert torch.equal(weight,torch.tensor([[0.,0.,1.,1.]])); "
                "assert torch.count_nonzero(weight).item() == 2",
            ],
        ),
        (
            "lm_eval_memory_kwargs",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "accuracy=Path('edge_benchmark/cli.py').read_text(); "
                "prewarm=Path('tools/prewarm_lm_eval_tasks.py').read_text(); "
                "assert 'model_args.extend(_lm_eval_memory_args(args.max_memory))' in accuracy; "
                "assert 'model_args.extend(_lm_eval_memory_args(args.max_memory))' in prewarm",
            ],
        ),
        (
            "precompressed_manifest_params",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('edge_benchmark/cli.py').read_text(); "
                "assert '\"backend\": \"precompressed_checkpoint\"' in text; "
                "assert '\"quant_bits\": args.quant_bits' in text; "
                "assert '\"group_size\": args.group_size' in text; "
                "assert '\"desc_act\": args.desc_act' in text; "
                "assert '\"awq_version\": args.awq_version' in text; "
                "assert '\"prune_lm_head\": args.prune_lm_head' in text; "
                "assert '\"sparsity\": args.sparsity' in text; "
                "assert '\"prune_pattern\": args.prune_pattern' in text; "
                "assert '\"prune_structure\"' in text; "
                "assert '\"calibration_samples\": args.calibration_samples' in text; "
                "assert '\"dtype\": args.dtype' in text; "
                "assert '\"max_memory\": args.max_memory' in text; "
                "assert '\"attn_implementation\": args.attn_implementation' in text; "
                "assert 'calibration_batch_size' in text",
            ],
        ),
        (
            "external_manifest_params",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('edge_benchmark/compression.py').read_text(); "
                "assert '\"backend\": \"external_command\"' in text; "
                "assert '\"quant_bits\": args.quant_bits' in text; "
                "assert '\"group_size\": args.group_size' in text; "
                "assert '\"sparsity\": args.sparsity' in text; "
                "assert '\"prune_pattern\": args.prune_pattern' in text; "
                "assert '\"prune_structure\"' in text; "
                "assert '\"calibration_samples\": args.calibration_samples' in text; "
                "assert '\"max_memory\": args.max_memory' in text; "
                "assert '\"attn_implementation\": args.attn_implementation' in text; "
                "assert '\"external_command\": rendered' in text; "
                "assert '\"external_returncode\": proc.returncode' in text; "
                "assert '\"stderr_tail\": proc.stderr[-4000:]' in text; "
                "assert 'calibration_batch_size' in text",
            ],
        ),
        (
            "external_readiness_templates",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('tools/check_method_readiness.py').read_text(); "
                "assert '{calibration_samples}' in text; "
                "assert '{calibration_batch_size}' in text; "
                "assert '{max_memory}' in text; "
                "assert '{attn_implementation}' in text",
            ],
        ),
        (
            "structured_pruning_support",
            [
                sys.executable,
                "-c",
                "import json; from pathlib import Path; "
                "cli=Path('edge_benchmark/cli.py').read_text(); comp=Path('edge_benchmark/compression.py').read_text(); "
                "structured=json.load(open('configs/edge_4090_structured_pruning.json')); "
                "assert '--prune-pattern' in cli; "
                "assert '--sparsegpt-blocksize' in cli and '--pruning-buffer-device' in cli; "
                "assert 'SUPPORTED_METHODS = (' in cli; "
                "assert '_apply_nm_prune' in comp and 'prune_n, prune_m = (2, 4)' in comp; "
                "assert 'sparsegpt_official_second_order' in comp and 'torch.linalg.cholesky' in comp; "
                "assert structured['methods']==['wanda','sparsegpt']; "
                "assert {m['name'] for m in structured['models']}=={'LLaMA3-8B','Qwen3-8B'}; "
                "assert structured['defaults']['calibration_samples'] == 128; "
                "assert structured['defaults']['pruning_calibration_source'] == 'c4'; "
                "assert structured['defaults']['calibration_seed'] == 0; "
                "assert structured['defaults']['calibration_dataset'] == 'allenai/c4'; "
                "assert structured['defaults']['calibration_dataset_config'] == 'allenai--c4'; "
                "assert structured['defaults']['calibration_c4_data_file'] == 'en/c4-train.00000-of-01024.json.gz'; "
                "assert structured['defaults']['calibration_c4_revision'] == '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'; "
                "assert structured['defaults']['pruning_buffer_device'] == 'cpu'; "
                "assert all(v['prune_pattern']=='2:4' and v['sparsity']==0.5 for v in structured['method_overrides'].values())",
            ],
        ),
        (
            "quantization_storage_semantics",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "inspect=Path('edge_benchmark/checkpoint_inspection.py').read_text(); "
                "collector=Path('tools/collect_edge_results.py').read_text(); "
                "validator=Path('tools/validate_edge_results.py').read_text(); "
                "assert 'packed_integer' in inspect and 'integer_weight' in inspect and 'floating_weight_or_fake_quant' in inspect; "
                "assert '_quantization_metadata' in inspect and 'weight_bits' in inspect and 'activation_bits' in inspect; "
                "assert 'quantization_storage_real' in collector and 'quantization_storage_format' in collector; "
                "assert 'checkpoint_weight_bits' in collector and 'checkpoint_group_size' in collector; "
                "assert 'floating-weight or fake-quant checkpoint' in validator",
            ],
        ),
        ("compress_help", [sys.executable, "-m", "edge_benchmark", "compress", "--help"]),
        ("pipeline_help", [sys.executable, "-m", "edge_benchmark", "pipeline", "--help"]),
        (
            "deployment_memory_gate_status",
            [
                sys.executable,
                "-c",
                "from edge_benchmark.cli import _deployment_memory_status; "
                "ok=_deployment_memory_status({'peak_memory_allocated_gb': 21.0, 'peak_memory_reserved_gb': 23.0}, 24); "
                "assert ok['deployment_memory_measured_peak_gb'] == 23.0; "
                "assert ok['deployment_memory_limit_exceeded'] is False; "
                "bad=_deployment_memory_status({'peak_memory_allocated_gb': 21.0, 'peak_memory_reserved_gb': 25.0}, 24); "
                "assert bad['deployment_memory_measured_peak_gb'] == 25.0; "
                "assert bad['deployment_memory_limit_exceeded'] is True; "
                "off=_deployment_memory_status({'peak_memory_allocated_gb': 30.0}, 0); "
                "assert off['deployment_memory_limit_gb'] is None and off['deployment_memory_limit_exceeded'] is False",
            ],
        ),
        (
            "resource_help_quant_args",
            [
                sys.executable,
                "-c",
                "import subprocess,sys; "
                "p=subprocess.run([sys.executable,'-m','edge_benchmark','resource','--help'],capture_output=True,text=True); "
                "assert p.returncode == 0; "
                "assert '--quant-bits' in p.stdout and '--group-size' in p.stdout",
            ],
        ),
        (
            "resource_config_only_fallback_code",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('edge_benchmark/cli.py').read_text(); "
                "assert 'config_only_fallback' in text; "
                "assert 'model_load_error' in text; "
                "assert '_config_parameter_stats' in text; "
                "assert '_analytical_flops_from_config' in text; "
                "assert 'AutoConfig.from_pretrained' in text",
            ],
        ),
        (
            "prepare_precompressed_assets_static",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('tools/prepare_precompressed_assets.py').read_text(); "
                "assert 'Ksgk-fy/Meta-Llama-3-8B-4bit-GPTQ' in text; "
                "assert 'solidrust/Meta-Llama-3-8B-hf-AWQ' in text; "
                "assert 'HRuiii/Meta-Llama-3-8B-smoothquant-w8a8' in text; "
                "assert 'Qwen/Qwen3-8B-AWQ' in text; "
                "assert 'kaitchup/Qwen3-8B-autoround-4bit-gptq' in text; "
                "assert 'Lotaliz/Qwen3-8B-smoothquant-w8a8' in text; "
                "assert 'yw223/Qwen3-8B-OmniQuant-4bit' in text; "
                "assert 'MISSING_DEFAULT_REPOS' in text; "
                "assert 'no_verified_public_default_repo' in text; "
                "assert 'precompressed_paths' in text; "
                "assert 'blockers' in text; "
                "assert 'blocking' in text; "
                "assert '--metadata-only' in text; "
                "assert '--fail-on-error' in text; "
                "assert '_weight_status' in text",
            ],
        ),
        (
            "target_readiness_static",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('tools/audit_target_readiness.py').read_text(); "
                "assert 'missing_before_matrix' in text; "
                "assert 'ready_for_matrix' in text; "
                "assert 'before_matrix_blockers' in text; "
                "assert 'require_non_rotational_storage' in text; "
                "assert 'asset_storage' in text; "
                "assert 'require_handoff_manifest' in text; "
                "assert 'edge_handoff_manifest.json' in text; "
                "assert 'tools/build_edge_handoff_manifest.py' in text; "
                "assert 'require_optional_backends_report' in text; "
                "assert 'require_quant_runtime_report' in text; "
                "assert 'next_commands' in text; "
                "assert '--require-runtime-cuda-prefix 12.' in text; "
                "assert '--require-runtime-gpu-total-memory-min-gb 23' in text; "
                "assert '--require-runtime-gpu-count 1' in text; "
                "assert '--require-runtime-cudnn' in text; "
                "assert '--require-runtime-torch-prefix 2.' in text; "
                "assert '--require-runtime-transformers' in text; "
                "assert '--require-runtime-cpu-count-min 8' in text; "
                "assert '--require-runtime-system-memory-min-gb 240' in text; "
                "assert '--require-runtime-hf-endpoint https://hf-mirror.com' in text; "
                "assert 'edge_quant_runtime_report.json' in text; "
                "assert 'completion_audit.json' in text",
            ],
        ),
        (
            "target_readiness_method_blocker_details",
            [
                sys.executable,
                "-c",
                "from tools.audit_target_readiness import _attach_optional_runtime_details, _method_readiness_blockers; "
                "readiness={'ok':True,'payload':{'ok':False,'rows':[{'model':'Qwen3-8B','method':'gptq','ok':False,'uses_precompressed':True,'precompressed_path':'/tmp/qwen-gptq','runtime_requirements_reason':'gptqmodel_checkpoint_metadata','quantization_metadata':{'quantizer':['gptqmodel:4.0.0']},'blockers':['missing_runtime_import:any_of:gptqmodel']}]} }; "
                "blockers=_method_readiness_blockers(readiness); "
                "assert blockers[0]['runtime_requirements_reason'] == 'gptqmodel_checkpoint_metadata'; "
                "assert blockers[0]['quantization_metadata']['quantizer'] == ['gptqmodel:4.0.0']; "
                "blockers=_attach_optional_runtime_details(blockers, {'gptqmodel_install': {'requirements_ok': False, 'requirement_status': {'torch': {'ok': False}}}}); "
                "assert blockers[0]['gptqmodel_requirement_status']['torch']['ok'] is False; "
                "assert 'gptqmodel_requirements_not_satisfied' in blockers[0]['blockers']",
            ],
        ),
        (
            "handoff_manifest_static",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('tools/build_edge_handoff_manifest.py').read_text(); "
                "assert 'TARGET_RUNTIME_GATES' in text; "
                "assert 'single NVIDIA RTX 4090 24GB' in text; "
                "assert 'edge_optional_backends_report.json with ok=true' in text; "
                "assert 'edge_quant_runtime_report.json with ok=true' in text; "
                "assert 'edge_optional_backends_report.json when optional backend resolution is enabled' not in text; "
                "assert 'edge_quant_runtime_report.json when quant runtime probing is enabled' not in text; "
                "assert 'completion_audit.json with ok=true' in text; "
                "assert 'tools/plan_edge_matrix_resume.py' in text; "
                "assert 'tools/export_edge_results_bundle.py' in text; "
                "assert 'CHECK_OPTIONAL_BACKEND_RESOLUTION=1 CHECK_QUANT_RUNTIME=1' in text; "
                "assert 'git@code.byted.org:data/llmcbench.git' in text",
            ],
        ),
        (
            "handoff_manifest_build",
            [
                sys.executable,
                "tools/build_edge_handoff_manifest.py",
                "--config",
                "configs/edge_4090_matrix.json",
                "--output-root",
                "/tmp/llmcbench_handoff_output",
                "--output",
                "/tmp/llmcbench_edge_handoff_manifest.json",
            ],
        ),
        (
            "handoff_manifest_assert",
            [
                sys.executable,
                "-c",
                "import json; r=json.load(open('/tmp/llmcbench_edge_handoff_manifest.json')); "
                "assert r['schema_version'] == 1; "
                "assert r['config']['methods'] == ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']; "
                "assert {m['name'] for m in r['config']['models']} == {'LLaMA3-8B','Qwen3-8B'}; "
                "assert r['target_environment']['software']['hf_endpoint'] == 'https://hf-mirror.com'; "
                "assert r['target_environment']['software']['gpu_count'] == 1; "
                "assert r['target_environment']['memory_controls']['deployment_memory_limit_gb'] == 24; "
                "assert r['target_environment']['memory_controls']['deployment_batch_size'] == 1; "
                "assert r['paths']['output_root'] == '/tmp/llmcbench_handoff_output'; "
                "assert r['completion_gates']['coverage']['required_mmlu_subtask_count'] == 57; "
                "assert 'mmlu_abstract_algebra' in r['completion_gates']['coverage']['required_mmlu_subtasks']; "
                "assert 'edge_target_paths_report.json with ok=true' in r['completion_gates']['reports']; "
                "assert 'edge_optional_backends_report.json with ok=true' in r['completion_gates']['reports']; "
                "assert 'edge_quant_runtime_report.json with ok=true' in r['completion_gates']['reports']; "
                "assert any('target_readiness.json' in item and 'ok=true' in item for item in r['completion_gates']['reports']); "
                "assert 'completion_audit.json with ok=true and validator.ok=true' in r['completion_gates']['reports']; "
                "assert 'edge_results_bundle_verify.json with ok=true, complete=true, validator.ok=true and target completion_ok=true' in r['completion_gates']['reports']; "
                "assert 'edge_delivery_verify.json with ok=true and all substatus checks passing' in r['completion_gates']['reports']; "
                "assert 'completion_audit.json' in r['required_artifacts']; "
                "assert 'edge_handoff_manifest.json' in r['required_artifacts']; "
                "assert 'edge_target_paths_report.json' in r['required_artifacts']; "
                "assert 'edge_optional_backends_report.json' in r['required_artifacts']; "
                "assert 'edge_quant_runtime_report.json' in r['required_artifacts']; "
                "assert 'target_readiness.json' in r['required_artifacts']; "
                "assert 'edge_scorecard.json' in r['required_artifacts']; "
                "assert 'edge_scorecard.md' in r['required_artifacts']; "
                "assert 'edge_results_bundle.tar.gz' in r['required_artifacts']; "
                "assert 'edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json' in r['required_artifacts']; "
                "assert 'edge_results_bundle_verify.json' in r['required_artifacts']; "
                "assert 'edge_delivery_verify.json' in r['required_artifacts']; "
                "assert 'edge_target_paths_report.json' not in r['optional_artifacts']; "
                "assert 'edge_optional_backends_report.json' not in r['optional_artifacts']; "
                "assert 'edge_quant_runtime_report.json' not in r['optional_artifacts']; "
                "assert 'tools/prepare_target_paths.py' in r['commands']['prepare'][0]; "
                "assert 'edge_target_paths_report.json' in r['commands']['prepare'][0]; "
                "assert '--fail-on-error' in r['commands']['prepare'][0]; "
                "assert 'RUN_PRECOMPRESSED_PREPARE=1' in r['commands']['prepare_precompressed'][0]; "
                "assert 'PRECOMPRESSED_PREPARE_ONLY=1' in r['commands']['prepare_precompressed'][0]; "
                "assert 'PRECOMPRESSED_GROUPS=' in r['commands']['prepare_precompressed'][0]; "
                "assert 'scripts/run_edge_4090_full.sh' in r['commands']['prepare_precompressed'][0]; "
                "assert 'edge_prepare_precompressed_report.json' in r['commands']['prepare_precompressed'][0]; "
                "assert 'edge_matrix_with_precompressed.json' in r['commands']['prepare_precompressed'][0]; "
                "assert '--check-optional-backend-resolution' in r['commands']['preflight'][0]; "
                "assert '--check-quant-runtime' in r['commands']['preflight'][0]; "
                "assert 'tools/prewarm_lm_eval_tasks.py' in r['commands']['prewarm'][0]; "
                "assert '--expand-mmlu' in r['commands']['prewarm'][0]; "
                "assert 'edge_prewarm_report.json' in r['commands']['prewarm'][0]; "
                "assert '--require-non-rotational-storage' in r['commands']['readiness'][0]; "
                "assert '--require-handoff-manifest' in r['commands']['readiness'][0]; "
                "assert '--require-optional-backends-report' in r['commands']['readiness'][0]; "
                "assert '--require-quant-runtime-report' in r['commands']['readiness'][0]; "
                "followup='\\n'.join(r['commands']['precompressed_config_followup']); "
                "assert 'edge_matrix_with_precompressed.json' in followup; "
                "assert 'tools/run_edge_matrix.py' in followup; "
                "assert 'tools/audit_edge_completion.py' in followup; "
                "assert '--require-prepare-precompressed-report' in followup; "
                "post_readiness=[cmd for cmd in r['commands']['post_run_audit'] if 'tools/audit_target_readiness.py' in cmd][0]; "
                "post_goal=[cmd for cmd in r['commands']['post_run_audit'] if 'tools/build_goal_audit.py' in cmd][0]; "
                "assert r['commands']['post_run_audit'].index(post_readiness) < r['commands']['post_run_audit'].index(post_goal); "
                "assert '--require-non-rotational-storage' in post_readiness; "
                "assert '--require-handoff-manifest' in post_readiness; "
                "assert '--require-optional-backends-report' in post_readiness; "
                "assert '--require-quant-runtime-report' in post_readiness; "
                "assert '--final-output-root /tmp/llmcbench_handoff_output' in post_goal; "
                "assert '--portable-bundle-verify portable_bundle_verify.json' in post_goal; "
                "assert '--output /tmp/llmcbench_handoff_output/current_goal_audit.json' in post_goal; "
                "post_baseline='\\n'.join(r['commands']['baseline_post_run_audit']); "
                "assert 'tools/audit_target_readiness.py' in post_baseline; "
                "assert 'edge_matrix_with_precompressed.json' not in post_baseline; "
                "assert 'tools/build_goal_audit.py' in post_baseline; "
                "assert '--final-output-root /tmp/llmcbench_handoff_output' in post_baseline; "
                "assert '--portable-bundle-verify portable_bundle_verify.json' in post_baseline; "
                "assert '--output /tmp/llmcbench_handoff_output/current_goal_audit.json' in post_baseline; "
                "assert '--require-optional-backends-report' in post_baseline; "
                "assert '--require-quant-runtime-report' in post_baseline; "
                "post_precompressed='\\n'.join(r['commands']['precompressed_config_post_run_audit']); "
                "assert 'edge_matrix_with_precompressed.json' in post_precompressed; "
                "assert '--require-prepare-precompressed-report' in post_precompressed; "
                "assert 'tools/build_goal_audit.py' in post_precompressed; "
                "assert '--final-output-root /tmp/llmcbench_handoff_output' in post_precompressed; "
                "assert '--portable-bundle-verify portable_bundle_verify.json' in post_precompressed; "
                "assert '--output /tmp/llmcbench_handoff_output/current_goal_audit.json' in post_precompressed; "
                "assert '--require-runtime-gpu-name-contains 4090' in r['commands']['manual_finalize'][0]; "
                "assert '--require-handoff-manifest' in r['commands']['manual_finalize'][0]; "
                "assert '--require-target-readiness-report' in r['commands']['manual_finalize'][0]; "
                "assert '--require-handoff-manifest' in r['commands']['manual_finalize'][1]; "
                "assert '--require-target-readiness-report' in r['commands']['manual_finalize'][1]; "
                "assert '--require-quant-runtime-report' in r['commands']['manual_finalize'][0]; "
                "assert 'resume' in r['commands']; "
                "assert 'tools/plan_edge_matrix_resume.py' in r['commands']['resume'][0]; "
                "assert 'matrix_resume_plan.json' in r['commands']['resume'][0]; "
                "assert 'scorecard' in r['commands']; "
                "assert 'tools/summarize_edge_results.py' in r['commands']['scorecard'][0]; "
                "assert 'edge_scorecard.json' in r['commands']['scorecard'][0]; "
                "assert 'export_results' in r['commands']; "
                "assert 'tools/export_edge_results_bundle.py' in r['commands']['export_results'][0]; "
                "assert '--require-complete' in r['commands']['export_results'][0]; "
                "assert 'verify_results_bundle' in r['commands']; "
                "assert 'tools/verify_edge_results_bundle.py' in r['commands']['verify_results_bundle'][0]; "
                "assert '--require-complete' in r['commands']['verify_results_bundle'][0]; "
                "assert '--require-target-readiness-ok' in r['commands']['verify_results_bundle'][0]; "
                "assert '--require-target-runtime' in r['commands']['verify_results_bundle'][0]; "
                "assert 'edge_results_bundle_verify.json' in r['commands']['verify_results_bundle'][0]; "
                "assert 'verify_delivery' in r['commands']; "
                "assert 'tools/verify_edge_delivery.py' in r['commands']['verify_delivery'][0]; "
                "assert '--results-archive' in r['commands']['verify_delivery'][0]; "
                "assert 'edge_results_bundle.tar.gz' in r['commands']['verify_delivery'][0]; "
                "assert 'edge_delivery_verify.json' in r['commands']['verify_delivery'][0]; "
                "assert any('authorized LLaMA3-8B checkpoint' in item for item in r['known_blockers_to_clear']); "
                "assert any('edge_delivery_verify.json with ok=true' in item for item in r['known_blockers_to_clear'])",
            ],
        ),
        (
            "handoff_manifest_precompressed_config_keeps_baseline_post_audit_baseline",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_handoff_precompressed_input_')); "
                "cfg=json.loads(Path('configs/edge_4090_matrix.json').read_text()); "
                "cfg['precompressed_paths']={'Qwen3-8B':{'gptq':'/data/models/precompressed/Qwen3-8B-gptq'}}; "
                "cfg_path=root/'edge_matrix_with_precompressed.json'; cfg_path.write_text(json.dumps(cfg)); "
                "out=root/'handoff.json'; "
                "p=subprocess.run([sys.executable,'tools/build_edge_handoff_manifest.py','--config',str(cfg_path),'--output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "baseline='\\n'.join(r['commands']['baseline_post_run_audit']); "
                "precompressed='\\n'.join(r['commands']['precompressed_config_post_run_audit']); "
                "assert 'edge_matrix_with_precompressed.json' not in baseline; "
                "assert 'configs/edge_4090_matrix.json' in baseline; "
                "assert 'edge_matrix_with_precompressed.json' in precompressed; "
                "assert '--require-prepare-precompressed-report' in precompressed",
            ],
        ),
        (
            "readiness_precompressed_config_keeps_baseline_commands_baseline",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_readiness_precompressed_input_')); "
                "cfg=json.loads(Path('configs/edge_4090_matrix.json').read_text()); "
                "cfg['precompressed_paths']={'Qwen3-8B':{'gptq':'/data/models/precompressed/Qwen3-8B-gptq'}}; "
                "cfg_path=root/'edge_matrix_with_precompressed.json'; cfg_path.write_text(json.dumps(cfg)); "
                "out=root/'readiness.json'; "
                "p=subprocess.run([sys.executable,'tools/audit_target_readiness.py','--config',str(cfg_path),'--root',str(root),'--require-optional-backends-report','--require-quant-runtime-report','--require-prepare-precompressed-report','--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); groups=r['next_command_groups']; "
                "baseline='\\n'.join(groups['baseline_preflight']+groups['baseline_readiness']+groups['baseline_finalize']+groups['baseline_post_run_audit']); "
                "precompressed='\\n'.join(groups['precompressed_config_followup']+groups['precompressed_config_post_run_audit']); "
                "assert 'edge_matrix_with_precompressed.json' not in baseline; "
                "assert 'configs/edge_4090_matrix.json' in baseline; "
                "assert 'edge_matrix_with_precompressed.json' in precompressed; "
                "assert '--require-prepare-precompressed-report' in precompressed",
            ],
        ),
        (
            "portable_bundle_static",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "text=Path('tools/export_edge_portable_bundle.py').read_text(); "
                "readme=Path('README.md').read_text(); "
                "goal=Path('tools/build_goal_audit.py').read_text(); "
                "assert 'PORTABLE_PATHS' in text; "
                "assert 'PORTABLE_BUNDLE_MANIFEST.json' in text; "
                "assert 'RUN_ON_TARGET.md' in text; "
                "assert 'edge_handoff_manifest.json' in text; "
                "assert 'tarfile.open' in text; "
                "assert '--archive <llmcbench_edge_portable.tar.gz>' in text; "
                "assert 'scripts/init_edge_storage.sh' in text; "
                "assert 'tools/export_edge_portable_bundle.py' in readme; "
                "assert 'tools/verify_edge_portable_bundle.py' in readme; "
                "assert 'scripts/init_edge_storage.sh' in readme; "
                "assert 'sudo DATA_ROOT=/data bash scripts/init_edge_storage.sh' in readme; "
                "assert '--archive /tmp/llmcbench_edge_portable.tar.gz' in readme; "
                "assert '--require-clean-code' in readme; "
                "assert '--require-commit $(git rev-parse HEAD)' in readme; "
                "assert 'scripts/init_edge_storage.sh' in Path('tools/verify_edge_portable_bundle.py').read_text(); "
                "assert 'tools/build_edge_handoff_manifest.py' in Path('tools/verify_edge_portable_bundle.py').read_text(); "
                "assert 'tools/build_goal_audit.py' in Path('tools/verify_edge_portable_bundle.py').read_text(); "
                "assert 'tools/verify_edge_env.py' in Path('tools/verify_edge_portable_bundle.py').read_text(); "
                "assert 'tools/prepare_target_paths.py' in Path('tools/verify_edge_portable_bundle.py').read_text(); "
                "assert 'tools/configure_external_methods.py' in Path('tools/verify_edge_portable_bundle.py').read_text(); "
                "assert 'tools/probe_quant_runtime.py' in Path('tools/verify_edge_portable_bundle.py').read_text(); "
                "assert 'tools/validate_edge_results.py' in Path('tools/verify_edge_portable_bundle.py').read_text(); "
                "assert 'require_clean_code' in Path('tools/verify_edge_portable_bundle.py').read_text(); "
                "assert 'require_commit' in Path('tools/verify_edge_portable_bundle.py').read_text(); "
                "assert 'tools/plan_edge_matrix_resume.py' in readme; "
                "assert 'tools/verify_edge_results_bundle.py' in readme; "
                "assert 'tools/verify_edge_delivery.py' in readme; "
                "assert '`tools/verify_edge_results_bundle.py`' in readme; "
                "assert '`tools/verify_edge_delivery.py`' in readme; "
                "assert '--results-archive /data/outputs/llmcbench_edge_4090/edge_results_bundle.tar.gz' in readme; "
                "assert 'root/bundle/archive delivery consistency' in readme; "
                "assert 'llmcbench_edge_portable.tar.gz' in readme; "
                "assert 'portable_bundle' in goal and 'tools/export_edge_portable_bundle.py' in goal; "
                "assert 'portable_bundle_verify' in goal and 'tools/verify_edge_portable_bundle.py' in goal; "
                "assert 'results_bundle_verify' in goal and 'tools/verify_edge_results_bundle.py' in goal",
            ],
        ),
        (
            "delivery_archive_docs_static",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "readme=Path('README.md').read_text(); "
                "audit=Path('docs/COMPLETION_AUDIT.md').read_text(); "
                "status=Path('docs/RUN_STATUS.md').read_text(); "
                "assert 'edge_results_bundle.tar.gz' in readme; assert 'target path preparation report' in readme; assert 'current-goal audit when present' in readme; "
                "assert 'stale tarballs are rejected' in readme; "
                "assert 'results_bundle_verify_ok' in readme; "
                "assert 'results_bundle_verify_complete' in readme; "
                "assert 'delivery_verify_ok' in readme; "
                "assert 'delivery_verify_substatus_ok' in readme; "
                "assert 'target_completion_ok' in readme; "
                "assert 'method-specific allowed compression backend provenance' in readme; "
                "assert 'compression_materialized` and `compression_uses_precompressed` booleans' in readme; "
                "assert 'compression_external_command` with `compression_external_returncode=0' in readme; "
                "assert 'validator.ok=true' in readme; "
                "assert 'has `ok=true` with `validator.ok=true`' in readme; "
                "assert 'target readiness has `ok=true`' in readme; "
                "assert '`completion_ok=true`' in readme; "
                "assert 'no `before_matrix_blockers`' in readme; "
                "assert 'With `--require-target-readiness-ok`' in readme; "
                "assert 'root `completion_audit.json`' in readme; "
                "assert 'edge_results_bundle_verify.json' in readme; "
                "assert '`\"ok\": true` and `\"complete\": true`' in readme; "
                "assert 'output root, `edge_results_bundle/` directory and `edge_results_bundle.tar.gz` archive' in audit; assert 'edge_target_paths_report.json' in audit; "
                "assert 'stale bundle directories or stale tarballs are rejected' in audit; "
                "assert 'Goal audit matrix completion' in audit; "
                "assert 'method-specific allowed compression backend provenance' in audit; "
                "assert 'compression_external_command` and `compression_external_returncode=0' in audit; "
                "assert 'validator.ok=true' in audit; "
                "assert 'self-contained result/delivery verifiers' in audit; "
                "assert 'root completion audit must have `\"ok\": true` with `validator.ok=true`' in audit; "
                "assert 'target_readiness.json` has `\"ok\": true` with `completion_ok=true`' in audit; "
                "assert 'no `before_matrix_blockers`' in audit; "
                "assert 'edge_results_bundle_verify.json` to report `\"ok\": true` and `\"complete\": true' in audit; "
                "assert 'edge_delivery_verify.json` to report `\"ok\": true` with root, bundle-directory and archive substatus checks passing' in audit; "
                "assert 'JSON/CSV row/column/value consistency' in audit; "
                "assert 'compression backend/materialized/precompressed/external-command provenance' in audit; "
                "assert 'deployment metrics and token/s plus memory-peak consistency' in audit; "
                "assert 'resource metrics and storage/sparse-FLOPs/quantized-storage consistency' in audit; "
                "assert 'Delivery verification has also been tightened for portable handoff' in status; "
                "assert 'matrix_results.json` and `matrix_results.csv` must agree by row' in status; "
                "assert 'deployment tokens/s, peak memory and memory-limit fields must agree' in status; "
                "assert 'resource' in status and 'sparse effective storage/FLOPs' in status and 'quantized storage estimates' in status; "
                "assert 'results_bundle_verify_ok' in status; "
                "assert 'results_bundle_verify_complete' in status; "
                "assert 'delivery_verify_ok' in status; "
                "assert 'delivery_verify_substatus_ok' in status; "
                "assert 'target_completion_ok' in status; "
                "assert 'Current-goal audit matrix completion now also requires exact target pair' in status; "
                "assert 'method-specific allowed compression backend' in status; "
                "assert 'compression_external_command' in status and 'compression_external_returncode=0' in status; "
                "assert 'validator.ok=true' in status; "
                "assert 'results-bundle verifier and' in status; "
                "assert 'results-bundle export and verifier now also require target readiness' in status; "
                "assert 'target_readiness.json` to report `ok=true`' in status; "
                "assert '`completion_ok=true`' in status; "
                "assert '`before_matrix_blockers`' in status; "
                "assert 'delivery verification' in status and 'root output directory' in status; "
                "assert 'edge_results_bundle_verify.json` reports `\"ok\": true` and `\"complete\": true' in status; "
                "assert 'stale root files, stale bundle directories and stale' in status",
            ],
        ),
        (
            "portable_bundle_export",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys, tarfile; "
                "from pathlib import Path; "
                "root=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "archive=Path('/tmp/llmcbench_portable_bundle_smoke.tar.gz'); "
                "shutil.rmtree(root, ignore_errors=True); "
                "archive.unlink(missing_ok=True); "
                "proc=subprocess.run([sys.executable,'tools/export_edge_portable_bundle.py',"
                "'--config','configs/edge_4090_matrix.json',"
                "'--output-root','/tmp/llmcbench_portable_target',"
                "'--output-dir',str(root),'--archive',str(archive),'--force'],capture_output=True,text=True); "
                "assert proc.returncode == 0, proc.stderr + proc.stdout; "
                "out=json.loads(proc.stdout); "
                "assert out['ok'] is True and out['archive'] == str(archive); "
                "assert root.joinpath('README.md').exists(); "
                "assert root.joinpath('edge_benchmark','cli.py').exists(); "
                "assert root.joinpath('scripts','init_edge_storage.sh').exists(); "
                "assert root.joinpath('scripts','setup_edge_env.sh').exists(); "
                "assert root.joinpath('scripts','setup_optional_backends.sh').exists(); "
                "assert root.joinpath('scripts','run_edge_4090_full.sh').exists(); "
                "assert root.joinpath('tools','run_edge_matrix.py').exists(); "
                "assert root.joinpath('tools','verify_edge_portable_bundle.py').exists(); "
                "assert root.joinpath('tools','build_edge_handoff_manifest.py').exists(); "
                "assert root.joinpath('tools','build_goal_audit.py').exists(); "
                "assert root.joinpath('tools','plan_edge_matrix_resume.py').exists(); "
                "assert root.joinpath('tools','summarize_edge_results.py').exists(); "
                "assert root.joinpath('tools','collect_edge_results.py').exists(); "
                "assert root.joinpath('tools','render_edge_report.py').exists(); "
                "assert root.joinpath('tools','export_edge_results_bundle.py').exists(); "
                "assert root.joinpath('tools','verify_edge_results_bundle.py').exists(); "
                "assert root.joinpath('tools','verify_edge_delivery.py').exists(); "
                "assert root.joinpath('tools','audit_target_readiness.py').exists(); "
                "assert root.joinpath('tools','run_edge_preflight.py').exists(); "
                "assert root.joinpath('tools','verify_edge_env.py').exists(); "
                "assert root.joinpath('tools','prepare_target_paths.py').exists(); "
                "assert root.joinpath('tools','prewarm_lm_eval_tasks.py').exists(); "
                "assert root.joinpath('tools','prepare_edge_assets.py').exists(); "
                "assert root.joinpath('tools','prepare_precompressed_assets.py').exists(); "
                "assert root.joinpath('tools','configure_external_methods.py').exists(); "
                "assert root.joinpath('tools','check_method_readiness.py').exists(); "
                "assert root.joinpath('tools','probe_quant_runtime.py').exists(); "
                "assert root.joinpath('configs','edge_4090_structured_pruning.json').exists(); "
                "assert root.joinpath('tools','validate_edge_results.py').exists(); "
                "manifest=json.loads(root.joinpath('PORTABLE_BUNDLE_MANIFEST.json').read_text()); "
                "handoff=json.loads(root.joinpath('edge_handoff_manifest.json').read_text()); "
                "runbook=root.joinpath('RUN_ON_TARGET.md').read_text(); "
                "paths={item['path'] for item in manifest['files']}; "
                "assert manifest['schema_version'] == 1; "
                "assert manifest['archive'] == str(archive); "
                "assert manifest['output_root'] == '/tmp/llmcbench_portable_target'; "
                "assert 'README.md' in paths and 'edge_handoff_manifest.json' in paths and 'RUN_ON_TARGET.md' in paths; "
                "assert 'PORTABLE_BUNDLE_MANIFEST.json' not in paths; "
                "assert handoff['config']['path'] == 'configs/edge_4090_matrix.json'; "
                "assert handoff['paths']['output_root'] == '/tmp/llmcbench_portable_target'; "
                "assert 'python tools/verify_edge_portable_bundle.py --bundle-root . --require-clean-code --require-commit ' in runbook; "
                "assert ' --output portable_bundle_verify.json' in runbook; "
                "assert 'python tools/verify_edge_portable_bundle.py --bundle-root . --archive <llmcbench_edge_portable.tar.gz> --require-clean-code --require-commit ' in runbook; "
                "assert ' --output portable_bundle_archive_verify.json' in runbook; "
                "assert 'sudo DATA_ROOT=/data bash scripts/init_edge_storage.sh' in runbook; "
                "assert 'EDGE_RUNTIME_PROFILE=gptqmodel' in runbook; "
                "assert 'scripts/init_edge_storage.sh' in handoff['commands']['setup'][2]; "
                "assert 'EDGE_RUNTIME_PROFILE=gptqmodel' in '\\n'.join(handoff['commands']['setup']); "
                "assert 'tools/plan_edge_matrix_resume.py' in runbook; "
                "assert 'tools/summarize_edge_results.py' in runbook; "
                "assert 'tools/verify_edge_delivery.py' in runbook; "
                "assert 'tools/verify_edge_results_bundle.py' in runbook; "
                "assert '### prepare_precompressed' in runbook; "
                "assert '### precompressed_config_followup' in runbook; "
                "assert '### baseline_post_run_audit' in runbook; "
                "assert '### precompressed_config_post_run_audit' in runbook; "
                "assert 'RUN_PRECOMPRESSED_PREPARE=1' in runbook; "
                "assert 'PRECOMPRESSED_PREPARE_ONLY=1' in runbook; "
                "assert 'PRECOMPRESSED_GROUPS=' in runbook; "
                "assert 'scripts/run_edge_4090_full.sh' in runbook; "
                "assert 'edge_matrix_with_precompressed.json' in runbook; "
                "assert '--require-prepare-precompressed-report' in runbook; "
                "assert '--require-target-readiness-ok' in runbook; "
                "assert '--require-target-runtime' in runbook; "
                "assert 'CHECK_OPTIONAL_BACKEND_RESOLUTION=1 CHECK_QUANT_RUNTIME=1 bash scripts/run_edge_4090_full.sh' in runbook; "
                "assert 'validator.ok=true' in runbook; "
                "assert 'edge_results_bundle_verify.json' in runbook; "
                "assert 'complete=true' in runbook; "
                "assert 'target_readiness.json' in runbook; "
                "assert 'completion_ok=true' in runbook; "
                "assert 'edge_delivery_verify.json' in runbook; "
                "assert archive.exists() and archive.stat().st_size > 0; "
                "tar=tarfile.open(archive); names=set(tar.getnames()); tar.close(); "
                "assert any(name.endswith('/PORTABLE_BUNDLE_MANIFEST.json') for name in names); "
                "assert any(name.endswith('/RUN_ON_TARGET.md') for name in names); "
                "verify=subprocess.run([sys.executable,str(root/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(root),'--archive',str(archive),'--require-commit',manifest['code']['commit'],'--output',str(root/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert verify.returncode == 0, verify.stderr + verify.stdout; "
                "verify_report=json.loads((root/'portable_bundle_verify.json').read_text()); "
                "assert verify_report['ok'] is True; "
                "assert verify_report['archive_status']['ok'] is True; "
                "assert verify_report['archive_status']['checked'] is True; "
                "assert verify_report['archive_status']['expected_top_level_dir'] == root.name; "
                "assert verify_report['archive_status']['archive_top_level_dirs'] == [root.name]; "
                "assert verify_report['archive_status']['verified_archive_file_count'] == verify_report['file_count'] + 1; "
                "assert verify_report['archive_status']['size_mismatches'] == []; "
                "assert verify_report['archive_status']['sha256_mismatches'] == []; "
                "assert verify_report['archive_status']['duplicate_archive_files'] == []; "
                "assert verify_report['archive_status']['unsafe_archive_members'] == []; "
                "assert verify_report['archive_status']['unsupported_archive_members'] == []; "
                "assert verify_report['code_status']['ok'] is True; "
                "assert verify_report['code_status']['require_commit'] == manifest['code']['commit']; "
                "assert verify_report['code_status']['manifest_commit'] == manifest['code']['commit']; "
                "assert verify_report['code_status']['handoff_commit'] == handoff['code']['commit']; "
                "assert verify_report['code_status']['manifest_dirty'] == manifest['code']['dirty']; "
                "assert verify_report['code_status']['handoff_dirty'] == handoff['code']['dirty']; "
                "required_paths={item['path']: item for item in verify_report['required_path_status']}; "
                "assert required_paths['scripts/init_edge_storage.sh']['ok'] is True; "
                "assert required_paths['scripts/setup_edge_env.sh']['ok'] is True; "
                "assert required_paths['scripts/setup_optional_backends.sh']['ok'] is True; "
                "assert required_paths['tools/build_edge_handoff_manifest.py']['ok'] is True; "
                "assert required_paths['tools/build_goal_audit.py']['ok'] is True; "
                "assert required_paths['tools/audit_target_readiness.py']['ok'] is True; "
                "assert required_paths['tools/run_edge_preflight.py']['ok'] is True; "
                "assert required_paths['tools/verify_edge_env.py']['ok'] is True; "
                "assert required_paths['tools/prepare_target_paths.py']['ok'] is True; "
                "assert required_paths['tools/prewarm_lm_eval_tasks.py']['ok'] is True; "
                "assert required_paths['tools/configure_external_methods.py']['ok'] is True; "
                "assert required_paths['tools/probe_quant_runtime.py']['ok'] is True; "
                "assert required_paths['tools/validate_edge_results.py']['ok'] is True; "
                "assert verify_report['handoff_manifest_status']['missing_completion_gates'] == []; "
                "assert verify_report['handoff_manifest_status']['missing_required_artifacts'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['setup']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['run']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['prepare']['missing_tokens'] == []; "
                "assert '--fail-on-error' in handoff['commands']['prepare'][0]; "
                "assert verify_report['handoff_manifest_status']['command_status']['prepare_precompressed']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['prewarm']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['precompressed_config_followup']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['preflight']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['readiness']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['manual_finalize']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['post_run_audit']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['baseline_post_run_audit']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['precompressed_config_post_run_audit']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['scorecard']['exists'] is True; "
                "assert verify_report['handoff_manifest_status']['command_status']['scorecard']['missing_tokens'] == []; "
                "assert verify_report['handoff_manifest_status']['command_status']['verify_results_bundle']['exists'] is True; "
                "assert verify_report['handoff_manifest_status']['command_status']['verify_results_bundle']['missing_tokens'] == []; "
                "assert verify_report['runbook_status']['missing_required_text'] == []; "
                "assert verify_report['verified_file_count'] == verify_report['file_count']",
            ],
        ),
        (
            "portable_bundle_rejects_dirty_code_when_required",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_dirty_code_required'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "manifest_path=dst/'PORTABLE_BUNDLE_MANIFEST.json'; "
                "manifest=json.loads(manifest_path.read_text()); "
                "manifest['code']['dirty']=True; "
                "manifest_path.write_text(json.dumps(manifest)); "
                "handoff_path=dst/'edge_handoff_manifest.json'; "
                "handoff=json.loads(handoff_path.read_text()); "
                "handoff['code']['dirty']=True; "
                "handoff_path.write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--require-clean-code','--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "assert r['code_status']['ok'] is False; "
                "assert 'portable manifest was exported from a dirty worktree' in r['code_status']['violations']; "
                "assert 'handoff manifest was exported from a dirty worktree' in r['code_status']['violations']; "
                "assert 'portable code provenance verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_wrong_required_commit",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_wrong_commit_required'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--require-commit','0'*40,'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "assert r['code_status']['ok'] is False; "
                "assert r['code_status']['require_commit'] == '0'*40; "
                "assert 'portable manifest commit does not match required commit' in r['code_status']['violations']; "
                "assert 'handoff manifest commit does not match required commit' in r['code_status']['violations']; "
                "assert 'portable code provenance verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_archive_mismatch",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys, tarfile; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "archive=Path('/tmp/llmcbench_portable_bundle_bad_archive.tar.gz'); "
                "archive.unlink(missing_ok=True); "
                "tar=tarfile.open(archive,'w:gz'); "
                "[tar.add(path, arcname=str(Path(src.name)/path.relative_to(src))) for path in src.rglob('*') if path.is_file() and path.name!='RUN_ON_TARGET.md']; "
                "tar.close(); "
                "p=subprocess.run([sys.executable,str(src/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(src),'--archive',str(archive),'--output',str(src/'portable_bundle_bad_archive_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((src/'portable_bundle_bad_archive_verify.json').read_text()); "
                "assert r['archive_status']['ok'] is False; "
                "assert 'RUN_ON_TARGET.md' in r['archive_status']['missing_archive_files']; "
                "assert 'portable archive verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_archive_content_mismatch",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys, tarfile, tempfile; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "archive=Path('/tmp/llmcbench_portable_bundle_bad_content.tar.gz'); "
                "tmp=Path(tempfile.mkdtemp(prefix='llmcbench_portable_bad_content_')); "
                "shutil.copytree(src, tmp/src.name); "
                "(tmp/src.name/'RUN_ON_TARGET.md').write_text('stale runbook content'); "
                "archive.unlink(missing_ok=True); "
                "tar=tarfile.open(archive,'w:gz'); tar.add(tmp/src.name, arcname=src.name); tar.close(); "
                "p=subprocess.run([sys.executable,str(src/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(src),'--archive',str(archive),'--output',str(src/'portable_bundle_bad_content_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((src/'portable_bundle_bad_content_verify.json').read_text()); "
                "assert r['archive_status']['ok'] is False; "
                "assert any(item['path']=='RUN_ON_TARGET.md' for item in r['archive_status']['size_mismatches']); "
                "assert any(item['path']=='RUN_ON_TARGET.md' for item in r['archive_status']['sha256_mismatches']); "
                "assert 'archive file size mismatch' in r['archive_status']['violations']; "
                "assert 'archive file sha256 mismatch' in r['archive_status']['violations']; "
                "assert 'portable archive verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_archive_manifest_content_mismatch",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys, tarfile, tempfile; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "archive=Path('/tmp/llmcbench_portable_bundle_bad_manifest_content.tar.gz'); "
                "tmp=Path(tempfile.mkdtemp(prefix='llmcbench_portable_bad_manifest_')); "
                "shutil.copytree(src, tmp/src.name); "
                "(tmp/src.name/'PORTABLE_BUNDLE_MANIFEST.json').write_text(json.dumps({'schema_version':1,'stale':True})); "
                "archive.unlink(missing_ok=True); "
                "tar=tarfile.open(archive,'w:gz'); tar.add(tmp/src.name, arcname=src.name); tar.close(); "
                "p=subprocess.run([sys.executable,str(src/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(src),'--archive',str(archive),'--output',str(src/'portable_bundle_bad_manifest_content_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((src/'portable_bundle_bad_manifest_content_verify.json').read_text()); "
                "assert r['archive_status']['ok'] is False; "
                "assert any(item['path']=='PORTABLE_BUNDLE_MANIFEST.json' for item in r['archive_status']['size_mismatches']); "
                "assert any(item['path']=='PORTABLE_BUNDLE_MANIFEST.json' for item in r['archive_status']['sha256_mismatches']); "
                "assert 'archive file size mismatch' in r['archive_status']['violations']; "
                "assert 'archive file sha256 mismatch' in r['archive_status']['violations']; "
                "assert 'portable archive verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_duplicate_archive_path",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys, tarfile; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "archive=Path('/tmp/llmcbench_portable_bundle_duplicate_archive_path.tar.gz'); "
                "archive.unlink(missing_ok=True); "
                "tar=tarfile.open(archive,'w:gz'); "
                "[tar.add(path, arcname=str(Path(src.name)/path.relative_to(src))) for path in src.rglob('*') if path.is_file()]; "
                "tar.add(src/'RUN_ON_TARGET.md', arcname=str(Path(src.name)/'RUN_ON_TARGET.md')); "
                "tar.close(); "
                "p=subprocess.run([sys.executable,str(src/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(src),'--archive',str(archive),'--output',str(src/'portable_bundle_duplicate_archive_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((src/'portable_bundle_duplicate_archive_verify.json').read_text()); "
                "assert r['archive_status']['ok'] is False; "
                "assert 'RUN_ON_TARGET.md' in r['archive_status']['duplicate_archive_files']; "
                "assert 'archive contains duplicate file paths' in r['archive_status']['violations']; "
                "assert 'portable archive verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_unsafe_archive_member",
            [
                sys.executable,
                "-c",
                "import io, json, shutil, subprocess, sys, tarfile; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "archive=Path('/tmp/llmcbench_portable_bundle_unsafe_member.tar.gz'); "
                "archive.unlink(missing_ok=True); "
                "tar=tarfile.open(archive,'w:gz'); "
                "[tar.add(path, arcname=str(Path(src.name)/path.relative_to(src))) for path in src.rglob('*') if path.is_file()]; "
                "payload=b'unsafe'; info=tarfile.TarInfo('../escape.txt'); info.size=len(payload); "
                "tar.addfile(info, io.BytesIO(payload)); tar.close(); "
                "p=subprocess.run([sys.executable,str(src/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(src),'--archive',str(archive),'--output',str(src/'portable_bundle_unsafe_member_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((src/'portable_bundle_unsafe_member_verify.json').read_text()); "
                "assert r['archive_status']['ok'] is False; "
                "assert '../escape.txt' in r['archive_status']['unsafe_archive_members']; "
                "assert 'archive contains unsafe member paths' in r['archive_status']['violations']; "
                "assert 'portable archive verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_unsupported_archive_member",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys, tarfile; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "archive=Path('/tmp/llmcbench_portable_bundle_unsupported_member.tar.gz'); "
                "archive.unlink(missing_ok=True); "
                "tar=tarfile.open(archive,'w:gz'); "
                "[tar.add(path, arcname=str(Path(src.name)/path.relative_to(src))) for path in src.rglob('*') if path.is_file()]; "
                "info=tarfile.TarInfo(str(Path(src.name)/'unsafe_link')); info.type=tarfile.SYMTYPE; info.linkname='../outside'; "
                "tar.addfile(info); tar.close(); "
                "p=subprocess.run([sys.executable,str(src/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(src),'--archive',str(archive),'--output',str(src/'portable_bundle_unsupported_member_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((src/'portable_bundle_unsupported_member_verify.json').read_text()); "
                "assert r['archive_status']['ok'] is False; "
                "assert any(item['name'].endswith('unsafe_link') for item in r['archive_status']['unsupported_archive_members']); "
                "assert 'archive contains unsupported member types' in r['archive_status']['violations']; "
                "assert 'portable archive verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_wrong_archive_top_level",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys, tarfile, tempfile; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "archive=Path('/tmp/llmcbench_portable_bundle_wrong_top.tar.gz'); "
                "tmp=Path(tempfile.mkdtemp(prefix='llmcbench_portable_wrong_top_')); "
                "wrong=tmp/'wrong_top_level'; shutil.copytree(src, wrong); "
                "archive.unlink(missing_ok=True); "
                "tar=tarfile.open(archive,'w:gz'); tar.add(wrong, arcname=wrong.name); tar.close(); "
                "p=subprocess.run([sys.executable,str(src/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(src),'--archive',str(archive),'--output',str(src/'portable_bundle_wrong_top_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((src/'portable_bundle_wrong_top_verify.json').read_text()); "
                "assert r['archive_status']['ok'] is False; "
                "assert r['archive_status']['expected_top_level_dir'] == src.name; "
                "assert r['archive_status']['archive_top_level_dirs'] == ['wrong_top_level']; "
                "assert 'archive top-level directory does not match bundle root' in r['archive_status']['violations']; "
                "assert 'portable archive verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_stale_handoff_gates",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_stale_handoff'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "handoff_path=dst/'edge_handoff_manifest.json'; "
                "handoff=json.loads(handoff_path.read_text()); "
                "handoff['completion_gates']['reports']=[item for item in handoff['completion_gates']['reports'] if 'edge_delivery_verify.json' not in item]; "
                "handoff_path.write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "assert r['ok'] is False; "
                "missing=r['handoff_manifest_status']['missing_completion_gates']; "
                "assert 'edge_delivery_verify.json with ok=true and all substatus checks passing' in missing; "
                "assert 'handoff manifest verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_missing_scorecard_command",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_missing_scorecard_command'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "handoff_path=dst/'edge_handoff_manifest.json'; "
                "handoff=json.loads(handoff_path.read_text()); "
                "handoff['commands'].pop('scorecard', None); "
                "handoff_path.write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "status=r['handoff_manifest_status']['command_status']['scorecard']; "
                "assert status['exists'] is False; "
                "assert 'handoff manifest commands.scorecard is missing' in r['handoff_manifest_status']['violations']; "
                "assert 'handoff manifest verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_missing_target_paths_completion_gate",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_missing_target_paths_gate'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "handoff_path=dst/'edge_handoff_manifest.json'; "
                "handoff=json.loads(handoff_path.read_text()); "
                "handoff['completion_gates']['reports']=[item for item in handoff['completion_gates']['reports'] if 'edge_target_paths_report.json' not in item]; "
                "handoff_path.write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "missing=r['handoff_manifest_status']['missing_completion_gates']; "
                "assert 'edge_target_paths_report.json with ok=true' in missing; "
                "assert 'handoff manifest completion_gates.reports is missing final delivery gates' in r['handoff_manifest_status']['violations']; "
                "assert 'handoff manifest verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_missing_handoff_target_paths_artifact",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_missing_target_paths_artifact'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "handoff_path=dst/'edge_handoff_manifest.json'; "
                "handoff=json.loads(handoff_path.read_text()); "
                "handoff['required_artifacts'].pop('edge_target_paths_report.json', None); "
                "handoff_path.write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "missing=r['handoff_manifest_status']['missing_required_artifacts']; "
                "assert 'edge_target_paths_report.json' in missing; "
                "assert 'handoff manifest required_artifacts is missing final artifacts' in r['handoff_manifest_status']['violations']; "
                "assert 'handoff manifest verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_weak_readiness_command",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_weak_readiness_command'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "handoff_path=dst/'edge_handoff_manifest.json'; "
                "handoff=json.loads(handoff_path.read_text()); "
                "handoff['commands']['readiness']=[cmd.replace(' --require-quant-runtime-report','') for cmd in handoff['commands']['readiness']]; "
                "handoff_path.write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "missing=r['handoff_manifest_status']['command_status']['readiness']['missing_tokens']; "
                "assert '--require-quant-runtime-report' in missing; "
                "assert 'handoff manifest commands.readiness is missing required tokens' in r['handoff_manifest_status']['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_weak_run_command",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_weak_run_command'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "handoff_path=dst/'edge_handoff_manifest.json'; "
                "handoff=json.loads(handoff_path.read_text()); "
                "handoff['commands']['run']=[cmd.replace(' CHECK_QUANT_RUNTIME=1','') for cmd in handoff['commands']['run']]; "
                "handoff_path.write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "missing=r['handoff_manifest_status']['command_status']['run']['missing_tokens']; "
                "assert 'CHECK_QUANT_RUNTIME=1' in missing; "
                "assert 'handoff manifest commands.run is missing required tokens' in r['handoff_manifest_status']['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_weak_precompressed_followup",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_weak_precompressed_followup'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "handoff_path=dst/'edge_handoff_manifest.json'; "
                "handoff=json.loads(handoff_path.read_text()); "
                "handoff['commands']['precompressed_config_followup']=[cmd.replace(' --require-prepare-precompressed-report','') for cmd in handoff['commands']['precompressed_config_followup']]; "
                "handoff_path.write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "missing=r['handoff_manifest_status']['command_status']['precompressed_config_followup']['missing_tokens']; "
                "assert '--require-prepare-precompressed-report' in missing; "
                "assert 'handoff manifest commands.precompressed_config_followup is missing required tokens' in r['handoff_manifest_status']['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_weak_precompressed_post_audit",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_weak_precompressed_post'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "handoff_path=dst/'edge_handoff_manifest.json'; "
                "handoff=json.loads(handoff_path.read_text()); "
                "handoff['commands']['precompressed_config_post_run_audit']=[cmd.replace(' --require-prepare-precompressed-report','') for cmd in handoff['commands']['precompressed_config_post_run_audit']]; "
                "handoff_path.write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "missing=r['handoff_manifest_status']['command_status']['precompressed_config_post_run_audit']['missing_tokens']; "
                "assert '--require-prepare-precompressed-report' in missing; "
                "assert 'handoff manifest commands.precompressed_config_post_run_audit is missing required tokens' in r['handoff_manifest_status']['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_missing_results_bundle_command",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_missing_results_command'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "handoff_path=dst/'edge_handoff_manifest.json'; "
                "handoff=json.loads(handoff_path.read_text()); "
                "handoff['commands'].pop('verify_results_bundle', None); "
                "handoff_path.write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "status=r['handoff_manifest_status']['command_status']['verify_results_bundle']; "
                "assert status['exists'] is False; "
                "assert 'handoff manifest commands.verify_results_bundle is missing' in r['handoff_manifest_status']['violations']; "
                "assert 'handoff manifest verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_missing_target_readiness_tool",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_missing_target_tool'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "(dst/'tools'/'audit_target_readiness.py').unlink(); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "missing=[item for item in r['required_path_status'] if item['path']=='tools/audit_target_readiness.py'][0]; "
                "assert missing['ok'] is False; "
                "assert 'required bundle file is missing: tools/audit_target_readiness.py' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_unlisted_required_file",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_unlisted_required'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "manifest_path=dst/'PORTABLE_BUNDLE_MANIFEST.json'; "
                "manifest=json.loads(manifest_path.read_text()); "
                "manifest['files']=[item for item in manifest['files'] if item['path']!='tools/audit_target_readiness.py']; "
                "manifest_path.write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "status=[item for item in r['required_path_status'] if item['path']=='tools/audit_target_readiness.py'][0]; "
                "assert status['ok'] is True and status['exists'] is True and status['in_manifest_files'] is False; "
                "assert 'required bundle file is missing from manifest files: tools/audit_target_readiness.py' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_duplicate_manifest_path",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_duplicate_manifest_path'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "manifest_path=dst/'PORTABLE_BUNDLE_MANIFEST.json'; "
                "manifest=json.loads(manifest_path.read_text()); "
                "manifest['files'].append(dict(manifest['files'][0])); "
                "manifest_path.write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "assert manifest['files'][0]['path'] in r['duplicate_manifest_paths']; "
                "assert 'manifest files contain duplicate paths' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_missing_optional_backend_script",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_missing_optional_script'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "(dst/'scripts'/'setup_optional_backends.sh').unlink(); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "missing=[item for item in r['required_path_status'] if item['path']=='scripts/setup_optional_backends.sh'][0]; "
                "assert missing['ok'] is False; "
                "assert 'required bundle file is missing: scripts/setup_optional_backends.sh' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_weak_results_bundle_gate",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_weak_results_gate'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "handoff_path=dst/'edge_handoff_manifest.json'; "
                "handoff=json.loads(handoff_path.read_text()); "
                "handoff['completion_gates']['reports']=[item if 'edge_results_bundle_verify.json' not in item else 'edge_results_bundle_verify.json with ok=true and complete=true' for item in handoff['completion_gates']['reports']]; "
                "handoff_path.write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "missing=r['handoff_manifest_status']['missing_completion_gates']; "
                "assert 'edge_results_bundle_verify.json with ok=true, complete=true, validator.ok=true and target completion_ok=true' in missing; "
                "assert 'handoff manifest verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_stale_completion_rule",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_stale_rule'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "runbook=dst/'RUN_ON_TARGET.md'; "
                "runbook.write_text(runbook.read_text().replace('with `validator.ok=true`, ','')); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "assert r['ok'] is False; "
                "assert 'validator.ok=true' in r['runbook_status']['missing_required_text']; "
                "assert 'target runbook verification failed' in r['violations']",
            ],
        ),
        (
            "portable_bundle_rejects_missing_target_completion_rule",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_portable_bundle_smoke'); "
                "dst=Path('/tmp/llmcbench_portable_bundle_stale_target_rule'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "runbook=dst/'RUN_ON_TARGET.md'; "
                "runbook.write_text(runbook.read_text().replace('with `completion_ok=true`, ','')); "
                "p=subprocess.run([sys.executable,str(dst/'tools'/'verify_edge_portable_bundle.py'),"
                "'--bundle-root',str(dst),'--output',str(dst/'portable_bundle_verify.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads((dst/'portable_bundle_verify.json').read_text()); "
                "assert r['ok'] is False; "
                "assert 'completion_ok=true' in r['runbook_status']['missing_required_text']; "
                "assert 'target runbook verification failed' in r['violations']",
            ],
        ),
        (
            "target_readiness_blockers",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_target_ready_')); "
                "cfg=root/'config.json'; out=root/'target_readiness.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'Tiny','path':str(root/'missing')}],"
                "'methods':['dense'],'defaults':{'output_root':str(root/'out'),'deployment_memory_limit_gb':24}})); "
                "p=subprocess.run([sys.executable,'tools/audit_target_readiness.py','--config',str(cfg),"
                "'--root',str(root/'out'),'--check-method-readiness','--require-prewarm','--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "assert r['completion_ok'] is False; "
                "assert r['ready_for_matrix'] is False; "
                "assert r['required_before_matrix'] == ['asset_audit','preflight','prewarm']; "
                "assert {'asset_audit','preflight','prewarm'} <= set(r['missing_before_matrix']); "
                "(root/'out').mkdir(parents=True, exist_ok=True); "
                "(root/'out'/'edge_asset_audit.json').write_text(json.dumps({'ok':True,'paths':{'hf_cache_dir':{'block_device':{'is_non_rotational':True},'can_create':True},'data_dir':{'block_device':{'is_non_rotational':True},'can_create':True},'output_root':{'block_device':{'is_non_rotational':False},'can_create':False,'writable_probe_path':str(root)}}})); "
                "p=subprocess.run([sys.executable,'tools/audit_target_readiness.py','--config',str(cfg),"
                "'--root',str(root/'out'),'--require-non-rotational-storage','--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "rs=json.loads(out.read_text()); "
                "assert 'asset_storage' in rs['required_before_matrix']; "
                "assert 'asset_storage' in rs['missing_before_matrix']; "
                "assert any(item['artifact']=='asset_storage' and item['type']=='artifact_not_ok' for item in rs['before_matrix_blockers']); "
                "p=subprocess.run([sys.executable,'tools/audit_target_readiness.py','--config',str(cfg),"
                "'--root',str(root/'out'),'--require-prepare-precompressed-report','--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "rp=json.loads(out.read_text()); "
                "assert 'prepare_precompressed' in rp['required_before_matrix']; "
                "assert 'prepare_precompressed' in rp['missing_before_matrix']; "
                "assert any(item['artifact']=='prepare_precompressed' and item['type']=='missing_artifact' for item in rp['before_matrix_blockers']); "
                "(root/'out'/'edge_prepare_precompressed_report.json').write_text(json.dumps({'ok':False,'blockers':[{'model':'LLaMA3-8B','method':'omniquant','reason':'repo_not_configured'}]})); "
                "p=subprocess.run([sys.executable,'tools/audit_target_readiness.py','--config',str(cfg),"
                "'--root',str(root/'out'),'--require-prepare-precompressed-report','--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "rpb=json.loads(out.read_text()); "
                "blocked=[item for item in rpb['before_matrix_blockers'] if item.get('artifact')=='prepare_precompressed'][0]; "
                "assert blocked['blockers'][0]['reason'] == 'repo_not_configured'; "
                "p=subprocess.run([sys.executable,'tools/audit_target_readiness.py','--config',str(cfg),"
                "'--root',str(root/'out'),'--require-handoff-manifest','--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r2=json.loads(out.read_text()); "
                "assert 'handoff_manifest' in r2['required_before_matrix']; "
                "assert 'handoff_manifest' in r2['missing_before_matrix']; "
                "(root/'out'/'edge_handoff_manifest.json').parent.mkdir(parents=True, exist_ok=True); "
                "(root/'out'/'edge_handoff_manifest.json').write_text(json.dumps({'schema_version':1,'commands':{}})); "
                "p=subprocess.run([sys.executable,'tools/audit_target_readiness.py','--config',str(cfg),"
                "'--root',str(root/'out'),'--require-handoff-manifest','--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r3=json.loads(out.read_text()); "
                "assert 'handoff_manifest' in r3['missing_before_matrix']; "
                "assert any(item['artifact']=='handoff_manifest' and item['type']=='artifact_not_ok' for item in r3['before_matrix_blockers']); "
                "(root/'out'/'edge_handoff_manifest.json').write_text(json.dumps({"
                "'schema_version':1,"
                "'code':{'commit':'x','branch':'edge-pipeline-updates','origin':'git@code.byted.org:data/llmcbench.git','dirty':False},"
                "'config':{},'target_environment':{},'paths':{'output_root':str(root/'out')},"
                "'commands':{'setup':['setup'],'prepare':['prepare'],'preflight':['preflight'],'run':['run'],"
                "'manual_finalize':['python tools/run_edge_matrix.py --finalize','python tools/audit_edge_completion.py --require-finalize-report'],"
                "'post_run_audit':['audit']},"
                "'required_artifacts':{'matrix_results.json':{},'matrix_results.csv':{},'edge_report.md':{},'completion_audit.json':{}},"
                "'completion_gates':{'coverage':{'required_mmlu_subtask_count':57,'required_mmlu_subtasks':__import__('tools.prewarm_lm_eval_tasks',fromlist=['DEFAULT_MMLU_SUBTASKS']).DEFAULT_MMLU_SUBTASKS},'reports':['completion_audit.json with ok=true']}})); "
                "p=subprocess.run([sys.executable,'tools/audit_target_readiness.py','--config',str(cfg),"
                "'--root',str(root/'out'),'--require-handoff-manifest','--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r4=json.loads(out.read_text()); "
                "assert 'handoff_manifest' in r4['missing_before_matrix']; "
                "assert any('commands.prepare_precompressed must be a non-empty list' in ';'.join(item.get('violations', [])) for item in r4['before_matrix_blockers'] if item.get('artifact')=='handoff_manifest'); "
                "assert any('commands.prewarm must be a non-empty list' in ';'.join(item.get('violations', [])) for item in r4['before_matrix_blockers'] if item.get('artifact')=='handoff_manifest'); "
                "assert any('commands.precompressed_config_followup must be a non-empty list' in ';'.join(item.get('violations', [])) for item in r4['before_matrix_blockers'] if item.get('artifact')=='handoff_manifest'); "
                "assert any('commands.baseline_post_run_audit must be a non-empty list' in ';'.join(item.get('violations', [])) for item in r4['before_matrix_blockers'] if item.get('artifact')=='handoff_manifest'); "
                "assert any('commands.precompressed_config_post_run_audit must be a non-empty list' in ';'.join(item.get('violations', [])) for item in r4['before_matrix_blockers'] if item.get('artifact')=='handoff_manifest'); "
                "assert any('commands.manual_finalize[0] missing --require-handoff-manifest' in ';'.join(item.get('violations', [])) for item in r4['before_matrix_blockers'] if item.get('artifact')=='handoff_manifest'); "
                "handoff=json.loads((root/'out'/'edge_handoff_manifest.json').read_text()); "
                "handoff['code']['commit']='stale'; "
                "(root/'out'/'edge_handoff_manifest.json').write_text(json.dumps(handoff)); "
                "p=subprocess.run([sys.executable,'tools/audit_target_readiness.py','--config',str(cfg),"
                "'--root',str(root/'out'),'--require-handoff-manifest','--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "rh=json.loads(out.read_text()); "
                "handoff_artifact=rh['artifacts']['handoff_manifest']; "
                "assert handoff_artifact['ok'] is False; "
                "assert handoff_artifact['code_matches_current']['commit'] is False; "
                "assert any('code.commit does not match current checkout' in ';'.join(item.get('violations', [])) for item in rh['before_matrix_blockers'] if item.get('artifact')=='handoff_manifest'); "
                "p=subprocess.run([sys.executable,'tools/audit_target_readiness.py','--config',str(cfg),"
                "'--root',str(root/'out'),'--require-optional-backends-report','--require-quant-runtime-report','--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "rq=json.loads(out.read_text()); "
                "groups=rq['next_command_groups']; "
                "assert len(rq['next_commands']) == len(set(rq['next_commands'])); "
                "assert all(len(group) == len(set(group)) for group in groups.values()); "
                "assert sum(1 for cmd in groups['precompressed_config_followup'] if 'tools/audit_edge_assets.py' in cmd) == 1; "
                "precompressed_cmd=[cmd for cmd in rq['next_commands'] if 'RUN_PRECOMPRESSED_PREPARE=1' in cmd][0]; "
                "preflight_cmd=[cmd for cmd in rq['next_commands'] if 'tools/run_edge_preflight.py' in cmd][0]; "
                "matrix_quant_cmd=[cmd for cmd in rq['next_commands'] if 'tools/run_edge_matrix.py' in cmd and 'edge_matrix_with_precompressed.json' not in cmd][0]; "
                "audit_quant_cmd=[cmd for cmd in rq['next_commands'] if 'tools/audit_edge_completion.py' in cmd and 'edge_matrix_with_precompressed.json' not in cmd][0]; "
                "matrix_precompressed_cmd=[cmd for cmd in rq['next_commands'] if 'tools/run_edge_matrix.py' in cmd and 'edge_matrix_with_precompressed.json' in cmd][0]; "
                "audit_precompressed_cmd=[cmd for cmd in rq['next_commands'] if 'tools/audit_edge_completion.py' in cmd and 'edge_matrix_with_precompressed.json' in cmd][0]; "
                "assert 'edge_matrix_with_precompressed.json' in precompressed_cmd; "
                "assert 'edge_prepare_precompressed_report.json' in precompressed_cmd; "
                "assert 'PRECOMPRESSED_PREPARE_ONLY=1' in precompressed_cmd; "
                "assert 'PRECOMPRESSED_GROUPS=' in precompressed_cmd; "
                "assert 'scripts/run_edge_4090_full.sh' in precompressed_cmd; "
                "assert '--require-target-readiness-report' in matrix_precompressed_cmd; "
                "assert '--require-target-readiness-report' in audit_precompressed_cmd; "
                "assert '--require-target-paths-report' in matrix_precompressed_cmd; "
                "assert '--require-target-paths-report' in audit_precompressed_cmd; "
                "assert '--require-prepare-precompressed-report' in matrix_precompressed_cmd; "
                "assert '--require-prepare-precompressed-report' in audit_precompressed_cmd; "
                "assert any('--require-prepare-precompressed-report' in cmd for cmd in groups['precompressed_config_followup'] if 'tools/audit_target_readiness.py' in cmd); "
                "assert '--check-optional-backend-resolution' in preflight_cmd; "
                "assert '--check-quant-runtime' in preflight_cmd; "
                "assert '--require-optional-backends-report' in matrix_quant_cmd; "
                "assert '--require-quant-runtime-report' in matrix_quant_cmd; "
                "assert '--require-target-readiness-report' in matrix_quant_cmd; "
                "assert '--require-target-paths-report' in matrix_quant_cmd; "
                "assert '--require-optional-backends-report' in audit_quant_cmd; "
                "assert '--require-quant-runtime-report' in audit_quant_cmd; "
                "assert '--require-target-readiness-report' in audit_quant_cmd; "
                "assert '--require-target-paths-report' in audit_quant_cmd; "
                "assert 'RUN_PRECOMPRESSED_PREPARE=1' in groups['precompressed_prepare'][0]; "
                "assert 'PRECOMPRESSED_GROUPS=' in groups['precompressed_prepare'][0]; "
                "assert any('edge_matrix_with_precompressed.json' in cmd and 'tools/run_edge_matrix.py' in cmd for cmd in groups['precompressed_config_followup']); "
                "assert any('tools/audit_target_readiness.py' in cmd and 'edge_matrix_with_precompressed.json' not in cmd for cmd in groups['baseline_readiness']); "
                "assert any('edge_matrix_with_precompressed.json' not in cmd and 'tools/run_edge_matrix.py' in cmd for cmd in groups['baseline_finalize']); "
                "precompressed_post='\\n'.join(groups['precompressed_config_post_run_audit']); "
                "baseline_post='\\n'.join(groups['baseline_post_run_audit']); "
                "assert 'tools/audit_target_readiness.py' in precompressed_post; "
                "assert 'edge_matrix_with_precompressed.json' in precompressed_post; "
                "assert '--require-prepare-precompressed-report' in precompressed_post; "
                "assert 'tools/build_goal_audit.py' in precompressed_post; "
                "assert '--final-output-root ' in precompressed_post; "
                "assert '--portable-bundle-verify portable_bundle_verify.json' in precompressed_post; "
                "assert 'current_goal_audit.json' in precompressed_post; "
                "assert 'tools/audit_target_readiness.py' in baseline_post; "
                "assert 'edge_matrix_with_precompressed.json' not in baseline_post; "
                "assert 'tools/build_goal_audit.py' in baseline_post; "
                "assert '--final-output-root ' in baseline_post; "
                "assert '--portable-bundle-verify portable_bundle_verify.json' in baseline_post; "
                "assert 'current_goal_audit.json' in baseline_post; "
                "matrix_cmd=[cmd for cmd in r['next_commands'] if 'tools/run_edge_matrix.py' in cmd][0]; "
                "assert any('tools/build_edge_handoff_manifest.py' in cmd for cmd in r['next_commands']); "
                "assert '--require-runtime-python-prefix 3.10' in matrix_cmd; "
                "assert '--require-runtime-gpu-name-contains 4090' in matrix_cmd; "
                "assert '--require-runtime-gpu-total-memory-min-gb 23' in matrix_cmd; "
                "assert '--require-runtime-gpu-count 1' in matrix_cmd; "
                "assert '--require-runtime-cudnn' in matrix_cmd; "
                "assert '--require-runtime-torch-prefix 2.' in matrix_cmd; "
                "assert '--require-runtime-transformers' in matrix_cmd; "
                "assert '--require-runtime-bitsandbytes' in matrix_cmd; "
                "assert '--require-runtime-cpu-count-min 8' in matrix_cmd; "
                "assert '--require-runtime-system-memory-min-gb 240' in matrix_cmd; "
                "assert '--require-runtime-hf-endpoint https://hf-mirror.com' in matrix_cmd; "
                "assert '--require-target-readiness-report' in matrix_cmd; "
                "assert '--require-code-match' in matrix_cmd and '--require-clean-worktree' in matrix_cmd; "
                "assert any('--require-runtime-cuda-prefix 12.' in cmd for cmd in r['next_commands']); "
                "assert any(item['type']=='missing_artifact' and item['artifact']=='asset_audit' for item in r['before_matrix_blockers']); "
                "assert any(item['type']=='method_blocked' and item['model']=='Tiny' and item['method']=='dense' for item in r['before_matrix_blockers'])",
            ],
        ),
        (
            "target_readiness_rejects_preflight_cuda_mismatch",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_target_preflight_cuda_')); "
                "out_root=root/'out'; out_root.mkdir(parents=True); "
                "cfg=root/'config.json'; out=root/'target_readiness.json'; "
                "cfg.write_text(json.dumps({'models':[],'methods':['dense'],'defaults':{'output_root':str(out_root),'deployment_memory_limit_gb':24,'cuda_visible_devices':'0'}})); "
                "(out_root/'edge_asset_audit.json').write_text(json.dumps({'ok':True,'paths':{'hf_cache_dir':{'block_device':{'is_non_rotational':True},'can_create':True},'data_dir':{'block_device':{'is_non_rotational':True},'can_create':True},'output_root':{'block_device':{'is_non_rotational':True},'can_create':True}}})); "
                "verify_payload={'cuda_visible_devices':{'actual':'1','expected':'0','matches_config':False,'required':True}}; "
                "steps=[{'name':'asset_audit','required':True,'ok':True,'returncode':0},"
                "{'name':'verify_env','required':True,'ok':True,'returncode':0,'payload':verify_payload},"
                "{'name':'method_readiness','required':True,'ok':True,'returncode':0},"
                "{'name':'matrix_dry_run','required':True,'ok':True,'returncode':0}]; "
                "(out_root/'edge_preflight_report.json').write_text(json.dumps({'ok':True,'steps':steps})); "
                "p=subprocess.run([sys.executable,'tools/audit_target_readiness.py','--config',str(cfg),'--root',str(out_root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "assert r['ready_for_matrix'] is False; "
                "blocked=[item for item in r['before_matrix_blockers'] if item.get('artifact')=='preflight'][0]; "
                "assert blocked['type'] == 'artifact_not_ok'; "
                "assert 'preflight verify_env CUDA_VISIBLE_DEVICES did not match config' in blocked['violations']; "
                "assert blocked['verify_env_cuda_visible_devices']['matches_config'] is False",
            ],
        ),
        (
            "probe_quant_runtime_static",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('tools/probe_quant_runtime.py').read_text(); "
                "assert 'smoothquant' in text and 'omniquant' in text; "
                "assert 'runtime_imports' in text; "
                "assert 'gptqmodel_checkpoint_metadata' in text; "
                "assert 'required_imports' in text; "
                "assert 'compressed_tensors\"]' in text; "
                "assert 'quantization_metadata' in text; "
                "assert 'python_executable' in text and 'pythonpath' in text; "
                "assert 'importlib.metadata.version' in text; "
                "assert 'getattr(module, \"__file__\", None)' in text; "
                "assert 'submodule_search_locations' in text; "
                "assert 'model_load' in text; "
                "assert '--generate' in text; "
                "assert 'max_memory' in text; "
                "assert 'precompressed_path_not_configured' in text",
            ],
        ),
        ("accuracy_script", ["bash", "-n", "scripts/run_edge_accuracy.sh"]),
        ("deploy_script", ["bash", "-n", "scripts/run_edge_deploy.sh"]),
        ("resource_script", ["bash", "-n", "scripts/run_edge_resource.sh"]),
        ("pipeline_script", ["bash", "-n", "scripts/run_edge_pipeline.sh"]),
        (
            "standalone_wrapper_params",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "a=Path('scripts/run_edge_accuracy.sh').read_text(); "
                "d=Path('scripts/run_edge_deploy.sh').read_text(); "
                "r=Path('scripts/run_edge_resource.sh').read_text(); "
                "assert all('CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES:-0}\"' in text for text in (a,d,r)); "
                "assert '${TASKS:-' in a and '${BATCH_SIZE:-' in a and '${LIMIT:-' in a; "
                "assert '${INPUT_LENGTH:-' in d and '${MAX_NEW_TOKENS:-' in d and '${REPEAT:-' in d; "
                "assert '${SEQLEN:-' in r and '${SPARSITY_SAMPLE_TENSORS:-' in r; "
                "assert '${QUANT_BITS:-' in r and '${GROUP_SIZE:-' in r; "
                "assert '${DTYPE:-fp16}' in a and '${DTYPE:-fp16}' in d and '${DTYPE:-fp16}' in r; "
                "assert all('MAX_MEMORY=\"${MAX_MEMORY:-0:22GiB,cpu:120GiB}\"' in text for text in (a,d,r)); "
                "assert all('MIN_FREE_GPU_MEMORY_GB=\"${MIN_FREE_GPU_MEMORY_GB:-20}\"' in text for text in (a,d,r)); "
                "assert all('--max-memory \"${MAX_MEMORY}\"' in text for text in (a,d,r)); "
                "assert all('--min-free-gpu-memory-gb \"${MIN_FREE_GPU_MEMORY_GB}\"' in text for text in (a,d,r)); "
                "assert 'DEPLOYMENT_MEMORY_LIMIT_GB=\"${DEPLOYMENT_MEMORY_LIMIT_GB:-24}\"' in d; "
                "assert '--deployment-memory-limit-gb \"${DEPLOYMENT_MEMORY_LIMIT_GB}\"' in d; "
                "readme=Path('README.md').read_text(); "
                "assert 'Direct `python -m edge_benchmark ...` invocations default to' in readme; "
                "assert 'HF_ENDPOINT=https://hf-mirror.com' in readme; "
                "assert 'MIN_FREE_GPU_MEMORY_GB=20' in readme; "
                "assert 'max_memory=0:22GiB,cpu:120GiB' in readme",
            ],
        ),
        (
            "pipeline_dtype_from_env",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('scripts/run_edge_pipeline.sh').read_text(); "
                "assert 'CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES:-0}\"' in text; "
                "assert 'REVISION' in text and '--revision' in text; "
                "assert '--device-map \"${DEVICE_MAP:-auto}\"' in text; "
                "assert '--dtype \"${DTYPE:-fp16}\"' in text; "
                "assert 'MAX_MEMORY=\"${MAX_MEMORY:-0:22GiB,cpu:120GiB}\"' in text; "
                "assert 'MIN_FREE_GPU_MEMORY_GB=\"${MIN_FREE_GPU_MEMORY_GB:-20}\"' in text; "
                "assert 'DEPLOYMENT_MEMORY_LIMIT_GB=\"${DEPLOYMENT_MEMORY_LIMIT_GB:-24}\"' in text; "
                "assert '--max-memory \"${MAX_MEMORY}\"' in text; "
                "assert '--min-free-gpu-memory-gb \"${MIN_FREE_GPU_MEMORY_GB}\"' in text; "
                "assert '--deployment-memory-limit-gb \"${DEPLOYMENT_MEMORY_LIMIT_GB}\"' in text; "
                "assert 'NUM_FEWSHOT' in text and '--num-fewshot' in text; "
                "assert 'LIMIT' in text and '--limit' in text; "
                "assert 'PRUNE_LM_HEAD' in text and '--prune-lm-head' in text; "
                "assert 'DESC_ACT' in text and '--desc-act' in text; "
                "assert 'AWQ_VERSION' in text and '--awq-version' in text; "
                "assert 'SPARSITY_SAMPLE_TENSORS' in text and '--sparsity-sample-tensors' in text",
            ],
        ),
        (
            "readme_resource_quant_args",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('README.md').read_text(); "
                "assert '--quant-bits 4' in text; "
                "assert '--group-size 128' in text; "
                "assert 'QUANT_BITS=4' in text and 'GROUP_SIZE=128' in text",
            ],
        ),
        (
            "readme_manual_prewarm_before_required_finalize",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "handoff='python tools/build_edge_handoff_manifest.py --config configs/edge_4090_matrix.json --output-root /data/outputs/llmcbench_edge_4090 --output /data/outputs/llmcbench_edge_4090/edge_handoff_manifest.json'; "
                "preflight='python tools/run_edge_preflight.py --config configs/edge_4090_matrix.json --check-hf-access --check-model-forward --finalize-dry-run --cuda-visible-devices 0 --check-optional-backend-resolution --check-quant-runtime --output /data/outputs/llmcbench_edge_4090/edge_preflight_report.json'; "
                "prewarm='python tools/prewarm_lm_eval_tasks.py --config configs/edge_4090_matrix.json --only-model Qwen3-8B --expand-mmlu --sleep-seconds 60 --report-output /data/outputs/llmcbench_edge_4090/edge_prewarm_report.json'; "
                "matrix='python tools/run_edge_matrix.py --config configs/edge_4090_matrix.json --stop-on-error --finalize --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report --require-optional-backends-report --require-quant-runtime-report --require-target-paths-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree'; "
                "audit='python tools/audit_edge_completion.py --config configs/edge_4090_matrix.json --root /data/outputs/llmcbench_edge_4090 --require-finalize-report --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report --require-optional-backends-report --require-quant-runtime-report --require-target-paths-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree'; "
                "texts=[Path(path).read_text() for path in ['README.md','docs/COMPLETION_AUDIT.md']]; "
                "assert all(handoff in text and preflight in text and prewarm in text and matrix in text and audit in text and text.rindex(handoff) < text.rindex(preflight) < text.rindex(prewarm) < text.rindex(matrix) < text.rindex(audit) for text in texts); "
                "readme=Path('README.md').read_text(); "
                "assert 'python tools/run_edge_preflight.py --config configs/edge_4090_matrix.json --check-hf-access\\n' not in readme; "
                "assert 'python tools/run_edge_preflight.py --config configs/edge_4090_matrix.json --check-optional-backend-resolution --allow-blocked --finalize-dry-run' not in readme",
            ],
        ),
        (
            "docs_final_deliverables_include_strict_reports",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "required=['edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','target_readiness.json','edge_scorecard.json','edge_scorecard.md','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json','edge_results_bundle_verify.json','edge_delivery_verify.json']; "
                "docs={path: Path(path).read_text() for path in ['README.md','docs/COMPLETION_AUDIT.md']}; "
                "missing=[(path,name) for path,text in docs.items() for name in required if f'/data/outputs/llmcbench_edge_4090/{name}' not in text]; "
                "assert not missing, missing",
            ],
        ),
        (
            "target_configs_use_split_accuracy",
            [
                sys.executable,
                "-c",
                "import json; "
                "paths=['configs/edge_4090_matrix.json','configs/edge_local_h20_matrix.json']; "
                "configs=[json.load(open(path)) for path in paths]; "
                "assert all(cfg['defaults']['split_accuracy_tasks'] is True for cfg in configs); "
                "assert all(cfg['defaults']['continue_on_accuracy_error'] is True for cfg in configs); "
                "assert all(cfg['defaults']['revision'] is None for cfg in configs); "
                "assert all(cfg['defaults']['num_fewshot'] is None for cfg in configs); "
                "assert all(cfg['defaults']['accuracy_limit'] is None for cfg in configs); "
                "assert all(cfg['defaults']['device_map'] == 'auto' for cfg in configs); "
                "assert all(cfg['defaults']['dtype'] == 'fp16' for cfg in configs); "
                "assert all(cfg['defaults']['prune_lm_head'] is False for cfg in configs); "
                "assert all(cfg['defaults']['prune_pattern'] == 'unstructured' for cfg in configs); "
                "assert all(cfg['defaults']['desc_act'] is False for cfg in configs); "
                "assert all(cfg['defaults']['awq_version'] == 'GEMM' for cfg in configs)",
            ],
        ),
        ("full_4090_script", ["bash", "-n", "scripts/run_edge_4090_full.sh"]),
        ("prepare_4090_models_script", ["bash", "-n", "scripts/prepare_4090_models.sh"]),
        ("install_4090_environment_script", ["bash", "-n", "scripts/install_4090_environment.sh"]),
        (
            "prepare_4090_datasets_cli",
            [
                sys.executable,
                "scripts/prepare_4090_datasets.py",
                "--family",
                "piqa",
                "--cache-dir",
                "/tmp/llmcbench_dataset_cli_smoke",
                "--manifest",
                "/tmp/llmcbench_dataset_cli_smoke/manifest.json",
                "--dry-run",
            ],
        ),
        (
            "install_4090_environment_dry_run",
            ["bash", "-c", "DRY_RUN=1 bash scripts/install_4090_environment.sh"],
        ),
        (
            "matrix_finalize_prewarm_flag_assert",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('tools/run_edge_matrix.py').read_text(); "
                "assert '\"REVISION\": \"\" if defaults.get(\"revision\") is None else str(defaults.get(\"revision\"))' in text; "
                "assert '\"NUM_FEWSHOT\": \"\" if defaults.get(\"num_fewshot\") is None else str(defaults.get(\"num_fewshot\"))' in text; "
                "assert '\"LIMIT\": \"\" if defaults.get(\"accuracy_limit\") is None else str(defaults.get(\"accuracy_limit\"))' in text; "
                "assert '\"PRUNE_LM_HEAD\": \"1\" if defaults.get(\"prune_lm_head\", False) else \"0\"' in text; "
                "assert '\"DESC_ACT\": \"1\" if defaults.get(\"desc_act\", False) else \"0\"' in text; "
                "assert '\"AWQ_VERSION\": str(defaults.get(\"awq_version\", \"GEMM\"))' in text; "
                "assert '\"SPARSITY_SAMPLE_TENSORS\": str(defaults.get(\"sparsity_sample_tensors\", -1))' in text; "
                "assert '\"DEVICE_MAP\": str(defaults.get(\"device_map\", \"auto\"))' in text; "
                "assert '--require-prewarm-report' in text; "
                "assert '--require-prewarm-all-models' in text; "
                "assert '--require-asset-audit-report' in text; "
                "assert '--require-non-rotational-storage' in text; "
                "assert '--require-preflight-report' in text; "
                "assert '--require-handoff-manifest' in text; "
                "assert 'args.require_handoff_manifest' in text; "
                "assert 'edge_handoff_manifest.json' in text; "
                "assert '--require-optional-backends-report' in text; "
                "assert 'args.require_optional_backends_report' in text; "
                "assert 'edge_optional_backends_report.json' in text; "
                "assert '--require-quant-runtime-report' in text; "
                "assert 'args.require_quant_runtime_report' in text; "
                "assert 'edge_quant_runtime_report.json' in text; "
                "assert '--require-prepare-report' in text; "
                "assert '--require-prepare-metadata-report' in text; "
                "assert 'args.require_prewarm_all_models' in text; "
                "assert '--require-runtime-cuda-prefix' in text; "
                "assert 'args.require_runtime_cuda_prefix' in text; "
                "assert '--require-runtime-gpu-total-memory-min-gb' in text; "
                "assert 'args.require_runtime_gpu_total_memory_min_gb' in text; "
                "assert '--require-runtime-gpu-count' in text; "
                "assert 'args.require_runtime_gpu_count' in text; "
                "assert '--require-runtime-cudnn' in text; "
                "assert 'args.require_runtime_cudnn' in text; "
                "assert '--require-runtime-torch-prefix' in text; "
                "assert 'args.require_runtime_torch_prefix' in text; "
                "assert '--require-runtime-transformers' in text; "
                "assert 'args.require_runtime_transformers' in text; "
                "assert '--require-runtime-bitsandbytes' in text; "
                "assert 'args.require_runtime_bitsandbytes' in text; "
                "assert '--require-runtime-cpu-count-min' in text; "
                "assert 'args.require_runtime_cpu_count_min' in text; "
                "assert '--require-runtime-system-memory-min-gb' in text; "
                "assert 'args.require_runtime_system_memory_min_gb' in text",
            ],
        ),
        (
            "full_4090_final_audit_assert",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('scripts/run_edge_4090_full.sh').read_text(); "
                "assert '--require-finalize-report' in text; "
                "assert '--require-code-match' in text; "
                "assert '--require-clean-worktree' in text; "
                "assert '--require-runtime-cuda-prefix' in text; "
                "assert 'REQUIRE_RUNTIME_CUDA_PREFIX' in text; "
                "assert '--require-runtime-gpu-total-memory-min-gb' in text; "
                "assert 'REQUIRE_RUNTIME_GPU_TOTAL_MEMORY_MIN_GB' in text; "
                "assert '--require-runtime-gpu-count' in text; "
                "assert 'REQUIRE_RUNTIME_GPU_COUNT' in text; "
                "assert '--require-runtime-cudnn' in text; "
                "assert 'REQUIRE_RUNTIME_CUDNN' in text; "
                "assert '--require-runtime-torch-prefix' in text; "
                "assert 'REQUIRE_RUNTIME_TORCH_PREFIX' in text; "
                "assert '--require-runtime-transformers' in text; "
                "assert 'REQUIRE_RUNTIME_TRANSFORMERS' in text; "
                "assert '--require-runtime-bitsandbytes' in text; "
                "assert 'REQUIRE_RUNTIME_BITSANDBYTES' in text; "
                "assert '--require-runtime-cpu-count-min' in text; "
                "assert 'REQUIRE_RUNTIME_CPU_COUNT_MIN' in text; "
                "assert '--require-runtime-system-memory-min-gb' in text; "
                "assert 'REQUIRE_RUNTIME_SYSTEM_MEMORY_MIN_GB' in text; "
                "assert '--require-prewarm-report' in text; "
                "assert '--require-prewarm-all-models' in text; "
                "assert '--require-asset-audit-report' in text; "
                "assert '--require-non-rotational-storage' in text; "
                "assert '--require-preflight-report' in text; "
                "assert '--require-handoff-manifest' in text; "
                "assert '--require-optional-backends-report' in text; "
                "assert '--require-quant-runtime-report' in text; "
                "assert 'TARGET_READINESS_ARGS+=(--require-optional-backends-report)' in text; "
                "assert 'TARGET_READINESS_ARGS+=(--require-quant-runtime-report)' in text; "
                "assert 'TARGET_READINESS_ARGS+=(--require-prepare-precompressed-report)' in text; "
                "assert '--require-prepare-report' in text; "
                "assert '--require-prepare-metadata-report' in text; "
                "assert 'RUN_PRECOMPRESSED_PREPARE' in text; "
                "assert 'PRECOMPRESSED_GROUPS' in text; "
                "assert 'PRECOMPRESSED_PREPARE_ONLY' in text; "
                "assert 'PRECOMPRESSED_MODELS_SET' in text; "
                "assert 'PRECOMPRESSED_CONFIG_OUTPUT' in text; "
                "assert 'PRECOMPRESSED_REPOS' in text; "
                "assert 'PRECOMPRESSED_FORCE' in text; "
                "assert 'tools/prepare_precompressed_assets.py' in text; "
                "assert 'TARGET_READINESS_OUTPUT' in text; "
                "assert 'HANDOFF_MANIFEST_OUTPUT' in text; "
                "assert 'RUN_HANDOFF_MANIFEST' in text; "
                "assert 'tools/build_edge_handoff_manifest.py' in text; "
                "assert 'TARGET_READINESS_ARGS+=(--require-handoff-manifest)' in text; "
                "assert 'tools/audit_target_readiness.py' in text; "
                "assert '--require-prepare-precompressed-report' in text; "
                "assert 'REQUIRE_PREWARM_ALL_MODELS' in text; "
                "assert '--fail-on-blocked-methods' in text; "
                "assert 'FAIL_ON_BLOCKED_METHODS' in text; "
                "assert 'CHECK_OPTIONAL_BACKEND_RESOLUTION' in text; "
                "assert 'FAIL_ON_OPTIONAL_BACKEND_RESOLUTION' in text; "
                "assert '--check-optional-backend-resolution' in text; "
                "assert '--fail-on-optional-backend-resolution' in text; "
                "assert 'CHECK_QUANT_RUNTIME' in text; "
                "assert 'FAIL_ON_QUANT_RUNTIME' in text; "
                "assert 'QUANT_RUNTIME_GENERATE' in text; "
                "assert '--check-quant-runtime' in text; "
                "assert 'ASSET_AUDIT_OUTPUT' in text; "
                "assert 'PREFLIGHT_OUTPUT' in text; "
                "assert 'PREWARM_REPORT_OUTPUT' in text; "
                "assert 'PREPARE_OUTPUT' in text and 'PREPARE_METADATA_OUTPUT' in text; "
                "assert 'RUN_TARGET_PATH_PREPARE' in text; "
                "assert 'TARGET_PATH_PREPARE_OUTPUT' in text; "
                "assert 'tools/prepare_target_paths.py' in text; "
                "assert 'edge_target_paths_report.json' in text; "
                "assert '--require-target-paths-report' in text; "
                "assert 'edge_prepare_assets_report.json' in text; "
                "assert 'edge_prepare_metadata_report.json' in text; "
                "assert 'edge_prepare_precompressed_report.json' in text; "
                "assert 'DEFAULT_PREWARM_REPORT_ROOT' in text; "
                "assert 'DRY_RUN_OUTPUT_ROOT' in text; "
                "assert 'edge_prewarm_report.json' in text; "
                "assert 'CONFIG_OUTPUT_ROOT' in text; "
                "assert 'LINK_EXISTING_MODELS' in text and '--model-source-root' in text; "
                "assert 'tools/audit_edge_completion.py' in text; "
                "assert 'build_target_readiness_args()' in text; "
                "assert 'run_target_readiness pre_matrix' in text; "
                "assert 'run_target_readiness post_matrix' in text; "
                "assert 'check_target_ready_for_matrix' in text; "
                "assert 'FAIL_ON_TARGET_READINESS_BLOCKERS' in text; "
                "assert 'EXPORT_RESULTS_BUNDLE' in text; "
                "assert 'VERIFY_RESULTS_BUNDLE' in text; "
                "assert 'tools/export_edge_results_bundle.py' in text; "
                "assert 'verify_edge_results_bundle.py' in text; "
                "assert 'RESULTS_BUNDLE_VERIFY_OUTPUT' in text; "
                "assert '--require-target-runtime' in text; "
                "assert 'RESULTS_BUNDLE_ARCHIVE' in text; "
                "assert '--results-archive' in text; "
                "assert 'SCORECARD_EXTRA_ARGS' in text; "
                "assert text.index('run_target_readiness pre_matrix') < text.index('tools/run_edge_matrix.py'); "
                "assert text.index('run_target_readiness post_matrix') < text.index('tools/export_edge_results_bundle.py'); "
                "assert '\"${PYTHON_BIN}\" - <<' in text",
            ],
        ),
        ("init_edge_storage_script", ["bash", "-n", "scripts/init_edge_storage.sh"]),
        ("optional_backends_script", ["bash", "-n", "scripts/setup_optional_backends.sh"]),
        (
            "precompressed_prepare_report_plumbing",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "matrix=Path('tools/run_edge_matrix.py').read_text(); "
                "audit=Path('tools/audit_edge_completion.py').read_text(); "
                "render=Path('tools/render_edge_report.py').read_text(); "
                "goal=Path('tools/build_goal_audit.py').read_text(); "
                "assert 'edge_prepare_precompressed_report.json' in matrix; "
                "assert '--require-prepare-precompressed-report' in matrix; "
                "assert 'edge_prepare_precompressed_report.json' in audit; "
                "assert '--require-prepare-precompressed-report' in audit; "
                "assert '--prepare-precompressed-report' in render; "
                "assert 'prepare_precompressed_report' in goal",
            ],
        ),
        (
            "requirements_split",
            [
                sys.executable,
                "-c",
                "base=open('requirements.txt').read(); opt=open('requirements-optional.txt').read(); constraints=open('constraints-cu124.txt').read(); "
                "assert 'auto-gptq' not in base and 'gptqmodel' not in base and 'autoawq' not in base and 'bitsandbytes' not in base and 'optimum' not in base and 'compressed-tensors' not in base; "
                "assert 'torch==2.6.0' in base; "
                "assert 'auto-gptq==0.7.1' in opt and 'gptqmodel' not in opt and 'autoawq==0.2.9' in opt and 'bitsandbytes==0.50.1' in opt and 'optimum<2' in opt and 'compressed-tensors==0.15.0.1' in opt; "
                "assert 'torch==2.6.0' in constraints and 'torchvision==0.21.0' in constraints and 'torchaudio==2.6.0' in constraints and 'compressed-tensors==0.15.0.1' in constraints",
            ],
        ),
        (
            "setup_edge_env_pins_torch26",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('scripts/setup_edge_env.sh').read_text(); readme=Path('README.md').read_text(); "
                "assert 'cu124' in text; "
                "assert 'torch==2.6.0' in text; "
                "assert 'torchvision==0.21.0' in text; "
                "assert 'torchaudio==2.6.0' in text; "
                "assert 'EDGE_RUNTIME_PROFILE' in text; "
                "assert 'gptqmodel)' in text; "
                "assert 'torch==2.8.0' in text; "
                "assert 'numpy>=2.2.6 transformers>=4.55.2' in text; "
                "assert 'PYTORCH_PACKAGES' in text; "
                "assert 'original LLMCBench scripts' in readme; "
                "assert 'EDGE_RUNTIME_PROFILE=gptqmodel' in readme; "
                "assert 'Python 3.10 / CUDA 12.x environment' in readme",
            ],
        ),
        (
            "optional_backend_script_includes_optimum",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('scripts/setup_optional_backends.sh').read_text(); "
                "assert 'INSTALL_OPTIMUM' in text; "
                "assert 'OPTIMUM_PACKAGE=\"${OPTIMUM_PACKAGE:-optimum<2}\"' in text; "
                "assert 'INSTALL_GPTQMODEL' in text; "
                "assert 'INSTALL_COMPRESSED_TENSORS' in text; "
                "assert 'REQUESTED_PACKAGES+=(\"${OPTIMUM_PACKAGE}\")' in text; "
                "assert 'gptqmodel==4.0.0' in text; "
                "assert '--no-build-isolation' in text; "
                "assert 'gptqmodel_install' in text; "
                "assert 'requirements_ok' in text; "
                "assert '>=2.7.1' in text and '>=2.2.6' in text and '>=4.55.2' in text; "
                "assert 'target Python 3.10+ runtime dependency' in text; "
                "assert 'REQUESTED_PACKAGES+=(compressed-tensors)' in text; "
                "assert 'modules.append(\"optimum\")' in text; "
                "assert 'modules.append(\"gptqmodel\")' in text; "
                "assert 'modules.append(\"compressed_tensors\")' in text",
            ],
        ),
        (
            "generated_outputs_ignored",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('.gitignore').read_text(); "
                "assert 'outputs/' in text",
            ],
        ),
        (
            "prepare_dry_run",
            [
                sys.executable,
                "tools/prepare_edge_assets.py",
                "--config",
                args.config,
                "--only-model",
                "Qwen3-8B",
                "--metadata-only",
                "--dry-run",
                "--skip-lm-eval-probe",
                "--output",
                "/tmp/llmcbench_prepare_dry_run_report.json",
            ],
        ),
        (
            "prepare_dry_run_report_assert",
            [
                sys.executable,
                "-c",
                "import json; "
                "r=json.load(open('/tmp/llmcbench_prepare_dry_run_report.json')); "
                "assert r['ok'] is True; "
                "assert r['dry_run'] is True and r['metadata_only'] is True; "
                "assert r['environment']['HF_ENDPOINT'] == 'https://hf-mirror.com'; "
                "assert r['models'][0]['name'] == 'Qwen3-8B'",
            ],
        ),
        (
            "prepare_target_paths_creates_seed_files",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_target_paths_')); "
                "cfg=root/'config.json'; out=root/'target_paths_report.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'Qwen3-8B','path':str(root/'models'/'Qwen3-8B')}],"
                "'methods':['dense'],'defaults':{'hf_endpoint':'https://hf-mirror.com','hf_cache_dir':str(root/'hf_cache'),"
                "'data_dir':str(root/'datasets'),'output_root':str(root/'outputs'),'calibration_file':str(root/'calibration'/'wiki_samples.txt'),"
                "'prompt_file':str(root/'prompts'/'edge_prompts.txt')}})); "
                "p=subprocess.run([sys.executable,'tools/prepare_target_paths.py','--config',str(cfg),'--model-root',str(root/'model_root'),"
                "'--precompressed-model-root',str(root/'model_root'/'precompressed'),'--output',str(out),'--fail-on-error'],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "assert r['ok'] is True and r['dry_run'] is False; "
                "keys=['hf_cache_dir','data_dir','output_root','model_root','precompressed_model_root','model_parent:Qwen3-8B']; "
                "assert all(r['directories'][key]['ok'] is True for key in keys); "
                "assert r['files']['calibration_file']['ok'] is True and Path(r['files']['calibration_file']['path']).exists(); "
                "assert r['files']['calibration_file']['nonempty_lines'] == 128; "
                "assert r['files']['calibration_file']['required_nonempty_lines'] == 128; "
                "assert r['files']['prompt_file']['ok'] is True and Path(r['files']['prompt_file']['path']).exists(); "
                "assert '[LLMCBench bootstrap calibration sample 1/128]' in Path(r['files']['calibration_file']['path']).read_text(); "
                "assert 'edge-side large language model deployment' in Path(r['files']['prompt_file']['path']).read_text()",
            ],
        ),
        (
            "calibration_file_sample_count_gate",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; from tempfile import TemporaryDirectory; from types import SimpleNamespace; "
                "from edge_benchmark.compression import _calibration_texts; "
                "directory=TemporaryDirectory(prefix='llmcbench_calibration_gate_'); "
                "path=Path(directory.name)/'calibration.txt'; path.write_text('one\\ntwo\\n'); "
                "args=SimpleNamespace(calibration_file=str(path),calibration_samples=3); "
                "exec(\"try:\\n _calibration_texts(args)\\nexcept ValueError as exc:\\n assert 'contains 2 non-empty samples' in str(exc)\\nelse:\\n raise AssertionError('short calibration file was accepted')\"); "
                "directory.cleanup()",
            ],
        ),
        (
            "prepare_link_existing_dry_run",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_prepare_link_')); "
                "source=root/'source'; target=root/'target'/'Qwen3-8B'; "
                "ckpt=source/'Qwen3-8B'; ckpt.mkdir(parents=True); "
                "(ckpt/'config.json').write_text('{}\\n'); "
                "(ckpt/'model.safetensors').write_text('tiny\\n'); "
                "cfg=root/'config.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'Qwen3-8B','path':str(target)}],"
                "'methods':['dense'],'defaults':{'hf_endpoint':'https://hf-mirror.com','hf_cache_dir':str(root/'hf'),"
                "'data_dir':str(root/'data'),'calibration_file':str(root/'calib.txt'),'prompt_file':str(root/'prompts.txt')}})); "
                "out=root/'prepare_report.json'; "
                "p=subprocess.run([sys.executable,'tools/prepare_edge_assets.py','--config',str(cfg),"
                "'--only-model','Qwen3-8B','--model-source-root',str(source),'--link-existing-models',"
                "'--skip-lm-eval-probe','--dry-run','--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "payload=json.loads(out.read_text()); "
                "assert payload['ok'] is True; "
                "model=payload['models'][0]; "
                "assert model['linked_existing_model'] is True; "
                "assert model['source'] == str(ckpt); "
                "assert model['target'] == str(target)",
            ],
        ),
        (
            "asset_audit",
            [
                sys.executable,
                "tools/audit_edge_assets.py",
                "--config",
                "configs/edge_local_h20_matrix.json",
                "--only-model",
                "Qwen3-8B",
                "--output",
                "/tmp/llmcbench_asset_audit_smoke.json",
            ],
        ),
        (
            "asset_audit_assert",
            [
                sys.executable,
                "-c",
                "import json\n"
                "r=json.load(open('/tmp/llmcbench_asset_audit_smoke.json'))\n"
                "assert r['models'][0]['name'] == 'Qwen3-8B'\n"
                "assert r['models'][0]['checkpoint_ready'] is True\n"
                "for key in ['hf_cache_dir','data_dir','output_root']:\n"
                "    item=r['paths'][key]\n"
                "    assert 'mount' in item and 'block_device' in item\n"
                "    assert 'source' in item['mount'] and 'fstype' in item['mount']\n"
                "    assert 'can_create' in item and 'writable_probe_path' in item\n"
                "    assert 'is_non_rotational' in item['block_device']",
            ],
        ),
        (
            "asset_audit_precompressed_method_filter",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_asset_precompressed_')); "
                "pre=root/'precompressed'/'Qwen3-8B-gptq'; pre.mkdir(parents=True); "
                "(pre/'config.json').write_text('{}\\n'); "
                "(pre/'model.safetensors').write_text('tiny\\n'); "
                "(root/'calib.txt').write_text('calib\\n'); "
                "(root/'prompts.txt').write_text('prompt\\n'); "
                "cfg=root/'config.json'; out=root/'audit.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'Qwen3-8B','path':str(root/'missing-base')}],"
                "'methods':['dense','gptq'],'defaults':{'hf_cache_dir':str(root/'hf'),'data_dir':str(root/'data'),"
                "'output_root':str(root/'out'),'calibration_file':str(root/'calib.txt'),'prompt_file':str(root/'prompts.txt')},"
                "'precompressed_paths':{'Qwen3-8B':{'gptq':str(pre)}}})); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_assets.py','--config',str(cfg),"
                "'--only-model','Qwen3-8B','--only-method','gptq','--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "report=json.loads(out.read_text()); "
                "assert report['ok'] is True; "
                "assert report['selected_methods'] == ['gptq']; "
                "assert report['models'][0]['required'] is False; "
                "assert report['precompressed'][0]['checkpoint_ready'] is True; "
                "ready=root/'readiness.json'; "
                "p=subprocess.run([sys.executable,'tools/check_method_readiness.py','--config',str(cfg),"
                "'--only-model','Qwen3-8B','--only-method','gptq','--output',str(ready)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "readiness=json.loads(ready.read_text()); "
                "row=readiness['rows'][0]; "
                "assert readiness['ok'] == row['ok']; "
                "assert row['uses_precompressed'] is True; "
                "assert row['install_hint'] is None; "
                "assert row['imports'] == []; "
                "assert row['compression_imports'] == []; "
                "assert row['runtime_requirements_reason'] == 'default'; "
                "assert row['ok'] is True or any(item.startswith('missing_runtime_import:') for item in row['blockers']); "
                "cfg_payload=json.loads((pre/'config.json').read_text()); "
                "cfg_payload['quantization_config']={'quant_method':'gptq','bits':4,'group_size':128,'meta':{'quantizer':['gptqmodel:4.0.0']}}; "
                "(pre/'config.json').write_text(json.dumps(cfg_payload)); "
                "p=subprocess.run([sys.executable,'tools/check_method_readiness.py','--config',str(cfg),"
                "'--only-model','Qwen3-8B','--only-method','gptq','--output',str(ready)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "readiness=json.loads(ready.read_text()); row=readiness['rows'][0]; "
                "assert row['runtime_requirements_reason'] == 'gptqmodel_checkpoint_metadata'; "
                "assert row['quantization_metadata']['quantizer'] == ['gptqmodel:4.0.0']; "
                "assert any(item['alternatives'] == ['gptqmodel'] for item in row['runtime_imports']); "
                "assert row['ok'] is False; "
                "assert 'missing_runtime_import:any_of:gptqmodel' in row['blockers']",
            ],
        ),
        (
            "prepare_precompressed_assets_dry_run",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_precompressed_prepare_')); "
                "cfg=root/'config.json'; out=root/'report.json'; out_cfg=root/'with_precompressed.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'Qwen3-8B','path':str(root/'qwen')}],"
                "'methods':['gptq','awq'],'defaults':{'hf_endpoint':'https://hf-mirror.com'}})); "
                "p=subprocess.run([sys.executable,'tools/prepare_precompressed_assets.py','--config',str(cfg),"
                "'--local-root',str(root/'precompressed'),'--output-config',str(out_cfg),'--output',str(out),"
                "'--only-model','Qwen3-8B','--only-method','gptq','--dry-run','--write-dry-run-config'],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "assert r['ok'] is True; "
                "assert r['dry_run'] is True; "
                "assert r['items'][0]['repo'] == 'kaitchup/Qwen3-8B-autoround-4bit-gptq'; "
                "assert r['items'][0]['download']['dry_run'] is True; "
                "assert r['updated_config'] == str(out_cfg); "
                "assert out_cfg.exists(); "
                "updated=json.loads(out_cfg.read_text()); "
                "assert updated['precompressed_paths']['Qwen3-8B']['gptq'] == str(root/'precompressed'/'Qwen3-8B-gptq')",
            ],
        ),
        (
            "prepare_precompressed_assets_provenance_mismatch",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_precompressed_provenance_')); "
                "cfg=root/'config.json'; out=root/'report.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'Qwen3-8B','path':str(root/'qwen')}],"
                "'methods':['gptq'],'defaults':{'hf_endpoint':'https://hf-mirror.com'}})); "
                "ckpt=root/'precompressed'/'Qwen3-8B-gptq'; ckpt.mkdir(parents=True); "
                "(ckpt/'config.json').write_text(json.dumps({'model_type':'qwen3'})); "
                "(ckpt/'model.safetensors').write_bytes(b'x'); "
                "(ckpt/'.llmcbench_precompressed_source.json').write_text(json.dumps({'schema_version':1,'repo':'JunHowie/Qwen3-8B-GPTQ-Int4'})); "
                "p=subprocess.run([sys.executable,'tools/prepare_precompressed_assets.py','--config',str(cfg),"
                "'--local-root',str(root/'precompressed'),'--output',str(out),'--only-model','Qwen3-8B','--only-method','gptq'],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); item=r['items'][0]; "
                "assert r['ok'] is False; "
                "assert item['download']['reason'] == 'checkpoint_source_mismatch'; "
                "assert item['download']['expected_repo'] == 'kaitchup/Qwen3-8B-autoround-4bit-gptq'; "
                "assert item['download']['existing_repo'] == 'JunHowie/Qwen3-8B-GPTQ-Int4'; "
                "assert '--force' in item['download']['remediation']; "
                "p=subprocess.run([sys.executable,'tools/prepare_precompressed_assets.py','--config',str(cfg),"
                "'--local-root',str(root/'precompressed'),'--output',str(out),'--only-model','Qwen3-8B','--only-method','gptq','--dry-run','--force'],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "assert r['ok'] is True; "
                "assert r['items'][0]['download']['dry_run'] is True",
            ],
        ),
        (
            "prepare_precompressed_assets_stale_qwen_gptq_metadata",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_precompressed_stale_qwen_')); "
                "cfg=root/'config.json'; out=root/'report.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'Qwen3-8B','path':str(root/'qwen')}],"
                "'methods':['gptq'],'defaults':{'hf_endpoint':'https://hf-mirror.com'}})); "
                "ckpt=root/'precompressed'/'Qwen3-8B-gptq'; ckpt.mkdir(parents=True); "
                "q={'quant_method':'gptq','bits':4,'group_size':128,'meta':{'quantizer':['gptqmodel:4.0.0']}}; "
                "(ckpt/'config.json').write_text(json.dumps({'model_type':'qwen3','quantization_config':q})); "
                "(ckpt/'model.safetensors').write_bytes(b'x'); "
                "p=subprocess.run([sys.executable,'tools/prepare_precompressed_assets.py','--config',str(cfg),"
                "'--local-root',str(root/'precompressed'),'--output',str(out),'--only-model','Qwen3-8B','--only-method','gptq'],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); item=r['items'][0]; "
                "assert r['ok'] is False; "
                "assert item['download']['reason'] == 'checkpoint_metadata_mismatch'; "
                "assert item['download']['quantization_metadata']['quantizer'] == ['gptqmodel:4.0.0']; "
                "assert item['download']['expected_repo'] == 'kaitchup/Qwen3-8B-autoround-4bit-gptq'; "
                "assert '--force' in item['download']['remediation']",
            ],
        ),
        (
            "prepare_precompressed_assets_smooth_omni_dry_run",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_precompressed_prepare_ext_')); "
                "cfg=root/'config.json'; out=root/'report.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'Qwen3-8B','path':str(root/'qwen')}],"
                "'methods':['smoothquant','omniquant'],'defaults':{'hf_endpoint':'https://hf-mirror.com'}})); "
                "p=subprocess.run([sys.executable,'tools/prepare_precompressed_assets.py','--config',str(cfg),"
                "'--local-root',str(root/'precompressed'),'--output',str(out),'--only-model','Qwen3-8B',"
                "'--dry-run'],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "repos={item['method']: item['repo'] for item in r['items']}; "
                "assert repos['smoothquant'] == 'Lotaliz/Qwen3-8B-smoothquant-w8a8'; "
                "assert repos['omniquant'] == 'yw223/Qwen3-8B-OmniQuant-4bit'",
            ],
        ),
        (
            "prepare_precompressed_assets_llama_dry_run",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_precompressed_prepare_llama_')); "
                "cfg=root/'config.json'; out=root/'report.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'LLaMA3-8B','path':str(root/'llama')}],"
                "'methods':['gptq','awq','smoothquant','omniquant'],'defaults':{'hf_endpoint':'https://hf-mirror.com'}})); "
                "p=subprocess.run([sys.executable,'tools/prepare_precompressed_assets.py','--config',str(cfg),"
                "'--local-root',str(root/'precompressed'),'--output',str(out),'--only-model','LLaMA3-8B',"
                "'--dry-run'],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "assert r['ok'] is False; "
                "assert len(r['blockers']) == 1; "
                "repos={item['method']: item for item in r['items']}; "
                "assert repos['gptq']['repo'] == 'Ksgk-fy/Meta-Llama-3-8B-4bit-GPTQ'; "
                "assert repos['awq']['repo'] == 'solidrust/Meta-Llama-3-8B-hf-AWQ'; "
                "assert repos['smoothquant']['repo'] == 'HRuiii/Meta-Llama-3-8B-smoothquant-w8a8'; "
                "assert repos['omniquant']['skipped'] is True and repos['omniquant']['blocking'] is True and repos['omniquant']['reason'] == 'repo_not_configured'; "
                "assert repos['omniquant']['repo_status'] == 'no_verified_public_default_repo'; "
                "assert '--repo LLaMA3-8B:omniquant=<repo_id>' in repos['omniquant']['remediation']; "
                "p2=subprocess.run([sys.executable,'tools/prepare_precompressed_assets.py','--config',str(cfg),"
                "'--local-root',str(root/'precompressed'),'--output',str(out),'--only-model','LLaMA3-8B',"
                "'--only-method','omniquant','--dry-run','--fail-on-error'],capture_output=True,text=True); "
                "assert p2.returncode != 0",
            ],
        ),
        (
            "external_method_config_dry_run",
            [
                sys.executable,
                "tools/configure_external_methods.py",
                "--config",
                args.config,
                "--smoothquant-command",
                "python smooth.py --model {model_path} --out {output_dir} --samples {calibration_samples} --batch {calibration_batch_size} --max-memory {max_memory} --attn {attn_implementation}",
                "--omniquant-command",
                "python omni.py --model {model_path} --out {output_dir} --wbits {quant_bits} --group {group_size} --sparsity {sparsity} --cache {cache_dir}",
                "--dry-run",
                "--fail-on-missing",
                "--output",
                "/tmp/llmcbench_external_methods_smoke.json",
            ],
        ),
        (
            "matrix_dry_run",
            [
                sys.executable,
                "tools/run_edge_matrix.py",
                "--config",
                args.config,
                "--only-model",
                "Qwen3-8B",
                "--only-method",
                "gptq",
                "--dry-run",
                "--output-root",
                "/tmp/llmcbench_matrix_dry_smoke",
            ],
        ),
        (
            "matrix_dry_run_log_assert",
            [
                sys.executable,
                "-c",
                "import json; r=json.load(open('/tmp/llmcbench_matrix_dry_smoke/matrix_run_report.json')); "
                "assert r['jobs'] and r['jobs'][0]['log'].endswith('/job.log'); "
                "assert r['resolved_config'].endswith('/matrix_config_resolved.json'); "
                "cfg=json.load(open(r['resolved_config'])); "
                "assert 'commit' in cfg['code']",
            ],
        ),
        (
            "matrix_finalize_dry_run_step_count",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/run_edge_matrix.py','--config','configs/edge_4090_matrix.json','--only-model','Qwen3-8B','--only-method','dense','--dry-run','--finalize','--require-code-match','--require-clean-worktree','--require-prewarm-report','--require-prewarm-all-models','--require-handoff-manifest','--require-optional-backends-report','--require-quant-runtime-report','--require-asset-audit-report','--require-non-rotational-storage','--require-runtime-python-prefix','3.10','--require-runtime-gpu-name-contains','4090','--require-runtime-cuda-prefix','12.','--require-runtime-gpu-total-memory-min-gb','23','--require-runtime-gpu-count','1','--require-runtime-cudnn','--require-runtime-torch-prefix','2.','--require-runtime-transformers','--require-runtime-bitsandbytes','--require-runtime-cpu-count-min','8','--require-runtime-system-memory-min-gb','240','--require-runtime-hf-endpoint','https://hf-mirror.com','--output-root','/tmp/llmcbench_finalize_step_count'],capture_output=True,text=True); "
                "assert p.returncode == 0; marker='\\n{\\n  \"ok\": true,'; "
                "payload=json.loads(p.stdout[p.stdout.rfind(marker)+1:]); "
                "assert len(payload['steps']) == 4; "
                "assert all(step['status'] == 'dry_run' for step in payload['steps']); "
                "saved=json.load(open('/tmp/llmcbench_finalize_step_count/matrix_finalize_report.json')); "
                "assert saved['steps'] == payload['steps']; "
                "assert '--require-code-match' in payload['steps'][3]['cmd']; "
                "assert '--require-clean-worktree' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-python-prefix' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-gpu-name-contains' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-cuda-prefix' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-gpu-total-memory-min-gb' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-gpu-count' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-cudnn' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-torch-prefix' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-transformers' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-bitsandbytes' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-cpu-count-min' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-system-memory-min-gb' in payload['steps'][3]['cmd']; "
                "assert '--require-runtime-hf-endpoint' in payload['steps'][3]['cmd']; "
                "assert '--require-prewarm-report' in payload['steps'][3]['cmd']; "
                "assert '--require-prewarm-all-models' in payload['steps'][3]['cmd']; "
                "assert '--require-non-rotational-storage' in payload['steps'][3]['cmd']; "
                "assert '--require-handoff-manifest' in payload['steps'][3]['cmd']; "
                "assert '--require-optional-backends-report' in payload['steps'][3]['cmd']; "
                "assert '--require-quant-runtime-report' in payload['steps'][3]['cmd']; "
                "assert payload['edge_handoff_manifest'].endswith('/edge_handoff_manifest.json'); "
                "assert payload['edge_optional_backends_report'].endswith('/edge_optional_backends_report.json'); "
                "assert payload['edge_quant_runtime_report'].endswith('/edge_quant_runtime_report.json')",
            ],
        ),
        (
            "matrix_rejects_incomplete_existing_job",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                f"p=subprocess.run([sys.executable,'tools/run_edge_matrix.py','--config',{incomplete_existing_fixture['config']!r}],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                f"r=json.load(open({str(Path(incomplete_existing_fixture['output_root']) / 'matrix_run_report.json')!r})); "
                "assert r['failed'][0]['reason'] == 'incomplete_existing'; "
                "assert any(path.endswith('compression.json') for path in r['failed'][0]['missing_artifacts'])",
            ],
        ),
        (
            "matrix_resume_plan_incomplete_existing",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                f"out={str(Path(incomplete_existing_fixture['output_root']) / 'matrix_resume_plan.json')!r}; "
                f"p=subprocess.run([sys.executable,'tools/plan_edge_matrix_resume.py','--config',{incomplete_existing_fixture['config']!r},'--output',out],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.load(open(out)); "
                "assert r['expected_job_count'] == 1; "
                "assert r['needs_run_count'] == 1; "
                "assert r['status_counts']['incomplete_existing'] == 1; "
                "job=r['jobs'][0]; "
                "assert job['force_required'] is True; "
                "assert 'compression.json' in job['missing_artifacts']; "
                "cmd=r['command_groups']['rerun_incomplete_or_failed'][0]; "
                "assert '--only-model TinyModel' in cmd; "
                "assert '--only-method dense' in cmd; "
                "assert '--force' in cmd; "
                "assert 'finalize_after_resume' in r['command_groups']",
            ],
        ),
        (
            "matrix_finalize_readiness_requires_precompressed_report",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/run_edge_matrix.py','--config','configs/edge_4090_matrix.json',"
                "'--only-model','Qwen3-8B','--only-method','dense','--dry-run','--finalize',"
                "'--require-target-readiness-report','--require-prepare-precompressed-report',"
                "'--require-prewarm-report','--require-handoff-manifest','--require-optional-backends-report',"
                "'--require-quant-runtime-report','--output-root','/tmp/llmcbench_finalize_precompressed_gate'],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; marker='\\n{\\n  \"ok\": true,'; "
                "payload=json.loads(p.stdout[p.stdout.rfind(marker)+1:]); "
                "readiness=[step for step in payload['steps'] if 'tools/audit_target_readiness.py' in step['cmd']][0]['cmd']; "
                "final_audit=payload['steps'][-1]['cmd']; "
                "assert '--require-prepare-precompressed-report' in readiness; "
                "assert '--require-prepare-precompressed-report' in final_audit; "
                "assert '--require-target-readiness-report' in final_audit",
            ],
        ),
        (
            "matrix_precompressed_dry_run",
            [
                sys.executable,
                "tools/run_edge_matrix.py",
                "--config",
                precompressed_fixture["config"],
                "--dry-run",
            ],
        ),
        (
            "matrix_precompressed_assert",
            [
                sys.executable,
                "-c",
                "import json; r=json.load(open('/tmp/llmcbench_precompressed_matrix_smoke/outputs/matrix_run_report.json')); "
                "job=[item for item in r['jobs'] if item['method']=='dense'][0]; "
                "sq=[item for item in r['jobs'] if item['method']=='smoothquant'][0]; "
                "oq=[item for item in r['jobs'] if item['method']=='omniquant'][0]; "
                "assert job['precompressed_path'].endswith('/precompressed_tiny'); "
                "assert sq['precompressed_path'].endswith('/precompressed_tiny'); "
                "assert oq['external_command'].startswith('python omni.py'); "
                "assert not r['skipped_external_not_configured']; "
                "assert job['job_overrides']['seqlen'] == 16; "
                "assert job['job_overrides']['batch_size'] == '2'; "
                "assert job['job_overrides']['input_length'] == 16; "
                "assert job['job_overrides']['max_new_tokens'] == 2",
            ],
        ),
        (
            "asset_audit_precompressed",
            [
                sys.executable,
                "tools/audit_edge_assets.py",
                "--config",
                precompressed_fixture["config"],
                "--only-model",
                "TinyModel",
                "--output",
                "/tmp/llmcbench_precompressed_matrix_smoke/asset_audit.json",
            ],
        ),
        (
            "asset_audit_precompressed_assert",
            [
                sys.executable,
                "-c",
                "import json; r=json.load(open('/tmp/llmcbench_precompressed_matrix_smoke/asset_audit.json')); "
                "assert len(r['precompressed']) == 2; "
                "assert all(item['checkpoint_ready'] for item in r['precompressed']); "
                "assert {item['method'] for item in r['precompressed']} == {'dense', 'smoothquant'}",
            ],
        ),
        (
            "method_readiness_precompressed_external",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys; "
                f"p=subprocess.run([sys.executable,'tools/check_method_readiness.py','--config',{precompressed_fixture['config']!r},"
                "'--only-model','TinyModel','--only-method','smoothquant'],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(p.stdout); row=r['rows'][0]; "
                "assert row['uses_precompressed'] is True; "
                "assert row['model_weights_exist'] is True; "
                "runtime=row['runtime_imports'][0]; "
                "expected=('target_python_3_10_required_for:compressed_tensors' if sys.version_info < (3, 10) else 'missing_runtime_import:any_of:compressed_tensors'); "
                "assert (runtime['ok'] and r['ok'] is True and row['blockers'] == []) or "
                "((not runtime['ok']) and r['ok'] is False and expected in row['blockers'])",
            ],
        ),
        (
            "method_readiness_quant_runtime_imports",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('tools/check_method_readiness.py').read_text(); "
                "assert 'RUNTIME_IMPORTS' in text; "
                "assert 'optimum' in text and 'gptqmodel' in text; "
                "assert 'compressed_tensors' in text; "
                "assert 'target_python_3_10_required_for:compressed_tensors' in text; "
                "assert 'missing_runtime_import:any_of' in text; "
                "assert 'compression_imports' in text and 'runtime_imports' in text",
            ],
        ),
        (
            "method_readiness",
            [
                sys.executable,
                "tools/check_method_readiness.py",
                "--config",
                "configs/edge_local_h20_matrix.json",
                "--only-model",
                "Qwen3-8B",
                "--only-method",
                "dense",
            ],
        ),
        (
            "method_readiness_filter_assert",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/check_method_readiness.py','--config','configs/edge_local_h20_matrix.json','--only-model','Qwen3-8B','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode == 0; r=json.loads(p.stdout); "
                "assert list(r['summary']) == ['dense']; "
                "assert len(r['rows']) == 1 and r['rows'][0]['model'] == 'Qwen3-8B'",
            ],
        ),
        (
            "prewarm_dry_run",
            [
                sys.executable,
                "tools/prewarm_lm_eval_tasks.py",
                "--config",
                args.config,
                "--only-model",
                "Qwen3-8B",
                "--attn-implementation",
                "eager",
                "--tasks",
                "piqa",
                "--expand-mmlu",
                "--dry-run",
                "--sleep-seconds",
                "0",
                "--report-output",
                "/tmp/llmcbench_prewarm_dry_run_report.json",
            ],
        ),
        (
            "prewarm_dry_run_report_assert",
            [
                sys.executable,
                "-c",
                "import json; r=json.load(open('/tmp/llmcbench_prewarm_dry_run_report.json')); "
                f"cfg=json.load(open({args.config!r})); expected=cfg.get('defaults',{{}}).get('max_memory') or None; "
                "assert r['ok'] is True; "
                "assert r['dry_run'] is True; "
                "assert r['tasks'] == ['piqa']; "
                "assert r['dtype'] == 'fp16'; "
                "assert r['max_memory'] == expected; "
                "assert len(r['attempted']) == 1; "
                "assert r['attempted'][0]['task'] == 'piqa'; "
                "assert r['attempted'][0]['dtype'] == 'fp16'; "
                "assert r['attempted'][0]['max_memory'] == expected; "
                "assert r['attempted'][0]['attn_implementation'] == 'eager'",
            ],
        ),
        (
            "prewarm_mmlu_expand_report",
            [
                sys.executable,
                "tools/prewarm_lm_eval_tasks.py",
                "--config",
                args.config,
                "--only-model",
                "Qwen3-8B",
                "--tasks",
                "mmlu",
                "--expand-mmlu",
                "--dry-run",
                "--sleep-seconds",
                "0",
                "--report-output",
                "/tmp/llmcbench_prewarm_mmlu_expand_report.json",
            ],
        ),
        (
            "prewarm_mmlu_expand_report_assert",
            [
                sys.executable,
                "-c",
                "import json; r=json.load(open('/tmp/llmcbench_prewarm_mmlu_expand_report.json')); "
                f"cfg=json.load(open({args.config!r})); expected=cfg.get('defaults',{{}}).get('max_memory') or None; "
                "assert r['ok'] is True; "
                "assert r['dtype'] == 'fp16'; "
                "assert r['max_memory'] == expected; "
                "assert r['tasks'][0] == 'mmlu_abstract_algebra'; "
                "assert len(r['tasks']) >= 50; "
                "assert len(r['attempted']) == len(r['tasks'])",
            ],
        ),
        (
            "edge_preflight",
            [
                sys.executable,
                "tools/run_edge_preflight.py",
                "--config",
                args.config,
                "--only-model",
                "Qwen3-8B",
                "--only-method",
                "dense",
                "--prewarm-model",
                "Qwen3-8B",
                "--prewarm-tasks",
                "piqa",
                "--allow-blocked",
                "--finalize-dry-run",
                "--output",
                "/tmp/llmcbench_edge_preflight_smoke.json",
            ],
        ),
        (
            "edge_preflight_dtype_assert",
            [
                sys.executable,
                "-c",
                "import json; r=json.load(open('/tmp/llmcbench_edge_preflight_smoke.json')); "
                "assert r['resolved_forward_dtype'] == 'fp16'; "
                "assert r['resolved_forward_attn_implementation'] is None",
            ],
        ),
        (
            "edge_preflight_optional_backend_flags",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('tools/run_edge_preflight.py').read_text(); "
                "assert '--check-optional-backend-resolution' in text; "
                "assert '--fail-on-optional-backend-resolution' in text; "
                "assert 'edge_optional_backends_report.json' in text; "
                "assert 'OPTIONAL_BACKEND_DRY_RUN' in text",
            ],
        ),
        (
            "edge_preflight_quant_runtime_flags",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('tools/run_edge_preflight.py').read_text(); "
                "assert '--check-quant-runtime' in text; "
                "assert '--fail-on-quant-runtime' in text; "
                "assert '--quant-runtime-generate' in text; "
                "assert 'edge_quant_runtime_report.json' in text; "
                "assert 'tools/probe_quant_runtime.py' in text",
            ],
        ),
        (
            "edge_preflight_cuda_visible_devices_gate",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('tools/run_edge_preflight.py').read_text(); "
                "assert '--require-cuda-visible-devices-match' in text; "
                "assert 'CUDA_VISIBLE_DEVICES' in text; "
                "assert 'args.cuda_visible_devices' in text",
            ],
        ),
        (
            "verify_env_forward_dtype_from_config",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('tools/verify_edge_env.py').read_text(); "
                "assert \"args.forward_dtype or defaults.get(\\\"dtype\\\") or \\\"fp16\\\"\" in text; "
                "assert 'args.forward_attn_implementation' in text; "
                "assert '--require-cuda-visible-devices-match' in text; "
                "assert 'matches_config' in text and 'CUDA_VISIBLE_DEVICES' in text",
            ],
        ),
        (
            "full_4090_dry_run",
            [
                "bash",
                "-c",
                "DRY_RUN=1 RUN_PREPARE=0 RUN_METADATA_PREPARE=0 CHECK_HF_ACCESS=0 CHECK_MODEL_FORWARD=0 RUN_PREWARM=0 CONFIG=configs/edge_4090_matrix.json bash scripts/run_edge_4090_full.sh",
            ],
        ),
        (
            "full_4090_dry_run_reports_under_dry_root",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; root=Path('/tmp/llmcbench_edge_4090_dryrun'); "
                "assert (root/'edge_asset_audit.json').exists(); "
                "assert (root/'edge_preflight_report.json').exists(); "
                "assert (root/'edge_handoff_manifest.json').exists(); "
                "assert (root/'target_readiness.json').exists(); "
                "assert not (root/'edge_results_bundle').exists(); "
                "assert not (root/'edge_results_bundle.tar.gz').exists()",
            ],
        ),
        (
            "full_4090_dry_run_prepare_report",
            [
                "bash",
                "-c",
                "rm -rf /tmp/llmcbench_edge_4090_prepare_dryrun; "
                "DRY_RUN=1 RUN_PREPARE=0 RUN_METADATA_PREPARE=1 CHECK_HF_ACCESS=0 CHECK_MODEL_FORWARD=0 RUN_PREWARM=0 DRY_RUN_OUTPUT_ROOT=/tmp/llmcbench_edge_4090_prepare_dryrun CONFIG=configs/edge_4090_matrix.json bash scripts/run_edge_4090_full.sh >/tmp/llmcbench_edge_4090_prepare_dryrun.log && "
                "python3 - <<'PY'\n"
                "import json\n"
                "from pathlib import Path\n"
                "root=Path('/tmp/llmcbench_edge_4090_prepare_dryrun')\n"
                "report=root/'edge_prepare_metadata_report.json'\n"
                "assert report.exists(), report\n"
                "data=json.loads(report.read_text())\n"
                "assert data['ok'] is True\n"
                "assert data['dry_run'] is True\n"
                "assert data['metadata_only'] is True\n"
                "assert data['environment']['HF_ENDPOINT'] == 'https://hf-mirror.com'\n"
                "manifest=json.loads((root/'edge_handoff_manifest.json').read_text())\n"
                "assert manifest['paths']['output_root'] == str(root)\n"
                "assert manifest['config']['path'] == 'configs/edge_4090_matrix.json'\n"
                "assert manifest['target_environment']['memory_controls']['deployment_memory_limit_gb'] == 24\n"
                "assert manifest['target_environment']['memory_controls']['deployment_batch_size'] == 1\n"
                "PY",
            ],
        ),
        (
            "full_4090_precompressed_default_groups_prepare_only",
            [
                "bash",
                "-c",
                "rm -rf /tmp/llmcbench_edge_4090_precompressed_groups; "
                "DRY_RUN=1 RUN_PRECOMPRESSED_PREPARE=1 PRECOMPRESSED_PREPARE_ONLY=1 "
                "RUN_TARGET_PATH_PREPARE=0 RUN_HANDOFF_MANIFEST=0 RUN_PREWARM=0 "
                "CHECK_HF_ACCESS=0 CHECK_MODEL_FORWARD=0 "
                "DRY_RUN_OUTPUT_ROOT=/tmp/llmcbench_edge_4090_precompressed_groups "
                "PRECOMPRESSED_LOCAL_ROOT=/tmp/llmcbench_edge_4090_precompressed_groups/precompressed "
                "CONFIG=configs/edge_4090_matrix.json bash scripts/run_edge_4090_full.sh >/tmp/llmcbench_edge_4090_precompressed_groups.log && "
                "python3 - <<'PY'\n"
                "import json\n"
                "from pathlib import Path\n"
                "root=Path('/tmp/llmcbench_edge_4090_precompressed_groups')\n"
                "report=json.loads((root/'edge_prepare_precompressed_report.json').read_text())\n"
                "cfg=json.loads((root/'edge_matrix_with_precompressed.json').read_text())\n"
                "assert report['ok'] is True and report['combined'] is True\n"
                "assert len(report['reports']) == 2\n"
                "assert cfg['precompressed_paths']['LLaMA3-8B']['gptq'] == str(root/'precompressed'/'LLaMA3-8B-gptq')\n"
                "assert cfg['precompressed_paths']['LLaMA3-8B']['awq'] == str(root/'precompressed'/'LLaMA3-8B-awq')\n"
                "assert cfg['precompressed_paths']['LLaMA3-8B']['smoothquant'] == str(root/'precompressed'/'LLaMA3-8B-smoothquant')\n"
                "assert 'omniquant' not in cfg['precompressed_paths']['LLaMA3-8B']\n"
                "assert cfg['precompressed_paths']['Qwen3-8B']['omniquant'] == str(root/'precompressed'/'Qwen3-8B-omniquant')\n"
                "PY",
            ],
        ),
        (
            "full_4090_dry_run_precompressed_config",
            [
                "bash",
                "-c",
                "rm -rf /tmp/llmcbench_edge_4090_precompressed_dryrun; "
                "DRY_RUN=1 RUN_PRECOMPRESSED_PREPARE=1 PRECOMPRESSED_MODELS=LLaMA3-8B "
                "PRECOMPRESSED_METHODS=omniquant PRECOMPRESSED_METADATA_ONLY=1 "
                "PRECOMPRESSED_REPOS='LLaMA3-8B:omniquant=compressa-ai/Saiga-Llama-3-8B-OmniQuant' "
                "CHECK_HF_ACCESS=0 CHECK_MODEL_FORWARD=0 RUN_PREWARM=0 "
                "DRY_RUN_OUTPUT_ROOT=/tmp/llmcbench_edge_4090_precompressed_dryrun "
                "CONFIG=configs/edge_4090_matrix.json bash scripts/run_edge_4090_full.sh >/tmp/llmcbench_edge_4090_precompressed_dryrun.log && "
                "python3 - <<'PY'\n"
                "import json\n"
                "from pathlib import Path\n"
                "root=Path('/tmp/llmcbench_edge_4090_precompressed_dryrun')\n"
                "cfg=json.loads((root/'edge_matrix_with_precompressed.json').read_text())\n"
                "assert cfg['defaults']['output_root'] == '/data/outputs/llmcbench_edge_4090'\n"
                "path=cfg['precompressed_paths']['LLaMA3-8B']['omniquant']\n"
                "assert path == '/data/models/precompressed/LLaMA3-8B-omniquant'\n"
                "prepare=json.loads((root/'edge_prepare_precompressed_report.json').read_text())\n"
                "assert prepare['items'][0]['repo'] == 'compressa-ai/Saiga-Llama-3-8B-OmniQuant'\n"
                "manifest=json.loads((root/'edge_handoff_manifest.json').read_text())\n"
                "assert manifest['paths']['output_root'] == str(root)\n"
                "assert manifest['config']['path'] == str(root/'edge_matrix_with_precompressed.json')\n"
                "finalize=json.loads((root/'matrix_finalize_report.json').read_text())\n"
                "assert finalize['output_root'] == str(root)\n"
                "assert finalize['resolved_config'] == str(root/'matrix_config_resolved.json')\n"
                "assert all(step.get('status') == 'dry_run' for step in finalize['steps'])\n"
                "run=json.loads((root/'matrix_run_report.json').read_text())\n"
                "assert run['resolved_config'] == str(root/'matrix_config_resolved.json')\n"
                "job=[item for item in run['jobs'] if item['model']=='LLaMA3-8B' and item['method']=='omniquant'][0]\n"
                "assert job['status'] == 'dry_run'\n"
                "assert job['precompressed_path'] == path\n"
                "assert not any(item['model']=='LLaMA3-8B' and item['method']=='omniquant' for item in run['skipped_external_not_configured'])\n"
                "PY",
            ],
        ),
        (
            "goal_audit_build",
            [
                sys.executable,
                "tools/build_goal_audit.py",
                "--output",
                "/tmp/llmcbench_goal_audit_smoke.json",
            ],
        ),
        (
            "goal_audit_validation_schema_assert",
            [
                sys.executable,
                "-c",
                "import json; r=json.load(open('/tmp/llmcbench_goal_audit_smoke.json')); "
                "v=r['qwen3']['dense_partial_validation']; "
                "assert 'type_violations' in v; "
                "assert 'range_violations' in v; "
                "assert 'consistency_violations' in v; "
                "assert 'format_errors' in v; "
                "assert 'required_score_metric_fields' in v; "
                "checklist=r['requirement_checklist']; "
                "assert {'accuracy_tool','deployment_tool','resource_tool','method_coverage','target_platform'} <= set(checklist); "
                "prompt_checklist=r['prompt_to_artifact_checklist']; "
                "assert {'accuracy_tool','deployment_tool','resource_tool','method_and_model_coverage','portable_environment_and_handoff','final_two_model_results'} <= set(prompt_checklist); "
                "state=r['completion_state']; "
                "assert state['status'] == 'incomplete'; "
                "assert state['complete'] is False; "
                "assert state['validator_ok'] is False; "
                "assert state['all_prompt_sections_complete'] is False; "
                "assert 'final_two_model_results' in state['incomplete_prompt_sections']; "
                "assert state['final_output_root'] == '/data/outputs/llmcbench_edge_4090'; "
                "assert state['missing_required_artifact_count'] >= 1; "
                "assert state['completion_audit_ok'] is False; "
                "assert 'Final completion_audit.json ok=true for complete matrix outputs' in state['remaining_for_completion']; "
                "assert 'Final validate_edge_results.py sub-gate ok=true' in state['remaining_for_completion']; "
                "assert 'Final target_readiness.json completion_ok=true with no before_matrix_blockers' in state['remaining_for_completion']; "
                "assert 'Final edge_results_bundle_verify.json ok=true and complete=true' in state['remaining_for_completion']; "
                "assert 'Final edge_delivery_verify.json ok=true with all substatus checks passing' in state['remaining_for_completion']; "
                "assert 'results_bundle_verify_ok' in state; "
                "assert 'results_bundle_verify_complete' in state; "
                "assert 'results_bundle_completion_validator_ok' in state; "
                "assert 'results_bundle_target_completion_ok' in state; "
                "assert 'delivery_verify_ok' in state; "
                "assert 'delivery_verify_substatus_ok' in state; "
                "target_preflight=r['artifacts']['target_preflight']; "
                "assert '--cuda-visible-devices 0' in target_preflight; "
                "assert '--check-optional-backend-resolution' in target_preflight; "
                "assert '--check-quant-runtime' in target_preflight; "
                "assert 'delivery_completion_validator_ok' in state; "
                "assert 'delivery_target_completion_ok' in state; "
                "assert 'target_completion_ok' in state; "
                "final=r['final_artifacts']; "
                "assert final['validator']['exists'] is False; "
                "assert final['validator_ok'] is False; "
                "path_access=r['target_path_access']; "
                "assert isinstance(path_access['ok'], bool); "
                "assert {'hf_cache_dir','data_dir','output_root','calibration_file','prompt_file'} <= set(path_access['paths']); "
                "assert path_access['paths']['output_root']['path'] == '/data/outputs/llmcbench_edge_4090'; "
                "assert 'can_create' in path_access['paths']['output_root']; "
                "assert state['target_path_access_ok'] == path_access['ok']; "
                "assert state['target_path_access_blockers'] == path_access['blockers']; "
                "assert 'matrix_results.json' in prompt_checklist['accuracy_tool']['artifacts']; "
                "assert 'completion_audit.json ok=true' in prompt_checklist['final_two_model_results']['gates']; "
                "assert 'completion_audit validator sub-gate ok=true' in prompt_checklist['final_two_model_results']['gates']; "
                "assert 'target_readiness.json ok=true and completion_ok=true' in prompt_checklist['final_two_model_results']['gates']; "
                "assert 'edge_results_bundle_verify.json ok=true and complete=true' in prompt_checklist['final_two_model_results']['gates']; "
                "assert 'edge_delivery_verify.json ok=true with all substatus checks passing' in prompt_checklist['final_two_model_results']['gates']; "
                "final_artifacts=prompt_checklist['final_two_model_results']['artifacts']; "
                "assert 'tools/export_edge_results_bundle.py' in final_artifacts; "
                "assert 'tools/verify_edge_results_bundle.py' in final_artifacts; "
                "assert 'RESULTS_BUNDLE_MANIFEST.json' in final_artifacts; "
                "required_final_artifacts=['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','target_readiness.json','edge_scorecard.json','edge_scorecard.md','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json','edge_results_bundle_verify.json','edge_delivery_verify.json']; "
                "missing_final_artifacts=[name for name in required_final_artifacts if name not in final_artifacts]; "
                "assert not missing_final_artifacts, missing_final_artifacts; "
                "final_evidence=prompt_checklist['final_two_model_results']['evidence']; "
                "assert 'missing_required_artifact_count' in final_evidence; "
                "assert 'missing_required_artifacts' in final_evidence; "
                "assert 'results_bundle_verify_ok' in final_evidence; "
                "assert 'results_bundle_completion_validator_ok' in final_evidence; "
                "assert 'results_bundle_target_completion_ok' in final_evidence; "
                "assert 'delivery_verify_ok' in final_evidence; "
                "assert 'delivery_completion_validator_ok' in final_evidence; "
                "assert 'delivery_target_completion_ok' in final_evidence; "
                "assert 'matrix_results_json_exists' in final_evidence; "
                "assert 'matrix_results_csv_exists' in final_evidence; "
                "assert 'edge_report_exists' in final_evidence; "
                "assert 'final_matrix_expected_pair_count' in final_evidence; "
                "assert 'final_matrix_actual_pair_count' in final_evidence; "
                "assert 'final_matrix_missing_pairs' in final_evidence; "
                "assert 'final_matrix_unexpected_pairs' in final_evidence; "
                "assert 'final_matrix_duplicate_pairs' in final_evidence; "
                "assert 'final_matrix_invalid_row_count' in final_evidence; "
                "assert 'final_matrix_run_complete' in final_evidence; "
                "final_missing=';'.join(prompt_checklist['final_two_model_results']['missing_or_unverified']); "
                "assert 'Missing required final artifacts:' in final_missing; "
                "assert 'matrix_results.csv' in final_missing; "
                "assert 'edge_report.md' in final_missing; "
                "assert 'Final matrix_results.csv is missing' in final_missing; "
                "assert 'Final edge_report.md is missing' in final_missing; "
                "assert 'Final edge_results_bundle_verify.json ok=true and complete=true is missing' in final_missing; "
                "assert 'Final edge_delivery_verify.json ok=true with all substatus checks passing is missing' in final_missing; "
                "assert 'Final target_readiness.json completion_ok=true with no before_matrix_blockers is missing' in final_missing; "
                "assert 'edge_handoff_manifest.json' in prompt_checklist['portable_environment_and_handoff']['artifacts']; "
                "assert 'scripts/init_edge_storage.sh' in prompt_checklist['portable_environment_and_handoff']['artifacts']; "
                "portable_gates=prompt_checklist['portable_environment_and_handoff']['gates']; "
                "assert 'scripts/init_edge_storage.sh' in portable_gates; "
                "assert 'audit_target_readiness.py --require-non-rotational-storage' in portable_gates; "
                "assert 'audit_target_readiness.py --require-optional-backends-report' in portable_gates; "
                "assert 'audit_target_readiness.py --require-quant-runtime-report' in portable_gates; "
                "assert 'audit_target_readiness.py --require-prepare-precompressed-report' in portable_gates; "
                "assert 'audit_edge_completion.py --require-non-rotational-storage' in portable_gates; "
                "assert 'audit_edge_completion.py --require-target-readiness-report' in portable_gates; "
                "assert 'audit_edge_completion.py --require-optional-backends-report' in portable_gates; "
                "assert 'audit_edge_completion.py --require-quant-runtime-report' in portable_gates; "
                "assert 'audit_edge_completion.py --require-prepare-precompressed-report' in portable_gates; "
                "assert prompt_checklist['final_two_model_results']['complete'] is False; "
                "assert checklist['accuracy_tool']['target_tasks'] == ['mmlu','arc_challenge','hellaswag','piqa','winogrande']; "
                "assert checklist['accuracy_tool']['required_mmlu_subtask_count'] == 57; "
                "assert 'mmlu_abstract_algebra' in checklist['accuracy_tool']['required_mmlu_subtasks']; "
                "assert 'full MMLU subtask coverage' in prompt_checklist['accuracy_tool']['gates']; "
                "assert prompt_checklist['accuracy_tool']['evidence']['required_mmlu_subtask_count'] == 57; "
                "assert prompt_checklist['accuracy_tool']['evidence']['final_rows_with_full_mmlu_subtask_accuracy'] == 0; "
                "assert checklist['accuracy_tool']['current_evidence']['target_split_accuracy_tasks'] is True; "
                "assert checklist['accuracy_tool']['current_evidence']['target_continue_on_accuracy_error'] is True; "
                "assert 'qwen_wanda_four_task_smoke' in checklist['accuracy_tool']['current_evidence']; "
                "assert 'qwen_sparsegpt_four_task_smoke' in checklist['accuracy_tool']['current_evidence']; "
                "assert 'qwen_wanda_mmlu_subtask_smoke' in checklist['accuracy_tool']['current_evidence']; "
                "assert 'qwen_sparsegpt_mmlu_subtask_smoke' in checklist['accuracy_tool']['current_evidence']; "
                "assert 'final_accuracy_rows_complete' in checklist['accuracy_tool']['current_evidence']; "
                "assert 'final_rows_with_accuracy_metric_keys' in checklist['accuracy_tool']['current_evidence']; "
                "assert 'final_rows_with_final_accuracy_run' in checklist['accuracy_tool']['current_evidence']; "
                "assert 'final_rows_with_full_mmlu_subtask_metric_keys' in checklist['accuracy_tool']['current_evidence']; "
                "assert 'qwen_gptq_runtime_probe' in checklist['deployment_tool']['current_evidence']; "
                "assert 'qwen_awq_runtime_probe' in checklist['deployment_tool']['current_evidence']; "
                "assert 'qwen_smoothquant_runtime_probe' in checklist['deployment_tool']['current_evidence']; "
                "assert 'qwen_omniquant_runtime_probe' in checklist['deployment_tool']['current_evidence']; "
                "assert 'final_deployment_rows_complete' in checklist['deployment_tool']['current_evidence']; "
                "assert r['artifacts']['handoff_manifest'] == 'tools/build_edge_handoff_manifest.py'; "
                "assert r['qwen3']['current_verified_smoke_scope']['wanda_accuracy_four_target_tasks_limit_1'] is True; "
                "assert r['qwen3']['current_verified_smoke_scope']['sparsegpt_accuracy_four_target_tasks_limit_1'] is True; "
                "assert r['qwen3']['current_verified_smoke_scope']['wanda_accuracy_mmlu_subtask_limit_1'] is True; "
                "assert r['qwen3']['current_verified_smoke_scope']['sparsegpt_accuracy_mmlu_subtask_limit_1'] is True; "
                "assert 'gptq_runtime_imports_ok' in r['qwen3']['current_verified_smoke_scope']; "
                "assert 'awq_runtime_model_load_ok' in r['qwen3']['current_verified_smoke_scope']; "
                "assert 'smoothquant_runtime_config_tokenizer_ok' in r['qwen3']['current_verified_smoke_scope']; "
                "assert 'omniquant_runtime_probe_exists' in r['qwen3']['current_verified_smoke_scope']; "
                "assert checklist['deployment_tool']['target_memory_limit_gb'] == 24; "
                "assert 'CUDA 12.x' in checklist['target_platform']['requirement']; "
                "assert 'cuDNN' in checklist['target_platform']['requirement']; "
                "assert 'PyTorch 2.x' in checklist['target_platform']['requirement']; "
                "assert 'Transformers' in checklist['target_platform']['requirement']; "
                "assert 'multicore CPU' in checklist['target_platform']['requirement']; "
                "assert '256GB RAM' in checklist['target_platform']['requirement']; "
                "assert 'hf-mirror.com' in checklist['target_platform']['requirement']; "
                "assert 'runtime_requirements' in checklist['target_platform']; "
                "assert '24GB' in checklist['target_platform']['requirement']; "
                "assert checklist['target_platform']['configured_controls']['dtype'] == 'fp16'; "
                "assert r['target_4090_controls']['dtype'] == 'fp16'; "
                "assert r['target_4090_controls']['split_accuracy_tasks'] is True; "
                "assert r['target_4090_controls']['continue_on_accuracy_error'] is True; "
                "assert checklist['resource_tool']['current_evidence']['qwen_dense_resource_calflops']['compute_backend'] in ('calflops', None); "
                "assert 'compressed_checkpoint_ready' in checklist['resource_tool']['current_evidence']['qwen_wanda_resource_smoke']; "
                "assert 'wanda_compressed_checkpoint_ready' in r['qwen3']['current_verified_smoke_scope']; "
                "assert 'qwen_smoothquant_resource_precompressed' in checklist['resource_tool']['current_evidence']; "
                "assert 'qwen_omniquant_resource_precompressed' in checklist['resource_tool']['current_evidence']; "
                "assert 'llama_gptq_resource_precompressed' in checklist['resource_tool']['current_evidence']; "
                "assert 'llama_awq_resource_precompressed' in checklist['resource_tool']['current_evidence']; "
                "assert 'llama_smoothquant_resource_precompressed' in checklist['resource_tool']['current_evidence']; "
                "assert 'final_resource_rows_complete' in checklist['resource_tool']['current_evidence']; "
                "assert 'smoothquant_precompressed_resource_config_fallback' in r['qwen3']['current_verified_smoke_scope']; "
                "assert 'omniquant_precompressed_resource_model_load' in r['qwen3']['current_verified_smoke_scope']; "
                "assert 'awq_resource_precompressed' in r['llama3']; "
                "assert 'gptq_resource_precompressed' in r['llama3']; "
                "assert 'gptq_runtime_probe' in r['llama3']; "
                "assert 'awq_runtime_probe' in r['llama3']; "
                "assert 'smoothquant_resource_precompressed' in r['llama3']; "
                "assert 'smoothquant_runtime_probe' in r['llama3']; "
                "final=r['final_artifacts']; "
                "assert final['output_root'] == '/data/outputs/llmcbench_edge_4090'; "
                "assert 'completion_audit.json' in final['artifacts']; "
                "assert 'edge_asset_audit.json' in final['artifacts']; "
                "assert 'edge_preflight_report.json' in final['artifacts']; "
                "assert 'edge_prewarm_report.json' in final['artifacts']; "
                "assert 'edge_handoff_manifest.json' in final['artifacts']; "
                "assert 'edge_optional_backends_report.json' in final['artifacts']; "
                "assert 'edge_quant_runtime_report.json' in final['artifacts']; "
                "assert 'matrix_run_report.json' in final['artifacts']; "
                "assert 'matrix_finalize_report.json' in final['artifacts']; "
                "assert 'target_readiness.json' in final['artifacts']; "
                "assert 'edge_optional_backends_report.json' not in final['optional_artifacts']; "
                "assert 'edge_quant_runtime_report.json' not in final['optional_artifacts']; "
                "assert 'required_artifacts' in final; "
                "assert final['missing_required_artifact_count'] >= 1; "
                "missing_names={item['name'] for item in final['missing_required_artifacts']}; "
                "assert 'edge_scorecard.json' in missing_names; "
                "assert 'edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json' in missing_names; "
                "assert 'asset_audit' in final; "
                "assert 'preflight_report' in final; "
                "assert 'prewarm_report' in final; "
                "assert 'handoff_manifest' in final; "
                "assert 'optional_backends_report' in final; "
                "assert 'quant_runtime_report' in final; "
                "assert 'target_readiness' in final; "
                "assert 'results_bundle_verify' in final; "
                "assert 'delivery_verify' in final; "
                "assert final['target_readiness']['ok'] is False; "
                "assert final['target_readiness'].get('completion_ok') is not True; "
                "assert final['results_bundle_verify']['exists'] is False; "
                "assert final['results_bundle_verify']['ok'] is False; "
                "assert final['delivery_verify']['exists'] is False; "
                "assert final['delivery_verify']['ok'] is False; "
                "assert 'mmlu_subtask_coverage' in final; "
                "assert final['mmlu_subtask_coverage']['required_subtask_count'] == 57; "
                "assert final['mmlu_subtask_coverage']['complete_row_count'] == 0; "
                "assert 'matrix_results' in final; "
                "assert 'raw_row_count' in final['matrix_results']; "
                "assert 'invalid_row_count' in final['matrix_results']; "
                "assert 'invalid_rows' in final['matrix_results']; "
                "assert all('has_compression_provenance' in row for row in final['matrix_results'].get('rows', [])); "
                "assert all('has_method_compression_params' in row for row in final['matrix_results'].get('rows', [])); "
                "assert 'final_matrix_expected_pair_count' in final; "
                "assert 'final_matrix_actual_pair_count' in final; "
                "assert 'final_matrix_missing_pairs' in final; "
                "assert 'final_matrix_unexpected_pairs' in final; "
                "assert 'final_matrix_duplicate_pairs' in final; "
                "assert 'matrix_results_csv' in final; "
                "assert 'edge_report' in final; "
                "assert all('has_final_accuracy_run' in row for row in final['matrix_results'].get('rows', [])); "
                "assert final['matrix_results_csv']['path'].endswith('matrix_results.csv'); "
                "assert final['edge_report']['path'].endswith('edge_report.md'); "
                "assert 'completion_audit' in final; "
                "assert final['required_report_gates_ok'] is False; "
                "assert final['complete'] is False; "
                "assert checklist['target_platform']['complete'] is False; "
                "remaining=';'.join(r['remaining_for_completion']); "
                "assert 'Target multicore CPU / 256GB RAM / Python 3.10 / PyTorch 2.x / Transformers / 4090 24GB / CUDA 12.x / cuDNN / hf-mirror dependency environment verification' in remaining; "
                "assert 'SmoothQuant/OmniQuant external commands or complete precompressed checkpoints' in remaining; "
                "assert 'GPTQ/AWQ backend installation' not in remaining; "
                "assert r['method_readiness_blockers']['gptq']['precompressed_ready'] is True; "
                "assert r['method_readiness_blockers']['gptq']['precompressed_paths']['LLaMA3-8B']['ready'] is True; "
                "gptq_blockers=r['method_readiness_blockers']['gptq']; "
                "assert gptq_blockers['runtime_ready'] is True or 'GPTQ/AWQ compression backend or complete precompressed checkpoints plus loading runtime' in remaining or gptq_blockers.get('target_readiness_checked') is True; "
                "assert gptq_blockers['runtime_ready'] is True or gptq_blockers.get('target_ready_count', 0) >= 1 or any('GPTQ runtime loading requires optimum' in item for item in gptq_blockers['blockers']) or 'Qwen3-8B:missing_runtime_import:any_of:gptqmodel' in gptq_blockers['blockers']",
            ],
        ),
        (
            "goal_audit_requires_complete_precompressed_checkpoints",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_precompressed_')); "
                "llama=root/'llama-gptq'; qwen=root/'qwen-gptq'; "
                "llama.mkdir(); qwen.mkdir(); "
                "(llama/'config.json').write_text('{}\\n'); "
                "(qwen/'config.json').write_text('{}\\n'); "
                "(qwen/'model.safetensors').write_text('tiny\\n'); "
                "cfg=root/'config.json'; out=root/'audit.json'; "
                "base={'models':[{'name':'LLaMA3-8B','path':str(root/'missing-llama')},{'name':'Qwen3-8B','path':str(root/'missing-qwen')}],"
                "'methods':['gptq'],'defaults':{},'precompressed_paths':{'LLaMA3-8B':{'gptq':str(llama)},'Qwen3-8B':{'gptq':str(qwen)}}}; "
                "cfg.write_text(json.dumps(base)); "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--config',str(cfg),'--local-config',str(cfg),"
                "'--qwen-path',str(root/'missing-qwen'),'--llama-path',str(root/'missing-llama'),'--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "gptq=r['method_readiness_blockers']['gptq']; "
                "assert gptq['precompressed_ready'] is False; "
                "assert gptq['precompressed_paths']['LLaMA3-8B']['ready'] is False; "
                "(llama/'model.safetensors').write_text('tiny\\n'); "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--config',str(cfg),'--local-config',str(cfg),"
                "'--qwen-path',str(root/'missing-qwen'),'--llama-path',str(root/'missing-llama'),'--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "gptq=r['method_readiness_blockers']['gptq']; "
                "assert gptq['precompressed_ready'] is True; "
                "assert (gptq['ready'] is True if gptq['runtime_ready'] else gptq['ready'] is False); "
                "assert gptq['runtime_ready'] is True or any('GPTQ runtime loading requires optimum' in item for item in gptq['blockers'])",
            ],
        ),
        (
            "goal_audit_reports_target_path_access",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_path_access_')); "
                "missing_parent=root/'missing_parent'; "
                "cfg=root/'config.json'; out=root/'audit.json'; "
                "defaults={'hf_cache_dir':str(missing_parent/'hf_cache'),'data_dir':str(missing_parent/'datasets'),"
                "'output_root':str(missing_parent/'outputs'),'calibration_file':str(missing_parent/'calib.txt'),"
                "'prompt_file':str(missing_parent/'prompts.txt'),'deployment_memory_limit_gb':24}; "
                "cfg.write_text(json.dumps({'models':[{'name':'LLaMA3-8B','path':str(root/'missing-llama')},{'name':'Qwen3-8B','path':str(root/'missing-qwen')}],"
                "'methods':['dense'],'defaults':defaults})); "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--config',str(cfg),'--local-config',str(cfg),"
                "'--qwen-path',str(root/'missing-qwen'),'--llama-path',str(root/'missing-llama'),'--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); access=r['target_path_access']; "
                "assert access['ok'] is True; "
                "assert access['paths']['output_root']['nearest_existing_parent'] == str(root); "
                "assert access['paths']['output_root']['can_create'] is True; "
                "assert r['completion_state']['target_path_access_ok'] is True; "
                "blocked=root/'blocked_parent'; blocked.write_text('not a directory'); "
                "defaults['output_root']=str(blocked/'outputs'); cfg.write_text(json.dumps({'models':[{'name':'LLaMA3-8B','path':str(root/'missing-llama')},{'name':'Qwen3-8B','path':str(root/'missing-qwen')}],"
                "'methods':['dense'],'defaults':defaults})); "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--config',str(cfg),'--local-config',str(cfg),"
                "'--qwen-path',str(root/'missing-qwen'),'--llama-path',str(root/'missing-llama'),'--output',str(out)],"
                "capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); access=r['target_path_access']; "
                "assert access['ok'] is False; "
                "assert access['paths']['output_root']['can_create'] is False; "
                "assert any(item['name']=='output_root' for item in access['blockers']); "
                "assert r['completion_state']['target_path_access_ok'] is False; "
                "assert any(item['name']=='output_root' for item in r['completion_state']['target_path_access_blockers'])",
            ],
        ),
        (
            "goal_audit_summarizes_prewarm_completion_gate",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_prewarm_gate_')); "
                "[(root/name).write_text('[]' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_prewarm_report.json']]; "
                "(root/'edge_preflight_report.json').write_text(json.dumps({'ok':True,'steps':[{'name':'verify_env','ok':False,'required':False,'returncode':1},{'name':'matrix_dry_run','ok':True,'required':True,'returncode':0}]})); "
                "(root/'edge_prewarm_report.json').write_text(json.dumps({'ok':True,'tasks':['piqa','arc_challenge'],'target_count':1,'attempted':[{'model':'m1','task':'piqa'},{'model':'m1','task':'arc_challenge'}],'succeeded':[{'model':'m1','task':'piqa'}],'failed':[]})); "
                "(root/'edge_optional_backends_report.json').write_text(json.dumps({'ok':False,'dry_run':True,'selected_packages':['autoawq'],'pip_returncode':1,'imports':{'awq':{'ok':False}}})); "
                "(root/'edge_quant_runtime_report.json').write_text(json.dumps({'ok':False,'rows':[{'model':'m1','method':'gptq','model_path':'/tmp/m1-gptq','ok':False,'returncode':1}]})); "
                "(root/'edge_prepare_precompressed_report.json').write_text(json.dumps({'ok':False,'dry_run':False,'metadata_only':False,'updated_config':str(root/'with_precompressed.json'),'items':[{'model':'LLaMA3-8B','method':'omniquant','ok':False,'reason':'repo_not_configured'}],'blockers':[{'model':'LLaMA3-8B','method':'omniquant','reason':'repo_not_configured'}]})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':False,'config':'configs/edge_4090_matrix.json','root':str(root),'ready_for_matrix':False,'completion_ok':False,'required_before_matrix':['asset_audit'],'missing_before_matrix':['asset_audit'],'before_matrix_blockers':[{'type':'missing_artifact','artifact':'asset_audit'}],'method_readiness':{'payload':{'summary':{'gptq':{'ok':1,'blocked':1}},'rows':[{'model':'LLaMA3-8B','method':'gptq','ok':True,'uses_precompressed':True,'precompressed_path':'/tmp/llama-gptq','blockers':[]},{'model':'Qwen3-8B','method':'gptq','ok':False,'uses_precompressed':True,'precompressed_path':'/tmp/qwen-gptq','runtime_requirements_reason':'gptqmodel_checkpoint_metadata','quantization_metadata':{'quantizer':['gptqmodel:4.0.0']},'blockers':['missing_runtime_import:any_of:gptqmodel']}]}},'next_command_groups':{'precompressed_prepare':['python tools/prepare_precompressed_assets.py --fail-on-error'],'precompressed_config_followup':['python tools/run_edge_matrix.py --config edge_matrix_with_precompressed.json'],'baseline_finalize':['python tools/run_edge_matrix.py --config configs/edge_4090_matrix.json'],'precompressed_config_post_run_audit':['python tools/audit_target_readiness.py --config edge_matrix_with_precompressed.json --require-prepare-precompressed-report','python tools/build_goal_audit.py'],'baseline_post_run_audit':['python tools/audit_target_readiness.py --config configs/edge_4090_matrix.json','python tools/build_goal_audit.py']}})); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':False,'asset_audit_report_status':{'ok':False,'required':True,'report_ok':False},'preflight_report_status':{'ok':False,'required':True,'error':'required preflight report is missing'},'prewarm_report_status':{'ok':False,'failed':[{'task':'piqa'}],'missing_succeeded_pairs':[{'model':'m1','task':'arc_challenge'}],'violations':['succeeded is missing model/task pairs']},'optional_backends_report_status':{'ok':False,'required':True,'report_ok':False,'error':'optional backend resolution failed'},'quant_runtime_report_status':{'ok':False,'required':True,'report_ok':False,'error':'quant runtime probe failed'}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "prewarm=r['final_artifacts']['prewarm_report']; "
                "preflight=r['final_artifacts']['preflight_report']; "
                "assert preflight['step_count'] == 2; "
                "assert preflight['failed_optional_steps'][0]['name'] == 'verify_env'; "
                "assert r['final_artifacts']['asset_audit']['gate_ok'] is False; "
                "assert r['final_artifacts']['asset_audit']['gate_report_ok'] is False; "
                "assert r['final_artifacts']['preflight_report']['gate_error'] == 'required preflight report is missing'; "
                "assert prewarm['task_count'] == 2; "
                "assert prewarm['succeeded_count'] == 1; "
                "assert prewarm['missing_succeeded_pairs'][0]['task'] == 'arc_challenge'; "
                "assert 'succeeded is missing model/task pairs' in prewarm['violations']; "
                "optional=r['final_artifacts']['optional_backends_report']; "
                "quant=r['final_artifacts']['quant_runtime_report']; "
                "assert optional['exists'] is True and optional['ok'] is False; "
                "assert optional['gate_required'] is True and optional['gate_ok'] is False; "
                "assert optional['selected_packages'] == ['autoawq']; "
                "assert quant['exists'] is True and quant['ok'] is False; "
                "assert quant['gate_required'] is True and quant['gate_ok'] is False; "
                "assert quant['entry_source_field'] == 'rows'; "
                "assert quant['row_count'] == 1; "
                "assert quant['probe_count'] == 1; "
                "assert quant['failed_probes'][0]['path'] == '/tmp/m1-gptq'; "
                "assert quant['failed_probes'][0]['method'] == 'gptq'; "
                "prep=r['final_artifacts']['prepare_precompressed_report']; "
                "assert prep['exists'] is True and prep['ok'] is False; "
                "assert prep['blocker_count'] == 1; "
                "assert prep['blockers'][0]['reason'] == 'repo_not_configured'; "
                "assert prep['failed_items'][0]['method'] == 'omniquant'; "
                "target=r['final_artifacts']['target_readiness']; "
                "assert target['next_command_group_counts']['precompressed_prepare'] == 1; "
                "assert target['has_precompressed_config_followup_matrix'] is True; "
                "assert target['has_baseline_finalize_matrix'] is True; "
                "assert target['has_precompressed_config_post_run_readiness'] is True; "
                "assert target['has_baseline_post_run_readiness'] is True; "
                "gptq=r['method_readiness_blockers']['gptq']; "
                "assert gptq['target_readiness_checked'] is True; "
                "assert gptq['target_ready_count'] == 1; "
                "assert gptq['target_blocked_count'] == 1; "
                "assert gptq['ready'] is False and gptq['runtime_ready'] is False; "
                "assert 'Qwen3-8B:missing_runtime_import:any_of:gptqmodel' in gptq['blockers']; "
                "assert any(row['model']=='Qwen3-8B' and row['runtime_requirements_reason']=='gptqmodel_checkpoint_metadata' for row in gptq['target_rows']); "
                "state=r['completion_state']; "
                "assert state['status'] == 'incomplete'; "
                "assert state['next_command_group_names'] == target['next_command_group_names']; "
                "assert 'precompressed_prepare' in state['next_command_group_names']; "
                "portable=r['prompt_to_artifact_checklist']['portable_environment_and_handoff']['evidence']; "
                "assert 'precompressed_prepare' in portable['target_readiness_next_command_groups']; "
                "assert portable['has_precompressed_config_followup_matrix'] is True; "
                "assert portable['has_precompressed_config_post_run_readiness'] is True; "
                "assert portable['has_baseline_post_run_readiness'] is True; "
                "summary=r['final_artifacts']['completion_audit']; "
                "assert summary['prewarm_report_status_ok'] is False; "
                "assert summary['optional_backends_report_status_ok'] is False; "
                "assert summary['quant_runtime_report_status_ok'] is False; "
                "assert summary['prewarm_report_status']['failed'][0]['task'] == 'piqa'",
            ],
        ),
        (
            "goal_audit_reports_portable_bundle_verify",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_portable_verify_')); "
                "cfg=root/'config.json'; out=root/'goal.json'; verify=root/'portable_bundle_verify.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'LLaMA3-8B','path':str(root/'missing-llama')},{'name':'Qwen3-8B','path':str(root/'missing-qwen')}],"
                "'methods':['dense'],'defaults':{'hf_cache_dir':str(root/'hf_cache'),'data_dir':str(root/'datasets'),"
                "'output_root':str(root/'outputs'),'calibration_file':str(root/'calib.txt'),'prompt_file':str(root/'prompt.txt'),"
                "'deployment_memory_limit_gb':24,'max_memory':'0:22GiB,cpu:120GiB','dtype':'fp16'}})); "
                "verify.write_text(json.dumps({'ok':True,'file_count':47,'verified_file_count':47,'violations':[],"
                "'archive_status':{'ok':True,'checked':True,'expected_top_level_dir':'bundle','archive_top_level_dirs':['bundle'],'verified_archive_file_count':48,'duplicate_archive_files':[],'unsafe_archive_members':[],'unsupported_archive_members':[],'size_mismatches':[],'sha256_mismatches':[]},"
                "'code_status':{'ok':True,'manifest_commit':'abc123','handoff_commit':'abc123','manifest_dirty':False,"
                "'handoff_dirty':False,'require_clean_code':True,'require_commit':'abc123','violations':[]}})); "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--config',str(cfg),'--local-config',str(cfg),"
                "'--qwen-path',str(root/'missing-qwen'),'--llama-path',str(root/'missing-llama'),"
                "'--portable-bundle-verify',str(verify),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); pbv=r['portable_bundle_verify']; "
                "assert pbv['exists'] is True and pbv['ok'] is True; "
                "assert pbv['file_count'] == 47 and pbv['verified_file_count'] == 47; "
                "assert pbv['archive_ok'] is True and pbv['archive_checked'] is True; "
                "assert pbv['archive_expected_top_level_dir'] == 'bundle' and pbv['archive_top_level_dirs'] == ['bundle']; "
                "assert pbv['archive_verified_file_count'] == 48; "
                "assert pbv['archive_duplicate_file_count'] == 0 and pbv['archive_duplicate_files'] == []; "
                "assert pbv['archive_unsafe_member_count'] == 0 and pbv['archive_unsafe_members'] == []; "
                "assert pbv['archive_unsupported_member_count'] == 0 and pbv['archive_unsupported_members'] == []; "
                "assert pbv['archive_size_mismatch_count'] == 0 and pbv['archive_sha256_mismatch_count'] == 0; "
                "assert pbv['archive_size_mismatches'] == [] and pbv['archive_sha256_mismatches'] == []; "
                "assert pbv['code_ok'] is True; "
                "assert pbv['manifest_commit'] == 'abc123' and pbv['handoff_commit'] == 'abc123'; "
                "assert pbv['manifest_dirty'] is False and pbv['handoff_dirty'] is False; "
                "assert pbv['require_clean_code'] is True and pbv['require_commit'] == 'abc123'; "
                "portable=r['prompt_to_artifact_checklist']['portable_environment_and_handoff']['evidence']['portable_bundle_verify']; "
                "assert portable['ok'] is True and portable['code_ok'] is True; "
                "assert portable['archive_top_level_dirs'] == ['bundle']; "
                "assert portable['archive_verified_file_count'] == 48; "
                "assert portable['archive_duplicate_file_count'] == 0; "
                "assert portable['archive_unsafe_member_count'] == 0; "
                "assert portable['archive_unsupported_member_count'] == 0; "
                "assert r['completion_state']['status'] == 'incomplete'; "
                "assert 'final_two_model_results' in r['completion_state']['incomplete_prompt_sections']",
            ],
        ),
        (
            "goal_audit_requires_report_gates_for_completion",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_report_gates_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json','edge_results_bundle_verify.json','edge_delivery_verify.json']]; "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':False},'preflight_report_status':{'ok':True,'required':False},'prewarm_report_status':{'ok':True,'required':False},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':False,'required':True,'error':'missing runtime probe'}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "assert r['final_artifacts']['completion_audit_ok'] is True; "
                "assert r['final_artifacts']['all_artifacts_present'] is True; "
                "assert r['final_artifacts']['missing_required_artifact_count'] == 0; "
                "assert r['final_artifacts']['matrix_results_csv']['exists'] is True; "
                "assert r['final_artifacts']['edge_report']['exists'] is True; "
                "assert r['final_artifacts']['required_report_gates_ok'] is False; "
                "assert r['final_artifacts']['optional_backends_report']['gate_required'] is True; "
                "assert r['final_artifacts']['optional_backends_report']['gate_ok'] is True; "
                "assert r['final_artifacts']['quant_runtime_report']['gate_required'] is True; "
                "assert r['final_artifacts']['quant_runtime_report']['gate_ok'] is False; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['status'] == 'incomplete'; "
                "assert r['completion_state']['final_artifacts_complete'] is False; "
                "assert r['completion_state']['completion_audit_ok'] is True; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_requires_delivery_verifiers_for_completion",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_delivery_verifiers_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':False,'violations':['bundle incomplete'],'completion_audit_status':{'ok':True,'validator_ok':True,'validator_payload_ok':True},'target_readiness_status':{'ok':True,'ready_for_matrix':True,'completion_ok':True}})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True,'validator_ok':True,'validator_payload_ok':True},'target_readiness_status':{'ok':True,'ready_for_matrix':True,'completion_ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':False},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "final=r['final_artifacts']; "
                "assert final['all_artifacts_present'] is True; "
                "assert final['completion_audit_ok'] is True; "
                "assert final['target_readiness']['ok'] is True; "
                "assert final['results_bundle_verify']['exists'] is True; "
                "assert final['results_bundle_verify']['ok'] is True; "
                "assert final['results_bundle_verify']['complete'] is False; "
                "assert final['results_bundle_verify']['completion_validator_ok'] is True; "
                "assert final['results_bundle_verify']['target_completion_ok'] is True; "
                "assert final['delivery_verify']['exists'] is True; "
                "assert final['delivery_verify']['ok'] is True; "
                "assert final['delivery_verify']['completion_validator_ok'] is True; "
                "assert final['delivery_verify']['target_completion_ok'] is True; "
                "assert final['delivery_verify']['root_bundle_match_ok'] is False; "
                "assert final['delivery_verify']['substatus_ok'] is False; "
                "assert final['complete'] is False; "
                "state=r['completion_state']; "
                "assert state['final_artifacts_complete'] is False; "
                "assert state['results_bundle_verify_ok'] is True; "
                "assert state['results_bundle_verify_complete'] is False; "
                "assert state['delivery_verify_ok'] is True; "
                "assert state['delivery_verify_substatus_ok'] is False; "
                "assert state['target_completion_ok'] is True; "
                "assert 'Full two-model x all-method x all-task matrix run' not in state['remaining_for_completion']; "
                "assert 'Final completion_audit.json ok=true for complete matrix outputs' not in state['remaining_for_completion']; "
                "assert 'Final validate_edge_results.py sub-gate ok=true' in state['remaining_for_completion']; "
                "assert 'Final edge_results_bundle_verify.json ok=true and complete=true' in state['remaining_for_completion']; "
                "assert 'Final edge_delivery_verify.json ok=true with all substatus checks passing' in state['remaining_for_completion']; "
                "assert state['complete'] is False; "
                "evidence=r['prompt_to_artifact_checklist']['final_two_model_results']['evidence']; "
                "assert evidence['results_bundle_verify_ok'] is True; "
                "assert evidence['results_bundle_verify_complete'] is False; "
                "assert evidence['delivery_verify_ok'] is True; "
                "assert evidence['delivery_verify_substatus_ok'] is False; "
                "missing=';'.join(r['prompt_to_artifact_checklist']['final_two_model_results']['missing_or_unverified']); "
                "assert 'Final edge_results_bundle_verify.json ok=true and complete=true is missing' in missing; "
                "assert 'Final edge_delivery_verify.json ok=true with all substatus checks passing is missing' in missing",
            ],
        ),
        (
            "goal_audit_keeps_matrix_blocker_for_limited_accuracy",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_limited_matrix_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':1,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[],'completion_audit_status':{'ok':True,'validator_ok':True,'validator_payload_ok':True},'target_readiness_status':{'ok':True,'ready_for_matrix':True,'completion_ok':False}})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True,'validator_ok':True,'validator_payload_ok':True},'target_readiness_status':{'ok':True,'ready_for_matrix':True,'completion_ok':False},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "rows=r['final_artifacts']['matrix_results']['rows']; "
                "assert rows and all(row['has_final_accuracy_run'] is False for row in rows); "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_keeps_matrix_blocker_for_missing_resource_numbers",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_missing_resource_numbers_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[],'completion_audit_status':{'ok':True,'validator_ok':True,'validator_payload_ok':True},'target_readiness_status':{'ok':True,'ready_for_matrix':True,'completion_ok':False}})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True,'validator_ok':True,'validator_payload_ok':True},'target_readiness_status':{'ok':True,'ready_for_matrix':True,'completion_ok':False},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "rows=r['final_artifacts']['matrix_results']['rows']; "
                "assert rows and all(row['has_resource'] is False for row in rows); "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_keeps_matrix_blocker_for_resource_fallback",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_resource_fallback_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'config_only_fallback','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[],'completion_audit_status':{'ok':True,'validator_ok':True,'validator_payload_ok':True},'target_readiness_status':{'ok':True,'ready_for_matrix':True,'completion_ok':False}})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True,'validator_ok':True,'validator_payload_ok':True},'target_readiness_status':{'ok':True,'ready_for_matrix':True,'completion_ok':False},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "rows=r['final_artifacts']['matrix_results']['rows']; "
                "assert rows and all(row['resource_backend'] == 'config_only_fallback' for row in rows); "
                "assert all(row['has_resource'] is False for row in rows); "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_rejects_unexpected_matrix_pair",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_unexpected_pair_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "extra=dict(rows[0], model_name='OtherModel'); rows.append(extra); "
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[]})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True},'target_readiness_status':{'ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "assert r['final_artifacts']['final_matrix_expected_pair_count'] == 14; "
                "assert r['final_artifacts']['final_matrix_actual_pair_count'] == 15; "
                "assert r['final_artifacts']['final_matrix_unexpected_pairs'] == [{'model':'OtherModel','method':'dense'}]; "
                "assert r['final_artifacts']['final_matrix_duplicate_pairs'] == []; "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_rejects_duplicate_matrix_pair",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_duplicate_pair_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "rows.pop(); rows.append(dict(rows[0])); "
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[]})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True},'target_readiness_status':{'ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "assert r['final_artifacts']['final_matrix_expected_pair_count'] == 14; "
                "assert r['final_artifacts']['final_matrix_actual_pair_count'] == 14; "
                "assert r['final_artifacts']['final_matrix_duplicate_pairs'] == [{'model':'LLaMA3-8B','method':'dense'}]; "
                "assert r['final_artifacts']['final_matrix_missing_pairs'] == [{'model':'Qwen3-8B','method':'omniquant'}]; "
                "assert r['final_artifacts']['final_matrix_unexpected_pairs'] == []; "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_rejects_invalid_matrix_row",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_invalid_matrix_row_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "rows.append('not-a-row'); "
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[]})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True},'target_readiness_status':{'ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "matrix=r['final_artifacts']['matrix_results']; "
                "assert matrix['raw_row_count'] == 15; "
                "assert matrix['row_count'] == 14; "
                "assert matrix['invalid_row_count'] == 1; "
                "assert matrix['invalid_rows'] == [{'index':14,'type':'str'}]; "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_keeps_matrix_blocker_for_missing_compression_provenance",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_missing_compression_provenance_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[]})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True},'target_readiness_status':{'ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "rows=r['final_artifacts']['matrix_results']['rows']; "
                "assert rows and all(row['has_compression_provenance'] is False for row in rows); "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_keeps_matrix_blocker_for_missing_compression_booleans",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_missing_compression_booleans_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[]})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True},'target_readiness_status':{'ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "rows=r['final_artifacts']['matrix_results']['rows']; "
                "assert rows and all(row['has_compression_provenance'] is False for row in rows); "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_keeps_matrix_blocker_for_invalid_compression_backend",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_invalid_compression_backend_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':'manual_backend','compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[]})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True},'target_readiness_status':{'ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "rows=r['final_artifacts']['matrix_results']['rows']; "
                "assert rows and all(row['compression_backend'] == 'manual_backend' for row in rows); "
                "assert all(row['has_compression_provenance'] is False for row in rows); "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_keeps_matrix_blocker_for_failed_external_command",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_failed_external_command_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        backend={'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':backend,'compression_materialized':(backend not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(backend == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':(7 if backend == 'external_command' else 0),'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[]})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True},'target_readiness_status':{'ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "rows=r['final_artifacts']['matrix_results']['rows']; "
                "assert any(row['compression_backend'] == 'external_command' and row['has_compression_provenance'] is False for row in rows); "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_keeps_matrix_blocker_for_missing_method_params",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_missing_method_params_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[]})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True},'target_readiness_status':{'ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "rows=r['final_artifacts']['matrix_results']['rows']; "
                "assert any(row['compression_method']=='gptq' and row['has_method_compression_params'] is False for row in rows); "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_keeps_matrix_blocker_for_bad_accuracy_score",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_bad_accuracy_score_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':1.2,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[]})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True},'target_readiness_status':{'ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "rows=r['final_artifacts']['matrix_results']['rows']; "
                "assert rows and all(row['has_accuracy'] is False for row in rows); "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_keeps_matrix_blocker_for_missing_score_metrics",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_missing_score_metrics_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[]})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True},'target_readiness_status':{'ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "rows=r['final_artifacts']['matrix_results']['rows']; "
                "assert rows and all(row['has_accuracy'] is True for row in rows); "
                "assert all(row['has_accuracy_metric_keys'] is False for row in rows); "
                "assert all(row['has_full_mmlu_subtask_metric_keys'] is True for row in rows); "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_keeps_matrix_blocker_for_bad_deployment_metrics",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_bad_deployment_metrics_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':25.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[]})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True},'target_readiness_status':{'ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "rows=r['final_artifacts']['matrix_results']['rows']; "
                "assert rows and all(row['has_deployment'] is False for row in rows); "
                "assert r['final_artifacts']['final_matrix_run_complete'] is False; "
                "assert 'Full two-model x all-method x all-task matrix run' in r['completion_state']['remaining_for_completion']; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert r['completion_state']['complete'] is False",
            ],
        ),
        (
            "goal_audit_requires_target_completion_ok",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_target_completion_')); "
                "[(root/name).parent.mkdir(parents=True, exist_ok=True) or (root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_asset_audit.json','edge_preflight_report.json','edge_prewarm_report.json','edge_handoff_manifest.json','edge_target_paths_report.json','edge_optional_backends_report.json','edge_quant_runtime_report.json','matrix_run_report.json','matrix_finalize_report.json','matrix_results.json','matrix_results.csv','edge_report.md','edge_scorecard.json','edge_scorecard.md','target_readiness.json','edge_results_bundle.tar.gz','edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json']]; "
                "rows=[]; "
                "\nfor model in ['LLaMA3-8B','Qwen3-8B']:\n"
                "    for method in ['dense','wanda','sparsegpt','gptq','awq','smoothquant','omniquant']:\n"
                "        row={'model_name':model,'compression_method':method,'source_model_path':'/tmp/source','compressed_model_path':'/tmp/compressed','compression_backend':({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]),'compression_materialized':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) not in {'source_checkpoint_reference','precompressed_checkpoint'}),'compression_uses_precompressed':(({'dense':'source_checkpoint_reference','wanda':'built_in_activation_pruning','sparsegpt':'built_in_activation_pruning','gptq':'auto_gptq','awq':'autoawq','smoothquant':'external_command','omniquant':'external_command'}[method]) == 'precompressed_checkpoint'),'compression_sparsity':0.5,'compression_prune_lm_head':False,'compression_quant_bits':4,'compression_group_size':128,'compression_calibration_samples':2,'compression_desc_act':False,'compression_awq_version':'GEMM','compression_external_command':'python export.py','compression_external_returncode':0,'mmlu_score':0.1,'arc_challenge_score':0.1,'hellaswag_score':0.1,'piqa_score':0.1,'winogrande_score':0.1,'accuracy_limit':None,'accuracy_failed_task_count':0,'accuracy_failed_tasks':'','tokens_per_second':1.0,'deployment_memory_measured_peak_gb':1.0,'deployment_memory_limit_gb':24.0,'deployment_memory_limit_exceeded':False,'storage_gb':1.0,'flops':'1 FLOPS','flops_num':1.0,'resource_backend':'model_load','compute_backend':'analytical_config_estimate'}\n"
                "        row.update({'mmlu_score_metric':'aggregate:acc,none','arc_challenge_score_metric':'acc_norm,none','hellaswag_score_metric':'acc_norm,none','piqa_score_metric':'acc_norm,none','winogrande_score_metric':'acc,none'})\n"
                "        row.update({f'{task}_score':0.1 for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        row.update({f'{task}_score_metric':'acc,none' for task in DEFAULT_MMLU_SUBTASKS})\n"
                "        rows.append(row)\n"
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'completion_audit.json').write_text(json.dumps({'ok':True,'asset_audit_report_status':{'ok':True,'required':True},'preflight_report_status':{'ok':True,'required':True},'prewarm_report_status':{'ok':True,'required':True},'optional_backends_report_status':{'ok':True,'required':True},'quant_runtime_report_status':{'ok':True,'required':True}})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True,'before_matrix_blockers':[{'type':'artifact_not_ok','artifact':'preflight'}]})); "
                "(root/'edge_results_bundle_verify.json').write_text(json.dumps({'ok':True,'complete':True,'violations':[],'completion_audit_status':{'ok':True,'validator_ok':True,'validator_payload_ok':True},'target_readiness_status':{'ok':True,'ready_for_matrix':True,'completion_ok':True}})); "
                "(root/'edge_delivery_verify.json').write_text(json.dumps({'ok':True,'violations':[],'completion_audit_status':{'ok':True,'validator_ok':True,'validator_payload_ok':True},'target_readiness_status':{'ok':True,'ready_for_matrix':True,'completion_ok':True},'results_bundle_status':{'ok':True},'root_bundle_match_status':{'ok':True},'archive_bundle_match_status':{'ok':True}})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads(out.read_text()); "
                "state=r['completion_state']; "
                "assert r['final_artifacts']['target_readiness']['ok'] is True; "
                "assert r['final_artifacts']['target_readiness']['completion_ok'] is True; "
                "assert r['final_artifacts']['target_readiness']['before_matrix_blocker_count'] == 1; "
                "assert r['final_artifacts']['results_bundle_verify']['target_completion_ok'] is True; "
                "assert r['final_artifacts']['delivery_verify']['target_completion_ok'] is True; "
                "assert r['final_artifacts']['final_matrix_run_complete'] is True; "
                "assert r['final_artifacts']['complete'] is False; "
                "assert state['target_readiness_ok'] is True; "
                "assert state['target_completion_ok'] is True; "
                "assert state['results_bundle_target_completion_ok'] is True; "
                "assert state['delivery_target_completion_ok'] is True; "
                "assert 'Full two-model x all-method x all-task matrix run' not in state['remaining_for_completion']; "
                "assert 'Final validate_edge_results.py sub-gate ok=true' in state['remaining_for_completion']; "
                "assert 'Final target_readiness.json completion_ok=true with no before_matrix_blockers' in state['remaining_for_completion']; "
                "assert 'Final edge_results_bundle_verify.json ok=true and complete=true' not in state['remaining_for_completion']; "
                "assert 'Final edge_delivery_verify.json ok=true with all substatus checks passing' not in state['remaining_for_completion']; "
                "evidence=r['prompt_to_artifact_checklist']['final_two_model_results']['evidence']; "
                "assert evidence['results_bundle_target_completion_ok'] is True; "
                "assert evidence['delivery_target_completion_ok'] is True; "
                "missing=';'.join(r['prompt_to_artifact_checklist']['final_two_model_results']['missing_or_unverified']); "
                "assert 'Final target_readiness.json completion_ok=true with no before_matrix_blockers is missing' in missing; "
                "assert 'Final edge_results_bundle_verify.json ok=true and complete=true is missing' not in missing; "
                "assert 'Final edge_delivery_verify.json ok=true with all substatus checks passing is missing' not in missing",
            ],
        ),
        (
            "goal_audit_summarizes_asset_failures",
            [
                sys.executable,
                "-c",
                "import json, subprocess, sys, tempfile; "
                "from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_goal_asset_failures_')); "
                "[(root/name).write_text('{}' if name.endswith('.json') else 'x') for name in ['matrix_config_resolved.json','edge_preflight_report.json','edge_prewarm_report.json','matrix_results.json','matrix_results.csv','edge_report.md','completion_audit.json']]; "
                "(root/'edge_asset_audit.json').write_text(json.dumps({'ok':False,'missing_files':['calibration_file'],'low_disk':[{'name':'output_root'}],'unwritable_paths':[{'name':'output_root'}],'models':[{'name':'LLaMA3-8B','path':'/missing','required':True,'checkpoint_ready':False,'required_for_methods':['dense'],'weight_status':{'missing':['weight_files']}}],'precompressed':[{'model':'Qwen3-8B','method':'gptq','path':'/missing-gptq','checkpoint_ready':False,'weight_status':{'missing':['weight_files']}}]})); "
                "out=root/'goal.json'; "
                "p=subprocess.run([sys.executable,'tools/build_goal_audit.py','--final-output-root',str(root),'--output',str(out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "asset=json.loads(out.read_text())['final_artifacts']['asset_audit']; "
                "assert asset['missing_files'] == ['calibration_file']; "
                "assert asset['unwritable_paths'] == [{'name':'output_root'}]; "
                "assert asset['required_models_not_ready'][0]['name'] == 'LLaMA3-8B'; "
                "assert asset['precompressed_not_ready'][0]['method'] == 'gptq'",
            ],
        ),
        (
            "collect_mmlu_aggregate",
            [
                sys.executable,
                "tools/collect_edge_results.py",
                "--root",
                collect_fixture["root"],
                "--output-json",
                "/tmp/llmcbench_collect_smoke/matrix_results.json",
                "--output-csv",
                "/tmp/llmcbench_collect_smoke/matrix_results.csv",
            ],
        ),
        (
            "collect_mmlu_aggregate_assert",
            [
                sys.executable,
                "-c",
                "import json; rows=json.load(open('/tmp/llmcbench_collect_smoke/matrix_results.json')); "
                "assert rows[0]['mmlu_score'] == 0.25; "
                "assert rows[0]['mmlu_score_metric'] == 'aggregate:acc,none'; "
                "assert rows[0]['arc_challenge_score'] == 0.55; "
                "assert rows[0]['arc_challenge_score_metric'] == 'acc_norm,none'; "
                "assert rows[0]['accuracy_failed_task_count'] == 0; "
                "assert rows[0]['accuracy_dtype'] == 'fp16'; "
                "assert rows[0]['accuracy_lm_eval_dtype'] == 'float16'; "
                "assert rows[0]['accuracy_batch_size'] == 'auto'; "
                "assert rows[0]['accuracy_num_fewshot'] is None; "
                "assert rows[0]['accuracy_max_memory'] == '0:22GiB,cpu:120GiB'; "
                "assert rows[0]['accuracy_split_tasks'] is True; "
                "assert rows[0]['accuracy_continue_on_error'] is True; "
                "assert rows[0]['deployment_memory_limit_exceeded'] is False; "
                "assert rows[0]['deployment_dtype'] == 'fp16'; "
                "assert rows[0]['deployment_max_memory'] == '0:22GiB,cpu:120GiB'; "
                "assert rows[0]['deployment_device_map'] == 'auto'; "
                "assert rows[0]['deployment_prompt_source'] == 'file'; "
                "assert rows[0]['deployment_prompt_count'] == 1; "
                "assert rows[0]['deployment_batch_size'] == 1; "
                "assert rows[0]['deployment_input_length'] == 32; "
                "assert rows[0]['deployment_input_tokens'] == 4; "
                "assert rows[0]['deployment_max_new_tokens'] == 1; "
                "assert rows[0]['deployment_warmup'] == 0; "
                "assert rows[0]['deployment_repeat'] == 1; "
                "assert rows[0]['deployment_elapsed_seconds'] == 1.0; "
                "assert rows[0]['resource_backend'] == 'model_load'; "
                "assert rows[0]['resource_model_load_error'] is None; "
                "assert rows[0]['resource_seqlen'] == 32; "
                "assert rows[0]['resource_prompt_source'] == 'file'; "
                "assert rows[0]['resource_prompt_path'] == '/tmp/prompts.txt'; "
                "assert rows[0]['resource_prompt_count'] == 1; "
                "assert rows[0]['resource_input_length'] == 32; "
                "assert rows[0]['resource_max_new_tokens'] == 1; "
                "assert rows[0]['resource_sparsity_sample_tensors'] == -1; "
                "assert rows[0]['source_model_path'] == '/tmp/tiny-source'; "
                "assert rows[0]['model_revision'] is None; "
                "assert rows[0]['compression_materialized'] is False; "
                "assert rows[0]['compression_uses_precompressed'] is False; "
                "assert rows[0]['compression_note'] == 'collect fixture dense checkpoint'; "
                "assert rows[0]['compression_dtype'] == 'fp16'; "
                "assert rows[0]['compression_seqlen'] == 32; "
                "assert rows[0]['compression_quant_bits'] == 4 and rows[0]['compression_group_size'] == 128; "
                "assert rows[0]['compression_desc_act'] is False; "
                "assert rows[0]['compression_awq_version'] == 'GEMM'; "
                "assert rows[0]['compression_calibration_samples'] == 2; "
                "assert rows[0]['compression_calibration_batch_size'] == 1; "
                "assert rows[0]['effective_storage_bytes_quantized_approx'] is None; "
                "assert rows[0]['effective_storage_gb_quantized_approx'] is None; "
                "assert rows[0]['effective_storage_quantized_bits'] is None; "
                "assert rows[0]['effective_storage_quantized_backend'] is None; "
                "assert rows[0]['runtime_python'] == '3.10.0'; "
                "assert rows[0]['runtime_profile'] == 'stable'; "
                "assert rows[0]['runtime_cpu_count'] == 16; "
                "assert rows[0]['runtime_system_memory_total_gb'] == 256.0; "
                "assert rows[0]['accuracy_data_dir'] == '/tmp/datasets'; "
                "assert rows[0]['accuracy_hf_datasets_cache'] == '/tmp/datasets'; "
                "wanda=[row for row in rows if row['compression_method']=='wanda'][0]; "
                "assert wanda['accuracy_dtype'] == 'fp16'; "
                "assert wanda['compression_dtype'] == 'fp16'; "
                "assert round(wanda['accuracy_delta_vs_dense'], 6) == -0.09; "
                "assert round(wanda['accuracy_retention_vs_dense'], 6) == round(0.51/0.6, 6); "
                "assert rows[0]['storage_gb'] == 100 / (1024**3); "
                "assert rows[0]['effective_storage_gb_sparse_approx'] == 100 / (1024**3); "
                "assert wanda['storage_gb'] == 100 / (1024**3); "
                "assert wanda['effective_storage_gb_sparse_approx'] == 50 / (1024**3); "
                "assert wanda['nonzero_ratio'] == 0.5; "
                "assert wanda['effective_flops_num_sparse_approx'] == 0.5",
            ],
        ),
        (
            "matrix_finalize_only",
            [
                sys.executable,
                "-c",
                "import subprocess,sys; from pathlib import Path; "
                "p=Path('/tmp/llmcbench_collect_smoke/matrix_config_resolved.json'); p.unlink(missing_ok=True); "
                "cmd=[sys.executable,'tools/run_edge_matrix.py','--config',"
                f"{validator_fixture['config']!r},'--output-root',{collect_fixture['root']!r},"
                "'--only-model','TinyModel','--only-method','dense','--allow-unexpected','--finalize-only']; "
                "r=subprocess.run(cmd,capture_output=True,text=True); "
                "assert r.returncode == 0, r.stderr + r.stdout; "
                "assert p.exists()",
            ],
        ),
        (
            "matrix_finalize_report_auxiliary_paths_assert",
            [
                sys.executable,
                "-c",
                "import json; r=json.load(open('/tmp/llmcbench_collect_smoke/matrix_finalize_report.json')); "
                "assert r['edge_asset_audit'] == '/tmp/llmcbench_collect_smoke/edge_asset_audit.json'; "
                "assert r['edge_preflight_report'] == '/tmp/llmcbench_collect_smoke/edge_preflight_report.json'; "
                "assert r['edge_prewarm_report'] == '/tmp/llmcbench_collect_smoke/edge_prewarm_report.json'; "
                "assert r['edge_handoff_manifest'] == '/tmp/llmcbench_collect_smoke/edge_handoff_manifest.json'; "
                "assert r['target_readiness'] == '/tmp/llmcbench_collect_smoke/target_readiness.json'; "
                "assert r['edge_prepare_assets_report'] == '/tmp/llmcbench_collect_smoke/edge_prepare_assets_report.json'; "
                "assert r['edge_prepare_metadata_report'] == '/tmp/llmcbench_collect_smoke/edge_prepare_metadata_report.json'",
            ],
        ),
        (
            "completion_audit",
            [
                sys.executable,
                "tools/audit_edge_completion.py",
                "--config",
                validator_fixture["config"],
                "--root",
                collect_fixture["root"],
                "--only-model",
                "TinyModel",
                "--only-method",
                "dense",
                "--allow-unexpected",
                "--require-code-match",
                "--require-prewarm-report",
                "--require-handoff-manifest",
                "--output",
                "/tmp/llmcbench_collect_smoke/completion_audit.json",
            ],
        ),
        (
            "completion_audit_resolved_config_assert",
            [
                sys.executable,
                "-c",
                "import json; r=json.load(open('/tmp/llmcbench_collect_smoke/completion_audit.json')); "
                "assert r['uses_resolved_config'] is True; "
                "assert r['validator_config'].endswith('/matrix_config_resolved.json'); "
                "assert r['edge_report_sections']['ok'] is True; "
                "assert '## Runtime' in r['edge_report_sections']['required_sections']; "
                "assert '## Preflight Checks' in r['edge_report_sections']['required_sections']; "
                "assert '## Asset Paths' in r['edge_report_sections']['required_sections']; "
                "assert '## Prewarm' in r['edge_report_sections']['required_sections']; "
                "assert '## Asset Preparation' in r['edge_report_sections']['required_sections']; "
                "assert r['matrix_results_csv_status']['row_count_matches'] is True; "
                "assert r['matrix_results_csv_status']['pair_set_matches'] is True; "
                "assert r['mmlu_subtask_coverage_status']['ok'] is True; "
                "assert r['mmlu_subtask_coverage_status']['required_subtask_count'] == 57; "
                "assert r['mmlu_subtask_coverage_status']['rows'][0]['present_count'] == 57; "
                "assert r['matrix_run_report_status']['ok'] is True; "
                "assert r['matrix_finalize_report_status']['ok'] is True; "
                "assert r['prewarm_report_status']['ok'] is True; "
                "assert r['prewarm_report_status']['dtype'] == 'fp16'; "
                "assert r['prewarm_report_status']['expected_dtype'] == 'fp16'; "
                "assert r['prewarm_report_status']['dtype_mismatch'] is None; "
                "assert r['prewarm_report_status']['attempted_count'] >= 50; "
                "assert r['prewarm_report_status']['missing_required_tasks'] == []; "
                "assert r['prewarm_report_status']['missing_required_models'] == []; "
                "assert r['artifact_status']['edge_asset_audit'] is True; "
                "assert r['artifact_status']['edge_preflight_report'] is True; "
                "assert r['artifact_status']['edge_handoff_manifest'] is True; "
                "assert r['artifact_status']['edge_optional_backends_report'] is False; "
                "assert r['artifact_status']['edge_quant_runtime_report'] is False; "
                "assert r['artifact_sha256']['edge_asset_audit']; "
                "assert r['asset_storage_status']['ok'] is True; "
                "assert r['asset_storage_status']['media']['output_root']['is_non_rotational'] is True; "
                "assert r['artifact_sha256']['edge_preflight_report']; "
                "assert r['artifact_sha256']['edge_handoff_manifest']; "
                "assert r['artifact_status']['edge_prewarm_report'] is True; "
                "assert r['artifact_sha256']['edge_prewarm_report']; "
                "assert r['handoff_manifest_status']['ok'] is True; "
                "assert r['handoff_manifest_status']['required'] is True; "
                "assert r['handoff_manifest_status']['code_matches_current']['commit'] is True; "
                "assert r['handoff_manifest_status']['code_dirty'] == r['code_status']['current']['dirty']; "
                "assert r['optional_backends_report_status']['ok'] is True; "
                "assert r['optional_backends_report_status']['required'] is False; "
                "assert r['quant_runtime_report_status']['ok'] is True; "
                "assert r['quant_runtime_report_status']['required'] is False; "
                "assert r['artifact_status']['edge_prepare_assets_report'] is True; "
                "assert r['artifact_status']['edge_prepare_metadata_report'] is True; "
                "assert r['artifact_sha256']['edge_prepare_assets_report']; "
                "assert r['artifact_sha256']['edge_prepare_metadata_report']; "
                "assert r['prepare_report_status']['ok'] is True; "
                "assert r['prepare_metadata_report_status']['ok'] is True; "
                "assert r['validator']['payload_ok'] is True; "
                "assert r['code_status']['all_present'] is True; "
                "assert r['code_status']['matches_current']['origin'] is True; "
                "assert 'dirty' in r['code_status']['current']; "
                "assert 'dirty' in r['code_status']['resolved']; "
                "assert r['artifact_sha256']['matrix_results_json']; "
                "assert r['artifact_sha256']['matrix_config_resolved']",
            ],
        ),
        (
            "results_bundle_rejects_incomplete",
            [
                sys.executable,
                "-c",
                "import shutil,subprocess,sys; from pathlib import Path; "
                "root=Path('/tmp/llmcbench_incomplete_results_bundle'); bundle=Path('/tmp/llmcbench_incomplete_results_bundle_out'); "
                "shutil.rmtree(root, ignore_errors=True); shutil.rmtree(bundle, ignore_errors=True); root.mkdir(); "
                "(root/'matrix_results.json').write_text('[]\\n'); "
                "p=subprocess.run([sys.executable,'tools/export_edge_results_bundle.py','--output-root',str(root),'--bundle-dir',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "assert 'result bundle is incomplete' in (p.stderr + p.stdout); "
                "assert not bundle.exists()",
            ],
        ),
        (
            "scorecard_export",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; from pathlib import Path; "
                "root=Path('/tmp/llmcbench_collect_smoke'); "
                "json_out=root/'edge_scorecard.json'; md_out=root/'edge_scorecard.md'; "
                "p=subprocess.run([sys.executable,'tools/summarize_edge_results.py','--results',str(root/'matrix_results.json'),'--output-json',str(json_out),'--output-md',str(md_out)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "summary=json.loads(json_out.read_text()); "
                "assert summary['schema_version'] == 1; "
                "assert summary['models'][0]['model_name'] == 'TinyModel'; "
                "assert summary['models'][0]['recommended_method'] in {'dense','wanda'}; "
                "assert summary['models'][0]['pareto_methods']; "
                "ranking={row['compression_method']: row for row in summary['models'][0]['ranking']}; "
                "assert ranking['wanda']['score_storage_gb'] < ranking['dense']['score_storage_gb']; "
                "assert ranking['wanda']['score_flops_num'] < ranking['dense']['score_flops_num']; "
                "assert 'LLMCBench Edge Result Scorecard' in md_out.read_text(); "
                "constrained=root/'edge_scorecard_constrained.json'; "
                "p2=subprocess.run([sys.executable,'tools/summarize_edge_results.py','--results',str(root/'matrix_results.json'),'--output-json',str(constrained),'--min-accuracy-retention','0.9','--weights','accuracy_avg_score=0.8,tokens_per_second=0.2,storage_gb=0,deployment_memory_measured_peak_gb=0,flops_num=0'],capture_output=True,text=True); "
                "assert p2.returncode == 0, p2.stderr + p2.stdout; "
                "c=json.loads(constrained.read_text()); "
                "assert c['scoring']['constraints']['min_accuracy_retention'] == 0.9; "
                "assert c['models'][0]['ranking'][0]['passes_constraints'] is True; "
                "assert any(not row['passes_constraints'] for row in c['models'][0]['ranking'])",
            ],
        ),
        (
            "results_bundle_export_complete",
            [
                sys.executable,
                "-c",
                "import csv,json,shutil,subprocess,sys,tarfile; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); root=Path('/tmp/llmcbench_results_bundle_complete_src'); bundle=Path('/tmp/llmcbench_results_bundle_smoke'); archive=Path('/tmp/llmcbench_results_bundle_smoke.tar.gz'); "
                "shutil.rmtree(root, ignore_errors=True); shutil.copytree(src, root); shutil.rmtree(bundle, ignore_errors=True); archive.unlink(missing_ok=True); "
                "resolved=json.loads((root/'matrix_config_resolved.json').read_text()); "
                "resolved['selected_methods']=['dense','wanda']; resolved['methods']=['dense','wanda']; "
                "(root/'matrix_config_resolved.json').write_text(json.dumps(resolved)); "
                "rows=json.loads((root/'matrix_results.json').read_text()); "
                "[row.update({'runtime_gpu_name':'NVIDIA RTX 4090'}) for row in rows]; "
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "csv_path=root/'matrix_results.csv'; csv_rows=list(csv.DictReader(open(csv_path))); "
                "[row.update({'runtime_gpu_name':'NVIDIA RTX 4090'}) for row in csv_rows]; "
                "handle=open(csv_path,'w',newline=''); writer=csv.DictWriter(handle, fieldnames=csv_rows[0].keys()); writer.writeheader(); writer.writerows(csv_rows); handle.close(); "
                "(root/'edge_optional_backends_report.json').write_text(json.dumps({'ok':True,'dry_run':False})); "
                "(root/'edge_quant_runtime_report.json').write_text(json.dumps({'ok':True,'items':[]})); "
                "(root/'target_readiness.json').write_text(json.dumps({'ok':True,'ready_for_matrix':True,'completion_ok':True,'before_matrix_blockers':[]})); "
                "subprocess.run([sys.executable,'tools/summarize_edge_results.py','--results',str(root/'matrix_results.json'),'--output-json',str(root/'edge_scorecard.json'),'--output-md',str(root/'edge_scorecard.md')],check=True,capture_output=True,text=True); "
                "p=subprocess.run([sys.executable,'tools/export_edge_results_bundle.py','--output-root',str(root),'--bundle-dir',str(bundle),'--archive',str(archive),'--include-job-logs','--require-complete','--force'],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "out=json.loads(p.stdout); "
                "assert out['complete'] is True; "
                "assert out['job_log_count'] == 2; "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "assert manifest['complete'] is True; "
                "assert manifest['completion_audit_status']['ok'] is True; "
                "assert manifest['completion_audit_status']['validator_ok'] is True; "
                "assert manifest['completion_audit_status']['validator_payload_ok'] is True; "
                "assert manifest['target_readiness_status']['ok'] is True; "
                "assert manifest['target_readiness_status']['completion_ok'] is True; "
                "assert manifest['missing_required_artifacts'] == []; "
                "assert 'Completion validator ok: `True`' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); "
                "assert 'Target completion ok: `True`' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); "
                "assert '`tools/verify_edge_results_bundle.py`' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); "
                "assert '`tools/verify_edge_delivery.py`' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); "
                "assert '`before_matrix_blockers`' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); "
                "assert 'JSON/CSV matrix consistency' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); "
                "assert 'score metric keys' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); "
                "assert 'compression provenance' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); "
                "assert 'quantized storage estimates' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); assert 'edge_target_paths_report.json' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); assert 'current_goal_audit.json' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); "
                "assert 'tools/verify_edge_delivery.py --root <output_root> --results-bundle . --results-archive <edge_results_bundle.tar.gz> --output <output_root>/edge_delivery_verify.json' in (bundle/'RESULTS_BUNDLE_README.md').read_text(); "
                "paths={item['path'] for item in manifest['files']}; "
                "assert 'matrix_results.json' in paths; "
                "assert 'matrix_results.csv' in paths; "
                "assert 'edge_report.md' in paths; "
                "assert 'edge_scorecard.json' in paths; "
                "assert 'edge_scorecard.md' in paths; "
                "assert 'edge_target_paths_report.json' in paths; "
                "assert 'completion_audit.json' in paths; assert 'current_goal_audit.json' in paths; "
                "assert 'tools/verify_edge_results_bundle.py' in paths; "
                "assert 'tools/verify_edge_delivery.py' in paths; "
                "assert 'tools/prewarm_lm_eval_tasks.py' in paths; "
                "assert 'RESULTS_BUNDLE_README.md' in paths; "
                "assert any(path.startswith('job_logs/') and path.endswith('/job.log') for path in paths); "
                "verify=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete','--require-target-readiness-ok','--require-target-runtime'],capture_output=True,text=True); "
                "assert verify.returncode == 0, verify.stderr + verify.stdout; "
                "verify_report=json.loads(verify.stdout); "
                "assert verify_report['ok'] is True; "
                "assert verify_report['complete'] is True; "
                "assert verify_report['readme_status']['ok'] is True; "
                "assert verify_report['readme_status']['missing_required_text'] == []; "
                "assert verify_report['completion_audit_status']['validator_ok'] is True; "
                "assert verify_report['completion_audit_status']['validator_payload_ok'] is True; "
                "assert verify_report['target_readiness_status']['completion_ok'] is True; "
                "assert verify_report['target_paths_report_status']['ok'] is True; "
                "assert verify_report['scorecard_status']['ok'] is True; "
                "assert verify_report['scorecard_status']['model_status'][0]['ok'] is True; "
                "assert verify_report['runtime_environment_status']['ok'] is True; "
                "assert verify_report['runtime_environment_status']['runtime_profiles'] == ['stable']; "
                "assert verify_report['matrix_coverage_status']['ok'] is True; "
                "assert verify_report['matrix_coverage_status']['expected_pair_count'] == 2; "
                "assert verify_report['score_coverage_status']['ok'] is True; "
                "assert verify_report['score_coverage_status']['required_mmlu_subtask_count'] == 57; "
                "assert verify_report['metric_coverage_status']['ok'] is True; "
                "assert 'tokens_per_second' in verify_report['metric_coverage_status']['deployment_metric_fields']; "
                "assert 'deployment_memory_limit_gb' in verify_report['metric_coverage_status']['deployment_metric_fields']; "
                "assert 'deployment_batch_size' in verify_report['metric_coverage_status']['deployment_metric_fields']; "
                "assert 'flops_num' in verify_report['metric_coverage_status']['resource_metric_fields']; "
                "assert archive.exists() and archive.stat().st_size > 0; "
                "tar=tarfile.open(archive); names=set(tar.getnames()); tar.close(); "
                "assert any(name.endswith('/RESULTS_BUNDLE_MANIFEST.json') for name in names)",
            ],
        ),
        (
            "results_bundle_export_rejects_missing_completion_validator",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_complete_src'); root=Path('/tmp/llmcbench_results_bundle_missing_export_validator_src'); bundle=Path('/tmp/llmcbench_results_bundle_missing_export_validator'); "
                "shutil.rmtree(root, ignore_errors=True); shutil.copytree(src, root); shutil.rmtree(bundle, ignore_errors=True); "
                "audit=json.loads((root/'completion_audit.json').read_text()); "
                "audit.pop('validator', None); "
                "(root/'completion_audit.json').write_text(json.dumps(audit)); "
                "p=subprocess.run([sys.executable,'tools/export_edge_results_bundle.py','--output-root',str(root),'--bundle-dir',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "assert 'completion_audit.json ok/validator.ok is not true' in (p.stderr + p.stdout); "
                "assert not bundle.exists()",
            ],
        ),
        (
            "results_bundle_export_rejects_incomplete_target_readiness",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_complete_src'); root=Path('/tmp/llmcbench_results_bundle_incomplete_target_src'); bundle=Path('/tmp/llmcbench_results_bundle_incomplete_target'); "
                "shutil.rmtree(root, ignore_errors=True); shutil.copytree(src, root); shutil.rmtree(bundle, ignore_errors=True); "
                "target=json.loads((root/'target_readiness.json').read_text()); "
                "target['completion_ok']=False; "
                "(root/'target_readiness.json').write_text(json.dumps(target)); "
                "p=subprocess.run([sys.executable,'tools/export_edge_results_bundle.py','--output-root',str(root),'--bundle-dir',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "assert 'target_readiness.json ok/completion_ok is not true' in (p.stderr + p.stdout); "
                "assert not bundle.exists()",
            ],
        ),
        (
            "results_bundle_rejects_missing_completion_validator",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_missing_validator'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "audit=json.loads((bundle/'completion_audit.json').read_text()); "
                "audit.pop('validator', None); "
                "(bundle/'completion_audit.json').write_text(json.dumps(audit)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=(bundle/'completion_audit.json').read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='completion_audit.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'completion_audit.json ok and validator.ok are not true' in r['violations']; "
                "assert r['completion_audit_status']['validator_exists'] is False; "
                "assert 'validator.ok is not true' in r['completion_audit_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_incomplete_target_readiness",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_incomplete_target'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "target=json.loads((bundle/'target_readiness.json').read_text()); "
                "target['completion_ok']=False; "
                "(bundle/'target_readiness.json').write_text(json.dumps(target)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=(bundle/'target_readiness.json').read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='target_readiness.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete','--require-target-readiness-ok'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'target_readiness.json ok and completion_ok are not true' in r['violations']; "
                "assert r['target_readiness_status']['completion_ok'] is False; "
                "assert 'completion_ok is not true' in r['target_readiness_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_target_readiness_blockers",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_blocked_target'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "target=json.loads((bundle/'target_readiness.json').read_text()); "
                "target['before_matrix_blockers']=[{'type':'artifact_not_ok','artifact':'preflight'}]; "
                "(bundle/'target_readiness.json').write_text(json.dumps(target)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=(bundle/'target_readiness.json').read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='target_readiness.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete','--require-target-readiness-ok'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'target_readiness.json ok and completion_ok are not true' in r['violations']; "
                "assert r['target_readiness_status']['before_matrix_blocker_count'] == 1; "
                "assert 'before_matrix_blockers is not empty' in r['target_readiness_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_bad_target_paths_report",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_bad_target_paths'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "target_paths=bundle/'edge_target_paths_report.json'; "
                "target_paths.write_text(json.dumps({'ok':True,'dry_run':False,'directories':{},'files':{}})); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=target_paths.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='edge_target_paths_report.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'edge_target_paths_report.json is not valid' in r['violations']; "
                "status=r['target_paths_report_status']; "
                "assert 'target paths directories are empty or missing' in status['violations']; "
                "assert 'target paths files do not cover required seed files' in status['violations']",
            ],
        ),
        (
            "results_bundle_rejects_wrong_cuda_visible_devices",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_bad_cuda_visible'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['runtime_cuda_visible_devices']='1'; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete','--require-target-readiness-ok','--require-target-runtime'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'matrix results target runtime verification failed' in r['violations']; "
                "row=r['runtime_environment_status']['rows'][0]; "
                "assert 'runtime_cuda_visible_devices does not match config default' in row['violations']",
            ],
        ),
        (
            "results_bundle_rejects_mixed_runtime_profiles",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_mixed_profile'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[1]['runtime_profile']='gptqmodel'; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete','--require-target-readiness-ok','--require-target-runtime'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'matrix results target runtime verification failed' in r['violations']; "
                "assert 'matrix_results.json mixes runtime_profile values' in r['runtime_environment_status']['violations']; "
                "assert r['runtime_environment_status']['runtime_profiles'] == ['gptqmodel','stable']",
            ],
        ),
        (
            "results_bundle_rejects_limited_accuracy",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_limited_accuracy'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['accuracy_limit']=1; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'matrix results score coverage verification failed' in r['violations']; "
                "row=r['score_coverage_status']['rows'][0]; "
                "assert 'accuracy_limit' in row['accuracy_provenance_violations']; "
                "assert 'matrix_results.json row has non-final accuracy provenance' in r['score_coverage_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_failed_accuracy_tasks",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_failed_accuracy'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['accuracy_failed_task_count']=1; rows[0]['accuracy_failed_tasks']='hellaswag'; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['score_coverage_status']['rows'][0]; "
                "assert 'accuracy_failed_task_count' in row['accuracy_provenance_violations']; "
                "assert 'accuracy_failed_tasks' in row['accuracy_provenance_violations']",
            ],
        ),
        (
            "results_bundle_rejects_missing_target_score_metric",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_missing_target_metric'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0].pop('piqa_score_metric', None); results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['score_coverage_status']['rows'][0]; "
                "assert 'piqa_score_metric' in row['missing_score_metric_fields']; "
                "assert 'matrix_results.json row is missing target task score metric fields' in r['score_coverage_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_missing_mmlu_subtask_metric",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_missing_mmlu_metric'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0].pop('mmlu_abstract_algebra_score_metric', None); results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['score_coverage_status']['rows'][0]; "
                "assert 'mmlu_abstract_algebra_score_metric' in row['missing_mmlu_subtask_metric_fields']; "
                "assert 'matrix_results.json row is missing MMLU subtask score metric fields' in r['score_coverage_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_wrong_target_score_metric",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_wrong_target_metric'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['piqa_score_metric']='acc,none'; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['score_coverage_status']['rows'][0]; "
                "assert any(item['field']=='piqa_score_metric' and item['expected']=='acc_norm,none' for item in row['score_metric_mismatches']); "
                "assert 'matrix_results.json row has unexpected target task score metric keys' in r['score_coverage_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_wrong_mmlu_subtask_metric",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_wrong_mmlu_metric'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['mmlu_abstract_algebra_score_metric']='acc_norm,none'; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['score_coverage_status']['rows'][0]; "
                "assert any(item['field']=='mmlu_abstract_algebra_score_metric' and item['expected']=='acc,none' for item in row['mmlu_subtask_metric_mismatches']); "
                "assert 'matrix_results.json row has unexpected MMLU subtask score metric keys' in r['score_coverage_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_wrong_mmlu_aggregate",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_wrong_mmlu_aggregate'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['mmlu_score']=0.99; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['score_coverage_status']['rows'][0]; "
                "assert any(item['field']=='mmlu_score' for item in row['score_aggregate_mismatches']); "
                "assert 'matrix_results.json row has inconsistent aggregate accuracy scores' in r['score_coverage_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_wrong_accuracy_average",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_wrong_accuracy_average'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['accuracy_avg_score']=0.99; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['score_coverage_status']['rows'][0]; "
                "assert any(item['field']=='accuracy_avg_score' for item in row['score_aggregate_mismatches']); "
                "assert 'matrix_results.json row has inconsistent aggregate accuracy scores' in r['score_coverage_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_wrong_compression_backend",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_wrong_compression_backend'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['compression_backend']='manual_backend'; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['compression_provenance_status']['rows'][0]; "
                "assert 'compression_backend is not allowed for method' in row['violations']; "
                "assert 'matrix results compression provenance verification failed' in r['violations']",
            ],
        ),
        (
            "results_bundle_rejects_inconsistent_precompressed_flag",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_bad_precompressed_flag'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['compression_backend']='precompressed_checkpoint'; rows[0]['compression_uses_precompressed']=False; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['compression_provenance_status']['rows'][0]; "
                "assert 'compression_uses_precompressed is inconsistent' in row['violations']",
            ],
        ),
        (
            "results_bundle_rejects_inconsistent_storage_gb",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_bad_storage_gb'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['storage_gb']=0.01; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['metric_coverage_status']['rows'][0]; "
                "assert 'storage_gb' in row['resource_consistency_violations']; "
                "assert 'matrix_results.json row has inconsistent resource estimates' in r['metric_coverage_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_inconsistent_sparse_flops",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_bad_sparse_flops'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[1]['effective_flops_num_sparse_approx']=0.75; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['metric_coverage_status']['rows'][1]; "
                "assert 'effective_flops_num_sparse_approx' in row['resource_consistency_violations']",
            ],
        ),
        (
            "results_bundle_rejects_inconsistent_tokens_per_second",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_bad_tps'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['tokens_per_second']=3.0; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['metric_coverage_status']['rows'][0]; "
                "assert 'tokens_per_second' in row['deployment_consistency_violations']; "
                "assert 'matrix_results.json row has inconsistent deployment metrics' in r['metric_coverage_status']['violations']",
            ],
        ),
        (
            "results_bundle_rejects_inconsistent_memory_peak",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_bad_memory_peak'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "results_path=bundle/'matrix_results.json'; rows=json.loads(results_path.read_text()); "
                "rows[0]['deployment_memory_measured_peak_gb']=0.05; results_path.write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=results_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "row=r['metric_coverage_status']['rows'][0]; "
                "assert 'deployment_memory_measured_peak_gb' in row['deployment_consistency_violations']",
            ],
        ),
        (
            "results_bundle_metric_status_rejects_bad_quantized_storage",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.verify_edge_results_bundle import _metric_coverage_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_bundle_quant_metric_')); "
                "row={'model_name':'TinyModel','compression_method':'gptq','tokens_per_second':1.0,"
                "'deployment_generated_tokens':1,'deployment_elapsed_seconds':1.0,'peak_memory_allocated_gb':0.1,"
                "'peak_memory_reserved_gb':0.1,'deployment_memory_limit_gb':24,'deployment_memory_measured_peak_gb':0.1,"
                "'deployment_memory_limit_exceeded':False,'storage_bytes':100,'storage_gb':100/(1024**3),"
                "'total_params':123,'compression_quant_bits':4,'sparsity':0.0,'nonzero_ratio':1.0,"
                "'effective_storage_bytes_sparse_approx':100,'effective_storage_gb_sparse_approx':100/(1024**3),"
                "'flops_num':1000000.0,'effective_flops_num_sparse_approx':1000000.0,"
                "'resource_backend':'model_load','compute_backend':'analytical_config_estimate',"
                "'effective_storage_bytes_quantized_approx':61.5,'effective_storage_gb_quantized_approx':61.5/(1024**3),"
                "'effective_storage_quantized_bits':8,'effective_storage_quantized_backend':'manual_estimate'}; "
                "(root/'matrix_results.json').write_text(json.dumps([row])); "
                "status=_metric_coverage_status(root); "
                "assert status['ok'] is False; "
                "bad=status['rows'][0]['quantized_storage_violations']; "
                "assert 'effective_storage_quantized_bits' in bad; "
                "assert 'effective_storage_quantized_backend' in bad; "
                "assert 'matrix_results.json row has invalid quantized storage estimates' in status['violations']",
            ],
        ),
        (
            "results_bundle_metric_status_rejects_quantized_fields_on_dense",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.verify_edge_results_bundle import _metric_coverage_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_bundle_dense_quant_metric_')); "
                "row={'model_name':'TinyModel','compression_method':'dense','tokens_per_second':1.0,"
                "'deployment_generated_tokens':1,'deployment_elapsed_seconds':1.0,'peak_memory_allocated_gb':0.1,"
                "'peak_memory_reserved_gb':0.1,'deployment_memory_limit_gb':24,'deployment_memory_measured_peak_gb':0.1,"
                "'deployment_memory_limit_exceeded':False,'storage_bytes':100,'storage_gb':100/(1024**3),"
                "'total_params':123,'sparsity':0.0,'nonzero_ratio':1.0,"
                "'effective_storage_bytes_sparse_approx':100,'effective_storage_gb_sparse_approx':100/(1024**3),"
                "'flops_num':1000000.0,'effective_flops_num_sparse_approx':1000000.0,"
                "'resource_backend':'model_load','compute_backend':'analytical_config_estimate',"
                "'effective_storage_bytes_quantized_approx':61.5,'effective_storage_gb_quantized_approx':61.5/(1024**3),"
                "'effective_storage_quantized_bits':4,'effective_storage_quantized_backend':'param_count_times_quant_bits'}; "
                "(root/'matrix_results.json').write_text(json.dumps([row])); "
                "status=_metric_coverage_status(root); "
                "assert status['ok'] is False; "
                "bad=status['rows'][0]['quantized_storage_violations']; "
                "assert 'effective_storage_quantized_backend' in bad",
            ],
        ),
        (
            "results_bundle_rejects_stale_csv_values",
            [
                sys.executable,
                "-c",
                "import csv,hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_stale_csv_values'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "csv_path=bundle/'matrix_results.csv'; rows=list(csv.DictReader(open(csv_path))); "
                "rows[0]['tokens_per_second']='999.0'; "
                "handle=open(csv_path,'w',newline=''); writer=csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows); handle.close(); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=csv_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.csv']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'matrix results CSV consistency verification failed' in r['violations']; "
                "mismatches=r['csv_consistency_status']['value_mismatches']; "
                "assert mismatches and mismatches[0]['column'] == 'tokens_per_second'",
            ],
        ),
        (
            "results_bundle_rejects_missing_csv_column",
            [
                sys.executable,
                "-c",
                "import csv,hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_missing_csv_column'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "csv_path=bundle/'matrix_results.csv'; rows=list(csv.DictReader(open(csv_path))); "
                "fieldnames=[name for name in rows[0].keys() if name != 'runtime_python']; "
                "handle=open(csv_path,'w',newline=''); writer=csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); "
                "writer.writerows([{key:value for key,value in row.items() if key in fieldnames} for row in rows]); handle.close(); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=csv_path.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.csv']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'runtime_python' in r['csv_consistency_status']['missing_columns']",
            ],
        ),
        (
            "delivery_verify_complete",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; from pathlib import Path; "
                "root=Path('/tmp/llmcbench_results_bundle_complete_src'); bundle=Path('/tmp/llmcbench_results_bundle_smoke'); archive=Path('/tmp/llmcbench_results_bundle_smoke.tar.gz'); "
                "p=subprocess.run([sys.executable,'tools/verify_edge_delivery.py','--root',str(root),'--results-bundle',str(bundle),'--results-archive',str(archive),'--output',str(root/'edge_delivery_verify.json')],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; "
                "r=json.loads((root/'edge_delivery_verify.json').read_text()); "
                "assert r['ok'] is True; "
                "assert r['completion_audit_status']['ok'] is True; "
                "assert r['completion_audit_status']['validator_ok'] is True; "
                "assert r['completion_audit_status']['validator_payload_ok'] is True; "
                "assert r['target_readiness_status']['ok'] is True; "
                "assert r['root_bundle_match_status']['ok'] is True; "
                "assert any(item['path']=='edge_target_paths_report.json' and item['ok'] is True for item in r['root_bundle_match_status']['files']); "
                "assert r['archive_bundle_match_status']['ok'] is True; "
                "assert r['results_bundle_status']['ok'] is True",
            ],
        ),
        (
            "delivery_verify_rejects_missing_root_completion_validator",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_complete_src'); bundle=Path('/tmp/llmcbench_results_bundle_smoke'); root=Path('/tmp/llmcbench_delivery_missing_root_validator'); "
                "shutil.rmtree(root, ignore_errors=True); shutil.copytree(src, root); "
                "audit=json.loads((root/'completion_audit.json').read_text()); "
                "audit.pop('validator', None); "
                "(root/'completion_audit.json').write_text(json.dumps(audit)); "
                "p=subprocess.run([sys.executable,'tools/verify_edge_delivery.py','--root',str(root),'--results-bundle',str(bundle)],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'completion_audit.json is not complete or validator.ok is not true' in r['violations']; "
                "assert r['completion_audit_status']['validator_exists'] is False; "
                "assert 'validator.ok is not true' in r['completion_audit_status']['violations']",
            ],
        ),
        (
            "delivery_verify_rejects_root_target_readiness_blockers",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_complete_src'); bundle=Path('/tmp/llmcbench_results_bundle_smoke'); root=Path('/tmp/llmcbench_delivery_blocked_root_target'); "
                "shutil.rmtree(root, ignore_errors=True); shutil.copytree(src, root); "
                "target=json.loads((root/'target_readiness.json').read_text()); "
                "target['before_matrix_blockers']=[{'type':'artifact_not_ok','artifact':'preflight'}]; "
                "(root/'target_readiness.json').write_text(json.dumps(target)); "
                "p=subprocess.run([sys.executable,'tools/verify_edge_delivery.py','--root',str(root),'--results-bundle',str(bundle)],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'target_readiness.json is not complete' in r['violations']; "
                "assert r['target_readiness_status']['before_matrix_blocker_count'] == 1; "
                "assert 'before_matrix_blockers is not empty' in r['target_readiness_status']['violations']",
            ],
        ),
        (
            "delivery_verify_rejects_archive_bundle_mismatch",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys,tarfile; from pathlib import Path; "
                "root=Path('/tmp/llmcbench_results_bundle_complete_src'); bundle=Path('/tmp/llmcbench_results_bundle_smoke'); src_archive=Path('/tmp/llmcbench_results_bundle_smoke.tar.gz'); bad=Path('/tmp/llmcbench_results_bundle_bad_archive.tar.gz'); tmp=Path('/tmp/llmcbench_results_bundle_bad_archive_tree'); "
                "shutil.rmtree(tmp, ignore_errors=True); bad.unlink(missing_ok=True); "
                "tar=tarfile.open(src_archive, 'r:gz'); tar.extractall(tmp); tar.close(); "
                "victim=next(tmp.rglob('matrix_results.json')); victim.unlink(); "
                "top=next(tmp.iterdir()); tar=tarfile.open(bad, 'w:gz'); tar.add(top, arcname=top.name); tar.close(); "
                "p=subprocess.run([sys.executable,'tools/verify_edge_delivery.py','--root',str(root),'--results-bundle',str(bundle),'--results-archive',str(bad)],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'results bundle archive does not match bundle directory' in r['violations']; "
                "archive_status=r['archive_bundle_match_status']; "
                "assert archive_status['ok'] is False; "
                "assert 'matrix_results.json' in archive_status['missing_archive_files']",
            ],
        ),
        (
            "delivery_verify_rejects_root_bundle_mismatch",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_complete_src'); bundle=Path('/tmp/llmcbench_results_bundle_smoke'); root=Path('/tmp/llmcbench_delivery_bad_root_bundle_match'); "
                "shutil.rmtree(root, ignore_errors=True); shutil.copytree(src, root); "
                "rows=json.loads((root/'matrix_results.json').read_text()); "
                "rows[0]['tokens_per_second']=123.0; "
                "(root/'matrix_results.json').write_text(json.dumps(rows)); "
                "(root/'edge_target_paths_report.json').write_text(json.dumps({'ok':True,'changed':True})); "
                "p=subprocess.run([sys.executable,'tools/verify_edge_delivery.py','--root',str(root),'--results-bundle',str(bundle)],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'root and results bundle artifacts do not match' in r['violations']; "
                "match=r['root_bundle_match_status']; "
                "assert match['ok'] is False; "
                "bad=[item for item in match['files'] if item['path']=='matrix_results.json'][0]; "
                "assert 'sha256 mismatch' in bad['violations']; "
                "bad_target=[item for item in match['files'] if item['path']=='edge_target_paths_report.json'][0]; "
                "assert 'sha256 mismatch' in bad_target['violations']",
            ],
        ),
        (
            "delivery_verify_rejects_missing_bundle",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_complete_src'); root=Path('/tmp/llmcbench_delivery_missing_bundle'); archive=Path('/tmp/llmcbench_results_bundle_smoke.tar.gz'); "
                "shutil.rmtree(root, ignore_errors=True); shutil.copytree(src, root); "
                "p=subprocess.run([sys.executable,'tools/verify_edge_delivery.py','--root',str(root),'--results-bundle',str(root/'missing_bundle'),'--results-archive',str(archive)],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert r['ok'] is False; "
                "assert 'results bundle verification failed' in r['violations']",
            ],
        ),
        (
            "results_bundle_rejects_missing_scorecard",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_missing_scorecard'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "(bundle/'edge_scorecard.json').unlink(); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "manifest['files']=[item for item in manifest['files'] if item['path']!='edge_scorecard.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'required bundle file is missing: edge_scorecard.json' in r['violations']; "
                "assert 'scorecard verification failed' in r['violations']",
            ],
        ),
        (
            "results_bundle_rejects_missing_delivery_verifier",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_missing_delivery_verifier'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "(bundle/'tools'/'verify_edge_delivery.py').unlink(); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "manifest['files']=[item for item in manifest['files'] if item['path']!='tools/verify_edge_delivery.py']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'required bundle file is missing: tools/verify_edge_delivery.py' in r['violations']",
            ],
        ),
        (
            "results_bundle_rejects_unlisted_required_file",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_unlisted_required'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "manifest['files']=[item for item in manifest['files'] if item['path']!='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'required bundle file is missing from manifest files: matrix_results.json' in r['violations']; "
                "status=[item for item in r['required_path_status'] if item['path']=='matrix_results.json'][0]; "
                "assert status['exists'] is True and status['in_manifest_files'] is False",
            ],
        ),
        (
            "results_bundle_rejects_duplicate_manifest_path",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_duplicate_manifest_path'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "manifest['files'].append(dict(manifest['files'][0])); "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert manifest['files'][0]['path'] in r['duplicate_manifest_paths']; "
                "assert 'manifest files contain duplicate paths' in r['violations']",
            ],
        ),
        (
            "results_bundle_rejects_stale_readme",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_stale_readme'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "readme=bundle/'RESULTS_BUNDLE_README.md'; "
                "readme.write_text(readme.read_text().replace('tools/verify_edge_delivery.py','tools/verify_edge_delivery_OLD.py')); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=readme.read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='RESULTS_BUNDLE_README.md']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'results bundle README verification failed' in r['violations']; "
                "assert 'tools/verify_edge_delivery.py' in r['readme_status']['missing_required_text']",
            ],
        ),
        (
            "results_bundle_rejects_scorecard_mismatch",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_bad_scorecard'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "score=json.loads((bundle/'edge_scorecard.json').read_text()); "
                "score['models'][0]['ranking']=score['models'][0]['ranking'][:1]; "
                "(bundle/'edge_scorecard.json').write_text(json.dumps(score)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=(bundle/'edge_scorecard.json').read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='edge_scorecard.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'scorecard verification failed' in r['violations']; "
                "assert 'ranking methods do not match matrix_results methods' in r['scorecard_status']['model_status'][0]['violations']",
            ],
        ),
        (
            "results_bundle_rejects_scorecard_metric_mismatch",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_bad_scorecard_metrics'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "rows=json.loads((bundle/'matrix_results.json').read_text()); "
                "rows[0]['tokens_per_second']=3.0; "
                "(bundle/'matrix_results.json').write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=(bundle/'matrix_results.json').read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'scorecard verification failed' in r['violations']; "
                "model=r['scorecard_status']['model_status'][0]; "
                "assert 'ranking metrics do not match matrix_results' in model['violations']; "
                "assert any(item['field']=='tokens_per_second' for item in model['metric_mismatches'])",
            ],
        ),
        (
            "results_bundle_rejects_memory_limit_exceeded",
            [
                sys.executable,
                "-c",
                "import hashlib,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_results_bundle_smoke'); bundle=Path('/tmp/llmcbench_results_bundle_bad_memory'); "
                "shutil.rmtree(bundle, ignore_errors=True); shutil.copytree(src, bundle); "
                "rows=json.loads((bundle/'matrix_results.json').read_text()); "
                "rows[0]['deployment_memory_limit_gb']=1; rows[0]['deployment_memory_measured_peak_gb']=2; rows[0]['deployment_memory_limit_exceeded']=False; "
                "(bundle/'matrix_results.json').write_text(json.dumps(rows)); "
                "manifest=json.loads((bundle/'RESULTS_BUNDLE_MANIFEST.json').read_text()); "
                "payload=(bundle/'matrix_results.json').read_bytes(); "
                "[item.update({'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}) for item in manifest['files'] if item['path']=='matrix_results.json']; "
                "(bundle/'RESULTS_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest)); "
                "p=subprocess.run([sys.executable,str(bundle/'tools'/'verify_edge_results_bundle.py'),'--bundle-root',str(bundle),'--require-complete'],capture_output=True,text=True); "
                "assert p.returncode != 0; "
                "r=json.loads(p.stdout); "
                "assert 'matrix results deployment/resource metric verification failed' in r['violations']; "
                "row=r['metric_coverage_status']['rows'][0]; "
                "assert 'deployment_memory_measured_peak_gb' in row['memory_limit_violations']; "
                "assert 'deployment_memory_limit_exceeded' in row['memory_limit_violations']",
            ],
        ),
        (
            "completion_audit_accepts_target_readiness_report",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_target_ready_ok'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "\nfor fp in dst.rglob('*'):\n"
                "    if fp.is_file():\n"
                "        text=fp.read_text(encoding='utf-8', errors='ignore')\n"
                "        fp.write_text(text.replace(str(src), str(dst)), encoding='utf-8')\n"
                "p0=dst/'matrix_finalize_report.json'; rf=json.load(open(p0)); "
                "rf['target_readiness']=str(dst/'target_readiness.json'); "
                "rf['steps'].append({'cmd':[sys.executable,'tools/audit_target_readiness.py'],'returncode':0,'ok':True}); "
                "p0.write_text(json.dumps(rf)); "
                "payload={'ok':True,'config':"
                f"{validator_fixture['config']!r}"
                ",'root':str(dst),'ready_for_matrix':True,'completion_ok':True,'before_matrix_blockers':[],"
                "'artifacts':{'completion_audit':{'path':str(dst/'completion_audit.json'),'exists':True,'ok':True}}}; "
                "(dst/'target_readiness.json').write_text(json.dumps(payload)); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-target-readiness-report','--output',str(dst/'completion_audit_target_ready_ok.json')],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout; out=json.load(open(dst/'completion_audit_target_ready_ok.json')); "
                "assert out['target_readiness_status']['ok'] is True; "
                "assert out['artifact_status']['target_readiness'] is True; "
                "assert out['artifact_sha256']['target_readiness']",
            ],
        ),
        (
            "completion_audit_rejects_missing_target_readiness_report",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_missing_target_readiness'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "\nfor fp in dst.rglob('*'):\n"
                "    if fp.is_file():\n"
                "        text=fp.read_text(encoding='utf-8', errors='ignore')\n"
                "        fp.write_text(text.replace(str(src), str(dst)), encoding='utf-8')\n"
                "(dst/'target_readiness.json').unlink(missing_ok=True); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-target-readiness-report','--output',str(dst/'completion_audit_missing_target_readiness.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.load(open(dst/'completion_audit_missing_target_readiness.json')); "
                "assert 'target_readiness.json' in ';'.join(r['missing_artifacts']); "
                "assert r['target_readiness_status']['ok'] is False; "
                "assert r['target_readiness_status']['exists'] is False",
            ],
        ),
        (
            "completion_audit_rejects_blocked_target_readiness_report",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_blocked_target_readiness'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "\nfor fp in dst.rglob('*'):\n"
                "    if fp.is_file():\n"
                "        text=fp.read_text(encoding='utf-8', errors='ignore')\n"
                "        fp.write_text(text.replace(str(src), str(dst)), encoding='utf-8')\n"
                "payload={'ok':False,'config':"
                f"{validator_fixture['config']!r}"
                ",'root':str(dst),'ready_for_matrix':False,'completion_ok':False,"
                "'before_matrix_blockers':[{'type':'missing_artifact','artifact':'preflight'}],"
                "'artifacts':{'completion_audit':{'path':str(dst/'completion_audit.json'),'exists':True,'ok':False}}}; "
                "(dst/'target_readiness.json').write_text(json.dumps(payload)); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-target-readiness-report','--output',str(dst/'completion_audit_blocked_target_readiness.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.load(open(dst/'completion_audit_blocked_target_readiness.json')); "
                "status=r['target_readiness_status']; "
                "assert status['ok'] is False; "
                "assert status['completion_ok'] is False; "
                "assert status['ready_for_matrix'] is False; "
                "assert any(item.get('field') == 'before_matrix_blockers' for item in status['violations'])",
            ],
        ),
        (
            "completion_audit_rejects_empty_prewarm_report",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _prewarm_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_empty_prewarm_')); "
                "p=root/'edge_prewarm_report.json'; p.write_text(json.dumps({'ok':True,'tasks':[],'attempted':[],'failed':[]})); "
                "status=_prewarm_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert 'tasks is empty or missing' in status['violations']; "
                "assert 'attempted is empty or missing' in status['violations']; "
                "assert 'succeeded is empty or missing' in status['violations']",
            ],
        ),
        (
            "completion_audit_rejects_failed_asset_report",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _simple_ok_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_asset_report_')); "
                "p=root/'edge_asset_audit.json'; p.write_text(json.dumps({'ok':False,'blockers':[{'reason':'repo_not_configured'}],'items':[{'ok':False,'method':'omniquant'}]})); "
                "status=_simple_ok_report_status(p, True, 'asset audit'); "
                "assert status['ok'] is False and status['report_ok'] is False; "
                "assert status['blockers'][0]['reason'] == 'repo_not_configured'; "
                "assert status['failed_items'][0]['method'] == 'omniquant'",
            ],
        ),
        (
            "completion_audit_rejects_empty_asset_audit_report",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _asset_audit_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_empty_asset_audit_')); "
                "p=root/'edge_asset_audit.json'; p.write_text(json.dumps({'ok':True})); "
                "status=_asset_audit_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert 'asset audit models must be a list' in status['violations']; "
                "assert 'asset audit unwritable_paths must be a list' in status['violations']; "
                "assert 'asset audit paths must be a JSON object' in status['violations']; "
                "paths={name:{'path':'/tmp/'+name} for name in ['hf_cache_dir','data_dir','output_root','calibration_file','prompt_file']}; "
                "p.write_text(json.dumps({'ok':True,'models':[],'precompressed':[],'missing_files':[],'low_disk':[],'unwritable_paths':[],'paths':paths})); "
                "status=_asset_audit_report_status(p, True); "
                "assert status['ok'] is True; "
                "assert status['missing_path_keys'] == []",
            ],
        ),
        (
            "completion_audit_rejects_rotational_storage",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_rotational_storage'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'edge_asset_audit.json'; r=json.load(open(p)); "
                "r['paths']['output_root']['block_device']['is_non_rotational']=False; "
                "r['paths']['output_root']['block_device']['rotational']=True; p.write_text(json.dumps(r)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-non-rotational-storage','--output',str(dst/'completion_audit_rotational_storage.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_rotational_storage.json')); "
                "assert out['asset_storage_status']['ok'] is False; "
                "assert out['asset_storage_status']['violations'][0]['name'] == 'output_root'",
            ],
        ),
        (
            "completion_audit_rejects_failed_prepare_report",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _simple_ok_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_prepare_report_')); "
                "p=root/'edge_prepare_assets_report.json'; p.write_text(json.dumps({'ok':False})); "
                "status=_simple_ok_report_status(p, False, 'asset prepare'); "
                "assert status['ok'] is False; "
                "assert status['required'] is False; "
                "assert status['report_ok'] is False",
            ],
        ),
        (
            "completion_audit_rejects_bad_target_paths_report",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _target_paths_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_bad_target_paths_')); "
                "p=root/'edge_target_paths_report.json'; p.write_text(json.dumps({'ok':True,'dry_run':False,'directories':{},'files':{}})); "
                "status=_target_paths_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert 'target paths directories are empty or missing' in status['violations']; "
                "assert 'target paths files are empty or missing' in status['violations']; "
                "assert 'target paths files do not cover required seed files' in status['violations']; "
                "p.write_text(json.dumps({'ok':True,'dry_run':False,'directories':{'output_root':{'ok':False,'path':'/x','error':'denied'}},'files':{'calibration_file':{'ok':True,'path':'/tmp/c'},'prompt_file':{'ok':False,'path':'/tmp/p','error':'denied'}}})); "
                "status=_target_paths_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert status['failed_directories'][0]['name'] == 'output_root'; "
                "assert status['failed_files'][0]['name'] == 'prompt_file'; "
                "assert 'target path directory items failed' in status['violations']; "
                "assert 'target path file items failed' in status['violations']; "
                "p.write_text(json.dumps({'ok':True,'dry_run':False,'directories':{'output_root':{'ok':True,'path':'/x'}},'files':{'calibration_file':{'ok':True,'path':'/tmp/c'},'prompt_file':{'ok':True,'path':'/tmp/p'}}})); "
                "status=_target_paths_report_status(p, True); "
                "assert status['ok'] is True",
            ],
        ),
        (
            "completion_audit_rejects_bad_full_prepare_report",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _prepare_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_bad_prepare_report_')); "
                "p=root/'edge_prepare_assets_report.json'; p.write_text(json.dumps({'ok':True,'dry_run':False,'metadata_only':False,'models':[],'files':[]})); "
                "status=_prepare_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert 'asset prepare models are empty or missing' in status['violations']; "
                "assert 'asset prepare files are empty or missing' in status['violations']; "
                "p.write_text(json.dumps({'ok':True,'dry_run':False,'metadata_only':False,'models':[{'name':'m1','ok':False,'error':'missing'}],'files':[{'path':'/tmp/calib','ok':False,'error':'missing'}]})); "
                "status=_prepare_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert status['failed_models'][0]['name'] == 'm1'; "
                "assert status['failed_files'][0]['path'] == '/tmp/calib'; "
                "assert 'asset prepare model items failed' in status['violations']; "
                "assert 'asset prepare file items failed' in status['violations']",
            ],
        ),
        (
            "completion_audit_rejects_bad_metadata_prepare_report",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _prepare_metadata_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_bad_metadata_prepare_')); "
                "p=root/'edge_prepare_metadata_report.json'; p.write_text(json.dumps({'ok':True,'dry_run':False,'metadata_only':False,'models':[],'files':[]})); "
                "status=_prepare_metadata_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert 'metadata prepare report must be metadata_only' in status['violations']; "
                "assert 'metadata prepare models are empty or missing' in status['violations']; "
                "p.write_text(json.dumps({'ok':True,'dry_run':False,'metadata_only':True,'models':[{'name':'m1','ok':False,'error':'missing'}],'files':[{'path':'/tmp/calib','ok':False,'error':'missing'}]})); "
                "status=_prepare_metadata_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert status['failed_models'][0]['name'] == 'm1'; "
                "assert status['failed_files'][0]['path'] == '/tmp/calib'; "
                "p.write_text(json.dumps({'ok':True,'dry_run':False,'metadata_only':True,'models':[{'name':'m1','ok':True}],'files':[{'path':'/tmp/calib','ok':True}]})); "
                "status=_prepare_metadata_report_status(p, True); "
                "assert status['ok'] is True",
            ],
        ),
        (
            "completion_audit_rejects_dry_run_auxiliary_reports",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _optional_backends_report_status, _simple_ok_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_dry_aux_report_')); "
                "p=root/'edge_prepare_assets_report.json'; p.write_text(json.dumps({'ok':True,'dry_run':True})); "
                "status=_simple_ok_report_status(p, True, 'asset prepare'); "
                "assert status['ok'] is False and status['dry_run'] is True; "
                "assert 'dry-run report is not final evidence' in status['violations']; "
                "p.write_text(json.dumps({'ok':True,'metadata_only':True})); "
                "status=_simple_ok_report_status(p, True, 'asset prepare'); "
                "assert status['ok'] is False and status['metadata_only'] is True; "
                "assert 'metadata-only report is not final evidence' in status['violations']; "
                "meta=root/'edge_prepare_metadata_report.json'; meta.write_text(json.dumps({'ok':True,'metadata_only':True})); "
                "meta_status=_simple_ok_report_status(meta, True, 'metadata prepare'); "
                "assert meta_status['ok'] is True and meta_status['metadata_only'] is True; "
                "optional=root/'edge_optional_backends_report.json'; optional.write_text(json.dumps({'ok':True,'dry_run':True,'mode':'dry_run'})); "
                "optional_status=_optional_backends_report_status(optional, True); "
                "assert optional_status['ok'] is False; "
                "assert 'optional backend requested_packages are empty or missing' in optional_status['violations']; "
                "optional.write_text(json.dumps({'ok':True,'dry_run':True,'mode':'dry_run','requested_packages':['optimum','auto-gptq'],'current_imports':{'optimum':{'ok':True},'auto_gptq':{'ok':True}},'pip_returncode':0})); "
                "optional_status=_optional_backends_report_status(optional, True); "
                "assert optional_status['ok'] is True; "
                "assert optional_status['needs_gptqmodel_runtime'] is False; "
                "optional.write_text(json.dumps({'ok':True,'dry_run':True,'mode':'dry_run','requested_packages':['optimum','auto-gptq','gptqmodel==4.0.0'],'current_imports':{'optimum':{'ok':True},'auto_gptq':{'ok':True},'gptqmodel':{'ok':True,'skipped':True}},'pip_returncode':0})); "
                "optional_status=_optional_backends_report_status(optional, True); "
                "assert optional_status['ok'] is False; "
                "assert optional_status['needs_gptqmodel_runtime'] is True; "
                "assert 'optional backend gptqmodel_install is missing' in optional_status['violations']; "
                "optional.write_text(json.dumps({'ok':True,'dry_run':True,'mode':'dry_run','requested_packages':['optimum','auto-gptq','gptqmodel==4.0.0'],'current_imports':{'optimum':{'ok':True},'auto_gptq':{'ok':True},'gptqmodel':{'ok':True,'skipped':True}},'gptqmodel_install':{'package':'gptqmodel==4.0.0','uses_no_build_isolation':True,'requires_python_3_10':True,'requirements_ok':False,'pip_returncode':0},'pip_returncode':0})); "
                "optional_status=_optional_backends_report_status(optional, True); "
                "assert optional_status['ok'] is False; "
                "assert 'optional backend gptqmodel requirements are not satisfied' in optional_status['violations']; "
                "optional.write_text(json.dumps({'ok':True,'dry_run':True,'mode':'dry_run','requested_packages':['optimum','auto-gptq','gptqmodel==4.0.0'],'current_imports':{'optimum':{'ok':True},'auto_gptq':{'ok':True},'gptqmodel':{'ok':True}},'gptqmodel_install':{'package':'gptqmodel==4.0.0','uses_no_build_isolation':True,'requires_python_3_10':True,'requirements_ok':True,'pip_returncode':0},'pip_returncode':0})); "
                "optional_status=_optional_backends_report_status(optional, True); "
                "assert optional_status['ok'] is True; "
                "optional.write_text(json.dumps({'ok':True,'dry_run':True,'mode':'dry_run','requested_packages':['autoawq'],'current_imports':{'awq':{'ok':True}},'pip_returncode':0})); "
                "optional_status=_optional_backends_report_status(optional, True); "
                "assert optional_status['ok'] is True and optional_status['dry_run'] is True",
            ],
        ),
        (
            "completion_audit_rejects_empty_quant_runtime_report",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _quant_runtime_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_quant_report_')); "
                "p=root/'edge_quant_runtime_report.json'; p.write_text(json.dumps({'ok':True,'rows':[]})); "
                "status=_quant_runtime_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert 'quant runtime rows are empty or missing' in status['violations']; "
                "p.write_text(json.dumps({'ok':True,'rows':[{'model':'m1','method':'gptq','model_path':'/tmp/m1','ok':False,'returncode':1}]})); "
                "status=_quant_runtime_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert status['failed_rows'][0]['method'] == 'gptq'; "
                "assert status['entry_source_field'] == 'rows'; "
                "assert 'quant runtime rows failed' in status['violations']; "
                "p.write_text(json.dumps({'ok':True,'probes':[{'model':'m1','method':'awq','path':'/tmp/m1-awq','ok':True,'returncode':0}]})); "
                "status=_quant_runtime_report_status(p, True); "
                "assert status['ok'] is True; "
                "assert status['entry_source_field'] == 'probes'; "
                "assert status['row_count'] == 1",
            ],
        ),
        (
            "completion_audit_rejects_bad_precompressed_prepare_report",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _prepare_precompressed_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_bad_precompressed_report_')); "
                "p=root/'edge_prepare_precompressed_report.json'; p.write_text(json.dumps({'ok':True,'dry_run':False,'metadata_only':False,'items':[],'blockers':[],'updated_config':str(root/'cfg.json')})); "
                "status=_prepare_precompressed_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert 'precompressed preparation items are empty or missing' in status['violations']; "
                "p.write_text(json.dumps({'ok':True,'dry_run':False,'metadata_only':False,'items':[{'model':'m1','method':'gptq','ok':False,'reason':'download_failed'}],'blockers':[{'model':'m1','method':'gptq','reason':'download_failed'}],'updated_config':str(root/'cfg.json')})); "
                "status=_prepare_precompressed_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert status['failed_items'][0]['method'] == 'gptq'; "
                "assert 'precompressed preparation blockers are present' in status['violations']; "
                "p.write_text(json.dumps({'ok':True,'dry_run':False,'metadata_only':False,'items':[{'model':'m1','method':'gptq','ok':True}],'blockers':[],'updated_config':str(root/'missing_cfg.json')})); "
                "status=_prepare_precompressed_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert status['updated_config_exists'] is False; "
                "assert 'precompressed updated_config file is missing' in status['violations']; "
                "p.write_text(json.dumps({'ok':True,'dry_run':False,'metadata_only':False,'items':[{'model':'m1','method':'gptq','ok':True}],'blockers':[]})); "
                "status=_prepare_precompressed_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert 'precompressed updated_config is missing' in status['violations']; "
                "(root/'cfg.json').write_text(json.dumps({'precompressed_paths':{'m1':{'gptq':'/tmp/m1'}}})); "
                "p.write_text(json.dumps({'ok':True,'dry_run':False,'metadata_only':False,'items':[{'model':'m1','method':'gptq','ok':True}],'blockers':[],'updated_config':str(root/'cfg.json')})); "
                "status=_prepare_precompressed_report_status(p, True, str(root/'other.json')); "
                "assert status['ok'] is False; "
                "assert 'precompressed updated_config does not match audit config' in status['violations']; "
                "status=_prepare_precompressed_report_status(p, True, str(root/'cfg.json')); "
                "assert status['ok'] is True; "
                "assert status['updated_config_exists'] is True",
            ],
        ),
        (
            "completion_audit_rejects_missing_asset_preflight_reports",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_missing_aux_reports'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "(dst/'edge_asset_audit.json').unlink(); (dst/'edge_preflight_report.json').unlink(); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-asset-audit-report','--require-preflight-report','--output',str(dst/'completion_audit_missing_aux_reports.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.load(open(dst/'completion_audit_missing_aux_reports.json')); "
                "missing=';'.join(r['missing_artifacts']); "
                "assert 'edge_asset_audit.json' in missing; "
                "assert 'edge_preflight_report.json' in missing",
            ],
        ),
        (
            "completion_audit_rejects_empty_preflight_steps",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_empty_preflight'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "(dst/'edge_preflight_report.json').write_text(json.dumps({'ok':True,'steps':[]})); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-preflight-report','--output',str(dst/'completion_audit_empty_preflight.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; out=json.load(open(dst/'completion_audit_empty_preflight.json')); "
                "status=out['preflight_report_status']; "
                "assert status['ok'] is False; "
                "assert 'preflight steps are empty or missing' in status['violations']",
            ],
        ),
        (
            "completion_audit_requires_optional_preflight_steps_when_gated",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _preflight_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_preflight_optional_steps_')); "
                "p=root/'edge_preflight_report.json'; "
                "verify_payload={'cuda_visible_devices':{'actual':'0','expected':'0','matches_config':True,'required':True}}; "
                "steps=[{'name':name,'required':True,'ok':True,'returncode':0} for name in ['asset_audit','method_readiness','matrix_dry_run']]; "
                "steps.insert(1, {'name':'verify_env','required':True,'ok':True,'returncode':0,'payload':verify_payload}); "
                "p.write_text(json.dumps({'ok':True,'steps':steps})); "
                "status=_preflight_report_status(p, True, True, True); "
                "assert status['ok'] is False; "
                "assert status['missing_steps'] == ['optional_backend_resolution', 'quant_runtime']; "
                "assert 'preflight required steps are missing' in status['violations']; "
                "steps.extend([{'name':'optional_backend_resolution','required':False,'ok':True,'returncode':0},{'name':'quant_runtime','required':False,'ok':True,'returncode':0}]); "
                "p.write_text(json.dumps({'ok':True,'steps':steps})); "
                "status=_preflight_report_status(p, True, True, True); "
                "assert status['ok'] is True; "
                "assert status['missing_steps'] == []",
            ],
        ),
        (
            "completion_audit_rejects_preflight_cuda_mismatch",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _preflight_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_preflight_cuda_mismatch_')); "
                "p=root/'edge_preflight_report.json'; "
                "verify_payload={'cuda_visible_devices':{'actual':'1','expected':'0','matches_config':False,'required':True}}; "
                "steps=[{'name':'asset_audit','required':True,'ok':True,'returncode':0},"
                "{'name':'verify_env','required':True,'ok':True,'returncode':0,'payload':verify_payload},"
                "{'name':'method_readiness','required':True,'ok':True,'returncode':0},"
                "{'name':'matrix_dry_run','required':True,'ok':True,'returncode':0}]; "
                "p.write_text(json.dumps({'ok':True,'steps':steps})); "
                "status=_preflight_report_status(p, True, False, False); "
                "assert status['ok'] is False; "
                "assert status['verify_env_cuda_visible_devices']['matches_config'] is False; "
                "assert 'preflight verify_env CUDA_VISIBLE_DEVICES did not match config' in status['violations']",
            ],
        ),
        (
            "completion_audit_rejects_missing_handoff_manifest",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_missing_handoff'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "(dst/'edge_handoff_manifest.json').unlink(); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-handoff-manifest','--output',str(dst/'completion_audit_missing_handoff.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.load(open(dst/'completion_audit_missing_handoff.json')); "
                "assert 'edge_handoff_manifest.json' in ';'.join(r['missing_artifacts']); "
                "assert r['handoff_manifest_status']['ok'] is False; "
                "assert r['handoff_manifest_status']['exists'] is False",
            ],
        ),
        (
            "completion_audit_rejects_bad_handoff_manifest",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_handoff'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "(dst/'edge_handoff_manifest.json').write_text(json.dumps({'schema_version':1,'commands':{}})); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-handoff-manifest','--output',str(dst/'completion_audit_bad_handoff.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.load(open(dst/'completion_audit_bad_handoff.json')); "
                "assert r['handoff_manifest_status']['ok'] is False; "
                "assert any('commands.setup' in item for item in r['handoff_manifest_status']['violations'])",
            ],
        ),
        (
            "completion_audit_rejects_weak_handoff_finalize_commands",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_weak_handoff_finalize'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'edge_handoff_manifest.json'; r=json.load(open(p)); "
                "r['commands']['manual_finalize']=[cmd.replace('--require-handoff-manifest ', '') for cmd in r['commands']['manual_finalize']]; "
                "p.write_text(json.dumps(r)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-handoff-manifest','--output',str(dst/'completion_audit_weak_handoff_finalize.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_weak_handoff_finalize.json')); "
                "violations=out['handoff_manifest_status']['violations']; "
                "assert any('commands.manual_finalize[0] missing --require-handoff-manifest' in item for item in violations); "
                "assert any('commands.manual_finalize[1] missing --require-handoff-manifest' in item for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_stale_handoff_code",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_stale_handoff_code'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'edge_handoff_manifest.json'; r=json.load(open(p)); "
                "r['code']['commit']='stale'; p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-handoff-manifest','--require-code-match','--output',str(dst/'completion_audit_stale_handoff_code.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_stale_handoff_code.json')); "
                "assert out['handoff_manifest_status']['code_matches_current']['commit'] is False; "
                "assert 'code.commit does not match current checkout' in out['handoff_manifest_status']['violations']",
            ],
        ),
        (
            "completion_audit_rejects_missing_config_prewarm_tasks",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _expand_prewarm_tasks, _prewarm_report_status, _split_tasks; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_missing_config_prewarm_')); "
                "p=root/'edge_prewarm_report.json'; "
                "p.write_text(json.dumps({'ok':True,'tasks':['piqa'],'target_count':1,'targets':[{'name':'m1'}],'attempted':[{'model':'m1','task':'piqa'}],'succeeded':[{'model':'m1','task':'piqa'}],'failed':[]})); "
                "required=_expand_prewarm_tasks(_split_tasks('mmlu,arc_challenge,hellaswag,piqa,winogrande')); "
                "status=_prewarm_report_status(p, True, required); "
                "assert status['ok'] is False; "
                "assert 'mmlu_abstract_algebra' in status['missing_required_tasks']; "
                "assert 'prewarm tasks do not cover configured matrix tasks' in status['violations']",
            ],
        ),
        (
            "completion_audit_rejects_dry_run_prewarm_report",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _prewarm_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_prewarm_dry_final_')); "
                "p=root/'edge_prewarm_report.json'; "
                "p.write_text(json.dumps({'ok':True,'dry_run':True,'tasks':['piqa'],'target_count':1,'targets':[{'name':'m1'}],'attempted':[{'model':'m1','task':'piqa'}],'succeeded':[{'model':'m1','task':'piqa'}],'failed':[]})); "
                "status=_prewarm_report_status(p, True, ['piqa'], ['m1']); "
                "assert status['ok'] is False and status['dry_run'] is True; "
                "assert 'prewarm dry-run report is not final evidence' in status['violations']",
            ],
        ),
        (
            "completion_audit_rejects_missing_config_prewarm_models",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _prewarm_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_missing_model_prewarm_')); "
                "p=root/'edge_prewarm_report.json'; "
                "p.write_text(json.dumps({'ok':True,'tasks':['piqa'],'target_count':1,'targets':[{'name':'m1'}],'attempted':[{'model':'m1','task':'piqa'}],'succeeded':[{'model':'m1','task':'piqa'}],'failed':[]})); "
                "status=_prewarm_report_status(p, True, ['piqa'], ['m1','m2']); "
                "assert status['ok'] is False; "
                "assert status['missing_required_models'] == ['m2']; "
                "assert 'prewarm targets do not cover configured matrix models' in status['violations']",
            ],
        ),
        (
            "completion_audit_rejects_incomplete_multi_model_prewarm",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _prewarm_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_multi_prewarm_')); "
                "p=root/'edge_prewarm_report.json'; "
                "p.write_text(json.dumps({'ok':True,'tasks':['piqa','arc_challenge'],'target_count':2,'attempted':[{'model':'m1','task':'piqa'},{'model':'m1','task':'arc_challenge'}],'failed':[]})); "
                "status=_prewarm_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert status['expected_attempted_count'] == 4; "
                "assert status['attempted_count'] == 2; "
                "assert 'attempted count is smaller than expanded task count times target count' in status['violations']",
            ],
        ),
        (
            "completion_audit_rejects_duplicate_prewarm_pairs",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _prewarm_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_dup_prewarm_')); "
                "p=root/'edge_prewarm_report.json'; "
                "attempted=[{'model':'m1','task':'piqa'},{'model':'m1','task':'piqa'},{'model':'m2','task':'piqa'},{'model':'m2','task':'piqa'}]; "
                "p.write_text(json.dumps({'ok':True,'tasks':['piqa','arc_challenge'],'target_count':2,'targets':[{'name':'m1'},{'name':'m2'}],'attempted':attempted,'failed':[]})); "
                "status=_prewarm_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert status['expected_attempted_count'] == 4; "
                "assert status['attempted_count'] == 4; "
                "assert {'model':'m1','task':'arc_challenge'} in status['missing_attempted_pairs']; "
                "assert 'attempted is missing model/task pairs' in status['violations']",
            ],
        ),
        (
            "completion_audit_rejects_missing_succeeded_pairs",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _prewarm_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_succeeded_prewarm_')); "
                "p=root/'edge_prewarm_report.json'; "
                "attempted=[{'model':'m1','task':'piqa'},{'model':'m1','task':'arc_challenge'}]; "
                "succeeded=[{'model':'m1','task':'piqa'}]; "
                "p.write_text(json.dumps({'ok':True,'tasks':['piqa','arc_challenge'],'target_count':1,'targets':[{'name':'m1'}],'attempted':attempted,'succeeded':succeeded,'failed':[]})); "
                "status=_prewarm_report_status(p, True); "
                "assert status['ok'] is False; "
                "assert status['succeeded_count'] == 1; "
                "assert {'model':'m1','task':'arc_challenge'} in status['missing_succeeded_pairs']; "
                "assert 'succeeded is missing model/task pairs' in status['violations']",
            ],
        ),
        (
            "completion_audit_rejects_prewarm_dtype_mismatch",
            [
                sys.executable,
                "-c",
                "import json,tempfile; from pathlib import Path; "
                "from tools.audit_edge_completion import _prewarm_report_status; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_dtype_prewarm_')); "
                "p=root/'edge_prewarm_report.json'; "
                "p.write_text(json.dumps({'ok':True,'tasks':['piqa'],'target_count':1,'targets':[{'name':'m1'}],'attempted':[{'model':'m1','task':'piqa','dtype':'fp32'}],'succeeded':[{'model':'m1','task':'piqa'}],'failed':[],'dtype':'fp32'})); "
                "status=_prewarm_report_status(p, True, ['piqa'], ['m1'], 'fp16', ''); "
                "assert status['ok'] is False; "
                "assert status['dtype_mismatch'] == {'expected':'fp16','actual':'fp32'}; "
                "assert 'prewarm dtype does not match matrix config' in status['violations']",
            ],
        ),
        (
            "completion_audit_allows_skipped_external_status",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_allow_skipped_external'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_run_report.json'; r=json.load(open(p)); "
                "\nfor job in r['jobs']:\n"
                "    job['summary']=job.get('summary','').replace(str(src),str(dst)); job['log']=job.get('log','').replace(str(src),str(dst))\n"
                "    if job.get('model')=='TinyModel' and job.get('method')=='dense':\n"
                "        job['status']='skipped_external_not_configured'; job['reason']='external_command_not_configured'\n"
                "r['completed']=[item.replace(str(src),str(dst)) for item in r['completed'] if 'TinyModel/dense/summary.json' not in item]; "
                "r['skipped_external_not_configured']=[{'model':'TinyModel','method':'dense','reason':'external_command_not_configured'}]; "
                "p.write_text(json.dumps(r)); "
                "pf=dst/'matrix_finalize_report.json'; rf=json.load(open(pf)); "
                "\nfor key in ['output_root','results_json','results_csv','edge_report','completion_audit','edge_asset_audit','edge_preflight_report','edge_prewarm_report','edge_handoff_manifest','edge_target_paths_report','edge_prepare_assets_report','edge_prepare_metadata_report','edge_prepare_precompressed_report','resolved_config']:\n"
                "    if isinstance(rf.get(key), str): rf[key]=rf[key].replace(str(src),str(dst))\n"
                "pf.write_text(json.dumps(rf)); "
                "rc=dst/'matrix_config_resolved.json'; cfg=json.load(open(rc)); cfg['defaults']['output_root']=str(dst); rc.write_text(json.dumps(cfg)); "
                "subprocess.run([sys.executable,'tools/render_edge_report.py','--results',str(dst/'matrix_results.json'),'--run-report',str(p),'--asset-audit-report',str(dst/'edge_asset_audit.json'),'--preflight-report',str(dst/'edge_preflight_report.json'),'--prewarm-report',str(dst/'edge_prewarm_report.json'),'--prepare-report',str(dst/'edge_prepare_assets_report.json'),'--prepare-metadata-report',str(dst/'edge_prepare_metadata_report.json'),'--prepare-precompressed-report',str(dst/'edge_prepare_precompressed_report.json'),'--output',str(dst/'edge_report.md')],check=True,capture_output=True,text=True); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--allow-skipped-external','--output',str(dst/'completion_audit_allow_skipped_external.json')],capture_output=True,text=True); "
                "assert proc.returncode == 0; out=json.load(open(dst/'completion_audit_allow_skipped_external.json')); "
                "assert out['matrix_run_report_status']['ok'] is True; "
                "assert 'skipped_external_not_configured' in out['matrix_run_report_status']['allowed_statuses']",
            ],
        ),
        (
            "completion_audit_reports_parse_errors",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_json'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "(dst/'matrix_run_report.json').write_text('{bad-json\\n'); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_json.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.load(open(dst/'completion_audit_bad_json.json')); "
                "assert r['artifact_parse_errors'][0]['artifact'] == 'matrix_run_report'",
            ],
        ),
        (
            "completion_audit_rejects_stale_csv",
            [
                sys.executable,
                "-c",
                "import csv,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_csv'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "rows=list(csv.reader(open(dst/'matrix_results.csv'))); "
                "f=open(dst/'matrix_results.csv','w',newline=''); csv.writer(f).writerows(rows[:1]); f.close(); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_csv.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.load(open(dst/'completion_audit_bad_csv.json')); "
                "assert r['matrix_results_csv_status']['row_count_matches'] is False",
            ],
        ),
        (
            "completion_audit_rejects_stale_csv_pairs",
            [
                sys.executable,
                "-c",
                "import csv,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_csv_pairs'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "rows=list(csv.DictReader(open(dst/'matrix_results.csv'))); rows[0]['compression_method']='stale'; "
                "f=open(dst/'matrix_results.csv','w',newline=''); w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows); f.close(); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_csv_pairs.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.load(open(dst/'completion_audit_bad_csv_pairs.json')); "
                "assert r['matrix_results_csv_status']['row_count_matches'] is True; "
                "assert r['matrix_results_csv_status']['pair_set_matches'] is False",
            ],
        ),
        (
            "completion_audit_rejects_stale_csv_values",
            [
                sys.executable,
                "-c",
                "import csv,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_csv_values'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "rows=list(csv.DictReader(open(dst/'matrix_results.csv'))); rows[0]['tokens_per_second']='999.0'; "
                "f=open(dst/'matrix_results.csv','w',newline=''); w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows); f.close(); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_csv_values.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.load(open(dst/'completion_audit_bad_csv_values.json')); "
                "mismatches=r['matrix_results_csv_status']['value_mismatches']; "
                "assert mismatches and mismatches[0]['column'] == 'tokens_per_second'; "
                "assert mismatches[0]['csv_value'] == '999.0'",
            ],
        ),
        (
            "completion_audit_rejects_missing_csv_columns",
            [
                sys.executable,
                "-c",
                "import csv,json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_missing_csv_column'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "rows=list(csv.DictReader(open(dst/'matrix_results.csv'))); "
                "fieldnames=[name for name in rows[0].keys() if name != 'runtime_python']; "
                "f=open(dst/'matrix_results.csv','w',newline=''); w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); "
                "w.writerows([{k:v for k,v in row.items() if k in fieldnames} for row in rows]); f.close(); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_missing_csv_column.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.load(open(dst/'completion_audit_missing_csv_column.json')); "
                "assert 'runtime_python' in r['matrix_results_csv_status']['missing_columns']",
            ],
        ),
        (
            "completion_audit_rejects_stale_report_metadata",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_report_metadata'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'edge_report.md'; text=p.read_text(); p.write_text(text.replace('- Rows: 2','- Rows: 999')); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_report_metadata.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_bad_report_metadata.json')); "
                "mismatches=out['edge_report_sections']['metadata_mismatches']; "
                "assert any(item['field'] == 'rows' for item in mismatches)",
            ],
        ),
        (
            "completion_audit_rejects_stale_report_content",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_report_content'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'edge_report.md'; text=p.read_text(); p.write_text(text.replace('0.6000 | 0.1500', '0.6000 | 9.9999', 1)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_report_content.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_bad_report_content.json')); "
                "assert out['edge_report_render_status']['ok'] is False; "
                "assert out['edge_report_render_status']['matches_rendered_report'] is False",
            ],
        ),
        (
            "completion_audit_rejects_stale_prewarm_report_content",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_prewarm_report_content'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'edge_report.md'; text=p.read_text(); p.write_text(text.replace('- Succeeded pairs: 61', '- Succeeded pairs: 999', 1)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_prewarm_report_content.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_bad_prewarm_report_content.json')); "
                "assert out['edge_report_render_status']['ok'] is False; "
                "assert out['edge_report_render_status']['matches_rendered_report'] is False",
            ],
        ),
        (
            "completion_audit_rejects_missing_run_job",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_missing_run_job'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_run_report.json'; r=json.load(open(p)); "
                "r['jobs']=[job for job in r['jobs'] if not (job.get('model')=='TinyModel' and job.get('method')=='dense')]; "
                "p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_missing_run_job.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_missing_run_job.json')); "
                "assert out['matrix_run_report_status']['missing_jobs'][0]['method'] == 'dense'",
            ],
        ),
        (
            "completion_audit_rejects_dry_run_matrix_job",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_dry_run_job'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_run_report.json'; r=json.load(open(p)); "
                "r['jobs'][0]['status']='dry_run'; r['jobs'][0].pop('elapsed_seconds', None); "
                "p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_dry_run_job.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_dry_run_job.json')); "
                "bad=out['matrix_run_report_status']['bad_statuses']; "
                "assert bad and bad[0]['status'] == 'dry_run'",
            ],
        ),
        (
            "completion_audit_rejects_top_level_dry_run_matrix_report",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_top_dry_run'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_run_report.json'; r=json.load(open(p)); "
                "r['dry_run']=True; p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_top_dry_run.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_top_dry_run.json')); "
                "violations=out['matrix_run_report_status']['report_violations']; "
                "assert violations and violations[0]['error'] == 'dry_run_report_not_final'",
            ],
        ),
        (
            "completion_audit_rejects_unexpected_run_job",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_unexpected_run_job'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--output',str(dst/'completion_audit_unexpected_run_job.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_unexpected_run_job.json')); "
                "unexpected=out['matrix_run_report_status']['unexpected_jobs']; "
                "assert unexpected and unexpected[0]['method'] == 'wanda'",
            ],
        ),
        (
            "completion_audit_rejects_inconsistent_run_report_lists",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_run_report_lists'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_run_report.json'; r=json.load(open(p)); "
                "r['completed']=[item for item in r['completed'] if 'TinyModel/dense/summary.json' not in item]; p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_run_report_lists.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_bad_run_report_lists.json')); "
                "violations=out['matrix_run_report_status']['list_violations']; "
                "assert any(item['field'] == 'completed' for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_stale_job_overrides",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_job_overrides'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_run_report.json'; r=json.load(open(p)); "
                "r['jobs'][0]['job_overrides']['dtype']='fp32'; p.write_text(json.dumps(r)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_job_overrides.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_bad_job_overrides.json')); "
                "violations=out['matrix_run_report_status']['override_violations']; "
                "assert violations and violations[0]['actual']['dtype'] == 'fp32'",
            ],
        ),
        (
            "completion_audit_rejects_stale_run_report_list_extra",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_stale_run_report_list_extra'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_run_report.json'; r=json.load(open(p)); "
                "r['completed'].append(str(dst/'TinyModel'/'stale'/'summary.json')); p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--output',str(dst/'completion_audit_stale_run_report_list_extra.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_stale_run_report_list_extra.json')); "
                "violations=out['matrix_run_report_status']['list_violations']; "
                "assert any(item['field'] == 'completed' and item.get('extra_items') for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_duplicate_run_jobs",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_duplicate_run_jobs'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_run_report.json'; r=json.load(open(p)); "
                "r['jobs'].append(dict(r['jobs'][0])); p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_duplicate_run_jobs.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_duplicate_run_jobs.json')); "
                "assert out['matrix_run_report_status']['duplicate_jobs'][0]['method'] == 'dense'",
            ],
        ),
        (
            "completion_audit_rejects_missing_run_job_log",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_missing_run_log'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p0=dst/'matrix_run_report.json'; r0=json.load(open(p0)); "
                "[job.update({'summary': job['summary'].replace(str(src),str(dst)), 'log': job['log'].replace(str(src),str(dst))}) for job in r0['jobs']]; "
                "p0.write_text(json.dumps(r0)); "
                "(dst/'TinyModel'/'dense'/'job.log').unlink(); "
                "p=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_missing_run_log.json')],capture_output=True,text=True); "
                "assert p.returncode != 0; out=json.load(open(dst/'completion_audit_missing_run_log.json')); "
                "assert out['matrix_run_report_status']['missing_artifacts'][0]['field'] == 'log'",
            ],
        ),
        (
            "completion_audit_rejects_summary_identity_mismatch",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_summary_identity'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p0=dst/'matrix_run_report.json'; r0=json.load(open(p0)); "
                "[job.update({'summary': job['summary'].replace(str(src),str(dst)), 'log': job['log'].replace(str(src),str(dst))}) for job in r0['jobs']]; "
                "p0.write_text(json.dumps(r0)); "
                "p=dst/'TinyModel'/'dense'/'summary.json'; summary=json.load(open(p)); "
                "summary['compression_method']='wanda'; p.write_text(json.dumps(summary)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_summary_identity.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_bad_summary_identity.json')); "
                "violations=out['matrix_run_report_status']['summary_violations']; "
                "assert violations and violations[0]['error'] == 'summary identity does not match matrix job'; "
                "assert violations[0]['summary_compression_method'] == 'wanda'",
            ],
        ),
        (
            "completion_audit_rejects_summary_runtime_mismatch",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_summary_runtime'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p0=dst/'matrix_run_report.json'; r0=json.load(open(p0)); "
                "[job.update({'summary': job['summary'].replace(str(src),str(dst)), 'log': job['log'].replace(str(src),str(dst))}) for job in r0['jobs']]; "
                "p0.write_text(json.dumps(r0)); "
                "p=dst/'TinyModel'/'dense'/'summary.json'; summary=json.load(open(p)); "
                "summary['runtime']['python']='3.9.0'; p.write_text(json.dumps(summary)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_summary_runtime.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_bad_summary_runtime.json')); "
                "violations=out['matrix_run_report_status']['result_violations']; "
                "assert any(item['field'] == 'runtime_python' and item['artifact'] == 'summary.json' for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_result_summary_path_mismatch",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_result_summary_path'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p0=dst/'matrix_run_report.json'; r0=json.load(open(p0)); "
                "[job.update({'summary': job['summary'].replace(str(src),str(dst)), 'log': job['log'].replace(str(src),str(dst))}) for job in r0['jobs']]; "
                "p0.write_text(json.dumps(r0)); "
                "p=dst/'matrix_results.json'; rows=json.load(open(p)); "
                "rows[0]['compressed_model_path']='/tmp/stale-compressed-model'; p.write_text(json.dumps(rows)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_result_summary_path.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_bad_result_summary_path.json')); "
                "violations=out['matrix_run_report_status']['result_violations']; "
                "assert violations and violations[0]['field'] == 'compressed_model_path'; "
                "assert violations[0]['matrix_results_value'] == '/tmp/stale-compressed-model'",
            ],
        ),
        (
            "completion_audit_rejects_result_source_artifact_mismatch",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_result_source_artifact'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p0=dst/'matrix_run_report.json'; r0=json.load(open(p0)); "
                "[job.update({'summary': job['summary'].replace(str(src),str(dst)), 'log': job['log'].replace(str(src),str(dst))}) for job in r0['jobs']]; "
                "p0.write_text(json.dumps(r0)); "
                "p=dst/'matrix_results.json'; rows=json.load(open(p)); "
                "rows[0]['tokens_per_second']=999.0; p.write_text(json.dumps(rows)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_result_source_artifact.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_bad_result_source_artifact.json')); "
                "violations=out['matrix_run_report_status']['result_violations']; "
                "assert any(item['field'] == 'tokens_per_second' and item['artifact'] == 'deployment.json' and item['artifact_field_path'] == 'tokens_per_second' for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_accuracy_source_artifact_mismatch",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_accuracy_source_artifact'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p0=dst/'matrix_run_report.json'; r0=json.load(open(p0)); "
                "[job.update({'summary': job['summary'].replace(str(src),str(dst)), 'log': job['log'].replace(str(src),str(dst))}) for job in r0['jobs']]; "
                "p0.write_text(json.dumps(r0)); "
                "p=dst/'matrix_results.json'; rows=json.load(open(p)); "
                "rows[0]['mmlu_score']=0.99; rows[0]['accuracy_data_dir']='/tmp/wrong-datasets'; p.write_text(json.dumps(rows)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_accuracy_source_artifact.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_bad_accuracy_source_artifact.json')); "
                "violations=out['matrix_run_report_status']['result_violations']; "
                "assert any(item['field'] == 'mmlu_score' and item['artifact'] == 'accuracy.json' for item in violations); "
                "assert any(item['field'] == 'accuracy_data_dir' and item['artifact'] == 'accuracy.json' for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_missing_mmlu_subtask_column",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_missing_mmlu_subtask_column'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.load(open(p)); "
                "rows[0].pop('mmlu_abstract_algebra_score', None); p.write_text(json.dumps(rows)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_missing_mmlu_subtask_column.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_missing_mmlu_subtask_column.json')); "
                "status=out['mmlu_subtask_coverage_status']; "
                "assert status['ok'] is False; "
                "assert status['violations'][0]['missing_count'] == 1; "
                "assert status['violations'][0]['missing_subtasks'] == ['mmlu_abstract_algebra']",
            ],
        ),
        (
            "completion_audit_rejects_compression_source_artifact_mismatch",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_compression_source_artifact'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p0=dst/'matrix_run_report.json'; r0=json.load(open(p0)); "
                "[job.update({'summary': job['summary'].replace(str(src),str(dst)), 'log': job['log'].replace(str(src),str(dst))}) for job in r0['jobs']]; "
                "p0.write_text(json.dumps(r0)); "
                "p=dst/'matrix_results.json'; rows=json.load(open(p)); "
                "rows[0]['compression_backend']='stale_backend'; rows[0]['compression_seqlen']=64; p.write_text(json.dumps(rows)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_compression_source_artifact.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_bad_compression_source_artifact.json')); "
                "violations=out['matrix_run_report_status']['result_violations']; "
                "assert any(item['field'] == 'compression_backend' and item['artifact'] == 'compression.json' and item['artifact_field_path'] == 'manifest.backend' for item in violations); "
                "assert any(item['field'] == 'compression_seqlen' and item['artifact'] == 'compression.json' and item['artifact_field_path'] == 'manifest.seqlen' for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_resource_source_artifact_mismatch",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_resource_source_artifact'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p0=dst/'matrix_run_report.json'; r0=json.load(open(p0)); "
                "[job.update({'summary': job['summary'].replace(str(src),str(dst)), 'log': job['log'].replace(str(src),str(dst))}) for job in r0['jobs']]; "
                "p0.write_text(json.dumps(r0)); "
                "p=dst/'matrix_results.json'; rows=json.load(open(p)); "
                "rows[0]['resource_prompt_path']='/tmp/wrong-prompts.txt'; p.write_text(json.dumps(rows)); "
                "proc=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_resource_source_artifact.json')],capture_output=True,text=True); "
                "assert proc.returncode != 0; out=json.load(open(dst/'completion_audit_bad_resource_source_artifact.json')); "
                "violations=out['matrix_run_report_status']['result_violations']; "
                "assert any(item['field'] == 'resource_prompt_path' and item['artifact'] == 'resource.json' for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_incomplete_finalize_report",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_finalize'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_finalize_report.json'; r=json.load(open(p)); "
                "r['steps']=[step for step in r['steps'] if 'tools/render_edge_report.py' not in step.get('cmd',[])]; "
                "p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_finalize.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_bad_finalize.json')); "
                "assert 'tools/render_edge_report.py' in out['matrix_finalize_report_status']['missing_tools']",
            ],
        ),
        (
            "completion_audit_rejects_dry_run_finalize_report",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_dry_finalize'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_finalize_report.json'; r=json.load(open(p)); "
                "r['steps'][0].pop('ok', None); r['steps'][0].pop('returncode', None); r['steps'][0]['status']='dry_run'; "
                "p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_dry_finalize.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_dry_finalize.json')); "
                "failed=out['matrix_finalize_report_status']['failed_steps']; "
                "assert any(item.get('error') == 'dry_run_step_not_final' for item in failed)",
            ],
        ),
        (
            "completion_audit_rejects_stale_finalize_paths",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_finalize_paths'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_finalize_report.json'; r=json.load(open(p)); "
                "r['results_json']=str(src/'matrix_results.json'); p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_finalize_paths.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_bad_finalize_paths.json')); "
                "violations=out['matrix_finalize_report_status']['path_violations']; "
                "assert any(item['field'] == 'results_json' for item in violations)",
            ],
        ),
        (
            "completion_audit_requires_target_readiness_finalize_step",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_strict_target_readiness_finalize'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "\nfor fp in dst.rglob('*'):\n"
                "    if fp.is_file():\n"
                "        text=fp.read_text(encoding='utf-8', errors='ignore')\n"
                "        fp.write_text(text.replace(str(src), str(dst)), encoding='utf-8')\n"
                "payload={'ok':True,'config':"
                f"{validator_fixture['config']!r}"
                ",'root':str(dst),'ready_for_matrix':True,'completion_ok':True,'before_matrix_blockers':[],"
                "'artifacts':{'completion_audit':{'path':str(dst/'completion_audit.json'),'exists':True,'ok':True}}}; "
                "(dst/'target_readiness.json').write_text(json.dumps(payload)); "
                "p=dst/'matrix_finalize_report.json'; r=json.load(open(p)); "
                "r['target_readiness']=str(dst/'target_readiness.json'); p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-finalize-report','--require-target-readiness-report','--output',str(dst/'completion_audit_missing_target_readiness_finalize_step.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_missing_target_readiness_finalize_step.json')); "
                "assert 'tools/audit_target_readiness.py' in out['matrix_finalize_report_status']['missing_tools']; "
                "r['steps'].append({'cmd':[sys.executable,'tools/audit_target_readiness.py'],'returncode':0,'ok':True}); p.write_text(json.dumps(r)); "
                "p3=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-finalize-report','--require-target-readiness-report','--output',str(dst/'completion_audit_target_readiness_finalize_step_ok.json')],capture_output=True,text=True); "
                "assert p3.returncode == 0, p3.stderr + p3.stdout; ok=json.load(open(dst/'completion_audit_target_readiness_finalize_step_ok.json')); "
                "assert ok['matrix_finalize_report_status']['ok'] is True; "
                "assert ok['target_readiness_status']['ok'] is True",
            ],
        ),
        (
            "completion_audit_rejects_resolved_selection_drift",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_resolved_selection'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_config_resolved.json'; r=json.load(open(p)); "
                "r['methods']=[]; r['selected_methods']=[]; p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_resolved_selection.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_bad_resolved_selection.json')); "
                "violations=out['resolved_selection_status']['violations']; "
                "assert any(item['field'] == 'methods' for item in violations); "
                "assert any(item['field'] == 'selected_methods' for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_resolved_defaults_drift",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_resolved_defaults'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_config_resolved.json'; r=json.load(open(p)); "
                "r['defaults']['tasks']='mmlu'; p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_resolved_defaults.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_bad_resolved_defaults.json')); "
                "violations=out['resolved_selection_status']['violations']; "
                "assert any(item['field'] == 'defaults.tasks' for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_resolved_output_root_drift",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_resolved_output_root'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_config_resolved.json'; r=json.load(open(p)); "
                "r['defaults']['output_root']='/tmp/stale-output-root'; p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_resolved_output_root.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_bad_resolved_output_root.json')); "
                "violations=out['resolved_selection_status']['violations']; "
                "assert any(item['field'] == 'defaults.output_root' for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_resolved_source_config_drift",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_resolved_source_config'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_config_resolved.json'; r=json.load(open(p)); "
                "r['source_config']='configs/edge_4090_matrix.json'; p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_resolved_source_config.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_bad_resolved_source_config.json')); "
                "violations=out['resolved_selection_status']['violations']; "
                "assert any(item['field'] == 'source_config' for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_resolved_model_path_drift",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_resolved_model_path'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_config_resolved.json'; r=json.load(open(p)); "
                "r['models'][0]['path']='/tmp/wrong-model-path'; p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_resolved_model_path.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_bad_resolved_model_path.json')); "
                "violations=out['resolved_selection_status']['violations']; "
                "assert any(item['field'] == 'model_entries' for item in violations)",
            ],
        ),
        (
            "completion_audit_rejects_resolved_extra_config_drift",
            [
                sys.executable,
                "-c",
                "import json,shutil,subprocess,sys; from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_collect_smoke_bad_resolved_extra_config'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_config_resolved.json'; r=json.load(open(p)); "
                "r['external_commands']={'smoothquant':'stale command'}; p.write_text(json.dumps(r)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--output',str(dst/'completion_audit_bad_resolved_extra_config.json')],capture_output=True,text=True); "
                "assert p2.returncode != 0; out=json.load(open(dst/'completion_audit_bad_resolved_extra_config.json')); "
                "violations=out['resolved_selection_status']['violations']; "
                "assert any(item['field'] == 'external_commands' for item in violations)",
            ],
        ),
        (
            "completion_audit_runtime_gate",
            [
                sys.executable,
                "tools/audit_edge_completion.py",
                "--config",
                validator_fixture["config"],
                "--root",
                collect_fixture["root"],
                "--only-model",
                "TinyModel",
                "--only-method",
                "dense",
                "--allow-unexpected",
                "--require-runtime-python-prefix",
                "3.10",
                "--require-runtime-cuda-prefix",
                "12.",
                "--require-runtime-gpu-total-memory-min-gb",
                "23",
                "--require-runtime-cudnn",
                "--require-runtime-torch-prefix",
                "2.",
                "--require-runtime-transformers",
                "--require-runtime-bitsandbytes",
                "--require-runtime-cpu-count-min",
                "8",
                "--require-runtime-system-memory-min-gb",
                "240",
                "--require-runtime-hf-endpoint",
                "https://hf-mirror.com",
                "--output",
                "/tmp/llmcbench_collect_smoke/completion_audit_runtime_gate.json",
            ],
        ),
        (
            "completion_audit_rejects_cuda_runtime",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_cuda_runtime'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_cuda_version']='11.8'; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-cuda-prefix','12.','--output',str(dst/'completion_audit_bad_cuda.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_cuda.json')); "
                "assert out['runtime_requirements']['cuda_prefix'] == '12.'; "
                "assert any(item['field']=='runtime_cuda_version' and item['required_prefix']=='12.' for item in out['runtime_violations'])",
            ],
        ),
        (
            "completion_audit_rejects_wrong_hf_endpoint",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_hf_endpoint'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_hf_endpoint']='https://huggingface.co'; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-hf-endpoint','https://hf-mirror.com','--output',str(dst/'completion_audit_bad_hf_endpoint.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_hf_endpoint.json')); "
                "assert out['runtime_requirements']['hf_endpoint'] == 'https://hf-mirror.com'; "
                "assert any(item['field']=='runtime_hf_endpoint' and item['required']=='https://hf-mirror.com' for item in out['runtime_violations'])",
            ],
        ),
        (
            "completion_audit_rejects_cuda_unavailable_when_required",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_cuda_available'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_cuda_available']=False; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-cuda-prefix','12.','--output',str(dst/'completion_audit_bad_cuda_available.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_cuda_available.json')); "
                "assert out['runtime_requirements']['cuda_prefix'] == '12.'; "
                "assert any(item['field']=='runtime_cuda_available' and item['required'] is True for item in out['runtime_violations'])",
            ],
        ),
        (
            "completion_audit_rejects_small_gpu_memory",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_gpu_memory'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_gpu_total_memory_gb']=16.0; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-gpu-total-memory-min-gb','23','--output',str(dst/'completion_audit_bad_gpu_memory.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_gpu_memory.json')); "
                "assert out['runtime_requirements']['gpu_total_memory_min_gb'] == 23.0; "
                "assert any(item['field']=='runtime_gpu_total_memory_gb' and item['required_min_gb']==23.0 for item in out['runtime_violations'])",
            ],
        ),
        (
            "completion_audit_rejects_wrong_gpu_count",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_gpu_count'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_gpu_count']=2; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-gpu-count','1','--output',str(dst/'completion_audit_bad_gpu_count.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_gpu_count.json')); "
                "assert out['runtime_requirements']['gpu_count'] == 1; "
                "assert any(item['field']=='runtime_gpu_count' and item['required']==1 for item in out['runtime_violations'])",
            ],
        ),
        (
            "completion_audit_rejects_wrong_cuda_visible_devices",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_cuda_visible'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_cuda_visible_devices']='1'; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-gpu-count','1','--output',str(dst/'completion_audit_bad_cuda_visible.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_cuda_visible.json')); "
                "assert out['runtime_requirements']['cuda_visible_devices'] == '0'; "
                "assert any(item['field']=='runtime_cuda_visible_devices' and item['required']=='0' for item in out['runtime_violations'])",
            ],
        ),
        (
            "completion_audit_rejects_low_cpu_count",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_cpu_count'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_cpu_count']=4; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-cpu-count-min','8','--output',str(dst/'completion_audit_bad_cpu_count.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_cpu_count.json')); "
                "assert out['runtime_requirements']['cpu_count_min'] == 8; "
                "assert any(item['field']=='runtime_cpu_count' and item['required_min']==8 for item in out['runtime_violations'])",
            ],
        ),
        (
            "completion_audit_rejects_low_system_memory",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_system_memory'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_system_memory_total_gb']=128.0; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-system-memory-min-gb','240','--output',str(dst/'completion_audit_bad_system_memory.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_system_memory.json')); "
                "assert out['runtime_requirements']['system_memory_min_gb'] == 240.0; "
                "assert any(item['field']=='runtime_system_memory_total_gb' and item['required_min_gb']==240.0 for item in out['runtime_violations'])",
            ],
        ),
        (
            "completion_audit_rejects_missing_cudnn",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_cudnn'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_cudnn_version']=''; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-cudnn','--output',str(dst/'completion_audit_bad_cudnn.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_cudnn.json')); "
                "assert out['runtime_requirements']['cudnn_required'] is True; "
                "assert any(item['field']=='runtime_cudnn_version' for item in out['runtime_violations'])",
            ],
        ),
        (
            "completion_audit_rejects_torch_runtime",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_torch_runtime'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_torch']='1.13.1'; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-torch-prefix','2.','--output',str(dst/'completion_audit_bad_torch.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_torch.json')); "
                "assert out['runtime_requirements']['torch_prefix'] == '2.'; "
                "assert any(item['field']=='runtime_torch' and item['required_prefix']=='2.' for item in out['runtime_violations'])",
            ],
        ),
        (
            "completion_audit_rejects_missing_transformers",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_transformers_runtime'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_transformers']=''; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-transformers','--output',str(dst/'completion_audit_bad_transformers.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_transformers.json')); "
                "assert out['runtime_requirements']['transformers_required'] is True; "
                "assert any(item['field']=='runtime_transformers' for item in out['runtime_violations'])",
            ],
        ),
        (
            "completion_audit_rejects_missing_bitsandbytes",
            [
                sys.executable,
                "-c",
                "import json, shutil, subprocess, sys; "
                "from pathlib import Path; "
                "src=Path('/tmp/llmcbench_collect_smoke'); dst=Path('/tmp/llmcbench_bad_bitsandbytes_runtime'); "
                "shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst); "
                "p=dst/'matrix_results.json'; rows=json.loads(p.read_text()); "
                "rows[0]['runtime_bitsandbytes_available']=False; rows[0]['runtime_bitsandbytes_error']='missing'; p.write_text(json.dumps(rows)); "
                "p2=subprocess.run([sys.executable,'tools/audit_edge_completion.py','--config',"
                f"{validator_fixture['config']!r},'--root',str(dst),'--only-model','TinyModel','--only-method','dense',"
                "'--allow-unexpected','--require-runtime-bitsandbytes','--output',str(dst/'completion_audit_bad_bitsandbytes.json')],"
                "capture_output=True,text=True); "
                "assert p2.returncode != 0; "
                "out=json.load(open(dst/'completion_audit_bad_bitsandbytes.json')); "
                "assert out['runtime_requirements']['bitsandbytes_required'] is True; "
                "assert any(item['field']=='runtime_bitsandbytes_available' for item in out['runtime_violations'])",
            ],
        ),
        (
            "collect_empty",
            [
                sys.executable,
                "tools/collect_edge_results.py",
                "--root",
                "/tmp/nonexistent_llmcbench_results_smoke",
            ],
        ),
        (
            "tiny_pipeline",
            [
                sys.executable,
                "tools/run_tiny_pipeline_smoke.py",
                "--clean",
            ],
        ),
        (
            "tiny_precompressed_pipeline",
            [
                sys.executable,
                "tools/run_tiny_pipeline_smoke.py",
                "--workdir",
                "/tmp/llmcbench_tiny_precompressed_smoke",
                "--use-precompressed",
                "--clean",
            ],
        ),
        (
            "tiny_gptq_precompressed_pipeline",
            [
                sys.executable,
                "tools/run_tiny_pipeline_smoke.py",
                "--workdir",
                "/tmp/llmcbench_tiny_gptq_precompressed_smoke",
                "--compression-method",
                "gptq",
                "--use-precompressed",
                "--clean",
            ],
        ),
        (
            "tiny_wanda_pipeline",
            [
                sys.executable,
                "tools/run_tiny_pipeline_smoke.py",
                "--workdir",
                "/tmp/llmcbench_tiny_wanda_smoke",
                "--compression-method",
                "wanda",
                "--clean",
            ],
        ),
        (
            "tiny_sparsegpt_pipeline",
            [
                sys.executable,
                "tools/run_tiny_pipeline_smoke.py",
                "--workdir",
                "/tmp/llmcbench_tiny_sparsegpt_smoke",
                "--compression-method",
                "sparsegpt",
                "--clean",
            ],
        ),
        (
            "tiny_llama_wanda_official_2to4_pipeline",
            [
                sys.executable,
                "tools/run_tiny_pipeline_smoke.py",
                "--workdir",
                "/tmp/llmcbench_tiny_llama_wanda_official_2to4",
                "--architecture",
                "llama",
                "--compression-method",
                "wanda",
                "--prune-pattern",
                "2:4",
                "--clean",
            ],
        ),
        (
            "tiny_llama_sparsegpt_official_2to4_pipeline",
            [
                sys.executable,
                "tools/run_tiny_pipeline_smoke.py",
                "--workdir",
                "/tmp/llmcbench_tiny_llama_sparsegpt_official_2to4",
                "--architecture",
                "llama",
                "--compression-method",
                "sparsegpt",
                "--prune-pattern",
                "2:4",
                "--sparsegpt-blocksize",
                "16",
                "--clean",
            ],
        ),
        (
            "tiny_smoothquant_external_pipeline",
            [
                sys.executable,
                "tools/run_tiny_pipeline_smoke.py",
                "--workdir",
                "/tmp/llmcbench_tiny_smoothquant_smoke",
                "--compression-method",
                "smoothquant",
                "--external-command",
                "cp -a {model_path}/. {output_dir}/",
                "--clean",
            ],
        ),
        (
            "tiny_omniquant_external_pipeline",
            [
                sys.executable,
                "tools/run_tiny_pipeline_smoke.py",
                "--workdir",
                "/tmp/llmcbench_tiny_omniquant_smoke",
                "--compression-method",
                "omniquant",
                "--external-command",
                "cp -a {model_path}/. {output_dir}/",
                "--clean",
            ],
        ),
        (
            "validate_results",
            [
                sys.executable,
                "tools/validate_edge_results.py",
                "--config",
                validator_fixture["config"],
                "--results",
                validator_fixture["results"],
                "--only-model",
                "TinyModel",
                "--only-method",
                "dense",
                "--only-method",
                "wanda",
            ],
        ),
        (
            "validate_rejects_mixed_runtime_profiles",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                f"src=Path({validator_fixture['results']!r}); "
                "rows=json.loads(src.read_text()); "
                "rows[1]['runtime_profile']='gptqmodel'; "
                "dst=Path(tempfile.mkdtemp(prefix='llmcbench_mixed_profile_'))/'results.json'; dst.write_text(json.dumps(rows)); "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',str(dst),"
                "'--only-model','TinyModel','--only-method','dense','--only-method','wanda'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'runtime_profile' and 'gptqmodel' in item['all_profiles'] and 'stable' in item['all_profiles'] for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_allows_configured_runtime_profiles",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                f"src=Path({validator_fixture['results']!r}); cfg_src=Path({validator_fixture['config']!r}); "
                "rows=json.loads(src.read_text()); rows[1]['runtime_profile']='smoothquant'; "
                "tmp=Path(tempfile.mkdtemp(prefix='llmcbench_configured_profile_')); "
                "results=tmp/'results.json'; config=tmp/'config.json'; "
                "results.write_text(json.dumps(rows)); "
                "cfg=json.loads(cfg_src.read_text()); cfg['method_overrides']={'wanda': {'runtime_profile': 'smoothquant'}}; "
                "config.write_text(json.dumps(cfg)); "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',str(config),'--results',str(results),"
                "'--only-model','TinyModel','--only-method','dense','--only-method','wanda'],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stdout + p.stderr",
            ],
        ),
        (
            "validate_rejects_failed_accuracy_tasks",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['failed_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['accuracy_violations'][0]['failed_tasks'] == 'hellaswag'",
            ],
        ),
        (
            "validate_rejects_inconsistent_failed_accuracy_tasks",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_failed_tasks_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'accuracy_failed_tasks' and item['actual_list_count'] == 1 for item in r['accuracy_violations'])",
            ],
        ),
        (
            "validate_required_fields_are_unique",
            [
                sys.executable,
                "-c",
                "from tools.validate_edge_results import REQUIRED_BASE_FIELDS; "
                "dupes=sorted({field for field in REQUIRED_BASE_FIELDS if REQUIRED_BASE_FIELDS.count(field) > 1}); "
                "assert not dupes, dupes",
            ],
        ),
        (
            "validate_rejects_limited_accuracy_results",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['limited_accuracy_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'accuracy_limit' and item['value'] == 1 for item in r['accuracy_violations'])",
            ],
        ),
        (
            "validate_rejects_out_of_range_mmlu_subtask_score",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_mmlu_subtask_score_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'mmlu_abstract_algebra_score' and item['constraint'] == '0 <= value <= 1' for item in r['range_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_mmlu_subtask_metric_type",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_mmlu_subtask_metric_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'mmlu_abstract_algebra_score_metric' and item['expected_type'] == 'string' for item in r['type_violations'])",
            ],
        ),
        (
            "validate_rejects_duplicate_pairs",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['duplicate_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['duplicate_pairs'][0]['model'] == 'TinyModel'; "
                "assert r['duplicate_pairs'][0]['method'] == 'dense'; "
                "assert r['duplicate_pairs'][0]['count'] == 2",
            ],
        ),
        (
            "validate_rejects_non_list_results",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['malformed_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['format_errors'][0]['error'] == 'expected JSON list'; "
                "assert r['format_errors'][0]['actual_type'] == 'dict'",
            ],
        ),
        (
            "validate_rejects_invalid_json_results",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['invalid_json_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['format_errors'][0]['error'] == 'invalid JSON'",
            ],
        ),
        (
            "validate_rejects_missing_results_file",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['missing_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['format_errors'][0]['error'] == 'results file not found'",
            ],
        ),
        (
            "validate_rejects_non_object_rows",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['invalid_row_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['format_errors'][0]['error'] == 'all rows must be JSON objects'; "
                "assert r['format_errors'][0]['invalid_rows'][0]['index'] == 2",
            ],
        ),
        (
            "validate_rejects_wrong_metric_types",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_type_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['type_violations'][0]['field'] == 'tokens_per_second'; "
                "assert r['type_violations'][0]['expected_type'] == 'number'; "
                "assert r['type_violations'][0]['actual_type'] == 'str'",
            ],
        ),
        (
            "validate_rejects_non_integer_metrics",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['non_integer_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'deployment_repeat' and item['expected_type'] == 'integer' for item in r['type_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_string_types",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_string_type_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['type_violations'][0]['field'] == 'runtime_python'; "
                "assert r['type_violations'][0]['expected_type'] == 'string'; "
                "assert r['type_violations'][0]['actual_type'] == 'float'",
            ],
        ),
        (
            "validate_rejects_inconsistent_runtime_cuda",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_runtime_cuda_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'runtime_cuda_available' and item['expected'] is True for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_runtime_cuda_visible_devices",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_runtime_cuda_visible_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'runtime_cuda_visible_devices' and item['expected'] == '0' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_score_metric_types",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_score_metric_type_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['type_violations'][0]['field'] == 'piqa_score_metric'; "
                "assert r['type_violations'][0]['expected_type'] == 'string'; "
                "assert r['type_violations'][0]['actual_type'] == 'int'",
            ],
        ),
        (
            "validate_rejects_wrong_target_metric_key",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_target_metric_key_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'piqa_score_metric' and item['expected'] == 'acc_norm,none' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_mmlu_subtask_metric_key",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_mmlu_subtask_metric_key_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'mmlu_abstract_algebra_score_metric' and item['expected'] == 'acc,none' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_missing_compression_backend",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['missing_backend_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_backend' for item in r['missing_fields'])",
            ],
        ),
        (
            "validate_rejects_wrong_compression_backend",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_compression_backend_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_backend' and 'source_checkpoint_reference' in item['expected'] for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_missing_external_command_provenance",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                f"base=json.load(open({validator_fixture['results']!r}))[0]; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_external_cmd_validate_')); "
                "cfg=root/'config.json'; res=root/'results.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'TinyModel','path':'/tmp/tiny'}],"
                "'methods':['smoothquant'],'defaults':{'tasks':'mmlu,arc_challenge,hellaswag,piqa,winogrande','dtype':'fp16','batch_size':'auto',"
                "'split_accuracy_tasks':True,'continue_on_accuracy_error':True,'calibration_batch_size':1,'max_memory':'0:22GiB,cpu:120GiB','seqlen':32,"
                "'input_length':32,'max_new_tokens':1,'warmup':0,'repeat':1,'deployment_memory_limit_gb':1,'hf_cache_dir':'/tmp/hf-cache',"
                "'data_dir':'/tmp/datasets','prompt_file':'/tmp/prompts.txt','calibration_file':'/tmp/calib.txt','calibration_samples':2,'sparsity':0.5,"
                "'quant_bits':4,'group_size':128}})); "
                "row=dict(base, compression_method='smoothquant', compression_backend='external_command', compression_materialized=True, compression_external_returncode=0, "
                "effective_storage_bytes_quantized_approx=61.5, effective_storage_gb_quantized_approx=61.5/(1024**3), "
                "effective_storage_quantized_bits=4, effective_storage_quantized_backend='param_count_times_quant_bits'); "
                "row.pop('compression_external_command', None); res.write_text(json.dumps([row])); "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',str(cfg),'--results',str(res),"
                "'--only-model','TinyModel','--only-method','smoothquant'],capture_output=True,text=True); "
                "assert p.returncode != 0, p.stdout; r=json.loads(p.stdout); "
                "assert any(item['field']=='compression_external_command' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_failed_external_command_provenance",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                f"base=json.load(open({validator_fixture['results']!r}))[0]; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_external_returncode_validate_')); "
                "cfg=root/'config.json'; res=root/'results.json'; "
                "cfg.write_text(json.dumps({'models':[{'name':'TinyModel','path':'/tmp/tiny'}],"
                "'methods':['smoothquant'],'defaults':{'tasks':'mmlu,arc_challenge,hellaswag,piqa,winogrande','dtype':'fp16','batch_size':'auto',"
                "'split_accuracy_tasks':True,'continue_on_accuracy_error':True,'calibration_batch_size':1,'max_memory':'0:22GiB,cpu:120GiB','seqlen':32,"
                "'input_length':32,'max_new_tokens':1,'warmup':0,'repeat':1,'deployment_memory_limit_gb':1,'hf_cache_dir':'/tmp/hf-cache',"
                "'data_dir':'/tmp/datasets','prompt_file':'/tmp/prompts.txt','calibration_file':'/tmp/calib.txt','calibration_samples':2,'sparsity':0.5,"
                "'quant_bits':4,'group_size':128}})); "
                "row=dict(base, compression_method='smoothquant', compression_backend='external_command', compression_materialized=True, "
                "compression_external_command='python export.py', compression_external_returncode=7, effective_storage_bytes_quantized_approx=61.5, "
                "effective_storage_gb_quantized_approx=61.5/(1024**3), effective_storage_quantized_bits=4, "
                "effective_storage_quantized_backend='param_count_times_quant_bits'); "
                "res.write_text(json.dumps([row])); "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',str(cfg),'--results',str(res),"
                "'--only-model','TinyModel','--only-method','smoothquant'],capture_output=True,text=True); "
                "assert p.returncode != 0, p.stdout; r=json.loads(p.stdout); "
                "assert any(item['field']=='compression_external_returncode' and item['expected']==0 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_missing_compression_dtype",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['missing_dtype_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_dtype' for item in r['missing_fields'])",
            ],
        ),
        (
            "validate_rejects_missing_quant_compression_params",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['missing_quant_param_results']!r},"
                "'--only-model','TinyModel','--only-method','gptq'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_quant_bits' for item in r['missing_fields'])",
            ],
        ),
        (
            "validate_rejects_wrong_compression_dtype",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_dtype_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_dtype' and item['expected'] == 'fp16' and item['value'] == 'fp32' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_lm_eval_dtype",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_lm_eval_dtype_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'accuracy_lm_eval_dtype' and item['expected'] == 'float16' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_num_fewshot",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_num_fewshot_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'accuracy_num_fewshot' and item['expected'] is None for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_hf_cache",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_hf_cache_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'runtime_hf_home' and item['expected'] == '/tmp/hf-cache' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_compression_max_memory",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_max_memory_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_max_memory' and item['expected'] == '0:22GiB,cpu:120GiB' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_deployment_device_map",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_device_map_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'deployment_device_map' and item['expected'] == 'auto' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_attn_implementation",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_validate_attn_')); "
                f"row=json.load(open({validator_fixture['results']!r}))[0]; "
                "row.update({'compression_attn_implementation':'eager','accuracy_attn_implementation':None,'deployment_attn_implementation':None}); "
                f"cfg=json.load(open({validator_fixture['config']!r})); "
                "cfg['methods']=['dense']; cfg['defaults']['attn_implementation']='eager'; "
                "cfg_path=root/'config.json'; res_path=root/'results.json'; "
                "cfg_path.write_text(json.dumps(cfg)); res_path.write_text(json.dumps([row])); "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',str(cfg_path),'--results',str(res_path)],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "fields={item['field'] for item in r['consistency_violations']}; "
                "assert {'accuracy_attn_implementation','deployment_attn_implementation'} <= fields",
            ],
        ),
        (
            "validate_rejects_wrong_compression_seqlen",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_compression_seqlen_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_seqlen' and item['expected'] == 32 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_compression_params",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_compression_param_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_quant_bits' and item['expected'] == 4 for item in r['consistency_violations']); "
                "assert any(item['field'] == 'compression_group_size' and item['expected'] == 128 for item in r['consistency_violations']); "
                "assert any(item['field'] == 'compression_desc_act' and item['expected'] is False for item in r['consistency_violations']); "
                "assert any(item['field'] == 'compression_awq_version' and item['expected'] == 'GEMM' for item in r['consistency_violations']); "
                "assert any(item['field'] == 'compression_sparsity' and item['expected'] == 0.5 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_compression_calibration_batch",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_calibration_batch_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_calibration_batch_size' and item['expected'] == 1 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_calibration_file",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_calibration_file_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_calibration_file' and item['expected'] == '/tmp/calib.txt' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_deployment_setting",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_deployment_setting_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'deployment_input_length' and item['expected'] == 32 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_resource_setting",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_resource_setting_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'resource_seqlen' and item['expected'] == 32 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_prompt_paths",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_prompt_path_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'deployment_prompt_path' and item['expected'] == '/tmp/prompts.txt' for item in r['consistency_violations']); "
                "assert any(item['field'] == 'resource_prompt_path' and item['expected'] == '/tmp/prompts.txt' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_prompt_sources",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_prompt_source_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'deployment_prompt_source' and item['expected'] == 'file' for item in r['consistency_violations']); "
                "assert any(item['field'] == 'resource_prompt_source' and item['expected'] == 'file' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_config_only_resource_backend",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_resource_backend_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'resource_backend' and item['value'] == 'config_only_fallback' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_unknown_resource_backend",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['unknown_resource_backend_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'resource_backend' and item['expected'] == 'model_load' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_compute_backend",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_compute_backend_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compute_backend' and item['expected'] == 'calflops or analytical_config_estimate' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_accuracy_setting",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_accuracy_setting_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'accuracy_batch_size' and item['expected'] == 'auto' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_dataset_paths",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_dataset_path_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'accuracy_data_dir' and item['expected'] == '/tmp/datasets' for item in r['consistency_violations']); "
                "assert any(item['field'] == 'accuracy_hf_datasets_cache' and item['expected'] == '/tmp/datasets' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_uses_model_method_overrides",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys,tempfile; from pathlib import Path; "
                "root=Path(tempfile.mkdtemp(prefix='llmcbench_validate_overrides_')); "
                f"row=json.load(open({validator_fixture['results']!r}))[0]; "
                "row.update({'deployment_input_length':16,'deployment_max_new_tokens':2,'compression_seqlen':16,'resource_seqlen':16,'resource_input_length':16,'resource_max_new_tokens':2,'accuracy_batch_size':'2'}); "
                "cfg=json.load(open(" + repr(validator_fixture["config"]) + ")); "
                "cfg['methods']=['dense']; cfg['defaults'].update({'input_length':32,'max_new_tokens':1,'seqlen':32,'batch_size':'auto'}); "
                "cfg['model_method_overrides']={'TinyModel/dense':{'input_length':16,'max_new_tokens':2,'seqlen':16,'batch_size':'2'}}; "
                "cfg_path=root/'config.json'; res_path=root/'results.json'; "
                "cfg_path.write_text(json.dumps(cfg)); res_path.write_text(json.dumps([row])); "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',str(cfg_path),'--results',str(res_path)],capture_output=True,text=True); "
                "assert p.returncode == 0, p.stderr + p.stdout",
            ],
        ),
        (
            "validate_rejects_non_finite_metrics",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['non_finite_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['type_violations'][0]['field'] == 'tokens_per_second'; "
                "assert r['type_violations'][0]['expected_type'] == 'finite number'",
            ],
        ),
        (
            "validate_rejects_missing_deployment_gate_fields",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['missing_deployment_gate_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'deployment_memory_measured_peak_gb' for item in r['missing_fields'])",
            ],
        ),
        (
            "validate_rejects_wrong_deployment_limit",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_deployment_limit_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['memory_violations'][0]['source'] == 'deployment_memory_limit_gb'; "
                "assert r['memory_violations'][0]['row_limit_gb'] == 2",
            ],
        ),
        (
            "validate_rejects_negative_quantized_storage",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_quantized_storage_results']!r},"
                "'--only-model','TinyModel','--only-method','gptq'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'effective_storage_gb_quantized_approx' for item in r['range_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_quantized_storage_bits",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_quantized_storage_results']!r},"
                "'--only-model','TinyModel','--only-method','gptq'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'effective_storage_quantized_bits' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_quantized_storage_bytes",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_quantized_storage_bytes_results']!r},"
                "'--only-model','TinyModel','--only-method','gptq'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'effective_storage_bytes_quantized_approx' and item['expected'] == 61.5 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_quantized_storage_gb",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_quantized_storage_gb_results']!r},"
                "'--only-model','TinyModel','--only-method','gptq'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'effective_storage_gb_quantized_approx' and item['source'] == 'effective_storage_bytes_quantized_approx / 1024**3' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_quantized_storage_backend",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_quantized_storage_backend_results']!r},"
                "'--only-model','TinyModel','--only-method','gptq'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'effective_storage_quantized_backend' and item['expected'] == 'checkpoint_directory_size' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_quantized_storage_on_non_quantized_method",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['non_quantized_storage_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'effective_storage_quantized_backend' and item['source'] == 'non-quantized compression method' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_storage_gb",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_storage_gb_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'storage_gb' and item['source'] == 'storage_bytes / 1024**3' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_nonzero_ratio",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_nonzero_ratio_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'nonzero_ratio' and item['expected'] == 0.75 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_sparse_storage_estimate",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_sparse_storage_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'effective_storage_bytes_sparse_approx' and item['expected'] == 50.0 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_sparse_flops_estimate",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_sparse_flops_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'effective_flops_num_sparse_approx' and item['expected'] == 500000.0 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_precompressed_provenance",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_precompressed_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_uses_precompressed' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_compression_materialized",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_materialized_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'compression_materialized' and item['expected'] is False for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_missing_quantized_storage",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['missing_quantized_storage_results']!r},"
                "'--only-model','TinyModel','--only-method','gptq','--allow-unexpected'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "missing={item['field'] for item in r['missing_fields']}; "
                "assert 'effective_storage_bytes_quantized_approx' in missing; "
                "assert 'effective_storage_gb_quantized_approx' in missing; "
                "assert 'effective_storage_quantized_bits' in missing; "
                "assert 'effective_storage_quantized_backend' in missing",
            ],
        ),
        (
            "validate_rejects_out_of_range_metrics",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['out_of_range_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "fields={item['field']: item['constraint'] for item in r['range_violations']}; "
                "assert fields['tokens_per_second'] == '>= 0'; "
                "assert fields['piqa_score'] == '0 <= value <= 1'",
            ],
        ),
        (
            "validate_rejects_zero_final_measurements",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['zero_measurement_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "fields={item['field']: item['constraint'] for item in r['range_violations']}; "
                "assert fields['tokens_per_second'] == '> 0'; "
                "assert fields['deployment_generated_tokens'] == '> 0'; "
                "assert fields['storage_bytes'] == '> 0'; "
                "assert fields['total_params'] == '> 0'; "
                "assert fields['flops_num'] == '> 0'",
            ],
        ),
        (
            "validate_rejects_inconsistent_deployment_gate",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_deployment_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert r['consistency_violations'][0]['field'] == 'deployment_memory_measured_peak_gb'; "
                "assert r['consistency_violations'][0]['expected_min'] == 0.5",
            ],
        ),
        (
            "validate_rejects_inconsistent_deployment_input_tokens",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_deployment_input_tokens_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'deployment_input_tokens' and item['expected_max'] == 32 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_deployment_generated_tokens",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_deployment_generated_tokens_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'deployment_generated_tokens' and item['expected_max'] == 1 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_inconsistent_deployment_tps",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['inconsistent_deployment_tps_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'tokens_per_second' and item['expected'] == 0.5 for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_accuracy_retention",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_retention_results']!r},"
                "'--only-model','TinyModel','--only-method','dense','--only-method','wanda'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'accuracy_retention_vs_dense' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_mmlu_aggregate",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_mmlu_aggregate_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'mmlu_score' and item['source'] == 'mean(DEFAULT_MMLU_SUBTASKS)' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_mmlu_metric_aggregate",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_mmlu_metric_aggregate_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'mmlu_score_metric' and item['expected'] == 'aggregate:acc,none' for item in r['consistency_violations'])",
            ],
        ),
        (
            "validate_rejects_wrong_accuracy_average",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; "
                "p=subprocess.run([sys.executable,'tools/validate_edge_results.py','--config',"
                f"{validator_fixture['config']!r},'--results',{validator_fixture['wrong_accuracy_average_results']!r},"
                "'--only-model','TinyModel','--only-method','dense'],capture_output=True,text=True); "
                "assert p.returncode != 0; r=json.loads(p.stdout); "
                "assert any(item['field'] == 'accuracy_avg_score' and item['source'] == 'mean(target_task_scores)' for item in r['consistency_violations'])",
            ],
        ),
        (
            "render_report",
            [
                sys.executable,
                "tools/render_edge_report.py",
                "--results",
                validator_fixture["results"],
                "--run-report",
                validator_fixture["run_report"],
                "--output",
                "/tmp/llmcbench_validator_smoke/edge_report.md",
            ],
        ),
        (
            "render_report_sections_assert",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('/tmp/llmcbench_validator_smoke/edge_report.md').read_text(); "
                "assert '## Accuracy Metric Keys' in text; "
                "assert 'acc_norm,none' in text; "
                "assert '## Compression' in text; "
                "assert 'Source Path' in text; "
                "assert 'Revision' in text; "
                "assert 'Max Memory' in text; "
                "assert 'Num Fewshot' in text; "
                "assert '0:22GiB,cpu:120GiB' in text; "
                "assert 'fp16' in text; "
                "assert 'Materialized' in text; "
                "assert 'Precompressed' in text; "
                "assert 'Desc Act' in text; "
                "assert 'AWQ Version' in text; "
                "assert 'External Command' in text; "
                "assert 'External Returncode' in text; "
                "assert '/tmp/tiny-source' in text; "
                "assert 'False' in text; "
                "assert 'source_checkpoint_reference' in text; "
                "assert '## Runtime' in text; "
                "assert 'NVIDIA Test GPU' in text; "
                "assert 'Elapsed Seconds' in text; "
                "assert 'Input Tokens' in text; "
                "assert 'Generated Tokens' in text; "
                "assert 'Storage Bytes' in text; "
                "assert 'Effective Storage Bytes' in text; "
                "assert 'Quantized Storage Bytes' in text; "
                "assert 'FLOPs Num' in text; "
                "assert 'Quantized Storage GB' in text; "
                "assert 'Quantized Storage Backend' in text",
            ],
        ),
        (
            "render_report_rejects_table_width_mismatch",
            [
                sys.executable,
                "-c",
                "from tools.render_edge_report import _table\n"
                "try:\n"
                "    _table(['A','B'], [[1]])\n"
                "except ValueError as exc:\n"
                "    assert 'expected 2' in str(exc)\n"
                "else:\n"
                "    raise AssertionError('table width mismatch was not rejected')\n",
            ],
        ),
        (
            "render_report_with_prewarm",
            [
                sys.executable,
                "tools/render_edge_report.py",
                "--results",
                "/tmp/llmcbench_collect_smoke/matrix_results.json",
                "--run-report",
                "/tmp/llmcbench_collect_smoke/matrix_run_report.json",
                "--asset-audit-report",
                "/tmp/llmcbench_collect_smoke/edge_asset_audit.json",
                "--preflight-report",
                "/tmp/llmcbench_collect_smoke/edge_preflight_report.json",
                "--prewarm-report",
                "/tmp/llmcbench_collect_smoke/edge_prewarm_report.json",
                "--output",
                "/tmp/llmcbench_collect_smoke/edge_report_with_prewarm.md",
            ],
        ),
        (
            "render_report_with_prewarm_assert",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; text=Path('/tmp/llmcbench_collect_smoke/edge_report_with_prewarm.md').read_text(); "
                "assert '## Preflight Checks' in text; "
                "assert 'Asset Audit' in text; "
                "assert '## Asset Paths' in text; "
                "assert 'Non-Rotational' in text; "
                "assert '## Prewarm' in text; "
                "assert '- Dtype: fp16' in text; "
                "assert '- Expanded tasks: 61' in text; "
                "assert '- Attempted pairs: 61' in text; "
                "assert '- Succeeded pairs: 61' in text",
            ],
        ),
        (
            "render_report_with_precompressed_blockers",
            [
                sys.executable,
                "-c",
                "import json,subprocess,sys; from pathlib import Path; "
                "p=Path('/tmp/llmcbench_precompressed_blockers_report.json'); "
                "p.write_text(json.dumps({'ok':False,'blockers':[{'model':'LLaMA3-8B','method':'omniquant','reason':'repo_not_configured','repo_status':'no_verified_public_default_repo','detail':'missing public base checkpoint','remediation':'provide --repo override'}]})); "
                "out=Path('/tmp/llmcbench_collect_smoke/edge_report_with_precompressed_blockers.md'); "
                "proc=subprocess.run([sys.executable,'tools/render_edge_report.py','--results','/tmp/llmcbench_collect_smoke/matrix_results.json','--prepare-precompressed-report',str(p),'--output',str(out)],capture_output=True,text=True); "
                "assert proc.returncode == 0, proc.stderr + proc.stdout; "
                "text=out.read_text(); "
                "assert '### Precompressed Preparation Blockers' in text; "
                "assert 'repo_not_configured' in text; "
                "assert 'provide --repo override' in text",
            ],
        ),
    ]
    results = [_run(name, cmd) for name, cmd in checks]
    report = {
        "ok": all(item["ok"] for item in results),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "output": str(output)}, indent=2))
    if not report["ok"]:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight LLMCBench edge smoke checks.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--output", default="outputs/smoke_report.json")
    return parser


def main() -> None:
    run_checks(build_parser().parse_args())


if __name__ == "__main__":
    main()
