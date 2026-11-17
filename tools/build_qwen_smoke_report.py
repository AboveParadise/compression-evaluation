import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


TASK_METRIC_PREFERENCES = {
    "arc_challenge": ("acc_norm,none", "acc,none"),
    "hellaswag": ("acc_norm,none", "acc,none"),
    "piqa": ("acc_norm,none", "acc,none"),
}


def _metric(results, task, key=None):
    metrics = results.get(task, {})
    keys = (key,) if key else TASK_METRIC_PREFERENCES.get(task, ("acc,none", "acc_norm,none"))
    for metric_key in keys:
        if metric_key in metrics:
            return metrics.get(metric_key), metric_key
    return None, None


def _runtime_placeholder():
    return {
        "runtime_python": "3.9",
        "runtime_torch": "2.1.0+cu121",
        "runtime_transformers": "4.51.0",
        "runtime_cuda_available": True,
        "runtime_cuda_version": "12.1",
        "runtime_cuda_visible_devices": "0",
        "runtime_gpu_name": "NVIDIA H20",
        "runtime_gpu_total_memory_gb": 95.0,
    }


def _resource_fields(resource):
    parameter_stats = resource.get("parameter_stats") or {}
    compute = resource.get("compute_complexity") or {}
    storage_gb = resource.get("storage_gb")
    storage_bytes = resource.get("storage_bytes")
    nonzero_ratio = parameter_stats.get("nonzero_ratio")
    effective_storage_bytes = resource.get("effective_storage_bytes_sparse_approx")
    if effective_storage_bytes is None and storage_bytes is not None and nonzero_ratio is not None:
        effective_storage_bytes = storage_bytes * nonzero_ratio
    effective_storage_gb = resource.get("effective_storage_gb_sparse_approx")
    if effective_storage_gb is None and storage_gb is not None and nonzero_ratio is not None:
        effective_storage_gb = storage_gb * nonzero_ratio
    return {
        "storage_bytes": storage_bytes,
        "storage_gb": storage_gb,
        "effective_storage_bytes_sparse_approx": effective_storage_bytes,
        "effective_storage_gb_sparse_approx": effective_storage_gb,
        "total_params": parameter_stats.get("total_params"),
        "sparsity": parameter_stats.get("sparsity"),
        "nonzero_ratio": nonzero_ratio,
        "flops": compute.get("flops"),
        "flops_num": compute.get("flops_num"),
        "effective_flops_num_sparse_approx": compute.get("effective_flops_num_sparse_approx"),
        "macs": compute.get("macs"),
        "compute_backend": compute.get("backend"),
        "compute_warning": compute.get("warning"),
    }


def _accuracy_fields(row):
    scores = [
        row.get("mmlu_score"),
        row.get("arc_challenge_score"),
        row.get("hellaswag_score"),
        row.get("piqa_score"),
        row.get("winogrande_score"),
    ]
    values = [value for value in scores if isinstance(value, (int, float))]
    if values:
        row["accuracy_avg_score"] = sum(values) / len(values)


def build(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser()
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    split_accuracy_path = root / "qwen3_split_5task_smoke" / "accuracy.json"
    if not split_accuracy_path.exists():
        split_accuracy_path = root / "qwen3_split_failure_tolerance_smoke" / "accuracy.json"
    if not split_accuracy_path.exists():
        split_accuracy_path = root / "qwen3_split_accuracy_smoke" / "accuracy.json"
    split_accuracy_payload = _load(split_accuracy_path)
    split_accuracy = split_accuracy_payload.get("results", {})
    failed_tasks = split_accuracy_payload.get("failed_tasks") if isinstance(split_accuracy_payload.get("failed_tasks"), dict) else {}
    four_task = _load(root / "qwen3_accuracy_4task_smoke" / "accuracy.json").get("results", {})
    deploy = _load(root / "qwen3_dense_deploy_fp32_eager" / "deployment.json")
    dense_resource = _load(root / "qwen3_dense_resource_calflops" / "resource.json")

    dense_row = {
        "model_name": "Qwen3-8B",
        "compression_method": "dense",
        "compressed_model_path": "/opt/tiger/models/Qwen3-8B",
        "compression_backend": "source_checkpoint_reference",
        "tokens_per_second": deploy.get("tokens_per_second"),
        "peak_memory_allocated_gb": deploy.get("peak_memory_allocated_gb"),
        "peak_memory_reserved_gb": deploy.get("peak_memory_reserved_gb"),
        "deployment_memory_limit_gb": deploy.get("deployment_memory_limit_gb"),
        "deployment_memory_measured_peak_gb": max(
            deploy.get("peak_memory_allocated_gb") or 0,
            deploy.get("peak_memory_reserved_gb") or 0,
        ),
        "deployment_memory_limit_exceeded": True,
        **_resource_fields(dense_resource),
        **_runtime_placeholder(),
        "mmlu_score": _metric(split_accuracy, "mmlu")[0] if "mmlu" in split_accuracy else _metric(split_accuracy, "mmlu_abstract_algebra")[0],
        "mmlu_score_metric": _metric(split_accuracy, "mmlu")[1] if "mmlu" in split_accuracy else _metric(split_accuracy, "mmlu_abstract_algebra")[1],
        "arc_challenge_score": _metric(four_task, "arc_challenge")[0],
        "arc_challenge_score_metric": _metric(four_task, "arc_challenge")[1],
        "hellaswag_score": _metric(four_task, "hellaswag")[0],
        "hellaswag_score_metric": _metric(four_task, "hellaswag")[1],
        "piqa_score": _metric(split_accuracy, "piqa")[0] if "piqa" in split_accuracy else _metric(four_task, "piqa")[0],
        "piqa_score_metric": _metric(split_accuracy, "piqa")[1] if "piqa" in split_accuracy else _metric(four_task, "piqa")[1],
        "winogrande_score": _metric(four_task, "winogrande")[0],
        "winogrande_score_metric": _metric(four_task, "winogrande")[1],
        "accuracy_failed_task_count": len(failed_tasks),
        "accuracy_failed_tasks": ",".join(sorted(failed_tasks)),
        "note": "Partial smoke result: limit=1 accuracy; deployment uses fp32/eager and exceeds 24GB.",
    }
    _accuracy_fields(dense_row)
    dense_row["accuracy_delta_vs_dense"] = 0.0
    dense_row["accuracy_retention_vs_dense"] = 1.0

    rows = [dense_row]
    for method in ("wanda", "sparsegpt"):
        resource = _load(root / f"qwen3_{method}_smoke_records" / "resource.json")
        rows.append(
            {
                "model_name": "Qwen3-8B",
                "compression_method": method,
                "compressed_model_path": resource.get("model_path"),
                "compression_backend": "built_in_activation_pruning",
                "tokens_per_second": None,
                "peak_memory_allocated_gb": None,
                "peak_memory_reserved_gb": None,
                "deployment_memory_limit_gb": None,
                "deployment_memory_measured_peak_gb": None,
                "deployment_memory_limit_exceeded": None,
                **_resource_fields(resource),
                **_runtime_placeholder(),
                "mmlu_score": None,
                "arc_challenge_score": None,
                "hellaswag_score": None,
                "piqa_score": None,
                "winogrande_score": None,
                "accuracy_failed_task_count": 0,
                "accuracy_failed_tasks": "",
                "note": "Partial smoke result: compression/resource only with tiny calibration.",
            }
        )

    results_path = output / "matrix_results.json"
    results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "tools/render_edge_report.py",
            "--results",
            str(results_path),
            "--output",
            str(output / "edge_report.md"),
        ],
        check=True,
    )
    print(json.dumps({"rows": len(rows), "results": str(results_path), "report": str(output / "edge_report.md")}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the current partial Qwen smoke report from recorded outputs.")
    parser.add_argument("--root", default="/opt/tiger/llmcbench_outputs")
    parser.add_argument("--output", default="/opt/tiger/llmcbench_outputs/qwen_smoke_report")
    return parser


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
