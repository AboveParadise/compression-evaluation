import argparse
import json
from pathlib import Path
from typing import Dict, List


ACCURACY_FIELDS = [
    "mmlu_score",
    "arc_challenge_score",
    "hellaswag_score",
    "piqa_score",
    "winogrande_score",
]


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(headers: List[str], rows: List[List]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    expected_columns = len(headers)
    for index, row in enumerate(rows):
        if len(row) != expected_columns:
            raise ValueError(
                f"Markdown table row {index} has {len(row)} columns; "
                f"expected {expected_columns}. headers={headers!r}"
            )
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def _sorted_rows(rows: List[Dict]) -> List[Dict]:
    return sorted(rows, key=lambda row: (row.get("model_name", ""), row.get("compression_method", "")))


def _pair_rows(items: List[Dict]) -> List[List]:
    return [[item.get("model"), item.get("task")] for item in items if isinstance(item, dict)]


def _precompressed_blocker_rows(report: Dict) -> List[List]:
    if not isinstance(report, dict):
        return []
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    rows = []
    for item in blockers:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                item.get("model"),
                item.get("method"),
                item.get("reason") or item.get("repo_status"),
                item.get("repo_status"),
                item.get("detail"),
                item.get("remediation"),
            ]
        )
    return rows


def render(args: argparse.Namespace) -> str:
    rows = _sorted_rows(_load(Path(args.results).expanduser()))
    run_report = _load(Path(args.run_report).expanduser()) if args.run_report else None
    asset_audit_report = _load(Path(args.asset_audit_report).expanduser()) if args.asset_audit_report else None
    preflight_report = _load(Path(args.preflight_report).expanduser()) if args.preflight_report else None
    prewarm_report = _load(Path(args.prewarm_report).expanduser()) if args.prewarm_report else None
    prepare_report = _load(Path(args.prepare_report).expanduser()) if args.prepare_report else None
    prepare_metadata_report = _load(Path(args.prepare_metadata_report).expanduser()) if args.prepare_metadata_report else None
    prepare_precompressed_report = _load(Path(args.prepare_precompressed_report).expanduser()) if args.prepare_precompressed_report else None
    accuracy_rows = [
        [row.get("model_name"), row.get("compression_method")]
        + [row.get(field) for field in ACCURACY_FIELDS]
        + [
            row.get("accuracy_avg_score"),
            row.get("accuracy_delta_vs_dense"),
            row.get("accuracy_retention_vs_dense"),
        ]
        for row in rows
    ]
    accuracy_metric_rows = [
        [row.get("model_name"), row.get("compression_method")]
        + [row.get(f"{field}_metric") for field in ACCURACY_FIELDS]
        for row in rows
    ]
    accuracy_setting_rows = [
        [
            row.get("model_name"),
            row.get("compression_method"),
            row.get("accuracy_dtype"),
            row.get("accuracy_lm_eval_dtype"),
            row.get("accuracy_batch_size"),
            row.get("accuracy_num_fewshot"),
            row.get("accuracy_limit"),
            row.get("accuracy_data_dir"),
            row.get("accuracy_hf_datasets_cache"),
            row.get("accuracy_max_memory"),
            row.get("accuracy_split_tasks"),
            row.get("accuracy_continue_on_error"),
        ]
        for row in rows
    ]
    deployment_rows = [
        [
            row.get("model_name"),
            row.get("compression_method"),
            row.get("tokens_per_second"),
            row.get("deployment_elapsed_seconds"),
            row.get("deployment_generated_tokens"),
            row.get("deployment_dtype"),
            row.get("deployment_max_memory"),
            row.get("deployment_device_map"),
            row.get("deployment_prompt_source"),
            row.get("deployment_prompt_count"),
            row.get("deployment_batch_size"),
            row.get("deployment_input_length"),
            row.get("deployment_input_tokens"),
            row.get("deployment_max_new_tokens"),
            row.get("deployment_warmup"),
            row.get("deployment_repeat"),
            row.get("peak_memory_allocated_gb"),
            row.get("peak_memory_reserved_gb"),
            row.get("deployment_memory_limit_gb"),
            row.get("deployment_memory_measured_peak_gb"),
            row.get("deployment_memory_limit_exceeded"),
        ]
        for row in rows
    ]
    compression_rows = [
        [
            row.get("model_name"),
            row.get("compression_method"),
            row.get("source_model_path"),
            row.get("model_revision"),
            row.get("compression_base_model_repository"),
            row.get("compression_base_model_revision"),
            row.get("compression_base_model_attested"),
            row.get("compression_backend"),
            row.get("compression_dtype"),
            row.get("compression_seqlen"),
            row.get("compression_max_memory"),
            row.get("compression_attn_implementation"),
            row.get("compression_materialized"),
            row.get("compression_uses_precompressed"),
            row.get("compression_quant_bits"),
            row.get("compression_group_size"),
            row.get("compression_desc_act"),
            row.get("compression_awq_version"),
            row.get("compression_sparsity"),
            row.get("compression_sparsity_semantics"),
            row.get("compression_prune_pattern"),
            row.get("compression_prune_structure"),
            row.get("compression_prune_n"),
            row.get("compression_prune_m"),
            row.get("compression_prune_lm_head"),
            row.get("compression_pruning_algorithm"),
            row.get("compression_pruning_buffer_device"),
            row.get("compression_sparsegpt_blocksize"),
            row.get("compression_sparsegpt_percdamp"),
            row.get("compression_calibration_samples"),
            row.get("compression_calibration_batch_size"),
            row.get("compression_calibration_protocol"),
            row.get("compression_calibration_source"),
            row.get("compression_calibration_sampling_algorithm"),
            row.get("compression_calibration_seed"),
            row.get("compression_calibration_dataset"),
            row.get("compression_calibration_data_file"),
            row.get("compression_calibration_dataset_revision"),
            row.get("compression_calibration_token_length"),
            row.get("compression_calibration_token_sha256"),
            row.get("compression_paper_protocol"),
            row.get("compression_paper_protocol_attested"),
            row.get("compression_gptq_true_sequential"),
            row.get("compression_gptq_symmetric"),
            row.get("compression_awq_calibration_source"),
            row.get("compression_awq_grid_size"),
            row.get("compression_smoothquant_alpha"),
            row.get("compression_smoothquant_alpha_selection"),
            row.get("compression_official_repository"),
            row.get("compression_official_commit"),
            row.get("compression_official_implementation_verified"),
            row.get("compression_omniquant_abits"),
            row.get("compression_omniquant_epochs"),
            row.get("compression_omniquant_seed"),
            row.get("compression_omniquant_lwc"),
            row.get("compression_omniquant_let"),
            row.get("compression_note"),
            row.get("compression_external_command"),
            row.get("compression_external_returncode"),
            row.get("compressed_model_path"),
        ]
        for row in rows
    ]
    runtime_rows = [
        [
            row.get("model_name"),
            row.get("compression_method"),
            row.get("runtime_python"),
            row.get("runtime_profile"),
            row.get("runtime_cpu_count"),
            row.get("runtime_system_memory_total_gb"),
            row.get("runtime_hf_endpoint"),
            row.get("runtime_hf_home"),
            row.get("runtime_transformers_cache"),
            row.get("runtime_hf_datasets_cache"),
            row.get("runtime_torch"),
            row.get("runtime_transformers"),
            row.get("runtime_bitsandbytes_available"),
            row.get("runtime_bitsandbytes_version"),
            row.get("runtime_cuda_available"),
            row.get("runtime_cuda_version"),
            row.get("runtime_cudnn_version"),
            row.get("runtime_cuda_visible_devices"),
            row.get("runtime_gpu_count"),
            row.get("runtime_gpu_name"),
            row.get("runtime_gpu_total_memory_gb"),
        ]
        for row in rows
    ]
    resource_rows = [
        [
            row.get("model_name"),
            row.get("compression_method"),
            row.get("resource_backend"),
            row.get("resource_model_load_error"),
            row.get("resource_seqlen"),
            row.get("resource_prompt_source"),
            row.get("resource_prompt_count"),
            row.get("resource_input_length"),
            row.get("resource_max_new_tokens"),
            row.get("resource_sparsity_sample_tensors"),
            row.get("storage_bytes"),
            row.get("storage_gb"),
            row.get("effective_storage_bytes_sparse_approx"),
            row.get("effective_storage_gb_sparse_approx"),
            row.get("effective_storage_bytes_quantized_approx"),
            row.get("effective_storage_gb_quantized_approx"),
            row.get("effective_storage_quantized_bits"),
            row.get("effective_storage_quantized_backend"),
            row.get("quantization_storage_format"),
            row.get("quantization_storage_real"),
            row.get("quantization_storage_dtypes"),
            row.get("checkpoint_quant_method"),
            row.get("checkpoint_weight_bits"),
            row.get("checkpoint_activation_bits"),
            row.get("checkpoint_group_size"),
            row.get("sparsity"),
            row.get("flops"),
            row.get("flops_num"),
            row.get("effective_flops_num_sparse_approx"),
            row.get("macs"),
            row.get("compute_backend"),
            row.get("total_params"),
        ]
        for row in rows
    ]

    parts = [
        "# LLMCBench Edge Compression Report",
        "",
        f"- Source results: `{args.results}`",
        f"- Rows: {len(rows)}",
        "",
        "## Accuracy",
        "",
        _table(
            [
                "Model",
                "Method",
                "MMLU",
                "ARC-Challenge",
                "HellaSwag",
                "PIQA",
                "WinoGrande",
                "Avg",
                "Delta vs Dense",
                "Retention vs Dense",
            ],
            accuracy_rows,
        ),
        "",
        "## Accuracy Metric Keys",
        "",
        _table(
            [
                "Model",
                "Method",
                "MMLU Metric",
                "ARC-Challenge Metric",
                "HellaSwag Metric",
                "PIQA Metric",
                "WinoGrande Metric",
            ],
            accuracy_metric_rows,
        ),
        "",
        "## Accuracy Settings",
        "",
        _table(
            [
                "Model",
                "Method",
                "Dtype",
                "LM Eval Dtype",
                "Batch",
                "Num Fewshot",
                "Limit",
                "Data Dir",
                "HF Datasets Cache",
                "Max Memory",
                "Split Tasks",
                "Continue On Error",
            ],
            accuracy_setting_rows,
        ),
        "",
        "## Deployment",
        "",
        _table(
            [
                "Model",
                "Method",
                "Tokens/s",
                "Elapsed Seconds",
                "Generated Tokens",
                "Dtype",
                "Max Memory",
                "Device Map",
                "Prompt Source",
                "Prompt Count",
                "Deploy Batch",
                "Input Len",
                "Input Tokens",
                "New Tokens",
                "Warmup",
                "Repeat",
                "Peak Allocated GB",
                "Peak Reserved GB",
                "Limit GB",
                "Measured Peak GB",
                "Exceeded",
            ],
            deployment_rows,
        ),
        "",
        "## Compression",
        "",
        _table(
            [
                "Model",
                "Method",
                "Source Path",
                "Revision",
                "Base Model Repository",
                "Base Model Revision",
                "Base Model Attested",
                "Backend",
                "Dtype",
                "Seq Len",
                "Max Memory",
                "Attention",
                "Materialized",
                "Precompressed",
                "Bits",
                "Group",
                "Desc Act",
                "AWQ Version",
                "Target Sparsity",
                "Sparsity Semantics",
                "Prune Pattern",
                "Prune Structure",
                "Prune N",
                "Prune M",
                "Prune LM Head",
                "Pruning Algorithm",
                "Pruning Buffer",
                "SparseGPT Block",
                "SparseGPT Damp",
                "Calib Samples",
                "Calib Batch",
                "Calibration Protocol",
                "Calibration Source",
                "Sampling Algorithm",
                "Calibration Seed",
                "Calibration Dataset",
                "Calibration Shard",
                "Calibration Revision",
                "Calibration Tokens",
                "Calibration Token SHA256",
                "Paper Protocol",
                "Paper Attested",
                "GPTQ True Sequential",
                "GPTQ Symmetric",
                "AWQ Calibration",
                "AWQ Grid",
                "SmoothQuant Alpha",
                "SmoothQuant Alpha Selection",
                "Official Repository",
                "Official Commit",
                "Official Implementation Verified",
                "OmniQuant A Bits",
                "OmniQuant Epochs",
                "OmniQuant Seed",
                "OmniQuant LWC",
                "OmniQuant LET",
                "Note",
                "External Command",
                "External Returncode",
                "Compressed Path",
            ],
            compression_rows,
        ),
        "",
        "## Runtime",
        "",
        _table(
            [
                "Model",
                "Method",
                "Python",
                "Profile",
                "CPU Count",
                "System Memory GB",
                "HF Endpoint",
                "HF Home",
                "Transformers Cache",
                "HF Datasets Cache",
                "Torch",
                "Transformers",
                "bitsandbytes",
                "bitsandbytes Version",
                "CUDA Available",
                "CUDA Version",
                "cuDNN Version",
                "CUDA Visible Devices",
                "GPU Count",
                "GPU",
                "GPU Memory GB",
            ],
            runtime_rows,
        ),
        "",
        "## Resources",
        "",
        _table(
            [
                "Model",
                "Method",
                "Resource Backend",
                "Model Load Error",
                "Seq Len",
                "Prompt Source",
                "Prompt Count",
                "Input Length",
                "Max New Tokens",
                "Sparsity Samples",
                "Storage Bytes",
                "Storage GB",
                "Effective Storage Bytes",
                "Effective Storage GB",
                "Quantized Storage Bytes",
                "Quantized Storage GB",
                "Quantized Bits",
                "Quantized Storage Backend",
                "Checkpoint Quant Format",
                "Real Quantized Storage",
                "Checkpoint Dtypes",
                "Checkpoint Quant Method",
                "Checkpoint W Bits",
                "Checkpoint A Bits",
                "Checkpoint Group",
                "Measured Sparsity",
                "FLOPs",
                "FLOPs Num",
                "Effective FLOPs Num",
                "MACs",
                "Compute Backend",
                "Params",
            ],
            resource_rows,
        ),
        "",
    ]
    if isinstance(asset_audit_report, dict) or isinstance(preflight_report, dict):
        parts.extend(
            [
                "## Preflight Checks",
                "",
                _table(
                    ["Report", "OK"],
                    [
                        ["Asset Audit", asset_audit_report.get("ok") if isinstance(asset_audit_report, dict) else None],
                        ["Preflight", preflight_report.get("ok") if isinstance(preflight_report, dict) else None],
                    ],
                ),
                "",
            ]
        )
    if isinstance(asset_audit_report, dict) and isinstance(asset_audit_report.get("paths"), dict):
        path_rows = []
        for name, item in asset_audit_report["paths"].items():
            if not isinstance(item, dict) or "free_gb" not in item:
                continue
            mount = item.get("mount") if isinstance(item.get("mount"), dict) else {}
            block = item.get("block_device") if isinstance(item.get("block_device"), dict) else {}
            path_rows.append(
                [
                    name,
                    item.get("path"),
                    item.get("probe_path"),
                    round(item.get("free_gb"), 3) if isinstance(item.get("free_gb"), (int, float)) else item.get("free_gb"),
                    round(item.get("total_gb"), 3) if isinstance(item.get("total_gb"), (int, float)) else item.get("total_gb"),
                    mount.get("target"),
                    mount.get("source"),
                    mount.get("fstype"),
                    block.get("device"),
                    block.get("is_non_rotational"),
                ]
            )
        if path_rows:
            parts.extend(
                [
                    "## Asset Paths",
                    "",
                    _table(
                        [
                            "Name",
                            "Path",
                            "Probe Path",
                            "Free GB",
                            "Total GB",
                            "Mount",
                            "Source",
                            "FSType",
                            "Block Device",
                            "Non-Rotational",
                        ],
                        path_rows,
                    ),
                    "",
                ]
            )
    if isinstance(prepare_report, dict) or isinstance(prepare_metadata_report, dict) or isinstance(prepare_precompressed_report, dict):
        precompressed_blocker_rows = _precompressed_blocker_rows(prepare_precompressed_report)
        parts.extend(
            [
                "## Asset Preparation",
                "",
                _table(
                    ["Report", "OK", "Dry Run", "Metadata Only", "Models", "Files"],
                    [
                        [
                            "Prepare",
                            prepare_report.get("ok") if isinstance(prepare_report, dict) else None,
                            prepare_report.get("dry_run") if isinstance(prepare_report, dict) else None,
                            prepare_report.get("metadata_only") if isinstance(prepare_report, dict) else None,
                            len(prepare_report.get("models", [])) if isinstance(prepare_report, dict) and isinstance(prepare_report.get("models"), list) else None,
                            len(prepare_report.get("files", [])) if isinstance(prepare_report, dict) and isinstance(prepare_report.get("files"), list) else None,
                        ],
                        [
                            "Metadata Prepare",
                            prepare_metadata_report.get("ok") if isinstance(prepare_metadata_report, dict) else None,
                            prepare_metadata_report.get("dry_run") if isinstance(prepare_metadata_report, dict) else None,
                            prepare_metadata_report.get("metadata_only") if isinstance(prepare_metadata_report, dict) else None,
                            len(prepare_metadata_report.get("models", [])) if isinstance(prepare_metadata_report, dict) and isinstance(prepare_metadata_report.get("models"), list) else None,
                            len(prepare_metadata_report.get("files", [])) if isinstance(prepare_metadata_report, dict) and isinstance(prepare_metadata_report.get("files"), list) else None,
                        ],
                        [
                            "Precompressed Prepare",
                            prepare_precompressed_report.get("ok") if isinstance(prepare_precompressed_report, dict) else None,
                            prepare_precompressed_report.get("dry_run") if isinstance(prepare_precompressed_report, dict) else None,
                            prepare_precompressed_report.get("metadata_only") if isinstance(prepare_precompressed_report, dict) else None,
                            len(prepare_precompressed_report.get("items", [])) if isinstance(prepare_precompressed_report, dict) and isinstance(prepare_precompressed_report.get("items"), list) else None,
                            None,
                        ],
                    ],
                ),
                "",
            ]
        )
        if precompressed_blocker_rows:
            parts.extend(
                [
                    "### Precompressed Preparation Blockers",
                    "",
                    _table(["Model", "Method", "Reason", "Repo Status", "Detail", "Remediation"], precompressed_blocker_rows),
                    "",
                ]
            )
    if isinstance(prewarm_report, dict):
        failed = prewarm_report.get("failed") if isinstance(prewarm_report.get("failed"), list) else []
        attempted = prewarm_report.get("attempted") if isinstance(prewarm_report.get("attempted"), list) else []
        succeeded = prewarm_report.get("succeeded") if isinstance(prewarm_report.get("succeeded"), list) else []
        tasks = prewarm_report.get("tasks") if isinstance(prewarm_report.get("tasks"), list) else []
        targets = prewarm_report.get("targets") if isinstance(prewarm_report.get("targets"), list) else []
        parts.extend(
            [
                "## Prewarm",
                "",
                f"- OK: {_fmt(prewarm_report.get('ok'))}",
                f"- Dtype: {_fmt(prewarm_report.get('dtype'))}",
                f"- Attention: {_fmt(prewarm_report.get('attn_implementation'))}",
                f"- Expanded tasks: {len(tasks)}",
                f"- Targets: {prewarm_report.get('target_count', len(targets))}",
                f"- Attempted pairs: {len(attempted)}",
                f"- Succeeded pairs: {len(succeeded)}",
                f"- Failed pairs: {len(failed)}",
                "",
            ]
        )
        if failed:
            parts.extend(
                [
                    "### Failed Prewarm Tasks",
                    "",
                    _table(["Model", "Task"], _pair_rows(failed)),
                    "",
                ]
            )
    if run_report:
        parts.extend(
            [
                "## Matrix Run Status",
                "",
                f"- Completed: {len(run_report.get('completed', []))}",
                f"- Skipped existing: {len(run_report.get('skipped', []))}",
                f"- Skipped external not configured: {len(run_report.get('skipped_external_not_configured', []))}",
                f"- Failed: {len(run_report.get('failed', []))}",
                "",
            ]
        )
        skipped_external = run_report.get("skipped_external_not_configured", [])
        failed = run_report.get("failed", [])
        if skipped_external:
            parts.extend(
                [
                    "### Skipped External Methods",
                    "",
                    _table(
                        ["Model", "Method", "Reason"],
                        [[item.get("model"), item.get("method"), item.get("reason")] for item in skipped_external],
                    ),
                    "",
                ]
            )
        if failed:
            parts.extend(
                [
                    "### Failed Jobs",
                    "",
                    _table(
                        ["Model", "Method", "Return Code", "Log"],
                        [[item.get("model"), item.get("method"), item.get("returncode"), item.get("log")] for item in failed],
                    ),
                    "",
                ]
            )
        jobs = run_report.get("jobs", [])
        if jobs:
            parts.extend(
                [
                    "### Job Details",
                    "",
                    _table(
                        ["Model", "Method", "Status", "Elapsed Seconds", "Reason", "Log"],
                        [
                            [
                                item.get("model"),
                                item.get("method"),
                                item.get("status"),
                                item.get("elapsed_seconds"),
                                item.get("reason"),
                                item.get("log"),
                            ]
                            for item in jobs
                        ],
                    ),
                    "",
                ]
            )
    report = "\n".join(parts)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    print(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render LLMCBench edge matrix results as a Markdown report.")
    parser.add_argument("--results", required=True, help="matrix_results.json from collect_edge_results.py")
    parser.add_argument("--run-report", default=None, help="Optional matrix_run_report.json from run_edge_matrix.py")
    parser.add_argument("--asset-audit-report", default=None, help="Optional edge_asset_audit.json from audit_edge_assets.py")
    parser.add_argument("--preflight-report", default=None, help="Optional edge_preflight_report.json from run_edge_preflight.py")
    parser.add_argument("--prewarm-report", default=None, help="Optional edge_prewarm_report.json from prewarm_lm_eval_tasks.py")
    parser.add_argument("--prepare-report", default=None, help="Optional edge_prepare_assets_report.json from prepare_edge_assets.py")
    parser.add_argument("--prepare-metadata-report", default=None, help="Optional edge_prepare_metadata_report.json from prepare_edge_assets.py")
    parser.add_argument("--prepare-precompressed-report", default=None, help="Optional edge_prepare_precompressed_report.json from prepare_precompressed_assets.py")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    render(build_parser().parse_args())


if __name__ == "__main__":
    main()
