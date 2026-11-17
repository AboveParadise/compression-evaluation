import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List


def _load_config(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_requires_paper_protocol_validation(path: str) -> bool:
    config = _load_config(path)
    return bool(config.get("require_paper_protocol_validation"))


def _validate_paper_protocol_preflight(args: argparse.Namespace) -> None:
    if not config_requires_paper_protocol_validation(args.config):
        return
    cmd = [
        sys.executable,
        "tools/validate_paper_protocol.py",
        "--config",
        args.config,
        "--require-official-checkouts",
        "--require-base-models",
    ]
    for model_name in args.only_model:
        cmd.extend(["--only-model", model_name])
    for method in args.only_method:
        cmd.extend(["--only-method", method])
    print(f"[paper-protocol-preflight] {' '.join(shlex.quote(part) for part in cmd)}")
    proc = subprocess.run(cmd, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(
            "Paper/official-code protocol preflight failed. Resolve every reported setting "
            "before starting a strict matrix job."
        )


def _selected(items: Iterable[Dict], names: List[str]) -> List[Dict]:
    if not names:
        return list(items)
    wanted = set(names)
    return [item for item in items if item["name"] in wanted]


def _selected_methods(methods: Iterable[str], names: List[str]) -> List[str]:
    if not names:
        return list(methods)
    wanted = set(names)
    return [method for method in methods if method in wanted]


def _job_output_root(defaults: Dict, model_name: str, method: str) -> Path:
    return Path(defaults["output_root"]).expanduser() / model_name / method


def _expand_env_value(value):
    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        name = match.group("name")
        default = match.group("default")
        return os.environ.get(name, default or "")

    expanded = re.sub(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}", replace, value)
    return os.path.expandvars(expanded)


def _job_defaults(config: Dict, model_name: str, method: str) -> Dict:
    merged = dict(config["defaults"])
    ignored_keys = {"output_root", "cuda_visible_devices"}
    for section_name, key in (
        ("method_overrides", method),
        ("model_overrides", model_name),
        ("model_method_overrides", f"{model_name}/{method}"),
    ):
        section = config.get(section_name, {})
        if isinstance(section, dict) and isinstance(section.get(key), dict):
            merged.update({name: value for name, value in section[key].items() if name not in ignored_keys})
    merged = {name: _expand_env_value(value) for name, value in merged.items()}
    return merged


def _precompressed_path(config: Dict, model_name: str, method: str) -> str:
    paths = config.get("precompressed_paths", {})
    if not isinstance(paths, dict):
        return ""
    model_paths = paths.get(model_name, {})
    if isinstance(model_paths, dict):
        return model_paths.get(method, "") or ""
    return ""


def _build_env(defaults: Dict, model: Dict, method: str, output_root: Path, precompressed_path: str = "") -> Dict[str, str]:
    env = os.environ.copy()
    mapping = {
        "HF_ENDPOINT": defaults["hf_endpoint"],
        "PYTHON_BIN": defaults.get("python_bin", ""),
        "EDGE_RUNTIME_PROFILE": defaults.get("runtime_profile", ""),
        "HF_CACHE_DIR": defaults["hf_cache_dir"],
        "DATA_DIR": defaults["data_dir"],
        "REVISION": "" if defaults.get("revision") is None else str(defaults.get("revision")),
        "CALIBRATION_FILE": defaults["calibration_file"],
        "PROMPT_FILE": defaults["prompt_file"],
        "CUDA_VISIBLE_DEVICES": defaults["cuda_visible_devices"],
        "BASE_MODEL_PATH": precompressed_path or model["path"],
        "BASE_MODEL_REPOSITORY": str(model.get("repository", "")),
        "BASE_MODEL_REVISION": str(model.get("revision", "")),
        "BASE_MODEL_ATTESTATION_PATH": str(
            Path(model["path"]).expanduser() / model.get("attestation_file", ".llmcbench_base_model_attestation.json")
        ),
        "MODEL_NAME": model["name"],
        "COMPRESSION_METHOD": method,
        "OUTPUT_ROOT": str(output_root),
        "SEQLEN": str(defaults["seqlen"]),
        "CALIBRATION_SAMPLES": str(defaults["calibration_samples"]),
        "CALIBRATION_BATCH_SIZE": str(defaults["calibration_batch_size"]),
        "PRUNING_CALIBRATION_SOURCE": str(defaults.get("pruning_calibration_source", "c4")),
        "CALIBRATION_SEED": str(defaults.get("calibration_seed", 0)),
        "CALIBRATION_DATASET": str(defaults.get("calibration_dataset", "allenai/c4")),
        "CALIBRATION_DATASET_CONFIG": str(defaults.get("calibration_dataset_config", "allenai--c4")),
        "CALIBRATION_C4_DATA_FILE": str(defaults.get("calibration_c4_data_file", "en/c4-train.00000-of-01024.json.gz")),
        "CALIBRATION_C4_REVISION": str(defaults.get("calibration_c4_revision", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2")),
        "CALIBRATION_DATASET_CACHE": str(defaults.get("calibration_dataset_cache", defaults["data_dir"])),
        "SPARSITY": str(defaults["sparsity"]),
        "PRUNE_PATTERN": str(defaults.get("prune_pattern", "unstructured")),
        "PRUNE_LM_HEAD": "1" if defaults.get("prune_lm_head", False) else "0",
        "SPARSEGPT_BLOCKSIZE": str(defaults.get("sparsegpt_blocksize", 128)),
        "SPARSEGPT_PERCDAMP": str(defaults.get("sparsegpt_percdamp", 0.01)),
        "PRUNING_BUFFER_DEVICE": str(defaults.get("pruning_buffer_device", "cpu")),
        "QUANT_BITS": str(defaults["quant_bits"]),
        "GROUP_SIZE": str(defaults["group_size"]),
        "DESC_ACT": "1" if defaults.get("desc_act", False) else "0",
        "GPTQ_CALIBRATION_SOURCE": str(defaults.get("gptq_calibration_source", "c4")),
        "GPTQ_DAMP_PERCENT": str(defaults.get("gptq_damp_percent", 0.01)),
        "GPTQ_TRUE_SEQUENTIAL": "1" if defaults.get("gptq_true_sequential", False) else "0",
        "GPTQ_STATIC_GROUPS": "1" if defaults.get("gptq_static_groups", False) else "0",
        "GPTQ_ASYMMETRIC": "1" if defaults.get("gptq_asymmetric", False) else "0",
        "AWQ_VERSION": str(defaults.get("awq_version", "GEMM")),
        "AWQ_CALIBRATION_SOURCE": str(defaults.get("awq_calibration_source", "pileval")),
        "AWQ_CALIBRATION_SEED": str(defaults.get("awq_calibration_seed", 42)),
        "AWQ_CALIBRATION_SAMPLES": str(defaults.get("awq_calibration_samples", 128)),
        "AWQ_CALIBRATION_SEQLEN": str(defaults.get("awq_calibration_seqlen", 512)),
        "AWQ_GRID_SIZE": str(defaults.get("awq_grid_size", 20)),
        "SMOOTHQUANT_ALPHA": ""
        if defaults.get("smoothquant_alpha") is None
        else str(defaults.get("smoothquant_alpha")),
        "SMOOTHQUANT_CALIBRATION_SOURCE": str(defaults.get("smoothquant_calibration_source", "pile_validation")),
        "SMOOTHQUANT_CALIBRATION_SENTENCES": str(defaults.get("smoothquant_calibration_sentences", 512)),
        "SMOOTHQUANT_ALPHA_SELECTION": str(defaults.get("smoothquant_alpha_selection", "official_grid_search")),
        "SMOOTHQUANT_OFFICIAL_CODE_DIR": str(defaults.get("smoothquant_official_code_dir", "")),
        "SMOOTHQUANT_OFFICIAL_REPOSITORY": str(defaults.get("smoothquant_official_repository", "")),
        "SMOOTHQUANT_OFFICIAL_COMMIT": str(defaults.get("smoothquant_official_commit", "")),
        "OMNIQUANT_ABITS": str(defaults.get("omniquant_abits", 16)),
        "OMNIQUANT_EPOCHS": str(defaults.get("omniquant_epochs", 20)),
        "OMNIQUANT_NSAMPLES": str(defaults.get("omniquant_nsamples", 128)),
        "OMNIQUANT_SEQLEN": str(defaults.get("omniquant_seqlen", 2048)),
        "OMNIQUANT_SEED": str(defaults.get("omniquant_seed", 2)),
        "OMNIQUANT_CALIBRATION_SOURCE": str(defaults.get("omniquant_calibration_source", "wikitext2")),
        "OMNIQUANT_LWC": "1" if defaults.get("omniquant_lwc", False) else "0",
        "OMNIQUANT_LET": "1" if defaults.get("omniquant_let", False) else "0",
        "OMNIQUANT_AUG_LOSS": "1" if defaults.get("omniquant_aug_loss", False) else "0",
        "OMNIQUANT_OFFICIAL_CODE_DIR": str(defaults.get("omniquant_official_code_dir", "")),
        "OMNIQUANT_OFFICIAL_REPOSITORY": str(defaults.get("omniquant_official_repository", "")),
        "OMNIQUANT_OFFICIAL_COMMIT": str(defaults.get("omniquant_official_commit", "")),
        "PAPER_PROTOCOL": str(defaults.get("paper_protocol", "")),
        "TARGET_MODEL_ADAPTATION": "1" if defaults.get("target_model_adaptation", False) else "0",
        "REQUIRE_PAPER_PROTOCOL_ATTESTATION": "1"
        if defaults.get("require_paper_protocol_attestation", False)
        else "0",
        "MIN_FREE_GPU_MEMORY_GB": str(defaults.get("min_free_gpu_memory_gb", 0)),
        "DEVICE_MAP": str(defaults.get("device_map", "auto")),
        "DTYPE": defaults.get("dtype", "fp16"),
        "ATTN_IMPLEMENTATION": defaults.get("attn_implementation", ""),
        "MAX_MEMORY": defaults.get("max_memory", ""),
        "TASKS": defaults["tasks"],
        "SPLIT_ACCURACY_TASKS": "1" if defaults.get("split_accuracy_tasks", False) else "0",
        "CONTINUE_ON_ACCURACY_ERROR": "1" if defaults.get("continue_on_accuracy_error", False) else "0",
        "BATCH_SIZE": str(defaults["batch_size"]),
        "NUM_FEWSHOT": "" if defaults.get("num_fewshot") is None else str(defaults.get("num_fewshot")),
        "LIMIT": "" if defaults.get("accuracy_limit") is None else str(defaults.get("accuracy_limit")),
        "INPUT_LENGTH": str(defaults["input_length"]),
        "MAX_NEW_TOKENS": str(defaults["max_new_tokens"]),
        "DEPLOYMENT_BATCH_SIZE": str(defaults.get("deployment_batch_size", 1)),
        "SPARSITY_SAMPLE_TENSORS": str(defaults.get("sparsity_sample_tensors", -1)),
        "WARMUP": str(defaults["warmup"]),
        "REPEAT": str(defaults["repeat"]),
        "DEPLOYMENT_MEMORY_LIMIT_GB": str(defaults.get("deployment_memory_limit_gb", 0)),
        "USE_PRECOMPRESSED": "1" if precompressed_path else "0",
    }
    for key, value in mapping.items():
        if value not in (None, ""):
            env[key] = value
    return env


def _external_command(config: Dict, method: str) -> str:
    commands = config.get("external_commands", {})
    configured = commands.get(method) or ""
    env_name = f"LLMCBENCH_{method.upper()}_CMD"
    return configured or os.environ.get(env_name, "")


def _tail_text(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _pipeline_artifacts(output_root: Path) -> List[Path]:
    return [
        output_root / "summary.json",
        output_root / "compression.json",
        output_root / "accuracy.json",
        output_root / "deployment.json",
        output_root / "resource.json",
    ]


def _missing_pipeline_artifacts(output_root: Path) -> List[str]:
    return [str(path) for path in _pipeline_artifacts(output_root) if not path.exists()]


def _run_logged(cmd: List[str], env: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_cmd = " ".join(shlex.quote(part) for part in cmd)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {shell_cmd}\n")
        handle.flush()
        proc = subprocess.run(
            ["bash", "-o", "pipefail", "-c", f"{shell_cmd} 2>&1 | tee -a {shlex.quote(str(log_path))}"],
            env=env,
            text=True,
            check=False,
        )
    return proc.returncode


def _git_value(args: List[str]) -> str:
    proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git_dirty() -> bool:
    proc = subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=False)
    return bool(proc.stdout.strip()) if proc.returncode == 0 else True


def _write_resolved_config(config: Dict, defaults: Dict, models: List[Dict], methods: List[str], args: argparse.Namespace) -> str:
    output_root = Path(defaults["output_root"]).expanduser()
    path = output_root / "matrix_config_resolved.json"
    payload = {
        "source_config": args.config,
        "code": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "branch": _git_value(["branch", "--show-current"]),
            "origin": _git_value(["remote", "get-url", "origin"]),
            "dirty": _git_dirty(),
        },
        "selected_models": [model["name"] for model in models],
        "selected_methods": methods,
        "defaults": defaults,
        "models": models,
        "methods": methods,
        "require_paper_protocol_validation": bool(config.get("require_paper_protocol_validation")),
        "external_commands": config.get("external_commands", {}),
        "precompressed_paths": config.get("precompressed_paths", {}),
        "method_overrides": config.get("method_overrides", {}),
        "model_overrides": config.get("model_overrides", {}),
        "model_method_overrides": config.get("model_method_overrides", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _finalize_outputs(args: argparse.Namespace, defaults: Dict) -> Dict:
    output_root = Path(defaults["output_root"]).expanduser()
    results_json = output_root / "matrix_results.json"
    results_csv = output_root / "matrix_results.csv"
    run_report = output_root / "matrix_run_report.json"
    edge_report = output_root / "edge_report.md"
    completion_audit = output_root / "completion_audit.json"
    resolved_config = output_root / "matrix_config_resolved.json"
    asset_audit_report = output_root / "edge_asset_audit.json"
    preflight_report = output_root / "edge_preflight_report.json"
    prewarm_report = output_root / "edge_prewarm_report.json"
    handoff_manifest = output_root / "edge_handoff_manifest.json"
    target_readiness = output_root / "target_readiness.json"
    optional_backends_report = output_root / "edge_optional_backends_report.json"
    quant_runtime_report = output_root / "edge_quant_runtime_report.json"
    target_paths_report = output_root / "edge_target_paths_report.json"
    prepare_report = output_root / "edge_prepare_assets_report.json"
    prepare_metadata_report = output_root / "edge_prepare_metadata_report.json"
    prepare_precompressed_report = output_root / "edge_prepare_precompressed_report.json"
    finalize_report = output_root / "matrix_finalize_report.json"
    if finalize_report.exists():
        finalize_report.unlink()

    steps = [
        [
            sys.executable,
            "tools/collect_edge_results.py",
            "--root",
            str(output_root),
            "--output-json",
            str(results_json),
            "--output-csv",
            str(results_csv),
        ],
        [
            sys.executable,
            "tools/validate_edge_results.py",
            "--config",
            args.config,
            "--results",
            str(results_json),
        ],
        [
            sys.executable,
            "tools/render_edge_report.py",
            "--results",
            str(results_json),
            "--output",
            str(edge_report),
        ],
        [
            sys.executable,
            "tools/audit_edge_completion.py",
            "--config",
            args.config,
            "--root",
            str(output_root),
            "--output",
            str(completion_audit),
        ],
    ]
    if config_requires_paper_protocol_validation(args.config):
        steps.insert(
            2,
            [
                sys.executable,
                "tools/validate_paper_protocol.py",
                "--config",
                args.config,
                "--results",
                str(results_json),
                "--require-official-checkouts",
                "--require-base-models",
            ],
        )
    validation_step = steps[1]
    paper_validation_step = steps[2] if len(steps) == 5 else None
    render_step = steps[-2]
    audit_step = steps[-1]
    target_readiness_step = [
        sys.executable,
        "tools/audit_target_readiness.py",
        "--config",
        args.config,
        "--root",
        str(output_root),
        "--output",
        str(target_readiness),
        "--check-method-readiness",
    ]
    for model_name in args.only_model:
        validation_step.extend(["--only-model", model_name])
        audit_step.extend(["--only-model", model_name])
        if paper_validation_step is not None:
            paper_validation_step.extend(["--only-model", model_name])
    for method in args.only_method:
        validation_step.extend(["--only-method", method])
        audit_step.extend(["--only-method", method])
        if paper_validation_step is not None:
            paper_validation_step.extend(["--only-method", method])
    if args.allow_unexpected:
        validation_step.append("--allow-unexpected")
        audit_step.append("--allow-unexpected")
    if args.allow_skipped_external:
        audit_step.append("--allow-skipped-external")
    if args.require_runtime_python_prefix:
        audit_step.extend(["--require-runtime-python-prefix", args.require_runtime_python_prefix])
    if args.require_runtime_gpu_name_contains:
        audit_step.extend(["--require-runtime-gpu-name-contains", args.require_runtime_gpu_name_contains])
    if args.require_runtime_cuda_prefix:
        audit_step.extend(["--require-runtime-cuda-prefix", args.require_runtime_cuda_prefix])
    if args.require_runtime_gpu_total_memory_min_gb > 0:
        audit_step.extend(
            [
                "--require-runtime-gpu-total-memory-min-gb",
                str(args.require_runtime_gpu_total_memory_min_gb),
            ]
        )
    if args.require_runtime_gpu_count > 0:
        audit_step.extend(["--require-runtime-gpu-count", str(args.require_runtime_gpu_count)])
    if args.require_runtime_cudnn:
        audit_step.append("--require-runtime-cudnn")
    if args.require_runtime_torch_prefix:
        audit_step.extend(["--require-runtime-torch-prefix", args.require_runtime_torch_prefix])
    if args.require_runtime_transformers:
        audit_step.append("--require-runtime-transformers")
    if args.require_runtime_bitsandbytes:
        audit_step.append("--require-runtime-bitsandbytes")
    if args.require_runtime_cpu_count_min > 0:
        audit_step.extend(["--require-runtime-cpu-count-min", str(args.require_runtime_cpu_count_min)])
    if args.require_runtime_system_memory_min_gb > 0:
        audit_step.extend(["--require-runtime-system-memory-min-gb", str(args.require_runtime_system_memory_min_gb)])
    if args.require_runtime_hf_endpoint:
        audit_step.extend(["--require-runtime-hf-endpoint", args.require_runtime_hf_endpoint])
    if args.require_code_match:
        audit_step.append("--require-code-match")
    if args.require_clean_worktree:
        audit_step.append("--require-clean-worktree")
    if args.require_prewarm_report:
        audit_step.append("--require-prewarm-report")
        target_readiness_step.append("--require-prewarm")
    if args.require_prewarm_all_models:
        audit_step.append("--require-prewarm-all-models")
    if args.require_asset_audit_report:
        audit_step.append("--require-asset-audit-report")
    if args.require_non_rotational_storage:
        audit_step.append("--require-non-rotational-storage")
        target_readiness_step.append("--require-non-rotational-storage")
    if args.require_preflight_report:
        audit_step.append("--require-preflight-report")
    if args.require_handoff_manifest:
        audit_step.append("--require-handoff-manifest")
        target_readiness_step.append("--require-handoff-manifest")
    if args.require_optional_backends_report:
        audit_step.append("--require-optional-backends-report")
        target_readiness_step.append("--require-optional-backends-report")
    if args.require_quant_runtime_report:
        audit_step.append("--require-quant-runtime-report")
        target_readiness_step.append("--require-quant-runtime-report")
    if args.require_target_paths_report:
        audit_step.append("--require-target-paths-report")
    if args.require_prepare_report:
        audit_step.append("--require-prepare-report")
    if args.require_prepare_metadata_report:
        audit_step.append("--require-prepare-metadata-report")
    if args.require_prepare_precompressed_report:
        audit_step.append("--require-prepare-precompressed-report")
        target_readiness_step.append("--require-prepare-precompressed-report")
    if run_report.exists():
        render_step.extend(["--run-report", str(run_report)])
    if asset_audit_report.exists():
        render_step.extend(["--asset-audit-report", str(asset_audit_report)])
    if preflight_report.exists():
        render_step.extend(["--preflight-report", str(preflight_report)])
    if prewarm_report.exists():
        render_step.extend(["--prewarm-report", str(prewarm_report)])
    if prepare_report.exists():
        render_step.extend(["--prepare-report", str(prepare_report)])
    if prepare_metadata_report.exists():
        render_step.extend(["--prepare-metadata-report", str(prepare_metadata_report)])
    if prepare_precompressed_report.exists():
        render_step.extend(["--prepare-precompressed-report", str(prepare_precompressed_report)])
    if args.require_target_readiness_report:
        final_audit_step = list(audit_step)
        final_audit_step.append("--require-target-readiness-report")
        steps.extend([target_readiness_step, final_audit_step])

    records = []
    for cmd in steps:
        print(f"[finalize] {' '.join(cmd)}")
        if args.dry_run:
            records.append({"cmd": cmd, "status": "dry_run"})
            continue
        proc = subprocess.run(cmd, text=True, check=False)
        record = {"cmd": cmd, "returncode": proc.returncode, "ok": proc.returncode == 0}
        records.append(record)
        if proc.returncode != 0:
            report = {
                "ok": False,
                "output_root": str(output_root),
                "results_json": str(results_json),
                "results_csv": str(results_csv),
                "edge_report": str(edge_report),
                "completion_audit": str(completion_audit),
                "edge_asset_audit": str(asset_audit_report),
                "edge_preflight_report": str(preflight_report),
                "edge_prewarm_report": str(prewarm_report),
                "edge_handoff_manifest": str(handoff_manifest),
                "target_readiness": str(target_readiness),
                "edge_optional_backends_report": str(optional_backends_report),
                "edge_quant_runtime_report": str(quant_runtime_report),
                "edge_target_paths_report": str(target_paths_report),
                "edge_prepare_assets_report": str(prepare_report),
                "edge_prepare_metadata_report": str(prepare_metadata_report),
                "edge_prepare_precompressed_report": str(prepare_precompressed_report),
                "steps": records,
            }
            raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))

    report = {
        "ok": all(item.get("ok", False) or item.get("status") == "dry_run" for item in records),
        "output_root": str(output_root),
        "results_json": str(results_json),
        "results_csv": str(results_csv),
        "edge_report": str(edge_report),
        "completion_audit": str(completion_audit),
        "edge_asset_audit": str(asset_audit_report),
        "edge_preflight_report": str(preflight_report),
        "edge_prewarm_report": str(prewarm_report),
        "edge_handoff_manifest": str(handoff_manifest),
        "target_readiness": str(target_readiness),
        "edge_optional_backends_report": str(optional_backends_report),
        "edge_quant_runtime_report": str(quant_runtime_report),
        "edge_target_paths_report": str(target_paths_report),
        "edge_prepare_assets_report": str(prepare_report),
        "edge_prepare_metadata_report": str(prepare_metadata_report),
        "edge_prepare_precompressed_report": str(prepare_precompressed_report),
        "resolved_config": str(resolved_config),
        "steps": records,
    }
    finalize_report.parent.mkdir(parents=True, exist_ok=True)
    finalize_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def run_matrix(args: argparse.Namespace) -> None:
    _validate_paper_protocol_preflight(args)
    config = _load_config(args.config)
    defaults = config["defaults"]
    if args.output_root:
        defaults["output_root"] = args.output_root
    if args.cuda_visible_devices:
        defaults["cuda_visible_devices"] = args.cuda_visible_devices

    models = _selected(config["models"], args.only_model)
    methods = _selected_methods(config["methods"], args.only_method)
    if not models:
        raise SystemExit("No models selected.")
    if not methods:
        raise SystemExit("No methods selected.")
    resolved_config = _write_resolved_config(config, defaults, models, methods, args)

    completed = []
    failed = []
    skipped = []
    skipped_external = []
    job_records = []
    for model in models:
        for method in methods:
            job_defaults = _job_defaults(config, model["name"], method)
            if args.output_root:
                job_defaults["output_root"] = args.output_root
            if args.cuda_visible_devices:
                job_defaults["cuda_visible_devices"] = args.cuda_visible_devices
            output_root = _job_output_root(job_defaults, model["name"], method)
            summary_path = output_root / "summary.json"
            log_path = output_root / "job.log"
            if summary_path.exists() and not args.force:
                missing_existing = _missing_pipeline_artifacts(output_root)
                if missing_existing:
                    failure = {
                        "model": model["name"],
                        "method": method,
                        "reason": "incomplete_existing",
                        "summary": str(summary_path),
                        "log": str(log_path) if log_path.exists() else None,
                        "missing_artifacts": missing_existing,
                    }
                    failed.append(failure)
                    job_records.append({**failure, "status": "failed"})
                    print(f"[fail] {model['name']} {method}: existing output is incomplete")
                    if args.stop_on_error:
                        break
                    continue
                skipped.append(str(summary_path))
                job_records.append(
                    {
                        "model": model["name"],
                        "method": method,
                        "status": "skipped_existing",
                        "summary": str(summary_path),
                        "log": str(log_path) if log_path.exists() else None,
                        "job_overrides": job_defaults,
                    }
                )
                print(f"[skip] {model['name']} {method}: {summary_path} exists")
                continue
            precompressed_path = _precompressed_path(config, model["name"], method)
            env = _build_env(job_defaults, model, method, output_root, precompressed_path)
            if args.force:
                env["OVERWRITE_COMPRESSED"] = "1"
            external_command = None
            if (
                method in {"smoothquant", "omniquant"}
                and not precompressed_path
                and not job_defaults.get("target_model_adaptation", False)
            ):
                command = _external_command(config, method)
                if not command:
                    item = {"model": model["name"], "method": method, "reason": "external_command_not_configured"}
                    skipped_external.append(item)
                    job_records.append({**item, "status": "skipped_external_not_configured", "external_command": None})
                    print(f"[skip] {model['name']} {method}: external command is not configured")
                    continue
                external_command = command
                env["EXTERNAL_COMMAND"] = command
            cmd = ["bash", "scripts/run_edge_pipeline.sh"]
            print(f"[run] {model['name']} {method} -> {output_root}")
            if args.dry_run:
                print(" ".join(cmd))
                job_records.append(
                    {
                        "model": model["name"],
                        "method": method,
                        "status": "dry_run",
                        "summary": str(summary_path),
                        "log": str(log_path),
                        "precompressed_path": precompressed_path or None,
                        "external_command": external_command,
                        "job_overrides": job_defaults,
                    }
                )
                continue
            started_at = time.time()
            returncode = _run_logged(cmd, env, log_path)
            elapsed = time.time() - started_at
            log_tail = _tail_text(log_path)
            if returncode == 0 and summary_path.exists():
                completed.append(str(summary_path))
                job_records.append(
                    {
                        "model": model["name"],
                        "method": method,
                        "status": "completed",
                        "summary": str(summary_path),
                        "log": str(log_path),
                        "log_tail": log_tail,
                        "precompressed_path": precompressed_path or None,
                        "external_command": external_command,
                        "job_overrides": job_defaults,
                        "elapsed_seconds": elapsed,
                    }
                )
            else:
                reason = "subprocess_failed" if returncode != 0 else "missing_summary"
                failure = {
                    "model": model["name"],
                    "method": method,
                    "returncode": returncode,
                    "reason": reason,
                    "summary": str(summary_path),
                    "log": str(log_path),
                    "log_tail": log_tail,
                    "precompressed_path": precompressed_path or None,
                    "external_command": external_command,
                    "job_overrides": job_defaults,
                    "elapsed_seconds": elapsed,
                }
                failed.append(failure)
                job_records.append({**failure, "status": "failed"})
                if args.stop_on_error:
                    break
        if failed and args.stop_on_error:
            break

    report = {
        "completed": completed,
        "skipped": skipped,
        "skipped_external_not_configured": skipped_external,
        "failed": failed,
        "jobs": job_records,
        "resolved_config": resolved_config,
    }
    report_path = Path(defaults["output_root"]).expanduser() / "matrix_run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)
    if args.finalize:
        try:
            _finalize_outputs(args, defaults)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full LLMCBench edge compression/evaluation matrix.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--only-model", action="append", default=[], help="Repeatable model name filter.")
    parser.add_argument("--only-method", action="append", default=[], help="Repeatable method filter.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--force", action="store_true", help="Re-run even if summary.json exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="After matrix execution, collect, validate and render final result artifacts.")
    parser.add_argument("--finalize-only", action="store_true", help="Skip matrix execution and only collect, validate and render existing outputs.")
    parser.add_argument("--allow-unexpected", action="store_true", help="Pass through to validate_edge_results.py during --finalize/--finalize-only.")
    parser.add_argument("--allow-skipped-external", action="store_true", help="Pass through to audit_edge_completion.py during --finalize/--finalize-only.")
    parser.add_argument("--require-runtime-python-prefix", default="", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-runtime-gpu-name-contains", default="", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-runtime-cuda-prefix", default="", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-runtime-gpu-total-memory-min-gb", type=float, default=0.0, help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-runtime-gpu-count", type=int, default=0, help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-runtime-cudnn", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-runtime-torch-prefix", default="", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-runtime-transformers", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-runtime-bitsandbytes", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-runtime-cpu-count-min", type=int, default=0, help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-runtime-system-memory-min-gb", type=float, default=0.0, help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-runtime-hf-endpoint", default="", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-code-match", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-clean-worktree", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-prewarm-report", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-prewarm-all-models", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-asset-audit-report", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-non-rotational-storage", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-preflight-report", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-handoff-manifest", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-target-readiness-report", action="store_true", help="Require and validate target_readiness.json during final completion audit.")
    parser.add_argument("--require-optional-backends-report", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-quant-runtime-report", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-target-paths-report", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-prepare-report", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-prepare-metadata-report", action="store_true", help="Pass through to audit_edge_completion.py.")
    parser.add_argument("--require-prepare-precompressed-report", action="store_true", help="Pass through to audit_edge_completion.py.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.finalize_only:
        _validate_paper_protocol_preflight(args)
        config = _load_config(args.config)
        defaults = config["defaults"]
        if args.output_root:
            defaults["output_root"] = args.output_root
        if args.cuda_visible_devices:
            defaults["cuda_visible_devices"] = args.cuda_visible_devices
        models = _selected(config["models"], args.only_model)
        methods = _selected_methods(config["methods"], args.only_method)
        if not models:
            raise SystemExit("No models selected.")
        if not methods:
            raise SystemExit("No methods selected.")
        _write_resolved_config(config, defaults, models, methods, args)
        try:
            _finalize_outputs(args, defaults)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1)
        return
    run_matrix(args)


if __name__ == "__main__":
    main()
