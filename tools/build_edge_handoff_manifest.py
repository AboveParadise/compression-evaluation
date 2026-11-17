import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

try:
    from tools.prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS
except ModuleNotFoundError:
    from prewarm_lm_eval_tasks import DEFAULT_MMLU_SUBTASKS


TARGET_RUNTIME_GATES = {
    "python_prefix": "3.10",
    "gpu_name_contains": "4090",
    "cuda_prefix": "12.",
    "gpu_total_memory_min_gb": 23,
    "gpu_count": 1,
    "require_cudnn": True,
    "torch_prefix": "2.",
    "require_transformers": True,
    "require_bitsandbytes": True,
    "cpu_count_min": 8,
    "system_memory_min_gb": 240,
    "hf_endpoint": "https://hf-mirror.com",
    "require_code_match": True,
    "require_clean_worktree": True,
}


def _load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_value(args: List[str]) -> str:
    proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git_dirty() -> bool:
    proc = subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=False)
    return bool(proc.stdout.strip()) if proc.returncode == 0 else True


def _artifact(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def _output_root(config: Dict[str, Any], override: str) -> Path:
    if override:
        return Path(override).expanduser()
    defaults = config.get("defaults", {}) if isinstance(config.get("defaults"), dict) else {}
    return Path(defaults.get("output_root", "outputs/llmcbench_edge_4090")).expanduser()


def _model_paths(config: Dict[str, Any]) -> List[Dict[str, str]]:
    models = config.get("models") if isinstance(config.get("models"), list) else []
    return [
        {"name": str(model.get("name", "")), "path": str(model.get("path", ""))}
        for model in models
        if isinstance(model, dict)
    ]


def _precompressed_paths(config: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = []
    paths = config.get("precompressed_paths") if isinstance(config.get("precompressed_paths"), dict) else {}
    for model_name, methods in sorted(paths.items()):
        if not isinstance(methods, dict):
            continue
        for method, path in sorted(methods.items()):
            if path:
                rows.append({"model": str(model_name), "method": str(method), "path": str(path)})
    return rows


def _external_command_status(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    commands = config.get("external_commands") if isinstance(config.get("external_commands"), dict) else {}
    return {
        method: {
            "configured": bool(commands.get(method)),
            "config_value": commands.get(method, ""),
            "env_override": f"LLMCBENCH_{method.upper()}_CMD",
        }
        for method in ("smoothquant", "omniquant")
    }


def _required_artifacts(root: Path) -> Dict[str, Dict[str, Any]]:
    names = [
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
    return {name: _artifact(root / name) for name in names}


def _optional_artifacts(root: Path) -> Dict[str, Dict[str, Any]]:
    names = [
        "edge_prepare_metadata_report.json",
        "edge_prepare_assets_report.json",
        "edge_prepare_precompressed_report.json",
    ]
    return {name: _artifact(root / name) for name in names}


def _baseline_config_path(config_path: Path) -> str:
    config_text = str(config_path)
    if config_path.name == "edge_matrix_with_precompressed.json":
        return "configs/edge_4090_matrix.json"
    return config_text


def _commands(config_path: Path, root: Path) -> Dict[str, List[str]]:
    config = str(config_path)
    baseline_config = _baseline_config_path(config_path)
    output_root = str(root)
    precompressed_config = f"{output_root}/edge_matrix_with_precompressed.json"
    goal_audit_cmd = (
        "python tools/build_goal_audit.py "
        f"--final-output-root {output_root} "
        "--portable-bundle-verify portable_bundle_verify.json "
        f"--output {output_root}/current_goal_audit.json"
    )
    return {
        "setup": [
            "git clone git@code.byted.org:data/llmcbench.git",
            "cd llmcbench && git checkout master",
            "sudo DATA_ROOT=/data bash scripts/init_edge_storage.sh",
            "bash scripts/setup_edge_env.sh",
            "# For Qwen GPTQModel checkpoints, use: EDGE_RUNTIME_PROFILE=gptqmodel bash scripts/setup_edge_env.sh",
            "export HF_ENDPOINT=https://hf-mirror.com",
            "OPTIONAL_BACKEND_DRY_RUN=1 REPORT_OUTPUT="
            f"{output_root}/edge_optional_backends_report.json bash scripts/setup_optional_backends.sh",
            "bash scripts/setup_optional_backends.sh",
            "# Strict paper/official-code matrix only: pin official external source checkouts.",
            "python tools/setup_official_compression_sources.py "
            "--root /opt/llmcbench_official "
            f"--output {output_root}/official_sources.json",
            "export LLMCBENCH_SMOOTHQUANT_DIR=/opt/llmcbench_official/smoothquant",
            "export LLMCBENCH_OMNIQUANT_DIR=/opt/llmcbench_official/omniquant",
            f"python tools/attest_base_models.py --config {config} "
            f"--output {output_root}/base_model_attestation.json",
        ],
        "prepare": [
            "python tools/prepare_target_paths.py "
            f"--config {config} --output {output_root}/edge_target_paths_report.json --fail-on-error",
            "python tools/prepare_edge_assets.py "
            f"--config {config} --metadata-only --skip-lm-eval-probe --continue-on-error "
            f"--output {output_root}/edge_prepare_metadata_report.json",
            "python tools/prepare_edge_assets.py "
            f"--config {config} --skip-lm-eval-probe --continue-on-error --min-free-disk-gb 80 "
            f"--output {output_root}/edge_prepare_assets_report.json",
            f"python tools/audit_edge_assets.py --config {config} --min-free-disk-gb 80 "
            f"--output {output_root}/edge_asset_audit.json",
        ],
        "prepare_precompressed": [
            "RUN_PRECOMPRESSED_PREPARE=1 PRECOMPRESSED_PREPARE_ONLY=1 "
            "RUN_TARGET_PATH_PREPARE=0 RUN_HANDOFF_MANIFEST=0 RUN_PREWARM=0 "
            "CHECK_HF_ACCESS=0 CHECK_MODEL_FORWARD=0 "
            f"CONFIG={config} PRECOMPRESSED_CONFIG_OUTPUT={output_root}/edge_matrix_with_precompressed.json "
            f"PRECOMPRESSED_PREPARE_OUTPUT={output_root}/edge_prepare_precompressed_report.json "
            "PRECOMPRESSED_GROUPS='LLaMA3-8B:gptq,awq,smoothquant;Qwen3-8B:gptq,awq,smoothquant,omniquant' "
            "bash scripts/run_edge_4090_full.sh",
        ],
        "preflight": [
            "python tools/run_edge_preflight.py "
            f"--config {config} --check-hf-access --check-model-forward --finalize-dry-run "
            "--cuda-visible-devices 0 "
            "--check-optional-backend-resolution --check-quant-runtime "
            f"--output {output_root}/edge_preflight_report.json",
        ],
        "prewarm": [
            "python tools/prewarm_lm_eval_tasks.py "
            f"--config {config} --only-model Qwen3-8B --expand-mmlu --sleep-seconds 60 "
            f"--report-output {output_root}/edge_prewarm_report.json",
        ],
        "readiness": [
            "python tools/audit_target_readiness.py "
            f"--config {config} --root {output_root} --check-method-readiness "
            "--require-non-rotational-storage --require-prewarm --require-handoff-manifest "
            "--require-optional-backends-report --require-quant-runtime-report "
            f"--output {output_root}/target_readiness.json",
        ],
        "precompressed_config_followup": [
            "python tools/audit_edge_assets.py "
            f"--config {precompressed_config} --min-free-disk-gb 80 "
            f"--output {output_root}/edge_asset_audit.json",
            "python tools/run_edge_preflight.py "
            f"--config {precompressed_config} --check-hf-access --check-model-forward --finalize-dry-run "
            "--cuda-visible-devices 0 "
            "--check-optional-backend-resolution --check-quant-runtime "
            f"--output {output_root}/edge_preflight_report.json",
            "python tools/prewarm_lm_eval_tasks.py "
            f"--config {precompressed_config} --only-model Qwen3-8B --expand-mmlu --sleep-seconds 60 "
            f"--report-output {output_root}/edge_prewarm_report.json",
            "python tools/audit_target_readiness.py "
            f"--config {precompressed_config} --root {output_root} --check-method-readiness "
            "--require-non-rotational-storage --require-prewarm --require-handoff-manifest "
            "--require-optional-backends-report --require-quant-runtime-report "
            "--require-prepare-precompressed-report "
            f"--output {output_root}/target_readiness.json",
            f"python tools/run_edge_matrix.py --config {precompressed_config} --stop-on-error --finalize "
            "--require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report "
            "--require-handoff-manifest --require-target-readiness-report "
            "--require-optional-backends-report --require-quant-runtime-report "
            "--require-target-paths-report "
            "--require-prepare-precompressed-report "
            "--require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 "
            "--require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 "
            "--require-runtime-gpu-count 1 "
            "--require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers "
            "--require-runtime-bitsandbytes "
            "--require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 "
            "--require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree",
            f"python tools/audit_edge_completion.py --config {precompressed_config} --root {output_root} "
            "--require-finalize-report --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage "
            "--require-preflight-report --require-handoff-manifest --require-target-readiness-report "
            "--require-optional-backends-report --require-quant-runtime-report "
            "--require-target-paths-report "
            "--require-prepare-precompressed-report "
            "--require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 "
            "--require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 "
            "--require-runtime-gpu-count 1 "
            "--require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers "
            "--require-runtime-bitsandbytes "
            "--require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 "
            "--require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree",
        ],
        "run": [
            "DRY_RUN=1 CHECK_MODEL_FORWARD=0 RUN_PREWARM=0 bash scripts/run_edge_4090_full.sh",
            "CHECK_OPTIONAL_BACKEND_RESOLUTION=1 CHECK_QUANT_RUNTIME=1 bash scripts/run_edge_4090_full.sh",
        ],
        "manual_finalize": [
            f"python tools/run_edge_matrix.py --config {config} --stop-on-error --finalize "
            "--require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report "
            "--require-handoff-manifest --require-target-readiness-report "
            "--require-optional-backends-report --require-quant-runtime-report "
            "--require-target-paths-report "
            "--require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 "
            "--require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 "
            "--require-runtime-gpu-count 1 "
            "--require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers "
            "--require-runtime-bitsandbytes "
            "--require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 "
            "--require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree",
            f"python tools/audit_edge_completion.py --config {config} --root {output_root} "
            "--require-finalize-report --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage "
            "--require-preflight-report --require-handoff-manifest --require-target-readiness-report "
            "--require-optional-backends-report --require-quant-runtime-report "
            "--require-target-paths-report "
            "--require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 "
            "--require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 "
            "--require-runtime-gpu-count 1 "
            "--require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers "
            "--require-runtime-bitsandbytes "
            "--require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 "
            "--require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree",
        ],
        "resume": [
            f"python tools/plan_edge_matrix_resume.py --config {config} --output-root {output_root} "
            f"--output {output_root}/matrix_resume_plan.json",
        ],
        "scorecard": [
            f"python tools/summarize_edge_results.py --results {output_root}/matrix_results.json "
            f"--output-json {output_root}/edge_scorecard.json --output-md {output_root}/edge_scorecard.md",
        ],
        "post_run_audit": [
            f"python tools/audit_target_readiness.py --config {config} --root {output_root} "
            "--check-method-readiness --require-non-rotational-storage --require-prewarm "
            "--require-handoff-manifest --require-optional-backends-report "
            f"--require-quant-runtime-report --output {output_root}/target_readiness.json",
            goal_audit_cmd,
        ],
        "export_results": [
            "python tools/export_edge_results_bundle.py "
            f"--output-root {output_root} --bundle-dir {output_root}/edge_results_bundle "
            f"--archive {output_root}/edge_results_bundle.tar.gz --include-job-logs --require-complete --force",
        ],
        "verify_results_bundle": [
            "python "
            f"{output_root}/edge_results_bundle/tools/verify_edge_results_bundle.py "
            f"--bundle-root {output_root}/edge_results_bundle "
            "--require-complete --require-target-readiness-ok --require-target-runtime "
            f"--output {output_root}/edge_results_bundle_verify.json",
        ],
        "verify_delivery": [
            "python tools/verify_edge_delivery.py "
            f"--root {output_root} --results-bundle {output_root}/edge_results_bundle "
            f"--results-archive {output_root}/edge_results_bundle.tar.gz "
            f"--output {output_root}/edge_delivery_verify.json",
        ],
        "baseline_post_run_audit": [
            f"python tools/audit_target_readiness.py --config {baseline_config} --root {output_root} "
            "--check-method-readiness --require-non-rotational-storage --require-prewarm "
            "--require-handoff-manifest --require-optional-backends-report "
            f"--require-quant-runtime-report --output {output_root}/target_readiness.json",
            goal_audit_cmd,
        ],
        "precompressed_config_post_run_audit": [
            f"python tools/audit_target_readiness.py --config {precompressed_config} --root {output_root} "
            "--check-method-readiness --require-non-rotational-storage --require-prewarm "
            "--require-handoff-manifest --require-optional-backends-report "
            f"--require-quant-runtime-report --require-prepare-precompressed-report --output {output_root}/target_readiness.json",
            goal_audit_cmd,
        ],
    }


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config).expanduser()
    config = _load_config(config_path)
    defaults = config.get("defaults", {}) if isinstance(config.get("defaults"), dict) else {}
    root = _output_root(config, args.output_root)
    methods = config.get("methods") if isinstance(config.get("methods"), list) else []
    manifest = {
        "schema_version": 1,
        "purpose": "Portable handoff manifest for the LLMCBench edge 4090 matrix.",
        "code": {
            "branch": _git_value(["branch", "--show-current"]),
            "commit": _git_value(["rev-parse", "HEAD"]),
            "origin": _git_value(["remote", "get-url", "origin"]),
            "dirty": _git_dirty(),
        },
        "config": {
            "path": str(config_path),
            "models": _model_paths(config),
            "methods": methods,
            "defaults": defaults,
            "external_commands": _external_command_status(config),
            "precompressed_paths": _precompressed_paths(config),
        },
        "target_environment": {
            "hardware": {
                "gpu": "single NVIDIA RTX 4090 24GB",
                "cpu_count_min": TARGET_RUNTIME_GATES["cpu_count_min"],
                "system_memory_min_gb": TARGET_RUNTIME_GATES["system_memory_min_gb"],
                "storage": "NVMe or local SSD recommended for checkpoints, datasets and outputs",
            },
            "software": {
                "python_prefix": TARGET_RUNTIME_GATES["python_prefix"],
                "pytorch_prefix": TARGET_RUNTIME_GATES["torch_prefix"],
                "cuda_prefix": TARGET_RUNTIME_GATES["cuda_prefix"],
                "gpu_count": TARGET_RUNTIME_GATES["gpu_count"],
                "require_cudnn": TARGET_RUNTIME_GATES["require_cudnn"],
                "require_transformers": TARGET_RUNTIME_GATES["require_transformers"],
                "require_bitsandbytes": TARGET_RUNTIME_GATES["require_bitsandbytes"],
                "hf_endpoint": TARGET_RUNTIME_GATES["hf_endpoint"],
            },
            "memory_controls": {
                "dtype": defaults.get("dtype"),
                "max_memory": defaults.get("max_memory"),
                "deployment_memory_limit_gb": defaults.get("deployment_memory_limit_gb"),
                "deployment_batch_size": defaults.get("deployment_batch_size"),
                "min_free_gpu_memory_gb": defaults.get("min_free_gpu_memory_gb"),
                "calibration_batch_size": defaults.get("calibration_batch_size"),
            },
        },
        "paths": {
            "output_root": str(root),
            "hf_cache_dir": defaults.get("hf_cache_dir"),
            "data_dir": defaults.get("data_dir"),
            "calibration_file": defaults.get("calibration_file"),
            "prompt_file": defaults.get("prompt_file"),
        },
        "commands": _commands(config_path, root),
        "required_artifacts": _required_artifacts(root),
        "optional_artifacts": _optional_artifacts(root),
        "completion_gates": {
            "runtime": TARGET_RUNTIME_GATES,
            "coverage": {
                "models": [model["name"] for model in _model_paths(config)],
                "methods": methods,
                "tasks": defaults.get("tasks"),
                "required_mmlu_subtask_count": len(DEFAULT_MMLU_SUBTASKS),
                "required_mmlu_subtasks": list(DEFAULT_MMLU_SUBTASKS),
                "split_accuracy_tasks": defaults.get("split_accuracy_tasks"),
                "continue_on_accuracy_error": defaults.get("continue_on_accuracy_error"),
            },
            "reports": [
                "edge_asset_audit.json",
                "edge_preflight_report.json",
                "edge_prewarm_report.json",
                "edge_target_paths_report.json with ok=true",
                "edge_optional_backends_report.json with ok=true",
                "edge_quant_runtime_report.json with ok=true",
                "target_readiness.json with ok=true, completion_ok=true and no before_matrix_blockers",
                "completion_audit.json with ok=true and validator.ok=true",
                "edge_results_bundle_verify.json with ok=true, complete=true, validator.ok=true and target completion_ok=true",
                "edge_delivery_verify.json with ok=true and all substatus checks passing",
            ],
        },
        "known_blockers_to_clear": [
            "Provide an authorized LLaMA3-8B checkpoint at the configured path.",
            "Verify Python 3.10, PyTorch 2.x, CUDA 12.x, cuDNN and Transformers on the target 4090 host.",
            "Install or validate GPTQ/AWQ/SmoothQuant quantized loading runtimes, or provide complete precompressed checkpoints.",
            "Configure SmoothQuant and OmniQuant external commands, or provide complete precompressed checkpoints.",
            "Run the full two-model x all-method x all-task matrix and produce completion_audit.json with ok=true and validator.ok=true.",
            "Export and verify edge_results_bundle.tar.gz, then produce edge_delivery_verify.json with ok=true.",
        ],
    }
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a portable handoff manifest for the LLMCBench edge 4090 workflow.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--output-root", default="", help="Override the output root used for artifact paths in the manifest.")
    parser.add_argument("--output", default="", help="Write the manifest JSON to this path. Defaults to stdout only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_manifest(args)
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
