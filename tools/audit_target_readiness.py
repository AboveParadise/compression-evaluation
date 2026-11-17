import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tools.audit_edge_completion import (
        _asset_audit_report_status,
        _handoff_manifest_status,
        _optional_backends_report_status,
        _preflight_report_status,
        _prepare_precompressed_report_status,
        _prewarm_report_status,
        _quant_runtime_report_status,
        _split_tasks,
        _expand_prewarm_tasks,
    )
except ModuleNotFoundError:
    from audit_edge_completion import (
        _asset_audit_report_status,
        _handoff_manifest_status,
        _optional_backends_report_status,
        _preflight_report_status,
        _prepare_precompressed_report_status,
        _prewarm_report_status,
        _quant_runtime_report_status,
        _split_tasks,
        _expand_prewarm_tasks,
    )


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_parse_error": str(exc)}


def _artifact(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    artifact = {
        "path": str(path),
        "exists": path.exists(),
        "ok": payload.get("ok") if isinstance(payload, dict) else None,
        "parse_error": payload.get("_parse_error") if isinstance(payload, dict) else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    if isinstance(payload, dict):
        for key in ("blockers", "violations", "error"):
            if payload.get(key):
                artifact[key] = payload.get(key)
    return artifact


def _status_artifact(status: Dict[str, Any]) -> Dict[str, Any]:
    artifact = {
        "path": status.get("path"),
        "exists": status.get("exists"),
        "ok": status.get("ok"),
        "parse_error": status.get("message") if status.get("error") == "invalid JSON" else None,
        "size_bytes": Path(status["path"]).stat().st_size if status.get("path") and Path(status["path"]).exists() else None,
    }
    for key in (
        "blockers",
        "violations",
        "error",
        "report_ok",
        "dry_run",
        "metadata_only",
        "failed_items",
        "failed_rows",
        "failed_required_steps",
        "missing_steps",
        "verify_env_cuda_visible_devices",
        "missing_required_tasks",
        "missing_required_models",
        "missing_path_keys",
        "gptqmodel_install",
    ):
        if status.get(key):
            artifact[key] = status.get(key)
    return artifact


def _handoff_artifact(path: Path, current_code: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    status = _handoff_manifest_status(
        path,
        path.exists(),
        current_code=current_code,
        require_code_match=True,
        require_clean_worktree=True,
    )
    return {
        **base,
        "ok": status.get("ok") if path.exists() else None,
        "parse_error": status.get("message") if status.get("error") == "invalid JSON" else None,
        "schema_version": status.get("schema_version"),
        "code_commit": status.get("code_commit"),
        "code_branch": status.get("code_branch"),
        "code_origin": status.get("code_origin"),
        "code_dirty": status.get("code_dirty"),
        "code_matches_current": status.get("code_matches_current"),
        "violations": status.get("violations", []),
        "error": status.get("error"),
    }


def _asset_storage_artifact(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    base = {
        "path": str(path),
        "exists": path.exists(),
        "parse_error": payload.get("_parse_error") if isinstance(payload, dict) else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    if not path.exists() or not isinstance(payload, dict) or payload.get("_parse_error"):
        return {**base, "ok": None}
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
    return {**base, "ok": not missing and not violations, "missing_paths": missing, "media": media, "violations": violations}


def _run_json(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    payload = None
    marker = proc.stdout.rfind("\n{")
    text = proc.stdout[marker + 1 :] if marker >= 0 else proc.stdout
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "payload": payload,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _git_value(args: List[str]) -> str:
    proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git_dirty() -> Optional[bool]:
    proc = subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=False)
    return bool(proc.stdout.strip()) if proc.returncode == 0 else None


def _current_code_status() -> Dict[str, Any]:
    return {
        "commit": _git_value(["rev-parse", "HEAD"]),
        "branch": _git_value(["branch", "--show-current"]),
        "origin": _git_value(["remote", "get-url", "origin"]),
        "dirty": _git_dirty(),
    }


def _config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    precompressed = config.get("precompressed_paths") if isinstance(config.get("precompressed_paths"), dict) else {}
    precompressed_pairs = []
    for model_name, method_map in precompressed.items():
        if isinstance(method_map, dict):
            for method, path in method_map.items():
                if path:
                    precompressed_pairs.append({"model": model_name, "method": method, "path": path})
    return {
        "models": [model.get("name") for model in config.get("models", []) if isinstance(model, dict)],
        "methods": config.get("methods", []),
        "output_root": defaults.get("output_root"),
        "dtype": defaults.get("dtype"),
        "max_memory": defaults.get("max_memory"),
        "deployment_memory_limit_gb": defaults.get("deployment_memory_limit_gb"),
        "deployment_batch_size": defaults.get("deployment_batch_size"),
        "min_free_gpu_memory_gb": defaults.get("min_free_gpu_memory_gb"),
        "tasks": defaults.get("tasks"),
        "split_accuracy_tasks": defaults.get("split_accuracy_tasks"),
        "precompressed_pair_count": len(precompressed_pairs),
        "precompressed_pairs": precompressed_pairs,
    }


def _optional_final_gate_text(
    require_optional_backends_report: bool = False,
    require_quant_runtime_report: bool = False,
) -> str:
    optional_final_gates = []
    if require_optional_backends_report:
        optional_final_gates.append("--require-optional-backends-report")
    if require_quant_runtime_report:
        optional_final_gates.append("--require-quant-runtime-report")
    optional_final_gate_text = " ".join(optional_final_gates)
    if optional_final_gate_text:
        optional_final_gate_text = f" {optional_final_gate_text}"
    return optional_final_gate_text


def _preflight_optional_text(
    require_optional_backends_report: bool = False,
    require_quant_runtime_report: bool = False,
) -> str:
    preflight_optional_flags = []
    if require_optional_backends_report:
        preflight_optional_flags.append("--check-optional-backend-resolution")
    if require_quant_runtime_report:
        preflight_optional_flags.append("--check-quant-runtime")
    preflight_optional_text = " ".join(preflight_optional_flags)
    if preflight_optional_text:
        preflight_optional_text = f" {preflight_optional_text}"
    return preflight_optional_text


def _readiness_command(
    config_path: str,
    root: Path,
    require_optional_backends_report: bool = False,
    require_quant_runtime_report: bool = False,
    require_prepare_precompressed_report: bool = False,
) -> str:
    optional_final_gate_text = _optional_final_gate_text(
        require_optional_backends_report,
        require_quant_runtime_report,
    )
    prepare_gate = " --require-prepare-precompressed-report" if require_prepare_precompressed_report else ""
    return (
        f"python tools/audit_target_readiness.py --config {config_path} --root {root} "
        "--check-method-readiness --require-non-rotational-storage --require-prewarm "
        f"--require-handoff-manifest{optional_final_gate_text}{prepare_gate} "
        f"--output {root / 'target_readiness.json'}"
    )


def _baseline_config_path(config_path: str) -> str:
    if Path(config_path).name == "edge_matrix_with_precompressed.json":
        return "configs/edge_4090_matrix.json"
    return config_path


def _unique_commands(commands: List[str]) -> List[str]:
    unique = []
    seen = set()
    for command in commands:
        if command in seen:
            continue
        seen.add(command)
        unique.append(command)
    return unique


def _missing_commands(
    root: Path,
    config_path: str,
    require_optional_backends_report: bool = False,
    require_quant_runtime_report: bool = False,
    require_prepare_precompressed_report: bool = False,
) -> List[str]:
    precompressed_config = root / "edge_matrix_with_precompressed.json"
    baseline_config = _baseline_config_path(config_path)
    optional_final_gate_text = _optional_final_gate_text(
        require_optional_backends_report,
        require_quant_runtime_report,
    )
    preflight_optional_text = _preflight_optional_text(
        require_optional_backends_report,
        require_quant_runtime_report,
    )
    baseline_readiness = _readiness_command(
        baseline_config,
        root,
        require_optional_backends_report,
        require_quant_runtime_report,
    )
    precompressed_readiness = _readiness_command(
        str(precompressed_config),
        root,
        require_optional_backends_report,
        require_quant_runtime_report,
        True,
    )
    return _unique_commands([
        f"python tools/build_edge_handoff_manifest.py --config {config_path} --output-root {root} --output {root / 'edge_handoff_manifest.json'}",
        f"python tools/audit_edge_assets.py --config {config_path} --min-free-disk-gb 80 --output {root / 'edge_asset_audit.json'}",
        _prepare_precompressed_command(config_path, root),
        f"python tools/audit_edge_assets.py --config {precompressed_config} --min-free-disk-gb 80 --output {root / 'edge_asset_audit.json'}",
        f"python tools/run_edge_preflight.py --config {precompressed_config} --check-hf-access --check-model-forward --finalize-dry-run --cuda-visible-devices 0{preflight_optional_text} --output {root / 'edge_preflight_report.json'}",
        f"python tools/prewarm_lm_eval_tasks.py --config {precompressed_config} --only-model Qwen3-8B --expand-mmlu --sleep-seconds 60 --report-output {root / 'edge_prewarm_report.json'}",
        precompressed_readiness,
        f"python tools/run_edge_preflight.py --config {baseline_config} --check-hf-access --check-model-forward --finalize-dry-run --cuda-visible-devices 0{preflight_optional_text} --output {root / 'edge_preflight_report.json'}",
        f"python tools/prewarm_lm_eval_tasks.py --config {baseline_config} --only-model Qwen3-8B --expand-mmlu --sleep-seconds 60 --report-output {root / 'edge_prewarm_report.json'}",
        baseline_readiness,
        f"python tools/run_edge_matrix.py --config {precompressed_config} --stop-on-error --finalize --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report{optional_final_gate_text} --require-target-paths-report --require-prepare-precompressed-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree",
        f"python tools/audit_edge_completion.py --config {precompressed_config} --root {root} --require-finalize-report --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report{optional_final_gate_text} --require-target-paths-report --require-prepare-precompressed-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree",
        f"python tools/run_edge_matrix.py --config {baseline_config} --stop-on-error --finalize --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report{optional_final_gate_text} --require-target-paths-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree",
        f"python tools/audit_edge_completion.py --config {baseline_config} --root {root} --require-finalize-report --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report{optional_final_gate_text} --require-target-paths-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree",
    ])


def _prepare_precompressed_command(config_path: str, root: Path) -> str:
    return (
        "RUN_PRECOMPRESSED_PREPARE=1 PRECOMPRESSED_PREPARE_ONLY=1 "
        "RUN_TARGET_PATH_PREPARE=0 RUN_HANDOFF_MANIFEST=0 RUN_PREWARM=0 "
        "CHECK_HF_ACCESS=0 CHECK_MODEL_FORWARD=0 "
        f"CONFIG={config_path} PRECOMPRESSED_CONFIG_OUTPUT={root / 'edge_matrix_with_precompressed.json'} "
        f"PRECOMPRESSED_PREPARE_OUTPUT={root / 'edge_prepare_precompressed_report.json'} "
        "PRECOMPRESSED_GROUPS='LLaMA3-8B:gptq,awq,smoothquant;Qwen3-8B:gptq,awq,smoothquant,omniquant' "
        "bash scripts/run_edge_4090_full.sh"
    )


def _is_prepare_precompressed_command(command: str) -> bool:
    return "tools/prepare_precompressed_assets.py" in command or "RUN_PRECOMPRESSED_PREPARE=1" in command


def _next_command_groups(
    root: Path,
    config_path: str,
    require_optional_backends_report: bool = False,
    require_quant_runtime_report: bool = False,
    require_prepare_precompressed_report: bool = False,
) -> Dict[str, List[str]]:
    commands = _missing_commands(
        root,
        config_path,
        require_optional_backends_report,
        require_quant_runtime_report,
        require_prepare_precompressed_report,
    )
    precompressed_marker = "edge_matrix_with_precompressed.json"
    baseline_config = _baseline_config_path(config_path)
    baseline_post_readiness = _readiness_command(
        baseline_config,
        root,
        require_optional_backends_report,
        require_quant_runtime_report,
    )
    precompressed_post_readiness = _readiness_command(
        str(root / "edge_matrix_with_precompressed.json"),
        root,
        require_optional_backends_report,
        require_quant_runtime_report,
        True,
    )
    goal_audit = (
        "python tools/build_goal_audit.py "
        f"--final-output-root {root} "
        "--portable-bundle-verify portable_bundle_verify.json "
        f"--output {root / 'current_goal_audit.json'}"
    )
    groups = {
        "handoff_setup": [
            command for command in commands if "tools/build_edge_handoff_manifest.py" in command
        ],
        "baseline_preflight": [
            command
            for command in commands
            if precompressed_marker not in command
            and any(
                token in command
                for token in (
                    "tools/audit_edge_assets.py",
                    "tools/run_edge_preflight.py",
                    "tools/prewarm_lm_eval_tasks.py",
                )
            )
        ],
        "baseline_readiness": [
            command
            for command in commands
            if precompressed_marker not in command and "tools/audit_target_readiness.py" in command
        ],
        "precompressed_prepare": [
            command for command in commands if _is_prepare_precompressed_command(command)
        ],
        "precompressed_config_followup": [
            command
            for command in commands
            if precompressed_marker in command
            and not _is_prepare_precompressed_command(command)
        ],
        "baseline_finalize": [
            command
            for command in commands
            if precompressed_marker not in command
            and any(token in command for token in ("tools/run_edge_matrix.py", "tools/audit_edge_completion.py"))
        ],
        "precompressed_config_post_run_audit": [
            precompressed_post_readiness,
            goal_audit,
        ],
        "baseline_post_run_audit": [
            baseline_post_readiness,
            goal_audit,
        ],
    }
    return {name: _unique_commands(group) for name, group in groups.items()}


def _required_artifact_blockers(artifacts: Dict[str, Dict[str, Any]], required_names: List[str]) -> List[Dict[str, Any]]:
    blockers = []
    for name in required_names:
        artifact = artifacts[name]
        if not artifact["exists"]:
            blockers.append({"type": "missing_artifact", "artifact": name, "path": artifact["path"]})
        elif artifact["parse_error"]:
            blockers.append(
                {
                    "type": "invalid_json",
                    "artifact": name,
                    "path": artifact["path"],
                    "error": artifact["parse_error"],
                }
            )
        elif artifact["ok"] is not True:
            blocker = {
                "type": "artifact_not_ok",
                "artifact": name,
                "path": artifact["path"],
                "ok": artifact["ok"],
            }
            for key in ("blockers", "violations", "error", "parse_error", "verify_env_cuda_visible_devices"):
                if artifact.get(key):
                    blocker[key] = artifact.get(key)
            blockers.append(blocker)
    return blockers


def _method_readiness_blockers(readiness: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if readiness is None:
        return []
    if not readiness.get("ok"):
        return [
            {
                "type": "method_readiness_command_failed",
                "cmd": readiness.get("cmd"),
                "returncode": readiness.get("returncode"),
                "stderr_tail": readiness.get("stderr_tail"),
            }
        ]
    payload = readiness.get("payload")
    if not isinstance(payload, dict):
        return [{"type": "method_readiness_missing_payload", "cmd": readiness.get("cmd")}]
    if payload.get("ok") is True:
        return []
    blockers = []
    for row in payload.get("rows", []):
        if isinstance(row, dict) and row.get("blockers"):
            blockers.append(
                {
                    "type": "method_blocked",
                    "model": row.get("model"),
                    "method": row.get("method"),
                    "uses_precompressed": row.get("uses_precompressed"),
                    "precompressed_path": row.get("precompressed_path"),
                    "runtime_requirements_reason": row.get("runtime_requirements_reason"),
                    "quantization_metadata": row.get("quantization_metadata"),
                    "blockers": row.get("blockers"),
                }
            )
    return blockers or [{"type": "method_readiness_not_ok", "cmd": readiness.get("cmd")}]


def _attach_optional_runtime_details(blockers: List[Dict[str, Any]], optional_backends: Dict[str, Any]) -> List[Dict[str, Any]]:
    gptqmodel_install = optional_backends.get("gptqmodel_install") if isinstance(optional_backends.get("gptqmodel_install"), dict) else {}
    requirement_status = (
        gptqmodel_install.get("requirement_status")
        if isinstance(gptqmodel_install.get("requirement_status"), dict)
        else {}
    )
    requirements_ok = gptqmodel_install.get("requirements_ok")
    for blocker in blockers:
        if (
            blocker.get("type") == "method_blocked"
            and blocker.get("method") == "gptq"
            and blocker.get("runtime_requirements_reason") == "gptqmodel_checkpoint_metadata"
            and requirements_ok is False
        ):
            blocker["gptqmodel_requirement_status"] = requirement_status
            blocker_values = blocker.get("blockers") if isinstance(blocker.get("blockers"), list) else []
            blocker["blockers"] = list(dict.fromkeys([*blocker_values, "gptqmodel_requirements_not_satisfied"]))
    return blockers


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config).expanduser()
    config = _load_json(config_path)
    if not isinstance(config, dict) or config.get("_parse_error"):
        raise SystemExit(f"Invalid config: {config_path}")
    root = Path(args.root or config.get("defaults", {}).get("output_root", "")).expanduser()
    current_code = _current_code_status()
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    required_prewarm_tasks = _expand_prewarm_tasks(_split_tasks(defaults.get("tasks", "")))
    artifacts = {
        "asset_audit": _status_artifact(_asset_audit_report_status(root / "edge_asset_audit.json", True)),
        "asset_storage": _asset_storage_artifact(root / "edge_asset_audit.json"),
        "preflight": _status_artifact(
            _preflight_report_status(
                root / "edge_preflight_report.json",
                True,
                args.require_optional_backends_report,
                args.require_quant_runtime_report,
            )
        ),
        "prewarm": _status_artifact(
            _prewarm_report_status(
                root / "edge_prewarm_report.json",
                True,
                required_prewarm_tasks,
                [],
                defaults.get("dtype", "") or "",
                defaults.get("attn_implementation", "") or "",
            )
        ),
        "handoff_manifest": _handoff_artifact(root / "edge_handoff_manifest.json", current_code),
        "prepare_assets": _artifact(root / "edge_prepare_assets_report.json"),
        "prepare_metadata": _artifact(root / "edge_prepare_metadata_report.json"),
        "prepare_precompressed": _status_artifact(
            _prepare_precompressed_report_status(root / "edge_prepare_precompressed_report.json", True, str(config_path))
        ),
        "optional_backends": _status_artifact(
            _optional_backends_report_status(root / "edge_optional_backends_report.json", True)
        ),
        "quant_runtime": _status_artifact(
            _quant_runtime_report_status(root / "edge_quant_runtime_report.json", True)
        ),
        "matrix_results": _artifact(root / "matrix_results.json"),
        "matrix_results_csv": _artifact(root / "matrix_results.csv"),
        "edge_report": _artifact(root / "edge_report.md"),
        "completion_audit": _artifact(root / "completion_audit.json"),
        "matrix_config_resolved": _artifact(root / "matrix_config_resolved.json"),
    }
    readiness = None
    if args.check_method_readiness:
        readiness = _run_json([sys.executable, "tools/check_method_readiness.py", "--config", str(config_path)])
    required_before_matrix = ["asset_audit", "preflight"]
    if args.require_non_rotational_storage:
        required_before_matrix.append("asset_storage")
    if args.require_prewarm:
        required_before_matrix.append("prewarm")
    if args.require_handoff_manifest:
        required_before_matrix.append("handoff_manifest")
    if args.require_optional_backends_report:
        required_before_matrix.append("optional_backends")
    if args.require_quant_runtime_report:
        required_before_matrix.append("quant_runtime")
    if args.require_prepare_precompressed_report:
        required_before_matrix.append("prepare_precompressed")
    missing_before_matrix = [
        name
        for name in required_before_matrix
        if not artifacts[name]["exists"] or artifacts[name].get("parse_error") or artifacts[name].get("ok") is False
    ]
    before_matrix_blockers = _required_artifact_blockers(artifacts, required_before_matrix)
    if args.check_method_readiness:
        before_matrix_blockers.extend(_method_readiness_blockers(readiness))
    before_matrix_blockers = _attach_optional_runtime_details(before_matrix_blockers, artifacts["optional_backends"])
    ready_for_matrix = not before_matrix_blockers
    completion_ok = artifacts["completion_audit"]["ok"] is True
    report = {
        "ok": completion_ok,
        "config": str(config_path),
        "root": str(root),
        "current_code": current_code,
        "config_summary": _config_summary(config),
        "artifacts": artifacts,
        "method_readiness": readiness,
        "required_before_matrix": required_before_matrix,
        "missing_before_matrix": missing_before_matrix,
        "before_matrix_blockers": before_matrix_blockers,
        "ready_for_matrix": ready_for_matrix,
        "completion_ok": completion_ok,
        "next_commands": _missing_commands(
            root,
            str(config_path),
            args.require_optional_backends_report,
            args.require_quant_runtime_report,
            args.require_prepare_precompressed_report,
        ),
        "next_command_groups": _next_command_groups(
            root,
            str(config_path),
            args.require_optional_backends_report,
            args.require_quant_runtime_report,
            args.require_prepare_precompressed_report,
        ),
        "notes": [
            "This readiness audit does not run compression, lm-eval, deployment, or resource jobs.",
            "ready_for_matrix means required preparation, preflight and method readiness gates are satisfied; ok/completion_ok means the final matrix has already produced a passing completion_audit.json.",
            "Final completion still requires completion_audit.json with ok=true from the target Python 3.10 / RTX 4090 environment.",
        ],
    }
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"] and args.fail_on_incomplete:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize target-machine readiness for the LLMCBench edge matrix.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--check-method-readiness", action="store_true")
    parser.add_argument("--require-non-rotational-storage", action="store_true")
    parser.add_argument("--require-prewarm", action="store_true")
    parser.add_argument("--require-handoff-manifest", action="store_true")
    parser.add_argument("--require-optional-backends-report", action="store_true")
    parser.add_argument("--require-quant-runtime-report", action="store_true")
    parser.add_argument("--require-prepare-precompressed-report", action="store_true")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main() -> None:
    audit(build_parser().parse_args())


if __name__ == "__main__":
    main()
