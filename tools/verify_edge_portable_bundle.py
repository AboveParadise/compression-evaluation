import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_TOP_LEVEL_PATHS = [
    "README.md",
    "RUN_ON_TARGET.md",
    "PORTABLE_BUNDLE_MANIFEST.json",
    "edge_handoff_manifest.json",
    "requirements.txt",
    "requirements-optional.txt",
    "configs/edge_4090_matrix.json",
    "configs/edge_4090_structured_pruning.json",
    "configs/edge_paper_aligned_matrix.json",
    "configs/paper_protocols.json",
    "edge_benchmark/cli.py",
    "scripts/init_edge_storage.sh",
    "scripts/setup_edge_env.sh",
    "scripts/setup_optional_backends.sh",
    "scripts/run_edge_4090_full.sh",
    "tools/export_edge_portable_bundle.py",
    "tools/verify_edge_portable_bundle.py",
    "tools/build_edge_handoff_manifest.py",
    "tools/build_goal_audit.py",
    "tools/plan_edge_matrix_resume.py",
    "tools/summarize_edge_results.py",
    "tools/collect_edge_results.py",
    "tools/render_edge_report.py",
    "tools/export_edge_results_bundle.py",
    "tools/verify_edge_results_bundle.py",
    "tools/verify_edge_delivery.py",
    "tools/run_edge_matrix.py",
    "tools/audit_edge_completion.py",
    "tools/audit_edge_assets.py",
    "tools/audit_target_readiness.py",
    "tools/run_edge_preflight.py",
    "tools/verify_edge_env.py",
    "tools/prepare_target_paths.py",
    "tools/prewarm_lm_eval_tasks.py",
    "tools/prepare_edge_assets.py",
    "tools/prepare_precompressed_assets.py",
    "tools/configure_external_methods.py",
    "tools/setup_official_compression_sources.py",
    "tools/attest_base_models.py",
    "tools/check_method_readiness.py",
    "tools/probe_quant_runtime.py",
    "tools/validate_edge_results.py",
    "tools/validate_paper_protocol.py",
]

REQUIRED_HANDOFF_COMPLETION_GATES = [
    "edge_target_paths_report.json with ok=true",
    "completion_audit.json with ok=true and validator.ok=true",
    "edge_results_bundle_verify.json with ok=true, complete=true, validator.ok=true and target completion_ok=true",
    "edge_delivery_verify.json with ok=true and all substatus checks passing",
]

REQUIRED_HANDOFF_ARTIFACTS = [
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

REQUIRED_HANDOFF_COMMANDS = {
    "setup": [
        "scripts/init_edge_storage.sh",
        "DATA_ROOT=/data",
        "scripts/setup_edge_env.sh",
        "HF_ENDPOINT=https://hf-mirror.com",
        "scripts/setup_optional_backends.sh",
        "OPTIONAL_BACKEND_DRY_RUN=1",
        "edge_optional_backends_report.json",
        "tools/setup_official_compression_sources.py",
        "tools/attest_base_models.py",
    ],
    "prepare": [
        "tools/prepare_target_paths.py",
        "edge_target_paths_report.json",
        "--fail-on-error",
        "tools/prepare_edge_assets.py",
        "tools/audit_edge_assets.py",
        "--metadata-only",
        "--min-free-disk-gb",
        "edge_prepare_assets_report.json",
        "edge_asset_audit.json",
    ],
    "prepare_precompressed": [
        "RUN_PRECOMPRESSED_PREPARE=1",
        "PRECOMPRESSED_PREPARE_ONLY=1",
        "PRECOMPRESSED_GROUPS=",
        "scripts/run_edge_4090_full.sh",
        "edge_matrix_with_precompressed.json",
        "edge_prepare_precompressed_report.json",
    ],
    "preflight": [
        "tools/run_edge_preflight.py",
        "--cuda-visible-devices 0",
        "--check-hf-access",
        "--check-model-forward",
        "--check-optional-backend-resolution",
        "--check-quant-runtime",
        "edge_preflight_report.json",
    ],
    "prewarm": [
        "tools/prewarm_lm_eval_tasks.py",
        "--expand-mmlu",
        "edge_prewarm_report.json",
    ],
    "readiness": [
        "tools/audit_target_readiness.py",
        "--require-non-rotational-storage",
        "--require-prewarm",
        "--require-handoff-manifest",
        "--require-optional-backends-report",
        "--require-quant-runtime-report",
        "target_readiness.json",
    ],
    "precompressed_config_followup": [
        "edge_matrix_with_precompressed.json",
        "tools/run_edge_preflight.py",
        "tools/run_edge_matrix.py",
        "tools/audit_edge_completion.py",
        "--require-prepare-precompressed-report",
        "--require-target-readiness-report",
    ],
    "manual_finalize": [
        "tools/run_edge_matrix.py",
        "tools/audit_edge_completion.py",
        "--require-target-readiness-report",
        "--require-runtime-gpu-name-contains 4090",
        "--require-runtime-cuda-prefix 12.",
        "--require-code-match",
        "--require-clean-worktree",
    ],
    "run": [
        "scripts/run_edge_4090_full.sh",
        "DRY_RUN=1",
        "CHECK_OPTIONAL_BACKEND_RESOLUTION=1",
        "CHECK_QUANT_RUNTIME=1",
    ],
    "scorecard": [
        "tools/summarize_edge_results.py",
        "edge_scorecard.json",
        "edge_scorecard.md",
    ],
    "export_results": [
        "tools/export_edge_results_bundle.py",
        "--require-complete",
        "edge_results_bundle.tar.gz",
    ],
    "verify_results_bundle": [
        "tools/verify_edge_results_bundle.py",
        "--require-complete",
        "--require-target-readiness-ok",
        "--require-target-runtime",
        "edge_results_bundle_verify.json",
    ],
    "verify_delivery": [
        "tools/verify_edge_delivery.py",
        "--results-archive",
        "edge_delivery_verify.json",
    ],
    "post_run_audit": [
        "tools/audit_target_readiness.py",
        "tools/build_goal_audit.py",
        "--final-output-root",
        "--portable-bundle-verify",
        "portable_bundle_verify.json",
        "--output",
        "current_goal_audit.json",
        "--require-non-rotational-storage",
        "--require-prewarm",
        "--require-handoff-manifest",
        "--require-optional-backends-report",
        "--require-quant-runtime-report",
        "target_readiness.json",
    ],
    "baseline_post_run_audit": [
        "tools/audit_target_readiness.py",
        "tools/build_goal_audit.py",
        "--final-output-root",
        "--portable-bundle-verify",
        "portable_bundle_verify.json",
        "--output",
        "current_goal_audit.json",
        "--require-non-rotational-storage",
        "--require-prewarm",
        "--require-handoff-manifest",
        "--require-optional-backends-report",
        "--require-quant-runtime-report",
        "target_readiness.json",
    ],
    "precompressed_config_post_run_audit": [
        "tools/audit_target_readiness.py",
        "tools/build_goal_audit.py",
        "--final-output-root",
        "--portable-bundle-verify",
        "portable_bundle_verify.json",
        "--output",
        "current_goal_audit.json",
        "edge_matrix_with_precompressed.json",
        "--require-prepare-precompressed-report",
        "target_readiness.json",
    ],
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
    status["exists"] = path.is_file()
    if not path.is_file():
        status["violations"].append("file is missing")
        return status
    expected_size = item.get("size_bytes")
    actual_size = path.stat().st_size
    status["actual_size_bytes"] = actual_size
    status["expected_size_bytes"] = expected_size
    if expected_size == actual_size:
        status["size_ok"] = True
    else:
        status["violations"].append("size mismatch")
    expected_sha = item.get("sha256")
    actual_sha = _sha256(path)
    status["actual_sha256"] = actual_sha
    status["expected_sha256"] = expected_sha
    if expected_sha == actual_sha:
        status["sha256_ok"] = True
    else:
        status["violations"].append("sha256 mismatch")
    status["ok"] = status["exists"] and status["size_ok"] and status["sha256_ok"]
    return status


def _archive_status(
    bundle_root: Path,
    archive_arg: str,
    manifest_files: List[Dict[str, Any]],
    manifest_path: Path,
) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "ok": True,
        "archive": archive_arg or None,
        "checked": False,
        "violations": [],
        "expected_top_level_dir": bundle_root.name,
        "archive_top_level_dirs": [],
        "missing_archive_files": [],
        "extra_archive_files": [],
        "duplicate_archive_files": [],
        "unsafe_archive_members": [],
        "unsupported_archive_members": [],
        "size_mismatches": [],
        "sha256_mismatches": [],
        "verified_archive_file_count": 0,
    }
    if not archive_arg:
        return status
    archive_path = Path(archive_arg).expanduser().resolve()
    status["archive"] = str(archive_path)
    status["checked"] = True
    if not archive_path.is_file():
        status["ok"] = False
        status["violations"].append("archive is missing")
        return status
    expected_items = {
        item.get("path"): item
        for item in manifest_files
        if isinstance(item, dict) and _safe_relative_path(item.get("path"))
    }
    if manifest_path.is_file():
        expected_items["PORTABLE_BUNDLE_MANIFEST.json"] = {
            "path": "PORTABLE_BUNDLE_MANIFEST.json",
            "size_bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        }
    expected = set(expected_items)
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            top_level_dirs = {
                Path(member.name).parts[0]
                for member in members
                if Path(member.name).parts
            }
            status["archive_top_level_dirs"] = sorted(top_level_dirs)
            for member in members:
                if not member.isfile() and not member.isdir():
                    status["unsupported_archive_members"].append(
                        {"name": member.name, "type": member.type.decode("ascii", errors="replace")}
                    )
                member_path = Path(member.name)
                rel_parts = member_path.parts[1:] if len(member_path.parts) > 1 else ()
                if member.isdir() and len(member_path.parts) == 1:
                    continue
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or len(member_path.parts) <= 1
                    or not rel_parts
                ):
                    status["unsafe_archive_members"].append(member.name)
            archive_file_list = [
                "/".join(Path(member.name).parts[1:])
                for member in members
                if member.isfile() and len(Path(member.name).parts) > 1
            ]
            status["duplicate_archive_files"] = sorted(
                {path for path in archive_file_list if archive_file_list.count(path) > 1}
            )
            archive_members = {
                "/".join(Path(member.name).parts[1:]): member
                for member in members
                if member.isfile() and len(Path(member.name).parts) > 1
            }
            archive_files = set(archive_members)
            for rel, expected_item in expected_items.items():
                member = archive_members.get(rel)
                if member is None:
                    continue
                expected_size = expected_item.get("size_bytes")
                if member.size != expected_size:
                    status["size_mismatches"].append(
                        {"path": rel, "expected_size_bytes": expected_size, "actual_size_bytes": member.size}
                    )
                extracted = tar.extractfile(member)
                if extracted is None:
                    status["sha256_mismatches"].append({"path": rel, "reason": "archive member could not be read"})
                    continue
                digest = hashlib.sha256()
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                actual_sha = digest.hexdigest()
                expected_sha = expected_item.get("sha256")
                if actual_sha != expected_sha:
                    status["sha256_mismatches"].append(
                        {"path": rel, "expected_sha256": expected_sha, "actual_sha256": actual_sha}
                    )
                else:
                    status["verified_archive_file_count"] += 1
    except Exception as exc:
        status["ok"] = False
        status["violations"].append(f"archive is not readable: {type(exc).__name__}: {exc}")
        return status
    missing = sorted(expected - archive_files)
    extra = sorted(archive_files - expected)
    status["missing_archive_files"] = missing
    status["extra_archive_files"] = extra
    if missing:
        status["violations"].append("archive is missing bundle files")
    if extra:
        status["violations"].append("archive contains unexpected files")
    if status["archive_top_level_dirs"] != [bundle_root.name]:
        status["violations"].append("archive top-level directory does not match bundle root")
    if status["duplicate_archive_files"]:
        status["violations"].append("archive contains duplicate file paths")
    if status["unsafe_archive_members"]:
        status["violations"].append("archive contains unsafe member paths")
    if status["unsupported_archive_members"]:
        status["violations"].append("archive contains unsupported member types")
    if status["size_mismatches"]:
        status["violations"].append("archive file size mismatch")
    if status["sha256_mismatches"]:
        status["violations"].append("archive file sha256 mismatch")
    status["ok"] = not status["violations"]
    return status


def _code_status(manifest: Dict[str, Any], handoff: Any, args: argparse.Namespace) -> Dict[str, Any]:
    manifest_code = manifest.get("code") if isinstance(manifest.get("code"), dict) else {}
    handoff_code = handoff.get("code") if isinstance(handoff, dict) and isinstance(handoff.get("code"), dict) else {}
    status: Dict[str, Any] = {
        "ok": True,
        "manifest_commit": manifest_code.get("commit"),
        "handoff_commit": handoff_code.get("commit"),
        "manifest_dirty": manifest_code.get("dirty"),
        "handoff_dirty": handoff_code.get("dirty"),
        "manifest_origin": manifest_code.get("origin"),
        "handoff_origin": handoff_code.get("origin"),
        "require_clean_code": bool(args.require_clean_code),
        "require_commit": args.require_commit or None,
        "violations": [],
    }
    if not manifest_code:
        status["violations"].append("portable manifest code metadata is missing")
    if not handoff_code:
        status["violations"].append("handoff manifest code metadata is missing")
    if manifest_code and handoff_code:
        for field in ("commit", "origin", "dirty"):
            if manifest_code.get(field) != handoff_code.get(field):
                status["violations"].append(f"portable manifest and handoff code.{field} do not match")
    if args.require_clean_code:
        if manifest_code.get("dirty") is not False:
            status["violations"].append("portable manifest was exported from a dirty worktree")
        if handoff_code.get("dirty") is not False:
            status["violations"].append("handoff manifest was exported from a dirty worktree")
    if args.require_commit:
        if manifest_code.get("commit") != args.require_commit:
            status["violations"].append("portable manifest commit does not match required commit")
        if handoff_code.get("commit") != args.require_commit:
            status["violations"].append("handoff manifest commit does not match required commit")
    status["ok"] = not status["violations"]
    return status


def verify_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else bundle_root / "PORTABLE_BUNDLE_MANIFEST.json"
    report: Dict[str, Any] = {
        "ok": False,
        "bundle_root": str(bundle_root),
        "manifest_path": str(manifest_path),
        "violations": [],
        "file_status": [],
        "required_path_status": [],
        "handoff_manifest_status": {},
        "runbook_status": {},
        "archive_status": {},
        "code_status": {},
    }
    if not bundle_root.is_dir():
        report["violations"].append("bundle root is missing or not a directory")
        return report
    if not manifest_path.is_file():
        report["violations"].append("PORTABLE_BUNDLE_MANIFEST.json is missing")
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
    if manifest.get("schema_version") != 1:
        report["violations"].append("manifest schema_version must be 1")

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
    report["file_status"] = file_status if args.include_file_status else [
        {key: item[key] for key in ("path", "ok", "exists", "violations")} for item in file_status if not item["ok"]
    ]
    bad_files = [item for item in file_status if not item["ok"]]
    if bad_files:
        report["violations"].append(f"{len(bad_files)} manifest file entries failed verification")

    manifest_paths = {item.get("path") for item in files if isinstance(item, dict)}
    required_status = []
    for rel in REQUIRED_TOP_LEVEL_PATHS:
        path = bundle_root / rel
        ok = path.is_file()
        in_manifest_files = rel in manifest_paths
        required_status.append({"path": rel, "ok": ok, "exists": path.exists(), "in_manifest_files": in_manifest_files})
        if not ok:
            report["violations"].append(f"required bundle file is missing: {rel}")
        elif rel != "PORTABLE_BUNDLE_MANIFEST.json" and not in_manifest_files:
            report["violations"].append(f"required bundle file is missing from manifest files: {rel}")
    report["required_path_status"] = required_status

    handoff_rel = manifest.get("handoff_manifest") or "edge_handoff_manifest.json"
    handoff_status = {"path": handoff_rel, "ok": False, "violations": []}
    handoff = None
    if not _safe_relative_path(handoff_rel):
        handoff_status["violations"].append("handoff path is not a safe relative path")
    else:
        handoff_path = bundle_root / handoff_rel
        if not handoff_path.is_file():
            handoff_status["violations"].append("handoff manifest is missing")
        else:
            try:
                handoff = _load_json(handoff_path)
                handoff_status["schema_version"] = handoff.get("schema_version") if isinstance(handoff, dict) else None
                handoff_status["has_commands"] = isinstance(handoff.get("commands"), dict) if isinstance(handoff, dict) else False
                handoff_status["has_required_artifacts"] = isinstance(handoff.get("required_artifacts"), dict) if isinstance(handoff, dict) else False
                completion_gates = (
                    handoff.get("completion_gates", {})
                    if isinstance(handoff, dict) and isinstance(handoff.get("completion_gates"), dict)
                    else {}
                )
                completion_gate_reports = (
                    completion_gates.get("reports")
                    if isinstance(completion_gates.get("reports"), list)
                    else []
                )
                missing_completion_gates = [
                    gate for gate in REQUIRED_HANDOFF_COMPLETION_GATES if gate not in completion_gate_reports
                ]
                handoff_status["missing_completion_gates"] = missing_completion_gates
                required_artifacts = (
                    handoff.get("required_artifacts")
                    if isinstance(handoff, dict) and isinstance(handoff.get("required_artifacts"), dict)
                    else {}
                )
                missing_required_artifacts = [
                    artifact for artifact in REQUIRED_HANDOFF_ARTIFACTS if artifact not in required_artifacts
                ]
                handoff_status["missing_required_artifacts"] = missing_required_artifacts
                handoff_status["output_root"] = (
                    handoff.get("paths", {}).get("output_root")
                    if isinstance(handoff, dict) and isinstance(handoff.get("paths"), dict)
                    else None
                )
                if not isinstance(handoff, dict) or handoff.get("schema_version") != 1:
                    handoff_status["violations"].append("handoff manifest schema_version must be 1")
                if not handoff_status["has_commands"]:
                    handoff_status["violations"].append("handoff manifest commands are missing")
                if not handoff_status["has_required_artifacts"]:
                    handoff_status["violations"].append("handoff manifest required_artifacts are missing")
                elif missing_required_artifacts:
                    handoff_status["violations"].append("handoff manifest required_artifacts is missing final artifacts")
                if missing_completion_gates:
                    handoff_status["violations"].append("handoff manifest completion_gates.reports is missing final delivery gates")
                commands = handoff.get("commands") if isinstance(handoff, dict) and isinstance(handoff.get("commands"), dict) else {}
                command_status = {}
                for group, required_tokens in REQUIRED_HANDOFF_COMMANDS.items():
                    group_commands = commands.get(group)
                    group_text = "\n".join(str(item) for item in group_commands) if isinstance(group_commands, list) else ""
                    missing_tokens = [token for token in required_tokens if token not in group_text]
                    command_status[group] = {
                        "exists": isinstance(group_commands, list) and bool(group_commands),
                        "missing_tokens": missing_tokens,
                    }
                    if not command_status[group]["exists"]:
                        handoff_status["violations"].append(f"handoff manifest commands.{group} is missing")
                    elif missing_tokens:
                        handoff_status["violations"].append(f"handoff manifest commands.{group} is missing required tokens")
                handoff_status["command_status"] = command_status
            except Exception as exc:
                handoff_status["violations"].append(f"handoff manifest is not valid JSON: {type(exc).__name__}: {exc}")
    handoff_status["ok"] = not handoff_status["violations"]
    report["handoff_manifest_status"] = handoff_status
    if not handoff_status["ok"]:
        report["violations"].append("handoff manifest verification failed")

    code_status = _code_status(manifest, handoff, args)
    report["code_status"] = code_status
    if not code_status["ok"]:
        report["violations"].append("portable code provenance verification failed")

    runbook_rel = manifest.get("target_runbook") or "RUN_ON_TARGET.md"
    runbook_status = {"path": runbook_rel, "ok": False, "violations": []}
    if not _safe_relative_path(runbook_rel):
        runbook_status["violations"].append("runbook path is not a safe relative path")
    else:
        runbook_path = bundle_root / runbook_rel
        if not runbook_path.is_file():
            runbook_status["violations"].append("target runbook is missing")
        else:
            text = runbook_path.read_text(encoding="utf-8")
            required_text = [
                "HF_ENDPOINT=https://hf-mirror.com",
                "scripts/setup_edge_env.sh",
                "scripts/run_edge_4090_full.sh",
                "completion_audit.json",
                "validator.ok=true",
                "target_readiness.json",
                "completion_ok=true",
                "edge_results_bundle_verify.json",
                "complete=true",
                "edge_delivery_verify.json",
            ]
            missing_text = [needle for needle in required_text if needle not in text]
            runbook_status["missing_required_text"] = missing_text
            if missing_text:
                runbook_status["violations"].append("target runbook is missing required commands or completion rule")
    runbook_status["ok"] = not runbook_status["violations"]
    report["runbook_status"] = runbook_status
    if not runbook_status["ok"]:
        report["violations"].append("target runbook verification failed")

    archive_status = _archive_status(bundle_root, args.archive, files, manifest_path)
    report["archive_status"] = archive_status
    if not archive_status["ok"]:
        report["violations"].append("portable archive verification failed")

    report["file_count"] = len(files)
    report["verified_file_count"] = sum(1 for item in file_status if item["ok"])
    report["ok"] = not report["violations"]
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify an exported LLMCBench edge portable bundle.")
    parser.add_argument("--bundle-root", default=".", help="Root directory of the exported portable bundle.")
    parser.add_argument("--manifest", default="", help="Override path to PORTABLE_BUNDLE_MANIFEST.json.")
    parser.add_argument("--archive", default="", help="Optional portable bundle .tar.gz archive to compare with --bundle-root.")
    parser.add_argument("--output", default="", help="Optional JSON report output path.")
    parser.add_argument("--include-file-status", action="store_true", help="Include every manifest file entry in the report.")
    parser.add_argument("--require-clean-code", action="store_true", help="Require portable and handoff manifests to record dirty=false.")
    parser.add_argument("--require-commit", default="", help="Require portable and handoff manifests to record this exact git commit.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_bundle(args)
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
