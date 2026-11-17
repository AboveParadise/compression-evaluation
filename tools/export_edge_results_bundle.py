import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from tools.verify_edge_results_bundle import _completion_audit_status, _target_readiness_status
except ModuleNotFoundError:
    from verify_edge_results_bundle import _completion_audit_status, _target_readiness_status


REQUIRED_RESULT_FILES = [
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
]

OPTIONAL_RESULT_FILES = [
    "current_goal_audit.json",
    "edge_prepare_metadata_report.json",
    "edge_prepare_assets_report.json",
    "edge_prepare_precompressed_report.json",
    "edge_matrix_with_precompressed.json",
    "matrix_resume_plan.json",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _artifact(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _copy_named_files(output_root: Path, bundle_root: Path, names: List[str]) -> List[Dict[str, Any]]:
    copied = []
    for name in names:
        src = output_root / name
        item = _artifact(src)
        if src.is_file():
            _copy_file(src, bundle_root / name)
            item["bundle_path"] = name
        copied.append({"name": name, **item})
    return copied


def _copy_job_logs(output_root: Path, bundle_root: Path) -> List[Dict[str, Any]]:
    copied = []
    for path in sorted(output_root.glob("*/*/job.log")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_root)
        dst_rel = Path("job_logs") / rel
        _copy_file(path, bundle_root / dst_rel)
        copied.append(
            {
                "source_path": str(path),
                "bundle_path": dst_rel.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return copied


def _status(path: Path, key: str = "ok") -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "ok": False}
    try:
        payload = _load_json(path)
    except Exception as exc:
        return {"path": str(path), "exists": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"path": str(path), "exists": True, "ok": False, "error": "JSON root is not an object"}
    return {"path": str(path), "exists": True, "ok": payload.get(key) is True, "value": payload.get(key)}


def _write_readme(bundle_root: Path, manifest: Dict[str, Any]) -> None:
    lines = [
        "# LLMCBench Edge Results Bundle",
        "",
        "This bundle contains final or diagnostic LLMCBench edge matrix artifacts.",
        "",
        "## Source",
        "",
        f"- Output root: `{manifest['source_output_root']}`",
        f"- Complete: `{manifest['complete']}`",
        f"- Completion audit ok: `{manifest['completion_audit_status'].get('ok')}`",
        f"- Completion validator ok: `{manifest['completion_audit_status'].get('validator_ok')}`",
        f"- Target readiness ok: `{manifest['target_readiness_status'].get('ok')}`",
        f"- Target completion ok: `{manifest['target_readiness_status'].get('completion_ok')}`",
        "",
        "## Key Files",
        "",
        "- `matrix_results.json` / `matrix_results.csv`: collected tabular results",
        "- `edge_report.md`: human-readable report",
        "- `completion_audit.json`: final machine-readable gate",
        "- `current_goal_audit.json`: prompt-to-artifact completion snapshot",
        "- `edge_target_paths_report.json`: target path preparation evidence",
        "- `matrix_run_report.json`: per-job run status and log paths",
        "- `job_logs/`: copied per-job logs when present",
        "- `tools/verify_edge_results_bundle.py`: verifies the moved results bundle",
        "- `tools/verify_edge_delivery.py`: verifies the final root/bundle/archive delivery",
        "- `RESULTS_BUNDLE_MANIFEST.json`: file checksums and completeness status",
        "",
        "## Verification Scope",
        "",
        "`tools/verify_edge_results_bundle.py` checks manifest file hashes, duplicate",
        "manifest paths, JSON/CSV matrix consistency, model/method coverage, target",
        "task and MMLU subtask score coverage, score metric keys, non-limited and",
        "non-failed accuracy provenance, aggregate score consistency, deployment",
        "tokens/s and memory consistency, resource storage/FLOPs consistency,",
        "quantized storage estimates, compression provenance and target runtime",
        "metadata when `--require-target-runtime` is used.",
        "",
        "## Verify",
        "",
        "```bash",
        "python tools/verify_edge_results_bundle.py --bundle-root . --require-complete --require-target-readiness-ok --require-target-runtime",
        "python tools/verify_edge_delivery.py --root <output_root> --results-bundle . --results-archive <edge_results_bundle.tar.gz> --output <output_root>/edge_delivery_verify.json",
        "```",
        "",
        "A final result delivery is acceptable only when `complete` is `true` and",
        "`completion_audit.json` reports `ok=true` with `validator.ok=true` and",
        "`target_readiness.json` reports `ok=true` with `completion_ok=true` and no",
        "`before_matrix_blockers`.",
        "",
    ]
    (bundle_root / "RESULTS_BUNDLE_README.md").write_text("\n".join(lines), encoding="utf-8")


def export_results_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    bundle_root = Path(args.bundle_dir).expanduser().resolve()
    if not output_root.is_dir():
        raise SystemExit(f"output root is missing or not a directory: {output_root}")
    if bundle_root.exists():
        if not args.force:
            raise SystemExit(f"bundle directory already exists: {bundle_root}. Use --force to replace it.")
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)

    required = _copy_named_files(output_root, bundle_root, REQUIRED_RESULT_FILES)
    optional = _copy_named_files(output_root, bundle_root, OPTIONAL_RESULT_FILES)
    job_logs = _copy_job_logs(output_root, bundle_root) if args.include_job_logs else []
    verifier_src = Path(__file__).resolve().with_name("verify_edge_results_bundle.py")
    if verifier_src.is_file():
        _copy_file(verifier_src, bundle_root / "tools" / "verify_edge_results_bundle.py")
    delivery_verifier_src = Path(__file__).resolve().with_name("verify_edge_delivery.py")
    if delivery_verifier_src.is_file():
        _copy_file(delivery_verifier_src, bundle_root / "tools" / "verify_edge_delivery.py")
    prewarm_src = Path(__file__).resolve().with_name("prewarm_lm_eval_tasks.py")
    if prewarm_src.is_file():
        _copy_file(prewarm_src, bundle_root / "tools" / "prewarm_lm_eval_tasks.py")
    completion_status = _completion_audit_status(output_root / "completion_audit.json")
    readiness_status = _target_readiness_status(output_root / "target_readiness.json")
    missing_required = [item for item in required if not item["exists"]]
    complete = (
        not missing_required
        and completion_status.get("ok") is True
        and readiness_status.get("ok") is True
    )
    if args.require_complete and not complete:
        shutil.rmtree(bundle_root)
        raise SystemExit("result bundle is incomplete; missing required artifacts, completion_audit.json ok/validator.ok is not true, or target_readiness.json ok/completion_ok is not true")

    manifest = {
        "schema_version": 1,
        "purpose": "Portable LLMCBench edge benchmark results bundle.",
        "source_output_root": str(output_root),
        "complete": complete,
        "missing_required_artifacts": [item["name"] for item in missing_required],
        "completion_audit_status": completion_status,
        "target_readiness_status": readiness_status,
        "required_artifacts": required,
        "optional_artifacts": optional,
        "job_logs": job_logs,
        "archive": str(Path(args.archive).expanduser().resolve()) if args.archive else None,
    }
    _write_readme(bundle_root, manifest)
    manifest["files"] = [
        {
            "path": path.relative_to(bundle_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _iter_files(bundle_root)
        if path.name != "RESULTS_BUNDLE_MANIFEST.json"
    ]
    (bundle_root / "RESULTS_BUNDLE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.archive:
        archive = Path(args.archive).expanduser().resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(bundle_root, arcname=bundle_root.name)

    return {
        "ok": complete or not args.require_complete,
        "complete": complete,
        "bundle_dir": str(bundle_root),
        "archive": manifest["archive"],
        "manifest": str(bundle_root / "RESULTS_BUNDLE_MANIFEST.json"),
        "missing_required_artifacts": manifest["missing_required_artifacts"],
        "job_log_count": len(job_logs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LLMCBench edge result artifacts into a portable bundle.")
    parser.add_argument("--output-root", required=True, help="Matrix output root containing final artifacts.")
    parser.add_argument("--bundle-dir", required=True, help="Directory to create for the results bundle.")
    parser.add_argument("--archive", default="", help="Optional .tar.gz archive to write.")
    parser.add_argument("--include-job-logs", action="store_true", help="Copy <output_root>/<model>/<method>/job.log files.")
    parser.add_argument("--require-complete", action="store_true", help="Fail unless all required final artifacts exist, completion_audit.json has ok=true with validator.ok=true, and target_readiness.json has ok=true with completion_ok=true and no before_matrix_blockers.")
    parser.add_argument("--force", action="store_true", help="Replace --bundle-dir if it already exists.")
    return parser


def main() -> None:
    result = export_results_bundle(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
