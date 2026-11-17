import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, List

try:
    from tools.verify_edge_results_bundle import _completion_audit_status, verify_results_bundle
except ModuleNotFoundError:
    from verify_edge_results_bundle import _completion_audit_status, verify_results_bundle


ROOT_BUNDLE_MATCH_FILES = [
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_status(path: Path, key: str = "ok") -> Dict[str, Any]:
    status = {"path": str(path), "exists": path.is_file(), "ok": False, "violations": []}
    if not path.is_file():
        status["violations"].append("file is missing")
        return status
    try:
        payload = _load_json(path)
    except Exception as exc:
        status["violations"].append(f"invalid JSON: {type(exc).__name__}: {exc}")
        return status
    if not isinstance(payload, dict):
        status["violations"].append("JSON root is not an object")
        return status
    status["value"] = payload.get(key)
    status["ok"] = payload.get(key) is True
    if not status["ok"]:
        status["violations"].append(f"{key} is not true")
    if path.name == "target_readiness.json":
        blockers = payload.get("before_matrix_blockers")
        status["ready_for_matrix"] = payload.get("ready_for_matrix")
        status["completion_ok"] = payload.get("completion_ok")
        status["before_matrix_blocker_count"] = len(blockers) if isinstance(blockers, list) else None
        if payload.get("completion_ok") is not True:
            status["ok"] = False
            status["violations"].append("completion_ok is not true")
        if status["ok"] and isinstance(blockers, list) and blockers:
            status["ok"] = False
            status["violations"].append("before_matrix_blockers is not empty")
    return status


def _root_bundle_match_status(root: Path, bundle_root: Path, names: List[str]) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "ok": False,
        "root": str(root),
        "results_bundle": str(bundle_root),
        "files": [],
        "violations": [],
    }
    for name in names:
        root_path = root / name
        bundle_path = bundle_root / name
        item: Dict[str, Any] = {
            "path": name,
            "root_exists": root_path.is_file(),
            "bundle_exists": bundle_path.is_file(),
            "size_matches": False,
            "sha256_matches": False,
            "ok": False,
            "violations": [],
        }
        if not root_path.is_file():
            item["violations"].append("root file is missing")
        if not bundle_path.is_file():
            item["violations"].append("bundle file is missing")
        if root_path.is_file() and bundle_path.is_file():
            root_size = root_path.stat().st_size
            bundle_size = bundle_path.stat().st_size
            item["root_size_bytes"] = root_size
            item["bundle_size_bytes"] = bundle_size
            item["size_matches"] = root_size == bundle_size
            if not item["size_matches"]:
                item["violations"].append("size mismatch")
            root_sha = _sha256(root_path)
            bundle_sha = _sha256(bundle_path)
            item["root_sha256"] = root_sha
            item["bundle_sha256"] = bundle_sha
            item["sha256_matches"] = root_sha == bundle_sha
            if not item["sha256_matches"]:
                item["violations"].append("sha256 mismatch")
        item["ok"] = not item["violations"]
        if not item["ok"]:
            status["violations"].append(f"root and results bundle differ for {name}")
        status["files"].append(item)
    status["ok"] = not status["violations"]
    return status


def _safe_archive_relpath(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts and bool(path.parts)


def _portable_bundle_file(path: Path) -> bool:
    return "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}


def _archive_bundle_match_status(bundle_root: Path, archive_path: Path) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "ok": False,
        "results_bundle": str(bundle_root),
        "archive": str(archive_path),
        "archive_exists": archive_path.is_file(),
        "files": [],
        "extra_archive_files": [],
        "missing_archive_files": [],
        "violations": [],
    }
    if not bundle_root.is_dir():
        status["violations"].append("results bundle directory is missing")
        return status
    if not archive_path.is_file():
        status["violations"].append("results bundle archive is missing")
        return status
    bundle_files = {
        path.relative_to(bundle_root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(bundle_root.rglob("*"))
        if path.is_file() and _portable_bundle_file(path.relative_to(bundle_root))
    }
    archive_files: Dict[str, Dict[str, Any]] = {}
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = [member for member in tar.getmembers() if member.isfile()]
            prefixes = {
                Path(member.name).parts[0]
                for member in members
                if _safe_archive_relpath(member.name) and len(Path(member.name).parts) > 1
            }
            strip_prefix = next(iter(prefixes)) if len(prefixes) == 1 else ""
            for member in members:
                raw_parts = Path(member.name).parts
                if not _safe_archive_relpath(member.name):
                    status["violations"].append(f"archive contains unsafe path: {member.name}")
                    continue
                rel_parts = raw_parts[1:] if strip_prefix and raw_parts and raw_parts[0] == strip_prefix else raw_parts
                rel = Path(*rel_parts).as_posix()
                if not _safe_archive_relpath(rel):
                    status["violations"].append(f"archive contains unsafe relative path: {member.name}")
                    continue
                if not _portable_bundle_file(Path(rel)):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    status["violations"].append(f"archive member cannot be read: {member.name}")
                    continue
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
                archive_files[rel] = {"size_bytes": size, "sha256": digest.hexdigest(), "tar_path": member.name}
    except Exception as exc:
        status["violations"].append(f"archive cannot be read: {type(exc).__name__}: {exc}")
        return status

    for rel, expected in bundle_files.items():
        item: Dict[str, Any] = {
            "path": rel,
            "bundle_size_bytes": expected["size_bytes"],
            "bundle_sha256": expected["sha256"],
            "archive_exists": rel in archive_files,
            "size_matches": False,
            "sha256_matches": False,
            "ok": False,
            "violations": [],
        }
        actual = archive_files.get(rel)
        if actual is None:
            item["violations"].append("file is missing from archive")
            status["missing_archive_files"].append(rel)
        else:
            item["archive_tar_path"] = actual["tar_path"]
            item["archive_size_bytes"] = actual["size_bytes"]
            item["archive_sha256"] = actual["sha256"]
            item["size_matches"] = actual["size_bytes"] == expected["size_bytes"]
            item["sha256_matches"] = actual["sha256"] == expected["sha256"]
            if not item["size_matches"]:
                item["violations"].append("size mismatch")
            if not item["sha256_matches"]:
                item["violations"].append("sha256 mismatch")
        item["ok"] = not item["violations"]
        if not item["ok"]:
            status["violations"].append(f"archive and results bundle differ for {rel}")
        status["files"].append(item)

    extras = sorted(set(archive_files) - set(bundle_files))
    status["extra_archive_files"] = extras
    if extras:
        status["violations"].append("archive contains files not present in results bundle")
    status["ok"] = not status["violations"]
    return status


def verify_delivery(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    bundle_root = Path(args.results_bundle).expanduser().resolve() if args.results_bundle else root / "edge_results_bundle"
    archive_path = Path(args.results_archive).expanduser().resolve() if args.results_archive else root / "edge_results_bundle.tar.gz"
    report: Dict[str, Any] = {
        "ok": False,
        "root": str(root),
        "results_bundle": str(bundle_root),
        "results_archive": str(archive_path),
        "violations": [],
        "completion_audit_status": {},
        "target_readiness_status": {},
        "root_bundle_match_status": {},
        "archive_bundle_match_status": {},
        "results_bundle_status": {},
    }
    if not root.is_dir():
        report["violations"].append("delivery root is missing or not a directory")
        return report
    completion_status = _completion_audit_status(root / "completion_audit.json")
    readiness_status = _json_status(root / "target_readiness.json")
    report["completion_audit_status"] = completion_status
    report["target_readiness_status"] = readiness_status
    if not completion_status["ok"]:
        report["violations"].append("completion_audit.json is not complete or validator.ok is not true")
    if not readiness_status["ok"]:
        report["violations"].append("target_readiness.json is not complete")

    bundle_args = argparse.Namespace(
        bundle_root=str(bundle_root),
        manifest="",
        output="",
        config=args.config,
        require_complete=True,
        require_target_readiness_ok=True,
        require_target_runtime=not args.skip_target_runtime,
        include_file_status=args.include_file_status,
    )
    bundle_status = verify_results_bundle(bundle_args)
    report["results_bundle_status"] = bundle_status
    if not bundle_status.get("ok"):
        report["violations"].append("results bundle verification failed")
    match_status = _root_bundle_match_status(root, bundle_root, ROOT_BUNDLE_MATCH_FILES)
    report["root_bundle_match_status"] = match_status
    if not match_status["ok"]:
        report["violations"].append("root and results bundle artifacts do not match")
    archive_status = _archive_bundle_match_status(bundle_root, archive_path)
    report["archive_bundle_match_status"] = archive_status
    if not archive_status["ok"]:
        report["violations"].append("results bundle archive does not match bundle directory")
    report["ok"] = not report["violations"]
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a completed LLMCBench edge delivery directory.")
    parser.add_argument("--root", required=True, help="Final matrix output root.")
    parser.add_argument("--results-bundle", default="", help="Override path to edge_results_bundle.")
    parser.add_argument("--results-archive", default="", help="Override path to edge_results_bundle.tar.gz.")
    parser.add_argument("--config", default="", help="Optional matrix config passed to results-bundle verification.")
    parser.add_argument("--skip-target-runtime", action="store_true", help="Do not require target 4090 runtime metadata in the results bundle.")
    parser.add_argument("--include-file-status", action="store_true", help="Include every results-bundle file entry in the report.")
    parser.add_argument("--output", default="", help="Optional JSON report output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_delivery(args)
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
