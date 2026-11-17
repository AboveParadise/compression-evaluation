import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from tools.build_edge_handoff_manifest import build_manifest
except ModuleNotFoundError:
    from build_edge_handoff_manifest import build_manifest


PORTABLE_PATHS = [
    "README.md",
    "requirements.txt",
    "requirements-optional.txt",
    "configs/edge_4090_matrix.json",
    "configs/edge_4090_structured_pruning.json",
    "configs/edge_local_h20_matrix.json",
    "configs/edge_paper_aligned_matrix.json",
    "configs/paper_protocols.json",
    "docs/COMPLETION_AUDIT.md",
    "docs/RUN_STATUS.md",
    "edge_benchmark",
    "scripts",
    "tools",
]

EXCLUDE_NAMES = {"__pycache__", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def _git_value(args: List[str]) -> str:
    proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


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


def _ignore(_dir: str, names: List[str]) -> List[str]:
    ignored = []
    for name in names:
        path = Path(name)
        if name in EXCLUDE_NAMES or path.suffix in EXCLUDE_SUFFIXES:
            ignored.append(name)
    return ignored


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, ignore=_ignore)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_target_readme(bundle_root: Path, handoff: Dict[str, Any]) -> None:
    commands = handoff.get("commands", {}) if isinstance(handoff.get("commands"), dict) else {}
    expected_commit = (
        handoff.get("code", {}).get("commit")
        if isinstance(handoff.get("code"), dict)
        else ""
    ) or "<expected-git-commit>"
    lines = [
        "# LLMCBench Edge Portable Bundle",
        "",
        "This bundle contains the code, configs and runbook needed to reproduce the",
        "single-RTX-4090 edge compression benchmark workflow.",
        "",
        "## Target Assumptions",
        "",
        "- Single NVIDIA RTX 4090 24GB GPU",
        "- Python 3.10, PyTorch 2.x, CUDA 12.x, cuDNN",
        "- `EDGE_RUNTIME_PROFILE=stable` is the default Torch 2.6 profile; use `EDGE_RUNTIME_PROFILE=gptqmodel` only when validating GPTQModel checkpoints and rerun all affected probes/results under that profile",
        "- HF downloads through `HF_ENDPOINT=https://hf-mirror.com`",
        "- Local model, dataset, prompt and calibration paths matching the config",
        "",
        "## First Commands On Target",
        "",
        "```bash",
        f"python tools/verify_edge_portable_bundle.py --bundle-root . --require-clean-code --require-commit {expected_commit} --output portable_bundle_verify.json",
        f"python tools/verify_edge_portable_bundle.py --bundle-root . --archive <llmcbench_edge_portable.tar.gz> --require-clean-code --require-commit {expected_commit} --output portable_bundle_archive_verify.json",
        "export HF_ENDPOINT=https://hf-mirror.com",
        "sudo DATA_ROOT=/data bash scripts/init_edge_storage.sh",
        "bash scripts/setup_edge_env.sh",
        "# For Qwen GPTQModel checkpoints, use: EDGE_RUNTIME_PROFILE=gptqmodel bash scripts/setup_edge_env.sh",
        "OPTIONAL_BACKEND_DRY_RUN=1 REPORT_OUTPUT=/data/outputs/llmcbench_edge_4090/edge_optional_backends_report.json bash scripts/setup_optional_backends.sh",
        "bash scripts/setup_optional_backends.sh",
        "CHECK_OPTIONAL_BACKEND_RESOLUTION=1 CHECK_QUANT_RUNTIME=1 bash scripts/run_edge_4090_full.sh",
        "```",
        "",
        "## Handoff Command Groups",
        "",
        "`edge_handoff_manifest.json` is the authoritative command manifest. The most",
        "important command groups are repeated below for quick copy/paste.",
        "",
    ]
    for group in (
        "prepare",
        "prepare_precompressed",
        "preflight",
        "prewarm",
        "readiness",
        "precompressed_config_followup",
        "run",
        "manual_finalize",
        "resume",
        "scorecard",
        "post_run_audit",
        "baseline_post_run_audit",
        "precompressed_config_post_run_audit",
        "export_results",
        "verify_results_bundle",
        "verify_delivery",
    ):
        group_commands = commands.get(group)
        if not isinstance(group_commands, list) or not group_commands:
            continue
        lines.extend([f"### {group}", "", "```bash"])
        lines.extend(str(command) for command in group_commands)
        lines.extend(["```", ""])
    lines.extend(
        [
            "## Completion Rule",
            "",
            "The benchmark is complete only when the target output root contains the full",
            "two-model/all-method matrix, `completion_audit.json` reports `ok=true`",
            "with `validator.ok=true`, `target_readiness.json` reports `ok=true`",
            "with `completion_ok=true`, `edge_results_bundle_verify.json` reports",
            "`ok=true` and `complete=true`, and `edge_delivery_verify.json` reports",
            "`ok=true` with all delivery substatus checks passing.",
            "",
        ]
    )
    (bundle_root / "RUN_ON_TARGET.md").write_text("\n".join(lines), encoding="utf-8")


def _manifest(bundle_root: Path, copied_paths: List[str], archive: str, handoff: Dict[str, Any]) -> Dict[str, Any]:
    files = []
    for path in _iter_files(bundle_root):
        rel = path.relative_to(bundle_root).as_posix()
        if rel == "PORTABLE_BUNDLE_MANIFEST.json":
            continue
        files.append({"path": rel, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {
        "schema_version": 1,
        "purpose": "Portable LLMCBench edge benchmark bundle for a single RTX 4090 target.",
        "code": {
            "branch": _git_value(["branch", "--show-current"]),
            "commit": _git_value(["rev-parse", "HEAD"]),
            "origin": _git_value(["remote", "get-url", "origin"]),
            "dirty": bool(_git_value(["status", "--porcelain"])),
        },
        "copied_paths": copied_paths,
        "archive": archive or None,
        "handoff_manifest": "edge_handoff_manifest.json",
        "target_runbook": "RUN_ON_TARGET.md",
        "config_path": handoff.get("config", {}).get("path") if isinstance(handoff.get("config"), dict) else None,
        "output_root": handoff.get("paths", {}).get("output_root") if isinstance(handoff.get("paths"), dict) else None,
        "required_artifacts": sorted((handoff.get("required_artifacts") or {}).keys()),
        "files": files,
    }


def export_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"output directory already exists: {output_dir}. Use --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    copied = []
    for rel in PORTABLE_PATHS:
        src = repo_root / rel
        if not src.exists():
            raise SystemExit(f"required portable path is missing: {src}")
        _copy_path(src, output_dir / rel)
        copied.append(rel)

    cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        handoff_args = argparse.Namespace(config=args.config, output_root=args.output_root, output="")
        handoff = build_manifest(handoff_args)
    finally:
        os.chdir(cwd)
    (output_dir / "edge_handoff_manifest.json").write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_target_readme(output_dir, handoff)

    archive = str(Path(args.archive).expanduser().resolve()) if args.archive else ""
    manifest = _manifest(output_dir, copied, archive, handoff)
    (output_dir / "PORTABLE_BUNDLE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.archive:
        archive_path = Path(archive)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)

    return {
        "ok": True,
        "output_dir": str(output_dir),
        "archive": archive or None,
        "file_count": len(manifest["files"]),
        "manifest": str(output_dir / "PORTABLE_BUNDLE_MANIFEST.json"),
        "handoff_manifest": str(output_dir / "edge_handoff_manifest.json"),
        "target_runbook": str(output_dir / "RUN_ON_TARGET.md"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a portable LLMCBench edge benchmark bundle.")
    parser.add_argument("--repo-root", default=".", help="Repository root to copy from.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json", help="Matrix config path relative to --repo-root.")
    parser.add_argument("--output-root", default="", help="Override final output root recorded in the handoff manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory to create for the portable bundle.")
    parser.add_argument("--archive", default="", help="Optional .tar.gz path to write after creating the bundle directory.")
    parser.add_argument("--force", action="store_true", help="Replace --output-dir if it already exists.")
    return parser


def main() -> None:
    result = export_bundle(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
