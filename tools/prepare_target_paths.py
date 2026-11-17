import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from tools.prepare_edge_assets import DEFAULT_PROMPTS, calibration_seed_texts
except ModuleNotFoundError:
    from prepare_edge_assets import DEFAULT_PROMPTS, calibration_seed_texts


DIRECTORY_DEFAULT_KEYS = [
    "hf_cache_dir",
    "data_dir",
    "output_root",
]

FILE_DEFAULT_KEYS = [
    "calibration_file",
    "prompt_file",
]


def _load_config(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mkdir(path: Path, dry_run: bool) -> Dict[str, Any]:
    existed_before = path.exists()
    ok = True
    error = None
    if not dry_run:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
    exists_after = path.exists() if not dry_run else existed_before
    is_dir_after = path.is_dir() if exists_after else False
    writable = os.access(path, os.W_OK) if exists_after else None
    return {
        "path": str(path),
        "ok": ok and (dry_run or (exists_after and is_dir_after)),
        "dry_run": dry_run,
        "existed_before": existed_before,
        "exists_after": exists_after,
        "is_dir_after": is_dir_after,
        "writable": writable,
        "created": (not dry_run) and (not existed_before) and exists_after,
        "error": error,
    }


def _nonempty_line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _write_seed_file(
    path: Path,
    lines: Iterable[str],
    dry_run: bool,
    min_lines: int = 0,
) -> Dict[str, Any]:
    existed_before = path.exists()
    ok = True
    error = None
    if not dry_run:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
    exists_after = path.exists() if not dry_run else existed_before
    is_file_after = path.is_file() if exists_after else False
    writable = os.access(path, os.W_OK) if exists_after else None
    nonempty_lines = _nonempty_line_count(path) if exists_after and is_file_after else 0
    line_count_ok = min_lines <= 0 or nonempty_lines >= min_lines
    return {
        "path": str(path),
        "ok": ok and (dry_run or (exists_after and is_file_after and line_count_ok)),
        "dry_run": dry_run,
        "existed_before": existed_before,
        "exists_after": exists_after,
        "is_file_after": is_file_after,
        "writable": writable,
        "created": (not dry_run) and (not existed_before) and exists_after,
        "size_bytes": path.stat().st_size if exists_after and is_file_after else None,
        "nonempty_lines": nonempty_lines,
        "required_nonempty_lines": min_lines,
        "error": (
            error
            or (
                None
                if line_count_ok
                else f"file contains {nonempty_lines} non-empty lines; requires at least {min_lines}"
            )
        ),
    }


def _selected_model_dirs(config: Dict[str, Any], only_model: List[str]) -> List[Path]:
    wanted = set(only_model or [])
    dirs = []
    for model in config.get("models", []):
        if not isinstance(model, dict):
            continue
        if wanted and model.get("name") not in wanted:
            continue
        path = model.get("path")
        if path:
            dirs.append(Path(str(path)).expanduser())
    return dirs


def _path_from_default(defaults: Dict[str, Any], key: str, override: str = "") -> Path:
    return Path(override or str(defaults.get(key, ""))).expanduser()


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    config = _load_config(args.config)
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    if not defaults:
        raise SystemExit("config defaults are missing")

    directory_paths = {
        key: _path_from_default(defaults, key)
        for key in DIRECTORY_DEFAULT_KEYS
        if defaults.get(key)
    }
    if args.hf_cache_dir:
        directory_paths["hf_cache_dir"] = Path(args.hf_cache_dir).expanduser()
    if args.data_dir:
        directory_paths["data_dir"] = Path(args.data_dir).expanduser()
    if args.output_root:
        directory_paths["output_root"] = Path(args.output_root).expanduser()
    directory_paths["model_root"] = Path(args.model_root).expanduser()
    directory_paths["precompressed_model_root"] = Path(args.precompressed_model_root).expanduser()

    for model_dir in _selected_model_dirs(config, args.only_model):
        directory_paths[f"model_parent:{model_dir.name}"] = model_dir.parent

    calibration_file = _path_from_default(defaults, "calibration_file", args.calibration_file)
    prompt_file = _path_from_default(defaults, "prompt_file", args.prompt_file)

    directories = {
        name: _mkdir(path, args.dry_run)
        for name, path in sorted(directory_paths.items())
    }
    files = {
        "calibration_file": _write_seed_file(
            calibration_file,
            calibration_seed_texts(int(defaults.get("calibration_samples", 128))),
            args.dry_run,
            min_lines=int(defaults.get("calibration_samples", 128)),
        ),
        "prompt_file": _write_seed_file(prompt_file, DEFAULT_PROMPTS, args.dry_run),
    }
    report = {
        "ok": all(item["ok"] for item in directories.values()) and all(item["ok"] for item in files.values()),
        "config": str(Path(args.config).expanduser()),
        "dry_run": args.dry_run,
        "directories": directories,
        "files": files,
        "notes": [
            "This tool prepares local target paths only; it does not download model weights or datasets.",
            "Existing prompt files are preserved. Existing calibration files must contain at least the configured calibration_samples non-empty lines.",
            "The bootstrap calibration file is only an operational seed; replace it with representative calibration data before reporting method-quality results.",
            "Model checkpoint directories are not created directly; their parent directories are prepared for downloads or symlinks.",
        ],
    }
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "output": args.output or None}, ensure_ascii=False, indent=2))
    if args.fail_on_error and not report["ok"]:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local target paths for the LLMCBench 4090 workflow.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--only-model", action="append", default=[])
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--calibration-file", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--model-root", default="/data/models")
    parser.add_argument("--precompressed-model-root", default="/data/models/precompressed")
    return parser


def main() -> None:
    prepare(build_parser().parse_args())


if __name__ == "__main__":
    main()
