import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_CALIBRATION_TEXTS = [
    "Large language model compression reduces memory and bandwidth cost on edge devices.",
    "Sparse pruning removes less important weights while attempting to preserve model accuracy.",
    "Quantization stores model weights in lower precision formats such as int8 or int4.",
    "MMLU, ARC, HellaSwag, PIQA and WinoGrande evaluate different reasoning abilities.",
    "Deployment benchmarks should measure peak GPU memory and generated tokens per second.",
]

DEFAULT_PROMPTS = [
    "Explain why model compression is useful for edge-side large language model deployment.",
    "Summarize the tradeoff between quantization accuracy and inference speed.",
    "List two reasons why GPU memory usage matters for edge deployment.",
]

METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

OFFICIAL_C4_DATASET = "allenai/c4"
OFFICIAL_C4_TRAIN_SHARD = "en/c4-train.00000-of-01024.json.gz"
OFFICIAL_C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"


def _load_config(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run(cmd, env, dry_run: bool) -> Dict:
    print("+ " + " ".join(str(part) for part in cmd))
    if dry_run:
        return {"ok": True, "returncode": 0, "dry_run": True}
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def calibration_seed_texts(sample_count: int) -> List[str]:
    if sample_count <= 0:
        raise ValueError(f"calibration sample count must be positive, got {sample_count}")
    return [
        f"[LLMCBench bootstrap calibration sample {index + 1}/{sample_count}] {text}"
        for index in range(sample_count)
        for text in [DEFAULT_CALIBRATION_TEXTS[index % len(DEFAULT_CALIBRATION_TEXTS)]]
    ]


def _nonempty_line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _ensure_text_file(path: str, lines, dry_run: bool, min_lines: int = 0) -> Dict:
    target = Path(path).expanduser()
    print(f"[ensure] {target}")
    if dry_run:
        return {
            "ok": True,
            "path": str(target),
            "dry_run": True,
            "required_nonempty_lines": min_lines,
        }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        nonempty_lines = _nonempty_line_count(target)
        return {
            "ok": min_lines <= 0 or nonempty_lines >= min_lines,
            "path": str(target),
            "exists": target.exists(),
            "nonempty_lines": nonempty_lines,
            "required_nonempty_lines": min_lines,
            "error": (
                None
                if min_lines <= 0 or nonempty_lines >= min_lines
                else f"file contains {nonempty_lines} non-empty lines; requires at least {min_lines}"
            ),
        }
    except Exception as exc:
        return {"ok": False, "path": str(target), "error": str(exc)}


def _has_weight_files(path: Path) -> bool:
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin")) or any(path.glob("*.pt"))


def _has_metadata_files(path: Path) -> bool:
    return (path / "config.json").exists()


def _has_runnable_checkpoint(path: Path) -> bool:
    return _has_metadata_files(path) and _has_weight_files(path)


def _model_source_candidates(source_roots: List[str], model: Dict, model_id_or_path: str) -> List[Path]:
    candidates: List[Path] = []
    names = [
        model.get("name", ""),
        Path(model.get("path", "")).name,
        Path(str(model_id_or_path)).name,
    ]
    for root in source_roots:
        root_path = Path(root).expanduser()
        for name in names:
            if name:
                candidates.append(root_path / name)
    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _find_existing_checkpoint(source_roots: List[str], model: Dict, model_id_or_path: str) -> Optional[Path]:
    for candidate in _model_source_candidates(source_roots, model, model_id_or_path):
        if _has_runnable_checkpoint(candidate):
            return candidate
    return None


def _link_existing_checkpoint(source: Path, target: Path, dry_run: bool) -> Dict:
    source = source.expanduser()
    target = target.expanduser()
    if target.exists() or target.is_symlink():
        if _has_runnable_checkpoint(target):
            return {
                "ok": True,
                "source": str(source),
                "target": str(target),
                "skipped": True,
                "reason": "target_checkpoint_already_exists",
            }
        return {
            "ok": False,
            "source": str(source),
            "target": str(target),
            "error": "target_path_exists_but_is_not_a_complete_checkpoint",
        }
    print(f"[link] {target} -> {source}")
    if dry_run:
        return {"ok": True, "source": str(source), "target": str(target), "dry_run": True}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)
        return {"ok": True, "source": str(source), "target": str(target), "created": True}
    except Exception as exc:
        return {"ok": False, "source": str(source), "target": str(target), "error": str(exc)}


def _download_model(model_id_or_path: str, local_dir: str, env: Dict[str, str], dry_run: bool, metadata_only: bool) -> Dict:
    local = Path(local_dir).expanduser()
    if local.exists():
        if metadata_only and _has_metadata_files(local):
            print(f"[skip] model metadata exists: {local}")
            return {"ok": True, "model_id": model_id_or_path, "local_dir": str(local), "skipped": True, "metadata_only": True}
        if not metadata_only and _has_metadata_files(local) and _has_weight_files(local):
            print(f"[skip] model checkpoint exists: {local}")
            return {"ok": True, "model_id": model_id_or_path, "local_dir": str(local), "skipped": True, "metadata_only": False}
    if metadata_only:
        results = []
        for filename in METADATA_FILES:
            cmd = [
                "huggingface-cli",
                "download",
                model_id_or_path,
                filename,
                "--local-dir",
                str(local),
                "--local-dir-use-symlinks",
                "False",
            ]
            item = _run(cmd, env, dry_run)
            item["file"] = filename
            if not item["ok"] and filename != "config.json" and "Entry Not Found" in item.get("stderr_tail", ""):
                item["optional_missing"] = True
            results.append(item)
        ok = any(item["ok"] and item["file"] == "config.json" for item in results)
        result = {
            "ok": ok,
            "returncode": 0 if ok else 1,
            "files": results,
            "model_id": model_id_or_path,
            "local_dir": str(local),
            "metadata_only": True,
        }
        return result
    cmd = ["huggingface-cli", "download", model_id_or_path, "--local-dir", str(local), "--local-dir-use-symlinks", "False"]
    result = _run(cmd, env, dry_run)
    result.update({"model_id": model_id_or_path, "local_dir": str(local)})
    warnings = []
    if "Permission denied" in result.get("stderr_tail", ""):
        warnings.append("permission_denied_seen_in_stderr")
    if warnings:
        result["warnings"] = warnings
    if metadata_only:
        result["metadata_only"] = True
    return result


def _write_report(report: Dict, output: Optional[str]) -> None:
    if not output:
        return
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _report_ok(report: Dict) -> bool:
    models_ok = all(item.get("ok") for item in report.get("models", []))
    files_ok = all(item.get("ok") for item in report.get("files", []))
    calibration = report.get("pruning_calibration")
    calibration_ok = True if calibration is None else bool(calibration.get("ok"))
    probe = report.get("lm_eval_probe")
    probe_ok = True if probe is None else bool(probe.get("ok"))
    return models_ok and files_ok and calibration_ok and probe_ok


def _prepare_pruning_calibration(defaults: Dict, env: Dict[str, str], dry_run: bool) -> Dict:
    source = defaults.get("pruning_calibration_source", "text_file")
    if source != "c4":
        return {"ok": True, "source": source, "skipped": True}
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        return {"ok": False, "source": source, "error": f"missing_huggingface_hub:{exc}"}
    dataset = defaults.get("calibration_dataset", OFFICIAL_C4_DATASET)
    shard = defaults.get("calibration_c4_data_file", OFFICIAL_C4_TRAIN_SHARD)
    revision = defaults.get("calibration_c4_revision", OFFICIAL_C4_REVISION)
    cache_dir = defaults.get("calibration_dataset_cache") or defaults["data_dir"]
    endpoint = env["HF_ENDPOINT"]
    report = {
        "ok": False,
        "source": source,
        "dataset": dataset,
        "dataset_config": defaults.get("calibration_dataset_config"),
        "revision": revision,
        "shard": shard,
        "cache_dir": cache_dir,
        "endpoint": endpoint,
        "sample_count": defaults.get("calibration_samples"),
        "token_length": defaults.get("seqlen"),
        "seed": defaults.get("calibration_seed", 0),
    }
    if dry_run:
        report.update({"ok": True, "dry_run": True})
        return report
    try:
        local_file = hf_hub_download(
            repo_id=dataset,
            repo_type="dataset",
            filename=shard,
            revision=revision,
            cache_dir=str(Path(cache_dir).expanduser()),
            endpoint=endpoint,
        )
        report.update({"ok": True, "local_file": local_file, "size_bytes": Path(local_file).stat().st_size})
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def prepare(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    defaults = config["defaults"]
    if args.calibration_file:
        defaults["calibration_file"] = args.calibration_file
    if args.prompt_file:
        defaults["prompt_file"] = args.prompt_file
    calibration_samples = int(defaults.get("calibration_samples", 128))
    env = os.environ.copy()
    env["HF_ENDPOINT"] = args.hf_endpoint or defaults["hf_endpoint"]
    env["HF_HOME"] = args.hf_home or defaults["hf_cache_dir"]
    env["HF_DATASETS_CACHE"] = args.data_dir or defaults["data_dir"]
    env["TRANSFORMERS_CACHE"] = str(Path(env["HF_HOME"]).expanduser() / "transformers")
    _check_free_disk(args.min_free_disk_gb, [env["HF_HOME"], env["HF_DATASETS_CACHE"], *[model["path"] for model in config["models"]]])

    model_ids = {
        "LLaMA3-8B": args.llama_model_id,
        "Qwen3-8B": args.qwen_model_id,
    }
    report = {
        "ok": False,
        "config": str(Path(args.config).expanduser()),
        "dry_run": args.dry_run,
        "metadata_only": args.metadata_only,
        "environment": {
            "HF_ENDPOINT": env["HF_ENDPOINT"],
            "HF_HOME": env["HF_HOME"],
            "HF_DATASETS_CACHE": env["HF_DATASETS_CACHE"],
            "TRANSFORMERS_CACHE": env["TRANSFORMERS_CACHE"],
        },
        "models": [],
        "files": [],
        "pruning_calibration": _prepare_pruning_calibration(defaults, env, args.dry_run),
        "lm_eval_probe": None,
    }
    source_roots = args.model_source_root or []
    for model in config["models"]:
        if args.only_model and model["name"] not in args.only_model:
            continue
        model_id = model_ids.get(model["name"], model["path"])
        source_checkpoint = None
        if args.link_existing_models:
            source_checkpoint = _find_existing_checkpoint(source_roots, model, model_id)
        if source_checkpoint is not None:
            result = _link_existing_checkpoint(source_checkpoint, Path(model["path"]), args.dry_run)
            result.update(
                {
                    "model_id": model_id,
                    "local_dir": str(Path(model["path"]).expanduser()),
                    "linked_existing_model": True,
                }
            )
        else:
            result = _download_model(model_id, model["path"], env, args.dry_run, args.metadata_only)
            if args.link_existing_models:
                result["linked_existing_model"] = False
                result["source_roots"] = source_roots
                result["source_candidates"] = [
                    str(path) for path in _model_source_candidates(source_roots, model, model_id)
                ]
        result["name"] = model["name"]
        report["models"].append(result)
        if not result["ok"] and not args.continue_on_error:
            report["ok"] = _report_ok(report)
            _write_report(report, args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(result["returncode"])

    for file_result in (
        _ensure_text_file(
            defaults["calibration_file"],
            calibration_seed_texts(calibration_samples),
            args.dry_run,
            min_lines=calibration_samples,
        ),
        _ensure_text_file(defaults["prompt_file"], DEFAULT_PROMPTS, args.dry_run),
    ):
        report["files"].append(file_result)
        if not file_result["ok"] and not args.continue_on_error:
            report["ok"] = _report_ok(report)
            _write_report(report, args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(1)

    if not args.skip_lm_eval_probe:
        report["lm_eval_probe"] = _run([sys.executable, "-m", "lm_eval", "--help"], env, args.dry_run)
        if not report["lm_eval_probe"]["ok"] and not args.continue_on_error:
            report["ok"] = _report_ok(report)
            _write_report(report, args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(report["lm_eval_probe"]["returncode"])
    report["ok"] = _report_ok(report)
    _write_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _check_free_disk(min_free_gb: float, paths) -> None:
    if min_free_gb <= 0:
        return
    checked = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        probe = path if path.exists() else path.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        free_gb = usage.free / (1024**3)
        checked[str(probe)] = free_gb
    too_low = {path: free for path, free in checked.items() if free < min_free_gb}
    if too_low:
        raise SystemExit(
            "Insufficient free disk for asset preparation: "
            + ", ".join(f"{path} has {free:.1f}GB" for path, free in too_low.items())
            + f"; required >= {min_free_gb:.1f}GB"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare models, cache paths and prompt files for LLMCBench edge runs.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--llama-model-id", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--only-model", action="append", default=[])
    parser.add_argument(
        "--model-source-root",
        action="append",
        default=[],
        help="Repeatable directory containing existing checkpoints named by model name or configured path basename.",
    )
    parser.add_argument(
        "--link-existing-models",
        action="store_true",
        help="Symlink complete checkpoints from --model-source-root into configured model paths before downloading.",
    )
    parser.add_argument("--metadata-only", action="store_true", help="Download only lightweight model metadata/tokenizer files.")
    parser.add_argument("--min-free-disk-gb", type=float, default=0.0, help="Fail early unless target filesystems have this much free disk. 0 disables the guard.")
    parser.add_argument("--output", default=None, help="Optional JSON report path for asset preparation status.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-lm-eval-probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    prepare(build_parser().parse_args())


if __name__ == "__main__":
    main()
