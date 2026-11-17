import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from verify_edge_env import _weight_status


DEFAULT_REPOS = {
    ("LLaMA3-8B", "gptq"): "Ksgk-fy/Meta-Llama-3-8B-4bit-GPTQ",
    ("LLaMA3-8B", "awq"): "solidrust/Meta-Llama-3-8B-hf-AWQ",
    ("LLaMA3-8B", "smoothquant"): "HRuiii/Meta-Llama-3-8B-smoothquant-w8a8",
    ("Qwen3-8B", "gptq"): "kaitchup/Qwen3-8B-autoround-4bit-gptq",
    ("Qwen3-8B", "awq"): "Qwen/Qwen3-8B-AWQ",
    ("Qwen3-8B", "smoothquant"): "Lotaliz/Qwen3-8B-smoothquant-w8a8",
    ("Qwen3-8B", "omniquant"): "yw223/Qwen3-8B-OmniQuant-4bit",
}

MISSING_DEFAULT_REPOS = {
    ("LLaMA3-8B", "omniquant"): {
        "reason": "no_verified_public_default_repo",
        "detail": (
            "The official OpenGVLab/OmniQuant repository does not support LLaMA3, and no verified "
            "public OmniQuant checkpoint based on meta-llama/Meta-Llama-3-8B is available as a default."
        ),
        "remediation": (
            "Generate W4A16g128 from the exact base model with the LLaMA-V3-capable ModelTC/llmc "
            "OmniQuant implementation, provide a verified checkpoint with "
            "--repo LLaMA3-8B:omniquant=<repo_id>, or configure "
            "LLMCBENCH_OMNIQUANT_CMD/external_commands.omniquant."
        ),
    },
}

METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "quantize_config.json",
    "awq_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
]

PROVENANCE_FILE = ".llmcbench_precompressed_source.json"


def _load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected(items: List[str], filters: List[str]) -> List[str]:
    if not filters:
        return items
    wanted = set(filters)
    return [item for item in items if item in wanted]


def _parse_repo_overrides(values: List[str]) -> Dict[Tuple[str, str], str]:
    overrides = {}
    for value in values:
        left, sep, repo = value.partition("=")
        if not sep:
            raise SystemExit(f"Invalid --repo override {value!r}; expected MODEL:METHOD=repo_id.")
        model, sep, method = left.partition(":")
        if not sep or not model or not method or not repo:
            raise SystemExit(f"Invalid --repo override {value!r}; expected MODEL:METHOD=repo_id.")
        overrides[(model, method)] = repo
    return overrides


def _run_download(repo: str, output_dir: Path, env: Dict[str, str], dry_run: bool, metadata_only: bool) -> Dict:
    cmd = ["huggingface-cli", "download", repo]
    if metadata_only:
        cmd.extend(METADATA_FILES)
    cmd.extend(["--local-dir", str(output_dir), "--local-dir-use-symlinks", "False"])
    print("+ " + " ".join(cmd))
    if dry_run:
        return {"ok": True, "returncode": 0, "cmd": cmd, "dry_run": True}
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    ok = proc.returncode == 0
    if metadata_only and not ok:
        # Some repos do not carry every optional metadata filename. For metadata
        # preflight, config.json plus any available tokenizer/index files are enough.
        ok = (output_dir / "config.json").exists()
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "cmd": cmd,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _provenance_path(path: Path) -> Path:
    return path / PROVENANCE_FILE


def _load_provenance(path: Path) -> Dict:
    marker = _provenance_path(path)
    if not marker.exists():
        return {"exists": False, "path": str(marker)}
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"exists": True, "path": str(marker), "ok": False, "error": str(exc)}
    data.setdefault("exists", True)
    data.setdefault("path", str(marker))
    data.setdefault("ok", True)
    return data


def _write_provenance(
    path: Path,
    *,
    model_name: str,
    method: str,
    repo: str,
    hf_endpoint: str,
    metadata_only: bool,
) -> Dict:
    marker = _provenance_path(path)
    payload = {
        "schema_version": 1,
        "model": model_name,
        "method": method,
        "repo": repo,
        "hf_endpoint": hf_endpoint,
        "metadata_only": metadata_only,
    }
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _provenance_matches(provenance: Dict, repo: str) -> bool:
    return bool(provenance.get("ok", True) and provenance.get("repo") == repo)


def _read_quantization_metadata(path: Path) -> Dict:
    metadata = {
        "quant_method": None,
        "checkpoint_format": None,
        "bits": None,
        "group_size": None,
        "quantizer": [],
    }
    for name in ("quantize_config.json", "config.json"):
        file_path = path / name
        if not file_path.exists():
            continue
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        quant = payload.get("quantization_config") if isinstance(payload, dict) else None
        if not isinstance(quant, dict):
            quant = payload if isinstance(payload, dict) else {}
        metadata["quant_method"] = quant.get("quant_method", metadata["quant_method"])
        metadata["checkpoint_format"] = quant.get("checkpoint_format", metadata["checkpoint_format"])
        metadata["bits"] = quant.get("bits", metadata["bits"])
        metadata["group_size"] = quant.get("group_size", metadata["group_size"])
        meta = quant.get("meta") if isinstance(quant.get("meta"), dict) else {}
        quantizer = quant.get("quantizer") or meta.get("quantizer") or []
        if isinstance(quantizer, str):
            quantizer = [quantizer]
        if quantizer:
            metadata["quantizer"] = list(quantizer)
        break
    return metadata


def _stale_default_checkpoint_reason(model_name: str, method: str, repo: str, path: Path) -> Optional[Dict]:
    if (model_name, method) != ("Qwen3-8B", "gptq"):
        return None
    if repo != DEFAULT_REPOS[("Qwen3-8B", "gptq")]:
        return None
    metadata = _read_quantization_metadata(path)
    quantizers = [str(item).lower() for item in metadata.get("quantizer", [])]
    if any("gptqmodel" in item for item in quantizers):
        return {
            "reason": "checkpoint_metadata_mismatch",
            "expected_repo": repo,
            "detail": "Existing Qwen3-8B GPTQ checkpoint advertises GPTQModel metadata, but the default repo now uses the stable AutoGPTQ metadata path.",
            "quantization_metadata": metadata,
            "remediation": "Rerun with --force, or stage this method under a clean --local-root.",
        }
    return None


def _checkpoint_ready(path: Path) -> Dict:
    config_exists = (path / "config.json").exists()
    weight_status = _weight_status(path)
    return {
        "config_exists": config_exists,
        "weights_exist": weight_status["ok"],
        "weight_status": weight_status,
        "checkpoint_ready": config_exists and weight_status["ok"],
    }


def _apply_precompressed_path(config: Dict, model_name: str, method: str, path: Path) -> None:
    precompressed = config.setdefault("precompressed_paths", {})
    model_paths = precompressed.setdefault(model_name, {})
    model_paths[method] = str(path)


def prepare(args: argparse.Namespace) -> Dict:
    config_path = Path(args.config).expanduser()
    config = _load_config(config_path)
    methods = _selected(config.get("methods", []), args.only_method)
    models = [item for item in config.get("models", []) if not args.only_model or item.get("name") in set(args.only_model)]
    repo_map = dict(DEFAULT_REPOS)
    repo_map.update(_parse_repo_overrides(args.repo))
    env = os.environ.copy()
    endpoint = args.hf_endpoint or config.get("defaults", {}).get("hf_endpoint") or "https://hf-mirror.com"
    env["HF_ENDPOINT"] = endpoint
    if args.hf_home:
        env["HF_HOME"] = str(Path(args.hf_home).expanduser())

    output_root = Path(args.local_root).expanduser()
    report = {
        "ok": False,
        "config": str(config_path),
        "dry_run": args.dry_run,
        "metadata_only": args.metadata_only,
        "hf_endpoint": endpoint,
        "local_root": str(output_root),
        "selected_models": [model.get("name") for model in models],
        "selected_methods": methods,
        "items": [],
        "blockers": [],
        "updated_config": None,
    }
    updated_config = json.loads(json.dumps(config))
    for model in models:
        model_name = model.get("name")
        for method in methods:
            if method in {"dense", "wanda", "sparsegpt"}:
                continue
            repo = repo_map.get((model_name, method))
            if not repo:
                missing_default = MISSING_DEFAULT_REPOS.get((model_name, method), {})
                item = {
                    "model": model_name,
                    "method": method,
                    "ok": False,
                    "skipped": True,
                    "blocking": True,
                    "reason": "repo_not_configured",
                    "repo_status": missing_default.get("reason", "default_repo_not_configured"),
                    "detail": missing_default.get("detail"),
                    "remediation": missing_default.get(
                        "remediation",
                        f"Provide a checkpoint repo with --repo {model_name}:{method}=<repo_id>.",
                    ),
                }
                report["items"].append(item)
                report["blockers"].append(item)
                continue
            local_dir = output_root / f"{model_name}-{method}"
            before = _checkpoint_ready(local_dir)
            before_provenance = _load_provenance(local_dir)
            provenance_mismatch = (
                before["checkpoint_ready"]
                and not args.force
                and before_provenance.get("exists")
                and not _provenance_matches(before_provenance, repo)
            )
            stale_checkpoint = None
            if before["checkpoint_ready"] and not args.force and not provenance_mismatch:
                stale_checkpoint = _stale_default_checkpoint_reason(model_name, method, repo, local_dir)
            if provenance_mismatch:
                download = {
                    "ok": False,
                    "returncode": None,
                    "skipped": True,
                    "reason": "checkpoint_source_mismatch",
                    "expected_repo": repo,
                    "existing_repo": before_provenance.get("repo"),
                    "remediation": "Rerun with --force, or stage this method under a clean --local-root.",
                }
            elif stale_checkpoint:
                download = {
                    "ok": False,
                    "returncode": None,
                    "skipped": True,
                    **stale_checkpoint,
                }
            elif before["checkpoint_ready"] and not args.force:
                download = {"ok": True, "returncode": 0, "skipped": True, "reason": "checkpoint_already_ready"}
            else:
                download = _run_download(repo, local_dir, env, args.dry_run, args.metadata_only)
            provenance_after = before_provenance
            if bool(download.get("ok")) and not args.dry_run:
                provenance_after = _write_provenance(
                    local_dir,
                    model_name=model_name,
                    method=method,
                    repo=repo,
                    hf_endpoint=endpoint,
                    metadata_only=args.metadata_only,
                )
            after = before if args.dry_run else _checkpoint_ready(local_dir)
            _apply_precompressed_path(updated_config, model_name, method, local_dir)
            report["items"].append(
                {
                    "model": model_name,
                    "method": method,
                    "repo": repo,
                    "local_dir": str(local_dir),
                    "download": download,
                    "before": before,
                    "before_provenance": before_provenance,
                    "after": after,
                    "provenance": provenance_after,
                    "ok": bool(download.get("ok")) and (args.dry_run or args.metadata_only or after["checkpoint_ready"]),
                }
            )
            if not report["items"][-1]["ok"]:
                report["blockers"].append(report["items"][-1])
    if args.output_config:
        output_config = Path(args.output_config).expanduser()
        report["updated_config"] = str(output_config)
        if not args.dry_run or args.write_dry_run_config:
            output_config.parent.mkdir(parents=True, exist_ok=True)
            output_config.write_text(json.dumps(updated_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["ok"] = bool(report["items"]) and not report["blockers"]
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"] and args.fail_on_error:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare precompressed checkpoints and precompressed_paths config entries.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--local-root", default="/data/models/precompressed")
    parser.add_argument("--output-config", default=None, help="Optional path for a config copy with precompressed_paths updated.")
    parser.add_argument(
        "--write-dry-run-config",
        action="store_true",
        help="With --dry-run, still write the planned output config so wrapper dry-runs can exercise precompressed_paths.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--only-model", action="append", default=[])
    parser.add_argument("--only-method", action="append", default=[])
    parser.add_argument("--repo", action="append", default=[], help="Override repo as MODEL:METHOD=repo_id.")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--metadata-only", action="store_true", help="Download only metadata/tokenizer/index files for access checks.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def main() -> None:
    prepare(build_parser().parse_args())


if __name__ == "__main__":
    main()
