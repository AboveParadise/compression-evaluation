"""One-command compression launcher.

Example:
    python -m compression --base-model /data/models/Qwen3-8B --method awq
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List

from .profiles import METHODS, default_setting, method_arguments, setting_label


def _safe_name(value: str) -> str:
    name = Path(value.rstrip("/")).name or value
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "model"


def _infer_model_name(base_model: str) -> str:
    config_path = Path(base_model).expanduser() / "config.json"
    if not config_path.is_file():
        return "unknown"
    try:
        model_type = str(json.loads(config_path.read_text(encoding="utf-8")).get("model_type", "")).lower()
    except (OSError, json.JSONDecodeError):
        return "unknown"
    if model_type == "llama":
        return "LLaMA3-8B"
    if model_type == "qwen3":
        return "Qwen3-8B"
    return "unknown"


def build_command(args: argparse.Namespace) -> List[str]:
    setting = args.setting or default_setting(args.method)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        Path(args.output_root).expanduser() / f"{_safe_name(args.base_model)}-{args.method}"
    )
    model_name = args.model_name or _infer_model_name(args.base_model)
    command = [
        sys.executable,
        "-m",
        "edge_benchmark",
        "compress",
        "--model-path",
        args.base_model,
        "--model-name",
        model_name,
        "--compression-method",
        args.method,
        "--compressed-output-dir",
        str(output_dir),
        "--cache-dir",
        args.cache_dir,
        "--hf-endpoint",
        args.hf_endpoint,
        "--device",
        args.device,
        "--device-map",
        args.device_map,
        "--dtype",
        args.dtype,
        "--max-memory",
        args.max_memory,
        "--min-free-gpu-memory-gb",
        str(args.min_free_gpu_memory_gb),
        "--calibration-dataset-cache",
        args.dataset_cache,
        *method_arguments(args.method, setting),
    ]
    if args.overwrite:
        command.append("--overwrite-compressed")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress a local HF base model with one method-specific reproducible preset."
    )
    parser.add_argument("--base-model", required=True, help="Local HuggingFace base-model directory.")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument(
        "--setting",
        choices=("paper", "adapted"),
        default=None,
        help="Defaults to paper for Wanda/SparseGPT/GPTQ/AWQ and adapted for SmoothQuant/OmniQuant.",
    )
    parser.add_argument("--output-dir", default="", help="Compressed checkpoint directory.")
    parser.add_argument("--output-root", default="outputs/compressed", help="Parent when --output-dir is omitted.")
    parser.add_argument("--model-name", choices=("LLaMA3-8B", "Qwen3-8B", "unknown"), default="")
    parser.add_argument("--cache-dir", default="/data/hf_cache")
    parser.add_argument("--dataset-cache", default="/data/llm_eval_datasets")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--max-memory", default="0:22GiB,cpu:120GiB")
    parser.add_argument("--min-free-gpu-memory-gb", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setting = args.setting or default_setting(args.method)
    command = build_command(args)
    payload = {
        "base_model": args.base_model,
        "method": args.method,
        "setting": setting,
        "setting_label": setting_label(args.method, setting),
        "command": command,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()

