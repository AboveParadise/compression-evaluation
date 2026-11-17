import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REQUIRED_IMPORTS = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "lm_eval",
    "calflops",
]

OPTIONAL_IMPORTS = [
    "bitsandbytes",
    "auto_gptq",
    "awq",
]


def _load_config(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _filtered_config(config: Dict, only_model: List[str]) -> Dict:
    if not only_model:
        return config
    wanted = set(only_model)
    filtered = dict(config)
    filtered["models"] = [model for model in config.get("models", []) if model.get("name") in wanted]
    return filtered


def _check_import(module: str) -> Dict:
    try:
        loaded = importlib.import_module(module)
        return {"name": module, "ok": True, "version": getattr(loaded, "__version__", None)}
    except Exception as exc:
        return {"name": module, "ok": False, "error": str(exc)}


def _gpu_report(min_memory_gb: float) -> Dict:
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": str(exc), "gpus": []}
    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA is not available", "gpus": []}
    gpus: List[Dict] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        memory_gb = props.total_memory / (1024**3)
        gpus.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_gb": memory_gb,
                "meets_min_memory": memory_gb >= min_memory_gb,
            }
        )
    return {"ok": any(gpu["meets_min_memory"] for gpu in gpus), "gpus": gpus}


def _cuda_visible_devices_report(config: Dict, require_match: bool) -> Dict:
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    expected = defaults.get("cuda_visible_devices")
    actual = os.environ.get("CUDA_VISIBLE_DEVICES")
    matches = expected in (None, "") or actual == expected
    return {
        "ok": matches or not require_match,
        "actual": actual,
        "expected": expected,
        "matches_config": matches,
        "required": require_match,
    }


def _path_report(config: Dict) -> Dict:
    defaults = config["defaults"]
    models = []
    for model in config["models"]:
        path = Path(model["path"]).expanduser()
        weight_status = _weight_status(path)
        models.append(
            {
                "name": model["name"],
                "path": str(path),
                "exists": path.exists(),
                "has_config": (path / "config.json").exists(),
                "has_weights": weight_status["ok"],
                "weight_status": weight_status,
            }
        )
    return {
        "models": models,
        "calibration_file": {
            "path": defaults["calibration_file"],
            "exists": Path(defaults["calibration_file"]).expanduser().exists(),
        },
        "prompt_file": {
            "path": defaults["prompt_file"],
            "exists": Path(defaults["prompt_file"]).expanduser().exists(),
        },
        "data_dir": {
            "path": defaults["data_dir"],
            "exists": Path(defaults["data_dir"]).expanduser().exists(),
        },
        "output_root": defaults["output_root"],
    }


def _weight_status(path: Path) -> Dict:
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            expected = sorted(set(index.get("weight_map", {}).values()))
        except Exception as exc:
            return {"ok": False, "error": f"failed_to_read_index:{exc}"}
        missing = [name for name in expected if not (path / name).exists()]
        return {"ok": not missing and bool(expected), "expected": expected, "missing": missing}
    files = [file.name for pattern in ("*.safetensors", "*.bin", "*.pt") for file in path.glob(pattern)]
    return {"ok": bool(files), "expected": files, "missing": [] if files else ["weight_files"]}


def _resource_backend_report() -> Dict:
    calflops = _check_import("calflops")
    return {
        "ok": True,
        "calflops": calflops,
        "selected_backend": "calflops" if calflops["ok"] else "analytical_config_estimate",
        "note": (
            "calflops is available and will be tried first."
            if calflops["ok"]
            else "calflops is unavailable; resource evaluation will use the analytical config estimate fallback."
        ),
    }


def _hf_access_report(config: Dict) -> List[Dict]:
    endpoint = config["defaults"].get("hf_endpoint", "https://hf-mirror.com")
    env = os.environ.copy()
    env["HF_ENDPOINT"] = endpoint
    probes = {
        "LLaMA3-8B": "meta-llama/Meta-Llama-3-8B",
        "Qwen3-8B": "Qwen/Qwen3-8B",
    }
    results = []
    for model in config["models"]:
        repo_id = probes.get(model["name"], model["path"])
        proc = subprocess.run(
            [
                "huggingface-cli",
                "download",
                repo_id,
                "config.json",
                "--local-dir",
                f"/tmp/llmcbench_hf_probe/{model['name']}",
                "--local-dir-use-symlinks",
                "False",
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        results.append(
            {
                "name": model["name"],
                "repo_id": repo_id,
                "ok": proc.returncode == 0,
                "stdout_tail": proc.stdout[-1000:],
                "stderr_tail": proc.stderr[-2000:],
            }
        )
    return results


def _model_forward_report(config: Dict, dtype: str, attn_implementation: str) -> List[Dict]:
    results = []
    for model in config["models"]:
        model_path = Path(model["path"]).expanduser()
        if not (model_path / "config.json").exists() or not _weight_status(model_path)["ok"]:
            results.append(
                {
                    "name": model["name"],
                    "path": str(model_path),
                    "ok": False,
                    "skipped": True,
                    "reason": "complete_checkpoint_missing",
                }
            )
            continue
        dtype_expr = {
            "fp16": "torch.float16",
            "float16": "torch.float16",
            "bf16": "torch.bfloat16",
            "bfloat16": "torch.bfloat16",
            "fp32": "torch.float32",
            "float32": "torch.float32",
        }.get(dtype, "torch.float16")
        attn_arg = f",attn_implementation={attn_implementation!r}" if attn_implementation else ""
        script = (
            "import torch;"
            "from transformers import AutoModelForCausalLM,AutoTokenizer;"
            f"p={str(model_path)!r};"
            "tok=AutoTokenizer.from_pretrained(p,trust_remote_code=True);"
            f"m=AutoModelForCausalLM.from_pretrained(p,device_map='auto',torch_dtype={dtype_expr},trust_remote_code=True{attn_arg});"
            "x=tok('hello',return_tensors='pt').to(next(m.parameters()).device);"
            "y=m(**x,use_cache=False);"
            "print(tuple(y.logits.shape))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            check=False,
        )
        results.append(
            {
                "name": model["name"],
                "path": str(model_path),
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "dtype": dtype,
                "attn_implementation": attn_implementation or None,
                "stdout_tail": proc.stdout[-1000:],
                "stderr_tail": proc.stderr[-2000:],
            }
        )
    return results


def verify(args: argparse.Namespace) -> Dict:
    config = _filtered_config(_load_config(args.config), args.only_model)
    if not config.get("models"):
        raise SystemExit("No models selected.")
    report = {
        "gpu": _gpu_report(args.min_gpu_memory_gb),
        "cuda_visible_devices": _cuda_visible_devices_report(config, args.require_cuda_visible_devices_match),
        "required_imports": [_check_import(module) for module in REQUIRED_IMPORTS],
        "optional_imports": [_check_import(module) for module in OPTIONAL_IMPORTS],
        "paths": _path_report(config),
    }
    report["ok"] = (
        report["gpu"]["ok"]
        and report["cuda_visible_devices"]["ok"]
        and all(item["ok"] for item in report["required_imports"])
        and all(model["has_config"] for model in report["paths"]["models"])
        and report["paths"]["calibration_file"]["exists"]
        and report["paths"]["prompt_file"]["exists"]
    )
    if args.check_cli:
        proc = subprocess.run(
            [sys.executable, "-m", "edge_benchmark", "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        report["cli"] = {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }
        report["ok"] = report["ok"] and report["cli"]["ok"]
    if args.check_hf_access:
        report["hf_access"] = _hf_access_report(config)
        report["ok"] = report["ok"] and all(item["ok"] for item in report["hf_access"])
    if args.check_resource_backend:
        report["resource_backend"] = _resource_backend_report()
        report["ok"] = report["ok"] and report["resource_backend"]["ok"]
    if args.check_model_forward:
        defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
        forward_dtype = args.forward_dtype or defaults.get("dtype") or "fp16"
        forward_attn_implementation = (
            args.forward_attn_implementation
            if args.forward_attn_implementation not in (None, "")
            else defaults.get("attn_implementation", "")
        )
        report["model_forward"] = _model_forward_report(config, forward_dtype, forward_attn_implementation)
        report["ok"] = report["ok"] and all(item["ok"] for item in report["model_forward"])
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the LLMCBench edge runtime environment.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--only-model", action="append", default=[], help="Repeatable model name filter.")
    parser.add_argument("--min-gpu-memory-gb", type=float, default=24.0)
    parser.add_argument("--check-cli", action="store_true")
    parser.add_argument("--check-hf-access", action="store_true")
    parser.add_argument("--check-resource-backend", action="store_true")
    parser.add_argument("--check-model-forward", action="store_true")
    parser.add_argument("--require-cuda-visible-devices-match", action="store_true", help="Require CUDA_VISIBLE_DEVICES to match config defaults.cuda_visible_devices.")
    parser.add_argument("--forward-dtype", default=None, help="Dtype for --check-model-forward. Defaults to config defaults.dtype, then fp16.")
    parser.add_argument("--forward-attn-implementation", default="", help="Optional Transformers attention backend for --check-model-forward, for example eager.")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
