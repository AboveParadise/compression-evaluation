import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


QUANT_METHODS = {"gptq", "awq", "smoothquant", "omniquant"}


def _load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected(items: List[str], filters: List[str]) -> List[str]:
    if not filters:
        return items
    wanted = set(filters)
    return [item for item in items if item in wanted]


def _precompressed_path(config: Dict, model_name: str, method: str) -> str:
    paths = config.get("precompressed_paths", {})
    if not isinstance(paths, dict):
        return ""
    model_paths = paths.get(model_name, {})
    if not isinstance(model_paths, dict):
        return ""
    return model_paths.get(method, "") or ""


def _probe_script(model_path: str, method: str, args: argparse.Namespace) -> str:
    return f"""
import importlib
import importlib.metadata
import json
import os
import sys

result = {{
    "ok": False,
    "method": {method!r},
    "model_path": {model_path!r},
    "imports": {{}},
    "steps": [],
}}

def step(name, ok, **extra):
    item = {{"name": name, "ok": bool(ok)}}
    item.update(extra)
    result["steps"].append(item)
    return ok

try:
    import torch
    import transformers
    from edge_benchmark.runtime_compat import apply_runtime_compatibility_patches
    apply_runtime_compatibility_patches()
    result["python_executable"] = sys.executable
    result["python_version"] = sys.version
    result["pythonpath"] = os.environ.get("PYTHONPATH")
    result["runtime_profile"] = os.environ.get("EDGE_RUNTIME_PROFILE", "stable")
    result["torch"] = getattr(torch, "__version__", None)
    result["transformers"] = getattr(transformers, "__version__", None)
    result["cuda_available"] = torch.cuda.is_available()
    result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    for name in ["optimum", "auto_gptq", "gptqmodel", "awq", "accelerate", "triton", "compressed_tensors"]:
        try:
            module = importlib.import_module(name)
            try:
                package_version = importlib.metadata.version(name)
            except Exception:
                package_version = getattr(module, "__version__", None)
            spec = getattr(module, "__spec__", None)
            search_locations = getattr(spec, "submodule_search_locations", None) if spec is not None else None
            result["imports"][name] = {{
                "ok": True,
                "version": package_version,
                "module_version": getattr(module, "__version__", None),
                "file": getattr(module, "__file__", None),
                "origin": getattr(spec, "origin", None) if spec is not None else None,
                "search_locations": list(search_locations) if search_locations is not None else None,
            }}
        except Exception as exc:
            result["imports"][name] = {{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}}

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained({model_path!r}, trust_remote_code=True)
    result["quantization_config"] = getattr(cfg, "quantization_config", None)
    quantization_config = result["quantization_config"] if isinstance(result.get("quantization_config"), dict) else {{}}
    quantization_meta = quantization_config.get("meta") if isinstance(quantization_config.get("meta"), dict) else {{}}
    quantizers = quantization_meta.get("quantizer") if isinstance(quantization_meta.get("quantizer"), list) else []
    result["quantization_metadata"] = {{
        "quant_method": quantization_config.get("quant_method"),
        "checkpoint_format": quantization_config.get("checkpoint_format"),
        "bits": quantization_config.get("bits"),
        "group_size": quantization_config.get("group_size"),
        "quantizer": quantizers,
    }}
    if {method!r} == "gptq":
        if any(str(item).lower().startswith("gptqmodel") for item in quantizers):
            result["runtime_requirements_reason"] = "gptqmodel_checkpoint_metadata"
            step("runtime_imports", result["imports"].get("optimum", {{}}).get("ok") and result["imports"].get("gptqmodel", {{}}).get("ok"), required_imports=["optimum", "gptqmodel"])
        else:
            result["runtime_requirements_reason"] = "default"
            step("runtime_imports", result["imports"].get("optimum", {{}}).get("ok") and (result["imports"].get("auto_gptq", {{}}).get("ok") or result["imports"].get("gptqmodel", {{}}).get("ok")), required_imports=["optimum", "auto_gptq|gptqmodel"])
    elif {method!r} == "awq":
        result["runtime_requirements_reason"] = "default"
        step("runtime_imports", result["imports"].get("awq", {{}}).get("ok") and result["imports"].get("accelerate", {{}}).get("ok"), required_imports=["awq", "accelerate"])
    elif {method!r} == "smoothquant":
        result["runtime_requirements_reason"] = "default"
        step("runtime_imports", result["imports"].get("compressed_tensors", {{}}).get("ok"), required_imports=["compressed_tensors"])
    else:
        result["runtime_requirements_reason"] = "default"
        step("runtime_imports", True)
    if result["steps"] and result["steps"][-1]["name"] == "runtime_imports" and result["steps"][-1]["ok"] is not True:
        missing = [
            name
            for name in result["steps"][-1].get("required_imports", [])
            if "|" not in str(name) and result["imports"].get(str(name), {{}}).get("ok") is not True
        ]
        raise RuntimeError(f"missing required runtime imports: {{missing or result['steps'][-1].get('required_imports', [])}}")
    step("config", True, model_type=getattr(cfg, "model_type", None))
    tok = AutoTokenizer.from_pretrained({model_path!r}, trust_remote_code=True, use_fast=False)
    step("tokenizer", True, tokenizer=type(tok).__name__)
    dtype = torch.float16 if {args.dtype!r} in ("fp16", "float16") else "auto"
    max_memory = {{0: {args.gpu_max_memory!r}, "cpu": {args.cpu_max_memory!r}}} if {bool(args.gpu_max_memory)!r} else None
    kwargs = {{
        "trust_remote_code": True,
        "device_map": {args.device_map!r},
        "torch_dtype": dtype,
    }}
    if max_memory:
        kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained({model_path!r}, **kwargs)
    device = next(model.parameters()).device
    step("model_load", True, model=type(model).__name__, first_device=str(device))
    if {bool(args.generate)!r}:
        encoded = tok({args.prompt!r}, return_tensors="pt").to(device)
        generated = model.generate(**encoded, max_new_tokens={args.max_new_tokens}, do_sample=False, pad_token_id=tok.eos_token_id)
        step("generate", True, shape=list(generated.shape))
    result["ok"] = all(item["ok"] for item in result["steps"])
except Exception as exc:
    result["error"] = f"{{type(exc).__name__}}: {{exc}}"
    result["ok"] = False
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["ok"] else 1)
"""


def _run_probe(model_name: str, method: str, model_path: str, args: argparse.Namespace) -> Dict:
    env = os.environ.copy()
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint
    proc = subprocess.run(
        [args.python, "-c", _probe_script(model_path, method, args)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=args.timeout_seconds,
    )
    payload = None
    marker = proc.stdout.rfind("\n{")
    text = proc.stdout[marker + 1 :] if marker >= 0 else proc.stdout
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    return {
        "model": model_name,
        "method": method,
        "model_path": model_path,
        "ok": proc.returncode == 0 and isinstance(payload, dict) and payload.get("ok") is True,
        "returncode": proc.returncode,
        "payload": payload,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def probe(args: argparse.Namespace) -> Dict:
    config = _load_config(Path(args.config).expanduser())
    models = [
        model for model in config.get("models", [])
        if not args.only_model or model.get("name") in set(args.only_model)
    ]
    methods = _selected([method for method in config.get("methods", []) if method in QUANT_METHODS], args.only_method)
    rows = []
    for model in models:
        model_name = model.get("name")
        for method in methods:
            path = _precompressed_path(config, model_name, method)
            if not path:
                rows.append(
                    {
                        "model": model_name,
                        "method": method,
                        "ok": False,
                        "skipped": True,
                        "reason": "precompressed_path_not_configured",
                    }
                )
                continue
            rows.append(_run_probe(model_name, method, path, args))
    report = {"ok": all(row.get("ok") for row in rows), "rows": rows}
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"] and args.fail_on_error:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe quantized precompressed checkpoint runtime loading.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--only-model", action="append", default=[])
    parser.add_argument("--only-method", action="append", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pythonpath", default=os.environ.get("PYTHONPATH", ""))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--gpu-max-memory", default="22GiB")
    parser.add_argument("--cpu-max-memory", default="120GiB")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def main() -> None:
    probe(build_parser().parse_args())


if __name__ == "__main__":
    main()
