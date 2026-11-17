import argparse
import importlib
import inspect
import platform
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .adapted_quantization import adapted_checkpoint_metadata, is_adapted_quantization_checkpoint, load_adapted_model
from .checkpoint_inspection import inspect_quantization_storage
from .compression import run_compression
from .runtime_compat import apply_runtime_compatibility_patches


apply_runtime_compatibility_patches()


SUPPORTED_MODELS = ("LLaMA3-8B", "Qwen3-8B")
SUPPORTED_METHODS = (
    "dense",
    "wanda",
    "sparsegpt",
    "gptq",
    "awq",
    "smoothquant",
    "omniquant",
)
DEFAULT_ACCURACY_TASKS = ("mmlu", "arc_challenge", "hellaswag", "piqa", "winogrande")
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_MIN_FREE_GPU_MEMORY_GB = 20.0
DEFAULT_DEPLOYMENT_MEMORY_LIMIT_GB = 24.0
DEFAULT_DEPLOYMENT_BATCH_SIZE = 1
DEFAULT_CUDA_MAX_MEMORY = "0:22GiB,cpu:120GiB"


def _split_csv(values: Iterable[str]) -> List[str]:
    items: List[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                items.append(item)
    return items


def _ensure_parent(path: Optional[str]) -> None:
    if path:
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _write_json(payload: Dict, output: Optional[str]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        _ensure_parent(output)
        Path(output).write_text(text + "\n", encoding="utf-8")
    print(text)


def _configure_hf_cache(cache_dir: Optional[str], data_dir: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    if cache_dir:
        cache = str(Path(cache_dir).expanduser())
        env.setdefault("HF_HOME", cache)
        env.setdefault("TRANSFORMERS_CACHE", str(Path(cache) / "transformers"))
        env.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(cache) / "hub"))
    if data_dir:
        env.setdefault("HF_DATASETS_CACHE", str(Path(data_dir).expanduser()))
    return env


def _model_metadata(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "revision": args.revision,
        "base_model_repository": args.base_model_repository or None,
        "base_model_revision": args.base_model_revision or None,
        "base_model_attestation_path": args.base_model_attestation_path or None,
        "compression_method": args.compression_method,
    }


def _precompressed_paper_protocol(path_value: str, expected_protocol: str) -> Dict:
    manifest_path = Path(path_value).expanduser() / "llmcbench_compression_manifest.json"
    status = {
        "manifest_path": str(manifest_path),
        "exists": manifest_path.is_file(),
        "expected_protocol": expected_protocol,
        "paper_protocol": None,
        "attested": False,
        "manifest": None,
    }
    if not manifest_path.is_file():
        return status
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        return status
    status["paper_protocol"] = manifest.get("paper_protocol")
    status["attested"] = manifest.get("paper_protocol_attested") is True
    status["matches"] = status["paper_protocol"] == expected_protocol
    status["manifest"] = manifest
    return status


def _system_memory_total_gb() -> Optional[float]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) / (1024**2)
    return None


def _runtime_metadata() -> Dict:
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                }
            )
    optional_imports = {}
    for module_name in ("bitsandbytes",):
        try:
            module = importlib.import_module(module_name)
            optional_imports[module_name] = {
                "ok": True,
                "version": getattr(module, "__version__", None),
            }
        except Exception as exc:
            optional_imports[module_name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "system_memory_total_gb": _system_memory_total_gb(),
        "runtime_profile": os.environ.get("EDGE_RUNTIME_PROFILE", "stable"),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT),
        "hf_home": os.environ.get("HF_HOME"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
        "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
        "torch": getattr(torch, "__version__", None),
        "transformers": getattr(transformers, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.version() is not None else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpus,
        "optional_imports": optional_imports,
    }


def _check_gpu_memory_budget(args: argparse.Namespace) -> None:
    required = getattr(args, "min_free_gpu_memory_gb", 0.0) or 0.0
    if required <= 0 or not torch.cuda.is_available():
        return
    best_free = 0.0
    best_total = 0.0
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        best_free = max(best_free, free_bytes / (1024**3))
        best_total = max(best_total, total_bytes / (1024**3))
    if best_free < required:
        raise RuntimeError(
            f"Insufficient free GPU memory: best_free={best_free:.2f}GB, "
            f"best_total={best_total:.2f}GB, required_free={required:.2f}GB. "
            "Lower sequence/batch settings, free the GPU, or set --min-free-gpu-memory-gb 0 to disable this guard."
        )


def _apply_cuda_default_max_memory(args: argparse.Namespace) -> None:
    if getattr(args, "max_memory", None) or not torch.cuda.is_available():
        return
    args.max_memory = DEFAULT_CUDA_MAX_MEMORY


def run_compress(args: argparse.Namespace) -> None:
    _apply_cuda_default_max_memory(args)
    _check_gpu_memory_budget(args)
    manifest = run_compression(args)
    payload = {
        "benchmark": "edge_compression",
        **_model_metadata(args),
        "runtime": _runtime_metadata(),
        "compressed_model_path": manifest["compressed_model_path"],
        "manifest": manifest,
    }
    _write_json(payload, args.output)


def run_accuracy(args: argparse.Namespace) -> None:
    _apply_cuda_default_max_memory(args)
    _check_gpu_memory_budget(args)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    tasks = _split_csv(args.tasks) if args.tasks else list(DEFAULT_ACCURACY_TASKS)
    lm_eval_dtype = _lm_eval_dtype(args.dtype)
    adapted_checkpoint = is_adapted_quantization_checkpoint(args.model_path)
    model_args = [
        f"pretrained={args.model_path}",
        "trust_remote_code=True",
        f"dtype={lm_eval_dtype}",
    ]
    if args.revision:
        model_args.append(f"revision={args.revision}")
    if args.cache_dir:
        model_args.append(f"cache_dir={Path(args.cache_dir).expanduser()}")
    if args.max_memory:
        model_args.extend(_lm_eval_memory_args(args.max_memory))
    if args.attn_implementation:
        model_args.append(f"attn_implementation={args.attn_implementation}")

    cmd = [
        sys.executable,
        "-m",
        "edge_benchmark.lm_eval_compat",
        "--model",
        "hf",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        args.batch_size,
        "--output_path",
        args.lm_eval_output,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(args.num_fewshot)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    _ensure_parent(args.lm_eval_output)
    env = _configure_hf_cache(args.cache_dir, args.data_dir)
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint

    started_at = time.time()
    accuracy_backend = "lm_eval_subprocess"
    subprocess_error = None
    if adapted_checkpoint:
        results = _run_lm_eval_inprocess(args, tasks)
        elapsed = time.time() - started_at
        accuracy_backend = "lm_eval_inprocess_adapted_dequantized_model"
    else:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            capture_output=not args.stream_output,
            check=False,
        )
        elapsed = time.time() - started_at
        if proc.returncode != 0:
            subprocess_error = {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            try:
                started_at = time.time()
                results = _run_lm_eval_inprocess(args, tasks)
                elapsed = time.time() - started_at
                accuracy_backend = "lm_eval_inprocess_preloaded_model"
            except Exception as fallback_exc:
                if proc.stdout:
                    sys.stderr.write(proc.stdout)
                if proc.stderr:
                    sys.stderr.write(proc.stderr)
                sys.stderr.write(
                    "In-process lm-eval fallback failed: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}\n"
                )
                raise SystemExit(proc.returncode)
        else:
            results = _load_lm_eval_results(args.lm_eval_output)

    payload = {
        "benchmark": "edge_accuracy",
        **_model_metadata(args),
        "runtime": _runtime_metadata(),
        "tasks": tasks,
        "dtype": args.dtype,
        "lm_eval_dtype": lm_eval_dtype,
        "batch_size": args.batch_size,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "device": args.device,
        "data_dir": str(Path(args.data_dir).expanduser()) if args.data_dir else None,
        "hf_datasets_cache": env.get("HF_DATASETS_CACHE"),
        "max_memory": args.max_memory,
        "attn_implementation": args.attn_implementation or None,
        "elapsed_seconds": elapsed,
        "lm_eval_output": args.lm_eval_output,
        "accuracy_backend": accuracy_backend,
        "subprocess_error": subprocess_error,
        "results": _extract_accuracy_results(results),
    }
    _write_json(payload, args.output)


def _run_lm_eval_inprocess(args: argparse.Namespace, tasks: List[str]) -> Dict:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    tokenizer = _load_tokenizer(args.model_path, args.cache_dir, args.revision)
    model = _load_model(args)
    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
        dtype=_lm_eval_dtype(args.dtype),
        trust_remote_code=True,
    )
    raw = evaluator.simple_evaluate(
        model=hflm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        log_samples=False,
    )
    output_dir = Path(args.lm_eval_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return raw


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _lm_eval_dtype(dtype: str) -> str:
    mapping = {
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
    }
    return mapping.get(dtype, dtype)


def _lm_eval_memory_args(value: Optional[str]) -> List[str]:
    if not value:
        return []
    if "," not in value and ":" not in value:
        return [f"max_memory_per_gpu={value}"]
    parsed = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, memory = item.split(":", 1)
        parsed[key.strip()] = memory.strip()
    args = []
    gpu_memory = parsed.get("0") or next((memory for key, memory in parsed.items() if key.isdigit()), "")
    if gpu_memory:
        args.append(f"max_memory_per_gpu={gpu_memory}")
    if parsed.get("cpu"):
        args.append(f"max_cpu_memory={parsed['cpu']}")
    return args


def run_pipeline(args: argparse.Namespace) -> None:
    _apply_cuda_default_max_memory(args)
    _check_gpu_memory_budget(args)
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    compressed_dir = Path(args.compressed_output_dir).expanduser() if args.compressed_output_dir else (
        output_root / "compressed" / f"{args.model_name}_{args.compression_method}"
    )
    args.compressed_output_dir = str(compressed_dir)

    if args.use_precompressed:
        paper_protocol = getattr(args, "paper_protocol", "")
        paper_attestation = _precompressed_paper_protocol(args.model_path, paper_protocol)
        if getattr(args, "require_paper_protocol_attestation", False) and (
            not paper_protocol
            or paper_attestation.get("attested") is not True
            or paper_attestation.get("matches") is not True
        ):
            raise RuntimeError(
                "Paper-aligned precompressed checkpoint requires "
                "llmcbench_compression_manifest.json with matching paper_protocol and "
                "paper_protocol_attested=true. "
                f"Expected {paper_protocol!r}; status={paper_attestation}."
            )
        source_manifest = paper_attestation.get("manifest") or {}
        protocol_fields = {
            key: source_manifest[key]
            for key in (
                "quant_bits",
                "group_size",
                "desc_act",
                "awq_version",
                "sparsity",
                "sparsity_semantics",
                "prune_pattern",
                "prune_structure",
                "prune_n",
                "prune_m",
                "prune_lm_head",
                "pruning_algorithm",
                "pruning_buffer_device",
                "sparsegpt_blocksize",
                "sparsegpt_percdamp",
                "calibration_samples",
                "calibration_file",
                "calibration_batch_size",
                "calibration_protocol",
                "calibration_source",
                "calibration_sampling_algorithm",
                "calibration_seed",
                "calibration_dataset",
                "calibration_dataset_config",
                "calibration_data_file",
                "calibration_dataset_revision",
                "calibration_local_file",
                "calibration_token_length",
                "calibration_token_sha256",
                "gptq_damp_percent",
                "gptq_static_groups",
                "gptq_symmetric",
                "gptq_true_sequential",
                "awq_zero_point",
                "awq_calibration_source",
                "awq_calibration_seed",
                "awq_grid_size",
                "awq_runtime_scale_search_grid",
                "awq_runtime_calibration_seed",
                "quantization_protocol",
                "quantization_source",
                "quantization_dataset",
                "quantization_dataset_split",
                "quantization_seed",
                "quantization_samples",
                "quantization_token_length",
                "smoothquant_alpha",
                "smoothquant_alpha_selection",
                "smoothquant_calibration_source",
                "smoothquant_calibration_sentences",
                "omniquant_abits",
                "omniquant_epochs",
                "omniquant_nsamples",
                "omniquant_seqlen",
                "omniquant_seed",
                "omniquant_calibration_source",
                "omniquant_lwc",
                "omniquant_let",
                "omniquant_aug_loss",
            )
            if key in source_manifest
        }
        manifest = {
            "model_name": args.model_name,
            "source_model_path": args.model_path,
            "compressed_model_path": args.model_path,
            "compression_method": args.compression_method,
            "revision": args.revision,
            "base_model_repository": args.base_model_repository or None,
            "base_model_revision": args.base_model_revision or None,
            "base_model_attestation_path": args.base_model_attestation_path or None,
            "base_model_attested": bool(args.base_model_attestation_path),
            "backend": "precompressed_checkpoint",
            "quant_bits": args.quant_bits,
            "group_size": args.group_size,
            "desc_act": args.desc_act,
            "awq_version": args.awq_version,
            "sparsity": args.sparsity,
            "sparsity_semantics": (
                "element_zero_fraction"
                if args.compression_method in {"wanda", "sparsegpt"}
                else "method_config"
            ),
            "prune_pattern": args.prune_pattern,
            "prune_structure": (
                "semi_structured"
                if args.prune_pattern == "2:4"
                else "unstructured"
            ),
            "prune_n": 2 if args.prune_pattern == "2:4" else None,
            "prune_m": 4 if args.prune_pattern == "2:4" else None,
            "prune_lm_head": args.prune_lm_head,
            "calibration_samples": args.calibration_samples,
            "calibration_file": str(Path(args.calibration_file).expanduser()) if args.calibration_file else None,
            "calibration_batch_size": args.calibration_batch_size,
            "dtype": args.dtype,
            "seqlen": args.seqlen,
            "max_memory": args.max_memory,
            "attn_implementation": args.attn_implementation,
            "materialized": False,
            "paper_protocol": paper_protocol or None,
            "paper_protocol_attested": paper_attestation.get("attested", False),
            "paper_protocol_attestation": paper_attestation,
            "note": "Input --model-path is treated as an already compressed HuggingFace checkpoint.",
            **protocol_fields,
        }
    else:
        manifest = run_compression(args)
    compressed_model_path = manifest["compressed_model_path"]
    summary = {
        "benchmark": "edge_pipeline",
        **_model_metadata(args),
        "runtime": _runtime_metadata(),
        "compressed_model_path": compressed_model_path,
        "steps": {
            "compression": manifest,
        },
    }
    _write_json(
        {
            "benchmark": "edge_compression",
            **_model_metadata(args),
            "runtime": _runtime_metadata(),
            "compressed_model_path": compressed_model_path,
            "manifest": manifest,
        },
        str(output_root / "compression.json"),
    )

    if not args.skip_accuracy:
        if args.split_accuracy_tasks:
            accuracy_output = _run_split_accuracy(args, compressed_model_path, output_root)
            summary["steps"]["accuracy"] = accuracy_output
        else:
            accuracy_args = argparse.Namespace(**vars(args))
            accuracy_args.model_path = compressed_model_path
            accuracy_args.output = str(output_root / "accuracy.json")
            accuracy_args.lm_eval_output = str(output_root / "lm_eval_raw")
            run_accuracy(accuracy_args)
            summary["steps"]["accuracy"] = accuracy_args.output

    if not args.skip_deploy:
        deploy_args = argparse.Namespace(**vars(args))
        deploy_args.model_path = compressed_model_path
        deploy_args.output = str(output_root / "deployment.json")
        try:
            run_deploy(deploy_args)
        except Exception as exc:
            _write_deployment_error(deploy_args, exc)
            summary.setdefault("failed_steps", {})["deployment"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "output": deploy_args.output,
            }
        summary["steps"]["deployment"] = deploy_args.output

    if not args.skip_resource:
        resource_args = argparse.Namespace(**vars(args))
        resource_args.model_path = compressed_model_path
        resource_args.output = str(output_root / "resource.json")
        try:
            run_resource(resource_args)
        except Exception as exc:
            _write_resource_error(resource_args, exc)
            summary.setdefault("failed_steps", {})["resource"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "output": resource_args.output,
            }
        summary["steps"]["resource"] = resource_args.output

    _write_json(summary, str(output_root / "summary.json"))
    if summary.get("failed_steps"):
        raise RuntimeError(f"Pipeline failed steps: {sorted(summary['failed_steps'])}")


def _run_split_accuracy(args: argparse.Namespace, model_path: str, output_root: Path) -> str:
    tasks = _split_csv(args.tasks) if args.tasks else list(DEFAULT_ACCURACY_TASKS)
    merged = {
        "benchmark": "edge_accuracy",
        **_model_metadata(args),
        "model_path": model_path,
        "tasks": tasks,
        "dtype": args.dtype,
        "lm_eval_dtype": _lm_eval_dtype(args.dtype),
        "batch_size": args.batch_size,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "device": args.device,
        "max_memory": args.max_memory,
        "attn_implementation": args.attn_implementation or None,
        "split_accuracy_tasks": True,
        "continue_on_accuracy_error": args.continue_on_accuracy_error,
        "results": {},
        "task_outputs": {},
        "failed_tasks": {},
    }
    for task in tasks:
        task_args = argparse.Namespace(**vars(args))
        task_args.model_path = model_path
        task_args.tasks = [task]
        task_args.output = str(output_root / "accuracy_tasks" / f"{task}.json")
        task_args.lm_eval_output = str(output_root / "lm_eval_raw" / task)
        try:
            run_accuracy(task_args)
            task_payload = json.loads(Path(task_args.output).read_text(encoding="utf-8"))
            merged["results"].update(task_payload.get("results", {}))
            merged["task_outputs"][task] = task_args.output
        except (Exception, SystemExit) as exc:
            merged["failed_tasks"][task] = {
                "error": str(exc),
                "output": task_args.output,
                "lm_eval_output": task_args.lm_eval_output,
            }
            if not args.continue_on_accuracy_error:
                output = output_root / "accuracy.json"
                _write_json(merged, str(output))
                raise
    output = output_root / "accuracy.json"
    _write_json(merged, str(output))
    if merged["failed_tasks"] and not args.continue_on_accuracy_error:
        raise RuntimeError(f"Split accuracy failed tasks: {sorted(merged['failed_tasks'])}")
    return str(output)


def _load_lm_eval_results(output_path: str) -> Dict:
    path = Path(output_path)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    json_files = sorted(path.glob("**/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        raise FileNotFoundError(f"lm-eval result json not found under {output_path}")
    return json.loads(json_files[0].read_text(encoding="utf-8"))


def _extract_accuracy_results(raw: Dict) -> Dict[str, Dict[str, float]]:
    extracted: Dict[str, Dict[str, float]] = {}
    for task, metrics in raw.get("results", {}).items():
        task_metrics: Dict[str, float] = {}
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                task_metrics[name] = float(value)
        extracted[task] = task_metrics
    return extracted


def _load_tokenizer(model_path: str, cache_dir: Optional[str], revision: Optional[str]):
    if is_adapted_quantization_checkpoint(model_path):
        model_path = adapted_checkpoint_metadata(model_path)["source_model_path"]
    kwargs = {"trust_remote_code": True, "use_fast": False}
    if cache_dir:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    if revision:
        kwargs["revision"] = revision
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    except Exception:
        kwargs["use_fast"] = True
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    if not hasattr(tokenizer, "pad_token_id"):
        kwargs["use_fast"] = True
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(args: argparse.Namespace):
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _resolve_dtype(args.dtype),
    }
    if args.cache_dir:
        kwargs["cache_dir"] = str(Path(args.cache_dir).expanduser())
    if args.revision:
        kwargs["revision"] = args.revision
    if getattr(args, "device_map", None):
        kwargs["device_map"] = args.device_map
    if getattr(args, "max_memory", None):
        kwargs["max_memory"] = _parse_max_memory(args.max_memory)
    if getattr(args, "attn_implementation", None):
        kwargs["attn_implementation"] = args.attn_implementation
    if is_adapted_quantization_checkpoint(args.model_path):
        model = load_adapted_model(args.model_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    model.eval()
    if not getattr(args, "device_map", None) and args.device:
        model.to(torch.device(args.device))
    return model


def _resolve_dtype(dtype: str):
    if dtype in ("auto", "none", "None"):
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _parse_max_memory(value: str):
    if not value:
        return None
    if ":" not in value and "," not in value:
        return {0: value}
    parsed = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, memory = item.split(":", 1)
        key = key.strip()
        parsed[int(key) if key.isdigit() else key] = memory.strip()
    return parsed


def _first_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _greedy_generation_kwargs(tokenizer, max_new_tokens: int) -> Dict:
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "pad_token_id": tokenizer.eos_token_id,
    }


def _deployment_memory_status(payload: Dict, limit_gb: float) -> Dict:
    peaks = [
        value
        for value in (
            payload.get("peak_memory_allocated_gb"),
            payload.get("peak_memory_reserved_gb"),
        )
        if isinstance(value, (int, float))
    ]
    measured_peak = max(peaks) if peaks else None
    if limit_gb <= 0:
        return {
            "deployment_memory_limit_gb": None,
            "deployment_memory_measured_peak_gb": measured_peak,
            "deployment_memory_limit_exceeded": False,
        }
    return {
        "deployment_memory_limit_gb": limit_gb,
        "deployment_memory_measured_peak_gb": measured_peak,
        "deployment_memory_limit_exceeded": measured_peak is not None and measured_peak > limit_gb,
    }


def _write_deployment_error(args: argparse.Namespace, exc: Exception) -> None:
    payload = {
        "benchmark": "edge_deployment",
        **_model_metadata(args),
        "runtime": _runtime_metadata(),
        "device": getattr(args, "device", None),
        "dtype": getattr(args, "dtype", None),
        "device_map": getattr(args, "device_map", None),
        "max_memory": getattr(args, "max_memory", None),
        "attn_implementation": getattr(args, "attn_implementation", None) or None,
        "prompt_source": None,
        "prompt_path": None,
        "prompt_count": None,
        "input_length": getattr(args, "input_length", None),
        "max_new_tokens": getattr(args, "max_new_tokens", None),
        "deployment_batch_size": getattr(args, "deployment_batch_size", DEFAULT_DEPLOYMENT_BATCH_SIZE),
        "warmup": getattr(args, "warmup", None),
        "repeat": getattr(args, "repeat", None),
        "input_tokens": None,
        "generated_tokens": None,
        "elapsed_seconds": None,
        "tokens_per_second": None,
        "peak_memory_allocated_gb": None,
        "peak_memory_reserved_gb": None,
        "deployment_error": f"{type(exc).__name__}: {exc}",
    }
    payload.update(_deployment_memory_status(payload, getattr(args, "deployment_memory_limit_gb", 0)))
    _write_json(payload, args.output)


def _read_prompts(prompt: Optional[str], prompt_file: Optional[str]) -> tuple[List[str], str, Optional[str]]:
    if prompt_file:
        lines = Path(prompt_file).read_text(encoding="utf-8").splitlines()
        prompts = [line.strip() for line in lines if line.strip()]
        if prompts:
            return prompts, "file", str(Path(prompt_file).expanduser())
    if prompt:
        return [prompt], "inline", None
    return (
        ["Explain why model compression is useful for edge-side large language model deployment."],
        "default",
        None,
    )


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    batch_size = max(int(batch_size or 1), 1)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _generation_inputs(encoded: Dict[str, torch.Tensor], model) -> Dict[str, torch.Tensor]:
    accepted = {
        name
        for name, parameter in inspect.signature(model.forward).parameters.items()
        if parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    return {key: value for key, value in encoded.items() if key in accepted}


@torch.no_grad()
def run_deploy(args: argparse.Namespace) -> None:
    _apply_cuda_default_max_memory(args)
    _check_gpu_memory_budget(args)
    env = _configure_hf_cache(args.cache_dir, None)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.update({k: v for k, v in env.items() if k.startswith("HF_") or k.endswith("_CACHE")})

    tokenizer = _load_tokenizer(args.model_path, args.cache_dir, args.revision)
    model = _load_model(args)
    device = _first_model_device(model)
    prompts, prompt_source, prompt_path = _read_prompts(args.prompt, args.prompt_file)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    prompt_batches = list(_batched(prompts, getattr(args, "deployment_batch_size", DEFAULT_DEPLOYMENT_BATCH_SIZE)))
    encoded_batches = []
    for batch_prompts in prompt_batches:
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.input_length)
        encoded_batches.append(_generation_inputs({k: v.to(device) for k, v in encoded.items()}, model))

    for _ in range(args.warmup):
        for encoded in encoded_batches:
            _ = model.generate(
                **encoded,
                **_greedy_generation_kwargs(tokenizer, args.max_new_tokens),
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    generated_tokens = 0
    started_at = time.perf_counter()
    for _ in range(args.repeat):
        for encoded in encoded_batches:
            outputs = model.generate(
                **encoded,
                **_greedy_generation_kwargs(tokenizer, args.max_new_tokens),
            )
            generated_tokens += int(outputs.shape[-1] - encoded["input_ids"].shape[-1]) * int(encoded["input_ids"].shape[0])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at

    peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    payload = {
        "benchmark": "edge_deployment",
        **_model_metadata(args),
        "runtime": _runtime_metadata(),
        "device": str(device),
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_memory": args.max_memory,
        "attn_implementation": args.attn_implementation or None,
        "batch_size": max(len(batch) for batch in prompt_batches),
        "deployment_batch_size": getattr(args, "deployment_batch_size", DEFAULT_DEPLOYMENT_BATCH_SIZE),
        "prompt_source": prompt_source,
        "prompt_path": prompt_path,
        "prompt_count": len(prompts),
        "input_length": args.input_length,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "input_tokens": sum(int(encoded["input_ids"].numel()) for encoded in encoded_batches),
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": generated_tokens / elapsed if elapsed > 0 else None,
        "peak_memory_allocated_gb": peak_allocated / (1024**3),
        "peak_memory_reserved_gb": peak_reserved / (1024**3),
    }
    payload.update(_deployment_memory_status(payload, args.deployment_memory_limit_gb))
    _write_json(payload, args.output)


def run_resource(args: argparse.Namespace) -> None:
    _apply_cuda_default_max_memory(args)
    _check_gpu_memory_budget(args)
    env = _configure_hf_cache(args.cache_dir, None)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.update({k: v for k, v in env.items() if k.startswith("HF_") or k.endswith("_CACHE")})

    storage_bytes = _directory_size(Path(args.model_path))
    load_error = None
    resource_backend = "model_load"
    try:
        tokenizer = _load_tokenizer(args.model_path, args.cache_dir, args.revision)
        model = _load_model(args)
        param_stats = _parameter_stats(model, args.sparsity_sample_tensors)
        flops = _calculate_flops(model, tokenizer, args.seqlen)
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        resource_backend = "config_only_fallback"
        config = _load_config_only(args)
        param_stats = _config_parameter_stats(config)
        flops = _analytical_flops_from_config(
            config,
            args.seqlen,
            f"model load failed; using config-only analytical estimate: {load_error}",
            param_stats.get("total_params"),
        )
    effective_storage_bytes = None
    if storage_bytes is not None and param_stats["nonzero_ratio"] is not None:
        effective_storage_bytes = storage_bytes * param_stats["nonzero_ratio"]
    method = (args.compression_method or "").lower()
    quantization_storage = inspect_quantization_storage(Path(args.model_path))
    effective_quantized_storage_bytes = None
    if (
        method in {"gptq", "awq", "smoothquant", "omniquant"}
        and quantization_storage["is_real_quantized"]
        and storage_bytes is not None
    ):
        effective_quantized_storage_bytes = storage_bytes
    effective_quantized_storage_backend = (
        "checkpoint_directory_size" if effective_quantized_storage_bytes is not None else None
    )
    effective_quantized_storage_bits = (
        quantization_storage.get("weight_bits") or args.quant_bits
        if effective_quantized_storage_bytes is not None
        else None
    )
    effective_flops = None
    if flops.get("flops_num") is not None and param_stats["nonzero_ratio"] is not None:
        effective_flops = flops["flops_num"] * param_stats["nonzero_ratio"]
    prompts, prompt_source, prompt_path = _read_prompts(
        getattr(args, "prompt", None),
        getattr(args, "prompt_file", None),
    )

    payload = {
        "benchmark": "edge_resource",
        **_model_metadata(args),
        "runtime": _runtime_metadata(),
        "resource_backend": resource_backend,
        "model_load_error": load_error,
        "seqlen": args.seqlen,
        "prompt_source": prompt_source,
        "prompt_path": prompt_path,
        "prompt_count": len(prompts),
        "input_length": getattr(args, "input_length", args.seqlen),
        "max_new_tokens": getattr(args, "max_new_tokens", 0),
        "sparsity_sample_tensors": args.sparsity_sample_tensors,
        "storage_bytes": storage_bytes,
        "storage_gb": storage_bytes / (1024**3) if storage_bytes is not None else None,
        "effective_storage_bytes_sparse_approx": effective_storage_bytes,
        "effective_storage_gb_sparse_approx": effective_storage_bytes / (1024**3) if effective_storage_bytes is not None else None,
        "effective_storage_bytes_quantized_approx": effective_quantized_storage_bytes,
        "effective_storage_gb_quantized_approx": effective_quantized_storage_bytes / (1024**3) if effective_quantized_storage_bytes is not None else None,
        "effective_storage_quantized_bits": effective_quantized_storage_bits,
        "effective_storage_quantized_backend": effective_quantized_storage_backend,
        "quantization_storage_format": quantization_storage["format"],
        "quantization_storage_real": quantization_storage["is_real_quantized"],
        "quantization_storage_dtypes": quantization_storage["dtypes"],
        "parameter_stats": param_stats,
        "compute_complexity": {
            **flops,
            "effective_flops_num_sparse_approx": effective_flops,
        },
    }
    _write_json(payload, args.output)


def _write_resource_error(args: argparse.Namespace, exc: Exception) -> None:
    storage_bytes = _directory_size(Path(args.model_path))
    try:
        config = _load_config_only(args)
        param_stats = _config_parameter_stats(config)
        flops = _analytical_flops_from_config(
            config,
            getattr(args, "seqlen", 2048),
            f"resource step failed before model load; using config-only analytical estimate: {type(exc).__name__}: {exc}",
            param_stats.get("total_params"),
        )
    except Exception as fallback_exc:
        param_stats = {
            "total_params": None,
            "trainable_params": None,
            "tensor_count": None,
            "sparsity": None,
            "nonzero_ratio": None,
            "sparsity_sampled_tensors": 0,
            "sparsity_sampled_params": 0,
            "estimation_backend": "unavailable",
        }
        flops = {
            "flops": None,
            "macs": None,
            "params": None,
            "flops_num": None,
            "backend": "unavailable",
            "warning": f"resource fallback failed: {type(fallback_exc).__name__}: {fallback_exc}",
        }
    prompts, prompt_source, prompt_path = _read_prompts(
        getattr(args, "prompt", None),
        getattr(args, "prompt_file", None),
    )
    payload = {
        "benchmark": "edge_resource",
        **_model_metadata(args),
        "runtime": _runtime_metadata(),
        "resource_backend": "error_fallback",
        "model_load_error": f"{type(exc).__name__}: {exc}",
        "seqlen": getattr(args, "seqlen", None),
        "prompt_source": prompt_source,
        "prompt_path": prompt_path,
        "prompt_count": len(prompts),
        "input_length": getattr(args, "input_length", getattr(args, "seqlen", None)),
        "max_new_tokens": getattr(args, "max_new_tokens", None),
        "sparsity_sample_tensors": getattr(args, "sparsity_sample_tensors", None),
        "storage_bytes": storage_bytes,
        "storage_gb": storage_bytes / (1024**3) if storage_bytes is not None else None,
        "effective_storage_bytes_sparse_approx": None,
        "effective_storage_gb_sparse_approx": None,
        "effective_storage_bytes_quantized_approx": None,
        "effective_storage_gb_quantized_approx": None,
        "effective_storage_quantized_bits": None,
        "effective_storage_quantized_backend": None,
        "parameter_stats": param_stats,
        "compute_complexity": {
            **flops,
            "effective_flops_num_sparse_approx": None,
        },
    }
    _write_json(payload, args.output)


def _directory_size(path: Path) -> Optional[int]:
    path = path.expanduser()
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return None
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


@torch.no_grad()
def _parameter_stats(model, sample_tensors: int) -> Dict[str, Optional[float]]:
    total_params = 0
    sampled_params = 0
    zero_params = 0
    trainable_params = 0
    tensor_count = 0
    sampled = 0

    for param in model.parameters():
        numel = param.numel()
        total_params += numel
        tensor_count += 1
        if param.requires_grad:
            trainable_params += numel
        if sample_tensors < 0 or sampled < sample_tensors:
            tensor = param.detach()
            if tensor.is_floating_point() or tensor.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
                zero_params += int(torch.count_nonzero(tensor == 0).item())
                sampled_params += numel
                sampled += 1

    nonzero_ratio = None
    sparsity = None
    if sampled_params:
        sparsity = zero_params / sampled_params
        nonzero_ratio = 1.0 - sparsity

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "tensor_count": tensor_count,
        "sparsity": sparsity,
        "nonzero_ratio": nonzero_ratio,
        "sparsity_sampled_tensors": sampled,
        "sparsity_sampled_params": sampled_params,
    }


def _calculate_flops(model, tokenizer, seqlen: int) -> Dict[str, Optional[object]]:
    try:
        from calflops import calculate_flops
    except ImportError:
        return _analytical_flops(model, seqlen, "calflops is not installed; using analytical config estimate.")

    try:
        flops, macs, params = calculate_flops(
            model=model,
            input_shape=(1, seqlen),
            transformer_tokenizer=tokenizer,
            print_results=False,
            print_detailed=False,
            output_precision=2,
        )
        return {
            "flops": flops,
            "macs": macs,
            "params": params,
            "flops_num": _parse_human_number(flops),
            "backend": "calflops",
        }
    except Exception as exc:
        return _analytical_flops(model, seqlen, f"calflops failed; using analytical config estimate: {exc}")


def _analytical_flops(model, seqlen: int, warning: str) -> Dict[str, Optional[object]]:
    config = getattr(model, "config", None)
    total_params = sum(param.numel() for param in model.parameters())
    return _analytical_flops_from_config(config, seqlen, warning, total_params)


def _analytical_flops_from_config(
    config,
    seqlen: int,
    warning: str,
    total_params: Optional[int],
) -> Dict[str, Optional[object]]:
    hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    intermediate = getattr(config, "intermediate_size", None)
    vocab = getattr(config, "vocab_size", None)
    total_params = total_params or 0
    if hidden is None or layers is None:
        flops_num = 2.0 * total_params * seqlen
    else:
        intermediate = intermediate or (4 * hidden)
        attention_linear = 4 * hidden * hidden
        attention_scores = 2 * seqlen * hidden
        mlp = 3 * hidden * intermediate
        per_token_per_layer = 2 * (attention_linear + attention_scores + mlp)
        lm_head = 2 * hidden * vocab if vocab else 0
        flops_num = float(layers * seqlen * per_token_per_layer + seqlen * lm_head)
    macs_num = flops_num / 2.0
    return {
        "flops": _format_human_number(flops_num, "FLOPS"),
        "macs": _format_human_number(macs_num, "MACs"),
        "params": str(total_params),
        "flops_num": flops_num,
        "backend": "analytical_config_estimate",
        "warning": warning,
    }


def _load_config_only(args: argparse.Namespace):
    kwargs = {"trust_remote_code": True}
    if args.cache_dir:
        kwargs["cache_dir"] = str(Path(args.cache_dir).expanduser())
    if args.revision:
        kwargs["revision"] = args.revision
    return AutoConfig.from_pretrained(args.model_path, **kwargs)


def _config_parameter_stats(config) -> Dict[str, Optional[float]]:
    hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    intermediate = getattr(config, "intermediate_size", None)
    vocab = getattr(config, "vocab_size", None)
    total_params = None
    if hidden and layers:
        intermediate = intermediate or (4 * hidden)
        attention = 4 * hidden * hidden
        mlp = 3 * hidden * intermediate
        layer_norm = 2 * hidden
        per_layer = attention + mlp + layer_norm
        embeddings = (vocab or 0) * hidden
        lm_head = 0 if getattr(config, "tie_word_embeddings", False) else embeddings
        total_params = int(layers * per_layer + embeddings + lm_head)
    return {
        "total_params": total_params,
        "trainable_params": None,
        "tensor_count": None,
        "sparsity": None,
        "nonzero_ratio": 1.0,
        "sparsity_sampled_tensors": 0,
        "sparsity_sampled_params": 0,
        "estimation_backend": "config_architecture_estimate",
    }


def _format_human_number(value: float, suffix: str) -> str:
    units = [(1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]
    for scale, prefix in units:
        if abs(value) >= scale:
            return f"{value / scale:.2f} {prefix}{suffix}"
    return f"{value:.2f} {suffix}"


def _parse_human_number(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    parts = value.strip().split()
    if not parts:
        return None
    try:
        number = float(parts[0])
    except ValueError:
        return None
    suffix = parts[1].lower() if len(parts) > 1 else ""
    scales = {
        "kflops": 1e3,
        "mflops": 1e6,
        "gflops": 1e9,
        "tflops": 1e12,
        "pflops": 1e15,
        "flops": 1.0,
    }
    return number * scales.get(suffix, 1.0)


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True, help="Compressed HF checkpoint path or model id.")
    parser.add_argument("--model-name", default="unknown", choices=SUPPORTED_MODELS + ("unknown",), help="Model family label.")
    parser.add_argument("--compression-method", default="dense", choices=SUPPORTED_METHODS, help="Compression method label.")
    parser.add_argument("--cache-dir", default=None, help="HF model/cache directory.")
    parser.add_argument("--revision", default=None, help="HF revision.")
    parser.add_argument("--base-model-repository", default="", help="Attested upstream repository for a strict base-model run.")
    parser.add_argument("--base-model-revision", default="", help="Attested upstream revision for a strict base-model run.")
    parser.add_argument("--base-model-attestation-path", default="", help="Verified local base-model attestation JSON for a strict run.")
    parser.add_argument("--dtype", default="auto", help="auto, fp16, bf16, fp32.")
    parser.add_argument("--attn-implementation", default=None, help="Optional Transformers attention backend, for example eager or sdpa.")
    parser.add_argument("--max-memory", default=None, help=f"Transformers max_memory map, e.g. '0:22GiB,cpu:120GiB' or '22GiB'. Defaults to {DEFAULT_CUDA_MAX_MEMORY} on CUDA hosts.")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT), help="HF endpoint, for example https://hf-mirror.com.")
    parser.add_argument("--output", default=None, help="Write normalized benchmark JSON to this path.")
    parser.add_argument("--min-free-gpu-memory-gb", type=float, default=DEFAULT_MIN_FREE_GPU_MEMORY_GB, help="Fail early unless a visible GPU has at least this much free memory. 0 disables the guard.")


def add_compression_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--compressed-output-dir", default=None, help="Directory for the compressed checkpoint.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map, or empty string to use --device.")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--calibration-file", default=None, help="One calibration text per line.")
    parser.add_argument("--calibration-samples", type=int, default=128, help="Number of calibration sequences; official Wanda/SparseGPT protocol uses 128.")
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument(
        "--pruning-calibration-source",
        choices=("c4", "text_file"),
        default="c4",
        help="Calibration source for Wanda/SparseGPT. c4 uses the official random 2048-token C4 protocol.",
    )
    parser.add_argument("--calibration-seed", type=int, default=0, help="Seed for official C4 calibration sampling.")
    parser.add_argument("--calibration-dataset", default="allenai/c4", help="Dataset id used when --pruning-calibration-source=c4.")
    parser.add_argument("--calibration-dataset-config", default="allenai--c4", help="Dataset config used when --pruning-calibration-source=c4.")
    parser.add_argument(
        "--calibration-c4-data-file",
        default="en/c4-train.00000-of-01024.json.gz",
        help="C4 training shard used for deterministic official-style calibration.",
    )
    parser.add_argument(
        "--calibration-c4-revision",
        default="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        help="Pinned C4 dataset revision used for deterministic official-style calibration.",
    )
    parser.add_argument("--calibration-dataset-cache", default=None, help="Optional datasets cache for C4 calibration.")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target pruning sparsity.")
    parser.add_argument(
        "--prune-pattern",
        default="unstructured",
        choices=("unstructured", "2:4"),
        help="Weight sparsity pattern for Wanda/SparseGPT.",
    )
    parser.add_argument("--prune-lm-head", action="store_true", help="Also prune lm_head. Disabled by default to reduce accuracy risk.")
    parser.add_argument("--sparsegpt-blocksize", type=int, default=128, help="SparseGPT column block size for blockwise second-order updates.")
    parser.add_argument("--sparsegpt-percdamp", type=float, default=0.01, help="SparseGPT Hessian diagonal damping fraction.")
    parser.add_argument("--pruning-buffer-device", default="cpu", help="Device for layerwise calibration activations; cpu keeps 128x2048 buffers off the 24GB GPU.")
    parser.add_argument("--quant-bits", type=int, default=4, help="Quantization bits for GPTQ/AWQ.")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size for GPTQ/AWQ.")
    parser.add_argument("--desc-act", action="store_true", help="Enable GPTQ activation-order quantization.")
    parser.add_argument(
        "--gptq-calibration-source",
        choices=("c4",),
        default="c4",
        help="GPTQ calibration source. c4 follows the original GPTQ/GPTQ-for-LLaMA path.",
    )
    parser.add_argument("--gptq-damp-percent", type=float, default=0.01, help="GPTQ Hessian damping fraction.")
    parser.add_argument("--gptq-true-sequential", action="store_true", help="Enable original GPTQ-for-LLaMA true sequential quantization.")
    parser.add_argument("--gptq-static-groups", action="store_true", help="Use GPTQ static groups.")
    parser.add_argument("--gptq-asymmetric", action="store_true", help="Use asymmetric GPTQ quantization; this matches the official GPTQ-for-LLaMA command when --sym is omitted.")
    parser.add_argument("--awq-version", default="GEMM", help="AutoAWQ kernel version, usually GEMM or GEMV.")
    parser.add_argument(
        "--awq-calibration-source",
        choices=("pileval", "text_file"),
        default="pileval",
        help="AWQ calibration source. pileval uses AutoAWQ's official Pile validation defaults.",
    )
    parser.add_argument("--awq-calibration-seed", type=int, default=42, help="Expected AutoAWQ pileval shuffle seed for the strict paper profile.")
    parser.add_argument("--awq-calibration-samples", type=int, default=128, help="AutoAWQ maximum calibration samples.")
    parser.add_argument("--awq-calibration-seqlen", type=int, default=512, help="AutoAWQ maximum calibration sequence length.")
    parser.add_argument("--awq-grid-size", type=int, default=20, help="AWQ alpha grid-search size used by the reference implementation.")
    parser.add_argument("--smoothquant-alpha", type=float, default=None, help="SmoothQuant migration strength selected by the official recipe/grid search.")
    parser.add_argument(
        "--smoothquant-alpha-selection",
        choices=("official_recipe", "official_grid_search", "adapted_fixed_alpha"),
        default="official_grid_search",
        help="How SmoothQuant alpha is selected for the exported checkpoint.",
    )
    parser.add_argument("--smoothquant-calibration-source", default="pile_validation", help="SmoothQuant calibration corpus identifier.")
    parser.add_argument("--smoothquant-calibration-sentences", type=int, default=512, help="SmoothQuant activation-scale calibration sentence count.")
    parser.add_argument("--smoothquant-official-code-dir", default="", help="Pinned official SmoothQuant Git checkout required by the strict paper profile.")
    parser.add_argument("--smoothquant-official-repository", default="", help="Expected official SmoothQuant Git repository URL.")
    parser.add_argument("--smoothquant-official-commit", default="", help="Expected official SmoothQuant Git commit.")
    parser.add_argument("--omniquant-abits", type=int, default=16, help="OmniQuant activation bit width.")
    parser.add_argument("--omniquant-epochs", type=int, default=20, help="OmniQuant calibration optimization epochs.")
    parser.add_argument("--omniquant-nsamples", type=int, default=128, help="OmniQuant calibration segment count.")
    parser.add_argument("--omniquant-seqlen", type=int, default=2048, help="OmniQuant calibration segment length.")
    parser.add_argument("--omniquant-seed", type=int, default=2, help="OmniQuant calibration seed; the official main.py default is 2.")
    parser.add_argument("--omniquant-calibration-source", default="wikitext2", help="OmniQuant calibration corpus identifier.")
    parser.add_argument("--omniquant-lwc", action="store_true", help="Enable OmniQuant learnable weight clipping.")
    parser.add_argument("--omniquant-let", action="store_true", help="Enable OmniQuant learnable equivalent transformation.")
    parser.add_argument("--omniquant-aug-loss", action="store_true", help="Enable OmniQuant augmented loss.")
    parser.add_argument("--omniquant-official-code-dir", default="", help="Pinned official OmniQuant Git checkout required by the strict paper profile.")
    parser.add_argument("--omniquant-official-repository", default="", help="Expected official OmniQuant Git repository URL.")
    parser.add_argument("--omniquant-official-commit", default="", help="Expected official OmniQuant Git commit.")
    parser.add_argument(
        "--paper-protocol",
        default="",
        help="Named paper/official-implementation protocol recorded in manifests; required by the paper-aligned matrix.",
    )
    parser.add_argument(
        "--target-model-adaptation",
        action="store_true",
        help=(
            "Run the built-in target-model adaptation for SmoothQuant/OmniQuant when "
            "no upstream recipe exists. Results are explicitly marked as non-paper-reproduction."
        ),
    )
    parser.add_argument(
        "--require-paper-protocol-attestation",
        action="store_true",
        help="Reject precompressed checkpoints unless their LLMCBench manifest attests to --paper-protocol.",
    )
    parser.add_argument("--external-command", default=None, help="External SmoothQuant/OmniQuant command template.")
    parser.add_argument("--max-shard-size", default="2GB", help="Max checkpoint shard size for save_pretrained.")
    parser.add_argument("--overwrite-compressed", action="store_true", help="Remove an existing compressed checkpoint directory before writing.")


def add_pipeline_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", required=True, help="Directory for pipeline outputs.")
    parser.add_argument("--tasks", action="append", default=None, help="Comma-separated lm-eval task names.")
    parser.add_argument("--data-dir", default=None, help="HF datasets cache/local dataset cache directory.")
    parser.add_argument("--batch-size", default="auto", help="lm-eval batch size.")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None, help="Optional lm-eval sample limit for smoke tests.")
    parser.add_argument("--stream-output", action="store_true", help="Stream lm-eval output instead of capturing it.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None, help="One prompt per line.")
    parser.add_argument("--input-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--deployment-batch-size", type=int, default=DEFAULT_DEPLOYMENT_BATCH_SIZE, help="Number of prompts generated concurrently during deployment benchmarking.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--deployment-memory-limit-gb", type=float, default=DEFAULT_DEPLOYMENT_MEMORY_LIMIT_GB, help="Fail deployment if peak allocated/reserved GPU memory exceeds this value. 0 disables the check.")
    parser.add_argument("--sparsity-sample-tensors", type=int, default=-1, help="-1 checks all tensors.")
    parser.add_argument("--use-precompressed", action="store_true", help="Treat --model-path as an already compressed checkpoint and skip the compression step.")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--split-accuracy-tasks", action="store_true", help="Run lm-eval tasks one at a time and merge accuracy.json.")
    parser.add_argument("--continue-on-accuracy-error", action="store_true", help="With --split-accuracy-tasks, keep successful task outputs when one task fails.")
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--skip-resource", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edge-side compressed LLM benchmark tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress = subparsers.add_parser("compress", help="Compress a base checkpoint with the selected method.")
    add_common_model_args(compress)
    add_compression_args(compress)
    compress.set_defaults(func=run_compress)

    accuracy = subparsers.add_parser("accuracy", help="Run accuracy tasks with lm-evaluation-harness.")
    add_common_model_args(accuracy)
    accuracy.add_argument("--tasks", action="append", default=None, help="Comma-separated lm-eval task names.")
    accuracy.add_argument("--data-dir", default=None, help="HF datasets cache/local dataset cache directory.")
    accuracy.add_argument("--device", default="cuda:0", help="lm-eval device.")
    accuracy.add_argument("--batch-size", default="auto", help="lm-eval batch size.")
    accuracy.add_argument("--num-fewshot", type=int, default=None)
    accuracy.add_argument("--limit", type=float, default=None, help="Optional lm-eval sample limit for smoke tests.")
    accuracy.add_argument("--lm-eval-output", default="outputs/lm_eval", help="Raw lm-eval output path.")
    accuracy.add_argument("--stream-output", action="store_true", help="Stream lm-eval output instead of capturing it.")
    accuracy.set_defaults(func=run_accuracy)

    deploy = subparsers.add_parser("deploy", help="Measure inference memory and generation speed.")
    add_common_model_args(deploy)
    deploy.add_argument("--device", default="cuda:0")
    deploy.add_argument("--device-map", default="auto", help="Transformers device_map, or empty string to use --device.")
    deploy.add_argument("--prompt", default=None)
    deploy.add_argument("--prompt-file", default=None, help="One prompt per line.")
    deploy.add_argument("--input-length", type=int, default=2048)
    deploy.add_argument("--max-new-tokens", type=int, default=128)
    deploy.add_argument("--deployment-batch-size", type=int, default=DEFAULT_DEPLOYMENT_BATCH_SIZE, help="Number of prompts generated concurrently during deployment benchmarking.")
    deploy.add_argument("--warmup", type=int, default=1)
    deploy.add_argument("--repeat", type=int, default=3)
    deploy.add_argument("--deployment-memory-limit-gb", type=float, default=DEFAULT_DEPLOYMENT_MEMORY_LIMIT_GB, help="Fail if peak allocated/reserved GPU memory exceeds this value. 0 disables the check.")
    deploy.set_defaults(func=run_deploy)

    resource = subparsers.add_parser("resource", help="Measure storage, sparsity, params and FLOPs.")
    add_common_model_args(resource)
    resource.add_argument("--device", default="cuda:0")
    resource.add_argument("--device-map", default="auto", help="Transformers device_map, or empty string to use --device.")
    resource.add_argument("--seqlen", type=int, default=2048)
    resource.add_argument("--prompt", default=None)
    resource.add_argument("--prompt-file", default=None, help="One prompt per line; recorded as resource input provenance.")
    resource.add_argument("--input-length", type=int, default=2048)
    resource.add_argument("--max-new-tokens", type=int, default=128)
    resource.add_argument("--sparsity-sample-tensors", type=int, default=-1, help="-1 checks all tensors.")
    resource.add_argument("--quant-bits", type=int, default=4, help="Quantization bits used for quantized effective storage estimates.")
    resource.add_argument("--group-size", type=int, default=128, help="Quantization group size metadata for consistency with compression runs.")
    resource.set_defaults(func=run_resource)

    pipeline = subparsers.add_parser("pipeline", help="Compress, then run accuracy/deployment/resource benchmarks.")
    add_common_model_args(pipeline)
    add_compression_args(pipeline)
    add_pipeline_eval_args(pipeline)
    pipeline.set_defaults(func=run_pipeline)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "device_map") and args.device_map == "":
        args.device_map = None
    args.func(args)


if __name__ == "__main__":
    main()
