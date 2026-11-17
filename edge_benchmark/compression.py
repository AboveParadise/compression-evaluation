import json
import hashlib
import inspect
import math
import os
import random
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapted_quantization import export_adapted_quantization
from .runtime_compat import apply_runtime_compatibility_patches


apply_runtime_compatibility_patches()


PRUNE_METHODS = {"wanda", "sparsegpt"}
QUANT_METHODS = {"gptq", "awq"}
EXTERNAL_METHODS = {"smoothquant", "omniquant"}
SPARSEGPT_DEFAULT_BLOCKSIZE = 128
SPARSEGPT_DEFAULT_PERCDAMP = 0.01
OFFICIAL_C4_DATASET = "allenai/c4"
OFFICIAL_C4_CONFIG = "allenai--c4"
OFFICIAL_C4_TRAIN_SHARD = "en/c4-train.00000-of-01024.json.gz"
OFFICIAL_C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"


def run_compression(args) -> Dict:
    _configure_runtime_env(args)
    method = args.compression_method.lower()
    output_dir_value = args.compressed_output_dir or (
        Path("outputs") / "compressed" / f"{args.model_name}_{method}"
    )
    output_dir = Path(output_dir_value).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        if getattr(args, "overwrite_compressed", False):
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(
                f"Compressed output directory already exists and is not empty: {output_dir}. "
                "Use --overwrite-compressed or choose a new --compressed-output-dir."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "dense":
        return _save_dense(args, output_dir)
    if method in PRUNE_METHODS:
        return _run_pruning(args, output_dir, method)
    if method == "gptq":
        return _run_gptq(args, output_dir)
    if method == "awq":
        return _run_awq(args, output_dir)
    if method in EXTERNAL_METHODS:
        if getattr(args, "target_model_adaptation", False):
            return _run_target_model_adaptation(args, output_dir, method)
        return _run_external(args, output_dir, method)
    raise ValueError(f"Unsupported compression method: {method}")


def _paper_protocol_fields(args, attested: bool = True) -> Dict[str, Any]:
    protocol = getattr(args, "paper_protocol", "")
    if not protocol:
        return {}
    return {
        "paper_protocol": protocol,
        "paper_protocol_attested": attested,
    }


def _base_model_fields(args) -> Dict[str, Any]:
    if not getattr(args, "base_model_repository", ""):
        return {}
    return {
        "base_model_repository": args.base_model_repository,
        "base_model_revision": args.base_model_revision or None,
        "base_model_attestation_path": args.base_model_attestation_path or None,
        "base_model_attested": bool(args.base_model_attestation_path),
    }


def _configure_runtime_env(args) -> None:
    if getattr(args, "hf_endpoint", None):
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if getattr(args, "cache_dir", None):
        cache = str(Path(args.cache_dir).expanduser())
        os.environ.setdefault("HF_HOME", cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache) / "transformers"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(cache) / "hub"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _load_tokenizer(model_path: str, cache_dir: Optional[str], revision: Optional[str]):
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


def _load_model(args):
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _resolve_dtype(args.dtype),
        "low_cpu_mem_usage": True,
    }
    if args.cache_dir:
        kwargs["cache_dir"] = str(Path(args.cache_dir).expanduser())
    if args.revision:
        kwargs["revision"] = args.revision
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if getattr(args, "max_memory", None):
        kwargs["max_memory"] = _parse_max_memory(args.max_memory)
    if getattr(args, "attn_implementation", None):
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    model.eval()
    if not args.device_map and args.device:
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


def _save_dense(args, output_dir: Path) -> Dict:
    return _write_manifest(
        output_dir,
        args,
        {
            "backend": "source_checkpoint_reference",
            "compressed_model_path": args.model_path,
            "materialized": False,
            **_base_model_fields(args),
            **_paper_protocol_fields(args),
            "note": "Dense baseline reuses the source checkpoint to avoid duplicating 8B weights.",
        },
    )


def _calibration_texts(args) -> List[str]:
    requested = int(args.calibration_samples)
    if requested <= 0:
        raise ValueError(f"calibration_samples must be positive, got {requested}")
    if args.calibration_file:
        path = Path(args.calibration_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file does not exist: {path}")
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(texts) < requested:
            raise ValueError(
                f"Calibration file {path} contains {len(texts)} non-empty samples, "
                f"but --calibration-samples={requested}. Provide at least {requested} "
                "representative samples or lower --calibration-samples explicitly."
            )
        return texts[:requested]

    defaults = [
        "Large language model compression reduces memory and bandwidth cost on edge devices.",
        "The benchmark compares compressed model accuracy, deployment speed and resource usage.",
        "Quantization maps model weights to lower precision values while pruning removes weights.",
        "MMLU, ARC, HellaSwag, PIQA and WinoGrande measure general language understanding.",
    ]
    texts: List[str] = []
    while len(texts) < requested:
        texts.extend(defaults)
    return texts[:requested]


def _text_calibration_examples(args, tokenizer) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    examples = []
    for text in _calibration_texts(args):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=args.seqlen,
        )
        examples.append({key: value.cpu() for key, value in encoded.items()})
    return examples, {
        "calibration_protocol": "explicit_text_file",
        "calibration_source": "text_file" if args.calibration_file else "bootstrap_text",
        "calibration_seed": None,
        "calibration_dataset": None,
        "calibration_dataset_config": None,
        "calibration_data_file": None,
        "calibration_token_length": args.seqlen,
    }


def _official_c4_calibration_examples(
    args,
    tokenizer,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for --pruning-calibration-source c4. "
            "Install the base LLMCBench requirements before running pruning."
        ) from exc

    requested = int(args.calibration_samples)
    if requested <= 0:
        raise ValueError(f"calibration_samples must be positive, got {requested}")
    dataset_name = args.calibration_dataset or OFFICIAL_C4_DATASET
    dataset_config = args.calibration_dataset_config or OFFICIAL_C4_CONFIG
    data_file = args.calibration_c4_data_file or OFFICIAL_C4_TRAIN_SHARD
    revision = args.calibration_c4_revision or OFFICIAL_C4_REVISION
    cache_dir = str(Path(args.calibration_dataset_cache or args.cache_dir or ".").expanduser())
    endpoint = args.hf_endpoint or os.environ.get("HF_ENDPOINT")
    local_file = hf_hub_download(
        repo_id=dataset_name,
        repo_type="dataset",
        filename=data_file,
        revision=revision,
        cache_dir=cache_dir,
        endpoint=endpoint,
    )
    dataset = load_dataset(
        "json",
        data_files={"train": local_file},
        split="train",
        cache_dir=cache_dir,
    )
    if len(dataset) <= 0:
        raise RuntimeError(f"C4 calibration dataset is empty: {dataset_name}/{dataset_config} {data_file}")

    generator = random.Random(args.calibration_seed)
    examples: List[Dict[str, torch.Tensor]] = []
    max_attempts = max(len(dataset) * 4, requested * 64)
    attempts = 0
    while len(examples) < requested:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not draw {requested} C4 samples with at least {args.seqlen + 1} tokens "
                f"from {dataset_name}/{dataset_config} {data_file}; collected {len(examples)}."
            )
        attempts += 1
        record = dataset[generator.randrange(len(dataset))]
        text = record.get("text") if isinstance(record, dict) else None
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if token_ids.shape[1] <= args.seqlen:
            continue
        start = generator.randint(0, token_ids.shape[1] - args.seqlen - 1)
        input_ids = token_ids[:, start : start + args.seqlen].contiguous().cpu()
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            }
        )
    token_hash = hashlib.sha256(
        b"".join(example["input_ids"].numpy().tobytes() for example in examples)
    ).hexdigest()
    return examples, {
        "calibration_protocol": "official_c4_random_token_windows",
        "calibration_source": "c4",
        "calibration_sampling_algorithm": "python_random_document_and_window",
        "calibration_seed": args.calibration_seed,
        "calibration_dataset": dataset_name,
        "calibration_dataset_config": dataset_config,
        "calibration_data_file": data_file,
        "calibration_dataset_revision": revision,
        "calibration_local_file": local_file,
        "calibration_token_length": args.seqlen,
        "calibration_token_sha256": token_hash,
    }


def _pruning_calibration_examples(args, tokenizer) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    source = args.pruning_calibration_source
    if source == "c4":
        return _official_c4_calibration_examples(args, tokenizer)
    if source == "text_file":
        return _text_calibration_examples(args, tokenizer)
    raise ValueError(f"Unsupported pruning calibration source: {source}")


def _find_linear_layers(module: nn.Module, prefix: str = "") -> Dict[str, nn.Linear]:
    layers: Dict[str, nn.Linear] = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            full_name = f"{prefix}.{name}" if prefix and name else name or prefix
            layers[full_name] = child
    return layers


def _linear_modules(model, include_lm_head: bool = False) -> List[Tuple[str, torch.nn.Linear]]:
    modules: List[Tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if not include_lm_head and name.endswith("lm_head"):
                continue
            modules.append((name, module))
    return modules


@torch.no_grad()
def _decoder_layers(model) -> Optional[nn.ModuleList]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    return layers if isinstance(layers, nn.ModuleList) else None


def _model_device(model) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_decoder_calibration_inputs(
    model,
    examples: List[Dict[str, torch.Tensor]],
    device: torch.device,
    buffer_device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    layers = _decoder_layers(model)
    if layers is None:
        raise RuntimeError(
            "Official layerwise Wanda/SparseGPT currently supports decoder models with model.layers, "
            f"but got {type(model).__name__}."
        )
    use_cache = model.config.use_cache
    model.config.use_cache = False
    captured_inputs = None
    captured_kwargs: Dict[str, Any] = {}

    class StopCalibrationCapture(RuntimeError):
        pass

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states: torch.Tensor, *_, **kwargs):
            nonlocal captured_inputs, captured_kwargs
            captured_inputs = hidden_states.detach()
            captured_kwargs = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
                if key in {"attention_mask", "position_ids", "cache_position", "position_embeddings", "use_cache"}
            }
            captured_kwargs["use_cache"] = False
            raise StopCalibrationCapture

    original_first_layer = layers[0]
    layers[0] = Catcher(original_first_layer)
    calibration_inputs = None
    try:
        for index, example in enumerate(examples):
            encoded = {key: value.to(device) for key, value in example.items()}
            try:
                model(**encoded, use_cache=False)
            except StopCalibrationCapture:
                pass
            if captured_inputs is None:
                raise RuntimeError(f"Failed to capture decoder calibration input for sample {index}.")
            if calibration_inputs is None:
                calibration_inputs = torch.empty(
                    (len(examples), *captured_inputs.shape[1:]),
                    dtype=captured_inputs.dtype,
                    device=buffer_device,
                )
            calibration_inputs[index].copy_(captured_inputs[0].to(buffer_device))
    finally:
        layers[0] = original_first_layer
        model.config.use_cache = use_cache
    if calibration_inputs is None:
        raise RuntimeError("No decoder calibration inputs were captured.")
    return calibration_inputs, captured_kwargs


def _decoder_layer_output(layer: nn.Module, hidden_states: torch.Tensor, layer_kwargs: Dict[str, Any]) -> torch.Tensor:
    outputs = layer(hidden_states, **layer_kwargs)
    return outputs[0] if isinstance(outputs, tuple) else outputs


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


@torch.no_grad()
def _prune_generic_fallback(model, examples: List[Dict[str, torch.Tensor]], args, method: str) -> Dict[str, Any]:
    """Small-model fallback kept for CLI smoke tests outside LLaMA/Qwen architectures."""
    device = _model_device(model)
    statistics: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    linears = _linear_modules(model, include_lm_head=args.prune_lm_head)

    def activation_hook(name: str):
        def hook(_, inputs, __):
            activations = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            statistics[name] = statistics.get(name, torch.zeros(activations.shape[1], device=activations.device))
            statistics[name].add_(activations.square().sum(dim=0))
            counts[name] = counts.get(name, 0) + activations.shape[0]

        return hook

    handles = [module.register_forward_hook(activation_hook(name)) for name, module in linears]
    for example in examples:
        model(**{key: value.to(device) for key, value in example.items()}, use_cache=False)
    for handle in handles:
        handle.remove()
    prune_n, prune_m = (2, 4) if args.prune_pattern == "2:4" else (0, 0)
    for name, module in linears:
        scale = torch.sqrt(statistics[name] / max(counts[name], 1)).reshape(1, -1)
        score = module.weight.data.abs().float() * scale
        if method == "sparsegpt":
            score = score.square()
        if prune_n:
            _apply_nm_prune(module.weight.data, score, prune_n=prune_n, group_m=prune_m)
        else:
            _apply_unstructured_prune(module.weight.data, score, args.sparsity)
    return {
        "pruning_algorithm": f"{method}_generic_activation_fallback",
        "sparsegpt_blocksize": None,
        "sparsegpt_percdamp": None,
    }


class _SparseGPT:
    """Official SparseGPT blockwise second-order pruning, adapted for nn.Linear."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.device = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        self.hessian = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=self.device)
        self.sample_count = 0

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor) -> None:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        batch_size = inputs.shape[0]
        flattened = inputs.detach().float().reshape(-1, inputs.shape[-1]).t()
        self.hessian.mul_(self.sample_count / (self.sample_count + batch_size))
        self.sample_count += batch_size
        flattened.mul_(math.sqrt(2.0 / self.sample_count))
        self.hessian.add_(flattened.matmul(flattened.t()))

    @torch.no_grad()
    def faster_prune(
        self,
        sparsity: float,
        prune_n: int = 0,
        group_m: int = 0,
        blocksize: int = SPARSEGPT_DEFAULT_BLOCKSIZE,
        percdamp: float = SPARSEGPT_DEFAULT_PERCDAMP,
    ) -> None:
        weight = self.layer.weight.data.float().clone()
        hessian = self.hessian
        self.hessian = None
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        diagonal = torch.arange(self.columns, device=self.device)
        hessian[diagonal, diagonal] += percdamp * torch.mean(torch.diag(hessian))
        try:
            hessian = torch.linalg.cholesky(hessian)
            hessian = torch.cholesky_inverse(hessian)
            hessian_inverse = torch.linalg.cholesky(hessian, upper=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "SparseGPT Hessian factorization failed. Use more representative calibration samples "
                "or increase --sparsegpt-percdamp."
            ) from exc

        for start in range(0, self.columns, blocksize):
            end = min(start + blocksize, self.columns)
            count = end - start
            weight_block = weight[:, start:end].clone()
            quantized = torch.zeros_like(weight_block)
            errors = torch.zeros_like(weight_block)
            inverse_block = hessian_inverse[start:end, start:end]
            if prune_n == 0:
                metric = weight_block.square() / torch.diag(inverse_block).reshape(1, -1).square()
                threshold_index = min(int(metric.numel() * sparsity), metric.numel() - 1)
                threshold = torch.sort(metric.flatten())[0][threshold_index]
                mask = metric <= threshold
            else:
                mask = torch.zeros_like(weight_block, dtype=torch.bool)

            for column in range(count):
                values = weight_block[:, column]
                diagonal_value = inverse_block[column, column]
                if prune_n and column % group_m == 0:
                    group_end = min(column + group_m, count)
                    group_metric = (
                        weight_block[:, column:group_end].square()
                        / torch.diag(inverse_block)[column:group_end].reshape(1, -1).square()
                    )
                    keep_count = min(prune_n, group_end - column)
                    prune_indices = torch.topk(group_metric, keep_count, dim=1, largest=False).indices
                    mask.scatter_(1, column + prune_indices, True)
                pruned = values.clone()
                pruned[mask[:, column]] = 0
                quantized[:, column] = pruned
                error = (values - pruned) / diagonal_value
                weight_block[:, column:] -= error.unsqueeze(1).matmul(inverse_block[column, column:].unsqueeze(0))
                errors[:, column] = error

            weight[:, start:end] = quantized
            if end < self.columns:
                weight[:, end:] -= errors.matmul(hessian_inverse[start:end, end:])

        self.layer.weight.data.copy_(weight.to(dtype=self.layer.weight.dtype))

    def free(self) -> None:
        self.hessian = None


@torch.no_grad()
def _prune_decoder_layers(
    model,
    examples: List[Dict[str, torch.Tensor]],
    args,
    method: str,
) -> Dict[str, Any]:
    layers = _decoder_layers(model)
    if layers is None:
        raise RuntimeError(
            "Official layerwise Wanda/SparseGPT requires a LLaMA/Qwen-style model.layers backbone. "
            f"Unsupported architecture: {type(model).__name__}."
        )
    device = _model_device(model)
    buffer_device = torch.device(args.pruning_buffer_device)
    hidden_states, layer_kwargs = _prepare_decoder_calibration_inputs(
        model,
        examples,
        device,
        buffer_device,
    )
    prune_n, prune_m = (2, 4) if args.prune_pattern == "2:4" else (0, 0)
    for index, layer in enumerate(layers):
        layer_device = next(layer.parameters()).device
        if layer_device.type == "meta":
            raise RuntimeError(f"Decoder layer {index} is on a meta device; use a concrete device_map for pruning.")
        current_inputs = hidden_states
        current_kwargs = _move_to_device(layer_kwargs, layer_device)
        linears = _find_linear_layers(layer)
        layer_outputs = torch.empty_like(current_inputs, device=buffer_device)
        if method == "wanda":
            activation_norms: Dict[str, torch.Tensor] = {}
            sample_counts: Dict[str, int] = {}

            def wanda_hook(name: str):
                def hook(_, inputs, __):
                    activations = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
                    activation_norms[name] = activation_norms.get(name, torch.zeros(activations.shape[1], device=activations.device))
                    activation_norms[name].add_(activations.square().sum(dim=0))
                    sample_counts[name] = sample_counts.get(name, 0) + activations.shape[0]

                return hook

            handles = [module.register_forward_hook(wanda_hook(name)) for name, module in linears.items()]
            for sample_index in range(current_inputs.shape[0]):
                output = _decoder_layer_output(
                    layer,
                    current_inputs[sample_index : sample_index + 1].to(layer_device),
                    current_kwargs,
                )
                layer_outputs[sample_index].copy_(output[0].to(buffer_device))
            for handle in handles:
                handle.remove()
            for name, module in linears.items():
                score = module.weight.data.abs().float() * torch.sqrt(
                    activation_norms[name] / max(sample_counts[name], 1)
                ).reshape(1, -1)
                if prune_n:
                    _apply_nm_prune(module.weight.data, score, prune_n=prune_n, group_m=prune_m)
                else:
                    _apply_unstructured_prune(module.weight.data, score, args.sparsity)
        else:
            for module in linears.values():
                sparse_layer = _SparseGPT(module)

                def sparsegpt_hook(_, inputs, __):
                    sparse_layer.add_batch(inputs[0])

                handle = module.register_forward_hook(sparsegpt_hook)
                for sample_index in range(current_inputs.shape[0]):
                    _decoder_layer_output(
                        layer,
                        current_inputs[sample_index : sample_index + 1].to(layer_device),
                        current_kwargs,
                    )
                handle.remove()
                sparse_layer.faster_prune(
                    args.sparsity,
                    prune_n=prune_n,
                    group_m=prune_m,
                    blocksize=args.sparsegpt_blocksize,
                    percdamp=args.sparsegpt_percdamp,
                )
                sparse_layer.free()
                torch.cuda.empty_cache()
        for sample_index in range(current_inputs.shape[0]):
            output = _decoder_layer_output(
                layer,
                current_inputs[sample_index : sample_index + 1].to(layer_device),
                current_kwargs,
            )
            layer_outputs[sample_index].copy_(output[0].to(buffer_device))
        hidden_states = layer_outputs.detach()
        torch.cuda.empty_cache()
    return {
        "pruning_algorithm": "wanda_official_layerwise" if method == "wanda" else "sparsegpt_official_second_order",
        "pruning_buffer_device": str(buffer_device),
        "sparsegpt_blocksize": args.sparsegpt_blocksize if method == "sparsegpt" else None,
        "sparsegpt_percdamp": args.sparsegpt_percdamp if method == "sparsegpt" else None,
    }


@torch.no_grad()
def _run_pruning(args, output_dir: Path, method: str) -> Dict:
    tokenizer = _load_tokenizer(args.model_path, args.cache_dir, args.revision)
    model = _load_model(args)
    examples, calibration_metadata = _pruning_calibration_examples(args, tokenizer)
    if args.prune_pattern not in {"unstructured", "2:4"}:
        raise ValueError(f"{method} only supports unstructured or 2:4 pruning, got {args.prune_pattern}")
    if args.prune_lm_head:
        raise ValueError(
            "Official layerwise Wanda/SparseGPT only prunes decoder Linear modules. "
            "--prune-lm-head is not supported by this backend."
        )
    if _decoder_layers(model) is None:
        algorithm_metadata = _prune_generic_fallback(model, examples, args, method)
    else:
        algorithm_metadata = _prune_decoder_layers(model, examples, args, method)
    is_generic_fallback = algorithm_metadata["pruning_algorithm"].endswith("_generic_activation_fallback")

    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(output_dir)
    torch.cuda.empty_cache()
    return _write_manifest(
        output_dir,
        args,
        {
            "backend": "built_in_activation_pruning",
            "sparsity": args.sparsity,
            "sparsity_semantics": "element_zero_fraction",
            "prune_pattern": args.prune_pattern,
            "prune_structure": "semi_structured" if args.prune_pattern == "2:4" else "unstructured",
            "prune_n": 2 if args.prune_pattern == "2:4" else None,
            "prune_m": 4 if args.prune_pattern == "2:4" else None,
            "prune_lm_head": args.prune_lm_head,
            "calibration_samples": len(examples),
            **calibration_metadata,
            **algorithm_metadata,
            **_paper_protocol_fields(args),
            "note": (
                (
                    "Generic activation-statistics fallback for non-LLaMA/Qwen smoke architectures; "
                    "not an official layerwise Wanda reproduction."
                    if is_generic_fallback
                    else "Official layerwise Wanda: |W| times sqrt(mean squared input activations), "
                    "with layerwise calibration-output propagation."
                )
                if method == "wanda"
                else (
                    "Generic activation-statistics fallback for non-LLaMA/Qwen smoke architectures; "
                    "not an official SparseGPT reproduction."
                    if is_generic_fallback
                    else "Official SparseGPT second-order pruning with full per-linear input Hessian, "
                    "Cholesky inverse, blockwise error compensation and layerwise output propagation."
                )
            ),
        },
    )


def _apply_unstructured_prune(weight: torch.Tensor, score: torch.Tensor, sparsity: float) -> None:
    sparsity = min(max(float(sparsity), 0.0), 1.0)
    if sparsity <= 0:
        return
    prune_count = int(score.shape[1] * sparsity)
    if prune_count <= 0:
        return
    prune_count = min(prune_count, score.shape[1])
    prune_indices = torch.sort(score, dim=1, stable=True).indices[:, :prune_count]
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(1, prune_indices, True)
    weight.masked_fill_(mask.to(weight.device), 0)


def _apply_nm_prune(weight: torch.Tensor, score: torch.Tensor, prune_n: int, group_m: int) -> None:
    if prune_n <= 0 or group_m <= 0 or prune_n >= group_m:
        raise ValueError(f"Invalid N:M pruning pattern: prune_n={prune_n}, group_m={group_m}")
    width = score.shape[1]
    full_width = width - (width % group_m)
    if full_width <= 0:
        return
    grouped_scores = score[:, :full_width].reshape(score.shape[0], -1, group_m)
    prune_indices = torch.topk(grouped_scores, k=prune_n, dim=2, largest=False).indices
    grouped_mask = torch.zeros_like(grouped_scores, dtype=torch.bool)
    grouped_mask.scatter_(2, prune_indices, True)
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask[:, :full_width] = grouped_mask.reshape(score.shape[0], full_width)
    weight.masked_fill_(mask.to(weight.device), 0)


def _first_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _quantization_examples(args, tokenizer) -> List[Dict[str, torch.Tensor]]:
    texts = _calibration_texts(args)
    examples = []
    for text in texts:
        examples.append(
            tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.seqlen,
            )
        )
    return examples


def _gptq_calibration_examples(args, tokenizer) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    if args.gptq_calibration_source != "c4":
        raise ValueError(f"Unsupported GPTQ calibration source: {args.gptq_calibration_source}")
    examples, metadata = _official_c4_calibration_examples(args, tokenizer)
    metadata.update(
        {
            "quantization_protocol": "gptq_for_llama_c4_w4_act_order_true_sequential",
            "quantization_source": "gptq",
        }
    )
    return examples, metadata


def _awq_calibration_data(args) -> Tuple[Optional[List[str]], Dict[str, Any]]:
    if args.awq_calibration_source == "pileval":
        return None, {
            "quantization_protocol": "autoawq_pileval_w4_g128",
            "quantization_source": "pileval",
            "quantization_dataset": "mit-han-lab/pile-val-backup",
            "quantization_dataset_split": "validation",
            "quantization_seed": 42,
            "quantization_samples": args.awq_calibration_samples,
            "quantization_token_length": args.awq_calibration_seqlen,
        }
    if args.awq_calibration_source == "text_file":
        return _calibration_texts(args), {
            "quantization_protocol": "awq_text_file_override",
            "quantization_source": "text_file",
            "quantization_dataset": None,
            "quantization_dataset_split": None,
            "quantization_seed": None,
            "quantization_samples": args.awq_calibration_samples,
            "quantization_token_length": args.awq_calibration_seqlen,
        }
    raise ValueError(f"Unsupported AWQ calibration source: {args.awq_calibration_source}")


def _awq_runtime_scale_search_grid() -> Optional[int]:
    try:
        from awq.quantize.quantizer import AwqQuantizer

        source = inspect.getsource(AwqQuantizer._compute_best_scale)
    except (AttributeError, ImportError, OSError, TypeError):
        return None
    matches = re.findall(r"\bn_grid\s*=\s*(\d+)\b", source)
    if len(matches) != 1:
        return None
    return int(matches[0])


def _awq_runtime_calibration_seed() -> Optional[int]:
    try:
        from awq.utils.calib_data import get_calib_dataset

        source = inspect.getsource(get_calib_dataset)
    except (ImportError, OSError, TypeError):
        return None
    matches = re.findall(r"\.shuffle\(\s*seed\s*=\s*(\d+)\s*\)", source)
    if len(matches) != 1:
        return None
    return int(matches[0])


def _run_gptq(args, output_dir: Path) -> Dict:
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError as exc:
        raise RuntimeError("auto-gptq is required for --compression-method gptq") from exc

    tokenizer = _load_tokenizer(args.model_path, args.cache_dir, args.revision)
    quantize_config = BaseQuantizeConfig(
        bits=args.quant_bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        damp_percent=args.gptq_damp_percent,
        static_groups=args.gptq_static_groups,
        sym=not args.gptq_asymmetric,
        true_sequential=args.gptq_true_sequential,
    )
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _resolve_dtype(args.dtype),
        "device_map": args.device_map,
        "cache_dir": args.cache_dir,
    }
    if getattr(args, "max_memory", None):
        load_kwargs["max_memory"] = _parse_max_memory(args.max_memory)
    if getattr(args, "attn_implementation", None):
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_path,
        quantize_config,
        **load_kwargs,
    )
    examples, calibration_metadata = _gptq_calibration_examples(args, tokenizer)
    model.quantize(examples, batch_size=args.calibration_batch_size)
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)
    return _write_manifest(
        output_dir,
        args,
        {
            "backend": "auto_gptq",
            "quant_bits": args.quant_bits,
            "group_size": args.group_size,
            "desc_act": args.desc_act,
            "calibration_samples": len(examples),
            "gptq_damp_percent": args.gptq_damp_percent,
            "gptq_static_groups": args.gptq_static_groups,
            "gptq_symmetric": not args.gptq_asymmetric,
            "gptq_true_sequential": args.gptq_true_sequential,
            **calibration_metadata,
            **_paper_protocol_fields(args),
            "max_memory": args.max_memory,
            "attn_implementation": args.attn_implementation,
        },
    )


def _run_awq(args, output_dir: Path) -> Dict:
    try:
        from awq import AutoAWQForCausalLM
    except ImportError as exc:
        raise RuntimeError("autoawq is required for --compression-method awq") from exc

    runtime_scale_search_grid = _awq_runtime_scale_search_grid()
    runtime_calibration_seed = _awq_runtime_calibration_seed()
    if getattr(args, "require_paper_protocol_attestation", False) and runtime_scale_search_grid != args.awq_grid_size:
        raise RuntimeError(
            "Strict AWQ profile requires the installed AutoAWQ scale-search grid to match "
            f"--awq-grid-size={args.awq_grid_size}, but observed {runtime_scale_search_grid!r}."
        )
    if getattr(args, "require_paper_protocol_attestation", False) and runtime_calibration_seed != args.awq_calibration_seed:
        raise RuntimeError(
            "Strict AWQ profile requires the installed AutoAWQ pileval shuffle seed to match "
            f"--awq-calibration-seed={args.awq_calibration_seed}, but observed {runtime_calibration_seed!r}."
        )
    tokenizer = _load_tokenizer(args.model_path, args.cache_dir, args.revision)
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": args.device_map,
        "safetensors": True,
        "cache_dir": args.cache_dir,
    }
    if getattr(args, "max_memory", None):
        load_kwargs["max_memory"] = _parse_max_memory(args.max_memory)
    if getattr(args, "attn_implementation", None):
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        **load_kwargs,
    )
    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.quant_bits,
        "version": args.awq_version,
    }
    calibration_data, calibration_metadata = _awq_calibration_data(args)
    quantize_kwargs = {
        "quant_config": quant_config,
        "n_parallel_calib_samples": args.calibration_batch_size,
        "max_calib_samples": args.awq_calibration_samples,
        "max_calib_seq_len": args.awq_calibration_seqlen,
    }
    if calibration_data is not None:
        quantize_kwargs["calib_data"] = calibration_data
    else:
        quantize_kwargs["calib_data"] = "pileval"
    model.quantize(tokenizer, **quantize_kwargs)
    model.save_quantized(output_dir, safetensors=True, shard_size=args.max_shard_size)
    tokenizer.save_pretrained(output_dir)
    return _write_manifest(
        output_dir,
        args,
        {
            "backend": "autoawq",
            "quant_bits": args.quant_bits,
            "group_size": args.group_size,
            "awq_version": args.awq_version,
            "calibration_samples": args.awq_calibration_samples,
            "awq_zero_point": True,
            "awq_calibration_source": args.awq_calibration_source,
            "awq_calibration_seed": args.awq_calibration_seed,
            **calibration_metadata,
            "awq_grid_size": args.awq_grid_size,
            "awq_runtime_scale_search_grid": runtime_scale_search_grid,
            "awq_runtime_calibration_seed": runtime_calibration_seed,
            **_paper_protocol_fields(args),
            "max_memory": args.max_memory,
            "attn_implementation": args.attn_implementation,
        },
    )


def _normalized_git_remote(value: str) -> str:
    normalized = value.strip()
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@"):
        host_path = normalized[len("git@") :]
        host, separator, path = host_path.partition(":")
        if separator:
            normalized = f"https://{host}/{path}"
    return normalized.rstrip("/")


def _official_external_implementation(args, method: str) -> Dict[str, Any]:
    code_dir_value = getattr(args, f"{method}_official_code_dir", "")
    expected_repository = getattr(args, f"{method}_official_repository", "")
    expected_commit = getattr(args, f"{method}_official_commit", "")
    if not getattr(args, "require_paper_protocol_attestation", False):
        return {
            "official_code_dir": code_dir_value or None,
            "official_repository": expected_repository or None,
            "official_commit": expected_commit or None,
            "official_implementation_verified": False,
        }
    if not code_dir_value or not expected_repository or not expected_commit:
        raise RuntimeError(
            f"{method} paper-aligned export requires --{method}-official-code-dir, "
            f"--{method}-official-repository and --{method}-official-commit."
        )
    code_dir = Path(code_dir_value).expanduser()
    if not (code_dir / ".git").exists():
        raise RuntimeError(f"{method} official implementation directory is not a Git checkout: {code_dir}")
    commands = {
        "repository": ["git", "-C", str(code_dir), "remote", "get-url", "origin"],
        "commit": ["git", "-C", str(code_dir), "rev-parse", "HEAD"],
    }
    observed: Dict[str, str] = {}
    for key, command in commands.items():
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Could not inspect {method} official implementation {code_dir}: "
                f"{' '.join(command)}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        observed[key] = proc.stdout.strip()
    if _normalized_git_remote(observed["repository"]) != _normalized_git_remote(expected_repository):
        raise RuntimeError(
            f"{method} official implementation repository mismatch: "
            f"expected {expected_repository!r}, got {observed['repository']!r}"
        )
    if observed["commit"] != expected_commit:
        raise RuntimeError(
            f"{method} official implementation commit mismatch: "
            f"expected {expected_commit!r}, got {observed['commit']!r}"
        )
    return {
        "official_code_dir": str(code_dir),
        "official_repository": expected_repository,
        "official_commit": expected_commit,
        "official_implementation_verified": True,
    }


def _external_command_uses_official_checkout(rendered: str, official_code_dir: str) -> bool:
    try:
        tokens = shlex.split(rendered)
    except ValueError:
        return False
    code_dir = Path(official_code_dir).resolve()
    for token in tokens:
        candidate = Path(token).expanduser()
        if not candidate.is_file():
            continue
        try:
            resolved = candidate.resolve()
            resolved.relative_to(code_dir)
        except (OSError, ValueError):
            continue
        if resolved.suffix == ".py" or os.access(resolved, os.X_OK):
            return True
    return False


def _external_protocol_evidence(
    args,
    output_dir: Path,
    method: str,
    implementation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence_path = output_dir / "llmcbench_external_protocol.json"
    if not getattr(args, "require_paper_protocol_attestation", False):
        return {
            "external_protocol_evidence_path": str(evidence_path),
            "external_protocol_evidence_required": False,
        }
    if not evidence_path.is_file():
        raise RuntimeError(
            f"{method} paper-aligned export must write {evidence_path} with the actual "
            "official-command configuration and paper_protocol_attested=true."
        )
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not parse {method} protocol evidence {evidence_path}: {exc}") from exc
    if not isinstance(evidence, dict):
        raise RuntimeError(f"{method} protocol evidence must be a JSON object: {evidence_path}")
    if evidence.get("paper_protocol") != args.paper_protocol or evidence.get("paper_protocol_attested") is not True:
        raise RuntimeError(
            f"{method} protocol evidence does not attest to {args.paper_protocol!r}: {evidence}"
        )
    implementation = implementation or _official_external_implementation(args, method)
    if implementation["official_implementation_verified"]:
        expected_repository = implementation["official_repository"]
        expected_commit = implementation["official_commit"]
        if (
            _normalized_git_remote(str(evidence.get("official_repository") or ""))
            != _normalized_git_remote(expected_repository)
            or evidence.get("official_commit") != expected_commit
        ):
            raise RuntimeError(
                f"{method} protocol evidence must attest to the verified official implementation "
                f"{expected_repository}@{expected_commit}: {evidence}"
            )

    expected_fields = (
        (
            "smoothquant_alpha_selection",
            "smoothquant_calibration_source",
            "smoothquant_calibration_sentences",
            "quant_bits",
        )
        if method == "smoothquant"
        else (
            "quant_bits",
            "group_size",
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
    )
    expected = {
        "smoothquant_alpha": args.smoothquant_alpha,
        "smoothquant_alpha_selection": args.smoothquant_alpha_selection,
        "smoothquant_calibration_source": args.smoothquant_calibration_source,
        "smoothquant_calibration_sentences": args.smoothquant_calibration_sentences,
        "quant_bits": args.quant_bits,
        "group_size": args.group_size,
        "omniquant_abits": args.omniquant_abits,
        "omniquant_epochs": args.omniquant_epochs,
        "omniquant_nsamples": args.omniquant_nsamples,
        "omniquant_seqlen": args.omniquant_seqlen,
        "omniquant_seed": args.omniquant_seed,
        "omniquant_calibration_source": args.omniquant_calibration_source,
        "omniquant_lwc": args.omniquant_lwc,
        "omniquant_let": args.omniquant_let,
        "omniquant_aug_loss": args.omniquant_aug_loss,
    }
    if method == "smoothquant" and args.smoothquant_alpha is not None:
        expected_fields = ("smoothquant_alpha", *expected_fields)
    mismatches = {
        field: {"expected": expected[field], "actual": evidence.get(field)}
        for field in expected_fields
        if evidence.get(field) != expected[field]
    }
    if mismatches:
        raise RuntimeError(
            f"{method} protocol evidence differs from requested paper profile: {mismatches}"
        )
    if method == "smoothquant":
        alpha = evidence.get("smoothquant_alpha")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise RuntimeError(
                f"{method} protocol evidence must record a numeric selected smoothquant_alpha: {evidence}"
            )
    return {
        "external_protocol_evidence_path": str(evidence_path),
        "external_protocol_evidence_required": True,
        "external_protocol_evidence": evidence,
        "official_implementation": implementation,
    }


def _run_external(args, output_dir: Path, method: str) -> Dict:
    command = args.external_command or os.environ.get(f"LLMCBENCH_{method.upper()}_CMD")
    if not command:
        raise RuntimeError(
            f"{method} requires --external-command or LLMCBENCH_{method.upper()}_CMD. "
            "Use placeholders: {model_path}, {output_dir}, {calibration_file}, {seqlen}, "
            "{calibration_samples}, {calibration_batch_size}, {quant_bits}, {group_size}, "
            "{sparsity}, {max_memory}, {attn_implementation}, {cache_dir}."
        )
    implementation = _official_external_implementation(args, method)
    values = {
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "calibration_file": args.calibration_file or "",
        "seqlen": str(args.seqlen),
        "calibration_samples": str(args.calibration_samples),
        "calibration_batch_size": str(args.calibration_batch_size),
        "quant_bits": str(args.quant_bits),
        "group_size": str(args.group_size),
        "sparsity": str(args.sparsity),
        "prune_pattern": args.prune_pattern,
        "paper_protocol": args.paper_protocol,
        "smoothquant_alpha": "" if args.smoothquant_alpha is None else str(args.smoothquant_alpha),
        "smoothquant_alpha_selection": args.smoothquant_alpha_selection,
        "smoothquant_calibration_source": args.smoothquant_calibration_source,
        "smoothquant_calibration_sentences": str(args.smoothquant_calibration_sentences),
        "omniquant_abits": str(args.omniquant_abits),
        "omniquant_epochs": str(args.omniquant_epochs),
        "omniquant_nsamples": str(args.omniquant_nsamples),
        "omniquant_seqlen": str(args.omniquant_seqlen),
        "omniquant_seed": str(args.omniquant_seed),
        "omniquant_calibration_source": args.omniquant_calibration_source,
        "omniquant_lwc": "1" if args.omniquant_lwc else "0",
        "omniquant_let": "1" if args.omniquant_let else "0",
        "omniquant_aug_loss": "1" if args.omniquant_aug_loss else "0",
        "smoothquant_official_code_dir": args.smoothquant_official_code_dir or "",
        "smoothquant_official_repository": args.smoothquant_official_repository or "",
        "smoothquant_official_commit": args.smoothquant_official_commit or "",
        "omniquant_official_code_dir": args.omniquant_official_code_dir or "",
        "omniquant_official_repository": args.omniquant_official_repository or "",
        "omniquant_official_commit": args.omniquant_official_commit or "",
        "require_paper_protocol_attestation": "1" if args.require_paper_protocol_attestation else "0",
        "max_memory": args.max_memory or "",
        "attn_implementation": args.attn_implementation or "",
        "cache_dir": args.cache_dir or "",
    }
    rendered = command.format(**values)
    official_code_dir = implementation.get("official_code_dir")
    if official_code_dir and not _external_command_uses_official_checkout(rendered, official_code_dir):
        raise RuntimeError(
            f"{method} paper-aligned external command must directly execute a Python or executable file "
            f"inside its verified official code directory "
            f"{official_code_dir}: {rendered!r}"
        )
    proc = subprocess.run(shlex.split(rendered), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{method} external command failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    produced_files = [
        path
        for path in output_dir.iterdir()
        if path.name not in {"llmcbench_compression_manifest.json", "llmcbench_external_protocol.json"}
    ]
    if not produced_files:
        raise RuntimeError(f"{method} command succeeded but did not write checkpoint files to {output_dir}")
    protocol_evidence = _external_protocol_evidence(args, output_dir, method, implementation)

    selected_smoothquant_alpha = (
        protocol_evidence.get("external_protocol_evidence", {}).get("smoothquant_alpha")
        if method == "smoothquant"
        else args.smoothquant_alpha
    )
    method_fields = {
        "sparsity": args.sparsity,
        "sparsity_semantics": "method_config",
        "prune_pattern": args.prune_pattern,
        "prune_structure": None,
        "prune_lm_head": None,
        "quant_bits": args.quant_bits,
        "group_size": None if method == "smoothquant" else args.group_size,
        "smoothquant_alpha": selected_smoothquant_alpha,
        "smoothquant_alpha_selection": args.smoothquant_alpha_selection,
        "smoothquant_calibration_source": args.smoothquant_calibration_source,
        "smoothquant_calibration_sentences": args.smoothquant_calibration_sentences,
        "omniquant_abits": args.omniquant_abits,
        "omniquant_epochs": args.omniquant_epochs,
        "omniquant_nsamples": args.omniquant_nsamples,
        "omniquant_seqlen": args.omniquant_seqlen,
        "omniquant_seed": args.omniquant_seed,
        "omniquant_calibration_source": args.omniquant_calibration_source,
        "omniquant_lwc": args.omniquant_lwc,
        "omniquant_let": args.omniquant_let,
        "omniquant_aug_loss": args.omniquant_aug_loss,
    }
    manifest = _write_manifest(
        output_dir,
        args,
        {
            "backend": "external_command",
            **method_fields,
            "calibration_samples": args.calibration_samples,
            "max_memory": args.max_memory,
            "attn_implementation": args.attn_implementation,
            "external_command": rendered,
            "external_returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            **_paper_protocol_fields(
                args,
                attested=protocol_evidence.get("external_protocol_evidence_required") is not True
                or bool(protocol_evidence["external_protocol_evidence"].get("paper_protocol_attested")),
            ),
            **protocol_evidence,
        },
    )
    return manifest


def _run_target_model_adaptation(args, output_dir: Path, method: str) -> Dict:
    metadata = export_adapted_quantization(args, output_dir, method)
    calibration = metadata["calibration"]
    return _write_manifest(
        output_dir,
        args,
        {
            "backend": "target_model_adaptation",
            "target_model_adaptation": True,
            "target_model_adaptation_disclaimer": metadata["adaptation_disclaimer"],
            "runtime_loading": metadata["runtime_loading"],
            "storage_quantization": metadata["storage_quantization"],
            "adaptation_calibration_source": calibration["calibration_source"],
            "adaptation_calibration_seed": calibration.get("calibration_seed"),
            "adaptation_calibration_samples": calibration["calibration_samples"],
            "adaptation_calibration_sequence_length": calibration["calibration_sequence_length"],
            "adaptation_calibration_processed_batches": calibration["calibration_processed_batches"],
            "adaptation_calibration_activation_statistics": calibration["calibration_activation_statistics"],
            "quant_bits": metadata["weight_bits"],
            "group_size": metadata["group_size"],
            "smoothquant_alpha": metadata["smoothquant_alpha"],
            "smoothquant_alpha_selection": args.smoothquant_alpha_selection,
            "smoothquant_calibration_source": args.smoothquant_calibration_source,
            "smoothquant_calibration_sentences": args.smoothquant_calibration_sentences,
            "omniquant_abits": args.omniquant_abits,
            "omniquant_epochs": args.omniquant_epochs,
            "omniquant_nsamples": args.omniquant_nsamples,
            "omniquant_seqlen": args.omniquant_seqlen,
            "omniquant_seed": args.omniquant_seed,
            "omniquant_calibration_source": args.omniquant_calibration_source,
            "omniquant_lwc": args.omniquant_lwc,
            "omniquant_let": args.omniquant_let,
            "omniquant_aug_loss": args.omniquant_aug_loss,
            "calibration_samples": args.calibration_samples,
            "external_command": None,
            "external_returncode": 0,
            "adapted_quantization_metadata": metadata,
            **_paper_protocol_fields(args, attested=False),
        },
    )


def _write_manifest(output_dir: Path, args, extra: Dict) -> Dict:
    manifest = {
        "model_name": args.model_name,
        "source_model_path": args.model_path,
        "compressed_model_path": str(output_dir),
        "compression_method": args.compression_method,
        "revision": args.revision,
        "dtype": args.dtype,
        "seqlen": args.seqlen,
        "calibration_file": str(Path(args.calibration_file).expanduser()) if args.calibration_file else None,
        "calibration_samples": args.calibration_samples,
        "calibration_batch_size": args.calibration_batch_size,
        **_base_model_fields(args),
        **_paper_protocol_fields(args),
        "max_memory": args.max_memory,
        "attn_implementation": args.attn_implementation,
        "materialized": True,
        **extra,
    }
    (output_dir / "llmcbench_compression_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest

