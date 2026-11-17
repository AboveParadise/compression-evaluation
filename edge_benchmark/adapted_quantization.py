"""Portable target-model adaptations for external compression methods.

These adapters are intentionally separate from the strict paper/official-code
matrix. They make the external-method rows executable for model families that
the upstream repositories do not directly support, while retaining explicit
provenance in each exported checkpoint.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


ADAPTATION_METADATA = "llmcbench_adapted_quantization.json"
ADAPTATION_WEIGHTS = "llmcbench_adapted_quantized.safetensors"
ADAPTATION_VERSION = 1


def is_adapted_quantization_checkpoint(path_value: str | Path) -> bool:
    path = Path(path_value).expanduser()
    return (path / ADAPTATION_METADATA).is_file() and (path / ADAPTATION_WEIGHTS).is_file()


def _copy_hf_metadata(source: Path, output: Path) -> None:
    ignored = {
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "llmcbench_compression_manifest.json",
        ADAPTATION_METADATA,
        ADAPTATION_WEIGHTS,
    }
    for item in source.iterdir():
        if item.name in ignored or item.suffix in {".safetensors", ".bin"}:
            continue
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, destination)


def _load_source_model(args):
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if getattr(args, "cache_dir", None):
        kwargs["cache_dir"] = str(Path(args.cache_dir).expanduser())
    if getattr(args, "revision", None):
        kwargs["revision"] = args.revision
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    model.eval()
    return model


def _calibration_texts(args, method: str, required: int) -> Tuple[list[str], Dict[str, Any]]:
    calibration_file = getattr(args, "calibration_file", None)
    if calibration_file:
        path = Path(calibration_file).expanduser()
        if path.is_file():
            texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(texts) >= required:
                return texts[:required], {
                    "calibration_source": "explicit_text_file",
                    "calibration_file": str(path),
                    "calibration_samples": required,
                }
            raise RuntimeError(
                f"Explicit adaptation calibration file {path} has only {len(texts)}/{required} "
                "non-empty texts. Remove --calibration-file to use the configured dataset or provide full coverage."
            )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for target-model quantization calibration") from exc
    cache_dir = getattr(args, "cache_dir", None)
    if method == "smoothquant":
        dataset = load_dataset(
            "mit-han-lab/pile-val-backup",
            split="validation",
            cache_dir=cache_dir,
        ).shuffle(seed=42)
        source = "mit-han-lab/pile-val-backup validation"
        seed = 42
    else:
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
            cache_dir=cache_dir,
        ).shuffle(seed=int(getattr(args, "omniquant_seed", 2)))
        source = "wikitext-2-raw-v1 train"
        seed = int(getattr(args, "omniquant_seed", 2))
    texts = []
    for record in dataset:
        value = record.get("text") if isinstance(record, dict) else None
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
            if len(texts) >= required:
                break
    if len(texts) < required:
        raise RuntimeError(f"Collected only {len(texts)}/{required} usable calibration texts from {source}")
    return texts, {
        "calibration_source": source,
        "calibration_seed": seed,
        "calibration_samples": required,
    }


def _packed_token_batches(tokenizer, texts: list[str], sequence_length: int) -> list[torch.Tensor]:
    token_ids: list[int] = []
    batches: list[torch.Tensor] = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(token_ids) >= sequence_length:
            current = token_ids[:sequence_length]
            token_ids = token_ids[sequence_length:]
            batches.append(torch.tensor([current], dtype=torch.long))
    return batches


def _collect_activation_maxima(args, model, tokenizer, method: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    required = (
        int(getattr(args, "smoothquant_calibration_sentences", 512))
        if method == "smoothquant"
        else int(getattr(args, "omniquant_nsamples", 128))
    )
    texts, provenance = _calibration_texts(args, method, required)
    sequence_length = (
        min(int(getattr(args, "seqlen", 2048)), 512)
        if method == "smoothquant"
        else int(getattr(args, "omniquant_seqlen", 2048))
    )
    if method == "smoothquant":
        batches = [
            tokenizer(text, return_tensors="pt", truncation=True, max_length=sequence_length)["input_ids"]
            for text in texts
        ]
    else:
        batches = _packed_token_batches(tokenizer, texts, sequence_length)
        if len(batches) < required:
            raise RuntimeError(
                f"WikiText2 calibration text produced only {len(batches)}/{required} packed "
                f"{sequence_length}-token segments."
            )
        batches = batches[:required]
    device = torch.device(getattr(args, "device", "cuda:0") if torch.cuda.is_available() else "cpu")
    model.to(device)
    maxima: Dict[str, torch.Tensor] = {}
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        def hook(_module, inputs, _output, module_name=name):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            values = inputs[0].detach().abs().reshape(-1, inputs[0].shape[-1]).amax(dim=0).float().cpu()
            prior = maxima.get(module_name)
            maxima[module_name] = values if prior is None else torch.maximum(prior, values)

        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            for input_ids in batches:
                model(input_ids=input_ids.to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    provenance.update(
        {
            "calibration_sequence_length": sequence_length,
            "calibration_processed_batches": len(batches),
            "calibration_activation_statistics": "per_linear_input_absolute_max",
        }
    )
    return maxima, provenance


def _smoothquant_weight(
    weight: torch.Tensor,
    input_max: torch.Tensor | None,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = weight.detach().to(dtype=torch.float32, device="cpu")
    if input_max is None:
        input_max = torch.ones(values.shape[1], dtype=torch.float32)
    weight_max = values.abs().amax(dim=0).clamp(min=1e-8)
    activation_max = input_max.to(dtype=torch.float32).clamp(min=1e-8)
    input_scale = activation_max.pow(alpha) / weight_max.pow(1.0 - alpha)
    input_scale = input_scale / input_scale.max().clamp(min=1e-8)
    scaled = values * input_scale.unsqueeze(0)
    scale = scaled.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    quantized = torch.round(scaled / scale).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.to(torch.float16).contiguous(), input_scale.to(torch.float16).contiguous()


def _omniquant_weight(
    weight: torch.Tensor,
    group_size: int,
    input_max: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = weight.detach().to(dtype=torch.float32, device="cpu")
    if values.shape[1] % group_size:
        raise ValueError(
            f"OmniQuant adaptation requires input width divisible by group size {group_size}, "
            f"got tensor shape {tuple(values.shape)}."
        )
    grouped = values.reshape(values.shape[0], -1, group_size)
    if input_max is not None:
        activation = input_max.to(dtype=torch.float32).reshape(-1, group_size).amax(dim=1)
        activation = activation / activation.amax().clamp(min=1e-8)
        clip_factor = 0.90 + 0.10 * activation
        maximum_abs = grouped.abs().amax(dim=2, keepdim=True) * clip_factor.view(1, -1, 1)
        grouped = grouped.clamp(min=-maximum_abs, max=maximum_abs)
    minimum = grouped.amin(dim=2, keepdim=True)
    maximum = grouped.amax(dim=2, keepdim=True)
    scale = ((maximum - minimum) / 15.0).clamp(min=1e-8)
    zero = torch.round(-minimum / scale).clamp(0, 15)
    integers = torch.round(grouped / scale + zero).clamp(0, 15).to(torch.uint8)
    flattened = integers.reshape(values.shape[0], values.shape[1])
    packed = (flattened[:, 0::2] | (flattened[:, 1::2] << 4)).contiguous()
    return packed, scale.squeeze(2).to(torch.float16).contiguous(), zero.squeeze(2).to(torch.float16).contiguous()


def export_adapted_quantization(args, output_dir: Path, method: str) -> Dict[str, Any]:
    if method not in {"smoothquant", "omniquant"}:
        raise ValueError(f"Unsupported adapted quantization method: {method}")
    source = Path(args.model_path).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Adaptation source model path does not exist: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_hf_metadata(source, output_dir)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=False)
    tokenizer.save_pretrained(output_dir)
    model = _load_source_model(args)
    activation_maxima, calibration_provenance = _collect_activation_maxima(args, model, tokenizer, method)
    tensors: Dict[str, torch.Tensor] = {}
    tensor_formats: Dict[str, str] = {}
    modules = dict(model.named_modules())
    group_size = int(getattr(args, "group_size", -1))
    if method == "omniquant" and group_size <= 0:
        group_size = 128
    for name, tensor in model.state_dict().items():
        key_prefix = name.replace(".", "__")
        if tensor.is_floating_point() and tensor.ndim == 2:
            module_name = name[: -len(".weight")] if name.endswith(".weight") else ""
            input_max = activation_maxima.get(module_name)
            if method == "smoothquant":
                if not isinstance(modules.get(module_name), nn.Linear):
                    input_max = None
                quantized, scale, input_scale = _smoothquant_weight(
                    tensor,
                    input_max,
                    (
                        float(getattr(args, "smoothquant_alpha", 0.85) or 0.85)
                        if isinstance(modules.get(module_name), nn.Linear)
                        else 0.0
                    ),
                )
                tensors[f"q__{key_prefix}"] = quantized
                tensors[f"s__{key_prefix}"] = scale
                if isinstance(modules.get(module_name), nn.Linear):
                    tensors[f"i__{key_prefix}"] = input_scale
                    tensor_formats[name] = "int8_per_output_channel_smoothquant"
                else:
                    tensor_formats[name] = "int8_per_output_channel_symmetric_fallback"
            else:
                packed, scale, zero = _omniquant_weight(tensor, group_size, input_max)
                tensors[f"q__{key_prefix}"] = packed
                tensors[f"s__{key_prefix}"] = scale
                tensors[f"z__{key_prefix}"] = zero
                tensor_formats[name] = f"uint4_groupwise_asymmetric_g{group_size}_packed"
        else:
            tensors[f"f__{key_prefix}"] = tensor.detach().to(device="cpu", dtype=torch.float16).contiguous()
            tensor_formats[name] = "fp16_passthrough"
    save_file(tensors, str(output_dir / ADAPTATION_WEIGHTS), metadata={"format": "llmcbench_adapted_quantization"})
    metadata = {
        "schema_version": ADAPTATION_VERSION,
        "method": method,
        "source_model_path": str(source),
        "runtime_loading": "dequantize_to_fp16_transformers",
        "storage_quantization": "real_int8_weights" if method == "smoothquant" else "real_packed_uint4_weights",
        "weight_bits": 8 if method == "smoothquant" else 4,
        "activation_bits": 16,
        "group_size": None if method == "smoothquant" else group_size,
        "target_model_adaptation": True,
        "adaptation_disclaimer": (
            "Target-model adaptation derived from a nearby upstream method profile; "
            "not an upstream same-model recipe or a paper reproduction."
        ),
        "smoothquant_alpha": getattr(args, "smoothquant_alpha", None) if method == "smoothquant" else None,
        "omniquant_lwc": bool(getattr(args, "omniquant_lwc", False)) if method == "omniquant" else None,
        "omniquant_let": bool(getattr(args, "omniquant_let", False)) if method == "omniquant" else None,
        "omniquant_epochs": getattr(args, "omniquant_epochs", None) if method == "omniquant" else None,
        "tensor_formats": tensor_formats,
        "calibration": calibration_provenance,
    }
    (output_dir / ADAPTATION_METADATA).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del model
    return metadata


def _metadata(path: Path) -> Dict[str, Any]:
    return json.loads((path / ADAPTATION_METADATA).read_text(encoding="utf-8"))


def _source_model_path(path: Path) -> str:
    metadata = _metadata(path)
    source = metadata.get("source_model_path")
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"Adapted checkpoint has no usable source_model_path: {path}")
    return source


def dequantize_state_dict(path_value: str | Path, dtype: torch.dtype) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    path = Path(path_value).expanduser()
    metadata = _metadata(path)
    method = metadata["method"]
    group_size = metadata.get("group_size")
    tensors = load_file(str(path / ADAPTATION_WEIGHTS), device="cpu")
    state: Dict[str, torch.Tensor] = {}
    input_scales: Dict[str, torch.Tensor] = {}
    for tensor_name, tensor in tensors.items():
        prefix, encoded_name = tensor_name.split("__", 1)
        name = encoded_name.replace("__", ".")
        if prefix == "f":
            state[name] = tensor.to(dtype=dtype)
        elif prefix == "q":
            scale = tensors[f"s__{encoded_name}"].to(dtype=torch.float32)
            if method == "smoothquant":
                state[name] = (tensor.to(torch.float32) * scale).to(dtype=dtype)
            else:
                zero = tensors[f"z__{encoded_name}"].to(dtype=torch.float32)
                packed = tensor.to(torch.uint8)
                unpacked = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.float32)
                unpacked[:, 0::2] = (packed & 0x0F).to(torch.float32)
                unpacked[:, 1::2] = (packed >> 4).to(torch.float32)
                groups = unpacked.reshape(unpacked.shape[0], -1, int(group_size))
                restored = (groups - zero.unsqueeze(2)) * scale.unsqueeze(2)
                state[name] = restored.reshape(unpacked.shape).to(dtype=dtype)
        elif prefix == "i":
            input_scales[name] = tensor.to(dtype=dtype)
    return state, input_scales


class _InputScaledLinear(nn.Module):
    def __init__(self, linear: nn.Linear, input_scale: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", linear.weight.detach())
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach())
        self.register_buffer("input_scale", input_scale.to(device=linear.weight.device, dtype=linear.weight.dtype))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.linear(values / self.input_scale, self.weight, self.bias)


def _replace_module(root: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = dotted_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def load_adapted_model(path_value: str | Path, **kwargs):
    path = Path(path_value).expanduser()
    metadata = _metadata(path)
    source = _source_model_path(path)
    dtype = kwargs.get("torch_dtype")
    if not isinstance(dtype, torch.dtype):
        dtype = torch.float16
    load_kwargs = dict(kwargs)
    load_kwargs["torch_dtype"] = dtype
    load_kwargs.pop("revision", None)
    model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
    state, input_scales = dequantize_state_dict(path, dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapted quantization keys: {unexpected[:5]}")
    if missing:
        raise RuntimeError(f"Missing adapted quantization keys: {missing[:5]}")
    if metadata["method"] == "smoothquant":
        modules = dict(model.named_modules())
        for weight_name, input_scale in input_scales.items():
            module_name = weight_name[: -len(".weight")] if weight_name.endswith(".weight") else ""
            linear = modules.get(module_name)
            if not isinstance(linear, nn.Linear):
                raise RuntimeError(f"Could not locate SmoothQuant-adapted Linear module {module_name!r}")
            _replace_module(model, module_name, _InputScaledLinear(linear, input_scale))
    model.eval()
    return model


def adapted_checkpoint_metadata(path_value: str | Path) -> Dict[str, Any]:
    return _metadata(Path(path_value).expanduser())
