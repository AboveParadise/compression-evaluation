import json
from pathlib import Path
from typing import Dict


def _load_json(path: Path) -> Dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _quantization_metadata(path: Path) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "weight_bits": None,
        "activation_bits": None,
        "group_size": None,
        "quant_method": None,
    }
    for filename in ("quantize_config.json", "quant_config.json", "config.json"):
        payload = _load_json(path / filename)
        quantization = payload.get("quantization_config", payload)
        if not isinstance(quantization, dict):
            continue
        config_groups = quantization.get("config_groups")
        if isinstance(config_groups, dict):
            for group in config_groups.values():
                if not isinstance(group, dict):
                    continue
                weights = group.get("weights")
                activations = group.get("input_activations")
                if isinstance(weights, dict):
                    metadata["weight_bits"] = metadata["weight_bits"] or weights.get("num_bits")
                    metadata["group_size"] = metadata["group_size"] or weights.get("group_size")
                if isinstance(activations, dict):
                    metadata["activation_bits"] = metadata["activation_bits"] or activations.get("num_bits")
                break
        metadata["weight_bits"] = (
            metadata["weight_bits"]
            or quantization.get("bits")
            or quantization.get("w_bit")
            or quantization.get("wbits")
        )
        metadata["activation_bits"] = (
            metadata["activation_bits"]
            or quantization.get("act_bits")
            or quantization.get("abits")
        )
        metadata["group_size"] = (
            metadata["group_size"]
            or quantization.get("group_size")
            or quantization.get("q_group_size")
        )
        metadata["quant_method"] = metadata["quant_method"] or quantization.get("quant_method")
    return metadata


def inspect_quantization_storage(path: Path) -> Dict[str, object]:
    path = path.expanduser()
    adapted_metadata = _load_json(path / "llmcbench_adapted_quantization.json") if path.is_dir() else {}
    if adapted_metadata:
        method = adapted_metadata.get("method")
        weight_bits = adapted_metadata.get("weight_bits")
        group_size = adapted_metadata.get("group_size")
        storage_quantization = adapted_metadata.get("storage_quantization")
        if method == "smoothquant" and weight_bits == 8:
            return {
                "format": "llmcbench_adapted_int8_weight",
                "is_real_quantized": True,
                "dtypes": ["I8", "F16"],
                "weight_bits": weight_bits,
                "activation_bits": adapted_metadata.get("activation_bits"),
                "group_size": group_size,
                "quant_method": "smoothquant_target_model_adaptation",
                "runtime_loading": adapted_metadata.get("runtime_loading"),
                "storage_quantization": storage_quantization,
            }
        if method == "omniquant" and weight_bits == 4:
            return {
                "format": "llmcbench_adapted_packed_uint4_weight",
                "is_real_quantized": True,
                "dtypes": ["U8", "F16"],
                "weight_bits": weight_bits,
                "activation_bits": adapted_metadata.get("activation_bits"),
                "group_size": group_size,
                "quant_method": "omniquant_target_model_adaptation",
                "runtime_loading": adapted_metadata.get("runtime_loading"),
                "storage_quantization": storage_quantization,
            }
    safetensor_files = sorted(path.glob("*.safetensors")) if path.is_dir() else []
    metadata = _quantization_metadata(path) if path.is_dir() else {}
    dtypes = set()
    tensor_entries = []
    try:
        from safetensors import safe_open

        for file_path in safetensor_files[:2]:
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                for name in list(handle.keys())[:256]:
                    dtype = str(handle.get_slice(name).get_dtype())
                    tensor_entries.append((name, dtype))
                    dtypes.add(dtype)
    except Exception:
        return {
            "format": "unknown",
            "is_real_quantized": False,
            "dtypes": sorted(dtypes),
            **metadata,
        }

    has_packed_weights = any(name.endswith((".qweight", ".qzeros")) for name, _ in tensor_entries)
    has_integer_weights = any(
        name.endswith(".weight") and dtype in {"I8", "U8", "I16", "I32"}
        for name, dtype in tensor_entries
    )
    if has_packed_weights:
        storage_format = "packed_integer"
    elif has_integer_weights:
        storage_format = "integer_weight"
    elif safetensor_files:
        storage_format = "floating_weight_or_fake_quant"
    else:
        storage_format = "unknown"
    return {
        "format": storage_format,
        "is_real_quantized": storage_format in {"packed_integer", "integer_weight"},
        "dtypes": sorted(dtypes),
        **metadata,
    }
