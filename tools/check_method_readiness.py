import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from verify_edge_env import _weight_status


METHOD_IMPORTS = {
    "gptq": ["auto_gptq"],
    "awq": ["awq"],
}

RUNTIME_IMPORTS = {
    "gptq": [["optimum"], ["auto_gptq", "gptqmodel"]],
    "awq": [["awq"], ["accelerate"]],
    "smoothquant": [["compressed_tensors"]],
}

METHOD_INSTALL_HINTS = {
    "gptq": "Run: bash scripts/setup_optional_backends.sh, or python -m pip install optimum auto-gptq 'gptqmodel==4.0.0; python_version >= \"3.10\"'.",
    "awq": "Run: bash scripts/setup_optional_backends.sh, or python -m pip install autoawq.",
    "smoothquant": "Use the documented Python 3.10 target environment and install compressed-tensors, or configure LLMCBENCH_SMOOTHQUANT_CMD/external_commands.smoothquant for export.",
    "omniquant": "Configure LLMCBENCH_OMNIQUANT_CMD or external_commands.omniquant.",
}

EXTERNAL_METHODS = {"smoothquant", "omniquant"}
EXTERNAL_COMMAND_TEMPLATES = {
    "smoothquant": (
        "export LLMCBENCH_SMOOTHQUANT_CMD='python /path/to/SmoothQuant/export.py "
        "--model {model_path} --output {output_dir} --calib {calibration_file} "
        "--seq-len {seqlen} --samples {calibration_samples} --batch {calibration_batch_size} "
        "--max-memory {max_memory} --attn {attn_implementation}'"
    ),
    "omniquant": (
        "export LLMCBENCH_OMNIQUANT_CMD='python /path/to/OmniQuant/export.py "
        "--model {model_path} --output {output_dir} --calib {calibration_file} "
        "--wbits {quant_bits} --group-size {group_size} --samples {calibration_samples} "
        "--batch {calibration_batch_size} --max-memory {max_memory} --attn {attn_implementation}'"
    ),
}


def _load_config(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selected(items: List[Dict], names: List[str]) -> List[Dict]:
    if not names:
        return items
    wanted = set(names)
    return [item for item in items if item.get("name") in wanted]


def _selected_methods(methods: List[str], names: List[str]) -> List[str]:
    if not names:
        return methods
    wanted = set(names)
    return [method for method in methods if method in wanted]


def _effective_defaults(config: Dict, model_name: str, method: str) -> Dict:
    resolved = dict(config.get("defaults", {}))
    for section_name, key in (
        ("method_overrides", method),
        ("model_overrides", model_name),
        ("model_method_overrides", f"{model_name}/{method}"),
    ):
        section = config.get(section_name, {})
        override = section.get(key) if isinstance(section, dict) else None
        if isinstance(override, dict):
            resolved.update(override)
    return resolved


def _import_ok(name: str) -> Dict:
    try:
        module = importlib.import_module(name)
        return {"name": name, "ok": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"name": name, "ok": False, "error": str(exc)}


def _runtime_blocker(method: str, group: Dict) -> str:
    alternatives = group.get("alternatives") if isinstance(group.get("alternatives"), list) else []
    if method == "smoothquant" and "compressed_tensors" in alternatives and sys.version_info < (3, 10):
        return "target_python_3_10_required_for:compressed_tensors"
    return f"missing_runtime_import:any_of:{','.join(alternatives)}"


def _runtime_import_groups(method: str) -> List[Dict]:
    groups = []
    for alternatives in RUNTIME_IMPORTS.get(method, []):
        imports = [_import_ok(name) for name in alternatives]
        groups.append(
            {
                "alternatives": alternatives,
                "ok": any(item["ok"] for item in imports),
                "imports": imports,
            }
        )
    return groups


def _read_json(path: Path) -> Dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _quantization_metadata(model_path: Path) -> Dict:
    config = _read_json(model_path / "config.json")
    quantize_config = _read_json(model_path / "quantize_config.json")
    quantization_config = config.get("quantization_config") if isinstance(config.get("quantization_config"), dict) else {}
    metadata = quantization_config.get("meta") if isinstance(quantization_config.get("meta"), dict) else {}
    if not metadata and isinstance(quantize_config.get("meta"), dict):
        metadata = quantize_config.get("meta", {})
    quantizers = metadata.get("quantizer") if isinstance(metadata.get("quantizer"), list) else []
    return {
        "quant_method": quantization_config.get("quant_method") or quantize_config.get("quant_method"),
        "checkpoint_format": quantization_config.get("checkpoint_format") or quantize_config.get("checkpoint_format"),
        "bits": quantization_config.get("bits") or quantize_config.get("bits"),
        "group_size": quantization_config.get("group_size") or quantize_config.get("group_size"),
        "quantizer": quantizers,
    }


def _runtime_import_groups_for_checkpoint(method: str, model_path: Path) -> Tuple[List[Dict], Dict, str]:
    metadata = _quantization_metadata(model_path)
    reason = "default"
    if method == "gptq" and any(str(item).lower().startswith("gptqmodel") for item in metadata.get("quantizer", [])):
        groups = [["optimum"], ["gptqmodel"]]
        reason = "gptqmodel_checkpoint_metadata"
    else:
        groups = RUNTIME_IMPORTS.get(method, [])
    runtime_imports = []
    for alternatives in groups:
        imports = [_import_ok(name) for name in alternatives]
        runtime_imports.append(
            {
                "alternatives": alternatives,
                "ok": any(item["ok"] for item in imports),
                "imports": imports,
            }
        )
    return runtime_imports, metadata, reason


def _external_command(config: Dict, method: str) -> str:
    return (
        config.get("external_commands", {}).get(method)
        or os.environ.get(f"LLMCBENCH_{method.upper()}_CMD", "")
    )


def _precompressed_path(config: Dict, model_name: str, method: str) -> str:
    paths = config.get("precompressed_paths", {})
    if not isinstance(paths, dict):
        return ""
    model_paths = paths.get(model_name, {})
    if isinstance(model_paths, dict):
        return model_paths.get(method, "") or ""
    return ""


def check(args: argparse.Namespace) -> Dict:
    config = _load_config(args.config)
    models = _selected(config["models"], args.only_model)
    methods = _selected_methods(config["methods"], args.only_method)
    if not models:
        raise SystemExit("No models selected.")
    if not methods:
        raise SystemExit("No methods selected.")
    rows: List[Dict] = []
    summary = {method: {"ok": 0, "blocked": 0} for method in methods}
    for model in models:
        for method in methods:
            effective_defaults = _effective_defaults(config, model["name"], method)
            target_model_adaptation = bool(effective_defaults.get("target_model_adaptation", False))
            precompressed_path = _precompressed_path(config, model["name"], method)
            model_path = Path(precompressed_path or model["path"]).expanduser()
            uses_precompressed = bool(precompressed_path)
            model_ready = (model_path / "config.json").exists()
            weight_status = _weight_status(model_path)
            weights_ready = weight_status["ok"]
            blockers = []
            imports = [] if uses_precompressed else [_import_ok(name) for name in METHOD_IMPORTS.get(method, [])]
            for item in imports:
                if not item["ok"]:
                    blockers.append(f"missing_compression_import:{item['name']}")
            runtime_imports, quantization_metadata, runtime_requirements_reason = _runtime_import_groups_for_checkpoint(method, model_path)
            for group in runtime_imports:
                if not group["ok"]:
                    blockers.append(_runtime_blocker(method, group))
            external_command = None
            if method in EXTERNAL_METHODS and not uses_precompressed and not target_model_adaptation:
                external_command = _external_command(config, method)
                if not external_command:
                    blockers.append("external_command_not_configured")
            if not model_ready:
                blockers.append("model_config_missing")
            if not weights_ready:
                blockers.append("model_weights_missing")
            remediation = []
            if any(
                item.startswith("missing_compression_import:")
                or item.startswith("missing_runtime_import:")
                or item.startswith("target_python_3_10_required_for:")
                for item in blockers
            ):
                remediation.append(METHOD_INSTALL_HINTS.get(method))
            if "external_command_not_configured" in blockers:
                remediation.append(EXTERNAL_COMMAND_TEMPLATES.get(method))
            if "model_config_missing" in blockers or "model_weights_missing" in blockers:
                remediation.append("Run tools/prepare_edge_assets.py or place the full local checkpoint at the configured model path.")
            ok = not blockers
            summary[method]["ok" if ok else "blocked"] += 1
            rows.append(
                {
                    "model": model["name"],
                    "method": method,
                    "ok": ok,
                    "install_hint": None if uses_precompressed else METHOD_INSTALL_HINTS.get(method),
                    "remediation": [item for item in remediation if item],
                    "model_path": str(model_path),
                    "precompressed_path": precompressed_path or None,
                    "uses_precompressed": uses_precompressed,
                    "target_model_adaptation": target_model_adaptation,
                    "precompressed_note": (
                        "Compression backend/import checks are skipped because this row evaluates an already exported checkpoint."
                        if uses_precompressed
                        else None
                    ),
                    "adaptation_note": (
                        "Built-in target-model adaptation replaces the external-command requirement; "
                        "the exported result is explicitly non-paper-reproduction."
                        if target_model_adaptation
                        else None
                    ),
                    "model_config_exists": model_ready,
                    "model_weights_exist": weights_ready,
                    "weight_status": weight_status,
                    "compression_imports": imports,
                    "runtime_imports": runtime_imports,
                    "runtime_requirements_reason": runtime_requirements_reason,
                    "quantization_metadata": quantization_metadata,
                    "imports": imports,
                    "external_command": external_command,
                    "blockers": blockers,
                }
            )
    report = {"ok": all(row["ok"] for row in rows), "summary": summary, "rows": rows}
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"] and args.fail_on_blocked:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check per-method readiness for the LLMCBench edge matrix.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--only-model", action="append", default=[], help="Repeatable model name filter.")
    parser.add_argument("--only-method", action="append", default=[], help="Repeatable method filter.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main() -> None:
    check(build_parser().parse_args())


if __name__ == "__main__":
    main()
