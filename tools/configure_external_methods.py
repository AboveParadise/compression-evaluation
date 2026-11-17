import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict


BASE_PLACEHOLDER_VALUES = {
    "model_path": "/data/models/Qwen3-8B",
    "calibration_file": "/data/calibration/wiki_samples.txt",
    "seqlen": "2048",
    "calibration_samples": "128",
    "calibration_batch_size": "1",
    "quant_bits": "4",
    "group_size": "128",
    "sparsity": "0.5",
    "prune_pattern": "structured",
    "max_memory": "0:22GiB,cpu:120GiB",
    "attn_implementation": "eager",
    "cache_dir": "/data/hf_cache",
    "paper_protocol": "",
    "smoothquant_alpha": "",
    "smoothquant_alpha_selection": "official_grid_search",
    "smoothquant_calibration_source": "pile_validation",
    "smoothquant_calibration_sentences": "512",
    "smoothquant_official_code_dir": "",
    "smoothquant_official_repository": "https://github.com/mit-han-lab/smoothquant",
    "smoothquant_official_commit": "",
    "omniquant_abits": "16",
    "omniquant_epochs": "20",
    "omniquant_nsamples": "128",
    "omniquant_seqlen": "2048",
    "omniquant_calibration_source": "wikitext2",
    "omniquant_lwc": "1",
    "omniquant_let": "0",
    "omniquant_aug_loss": "0",
    "omniquant_official_code_dir": "",
    "omniquant_official_repository": "https://github.com/OpenGVLab/OmniQuant",
    "omniquant_official_commit": "",
}


def _load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_env_value(value: str) -> str:
    def replace(match: re.Match) -> str:
        name = match.group("name")
        default = match.group("default")
        return os.environ.get(name, default or "")

    expanded = re.sub(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}", replace, value)
    return os.path.expandvars(expanded)


def _command_for(method: str, args: argparse.Namespace, config: Dict) -> str:
    explicit = getattr(args, f"{method}_command")
    env_value = os.environ.get(f"LLMCBENCH_{method.upper()}_CMD", "")
    configured = config.get("external_commands", {}).get(method, "")
    return explicit or env_value or configured


def _render(method: str, command: str, config: Dict) -> Dict:
    values = dict(BASE_PLACEHOLDER_VALUES)
    defaults = config.get("defaults", {}) if isinstance(config.get("defaults"), dict) else {}
    overrides = config.get("method_overrides", {}) if isinstance(config.get("method_overrides"), dict) else {}
    method_override = overrides.get(method, {}) if isinstance(overrides.get(method), dict) else {}
    models = config.get("models", [])
    model = models[0] if models and isinstance(models[0], dict) else {}
    model_name = str(model.get("name") or "Model")
    values["model_path"] = str(model.get("path") or values["model_path"])
    for key in values:
        configured = method_override.get(key, defaults.get(key))
        if configured is not None:
            values[key] = _expand_env_value(str(configured))
    output_root = str(defaults.get("output_root") or "/data/outputs/llmcbench_edge_4090")
    values["output_dir"] = f"{output_root}/{model_name}/{method}/compressed/{model_name}_{method}"
    try:
        return {"ok": True, "rendered": command.format(**values)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def configure(args: argparse.Namespace) -> Dict:
    config_path = Path(args.config).expanduser()
    config = _load_config(config_path)
    external = dict(config.get("external_commands", {}))
    report = {"config": str(config_path), "dry_run": args.dry_run, "methods": {}}

    configured_methods = set(config.get("methods", []))
    configured_methods.update(
        method for method, command in config.get("external_commands", {}).items() if command
    )
    configured_methods.update(
        method
        for method in ("smoothquant", "omniquant")
        if getattr(args, f"{method}_command") or os.environ.get(f"LLMCBENCH_{method.upper()}_CMD")
    )
    for method in ("smoothquant", "omniquant"):
        if method not in configured_methods:
            continue
        command = _command_for(method, args, config)
        render = _render(method, command, config) if command else {"ok": False, "error": "external_command_not_configured"}
        report["methods"][method] = {
            "configured": bool(command),
            "command": command,
            "render_check": render,
        }
        if command:
            external[method] = command

    report["ok"] = all(item["render_check"]["ok"] for item in report["methods"].values())
    if not args.dry_run:
        config["external_commands"] = external
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output = Path(args.output).expanduser() if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"] and args.fail_on_missing:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configure and validate SmoothQuant/OmniQuant external command templates.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--smoothquant-command", default="")
    parser.add_argument("--omniquant-command", default="")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser


def main() -> None:
    configure(build_parser().parse_args())


if __name__ == "__main__":
    main()
