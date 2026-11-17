import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from verify_edge_env import _weight_status


OFFICIAL_C4_DATASET = "allenai/c4"
OFFICIAL_C4_TRAIN_SHARD = "en/c4-train.00000-of-01024.json.gz"
OFFICIAL_C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"


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


def _checkpoint_report(name: str, path: Path, extra: Dict = None) -> Dict:
    weight_status = _weight_status(path)
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "has_config": (path / "config.json").exists(),
        "has_weights": weight_status["ok"],
        "checkpoint_ready": (path / "config.json").exists() and weight_status["ok"],
        "weight_status": weight_status,
        **(extra or {}),
    }


def _findmnt_report(path: Path) -> Dict:
    proc = subprocess.run(
        ["findmnt", "-T", str(path), "-J", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout).strip()}
    try:
        payload = json.loads(proc.stdout)
        filesystems = payload.get("filesystems") or []
        if not filesystems:
            return {"ok": False, "error": "no_filesystem"}
        fs = filesystems[0]
        return {
            "ok": True,
            "target": fs.get("target"),
            "source": fs.get("source"),
            "fstype": fs.get("fstype"),
            "options": fs.get("options"),
        }
    except Exception as exc:
        return {"ok": False, "error": f"parse_error:{exc}"}


def _base_block_device(source: Optional[str]) -> Optional[str]:
    if not source:
        return None
    source = source.split("[", 1)[0]
    if source.startswith("/dev/"):
        source = source[len("/dev/") :]
    name = Path(source).name
    if not name:
        return None
    holder = Path("/sys/class/block") / name
    try:
        partition = (holder / "partition").exists()
        if partition:
            parent = (holder / "..").resolve().name
            return parent or name
    except OSError:
        return name
    return name


def _block_device_report(source: Optional[str]) -> Dict:
    device = _base_block_device(source)
    if not device:
        return {"device": None, "rotational": None, "is_non_rotational": None}
    rotational_path = Path("/sys/block") / device / "queue" / "rotational"
    if not rotational_path.exists():
        return {"device": device, "rotational": None, "is_non_rotational": None}
    try:
        rotational = rotational_path.read_text(encoding="utf-8").strip() == "1"
        return {"device": device, "rotational": rotational, "is_non_rotational": not rotational}
    except OSError as exc:
        return {"device": device, "rotational": None, "is_non_rotational": None, "error": str(exc)}


def _disk_report(path_value: str) -> Dict:
    path = Path(path_value).expanduser()
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        mount = _findmnt_report(probe)
        block = _block_device_report(mount.get("source") if mount.get("ok") else None)
        exists = path.exists()
        is_dir = path.is_dir() if exists else None
        is_writable = os.access(path, os.W_OK) if exists else None
        parent_writable = os.access(probe, os.W_OK)
        can_create = is_writable if exists else parent_writable
        return {
            "path": str(path),
            "probe_path": str(probe),
            "exists": exists,
            "is_dir": is_dir,
            "is_writable": is_writable,
            "can_create": can_create,
            "writable_probe_path": str(probe),
            "writable_probe_path_is_writable": parent_writable,
            "free_gb": usage.free / (1024**3),
            "total_gb": usage.total / (1024**3),
            "mount": mount,
            "block_device": block,
        }
    except Exception as exc:
        return {"path": str(path), "exists": path.exists(), "ok": False, "error": str(exc)}


def _file_report(path_value: str) -> Dict:
    path = Path(path_value).expanduser()
    nonempty_lines = None
    line_error = None
    if path.is_file():
        try:
            nonempty_lines = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        except Exception as exc:
            line_error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "nonempty_lines": nonempty_lines,
        "line_error": line_error,
    }


def _c4_calibration_report(defaults: Dict) -> Dict:
    source = defaults.get("pruning_calibration_source", "text_file")
    report = {"source": source, "required": source == "c4", "ok": source != "c4"}
    if source != "c4":
        return report
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        report["error"] = f"missing_huggingface_hub:{exc}"
        return report
    dataset = defaults.get("calibration_dataset", OFFICIAL_C4_DATASET)
    shard = defaults.get("calibration_c4_data_file", OFFICIAL_C4_TRAIN_SHARD)
    revision = defaults.get("calibration_c4_revision", OFFICIAL_C4_REVISION)
    cache_dir = defaults.get("calibration_dataset_cache") or defaults["data_dir"]
    try:
        local_file = hf_hub_download(
            repo_id=dataset,
            repo_type="dataset",
            filename=shard,
            revision=revision,
            cache_dir=str(Path(cache_dir).expanduser()),
            endpoint=defaults.get("hf_endpoint"),
            local_files_only=True,
        )
        path = Path(local_file)
        report.update(
            {
                "ok": path.is_file() and path.stat().st_size > 0,
                "dataset": dataset,
                "dataset_config": defaults.get("calibration_dataset_config"),
                "revision": revision,
                "shard": shard,
                "cache_dir": cache_dir,
                "local_file": local_file,
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    except Exception as exc:
        report.update(
            {
                "dataset": dataset,
                "dataset_config": defaults.get("calibration_dataset_config"),
                "revision": revision,
                "shard": shard,
                "cache_dir": cache_dir,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return report


def audit(args: argparse.Namespace) -> Dict:
    config = _load_config(args.config)
    defaults = config["defaults"]
    models = _selected(config["models"], args.only_model)
    methods = _selected_methods(config.get("methods", []), args.only_method)
    if not models:
        raise SystemExit("No models selected.")
    if not methods:
        raise SystemExit("No methods selected.")

    model_reports = []
    precompressed_paths = config.get("precompressed_paths", {})
    for model in models:
        path = Path(model["path"]).expanduser()
        method_paths = precompressed_paths.get(model["name"], {}) if isinstance(precompressed_paths, dict) else {}
        precompressed_methods = {
            method
            for method in methods
            if isinstance(method_paths, dict) and method_paths.get(method)
        }
        required_for_methods = [method for method in methods if method not in precompressed_methods]
        model_reports.append(
            _checkpoint_report(
                model["name"],
                path,
                {
                    "required": bool(required_for_methods),
                    "required_for_methods": required_for_methods,
                    "covered_by_precompressed_methods": sorted(precompressed_methods),
                },
            )
        )

    selected_names = {model["name"] for model in models}
    precompressed_reports = []
    if isinstance(precompressed_paths, dict):
        for model_name, method_map in precompressed_paths.items():
            if model_name not in selected_names or not isinstance(method_map, dict):
                continue
            for method, raw_path in method_map.items():
                if method not in methods or not raw_path:
                    continue
                precompressed_reports.append(
                    _checkpoint_report(
                        f"{model_name}/{method}",
                        Path(raw_path).expanduser(),
                        {"model": model_name, "method": method},
                    )
                )

    path_reports = {
        "hf_cache_dir": _disk_report(defaults["hf_cache_dir"]),
        "data_dir": _disk_report(defaults["data_dir"]),
        "output_root": _disk_report(defaults["output_root"]),
        "calibration_file": _file_report(defaults["calibration_file"]),
        "prompt_file": _file_report(defaults["prompt_file"]),
    }
    disk_targets = {
        key: value
        for key, value in path_reports.items()
        if isinstance(value, dict) and value.get("free_gb") is not None
    }
    low_disk = [
        {"name": key, "path": value["probe_path"], "free_gb": value["free_gb"], "required_gb": args.min_free_disk_gb}
        for key, value in disk_targets.items()
        if args.min_free_disk_gb and value["free_gb"] < args.min_free_disk_gb
    ]
    unwritable_paths = [
        {
            "name": key,
            "path": value["path"],
            "probe_path": value.get("writable_probe_path") or value.get("probe_path"),
            "exists": value.get("exists"),
            "is_writable": value.get("is_writable"),
            "can_create": value.get("can_create"),
        }
        for key, value in disk_targets.items()
        if value.get("can_create") is False
    ]

    text_calibration_required = any(method in {"gptq", "awq"} for method in methods)
    missing_files = [
        key
        for key in ("prompt_file",)
        if not path_reports[key]["exists"]
    ]
    if text_calibration_required and not path_reports["calibration_file"]["exists"]:
        missing_files.append("calibration_file")
    configured_calibration_samples = defaults.get("calibration_samples")
    calibration_file = path_reports["calibration_file"]
    calibration_samples_ok = (
        not isinstance(configured_calibration_samples, int)
        or isinstance(configured_calibration_samples, bool)
        or configured_calibration_samples <= 0
        or calibration_file.get("nonempty_lines") is not None
        and calibration_file["nonempty_lines"] >= configured_calibration_samples
    )
    calibration_file["required_nonempty_lines"] = configured_calibration_samples
    calibration_file["sample_count_ok"] = calibration_samples_ok
    if not text_calibration_required:
        calibration_samples_ok = True
    c4_calibration = _c4_calibration_report(defaults)
    c4_calibration_required = defaults.get("pruning_calibration_source", "text_file") == "c4" and any(
        method in {"wanda", "sparsegpt"} for method in methods
    )
    report = {
        "ok": (
            all((not item.get("required")) or item["checkpoint_ready"] for item in model_reports)
            and all(item["checkpoint_ready"] for item in precompressed_reports)
            and not missing_files
            and calibration_samples_ok
            and (not c4_calibration_required or c4_calibration["ok"])
            and not low_disk
            and not unwritable_paths
        ),
        "config": args.config,
        "selected_methods": methods,
        "models": model_reports,
        "precompressed": precompressed_reports,
        "paths": path_reports,
        "missing_files": missing_files,
        "calibration_samples": {
            "text_file_required": text_calibration_required,
            "configured": configured_calibration_samples,
            "available_nonempty_lines": calibration_file.get("nonempty_lines"),
            "ok": calibration_samples_ok,
        },
        "pruning_calibration": c4_calibration,
        "low_disk": low_disk,
        "unwritable_paths": unwritable_paths,
        "notes": [
            "metadata-only assets are not enough for compression/evaluation; checkpoint_ready requires config plus weight files.",
            "base model checkpoints are required only for selected methods without precompressed_paths entries.",
            "precompressed_paths entries are checked as runnable checkpoints when configured.",
            "Directory paths must either exist and be writable or have a writable nearest existing parent.",
            "GPTQ/AWQ text calibration files must contain at least defaults.calibration_samples non-empty lines.",
            "When Wanda/SparseGPT use pruning_calibration_source=c4, the pinned local C4 shard must be ready.",
            "Use tools/prepare_edge_assets.py to download assets or create calibration/prompt files.",
        ],
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "output": str(output)}, indent=2))
    if not report["ok"] and args.fail_on_missing:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local LLMCBench edge model/data assets.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--only-model", action="append", default=[], help="Repeatable model name filter.")
    parser.add_argument("--only-method", action="append", default=[], help="Repeatable compression method filter.")
    parser.add_argument("--min-free-disk-gb", type=float, default=0.0)
    parser.add_argument("--output", default="outputs/edge_asset_audit.json")
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser


def main() -> None:
    audit(build_parser().parse_args())


if __name__ == "__main__":
    main()
