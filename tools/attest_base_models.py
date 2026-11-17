import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected(models: List[Dict[str, Any]], names: List[str]) -> List[Dict[str, Any]]:
    if not names:
        return models
    wanted = set(names)
    return [model for model in models if model.get("name") in wanted]


def _attest_model(model: Dict[str, Any], defaults: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download

    name = str(model.get("name") or "")
    local_path = Path(str(model.get("path") or "")).expanduser()
    repository = str(model.get("repository") or "")
    revision = str(model.get("revision") or "")
    attestation_name = str(model.get("attestation_file") or ".llmcbench_base_model_attestation.json")
    attestation_path = local_path / attestation_name
    result: Dict[str, Any] = {
        "model": name,
        "local_path": str(local_path),
        "repository": repository,
        "revision": revision,
        "attestation_path": str(attestation_path),
        "ok": False,
    }
    if not repository or not revision:
        result["error"] = "model repository and revision are required for strict base-model attestation"
        return result
    override_marker = local_path / ".llmcbench_source_override.json"
    if override_marker.exists():
        result["error"] = "local model directory has a source override marker and is not accepted as the official base model"
        result["source_override"] = _load_json(override_marker)
        return result
    files = {
        "config.json": local_path / "config.json",
        "model.safetensors.index.json": local_path / "model.safetensors.index.json",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        result["error"] = f"local model is missing required files: {missing}"
        return result
    if dry_run:
        result.update({"ok": True, "dry_run": True, "expected_local_files": {name: str(path) for name, path in files.items()}})
        return result
    endpoint = str(defaults.get("hf_endpoint") or "https://hf-mirror.com")
    cache_dir = str(defaults.get("hf_cache_dir") or ".")
    try:
        remote = {
            name: Path(
                hf_hub_download(
                    repo_id=repository,
                    filename=name,
                    revision=revision,
                    cache_dir=cache_dir,
                    endpoint=endpoint,
                )
            )
            for name in files
        }
    except Exception as exc:
        result["error"] = f"could not retrieve pinned official model metadata: {type(exc).__name__}: {exc}"
        return result
    local_sha256 = {name: _sha256(path) for name, path in files.items()}
    remote_sha256 = {name: _sha256(path) for name, path in remote.items()}
    result.update(
        {
            "endpoint": endpoint,
            "local_sha256": local_sha256,
            "remote_sha256": remote_sha256,
            "remote_files": {name: str(path) for name, path in remote.items()},
        }
    )
    if local_sha256 != remote_sha256:
        result["error"] = "local config/index SHA256 does not match the pinned official repository revision"
        return result
    payload = {
        "schema_version": 1,
        "model": name,
        "repository": repository,
        "revision": revision,
        "endpoint": endpoint,
        "local_path": str(local_path),
        "files_sha256": local_sha256,
    }
    attestation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result.update({"ok": True, "attestation": payload})
    return result


def attest(args: argparse.Namespace) -> Dict[str, Any]:
    config = _load_json(Path(args.config).expanduser())
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    models = _selected(config.get("models") or [], args.only_model)
    if not models:
        raise SystemExit("No models selected.")
    rows = [_attest_model(model, defaults, args.dry_run) for model in models]
    report = {"ok": all(row.get("ok") for row in rows), "config": args.config, "rows": rows}
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Attest local strict base models against pinned Hugging Face repository revisions.")
    parser.add_argument("--config", default="configs/edge_paper_aligned_matrix.json")
    parser.add_argument("--only-model", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="")
    attest(parser.parse_args())


if __name__ == "__main__":
    main()
