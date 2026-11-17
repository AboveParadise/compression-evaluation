import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_env_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        name = match.group("name")
        default = match.group("default")
        return os.environ.get(name, default or "")

    expanded = re.sub(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}", replace, value)
    return os.path.expandvars(expanded)


def _effective_defaults(config: Dict[str, Any], model: str, method: str) -> Dict[str, Any]:
    resolved = dict(config.get("defaults", {}))
    for section_name, key in (
        ("method_overrides", method),
        ("model_overrides", model),
        ("model_method_overrides", f"{model}/{method}"),
    ):
        section = config.get(section_name, {})
        override = section.get(key) if isinstance(section, dict) else None
        if isinstance(override, dict):
            resolved.update(override)
    return {
        key: _expand_env_value(value)
        for key, value in resolved.items()
    }


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


def _official_checkout_violation(model: str, method: str, defaults: Dict[str, Any]) -> Dict[str, Any] | None:
    code_dir_value = defaults.get(f"{method}_official_code_dir")
    expected_repository = defaults.get(f"{method}_official_repository")
    expected_commit = defaults.get(f"{method}_official_commit")
    if not code_dir_value or not expected_repository or not expected_commit:
        return {
            "model": model,
            "method": method,
            "field": f"{method}_official_code_dir",
            "value": code_dir_value,
            "error": "strict external execution requires a pinned official Git checkout, repository and commit",
        }
    code_dir = Path(str(code_dir_value)).expanduser()
    if not (code_dir / ".git").exists():
        return {
            "model": model,
            "method": method,
            "field": f"{method}_official_code_dir",
            "value": str(code_dir),
            "error": "official implementation directory is not a Git checkout",
        }
    observations = {}
    for key, command in (
        ("repository", ["git", "-C", str(code_dir), "remote", "get-url", "origin"]),
        ("commit", ["git", "-C", str(code_dir), "rev-parse", "HEAD"]),
    ):
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            return {
                "model": model,
                "method": method,
                "field": f"{method}_official_code_dir",
                "value": str(code_dir),
                "error": f"could not inspect official checkout: {proc.stderr.strip() or proc.stdout.strip()}",
            }
        observations[key] = proc.stdout.strip()
    if _normalized_git_remote(observations["repository"]) != _normalized_git_remote(str(expected_repository)):
        return {
            "model": model,
            "method": method,
            "field": f"{method}_official_repository",
            "value": observations["repository"],
            "expected": expected_repository,
        }
    if observations["commit"] != expected_commit:
        return {
            "model": model,
            "method": method,
            "field": f"{method}_official_commit",
            "value": observations["commit"],
            "expected": expected_commit,
        }
    return None


def _base_model_violation(model_item: Dict[str, Any]) -> Dict[str, Any] | None:
    model = str(model_item.get("name") or "")
    local_path = Path(str(model_item.get("path") or "")).expanduser()
    repository = str(model_item.get("repository") or "")
    revision = str(model_item.get("revision") or "")
    attestation_name = str(model_item.get("attestation_file") or ".llmcbench_base_model_attestation.json")
    if not repository or not revision:
        return {
            "model": model,
            "field": "base_model_repository",
            "value": repository,
            "error": "strict base model requires pinned repository and revision",
        }
    override_marker = local_path / ".llmcbench_source_override.json"
    if override_marker.exists():
        return {
            "model": model,
            "field": "base_model_path",
            "value": str(local_path),
            "error": "local base model has a source override marker and is not accepted as the pinned official checkpoint",
        }
    attestation_path = local_path / attestation_name
    if not attestation_path.is_file():
        return {
            "model": model,
            "field": "base_model_attestation",
            "value": str(attestation_path),
            "error": "missing strict base-model attestation; run tools/attest_base_models.py after downloading the pinned official revision",
        }
    try:
        payload = _load_json(attestation_path)
    except Exception as exc:
        return {
            "model": model,
            "field": "base_model_attestation",
            "value": str(attestation_path),
            "error": f"invalid base-model attestation: {type(exc).__name__}: {exc}",
        }
    if payload.get("repository") != repository or payload.get("revision") != revision:
        return {
            "model": model,
            "field": "base_model_attestation",
            "value": {"repository": payload.get("repository"), "revision": payload.get("revision")},
            "expected": {"repository": repository, "revision": revision},
        }
    expected_sha256 = payload.get("files_sha256")
    required_files = {
        "config.json": local_path / "config.json",
        "model.safetensors.index.json": local_path / "model.safetensors.index.json",
    }
    if not isinstance(expected_sha256, dict) or set(expected_sha256) != set(required_files):
        return {
            "model": model,
            "field": "base_model_attestation.files_sha256",
            "value": expected_sha256,
            "expected": sorted(required_files),
            "error": "base-model attestation must include SHA256 values for config.json and model.safetensors.index.json",
        }
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing:
        return {
            "model": model,
            "field": "base_model_path",
            "value": str(local_path),
            "error": f"base model is missing attested files: {missing}",
        }
    current_sha256 = {name: _sha256(path) for name, path in required_files.items()}
    if current_sha256 != expected_sha256:
        return {
            "model": model,
            "field": "base_model_attestation.files_sha256",
            "value": current_sha256,
            "expected": expected_sha256,
            "error": "local base-model files no longer match the attestation SHA256 values",
        }
    return None


def _expect(violations: List[Dict[str, Any]], model: str, method: str, defaults: Dict[str, Any], key: str, expected: Any) -> None:
    value = defaults.get(key)
    if value != expected:
        violations.append(
            {
                "model": model,
                "method": method,
                "field": key,
                "value": value,
                "expected": expected,
            }
        )


def _expect_nonempty(violations: List[Dict[str, Any]], model: str, method: str, defaults: Dict[str, Any], key: str) -> None:
    value = defaults.get(key)
    if value in (None, ""):
        violations.append(
            {
                "model": model,
                "method": method,
                "field": key,
                "value": value,
                "error": "a model-specific official recipe or grid-search result is required",
            }
        )


def _expect_result(
    violations: List[Dict[str, Any]],
    model: str,
    method: str,
    row: Dict[str, Any],
    field: str,
    expected: Any,
) -> None:
    if row.get(field) != expected:
        violations.append(
            {
                "model": model,
                "method": method,
                "field": field,
                "value": row.get(field),
                "expected": expected,
            }
        )


def _is_selected(model: str, method: str, only_models: List[str], only_methods: List[str]) -> bool:
    return (not only_models or model in only_models) and (not only_methods or method in only_methods)


def _config_violations(
    config: Dict[str, Any],
    protocols: Dict[str, Any],
    only_models: List[str],
    only_methods: List[str],
    require_official_checkouts: bool,
    require_base_models: bool,
) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for model_item in config.get("models", []):
        model = model_item.get("name")
        if require_base_models and (not only_models or model in only_models):
            violation = _base_model_violation(model_item)
            if violation:
                violations.append(violation)
        for method in config.get("methods", []):
            if not _is_selected(model, method, only_models, only_methods):
                continue
            defaults = _effective_defaults(config, model, method)
            protocol_name = defaults.get("paper_protocol")
            protocol = protocols.get(protocol_name)
            if not isinstance(protocol, dict):
                violations.append(
                    {
                        "model": model,
                        "method": method,
                        "field": "paper_protocol",
                        "value": protocol_name,
                        "error": "protocol is missing from configs/paper_protocols.json",
                    }
                )
                continue
            if protocol.get("method") != method:
                violations.append(
                    {
                        "model": model,
                        "method": method,
                        "field": "paper_protocol",
                        "value": protocol_name,
                        "expected": f"protocol for {method}",
                    }
                )
                continue
            if defaults.get("require_paper_protocol_attestation") is not True:
                violations.append(
                    {
                        "model": model,
                        "method": method,
                        "field": "require_paper_protocol_attestation",
                        "value": defaults.get("require_paper_protocol_attestation"),
                        "expected": True,
                    }
                )
            if method in {"smoothquant", "omniquant"} and defaults.get("strict_model_recipe_available") is not True:
                violations.append(
                    {
                        "model": model,
                        "method": method,
                        "field": "strict_model_recipe_available",
                        "value": defaults.get("strict_model_recipe_available"),
                        "error": defaults.get("strict_model_recipe_reason")
                        or "pinned official implementation has no target-model recipe",
                    }
                )
                continue
            if method in {"wanda", "sparsegpt"}:
                for key, expected in (
                    ("pruning_calibration_source", "c4"),
                    ("calibration_seed", 0),
                    ("calibration_samples", 128),
                    ("seqlen", 2048),
                    ("calibration_dataset", "allenai/c4"),
                    ("calibration_dataset_config", "allenai--c4"),
                    ("calibration_c4_data_file", "en/c4-train.00000-of-01024.json.gz"),
                    ("calibration_c4_revision", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"),
                ):
                    _expect(violations, model, method, defaults, key, expected)
                if method == "wanda":
                    _expect(violations, model, method, defaults, "sparsity", 0.5)
                    _expect(violations, model, method, defaults, "prune_pattern", "unstructured")
                else:
                    for key, expected in (
                        ("sparsity", 0.5),
                        ("prune_pattern", "unstructured"),
                        ("sparsegpt_blocksize", 128),
                        ("sparsegpt_percdamp", 0.01),
                    ):
                        _expect(violations, model, method, defaults, key, expected)
            elif method == "gptq":
                for key, expected in (
                    ("quant_bits", 4),
                    ("group_size", -1),
                    ("gptq_calibration_source", "c4"),
                    ("calibration_seed", 0),
                    ("calibration_samples", 128),
                    ("seqlen", 2048),
                    ("calibration_dataset", "allenai/c4"),
                    ("calibration_dataset_config", "allenai--c4"),
                    ("calibration_c4_data_file", "en/c4-train.00000-of-01024.json.gz"),
                    ("calibration_c4_revision", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"),
                    ("desc_act", True),
                    ("gptq_true_sequential", True),
                    ("gptq_static_groups", False),
                    ("gptq_asymmetric", True),
                    ("gptq_damp_percent", 0.01),
                ):
                    _expect(violations, model, method, defaults, key, expected)
            elif method == "awq":
                for key, expected in (
                    ("quant_bits", 4),
                    ("group_size", 128),
                    ("awq_calibration_source", "pileval"),
                    ("awq_calibration_seed", 42),
                    ("awq_calibration_samples", 128),
                    ("awq_calibration_seqlen", 512),
                    ("awq_grid_size", 20),
                ):
                    _expect(violations, model, method, defaults, key, expected)
            elif method == "smoothquant":
                for key, expected in (
                    ("quant_bits", 8),
                    ("smoothquant_calibration_source", "pile_validation"),
                    ("smoothquant_calibration_sentences", 512),
                    ("smoothquant_official_repository", "https://github.com/mit-han-lab/smoothquant"),
                    ("smoothquant_official_commit", "c61476d728e42ae0d8a35e7e78494edcac3237b5"),
                ):
                    _expect(violations, model, method, defaults, key, expected)
                alpha = defaults.get("smoothquant_alpha")
                selection = defaults.get("smoothquant_alpha_selection")
                if selection not in {"official_recipe", "official_grid_search"}:
                    violations.append(
                        {
                            "model": model,
                            "method": method,
                            "field": "smoothquant_alpha_selection",
                            "value": selection,
                            "expected": "official_recipe or official_grid_search",
                        }
                    )
                if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
                    violations.append(
                        {
                            "model": model,
                            "method": method,
                            "field": "smoothquant_alpha",
                            "value": alpha,
                            "error": (
                                "strict execution requires a numeric alpha from an official model recipe "
                                "or a completed, attested official grid search"
                            ),
                        }
                    )
                if require_official_checkouts:
                    violation = _official_checkout_violation(model, method, defaults)
                    if violation:
                        violations.append(violation)
            elif method == "omniquant":
                for key, expected in (
                    ("quant_bits", 4),
                    ("group_size", -1),
                    ("omniquant_official_repository", "https://github.com/OpenGVLab/OmniQuant"),
                    ("omniquant_official_commit", "feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14"),
                    ("omniquant_abits", 16),
                    ("omniquant_epochs", 20),
                    ("omniquant_nsamples", 128),
                    ("omniquant_seqlen", 2048),
                    ("omniquant_seed", 2),
                    ("omniquant_calibration_source", "wikitext2"),
                    ("omniquant_lwc", True),
                    ("omniquant_let", False),
                    ("omniquant_aug_loss", False),
                ):
                    _expect(violations, model, method, defaults, key, expected)
                if require_official_checkouts:
                    violation = _official_checkout_violation(model, method, defaults)
                    if violation:
                        violations.append(violation)
    return violations


def _result_violations(
    config: Dict[str, Any],
    results: List[Dict[str, Any]],
    only_models: List[str],
    only_methods: List[str],
) -> List[Dict[str, Any]]:
    rows = {(row.get("model_name"), row.get("compression_method")): row for row in results if isinstance(row, dict)}
    violations: List[Dict[str, Any]] = []
    for model_item in config.get("models", []):
        model = model_item.get("name")
        for method in config.get("methods", []):
            if not _is_selected(model, method, only_models, only_methods):
                continue
            row = rows.get((model, method))
            defaults = _effective_defaults(config, model, method)
            expected_protocol = defaults.get("paper_protocol")
            if row is None:
                violations.append({"model": model, "method": method, "error": "result row is missing"})
                continue
            if row.get("compression_paper_protocol") != expected_protocol:
                violations.append(
                    {
                        "model": model,
                        "method": method,
                        "field": "compression_paper_protocol",
                        "value": row.get("compression_paper_protocol"),
                        "expected": expected_protocol,
                    }
                )
            if row.get("compression_paper_protocol_attested") is not True:
                violations.append(
                    {
                        "model": model,
                        "method": method,
                        "field": "compression_paper_protocol_attested",
                        "value": row.get("compression_paper_protocol_attested"),
                        "expected": True,
                    }
                )
            expected_repository = model_item.get("repository")
            expected_revision = model_item.get("revision")
            if expected_repository:
                _expect_result(
                    violations,
                    model,
                    method,
                    row,
                    "compression_base_model_repository",
                    expected_repository,
                )
            if expected_revision:
                _expect_result(
                    violations,
                    model,
                    method,
                    row,
                    "compression_base_model_revision",
                    expected_revision,
                )
            _expect_result(
                violations,
                model,
                method,
                row,
                "compression_base_model_attested",
                True,
            )
            if method == "dense":
                continue
            if method in {"wanda", "sparsegpt"}:
                for field, expected in (
                    ("compression_calibration_source", "c4"),
                    ("compression_calibration_seed", 0),
                    ("compression_calibration_samples", 128),
                    ("compression_calibration_token_length", 2048),
                    ("compression_calibration_dataset", "allenai/c4"),
                    ("compression_calibration_dataset_config", "allenai--c4"),
                    ("compression_calibration_data_file", "en/c4-train.00000-of-01024.json.gz"),
                    ("compression_calibration_dataset_revision", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"),
                    ("compression_sparsity", 0.5),
                    ("compression_prune_pattern", "unstructured"),
                ):
                    _expect_result(violations, model, method, row, field, expected)
                if method == "sparsegpt":
                    for field, expected in (
                        ("compression_sparsegpt_blocksize", 128),
                        ("compression_sparsegpt_percdamp", 0.01),
                    ):
                        _expect_result(violations, model, method, row, field, expected)
            elif method == "gptq":
                for field, expected in (
                    ("compression_quant_bits", 4),
                    ("compression_group_size", -1),
                    ("compression_calibration_source", "c4"),
                    ("compression_calibration_seed", 0),
                    ("compression_calibration_samples", 128),
                    ("compression_calibration_token_length", 2048),
                    ("compression_calibration_dataset", "allenai/c4"),
                    ("compression_calibration_dataset_config", "allenai--c4"),
                    ("compression_calibration_data_file", "en/c4-train.00000-of-01024.json.gz"),
                    ("compression_calibration_dataset_revision", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"),
                    ("compression_desc_act", True),
                    ("compression_gptq_true_sequential", True),
                    ("compression_gptq_symmetric", False),
                    ("compression_gptq_damp_percent", 0.01),
                    ("compression_gptq_static_groups", False),
                ):
                    _expect_result(violations, model, method, row, field, expected)
            elif method == "awq":
                for field, expected in (
                    ("compression_quant_bits", 4),
                    ("compression_group_size", 128),
                    ("compression_awq_calibration_source", "pileval"),
                    ("compression_awq_calibration_seed", 42),
                    ("compression_quantization_dataset", "mit-han-lab/pile-val-backup"),
                    ("compression_quantization_dataset_split", "validation"),
                    ("compression_quantization_seed", 42),
                    ("compression_quantization_samples", 128),
                    ("compression_quantization_token_length", 512),
                    ("compression_awq_grid_size", 20),
                    ("compression_awq_runtime_scale_search_grid", 20),
                    ("compression_awq_runtime_calibration_seed", 42),
                    ("compression_awq_zero_point", True),
                ):
                    _expect_result(violations, model, method, row, field, expected)
            elif method == "smoothquant":
                for field, expected in (
                    ("compression_quant_bits", 8),
                    ("compression_smoothquant_calibration_source", "pile_validation"),
                    ("compression_smoothquant_calibration_sentences", 512),
                    ("compression_official_repository", "https://github.com/mit-han-lab/smoothquant"),
                    ("compression_official_commit", "c61476d728e42ae0d8a35e7e78494edcac3237b5"),
                    ("compression_official_implementation_verified", True),
                ):
                    _expect_result(violations, model, method, row, field, expected)
                alpha = row.get("compression_smoothquant_alpha")
                if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
                    violations.append(
                        {
                            "model": model,
                            "method": method,
                            "field": "compression_smoothquant_alpha",
                            "value": alpha,
                            "error": "final exported result requires a selected numeric alpha",
                        }
                    )
                expected_selection = defaults.get("smoothquant_alpha_selection")
                _expect_result(
                    violations,
                    model,
                    method,
                    row,
                    "compression_smoothquant_alpha_selection",
                    expected_selection,
                )
                configured_alpha = defaults.get("smoothquant_alpha")
                if configured_alpha is not None:
                    _expect_result(
                        violations,
                        model,
                        method,
                        row,
                        "compression_smoothquant_alpha",
                        configured_alpha,
                    )
            elif method == "omniquant":
                for field, expected in (
                    ("compression_quant_bits", 4),
                    ("compression_group_size", -1),
                    ("compression_official_repository", "https://github.com/OpenGVLab/OmniQuant"),
                    ("compression_official_commit", "feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14"),
                    ("compression_official_implementation_verified", True),
                    ("compression_omniquant_abits", 16),
                    ("compression_omniquant_epochs", 20),
                    ("compression_omniquant_nsamples", 128),
                    ("compression_omniquant_seqlen", 2048),
                    ("compression_omniquant_seed", 2),
                    ("compression_omniquant_calibration_source", "wikitext2"),
                    ("compression_omniquant_lwc", True),
                    ("compression_omniquant_let", False),
                    ("compression_omniquant_aug_loss", False),
                ):
                    _expect_result(violations, model, method, row, field, expected)
    return violations


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    config = _load_json(Path(args.config).expanduser())
    protocol_payload = _load_json(Path(args.protocols).expanduser())
    protocols = protocol_payload.get("protocols", {}) if isinstance(protocol_payload, dict) else {}
    config_violations = _config_violations(
        config,
        protocols,
        args.only_model,
        args.only_method,
        args.require_official_checkouts,
        args.require_base_models,
    )
    result_violations: List[Dict[str, Any]] = []
    if args.results:
        results = _load_json(Path(args.results).expanduser())
        if not isinstance(results, list):
            result_violations.append({"field": "results", "error": "matrix results must be a JSON list"})
        else:
            result_violations = _result_violations(config, results, args.only_model, args.only_method)
    report = {
        "ok": not config_violations and not result_violations,
        "config": args.config,
        "protocols": args.protocols,
        "results": args.results or None,
        "config_violations": config_violations,
        "result_violations": result_violations,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).expanduser().write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["ok"]:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paper/official-implementation compression protocol settings.")
    parser.add_argument("--config", default="configs/edge_paper_aligned_matrix.json")
    parser.add_argument("--protocols", default="configs/paper_protocols.json")
    parser.add_argument("--results", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--only-model", action="append", default=[])
    parser.add_argument("--only-method", action="append", default=[])
    parser.add_argument(
        "--require-official-checkouts",
        action="store_true",
        help="Require strict external rows to point at matching pinned official Git checkouts.",
    )
    parser.add_argument(
        "--require-base-models",
        action="store_true",
        help="Require local base models to have matching pinned-repository attestations and no source override marker.",
    )
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
