"""Method presets used by the standalone compression launcher."""

from __future__ import annotations

from typing import Dict, List


METHODS = ("wanda", "sparsegpt", "gptq", "awq", "smoothquant", "omniquant")


def method_arguments(method: str, setting: str) -> List[str]:
    """Return the real edge_benchmark CLI arguments for a method preset."""
    if method not in METHODS:
        raise ValueError(f"Unsupported compression method: {method}")
    if setting not in {"paper", "adapted"}:
        raise ValueError(f"Unsupported setting: {setting}")
    if setting == "paper" and method in {"smoothquant", "omniquant"}:
        raise ValueError(
            f"{method} has no upstream LLaMA3-8B/Qwen3-8B target-model recipe. "
            "Use --setting adapted to create an explicitly labelled engineering-adaptation checkpoint."
        )

    common = ["--calibration-batch-size", "1"]
    if method in {"wanda", "sparsegpt"}:
        values = [
            "--pruning-calibration-source",
            "c4",
            "--calibration-seed",
            "0",
            "--calibration-samples",
            "128",
            "--seqlen",
            "2048",
            "--sparsity",
            "0.5",
            "--prune-pattern",
            "unstructured",
            "--pruning-buffer-device",
            "cpu",
        ]
        if method == "sparsegpt":
            values.extend(["--sparsegpt-blocksize", "128", "--sparsegpt-percdamp", "0.01"])
        return common + values
    if method == "gptq":
        return common + [
            "--gptq-calibration-source",
            "c4",
            "--calibration-seed",
            "0",
            "--calibration-samples",
            "128",
            "--seqlen",
            "2048",
            "--quant-bits",
            "4",
            "--group-size",
            "-1",
            "--gptq-damp-percent",
            "0.01",
            "--gptq-asymmetric",
            "--gptq-true-sequential",
        ]
    if method == "awq":
        return common + [
            "--quant-bits",
            "4",
            "--group-size",
            "128",
            "--awq-calibration-source",
            "pileval",
            "--awq-calibration-seed",
            "42",
            "--awq-calibration-samples",
            "128",
            "--awq-calibration-seqlen",
            "512",
            "--awq-grid-size",
            "20",
            "--awq-version",
            "GEMM",
        ]
    if method == "smoothquant":
        return common + [
            "--target-model-adaptation",
            "--quant-bits",
            "8",
            "--smoothquant-alpha",
            "0.85",
            "--smoothquant-alpha-selection",
            "adapted_fixed_alpha",
            "--smoothquant-calibration-source",
            "pile_validation",
            "--smoothquant-calibration-sentences",
            "512",
        ]
    return common + [
        "--target-model-adaptation",
        "--quant-bits",
        "4",
        "--group-size",
        "128",
        "--omniquant-abits",
        "16",
        "--omniquant-epochs",
        "20",
        "--omniquant-nsamples",
        "128",
        "--omniquant-seqlen",
        "2048",
        "--omniquant-seed",
        "2",
        "--omniquant-calibration-source",
        "wikitext2",
        "--omniquant-lwc",
    ]


def setting_label(method: str, setting: str) -> str:
    if setting == "paper":
        return "paper_or_official_profile"
    if method in {"smoothquant", "omniquant"}:
        return "target_model_adaptation_non_paper_reproduction"
    return "official_profile_transfer"


def default_setting(method: str) -> str:
    return "adapted" if method in {"smoothquant", "omniquant"} else "paper"

