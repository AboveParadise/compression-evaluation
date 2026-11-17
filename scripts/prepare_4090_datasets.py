#!/usr/bin/env python3
"""Download and verify the five LLMCBench evaluation dataset families."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


MMLU_CONFIGS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

DATASETS = [
    ("mmlu", "cais/mmlu", MMLU_CONFIGS),
    ("arc", "allenai/ai2_arc", ["ARC-Challenge", "ARC-Easy"]),
    ("hellaswag", "Rowan/hellaswag", [None]),
    ("piqa", "baber/piqa", [None]),
    ("winogrande", "allenai/winogrande", ["winogrande_xl"]),
]


def _summarize(dataset: Any) -> Dict[str, int]:
    return {name: len(split) for name, split in dataset.items()}


def _snapshot_path(snapshot_root: Path, dataset_id: str) -> Path:
    return snapshot_root / dataset_id.replace("/", "--")


def _download_snapshots(
    specs: list[tuple[str, str, list[Optional[str]]]],
    snapshot_root: Path,
    cache_dir: str,
    retries: int,
    retry_delay: float,
) -> Dict[str, str]:
    from huggingface_hub import snapshot_download

    snapshots: Dict[str, str] = {}
    for _family, dataset_id, _configs in specs:
        local_dir = _snapshot_path(snapshot_root, dataset_id)
        for attempt in range(1, retries + 1):
            try:
                print(f"[snapshot] {dataset_id} -> {local_dir} (attempt {attempt}/{retries})", flush=True)
                snapshot_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    local_dir=str(local_dir),
                    cache_dir=cache_dir,
                    max_workers=2,
                )
                snapshots[dataset_id] = str(local_dir)
                break
            except Exception:
                if attempt >= retries:
                    raise
                delay = min(retry_delay * (2 ** (attempt - 1)), 300.0)
                print(f"[retry] {dataset_id} after {delay:.1f}s", flush=True)
                time.sleep(delay)
    return snapshots


def _load_all(
    cache_dir: str,
    offline: bool,
    specs: list[tuple[str, str, list[Optional[str]]]],
    snapshots: Dict[str, str],
) -> list[Dict[str, Any]]:
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    import datasets
    from datasets import DownloadMode, load_dataset

    if offline:
        datasets.config.HF_DATASETS_OFFLINE = True
        try:
            import huggingface_hub.constants

            huggingface_hub.constants.HF_HUB_OFFLINE = True
        except ImportError:
            pass

    mode = DownloadMode.REUSE_DATASET_IF_EXISTS if offline else DownloadMode.REUSE_CACHE_IF_EXISTS
    records = []
    for family, dataset_id, configs in specs:
        for config in configs:
            source = snapshots[dataset_id]
            label = f"{dataset_id}/{config}" if config else dataset_id
            print(f"[{'verify' if offline else 'download'}] {label}", flush=True)
            dataset = load_dataset(
                source,
                config,
                cache_dir=cache_dir,
                download_mode=mode,
            )
            records.append(
                {
                    "family": family,
                    "dataset": dataset_id,
                    "local_snapshot": source,
                    "config": config,
                    "splits": _summarize(dataset),
                    "offline_verified": offline,
                }
            )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="/data/llm_eval_datasets")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--manifest", default="/data/llm_eval_datasets/llmcbench_datasets.json")
    parser.add_argument("--snapshot-root", default="")
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    parser.add_argument(
        "--family",
        action="append",
        choices=[item[0] for item in DATASETS],
        default=[],
        help="Repeat to prepare a subset; defaults to all five dataset families.",
    )
    parser.add_argument("--skip-offline-verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cache = Path(args.cache_dir).expanduser()
    manifest = Path(args.manifest).expanduser()
    snapshot_root = Path(args.snapshot_root).expanduser() if args.snapshot_root else cache / "snapshots"
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_DATASETS_CACHE"] = str(cache)
    wanted = set(args.family)
    specs = [item for item in DATASETS if not wanted or item[0] in wanted]
    plan = [
        {"family": family, "dataset": dataset, "configs": configs}
        for family, dataset, configs in specs
    ]
    if args.dry_run:
        print(json.dumps({"cache_dir": str(cache), "snapshot_root": str(snapshot_root), "manifest": str(manifest), "plan": plan}, ensure_ascii=False, indent=2))
        return
    cache.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshots = _download_snapshots(specs, snapshot_root, str(cache), args.retries, args.retry_delay)
    downloaded = _load_all(str(cache), offline=False, specs=specs, snapshots=snapshots)
    verified = [] if args.skip_offline_verify else _load_all(str(cache), offline=True, specs=specs, snapshots=snapshots)
    payload = {
        "ok": True,
        "hf_endpoint": args.hf_endpoint,
        "cache_dir": str(cache),
        "snapshot_root": str(snapshot_root),
        "snapshots": snapshots,
        "families": [item[0] for item in specs],
        "downloaded": downloaded,
        "offline_verified": verified,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "manifest": str(manifest), "entries": len(downloaded)}, indent=2))


if __name__ == "__main__":
    main()
