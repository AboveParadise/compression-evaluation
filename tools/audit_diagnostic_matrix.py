import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_MODELS = ("LLaMA3-8B", "Qwen3-8B")
DEFAULT_METHODS = ("dense", "wanda", "sparsegpt", "gptq", "awq", "smoothquant", "omniquant")
SUPPORTED_METHODS = DEFAULT_METHODS
DEFAULT_TASKS = ("mmlu", "arc_challenge", "hellaswag", "piqa", "winogrande")
TASK_METRICS = {
    "mmlu": ("acc,none",),
    "arc_challenge": ("acc_norm,none", "acc,none"),
    "hellaswag": ("acc_norm,none", "acc,none"),
    "piqa": ("acc_norm,none", "acc,none"),
    "winogrande": ("acc,none",),
}


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(results: Dict[str, Any], task: str) -> Optional[float]:
    metrics = results.get(task)
    if not isinstance(metrics, dict):
        return None
    for key in TASK_METRICS[task]:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _artifact_error(path: Path, keys: List[str]) -> Optional[str]:
    payload = _load(path)
    if not payload:
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _row(model: str, method: str, root: Path) -> Dict[str, Any]:
    job = root / model / method
    accuracy = _load(job / "accuracy.json")
    deployment = _load(job / "deployment.json")
    resource = _load(job / "resource.json")
    compression = _load(job / "compression.json")

    results = accuracy.get("results", {}) if accuracy else {}
    failed_tasks = sorted((accuracy.get("failed_tasks") or {}).keys()) if accuracy else []
    scores = {task: _metric(results, task) for task in DEFAULT_TASKS}
    missing_scores = [task for task, value in scores.items() if value is None]

    deployment_error = _artifact_error(job / "deployment.json", ["deployment_error"])
    resource_error = _artifact_error(job / "resource.json", ["model_load_error"])
    deployment_peak = deployment.get("deployment_memory_measured_peak_gb") if deployment else None
    deployment_limit = deployment.get("deployment_memory_limit_gb") if deployment else None
    deployment_exceeded = deployment.get("deployment_memory_limit_exceeded") if deployment else None
    resource_backend = resource.get("resource_backend") if resource else None

    status = "ok"
    blockers = []
    if not accuracy:
        blockers.append("missing_accuracy")
    if not deployment:
        blockers.append("missing_deployment")
    if not resource:
        blockers.append("missing_resource")
    if failed_tasks:
        blockers.append("failed_accuracy_tasks")
    if missing_scores:
        blockers.append("missing_scores")
    if deployment_error:
        blockers.append("deployment_error")
    if resource_error and resource_backend not in {"model_load", None}:
        blockers.append("resource_error")
    if deployment_exceeded:
        blockers.append("deployment_memory_exceeded")
    if blockers:
        status = "blocked"

    return {
        "model": model,
        "method": method,
        "job_dir": str(job),
        "status": status,
        "blockers": blockers,
        "scores": scores,
        "failed_tasks": failed_tasks,
        "missing_scores": missing_scores,
        "tokens_per_second": deployment.get("tokens_per_second") if deployment else None,
        "deployment_peak_gb": deployment_peak,
        "deployment_limit_gb": deployment_limit,
        "deployment_memory_limit_exceeded": deployment_exceeded,
        "deployment_error": deployment_error,
        "storage_gb": resource.get("storage_gb") if resource else None,
        "resource_backend": resource_backend,
        "resource_error": resource_error,
        "flops": (resource.get("compute_complexity") or {}).get("flops") if resource else None,
        "compression_backend": ((compression or {}).get("manifest") or {}).get("backend"),
        "uses_precompressed": ((compression or {}).get("manifest") or {}).get("backend") == "precompressed_checkpoint",
    }


def audit(root_map: Dict[str, Path], models: List[str], methods: List[str]) -> Dict[str, Any]:
    rows = []
    for model in models:
        root = root_map.get(model)
        if not root:
            rows.append({"model": model, "status": "blocked", "blockers": ["missing_model_root"]})
            continue
        for method in methods:
            rows.append(_row(model, method, root))

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    blocked_rows = [row for row in rows if row.get("status") != "ok"]
    return {
        "ok": not blocked_rows,
        "models": models,
        "methods": methods,
        "row_count": len(rows),
        "ok_row_count": len(ok_rows),
        "blocked_row_count": len(blocked_rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a diagnostic edge matrix output without final 4090 gates.")
    parser.add_argument("--llama-root", type=Path, required=True, help="Output root containing LLaMA3-8B/<method> outputs.")
    parser.add_argument("--qwen-root", type=Path, required=True, help="Output root containing Qwen3-8B/<method> outputs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=DEFAULT_MODELS, default=None)
    parser.add_argument("--method", action="append", choices=SUPPORTED_METHODS, default=None)
    args = parser.parse_args()

    models = args.model or list(DEFAULT_MODELS)
    methods = args.method or list(DEFAULT_METHODS)
    payload = audit(
        {
            "LLaMA3-8B": args.llama_root,
            "Qwen3-8B": args.qwen_root,
        },
        models,
        methods,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": payload["ok"], "output": str(args.output), "blocked_row_count": payload["blocked_row_count"]}, indent=2))


if __name__ == "__main__":
    main()
