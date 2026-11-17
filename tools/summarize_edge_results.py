import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


HIGHER_IS_BETTER = ["accuracy_avg_score", "tokens_per_second"]
LOWER_IS_BETTER = ["deployment_memory_measured_peak_gb", "storage_gb", "flops_num"]
DEFAULT_WEIGHTS = {
    "accuracy_avg_score": 0.35,
    "tokens_per_second": 0.25,
    "deployment_memory_measured_peak_gb": 0.20,
    "storage_gb": 0.10,
    "flops_num": 0.10,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _normalize(value: Any, values: List[float], higher_is_better: bool) -> float:
    if not _finite(value) or not values:
        return 0.0
    low = min(values)
    high = max(values)
    if high == low:
        return 1.0
    scaled = (float(value) - low) / (high - low)
    return scaled if higher_is_better else 1.0 - scaled


def _resource_metric_value(row: Dict[str, Any], field: str) -> Any:
    method = str(row.get("compression_method") or "").lower()
    if field == "storage_gb":
        if row.get("quantization_storage_real") is True:
            value = row.get("effective_storage_gb_quantized_approx")
            if _finite(value):
                return value
        if method in {"wanda", "sparsegpt"}:
            value = row.get("effective_storage_gb_sparse_approx")
            if _finite(value):
                return value
        return row.get("storage_gb")
    if field == "flops_num":
        if method in {"wanda", "sparsegpt"}:
            value = row.get("effective_flops_num_sparse_approx")
            if _finite(value):
                return value
        return row.get("flops_num")
    return row.get(field)


def _parse_weights(value: str) -> Dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if not value:
        return weights
    for item in value.split(","):
        if not item.strip():
            continue
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in weights:
            raise SystemExit(f"Unknown score weight field: {key}")
        weights[key] = float(raw)
    total = sum(weights.values())
    if total <= 0:
        raise SystemExit("Score weights must sum to a positive value")
    return {key: weight / total for key, weight in weights.items()}


def _passes_constraints(row: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    failures = []
    if args.min_accuracy_retention is not None:
        value = row.get("accuracy_retention_vs_dense")
        if not _finite(value) or value < args.min_accuracy_retention:
            failures.append("accuracy_retention_vs_dense")
    if args.max_peak_memory_gb is not None:
        value = row.get("deployment_memory_measured_peak_gb")
        if not _finite(value) or value > args.max_peak_memory_gb:
            failures.append("deployment_memory_measured_peak_gb")
    if args.max_storage_gb is not None:
        value = _resource_metric_value(row, "storage_gb")
        if not _finite(value) or value > args.max_storage_gb:
            failures.append("storage_gb")
    if args.min_tokens_per_second is not None:
        value = row.get("tokens_per_second")
        if not _finite(value) or value < args.min_tokens_per_second:
            failures.append("tokens_per_second")
    if args.max_flops_num is not None:
        value = _resource_metric_value(row, "flops_num")
        if not _finite(value) or value > args.max_flops_num:
            failures.append("flops_num")
    return failures


def _dominates(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    better_or_equal = []
    strictly_better = []
    for field in HIGHER_IS_BETTER:
        if not (_finite(left.get(field)) and _finite(right.get(field))):
            return False
        better_or_equal.append(left[field] >= right[field])
        strictly_better.append(left[field] > right[field])
    for field in LOWER_IS_BETTER:
        left_value = _resource_metric_value(left, field)
        right_value = _resource_metric_value(right, field)
        if not (_finite(left_value) and _finite(right_value)):
            return False
        better_or_equal.append(left_value <= right_value)
        strictly_better.append(left_value < right_value)
    return all(better_or_equal) and any(strictly_better)


def _pareto(rows: List[Dict[str, Any]]) -> List[str]:
    frontier = []
    for row in rows:
        method = row.get("compression_method")
        if not isinstance(method, str):
            continue
        dominated = any(_dominates(other, row) for other in rows if other is not row)
        if not dominated:
            frontier.append(method)
    return sorted(frontier)


def _score_rows(rows: List[Dict[str, Any]], weights: Dict[str, float], args: argparse.Namespace) -> List[Dict[str, Any]]:
    values = {
        field: [float(_resource_metric_value(row, field)) for row in rows if _finite(_resource_metric_value(row, field))]
        for field in [*HIGHER_IS_BETTER, *LOWER_IS_BETTER]
    }
    scored = []
    for row in rows:
        components = {}
        for field in HIGHER_IS_BETTER:
            components[field] = _normalize(row.get(field), values[field], True)
        for field in LOWER_IS_BETTER:
            components[field] = _normalize(_resource_metric_value(row, field), values[field], False)
        score = sum(components[field] * weights[field] for field in weights)
        constraint_failures = _passes_constraints(row, args)
        scored.append(
            {
                "model_name": row.get("model_name"),
                "compression_method": row.get("compression_method"),
                "score": score,
                "score_components": components,
                "constraint_failures": constraint_failures,
                "passes_constraints": not constraint_failures,
                "accuracy_avg_score": row.get("accuracy_avg_score"),
                "tokens_per_second": row.get("tokens_per_second"),
                "deployment_memory_measured_peak_gb": row.get("deployment_memory_measured_peak_gb"),
                "storage_gb": row.get("storage_gb"),
                "score_storage_gb": _resource_metric_value(row, "storage_gb"),
                "flops_num": row.get("flops_num"),
                "score_flops_num": _resource_metric_value(row, "flops_num"),
                "accuracy_retention_vs_dense": row.get("accuracy_retention_vs_dense"),
                "resource_backend": row.get("resource_backend"),
                "compute_backend": row.get("compute_backend"),
            }
        )
    return sorted(scored, key=lambda item: (item.get("passes_constraints", False), item.get("score", 0.0)), reverse=True)


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    rows = _load_json(Path(args.results).expanduser())
    if not isinstance(rows, list):
        raise SystemExit("matrix results must be a JSON list")
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("model_name"), str):
            by_model.setdefault(row["model_name"], []).append(row)
    weights = _parse_weights(args.weights)
    models = []
    for model_name, model_rows in sorted(by_model.items()):
        ranked = _score_rows(model_rows, weights, args)
        eligible = [row for row in ranked if row["passes_constraints"]]
        models.append(
            {
                "model_name": model_name,
                "method_count": len(model_rows),
                "pareto_methods": _pareto(model_rows),
                "recommended_method": eligible[0]["compression_method"] if eligible else None,
                "eligible_method_count": len(eligible),
                "ranking": ranked,
            }
        )
    return {
        "schema_version": 1,
        "source_results": str(Path(args.results).expanduser()),
        "scoring": {
            "higher_is_better": HIGHER_IS_BETTER,
            "lower_is_better": LOWER_IS_BETTER,
            "weights": weights,
            "score": "weighted mean of per-model min-max normalized metrics",
            "resource_metric_preference": {
                "storage_gb": {
                    "real_quantized": "effective_storage_gb_quantized_approx",
                    "pruned": "effective_storage_gb_sparse_approx",
                    "other": "storage_gb",
                },
                "flops_num": {
                    "pruned": "effective_flops_num_sparse_approx",
                    "other": "flops_num",
                },
            },
            "constraints": {
                "min_accuracy_retention": args.min_accuracy_retention,
                "max_peak_memory_gb": args.max_peak_memory_gb,
                "max_storage_gb": args.max_storage_gb,
                "min_tokens_per_second": args.min_tokens_per_second,
                "max_flops_num": args.max_flops_num,
            },
        },
        "models": models,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# LLMCBench Edge Result Scorecard",
        "",
        f"- Source results: `{summary.get('source_results')}`",
        "- Score: weighted mean of per-model min-max normalized metrics",
        f"- Weights: `{summary.get('scoring', {}).get('weights')}`",
        f"- Constraints: `{summary.get('scoring', {}).get('constraints')}`",
        "- Higher is better: accuracy_avg_score, tokens_per_second",
        "- Lower is better: deployment_memory_measured_peak_gb, storage_gb, flops_num",
        "",
    ]
    for model in summary.get("models", []):
        lines.extend(
            [
                f"## {model.get('model_name')}",
                "",
                f"- Recommended method: `{model.get('recommended_method')}`",
                f"- Pareto methods: `{', '.join(model.get('pareto_methods', []))}`",
                "",
                "| Rank | Method | Eligible | Score | Avg Accuracy | Retention | Tokens/s | Peak GB | Score Storage GB | Score FLOPs | Constraint Failures |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for index, row in enumerate(model.get("ranking", []), start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        _fmt(row.get("compression_method")),
                        _fmt(row.get("passes_constraints")),
                        _fmt(row.get("score")),
                        _fmt(row.get("accuracy_avg_score")),
                        _fmt(row.get("accuracy_retention_vs_dense")),
                        _fmt(row.get("tokens_per_second")),
                        _fmt(row.get("deployment_memory_measured_peak_gb")),
                        _fmt(row.get("score_storage_gb")),
                        _fmt(row.get("score_flops_num")),
                        _fmt(",".join(row.get("constraint_failures", []))),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize LLMCBench edge results for method selection.")
    parser.add_argument("--results", required=True, help="matrix_results.json")
    parser.add_argument("--output-json", default="", help="Optional scorecard JSON path.")
    parser.add_argument("--output-md", default="", help="Optional scorecard Markdown path.")
    parser.add_argument("--weights", default="", help="Comma-separated score weights, for example accuracy_avg_score=0.5,tokens_per_second=0.2.")
    parser.add_argument("--min-accuracy-retention", type=float, default=None)
    parser.add_argument("--max-peak-memory-gb", type=float, default=None)
    parser.add_argument("--max-storage-gb", type=float, default=None)
    parser.add_argument("--min-tokens-per-second", type=float, default=None)
    parser.add_argument("--max-flops-num", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize(args)
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        output = Path(args.output_json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    markdown = render_markdown(summary)
    if args.output_md:
        output = Path(args.output_md).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown + "\n", encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
