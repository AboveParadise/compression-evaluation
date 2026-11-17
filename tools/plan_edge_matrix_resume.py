import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PIPELINE_ARTIFACTS = [
    "summary.json",
    "compression.json",
    "accuracy.json",
    "deployment.json",
    "resource.json",
]


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_config(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"config is missing or invalid JSON: {path}")
    return payload


def _selected(items: Iterable[Dict[str, Any]], names: List[str]) -> List[Dict[str, Any]]:
    if not names:
        return list(items)
    wanted = set(names)
    return [item for item in items if item.get("name") in wanted]


def _selected_methods(methods: Iterable[str], names: List[str]) -> List[str]:
    if not names:
        return list(methods)
    wanted = set(names)
    return [method for method in methods if method in wanted]


def _run_report_jobs(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    payload = _load_json(path)
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list):
        return {}
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in jobs:
        if not isinstance(item, dict):
            continue
        model = item.get("model")
        method = item.get("method")
        if isinstance(model, str) and isinstance(method, str):
            records[(model, method)] = item
    return records


def _shell_command(parts: List[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _pair_command(args: argparse.Namespace, model: str, method: str, force: bool) -> str:
    parts = [
        "python",
        "tools/run_edge_matrix.py",
        "--config",
        args.config,
        "--only-model",
        model,
        "--only-method",
        method,
        "--stop-on-error",
    ]
    if args.output_root:
        parts.extend(["--output-root", args.output_root])
    if args.cuda_visible_devices:
        parts.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    if force:
        parts.append("--force")
    return _shell_command(parts)


def _finalize_command(args: argparse.Namespace) -> str:
    parts = [
        "python",
        "tools/run_edge_matrix.py",
        "--config",
        args.config,
        "--finalize-only",
    ]
    if args.output_root:
        parts.extend(["--output-root", args.output_root])
    for model in args.only_model:
        parts.extend(["--only-model", model])
    for method in args.only_method:
        parts.extend(["--only-method", method])
    return _shell_command(parts)


def _job_status(output_root: Path, run_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    artifacts = {name: output_root / name for name in PIPELINE_ARTIFACTS}
    existing = [name for name, path in artifacts.items() if path.exists()]
    missing = [name for name, path in artifacts.items() if not path.exists()]
    status = "complete" if not missing else "not_started"
    if existing and missing:
        status = "incomplete_existing"
    if run_record:
        run_status = run_record.get("status")
        if run_status == "failed":
            status = "incomplete_existing" if run_record.get("reason") == "incomplete_existing" else "failed"
        elif run_status == "skipped_external_not_configured":
            status = "skipped_external_not_configured"
        elif run_status == "dry_run":
            status = "dry_run"
    needs_run = status in {"not_started", "incomplete_existing", "failed", "dry_run"}
    force_required = status in {"incomplete_existing", "failed"}
    return {
        "status": status,
        "complete": status == "complete",
        "needs_run": needs_run,
        "force_required": force_required,
        "existing_artifacts": existing,
        "missing_artifacts": missing,
        "summary_path": str(output_root / "summary.json"),
        "log_path": str(output_root / "job.log"),
        "run_report_status": run_record.get("status") if isinstance(run_record, dict) else None,
        "run_report_reason": run_record.get("reason") if isinstance(run_record, dict) else None,
    }


def build_resume_plan(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config).expanduser()
    config = _load_config(config_path)
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    output_root = Path(args.output_root or defaults.get("output_root", "outputs")).expanduser()
    models = _selected(config.get("models", []), args.only_model)
    methods = _selected_methods(config.get("methods", []), args.only_method)
    run_report_path = Path(args.run_report).expanduser() if args.run_report else output_root / "matrix_run_report.json"
    run_jobs = _run_report_jobs(run_report_path)

    jobs = []
    command_groups: Dict[str, List[str]] = {
        "run_missing": [],
        "rerun_incomplete_or_failed": [],
        "resolve_external": [],
    }
    for model in models:
        model_name = str(model.get("name"))
        for method in methods:
            job_root = output_root / model_name / method
            run_record = run_jobs.get((model_name, method))
            job = {
                "model": model_name,
                "method": method,
                "output_root": str(job_root),
                **_job_status(job_root, run_record),
            }
            if job["status"] == "not_started":
                command_groups["run_missing"].append(_pair_command(args, model_name, method, False))
            elif job["status"] in {"incomplete_existing", "failed"}:
                command_groups["rerun_incomplete_or_failed"].append(_pair_command(args, model_name, method, True))
            elif job["status"] == "skipped_external_not_configured":
                command_groups["resolve_external"].append(
                    f"Configure LLMCBENCH_{method.upper()}_CMD or add precompressed_paths for {model_name}/{method}"
                )
            jobs.append(job)

    counts: Dict[str, int] = {}
    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    needs_run = [job for job in jobs if job["needs_run"]]
    blockers = [
        job
        for job in jobs
        if job["status"] == "skipped_external_not_configured"
    ]
    if jobs and not needs_run and not blockers:
        command_groups["finalize"] = [_finalize_command(args)]
    elif jobs:
        command_groups["finalize_after_resume"] = [_finalize_command(args)]

    return {
        "ok": not blockers,
        "config": str(config_path),
        "output_root": str(output_root),
        "run_report": str(run_report_path),
        "selected_models": [str(model.get("name")) for model in models],
        "selected_methods": methods,
        "expected_job_count": len(jobs),
        "status_counts": counts,
        "complete_job_count": counts.get("complete", 0),
        "needs_run_count": len(needs_run),
        "blocker_count": len(blockers),
        "jobs": jobs,
        "command_groups": {name: commands for name, commands in command_groups.items() if commands},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a resume plan for an LLMCBench edge matrix output root.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--output-root", default="", help="Override matrix output root. Defaults to config defaults.output_root.")
    parser.add_argument("--run-report", default="", help="Override matrix_run_report.json path.")
    parser.add_argument("--only-model", action="append", default=[], help="Repeatable model name filter.")
    parser.add_argument("--only-method", action="append", default=[], help="Repeatable method filter.")
    parser.add_argument("--cuda-visible-devices", default="", help="Include this CUDA_VISIBLE_DEVICES override in generated commands.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan = build_resume_plan(args)
    text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
