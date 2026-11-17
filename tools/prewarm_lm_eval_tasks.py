import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MMLU_SUBTASKS = [
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_astronomy",
    "mmlu_business_ethics",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_chemistry",
    "mmlu_college_computer_science",
    "mmlu_college_mathematics",
    "mmlu_college_medicine",
    "mmlu_college_physics",
    "mmlu_computer_security",
    "mmlu_conceptual_physics",
    "mmlu_econometrics",
    "mmlu_electrical_engineering",
    "mmlu_elementary_mathematics",
    "mmlu_formal_logic",
    "mmlu_global_facts",
    "mmlu_high_school_biology",
    "mmlu_high_school_chemistry",
    "mmlu_high_school_computer_science",
    "mmlu_high_school_european_history",
    "mmlu_high_school_geography",
    "mmlu_high_school_government_and_politics",
    "mmlu_high_school_macroeconomics",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_microeconomics",
    "mmlu_high_school_physics",
    "mmlu_high_school_psychology",
    "mmlu_high_school_statistics",
    "mmlu_high_school_us_history",
    "mmlu_high_school_world_history",
    "mmlu_human_aging",
    "mmlu_human_sexuality",
    "mmlu_international_law",
    "mmlu_jurisprudence",
    "mmlu_logical_fallacies",
    "mmlu_machine_learning",
    "mmlu_management",
    "mmlu_marketing",
    "mmlu_medical_genetics",
    "mmlu_miscellaneous",
    "mmlu_moral_disputes",
    "mmlu_moral_scenarios",
    "mmlu_nutrition",
    "mmlu_philosophy",
    "mmlu_prehistory",
    "mmlu_professional_accounting",
    "mmlu_professional_law",
    "mmlu_professional_medicine",
    "mmlu_professional_psychology",
    "mmlu_public_relations",
    "mmlu_security_studies",
    "mmlu_sociology",
    "mmlu_us_foreign_policy",
    "mmlu_virology",
    "mmlu_world_religions",
]


def _split_tasks(tasks: str):
    return [task.strip() for task in tasks.split(",") if task.strip()]


def _expand_tasks(tasks, expand_mmlu: bool):
    expanded = []
    for task in tasks:
        if expand_mmlu and task == "mmlu":
            expanded.extend(DEFAULT_MMLU_SUBTASKS)
        else:
            expanded.append(task)
    return expanded


def _load_config(path: str):
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selected_models(config, only_model, all_models: bool):
    if not config:
        return []
    wanted = set(only_model or [])
    models = [model for model in config.get("models", []) if not wanted or model.get("name") in wanted]
    if all_models:
        return models
    existing = [model for model in models if Path(model.get("path", "")).expanduser().exists()]
    return (existing or models)[:1]


def _config_default(args: argparse.Namespace, config, key: str, current):
    if current not in (None, ""):
        return current
    if not config:
        return current
    return config.get("defaults", {}).get(key, current)


def _lm_eval_memory_args(value: str):
    if not value:
        return []
    if "," not in value and ":" not in value:
        return [f"max_memory_per_gpu={value}"]
    parsed = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, memory = item.split(":", 1)
        parsed[key.strip()] = memory.strip()
    args = []
    gpu_memory = parsed.get("0") or next((memory for key, memory in parsed.items() if key.isdigit()), "")
    if gpu_memory:
        args.append(f"max_memory_per_gpu={gpu_memory}")
    if parsed.get("cpu"):
        args.append(f"max_cpu_memory={parsed['cpu']}")
    return args


def _lm_eval_dtype(dtype: str) -> str:
    mapping = {
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
    }
    return mapping.get(dtype, dtype)


def prewarm(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    args.tasks = _config_default(args, config, "tasks", args.tasks)
    args.hf_endpoint = _config_default(args, config, "hf_endpoint", args.hf_endpoint)
    args.cache_dir = _config_default(args, config, "hf_cache_dir", args.cache_dir)
    args.data_dir = _config_default(args, config, "data_dir", args.data_dir)
    args.dtype = _config_default(args, config, "dtype", args.dtype) or "fp16"
    lm_eval_dtype = _lm_eval_dtype(args.dtype)
    args.max_memory = _config_default(args, config, "max_memory", args.max_memory)
    args.attn_implementation = _config_default(args, config, "attn_implementation", args.attn_implementation)

    targets = []
    if args.model_path:
        targets.append({"name": args.model_name or "manual", "path": args.model_path})
    else:
        targets = _selected_models(config, args.only_model, args.all_models)
    if not targets:
        raise SystemExit("Provide --model-path, or provide --config with at least one selected model.")

    env = os.environ.copy()
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint
    if args.data_dir:
        env["HF_DATASETS_CACHE"] = str(Path(args.data_dir).expanduser())
    if args.cache_dir:
        cache = str(Path(args.cache_dir).expanduser())
        env["HF_HOME"] = cache
        env["TRANSFORMERS_CACHE"] = str(Path(cache) / "transformers")
        env["HUGGINGFACE_HUB_CACHE"] = str(Path(cache) / "hub")

    tasks = _expand_tasks(_split_tasks(args.tasks), args.expand_mmlu)
    failed = []
    attempted = []
    succeeded = []
    for target in targets:
        target_name = target.get("name") or Path(target["path"]).name
        for task in tasks:
            model_args = [
                f"pretrained={target['path']}",
                "trust_remote_code=True",
                f"dtype={lm_eval_dtype}",
            ]
            if args.cache_dir:
                model_args.append(f"cache_dir={Path(args.cache_dir).expanduser()}")
            if args.max_memory:
                model_args.extend(_lm_eval_memory_args(args.max_memory))
            if args.attn_implementation:
                model_args.append(f"attn_implementation={args.attn_implementation}")
            cmd = [
                sys.executable,
                "-m",
                "lm_eval",
                "--model",
                "hf",
                "--model_args",
                ",".join(model_args),
                "--tasks",
                task,
                "--limit",
                str(args.limit),
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
                "--output_path",
                str(Path(args.output_dir).expanduser() / target_name / task),
            ]
            print(f"[prewarm] model={target_name} task={task}")
            attempted.append(
                {
                    "model": target_name,
                    "task": task,
                    "output_path": cmd[-1],
                    "dry_run": args.dry_run,
                    "dtype": args.dtype,
                    "lm_eval_dtype": lm_eval_dtype,
                    "max_memory": args.max_memory or None,
                    "attn_implementation": args.attn_implementation or None,
                }
            )
            ok = False
            for attempt in range(1, args.retries + 1):
                print("+ " + " ".join(cmd) + f"  # attempt {attempt}/{args.retries}")
                if args.dry_run:
                    ok = True
                    break
                proc = subprocess.run(cmd, env=env, text=True, check=False)
                if proc.returncode == 0:
                    ok = True
                    break
                if attempt < args.retries:
                    time.sleep(args.sleep_seconds)
            if not ok:
                failed.append({"model": target_name, "task": task})
                if args.stop_on_error:
                    break
            else:
                succeeded.append({"model": target_name, "task": task})
            if args.sleep_seconds and task != tasks[-1]:
                time.sleep(args.sleep_seconds)
        if failed and args.stop_on_error:
            break
    report = {
        "ok": not failed,
        "config": args.config,
        "tasks": tasks,
        "target_count": len(targets),
        "targets": [{"name": target.get("name"), "path": target.get("path")} for target in targets],
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "dry_run": args.dry_run,
        "expand_mmlu": args.expand_mmlu,
        "dtype": args.dtype,
        "lm_eval_dtype": lm_eval_dtype,
        "max_memory": args.max_memory or None,
        "attn_implementation": args.attn_implementation or None,
        "hf_endpoint": args.hf_endpoint,
        "data_dir": args.data_dir,
        "cache_dir": args.cache_dir,
    }
    if args.report_output:
        output = Path(args.report_output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "attempted": len(attempted), "failed": len(failed), "report_output": args.report_output}, indent=2))
    if failed:
        raise SystemExit(f"Failed to prewarm tasks: {failed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prewarm lm-eval task datasets/cache one task at a time.")
    parser.add_argument("--config", default=None, help="Optional edge matrix config. Defaults fill tasks/cache/data/model paths.")
    parser.add_argument("--only-model", action="append", default=[], help="Repeatable model name filter when --config is used.")
    parser.add_argument("--all-models", action="store_true", help="With --config, prewarm selected tasks for every selected model instead of the first available model.")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--expand-mmlu", action="store_true", help="Expand mmlu into a smaller default list of MMLU subtasks for safer cache prewarming.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default=None, help="Model dtype for lm-eval prewarm. Defaults to config defaults.dtype, then fp16.")
    parser.add_argument("--max-memory", default=None, help="Transformers max_memory for lm-eval model loading. Defaults to config defaults.max_memory.")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--limit", type=float, default=1)
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/lm_eval_prewarm")
    parser.add_argument("--report-output", default=None, help="Write a JSON report with expanded tasks, targets and failures.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=30)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    prewarm(build_parser().parse_args())


if __name__ == "__main__":
    main()
