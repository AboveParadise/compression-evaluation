import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _load_config(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _config_default(config: Dict, key: str, fallback: str = "") -> str:
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    return defaults.get(key) or fallback


def _run_step(name: str, cmd: List[str], required: bool, env: Optional[Dict[str, str]] = None) -> Dict:
    print(f"[preflight] {name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
    step = {
        "name": name,
        "cmd": cmd,
        "required": required,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if name == "verify_env":
        try:
            payload = json.loads(proc.stdout)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            step["payload"] = payload
    return step


def _extend_filters(cmd: List[str], args: argparse.Namespace) -> None:
    for model_name in args.only_model:
        cmd.extend(["--only-model", model_name])
    for method in args.only_method:
        cmd.extend(["--only-method", method])


def run_preflight(args: argparse.Namespace) -> Dict:
    steps = []
    config = _load_config(args.config)
    if config.get("require_paper_protocol_validation") is True:
        protocol_cmd = [
            sys.executable,
            "tools/validate_paper_protocol.py",
            "--config",
            args.config,
            "--require-official-checkouts",
            "--require-base-models",
        ]
        _extend_filters(protocol_cmd, args)
        steps.append(_run_step("paper_protocol", protocol_cmd, required=not args.allow_blocked))
    forward_dtype = args.forward_dtype or _config_default(config, "dtype", "fp16")
    forward_attn_implementation = (
        args.forward_attn_implementation
        if args.forward_attn_implementation not in (None, "")
        else _config_default(config, "attn_implementation", "")
    )

    asset_cmd = [
        sys.executable,
        "tools/audit_edge_assets.py",
        "--config",
        args.config,
        "--output",
        str(Path(args.output).expanduser().parent / "edge_asset_audit.json"),
    ]
    for model_name in args.only_model:
        asset_cmd.extend(["--only-model", model_name])
    for method in args.only_method:
        asset_cmd.extend(["--only-method", method])
    steps.append(_run_step("asset_audit", asset_cmd, required=not args.allow_blocked))

    verify_cmd = [
        sys.executable,
        "tools/verify_edge_env.py",
        "--config",
        args.config,
        "--check-cli",
        "--check-resource-backend",
    ]
    if args.cuda_visible_devices:
        verify_cmd.append("--require-cuda-visible-devices-match")
    if args.check_hf_access:
        verify_cmd.append("--check-hf-access")
    for model_name in args.only_model:
        verify_cmd.extend(["--only-model", model_name])
    if args.check_model_forward:
        verify_cmd.extend(
            [
                "--check-model-forward",
                "--forward-dtype",
                forward_dtype,
            ]
        )
        if forward_attn_implementation:
            verify_cmd.extend(["--forward-attn-implementation", forward_attn_implementation])
    verify_env = None
    if args.cuda_visible_devices:
        verify_env = {**dict(os.environ), "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices}
    steps.append(_run_step("verify_env", verify_cmd, required=not args.allow_blocked, env=verify_env))

    readiness_cmd = [
        sys.executable,
        "tools/check_method_readiness.py",
        "--config",
        args.config,
    ]
    _extend_filters(readiness_cmd, args)
    if args.fail_on_blocked_methods:
        readiness_cmd.append("--fail-on-blocked")
    steps.append(_run_step("method_readiness", readiness_cmd, required=args.fail_on_blocked_methods))

    if args.check_optional_backend_resolution:
        optional_report = Path(args.output).expanduser().parent / "edge_optional_backends_report.json"
        optional_cmd = [
            "bash",
            "scripts/setup_optional_backends.sh",
        ]
        env = dict(**{
            "OPTIONAL_BACKEND_DRY_RUN": "1",
            "REPORT_OUTPUT": str(optional_report),
        })
        print(f"[preflight] optional_backend_resolution: {' '.join(optional_cmd)}")
        proc = subprocess.run(
            optional_cmd,
            text=True,
            capture_output=True,
            check=False,
            env={**dict(os.environ), **env},
        )
        steps.append(
            {
                "name": "optional_backend_resolution",
                "cmd": optional_cmd,
                "required": args.fail_on_optional_backend_resolution,
                "returncode": proc.returncode,
                "ok": proc.returncode == 0,
                "report": str(optional_report),
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
        )

    if args.check_quant_runtime:
        quant_report = Path(args.output).expanduser().parent / "edge_quant_runtime_report.json"
        quant_cmd = [
            sys.executable,
            "tools/probe_quant_runtime.py",
            "--config",
            args.config,
            "--output",
            str(quant_report),
            "--cuda-visible-devices",
            args.cuda_visible_devices or "",
        ]
        _extend_filters(quant_cmd, args)
        if args.quant_runtime_generate:
            quant_cmd.append("--generate")
        print(f"[preflight] quant_runtime: {' '.join(quant_cmd)}")
        proc = subprocess.run(quant_cmd, text=True, capture_output=True, check=False)
        steps.append(
            {
                "name": "quant_runtime",
                "cmd": quant_cmd,
                "required": args.fail_on_quant_runtime,
                "returncode": proc.returncode,
                "ok": proc.returncode == 0,
                "report": str(quant_report),
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
        )

    if not args.skip_prewarm_dry_run:
        prewarm_cmd = [
            sys.executable,
            "tools/prewarm_lm_eval_tasks.py",
            "--config",
            args.config,
            "--dry-run",
            "--sleep-seconds",
            "0",
            "--report-output",
            str(Path(args.output).expanduser().parent / "edge_prewarm_dry_run_report.json"),
        ]
        for model_name in args.prewarm_model:
            prewarm_cmd.extend(["--only-model", model_name])
        if args.prewarm_all_models:
            prewarm_cmd.append("--all-models")
        if args.expand_mmlu:
            prewarm_cmd.append("--expand-mmlu")
        if args.prewarm_tasks:
            prewarm_cmd.extend(["--tasks", args.prewarm_tasks])
        steps.append(_run_step("prewarm_dry_run", prewarm_cmd, required=True))

    matrix_cmd = [
        sys.executable,
        "tools/run_edge_matrix.py",
        "--config",
        args.config,
        "--dry-run",
    ]
    matrix_cmd.extend(["--output-root", args.output_root or "/tmp/llmcbench_edge_preflight_dryrun"])
    if args.cuda_visible_devices:
        matrix_cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    if args.finalize_dry_run:
        matrix_cmd.append("--finalize")
    _extend_filters(matrix_cmd, args)
    steps.append(_run_step("matrix_dry_run", matrix_cmd, required=True))

    ok = all(step["ok"] or not step["required"] for step in steps)
    report = {
        "ok": ok,
        "config": args.config,
        "resolved_forward_dtype": forward_dtype,
        "resolved_forward_attn_implementation": forward_attn_implementation or None,
        "steps": steps,
        "notes": [
            "This preflight is non-destructive: it does not compress 8B checkpoints or run full lm-eval.",
            "Use --check-hf-access and --check-model-forward on the target 4090 machine before the final matrix.",
        ],
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": ok, "output": str(output)}, indent=2))
    if not ok:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a non-destructive LLMCBench edge 4090 preflight.")
    parser.add_argument("--config", default="configs/edge_4090_matrix.json")
    parser.add_argument("--output", default="outputs/edge_preflight_report.json")
    parser.add_argument("--only-model", action="append", default=[], help="Repeatable model filter for matrix dry-run.")
    parser.add_argument("--only-method", action="append", default=[], help="Repeatable method filter for matrix dry-run.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--check-hf-access", action="store_true")
    parser.add_argument("--check-model-forward", action="store_true")
    parser.add_argument("--forward-dtype", default=None, help="Forward dtype for --check-model-forward. Defaults to config defaults.dtype, then fp16.")
    parser.add_argument("--forward-attn-implementation", default="")
    parser.add_argument("--allow-blocked", action="store_true", help="Record environment failures without failing the preflight process.")
    parser.add_argument("--fail-on-blocked-methods", action="store_true", help="Treat method readiness blockers as preflight failures.")
    parser.add_argument("--check-optional-backend-resolution", action="store_true", help="Run setup_optional_backends.sh in pip dry-run mode and write edge_optional_backends_report.json.")
    parser.add_argument("--fail-on-optional-backend-resolution", action="store_true", help="Fail preflight if optional backend pip dry-run resolution fails.")
    parser.add_argument("--check-quant-runtime", action="store_true", help="Run GPTQ/AWQ precompressed checkpoint runtime probes and write edge_quant_runtime_report.json.")
    parser.add_argument("--fail-on-quant-runtime", action="store_true", help="Fail preflight if GPTQ/AWQ runtime probes fail.")
    parser.add_argument("--quant-runtime-generate", action="store_true", help="Run one-token generation during GPTQ/AWQ runtime probes.")
    parser.add_argument("--skip-prewarm-dry-run", action="store_true")
    parser.add_argument("--prewarm-model", action="append", default=[], help="Repeatable model filter for prewarm dry-run.")
    parser.add_argument("--prewarm-all-models", action="store_true")
    parser.add_argument("--prewarm-tasks", default="piqa", help="Small task list for prewarm dry-run; set empty to use config tasks.")
    parser.add_argument("--expand-mmlu", action="store_true")
    parser.add_argument("--finalize-dry-run", action="store_true", help="Include matrix --finalize in the dry-run command.")
    return parser


def main() -> None:
    run_preflight(build_parser().parse_args())


if __name__ == "__main__":
    main()
