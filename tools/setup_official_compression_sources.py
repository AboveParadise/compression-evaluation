import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


SUPPORTED_METHODS = ("smoothquant", "omniquant")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _normalized_remote(value: str) -> str:
    normalized = value.strip()
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@"):
        host_path = normalized[len("git@") :]
        host, separator, path = host_path.partition(":")
        if separator:
            normalized = f"https://{host}/{path}"
    return normalized.rstrip("/")


def _run(command: List[str], dry_run: bool) -> Dict[str, Any]:
    if dry_run:
        return {"cmd": command, "dry_run": True, "ok": True}
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "cmd": command,
        "returncode": process.returncode,
        "ok": process.returncode == 0,
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-2000:],
    }


def _git_value(root: Path, args: List[str]) -> Dict[str, Any]:
    process = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, check=False)
    return {
        "ok": process.returncode == 0,
        "value": process.stdout.strip() if process.returncode == 0 else None,
        "error": process.stderr.strip() if process.returncode != 0 else None,
    }


def _protocol_for_method(protocols: Dict[str, Any], method: str) -> Dict[str, Any]:
    for name, protocol in protocols.items():
        if isinstance(protocol, dict) and protocol.get("method") == method:
            implementation = protocol.get("official_implementation")
            if isinstance(implementation, dict):
                return {"name": name, "protocol": protocol, "implementation": implementation}
    raise ValueError(f"No official_implementation record for {method} in paper protocols.")


def setup(args: argparse.Namespace) -> Dict[str, Any]:
    payload = _load_json(Path(args.protocols).expanduser())
    protocols = payload.get("protocols")
    if not isinstance(protocols, dict):
        raise ValueError("paper protocols JSON does not have a protocols object")
    selected = args.only_method or list(SUPPORTED_METHODS)
    unsupported = [method for method in selected if method not in SUPPORTED_METHODS]
    if unsupported:
        raise ValueError(f"Unsupported methods: {unsupported}")

    root = Path(args.root).expanduser()
    rows = []
    for method in selected:
        entry = _protocol_for_method(protocols, method)
        implementation = entry["implementation"]
        repository = implementation.get("repository")
        commit = implementation.get("commit")
        destination = root / method
        commands = []
        if not destination.exists():
            commands.append(
                _run(
                    [
                        "git",
                        "clone",
                        "--filter=blob:none",
                        "--no-checkout",
                        repository,
                        str(destination),
                    ],
                    args.dry_run,
                )
            )
        elif not (destination / ".git").exists():
            rows.append(
                {
                    "method": method,
                    "protocol": entry["name"],
                    "destination": str(destination),
                    "ok": False,
                    "error": "destination exists but is not a Git checkout",
                }
            )
            continue
        commands.append(
            _run(
                ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit],
                args.dry_run,
            )
        )
        commands.append(_run(["git", "-C", str(destination), "checkout", "--detach", commit], args.dry_run))
        if args.dry_run:
            rows.append(
                {
                    "method": method,
                    "protocol": entry["name"],
                    "destination": str(destination),
                    "repository": repository,
                    "commit": commit,
                    "commands": commands,
                    "ok": all(item["ok"] for item in commands),
                    "dry_run": True,
                }
            )
            continue
        remote = _git_value(destination, ["remote", "get-url", "origin"])
        head = _git_value(destination, ["rev-parse", "HEAD"])
        ok = (
            all(item["ok"] for item in commands)
            and remote["ok"]
            and head["ok"]
            and _normalized_remote(str(remote["value"])) == _normalized_remote(str(repository))
            and head["value"] == commit
        )
        rows.append(
            {
                "method": method,
                "protocol": entry["name"],
                "destination": str(destination),
                "repository": repository,
                "commit": commit,
                "observed_repository": remote["value"],
                "observed_commit": head["value"],
                "commands": commands,
                "ok": ok,
                "errors": [item.get("error") for item in (remote, head) if item.get("error")],
            }
        )
    report = {"ok": all(row["ok"] for row in rows), "root": str(root), "rows": rows}
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone and pin official SmoothQuant/OmniQuant source checkouts.")
    parser.add_argument("--protocols", default="configs/paper_protocols.json")
    parser.add_argument("--root", default="/opt/llmcbench_official")
    parser.add_argument("--only-method", action="append", default=[], choices=SUPPORTED_METHODS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="")
    setup(parser.parse_args())


if __name__ == "__main__":
    main()
