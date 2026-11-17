import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _write_tiny_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "Explain": 4,
        "why": 5,
        "model": 6,
        "compression": 7,
        "is": 8,
        "useful": 9,
        "for": 10,
        "edge": 11,
        "deployment": 12,
        ".": 13,
        "test": 14,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    wrapped.save_pretrained(path)


def _write_tiny_model(path: Path, architecture: str) -> None:
    if architecture == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=32,
            n_positions=64,
            n_ctx=64,
            n_embd=32,
            n_layer=2,
            n_head=4,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        model = GPT2LMHeadModel(config)
    elif architecture == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        model = LlamaForCausalLM(config)
    else:
        raise ValueError(f"Unsupported tiny model architecture: {architecture}")
    model.save_pretrained(path, safe_serialization=True)
    _write_tiny_tokenizer(path)


def _validate_outputs(output_root: Path) -> None:
    required = ["compression.json", "deployment.json", "resource.json", "summary.json"]
    missing = [name for name in required if not (output_root / name).exists()]
    if missing:
        raise RuntimeError(f"Missing tiny pipeline outputs: {missing}")
    deployment = json.loads((output_root / "deployment.json").read_text(encoding="utf-8"))
    resource = json.loads((output_root / "resource.json").read_text(encoding="utf-8"))
    compression = json.loads((output_root / "compression.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    if deployment.get("tokens_per_second") is None:
        raise RuntimeError("deployment.json does not contain tokens_per_second")
    if "deployment_memory_limit_exceeded" not in deployment:
        raise RuntimeError("deployment.json does not contain deployment_memory_limit_exceeded")
    if resource.get("storage_gb") is None:
        raise RuntimeError("resource.json does not contain storage_gb")
    if summary.get("compression_method") in {"gptq", "awq", "smoothquant", "omniquant"}:
        if not isinstance(resource.get("quantization_storage_real"), bool):
            raise RuntimeError("resource.json does not contain quantization_storage_real")
        if not resource.get("quantization_storage_format"):
            raise RuntimeError("resource.json does not contain quantization_storage_format")
        if resource["quantization_storage_real"]:
            if resource.get("effective_storage_gb_quantized_approx") is None:
                raise RuntimeError("resource.json does not contain effective_storage_gb_quantized_approx")
            if resource.get("effective_storage_quantized_bits") is None:
                raise RuntimeError("resource.json does not contain effective_storage_quantized_bits")
            if resource.get("effective_storage_quantized_backend") != "checkpoint_directory_size":
                raise RuntimeError("resource.json does not contain expected effective_storage_quantized_backend")
        elif any(
            resource.get(field) is not None
            for field in (
                "effective_storage_gb_quantized_approx",
                "effective_storage_quantized_bits",
                "effective_storage_quantized_backend",
            )
        ):
            raise RuntimeError("fake-quant checkpoint contains real-storage estimates")
    if compression.get("manifest", {}).get("compressed_model_path") is None:
        raise RuntimeError("compression.json does not contain compressed_model_path")
    if compression.get("manifest", {}).get("backend") == "precompressed_checkpoint":
        for field in ("quant_bits", "group_size", "desc_act", "awq_version", "sparsity", "calibration_samples", "calibration_batch_size"):
            if field not in compression.get("manifest", {}):
                raise RuntimeError(f"precompressed manifest does not contain {field}")
    if compression.get("manifest", {}).get("backend") == "external_command":
        for field in ("quant_bits", "group_size", "sparsity", "calibration_samples", "external_returncode", "stderr_tail"):
            if field not in compression.get("manifest", {}):
                raise RuntimeError(f"external manifest does not contain {field}")
    for name, payload in (("compression", compression), ("deployment", deployment), ("resource", resource), ("summary", summary)):
        if not isinstance(payload.get("runtime"), dict):
            raise RuntimeError(f"{name}.json does not contain runtime metadata")


def run(args: argparse.Namespace) -> None:
    root = Path(args.workdir).expanduser()
    if root.exists() and args.clean:
        shutil.rmtree(root)
    model_dir = root / "tiny_model"
    output_root = root / "pipeline_output"
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_tiny_model(model_dir, args.architecture)

    cmd = [
        sys.executable,
        "-m",
        "edge_benchmark",
        "pipeline",
        "--model-path",
        str(model_dir),
        "--model-name",
        "unknown",
        "--compression-method",
        args.compression_method,
        "--compressed-output-dir",
        str(root / f"compressed_{args.compression_method}"),
        "--output-root",
        str(output_root),
        "--device",
        args.device,
        "--device-map",
        "",
        "--dtype",
        "fp32",
        "--seqlen",
        "32",
        "--prune-pattern",
        args.prune_pattern,
        "--pruning-calibration-source",
        "text_file",
        "--sparsegpt-blocksize",
        str(args.sparsegpt_blocksize),
        "--sparsegpt-percdamp",
        str(args.sparsegpt_percdamp),
        "--pruning-buffer-device",
        args.pruning_buffer_device,
        "--input-length",
        "32",
        "--max-new-tokens",
        "4",
        "--warmup",
        "0",
        "--repeat",
        "1",
        "--skip-accuracy",
        "--overwrite-compressed",
    ]
    if args.use_precompressed:
        cmd.append("--use-precompressed")
    if args.external_command:
        cmd.extend(["--external-command", args.external_command])
    subprocess.run(cmd, check=True)
    _validate_outputs(output_root)
    manifest = json.loads((output_root / "compression.json").read_text(encoding="utf-8"))["manifest"]
    if args.architecture == "llama" and args.compression_method in {"wanda", "sparsegpt"}:
        expected = "wanda_official_layerwise" if args.compression_method == "wanda" else "sparsegpt_official_second_order"
        if manifest.get("pruning_algorithm") != expected:
            raise RuntimeError(f"Expected {expected}, got {manifest.get('pruning_algorithm')}")
        if manifest.get("pruning_buffer_device") != args.pruning_buffer_device:
            raise RuntimeError("Layerwise pruning did not preserve pruning_buffer_device metadata")
    print(json.dumps({"ok": True, "output_root": str(output_root)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny local model through the real edge pipeline.")
    parser.add_argument("--workdir", default="/tmp/llmcbench_tiny_pipeline_smoke")
    parser.add_argument("--compression-method", default="dense")
    parser.add_argument("--architecture", choices=("gpt2", "llama"), default="gpt2")
    parser.add_argument("--prune-pattern", choices=("unstructured", "2:4"), default="unstructured")
    parser.add_argument("--sparsegpt-blocksize", type=int, default=128)
    parser.add_argument("--sparsegpt-percdamp", type=float, default=0.01)
    parser.add_argument("--pruning-buffer-device", default="cpu")
    parser.add_argument("--external-command", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-precompressed", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
