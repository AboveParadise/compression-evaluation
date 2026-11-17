"""Compatibility entry point for lm-evaluation-harness."""

from __future__ import annotations

from typing import Any, Callable

from .runtime_compat import apply_runtime_compatibility_patches


def _patch_auto_model_quantization_kwargs() -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
    except Exception:
        return

    def patch_class(model_class: Any) -> None:
        original_from_pretrained = model_class.from_pretrained
        if getattr(original_from_pretrained, "_llmcbench_quant_kwarg_patch", False):
            return

        def patched_from_pretrained(*args: Any, **kwargs: Any) -> Any:
            quantization_config = kwargs.get("quantization_config")
            if quantization_config is None or isinstance(quantization_config, dict):
                kwargs.pop("quantization_config", None)
            return original_from_pretrained(*args, **kwargs)

        patched_from_pretrained._llmcbench_quant_kwarg_patch = True  # type: ignore[attr-defined]
        model_class.from_pretrained = patched_from_pretrained

    for model_class in (AutoModelForCausalLM, AutoModelForSeq2SeqLM):
        patch_class(model_class)


def main() -> None:
    apply_runtime_compatibility_patches()
    _patch_auto_model_quantization_kwargs()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
