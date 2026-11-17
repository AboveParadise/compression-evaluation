"""Runtime compatibility shims for optional quantization backends."""

from __future__ import annotations


def apply_runtime_compatibility_patches() -> None:
    """Patch small API drifts between Transformers and quantization backends."""
    try:
        import transformers.activations as activations
    except Exception:
        return

    if not hasattr(activations, "PytorchGELUTanh") and hasattr(activations, "GELUTanh"):
        activations.PytorchGELUTanh = activations.GELUTanh
