# Standalone Compression

This directory is the compact user-facing compression interface. The shared
algorithm implementations remain in `edge_benchmark/compression.py` so that
standalone exports and the accuracy/deployment/resource pipeline use exactly
the same checkpoint writer and manifest format.

Compress a local HuggingFace model with one required method argument:

```bash
CUDA_VISIBLE_DEVICES=1 bash compression/run_compression.sh \
  --base-model /data/models/Qwen3-8B \
  --method awq
```

Supported methods are `wanda`, `sparsegpt`, `gptq`, `awq`, `smoothquant`, and
`omniquant`. The launcher uses method-specific presets in `profiles.py`.

- Wanda, SparseGPT, GPTQ and AWQ default to their checked-in strict
  profiles.
- SmoothQuant and OmniQuant default to `adapted` because their upstream
  projects do not provide LLaMA3-8B/Qwen3-8B same-model recipes. Their outputs
  are explicitly marked as engineering adaptations.

Print the underlying compression command without writing a checkpoint:

```bash
python -m compression --base-model /data/models/Qwen3-8B --method gptq --dry-run
```

Compare numerical results only when the same model, compression setting,
calibration corpus/size, task suite, runtime and hardware are used. Otherwise
report the setting difference with the edge-benchmark result.
