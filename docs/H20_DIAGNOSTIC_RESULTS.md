# H20 Diagnostic Results

This report records the H20 diagnostic run for the edge compressed-model
benchmark. It is evidence from the available H20 environment, not the final
RTX 4090 acceptance run.

## Runtime

- GPU binding: `CUDA_VISIBLE_DEVICES=1`
- Visible GPU: NVIDIA H20, 95GB
- Python: 3.10.20
- Torch: 2.6.0+cu124
- CUDA runtime reported by PyTorch: 12.4
- cuDNN: 92000
- Transformers: 4.55.2
- bitsandbytes: 0.50.1
- auto-gptq: 0.7.1
- autoawq: 0.2.9
- compressed-tensors: 0.15.0.1
- Hugging Face endpoint: `https://hf-mirror.com`
- Data root: `/data/llm_eval_datasets`
- Output root: `/data/outputs/llmcbench_h20_full_matrix_py310`
- Full matrix config:
  `/data/outputs/llmcbench_h20_full_matrix_py310/h20_matrix_full_smoothquant_tf4576.json`
- SmoothQuant-compatible profile for compressed-tensors W8A8 probes:
  Python 3.10.20, Torch 2.6.0+cu124, Transformers 4.57.6,
  compressed-tensors 0.15.0.1.

## Model Sources

- `LLaMA3-8B` uses `/data/models/Meta-Llama-3-8B`, a symlink to the public
  `NousResearch/Meta-Llama-3-8B` checkpoint. The official
  `meta-llama/Meta-Llama-3-8B` repo was gated in this environment.
- `Qwen3-8B` uses `/data/models/Qwen3-8B`.
- Precompressed checkpoints were staged and verified from hf-mirror where
  available.
- The local LLaMA3 OmniQuant checkpoint is
  `compressa-ai/Saiga-Llama-3-8B-OmniQuant`; its metadata declares
  `base_model=IlyaGusev/saiga_llama3_8b`, so it is a Llama3-family diagnostic
  row rather than the official Meta base checkpoint.

## Calibration Caveat

- The historical H20 pruning rows were generated while
  `/data/calibration/wiki_samples.txt` had only 5 non-empty bootstrap lines,
  although the matrix configuration requested 64 samples. Their manifests
  correctly record `calibration_samples=5`.
- They remain useful diagnostic evidence for pipeline completeness and
  resource behavior, but are not 64-sample or final method-quality results.
- They also predate the current exact C4 setup: pinned
  `allenai/c4` training shard, seed 0, 128 random 2048-token windows, and the
  layerwise Wanda/full-Hessian SparseGPT implementations. They must not be
  compared numerically with the new strict-profile pruning rerun.
- Current code rejects an explicit calibration file with fewer non-empty lines
  than `--calibration-samples`; target-path and asset-preparation tools create
  a 128-line bootstrap file for the current default config. Replace that bootstrap file
  with representative C4/Pile or in-domain calibration text before the final
  RTX 4090 matrix.

## Commands

```bash
MAMBA_ROOT_PREFIX=/opt/tiger/micromamba_root \
HF_ENDPOINT=https://hf-mirror.com \
PYTHONPATH=/tmp/llmcbench_byted \
/opt/tiger/micromamba_bin/micromamba run -n llmcbench-edge-stable \
python tools/run_edge_matrix.py \
  --config /data/outputs/llmcbench_h20_full_matrix_py310/h20_matrix_full_smoothquant_tf4576.json \
  --cuda-visible-devices 1 \
  --finalize \
  --require-handoff-manifest \
  --require-runtime-python-prefix 3.10 \
  --require-runtime-cuda-prefix 12. \
  --require-runtime-gpu-total-memory-min-gb 23 \
  --require-runtime-torch-prefix 2. \
  --require-runtime-transformers \
  --require-runtime-bitsandbytes \
  --require-runtime-cpu-count-min 8 \
  --require-runtime-system-memory-min-gb 240 \
  --require-runtime-hf-endpoint https://hf-mirror.com \
  --require-code-match \
  --require-clean-worktree
```

## Results

Accuracy columns use the benchmark's selected metric: MMLU `acc`, ARC
`acc_norm`, HellaSwag `acc_norm`, PIQA `acc_norm`, and WinoGrande `acc`.
The completed diagnostic matrix contains all 14 model/method pairs. The strict
result validator reports 14 expected and 14 actual rows, no missing task or
MMLU-subtask fields, and no 24GB deployment-memory violations.

| model | method | MMLU | ARC | HellaSwag | PIQA | WinoGrande | avg | tokens/s | peak GB | storage GB | FLOPs | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LLaMA3-8B | dense | 0.6406 | 0.5410 | 0.7923 | 0.8052 | 0.7395 | 0.7037 | 26.5216 | 15.1211 | 14.9661 | 36.24 TFLOPS | ok |
| LLaMA3-8B | wanda | 0.4712 | 0.4471 | 0.6887 | 0.7584 | 0.6930 | 0.6117 | 27.5466 | 15.0430 | 14.9737 | 36.24 TFLOPS | ok |
| LLaMA3-8B | sparsegpt | 0.4717 | 0.4462 | 0.6887 | 0.7584 | 0.6898 | 0.6110 | 26.7059 | 15.0430 | 14.9737 | 36.24 TFLOPS | ok |
| LLaMA3-8B | gptq | 0.6191 | 0.4932 | 0.7805 | 0.7840 | 0.7309 | 0.6815 | 2.9693 | 5.8047 | 5.4774 | 36.24 TFLOPS | ok |
| LLaMA3-8B | awq | 0.6142 | 0.5205 | 0.7817 | 0.8069 | 0.7245 | 0.6896 | 20.5037 | 5.4043 | 5.3431 | 36.24 TFLOPS | ok |
| LLaMA3-8B | smoothquant | 0.6412 | 0.5358 | 0.7907 | 0.8036 | 0.7332 | 0.7009 | 26.5554 | 15.0137 | 14.9661 | 36.24 TFLOPS | ok |
| LLaMA3-8B | omniquant | 0.6025 | 0.5119 | 0.7547 | 0.7867 | 0.7119 | 0.6736 | 2.1583 | 5.7812 | 5.3478 | 36.24 TFLOPS | ok |
| Qwen3-8B | dense | 0.7554 | 0.5640 | 0.7494 | 0.7780 | 0.6756 | 0.7045 | 27.6355 | 15.3789 | 15.2713 | 37.18 TFLOPS | ok |
| Qwen3-8B | wanda | 0.6843 | 0.5239 | 0.6631 | 0.7524 | 0.6669 | 0.6581 | 27.0186 | 15.3398 | 15.2612 | 37.18 TFLOPS | ok |
| Qwen3-8B | sparsegpt | 0.6846 | 0.5230 | 0.6628 | 0.7524 | 0.6669 | 0.6580 | 27.9995 | 15.3398 | 15.2612 | 37.18 TFLOPS | ok |
| Qwen3-8B | gptq | 0.7442 | 0.5614 | 0.7383 | 0.7688 | 0.6953 | 0.7016 | 2.9166 | 6.0000 | 5.6996 | 37.18 TFLOPS | ok |
| Qwen3-8B | awq | 0.7466 | 0.5478 | 0.7360 | 0.7688 | 0.6827 | 0.6964 | 17.8372 | 5.7578 | 5.6946 | 37.18 TFLOPS | ok |
| Qwen3-8B | smoothquant | 0.7503 | 0.5648 | 0.7523 | 0.7704 | 0.6796 | 0.7035 | 0.2757 | 17.1719 | 8.8052 | 37.18 TFLOPS | ok |
| Qwen3-8B | omniquant | 0.7444 | 0.5333 | 0.7365 | 0.7704 | 0.6835 | 0.6936 | 25.9833 | 15.3203 | 15.4635 | 37.18 TFLOPS | ok |

## SmoothQuant Status

SmoothQuant was first probed in the stable profile:

```bash
MAMBA_ROOT_PREFIX=/opt/tiger/micromamba_root \
HF_ENDPOINT=https://hf-mirror.com \
PYTHONPATH=/tmp/llmcbench_byted \
/opt/tiger/micromamba_bin/micromamba run -n llmcbench-edge-stable \
python tools/probe_quant_runtime.py \
  --config /data/outputs/llmcbench_h20_full_matrix_py310/h20_matrix_with_precompressed.downloaded.json \
  --only-method smoothquant \
  --cuda-visible-devices 1 \
  --hf-endpoint https://hf-mirror.com \
  --device-map auto \
  --dtype float16 \
  --gpu-max-memory 22GiB \
  --cpu-max-memory 120GiB \
  --generate \
  --max-new-tokens 1 \
  --timeout-seconds 900 \
  --output /data/outputs/llmcbench_h20_full_matrix_py310/smoothquant_runtime_probe.json
```

Stable-profile probe result:

- `LLaMA3-8B/smoothquant`: `ok=true`; tokenizer, model load and 1-token
  generation succeeded.
- `Qwen3-8B/smoothquant`: `ok=false`; tokenizer/config load succeeded, then
  model load failed with:

  ```text
  RuntimeError: Error(s) in loading state_dict for Linear:
    While copying the parameter named \"weight\", whose dimensions in the model
    are torch.Size([4096, 12288]) and whose dimensions in the checkpoint are
    torch.Size([4096, 12288]), an exception occurred : ('Only Tensors of
    floating point and complex dtype can require gradients',).
  ```

The Qwen3 failure was resolved by running SmoothQuant rows with a
Transformers 4.57.6 profile. The follow-up probe is recorded at
`/data/outputs/llmcbench_h20_full_matrix_py310/smoothquant_runtime_probe_tf4576_shim.json`
and reports `ok=true` for both `LLaMA3-8B/smoothquant` and
`Qwen3-8B/smoothquant` load/generate checks. Both full SmoothQuant rows now
contain accuracy, deployment and resource artifacts and pass final validation.

## Notes

- Sparse Wanda and SparseGPT rows reduce effective sparse storage estimates but
  do not reduce dense checkpoint storage in these exported HuggingFace outputs;
  their effective sparse estimates are recorded in `matrix_results.json`.
- GPTQ and AWQ reduce checkpoint storage and measured deployment memory, but
  GPTQ generation is much slower in this Transformers/AutoGPTQ stack.
- Qwen3 OmniQuant loads as a precompressed checkpoint but does not reduce dense
  storage in the staged artifact; treat its resource result as checkpoint-layout
  evidence, not as proof of packed 4-bit storage.
- The final RTX 4090 run should use the documented Python 3.10 / CUDA 12.x
  isolated environment and rerun the full matrix before treating results as
  final acceptance evidence.

## Target Rerun Checklist

On the target 4090 host:

1. Create the documented Python 3.10 environment:

   ```bash
   bash scripts/setup_edge_env.sh
   python -m pip install -r requirements-optional.txt
   ```

2. Verify quantized runtime imports and checkpoint loading before long jobs:

   ```bash
   python tools/run_edge_preflight.py \
     --config /data/outputs/llmcbench_edge_4090/edge_matrix_with_precompressed.json \
     --check-hf-access \
     --check-model-forward \
     --check-optional-backend-resolution \
     --check-quant-runtime \
     --cuda-visible-devices 0 \
     --output /data/outputs/llmcbench_edge_4090/edge_preflight_report.json
   ```

3. Rerun the full matrix under the 4090 runtime:

   ```bash
   python tools/run_edge_matrix.py \
     --config /data/outputs/llmcbench_edge_4090/edge_matrix_with_precompressed.json \
     --output-root /data/outputs/llmcbench_edge_4090 \
     --cuda-visible-devices 0 \
     --stop-on-error \
     --finalize
   ```

4. Run the final strict matrix/finalization command from `README.md` once all
   rows pass readiness. The final objective is complete only after
   `completion_audit.json` reports `"ok": true`, result-bundle verification
   passes, and the final artifact coverage includes all requested methods on
   both models.
