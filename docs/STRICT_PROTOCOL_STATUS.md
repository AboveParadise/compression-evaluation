# Strict Reproducible Protocol Status

This document records the admissibility state of the strict matrix in
`configs/edge_paper_aligned_matrix.json`. It distinguishes a strict reproducible
strict setting from an engineering-only algorithm transfer.

## Decision Rule

A target-model row is eligible for the strict matrix only when all of the
following hold:

1. The local base model matches the pinned official repository and revision,
   proved by `tools/attest_base_models.py`.
2. The method configuration matches the named record in
   `configs/paper_protocols.json`.
3. An external method has a pinned official source checkout, runs a script
   inside that checkout, and writes matching export evidence.
4. The pinned official source contains a recipe for the requested target model.

`tools/validate_paper_protocol.py` and `tools/run_edge_matrix.py` enforce these
conditions before strict jobs run. `tools/run_edge_preflight.py` records the
same validation as a `protocol` preflight step.

`--allow-blocked` is diagnostic-only for the preflight reporter, and
`--allow-skipped-external` is a general-matrix final-audit option. Neither can
bypass strict matrix validation.

## Method Status

| Method | LLaMA3-8B | Qwen3-8B | Strict setting |
| --- | --- | --- | --- |
| Dense | Eligible after base-model attestation | Eligible after base-model attestation | Same pinned source checkpoint/revision |
| Wanda | Eligible after base-model attestation | Eligible after base-model attestation | C4, seed 0, 128 × 2048, 50% unstructured |
| SparseGPT | Eligible after base-model attestation | Eligible after base-model attestation | C4, seed 0, 128 × 2048, 50% unstructured, blocksize 128, percdamp 0.01 |
| GPTQ | Eligible after base-model attestation | Eligible after base-model attestation | C4, seed 0, 128 × 2048, W4 official asymmetric default, `group_size=-1`, act-order, true-sequential, damping 0.01 |
| AWQ | Eligible after base-model attestation | Eligible after base-model attestation | LLaMA3 official LLM-AWQ W4g128 recipe; Qwen3 setting transfer. Both use Pile validation via `pileval`, seed 42, 128 × 512, zero-point and a verified 20-point grid |
| SmoothQuant | Blocked | Blocked | Pinned upstream source has no LLaMA3-8B/Qwen3-8B target recipe/export; Qwen3 also has no official alpha |
| OmniQuant | Blocked | Blocked | WikiText2, official seed 2, W4A16, 20 epochs, LWC; pinned upstream source has no LLaMA3-8B/Qwen3-8B target script/export |

The blocked rows must not be substituted with third-party checkpoints, custom
adaptations, guessed alpha values, or recipes copied from another model family.
Those alternatives may be benchmarked in an engineering matrix, but remain
separate from strict-profile rows.

## Official Source Evidence

- AWQ: `mit-han-lab/llm-awq` commit
  `d6e797a42b9ef7778de8ee2352116e0f48a78d61`. Its README gives the
  LLaMA3-8B `--w_bit 4 --q_group_size 128 --run_awq` command. Its entrypoint
  invokes `run_awq(..., n_samples=128, seqlen=512)`, whose default corpus is
  `pileval`; zero-point is enabled unless `--no_zero_point` is passed. The
  strict runtime uses AutoAWQ only as a compatible execution realization and
  rejects it unless its actual `pileval` shuffle seed is 42 and its
  scale-search grid is 20. The official recipe does not name Qwen3-8B, so the
  Qwen3 row remains an explicitly labelled setting transfer.
- SmoothQuant: `mit-han-lab/smoothquant` commit
  `c61476d728e42ae0d8a35e7e78494edcac3237b5`. The repository tree exposes
  `examples/export_int8_model.py` and OPT-focused examples; it does not expose
  a LLaMA3-8B or Qwen3-8B recipe with target calibration provenance and alpha.
- OmniQuant: `OpenGVLab/OmniQuant` commit
  `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`. The repository tree contains
  LLaMA, LLaMA-2, OPT, Falcon and Mixtral scripts, but no LLaMA3-8B or
  Qwen3-8B target recipe.

## Current Host Status

At the time this status was last audited:

- `/data/models/Meta-Llama-3-8B` resolves to a NousResearch fallback and
  contains `.llmcbench_source_override.json`; strict LLaMA3 runs are rejected
  until the pinned official Meta checkpoint is installed and attested.
- The development host is not logged in to Hugging Face. The local
  `meta-llama/Meta-Llama-3-8B` cache contains only a `refs/main` pointer, with
  no official snapshot or weight blobs; an hf-mirror probe of the pinned
  official revision returns the expected gated-repository 403. A user with
  accepted Meta access must run `huggingface-cli login` (or provide an
  authorized local checkpoint) before the LLaMA3 strict slices can proceed.
- `/data/models/Qwen3-8B` has a successful strict base-model attestation:
  local `config.json` and `model.safetensors.index.json` match
  `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`. Its
  dense/Wanda/SparseGPT/GPTQ/AWQ strict slice passes method-readiness,
  matrix dry-run and pinned-C4 asset checks.
- Historical H20 pruning rows used five calibration examples and have no
  strict protocol attestation; they are diagnostic-only and cannot enter the
  strict matrix.
- The two available H20 GPUs are occupied, so no strict 8B recompression is
  scheduled from this host.

## Commands

Inspect static profile admissibility:

```bash
python tools/validate_paper_protocol.py \
  --config configs/edge_paper_aligned_matrix.json
```

Attest official base models after downloading the pinned revisions:

```bash
python tools/attest_base_models.py \
  --config configs/edge_paper_aligned_matrix.json \
  --output /data/outputs/llmcbench_strict_4090/base_model_attestation.json
```

Run the non-destructive strict preflight for the currently admissible Qwen3
slice:

```bash
python tools/run_edge_preflight.py \
  --config configs/edge_paper_aligned_matrix.json \
  --only-model Qwen3-8B \
  --only-method dense --only-method wanda --only-method sparsegpt \
  --only-method gptq --only-method awq \
  --allow-blocked \
  --output /data/outputs/llmcbench_strict_4090/qwen_builtin_preflight.json
```

Once all required target recipes and base attestations exist, launch only the
selected strict slice:

```bash
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com \
python tools/run_edge_matrix.py \
  --config configs/edge_paper_aligned_matrix.json \
  --only-model Qwen3-8B \
  --only-method wanda \
  --stop-on-error --finalize
```
