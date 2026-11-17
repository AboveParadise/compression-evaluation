# Completion Audit

This file maps the requested LLMCBench edge compression benchmark requirements
to concrete repository artifacts and verification commands.

## Objective Restatement

Deliver a portable LLMCBench-based toolkit that can:

1. Compress LLaMA3-8B and Qwen3-8B with Wanda, SparseGPT, GPTQ, AWQ,
   SmoothQuant and OmniQuant.
2. Evaluate compressed-model accuracy on MMLU, ARC-Challenge, HellaSwag, PIQA
   and WinoGrande.
3. Evaluate edge deployment capability with peak GPU memory and tokens/s.
4. Evaluate inference resource usage with storage, sparsity, parameter count and
   FLOPs/MACs.
5. Run on a 24GB RTX 4090-oriented environment.
6. Provide clear environment, model/data preparation, full matrix execution and
   result collection instructions.
7. Produce complete results for both target models once model and dataset assets
   are present.

## Prompt-to-Artifact Checklist

| Requirement | Artifact | Verification |
| --- | --- | --- |
| Unified entrypoint | `edge_benchmark/cli.py` exposes `compress`, `accuracy`, `deploy`, `resource`, `pipeline` | `python -m edge_benchmark --help` |
| Sparse/quant method selection | `SUPPORTED_METHODS` in `edge_benchmark/cli.py`; implementations/adapters in `edge_benchmark/compression.py` | `python -m edge_benchmark compress --help` |
| Pruning pattern selection | `--prune-pattern unstructured|2:4`; `configs/edge_4090_structured_pruning.json` | Wanda/SparseGPT default to 50% unstructured; the semi-structured pruning config applies exact 2:4 masks |
| Wanda | Built-in activation-aware pruning in `edge_benchmark/compression.py` | `--compression-method wanda` |
| SparseGPT | Official layerwise full input-Hessian SparseGPT implementation in `edge_benchmark/compression.py` | `--compression-method sparsegpt`; uses Cholesky inverse and blockwise error compensation |
| 4090-friendly pruning calibration | `use_cache=False`; decoder-layer activation buffers default to CPU; `lm_head` excluded; explicit calibration files must have at least `calibration_samples` non-empty lines | Keeps 128x2048 calibration state off the 24GB GPU, reduces accuracy risk and prevents silent under-calibration |
| GPTQ | `auto-gptq` backend in `edge_benchmark/compression.py` | `--compression-method gptq` |
| AWQ | `autoawq` backend in `edge_benchmark/compression.py` | `--compression-method awq` |
| SmoothQuant | External command adapter in `edge_benchmark/compression.py`; matrix command slot in `configs/edge_4090_matrix.json` | `--compression-method smoothquant --external-command ...` |
| OmniQuant | External command adapter in `edge_benchmark/compression.py`; matrix command slot in `configs/edge_4090_matrix.json` | `--compression-method omniquant --external-command ...` |
| External method template validation | `tools/configure_external_methods.py`; `tools/setup_official_compression_sources.py` | Validates/writes SmoothQuant and OmniQuant command templates and provisions pinned official Git checkouts before the strict matrix |
| Accuracy evaluation | `run_accuracy` in `edge_benchmark/cli.py` using lm-eval tasks | `python -m edge_benchmark accuracy --help` |
| Required accuracy datasets | Defaults in `DEFAULT_ACCURACY_TASKS`: `mmlu`, `arc_challenge`, `hellaswag`, `piqa`, `winogrande` | `rg DEFAULT_ACCURACY_TASKS edge_benchmark/cli.py` |
| Full MMLU coverage | `DEFAULT_MMLU_SUBTASKS` in `tools/prewarm_lm_eval_tasks.py`; `tools/collect_edge_results.py`; `tools/validate_edge_results.py`; `tools/build_goal_audit.py` | Final `matrix_results.json` must include every configured `mmlu_*_score` subtask column for each model/method row. `validate_edge_results.py` rejects rows with only `mmlu_score`, a single MMLU smoke subtask, or any missing/non-numeric MMLU subtask score. |
| Deployment metrics | `run_deploy` in `edge_benchmark/cli.py` outputs memory and tokens/s | `python -m edge_benchmark deploy --help` |
| Resource metrics | `run_resource` in `edge_benchmark/cli.py` outputs storage, params, sparsity and FLOPs | `python -m edge_benchmark resource --help` |
| FLOPs fallback | `edge_benchmark/cli.py` uses `calflops` first, then analytical config estimate; `verify_edge_env.py --check-resource-backend` reports selected backend | `resource.json` must include non-null FLOPs/MACs and backend |
| Full pipeline | `run_pipeline` in `edge_benchmark/cli.py`; `scripts/run_edge_pipeline.sh` | `python -m edge_benchmark pipeline --help` |
| Precompressed checkpoint input | `python -m edge_benchmark pipeline --use-precompressed` | Skips compression and evaluates an already exported compressed HuggingFace checkpoint |
| Matrix precompressed checkpoints | `precompressed_paths` in matrix config; `tools/run_edge_matrix.py`; `tools/audit_edge_assets.py --only-method`; `tools/check_method_readiness.py --only-method` | Allows selected model/method pairs to evaluate exported compressed checkpoints without recompression; scoped asset audit/preflight/readiness requires the exported checkpoint instead of the dense base checkpoint or compression backend for those pairs |
| Precompressed asset preparation | `tools/prepare_precompressed_assets.py` | Stages known public Qwen3-8B GPTQ/AWQ/SmoothQuant/OmniQuant checkpoints from hf-mirror, writes a preparation report, emits blockers for missing repo mappings, and can write a config copy with `precompressed_paths` populated |
| Full two-model matrix | `configs/edge_4090_matrix.json`; `tools/run_edge_matrix.py` | `python tools/run_edge_matrix.py --dry-run ...` |
| Resolved config snapshot | `matrix_config_resolved.json` in the output root | Captures selected models/methods, effective defaults, overrides, external commands and precompressed paths; final audit uses this snapshot when present |
| Matrix job overrides | `method_overrides`, `model_overrides`, `model_method_overrides` in matrix config | Allows per-method/model job settings for memory-sensitive 4090 runs |
| One-command 4090 workflow | `scripts/run_edge_4090_full.sh` | Runs asset audit, preflight, optional prewarm, matrix finalization, final audit, post-run readiness refresh and final results-bundle export/verification; supports `DRY_RUN=1` |
| Matrix resume planning | `tools/plan_edge_matrix_resume.py` | Reads the matrix config, output root and optional `matrix_run_report.json`; classifies each model/method pair and emits scoped commands for missing, failed or incomplete jobs plus finalization after resume |
| Target storage initialization | `scripts/init_edge_storage.sh`; handoff setup command | Creates the standard `/data` model/cache/dataset/calibration/prompt/output directory tree and assigns ownership to the target user before Python tooling runs on a fresh 4090 host |
| Target path preparation | `tools/prepare_target_paths.py`; generated `edge_target_paths_report.json` | Creates cache, dataset, output, model-parent and precompressed-model directories, writes a calibration seed with the configured number of non-empty lines and prompt seed file when absent, and emits an auditable path-preparation report before model downloads |
| Asset/preflight report audit | `tools/audit_edge_completion.py --require-asset-audit-report --require-non-rotational-storage --require-preflight-report` | Requires `<output_root>/edge_asset_audit.json` and `<output_root>/edge_preflight_report.json` to exist with `"ok": true`; the storage gate also requires hf cache, dataset and output paths to be creatable/writable by the current user and proven SSD/NVMe-like by asset audit path and block-device metadata |
| Handoff manifest audit | `tools/audit_edge_completion.py --require-handoff-manifest`; `scripts/run_edge_4090_full.sh` | Requires `<output_root>/edge_handoff_manifest.json` to exist and contain the portable setup, command, precompressed-preparation, prewarm, readiness, artifact and completion-gate schema generated by `tools/build_edge_handoff_manifest.py` |
| Portable bundle export and verification | `tools/export_edge_portable_bundle.py`; `tools/verify_edge_portable_bundle.py` | Creates a movable code/config/runbook bundle with `PORTABLE_BUNDLE_MANIFEST.json`, `RUN_ON_TARGET.md`, `edge_handoff_manifest.json`, requirements, scripts, tools and docs; the verifier checks file size/hash integrity, archive file-list/size/SHA256 consistency against the extracted bundle including the manifest file itself, rejects duplicate or unsafe archive paths and non-file/non-directory archive members, requires the archive's single top-level directory to match the extracted bundle directory, checks handoff manifest schema, target runbook commands, required migration/verification tools including environment validation, external-method configuration, quant-runtime probing, goal audit, result collection/rendering and final validation, plus optional `--require-clean-code`/`--require-commit` provenance gates for final handoff; model weights, datasets and final outputs stay external and are checked by asset/readiness/completion audits |
| Asset prepare report audit | `tools/audit_edge_completion.py --require-prepare-report --require-prepare-metadata-report` | Requires prepare reports when wrapper prepare steps are enabled; if prepare reports exist, completion audit validates that they are JSON objects with `"ok": true` |
| One-command finalization | `tools/run_edge_matrix.py --finalize` and `--finalize-only` | Runs collect, validate and render steps into final JSON/CSV/Markdown artifacts |
| Prewarm report audit | `tools/audit_edge_completion.py --require-prewarm-report`; optional `--require-prewarm-all-models` | Requires `<output_root>/edge_prewarm_report.json` to exist with `"ok": true`, configured task coverage and no failed prewarm tasks; all selected model coverage is opt-in |
| Result collection | `tools/collect_edge_results.py` | `python tools/collect_edge_results.py --root <output_root>` |
| Final result validation | `tools/validate_edge_results.py` | Checks every required model/method/task/metric in `matrix_results.json`, including full MMLU subtask score coverage, compression source path/backend allowlists/dtype/materialization/precompressed/model-revision/calibration-file/prune-lm-head/GPTQ-desc-act/AWQ-version/external-command/return-code provenance, method-specific compression parameter presence, malformed result files, duplicate rows, selected metric-key provenance, accuracy dtype/lm-eval-dtype/batch/num-fewshot/split/data-cache provenance, rejection of lm-eval `limit` smoke subsets, split-accuracy failure records, MMLU aggregate/average-score consistency, dense-relative accuracy delta/retention consistency, finite numeric metric types and ranges, integer-valued counters, deployment prompt/source/generation setting provenance, deployment token-count upper bounds, tokens/s consistency with generated tokens and elapsed seconds, deployment memory-gate field consistency, CUDA/GPU runtime metadata consistency, runtime profile provenance, HF cache path provenance, attention backend provenance when configured, storage bytes-to-GB conversion, sparse effective storage/FLOPs estimates, quantized storage estimate/provenance fields only for quantization methods, sparsity/nonzero-ratio consistency, numeric resource fields and resource prompt/source/sequence-length/sparsity-sample provenance, string score-metric/runtime/provenance metadata, FLOPs/MACs and compute/resource backend labels |
| Strict protocol validation | `configs/edge_paper_aligned_matrix.json`, `configs/paper_protocols.json`, `tools/validate_paper_protocol.py`, `tools/attest_base_models.py` | Enforces pinned base model repository/revision; Wanda/SparseGPT C4 seed-0 128×2048 50% unstructured settings; GPTQ C4 seed-0 128×2048 W4 per-channel asymmetric/act-order/true-sequential/non-static-group settings; and AWQ Pile validation seed-42 W4g128/20-grid settings. The pinned LLM-AWQ source has an explicit LLaMA3-8B W4g128 command; Qwen3 AWQ is labelled a setting transfer. AutoAWQ realization is rejected if its observed seed/grid diverge. SmoothQuant/OmniQuant target rows are blocked unless their pinned upstream source has a target-model recipe; if enabled later, they also require matching source repository/commit evidence and direct execution from that checkout. Five-line calibration outputs and source-override base models are rejected. Diagnostic options such as `--allow-blocked` and `--allow-skipped-external` cannot bypass the strict matrix preflight. |
| Goal audit matrix completion | `tools/build_goal_audit.py`; generated `current_goal_audit.json` | Requires exact LLaMA3-8B/Qwen3-8B x dense/Wanda/SparseGPT/GPTQ/AWQ/SmoothQuant/OmniQuant row coverage with no invalid, duplicate or unexpected rows; each row must have valid target-task and MMLU-subtask scores and metric keys, non-limited/no-failed-task accuracy provenance, deployment tokens/s and memory within the 24GB limit, numeric resource metrics with accepted resource/compute backends, and method-specific allowed compression backend provenance including `compression_materialized` and `compression_uses_precompressed` booleans; `external_command` rows must also record non-empty `compression_external_command` and `compression_external_returncode=0`; final artifacts cannot be complete unless `completion_audit.json` contains a successful `validate_edge_results.py` sub-gate with `validator.ok=true`; the goal audit also records `portable_bundle_verify.json` strict provenance evidence for the movable code package, but that evidence does not replace the final matrix gates |
| Final report rendering | `tools/render_edge_report.py` | Generates `edge_report.md` with accuracy scores, selected accuracy metric keys, accuracy dtype/batch/num-fewshot settings, compression source/materialization/precompressed/external-command provenance, runtime environment, deployment throughput/memory/input-token/generation settings, resource storage bytes/GB and FLOPs/MACs, preflight check summary, prewarm summary and matrix run status tables |
| Final scorecard | `tools/summarize_edge_results.py`; generated `edge_scorecard.json` and `edge_scorecard.md` | Ranks compression methods per model using configurable weighted normalized accuracy, tokens/s, peak memory, effective storage and effective FLOPs, records eligibility against edge constraints and reports a simple Pareto set for choosing edge deployment candidates; real quantized methods use physical checkpoint directory size, theoretical sparse storage/FLOPs are limited to Wanda/SparseGPT, and fake-quantized checkpoints receive no low-bit storage discount |
| Final completion audit | `tools/audit_edge_completion.py`; generated `completion_audit.json` | Checks final artifacts, artifact JSON parse errors, JSON/CSV row, column and value consistency, standard validator output and, for the strict matrix, `completion_paper_protocol_validation.json`, report section structure/metadata and full re-rendered content, resolved config source/model-entry/method/default/output-root/auxiliary selection, matrix finalization step coverage and final artifact paths, target readiness status when `--require-target-readiness-report` is enabled, matrix run job summary/log artifacts, unexpected run jobs, duplicate run jobs, matrix run status-list consistency including missing/stale entries, matrix run failures/skips, expected coverage, job summary identity, output-root containment for job artifacts, `compressed_model_path` consistency between job summaries and `matrix_results.json`, compression source path/backend/dtype/calibration/parameter/sequence-length provenance against each job's `compression.json`, accuracy values and accuracy settings against each job's `accuracy.json`, and key deployment/resource values including prompt, generation and sequence-length settings against each job's `deployment.json` and `resource.json` |
| Final results bundle | `tools/export_edge_results_bundle.py`; `tools/verify_edge_results_bundle.py`; generated `edge_results_bundle_verify.json` | Exports final JSON/CSV/Markdown results, scorecard, readiness/runtime reports, `edge_target_paths_report.json`, completion audit, optional `current_goal_audit.json`, optional job logs and self-contained result/delivery verifiers into a portable bundle with `RESULTS_BUNDLE_MANIFEST.json`; `--require-complete` fails unless required final artifacts including `edge_scorecard.json/md` exist, `completion_audit.json` has `"ok": true` with `validator.ok=true`, and `target_readiness.json` has `"ok": true` with `completion_ok=true` and no `before_matrix_blockers`; verifier checks file size/hash integrity, duplicate manifest paths, scorecard schema, completion/readiness status, JSON/CSV row/column/value consistency, `matrix_results.json` model/method coverage, target-task/MMLU-subtask score and score-metric coverage, non-limited/no-failed-task accuracy provenance, MMLU and target-task aggregate-score consistency, deployment metrics and token/s plus memory-peak consistency, deployment memory-limit compliance, resource metrics and storage/sparse-FLOPs/quantized-storage consistency, compression backend/materialized/precompressed/external-command provenance, and, with `--require-target-runtime`, 4090 target runtime metadata after transfer; `tools/build_goal_audit.py` requires `edge_results_bundle_verify.json` to report `"ok": true` and `"complete": true` before final artifacts can be complete |
| Final delivery verification | `tools/verify_edge_delivery.py`; generated `edge_delivery_verify.json` | Combines root-level `completion_audit.json`, root-level `target_readiness.json` and the exported results-bundle verifier into one final delivery verdict; the root completion audit must have `"ok": true` with `validator.ok=true`, and root target readiness must have `"ok": true`, `completion_ok=true` and no `before_matrix_blockers`; also compares the output root, `edge_results_bundle/` directory and `edge_results_bundle.tar.gz` archive by size and SHA256 for required artifacts, including `edge_target_paths_report.json`, so stale bundle directories or stale tarballs are rejected; `tools/build_goal_audit.py` requires `edge_delivery_verify.json` to report `"ok": true` with root, bundle-directory and archive substatus checks passing before final artifacts can be complete |
| Diagnostic matrix audit | `tools/audit_diagnostic_matrix.py`; `docs/H20_DIAGNOSTIC_RESULTS.md` | Scans diagnostic output roots and reports per-model/per-method coverage, failed accuracy tasks, missing deployment/resource artifacts and 24GB deployment-memory violations. This is useful for H20 or partial runs, but it is not a replacement for final 4090 `validate_edge_results.py` and `audit_edge_completion.py` gates. |
| Final runtime/code gate | `audit_edge_completion.py --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree` | Prevents CPU-only/CUDA-unavailable, non-4090, non-24GB-class, non-single-visible-GPU, non-multicore-CPU, low-RAM, non-Python-3.10, non-PyTorch-2.x, missing-Transformers, missing-bitsandbytes, non-CUDA-12.x, missing-cuDNN, non-hf-mirror, code-mismatched, or dirty-worktree outputs from being accepted as final 4090 results |
| Safe reruns | `--overwrite-compressed`; `OVERWRITE_COMPRESSED=1`; matrix `--force` | Existing compressed dirs are rejected unless explicitly overwritten |
| GPU memory guard | `--min-free-gpu-memory-gb`; `MIN_FREE_GPU_MEMORY_GB`; `configs/edge_4090_matrix.json` | Full-matrix default requires 20GB free GPU memory before long jobs |
| Asset preparation | `tools/prepare_edge_assets.py` | `python tools/prepare_edge_assets.py --metadata-only --dry-run ...` then full download |
| Asset readiness audit | `tools/audit_edge_assets.py` | Checks local checkpoint completeness, calibration/prompt files including configured calibration sample count, data/cache/output paths, free disk and best-effort mount/block-device media metadata before long jobs |
| Dataset cache prewarm | `tools/prewarm_lm_eval_tasks.py --config ... --expand-mmlu` | Reads matrix config paths/tasks and prewarms lm-eval datasets one task at a time |
| Unified 4090 preflight | `tools/run_edge_preflight.py` | Runs environment verification, method readiness, prewarm dry-run and matrix dry-run before long jobs; supports `--only-model` and `--only-method` scoped checks; `--cuda-visible-devices 0` also injects and strictly checks the configured single-GPU binding |
| Runtime verification | `tools/verify_edge_env.py` | `python tools/verify_edge_env.py --check-cli ...`; records `CUDA_VISIBLE_DEVICES` and can require it to match config `defaults.cuda_visible_devices` |
| Per-method readiness | `tools/check_method_readiness.py` | Reports model/config/import/external-command blockers per model and method |
| Target readiness summary | `tools/audit_target_readiness.py` | Summarizes target config, existing prepare/preflight/prewarm/handoff/final artifacts, optional backend/runtime/precompressed-preparation reports and next commands without running the matrix; reports `ready_for_matrix` and `before_matrix_blockers` separately from final `completion_ok`; groups baseline and generated precompressed-config post-run audit commands so `target_readiness.json` is refreshed with the same config as the matrix run |
| Optional backend installation | `requirements-optional.txt`; `scripts/setup_optional_backends.sh` | Installs/checks `bitsandbytes`, `auto-gptq`, `autoawq` and compressed-tensors via optional requirements, then installs/checks `gptqmodel==4.0.0` separately with `--no-build-isolation` on Python 3.10+ so GPTQModel checkpoints can see the active Torch install; the dry-run report records GPTQModel's Python/Torch/NumPy/Transformers requirements so incompatible target stacks fail before the matrix |
| Optional backend dry-run report | `OPTIONAL_BACKEND_DRY_RUN=1 scripts/setup_optional_backends.sh`; `tools/run_edge_preflight.py --check-optional-backend-resolution` | Records pip dependency resolution and current import status in `edge_optional_backends_report.json` before installing GPTQ/AWQ dependencies |
| Optional backend final gate | `tools/audit_edge_completion.py --require-optional-backends-report`; `scripts/run_edge_4090_full.sh` with `CHECK_OPTIONAL_BACKEND_RESOLUTION=1` | Requires `<output_root>/edge_optional_backends_report.json` to exist with `"ok": true` when optional backend resolution is part of the final 4090 workflow |
| Quantized runtime readiness | `tools/check_method_readiness.py`; `tools/build_goal_audit.py` | Separates precompressed checkpoint-file readiness from GPTQ/AWQ loading runtime readiness; GPTQ requires `optimum` plus `auto-gptq` or `gptqmodel`, AWQ requires `autoawq` |
| Quantized runtime probe | `tools/probe_quant_runtime.py`; `tools/run_edge_preflight.py --check-quant-runtime` | Reads checkpoint quantization metadata, verifies method-specific runtime imports such as GPTQModel and compressed-tensors before model load, then runs config/tokenizer/model-load and optional generation probes for quantized precompressed checkpoints before long matrix runs; preflight writes `edge_quant_runtime_report.json` |
| Quantized runtime final gate | `tools/audit_edge_completion.py --require-quant-runtime-report`; `scripts/run_edge_4090_full.sh` with `CHECK_QUANT_RUNTIME=1` | Requires `<output_root>/edge_quant_runtime_report.json` to exist with `"ok": true` when GPTQ/AWQ runtime probing is part of the final 4090 workflow |
| Full checkpoint gate | `tools/check_method_readiness.py`; `tools/verify_edge_env.py` | Distinguishes metadata-only assets from runnable checkpoints by checking weight files |
| Quantized resource fallback | `python -m edge_benchmark resource` | When a quantized checkpoint cannot load because the runtime backend is missing, records `resource_backend=config_only_fallback`, checkpoint storage, config-estimated params/FLOPs and `model_load_error` without claiming deployment or accuracy success |
| Final resource backend gate | `tools/validate_edge_results.py` | Rejects final matrix rows with `resource_backend=config_only_fallback`; diagnostic fallback outputs are allowed as intermediate evidence but not as complete final resource results |
| Environment setup | `scripts/setup_edge_env.sh` | Creates/updates the Python 3.10 conda environment and installs requirements |
| 4090 load memory constraint | README 4090 sections; defaults `dtype=fp16`, batch size 1 for calibration, 2048 sequence length and `max_memory=0:22GiB,cpu:120GiB`; compression, lm-eval accuracy/prewarm, deployment and resource loaders receive the configured dtype and `max_memory` | `configs/edge_4090_matrix.json` |
| 4090 deployment peak memory gate | `--deployment-memory-limit-gb`; `DEPLOYMENT_MEMORY_LIMIT_GB`; `deployment_memory_limit_gb=24` in the 4090 matrix config | Deployment step fails and `validate_edge_results.py` rejects rows above the configured peak-memory limit |
| Portability documentation | README sections: Environment Setup, Portable 4090 Setup Checklist, Full Matrix Runner | `README.md` |
| Current result status | README `Current Result Status` | `rg \"Current Result Status\" README.md` |

## Lightweight Verification Commands

Run these before starting long 8B jobs:

```bash
python -m compileall edge_benchmark tools
python -m edge_benchmark --help
python -m edge_benchmark compress --help
python -m edge_benchmark pipeline --help
bash -n scripts/run_edge_accuracy.sh
bash -n scripts/run_edge_deploy.sh
bash -n scripts/run_edge_resource.sh
bash -n scripts/run_edge_pipeline.sh
python tools/prepare_edge_assets.py --config configs/edge_4090_matrix.json --only-model Qwen3-8B --dry-run --skip-lm-eval-probe --continue-on-error --output /tmp/edge_prepare_dry_run.json
python tools/verify_edge_env.py --config configs/edge_4090_matrix.json --check-cli --check-hf-access --check-resource-backend
python tools/run_edge_matrix.py --config configs/edge_4090_matrix.json --only-model LLaMA3-8B --only-method gptq --dry-run --output-root /tmp/llmcbench_matrix_dry
python tools/collect_edge_results.py --root /tmp/nonexistent_llmcbench_results
```

Or run the wrapper:

```bash
python tools/run_smoke_checks.py
```

The current active-goal status can be regenerated as machine-readable JSON with:

```bash
python tools/build_goal_audit.py
```

`tools/run_smoke_checks.py` includes `tools/run_tiny_pipeline_smoke.py`, which
creates a tiny local GPT-style checkpoint and runs the real `pipeline` command
through compression, deployment and resource outputs without downloading 8B
weights. This is a functional smoke test for the JSON output contract, not a
substitute for the final LLaMA3/Qwen3 benchmark matrix.

## Known Remaining Gate

The complete two-model benchmark result is not checked into the repository until
the target runtime has:

- Local LLaMA3-8B checkpoint.
- Local Qwen3-8B checkpoint.
- lm-eval datasets/cache for MMLU, ARC-Challenge, HellaSwag, PIQA and
  WinoGrande.
- Installed optional compression backends for selected methods.
- SmoothQuant/OmniQuant external command templates if those methods are included
  in the final matrix.

If SmoothQuant/OmniQuant command templates are absent, `tools/run_edge_matrix.py`
records those jobs under `skipped_external_not_configured` so the result gap is
explicit. The final audit accepts that status only when
`--allow-skipped-external` is passed; otherwise skipped external methods remain
completion blockers.

Development-environment HF access probe:

- `Qwen/Qwen3-8B` config download from `hf-mirror.com` succeeded.
- `meta-llama/Meta-Llama-3-8B` config download returned a gated-repository 403
  without an authorized token. Use `huggingface-cli login` or provide a local
  authorized checkpoint before running the full LLaMA3 matrix.

Once available, the final result gate is:

```bash
python tools/build_edge_handoff_manifest.py --config configs/edge_4090_matrix.json --output-root /data/outputs/llmcbench_edge_4090 --output /data/outputs/llmcbench_edge_4090/edge_handoff_manifest.json
python tools/run_edge_preflight.py --config configs/edge_4090_matrix.json --check-hf-access --check-model-forward --finalize-dry-run --cuda-visible-devices 0 --check-optional-backend-resolution --check-quant-runtime --output /data/outputs/llmcbench_edge_4090/edge_preflight_report.json
python tools/prewarm_lm_eval_tasks.py --config configs/edge_4090_matrix.json --only-model Qwen3-8B --expand-mmlu --sleep-seconds 60 --report-output /data/outputs/llmcbench_edge_4090/edge_prewarm_report.json
python tools/run_edge_matrix.py --config configs/edge_4090_matrix.json --stop-on-error --finalize --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report --require-optional-backends-report --require-quant-runtime-report --require-target-paths-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree
python tools/audit_edge_completion.py --config configs/edge_4090_matrix.json --root /data/outputs/llmcbench_edge_4090 --require-finalize-report --require-prewarm-report --require-asset-audit-report --require-non-rotational-storage --require-preflight-report --require-handoff-manifest --require-target-readiness-report --require-optional-backends-report --require-quant-runtime-report --require-target-paths-report --require-runtime-python-prefix 3.10 --require-runtime-gpu-name-contains 4090 --require-runtime-cuda-prefix 12. --require-runtime-gpu-total-memory-min-gb 23 --require-runtime-gpu-count 1 --require-runtime-cudnn --require-runtime-torch-prefix 2. --require-runtime-transformers --require-runtime-bitsandbytes --require-runtime-cpu-count-min 8 --require-runtime-system-memory-min-gb 240 --require-runtime-hf-endpoint https://hf-mirror.com --require-code-match --require-clean-worktree
```

When the wrapper runs with `RUN_TARGET_PATH_PREPARE=1`, it adds
`--require-target-paths-report` to the final audit commands automatically.
When the wrapper runs with `RUN_PREPARE=1` or `RUN_METADATA_PREPARE=1`, it adds
`--require-prepare-report` or `--require-prepare-metadata-report` to the final
audit commands automatically.
The strict manual commands above already enable optional backend resolution and
quantized runtime probes in preflight, then require
`edge_optional_backends_report.json` and `edge_quant_runtime_report.json` during
finalization and final audit.

The final deliverables are:

```text
/data/outputs/llmcbench_edge_4090/matrix_results.csv
/data/outputs/llmcbench_edge_4090/matrix_results.json
/data/outputs/llmcbench_edge_4090/edge_report.md
/data/outputs/llmcbench_edge_4090/edge_scorecard.json
/data/outputs/llmcbench_edge_4090/edge_scorecard.md
/data/outputs/llmcbench_edge_4090/edge_asset_audit.json
/data/outputs/llmcbench_edge_4090/edge_preflight_report.json
/data/outputs/llmcbench_edge_4090/edge_prewarm_report.json
/data/outputs/llmcbench_edge_4090/edge_handoff_manifest.json
/data/outputs/llmcbench_edge_4090/edge_target_paths_report.json
/data/outputs/llmcbench_edge_4090/edge_optional_backends_report.json
/data/outputs/llmcbench_edge_4090/edge_quant_runtime_report.json
/data/outputs/llmcbench_edge_4090/edge_prepare_assets_report.json
/data/outputs/llmcbench_edge_4090/edge_prepare_metadata_report.json
/data/outputs/llmcbench_edge_4090/matrix_run_report.json
/data/outputs/llmcbench_edge_4090/matrix_finalize_report.json
/data/outputs/llmcbench_edge_4090/completion_audit.json
/data/outputs/llmcbench_edge_4090/matrix_config_resolved.json
/data/outputs/llmcbench_edge_4090/target_readiness.json
/data/outputs/llmcbench_edge_4090/edge_results_bundle.tar.gz
/data/outputs/llmcbench_edge_4090/edge_results_bundle/RESULTS_BUNDLE_MANIFEST.json
/data/outputs/llmcbench_edge_4090/edge_results_bundle_verify.json
/data/outputs/llmcbench_edge_4090/edge_delivery_verify.json
```
