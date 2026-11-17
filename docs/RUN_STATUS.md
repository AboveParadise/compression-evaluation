# Run Status

This file records real runtime progress from the development environment.

## Latest Status

- A Python 3.10 / CUDA 12.4 H20 diagnostic matrix is complete under
  `/data/outputs/llmcbench_h20_full_matrix_py310`.
- Completed rows: 14/14 for `LLaMA3-8B,Qwen3-8B` x
  `dense,wanda,sparsegpt,gptq,awq,smoothquant,omniquant`.
- The refreshed `matrix_run_report.json` records all 14 complete artifacts as
  `skipped_existing`, with 0 failed jobs.
- `matrix_results.json`, `matrix_results.csv` and `edge_report.md` have been
  generated for those 14 rows.
- The historical `completion_audit.json` reports `ok=true` for the 14-row H20
  matrix with runtime gates for Python 3.10, CUDA 12.x, PyTorch 2.x,
  Transformers, bitsandbytes, hf-mirror, clean worktree and code-match.
  It predates the strict calibration-sample gate: the Wanda/SparseGPT rows used
  5 non-empty calibration lines while the config requested 64. Current code
  correctly treats those rows as diagnostic rather than final-quality results.
- Both SmoothQuant rows are complete using the Python 3.10 / Transformers
  4.57.6 compatibility profile. Qwen3 SmoothQuant records MMLU `0.7503`,
  average accuracy `0.7035`, `0.2757 tokens/s`, `17.1719GB` peak memory and
  `8.8052GB` physical checkpoint storage.
- This is still diagnostic H20 evidence, not final RTX 4090 acceptance evidence.
  Final acceptance still requires the documented single RTX 4090 run and a
  same 14-row matrix on the documented RTX 4090 target.

## Current Development Environment

- GPU: NVIDIA H20, 95GB visible per device.
- Current successful diagnostic environment:
  `/opt/tiger/micromamba_root/envs/llmcbench-edge-stable`, Python 3.10.20,
  Torch 2.6.0+cu124, Transformers 4.55.2, bitsandbytes 0.50.1,
  auto-gptq 0.7.1, autoawq 0.2.9 and compressed-tensors 0.15.0.1.
- Older Python 3.9 notes below are retained as history for the first diagnostic
  pass and are superseded by the Python 3.10 H20 matrix above.
- The shared Python 3.9 environment now imports `optimum`, `auto_gptq==0.7.1`
  and `awq==0.2.9`, so GPTQ/AWQ readiness has advanced from missing imports to
  target-runtime validation. `compressed_tensors` installs but still fails to
  import on Python 3.9 with `TypeError: unsupported operand type(s) for |:
  'type' and 'types.GenericAlias'`; readiness reports this as
  `target_python_3_10_required_for:compressed_tensors`. Recheck SmoothQuant on
  the target Python 3.10 environment.
- Optional backend dry-run resolution now works through
  `OPTIONAL_BACKEND_DRY_RUN=1 bash scripts/setup_optional_backends.sh`. In the
  current Python 3.9 development environment it resolves
  `bitsandbytes==0.48.2`, `optimum`, `auto-gptq==0.7.1` and
  `autoawq==0.2.9` plus `compressed-tensors`, but also resolves `torch==2.8.0`
  plus CUDA 12.8 dependency wheels; do not install that into the shared system
  Python. `gptqmodel==4.0.0` is handled as a separate Python 3.10+ install step
  with `--no-build-isolation`, and is skipped in the Python 3.9 diagnostic
  environment. The dry-run report now records GPTQModel's own requirements:
  Python `>=3.10`, Torch `>=2.7.1`, NumPy `>=2.2.6` and Transformers
  `>=4.55.2`. The default Qwen3-8B GPTQ repo now uses a non-GPTQModel
  AutoRound GPTQ checkpoint, so the stable Torch 2.6 profile remains the
  preferred target unless a GPTQModel-format override checkpoint is selected.
- `datasets`, `lm_eval` and `calflops` were installed into the development
  Python 3.9 environment for smoke checks only. Final runs should use the
  documented isolated Python 3.10 environment. The setup helper now defaults to
  the CUDA 12.4 PyTorch 2.6.0 stack (`torch==2.6.0`, `torchvision==0.21.0`,
  `torchaudio==2.6.0`) to avoid the older torch 2.1 / Triton 2.1 runtime issues
  observed in H20 quantized-generation probes. It also exposes
  `EDGE_RUNTIME_PROFILE=gptqmodel`, which installs a Torch 2.8 / CUDA 12.8 stack
  plus NumPy `>=2.2.6` and Transformers `>=4.55.2` for the Qwen GPTQModel
  override checkpoint path. Rows produced under different runtime profiles must
  be kept separate or rerun with matching runtime metadata before final
  bundling.
- The latest dense resource smoke uses the exact `calflops` backend. Older
  Wanda/SparseGPT smoke records were generated before `calflops` was available
  and explicitly record `analytical_config_estimate`.

## Model Assets

`tools/prepare_edge_assets.py` now supports reusing staged checkpoints via
`--model-source-root <dir> --link-existing-models`. It searches for complete
checkpoints named by model display name or configured path basename and creates
a symlink into the config's target model path only when that target path is
absent. This is intended for moving the final 4090 workflow to a machine where
LLaMA3/Qwen3 checkpoints already exist on a shared disk; otherwise it continues
to use `HF_ENDPOINT=https://hf-mirror.com` downloads. Asset preparation now also
accepts `--output` and the 4090 wrapper writes prepare reports beside the final
or dry-run audit artifacts when `RUN_PREPARE=1` or
`RUN_METADATA_PREPARE=1`, making staged model reuse, download failures and
generated prompt/calibration files auditable. Completion audit validates
prepare reports when they exist and the wrapper requires them when the
corresponding prepare step is enabled.
Asset audit and preflight are now method-aware for `precompressed_paths`: a
scoped `--only-method` check for an already exported checkpoint requires that
compressed checkpoint and does not also require the dense base checkpoint for
that model/method pair.
Method readiness follows the same rule for compression/export backends, so
precompressed GPTQ/AWQ checkpoints do not require in-tool compression, and
precompressed SmoothQuant/OmniQuant checkpoints do not require external
compression commands. GPTQ/AWQ still require loading runtimes for accuracy and
deployment: current Transformers needs `optimum` plus `auto-gptq` or
`gptqmodel` for GPTQ, `autoawq` for AWQ, and `compressed-tensors` for
SmoothQuant checkpoints exported through the compressed-tensors format. The
goal-audit blocker wording now separates checkpoint-file readiness from
quantized runtime readiness and checks that each precompressed checkpoint path
contains both `config.json` and a weight file.
The one-command 4090 wrapper now fails preflight on blocked methods during
non-dry-run launches, catching missing backends or incomplete precompressed
checkpoints before the full matrix starts.
Deployment memory gating is also covered by smoke checks: the measured peak is
the max of allocated and reserved CUDA memory, and rows over the configured
24GB limit are marked exceeded and cause the deployment step to fail early.
The 4090 matrix now carries `dtype=fp16` as an explicit config default and the
matrix runner exports it as `DTYPE`, so pipeline, accuracy, deployment and
resource wrappers use the same memory-oriented dtype unless overridden.
Model-forward preflight and direct environment verification now also default to
that matrix dtype unless `--forward-dtype` is explicitly provided.
Dataset prewarm now writes a JSON report when `--report-output` is set; the
4090 wrapper records expanded MMLU subtasks, attempted task/model pairs and
failures under the matrix output root as `edge_prewarm_report.json`; dry-runs
write the report under `DRY_RUN_OUTPUT_ROOT` by default. Prewarm also inherits
the matrix dtype and attention backend and records them in the report. When
prewarm is enabled, final completion audit requires that report to exist with
`"ok": true` with non-empty expanded tasks/attempts, enough attempts to cover
every expanded task for every prewarm target, no missing model/task pairs when
target names are recorded, matching succeeded model/task coverage, coverage of
all configured matrix tasks after MMLU expansion, and no failed tasks. Full
selected-model prewarm coverage is optional through
`REQUIRE_PREWARM_ALL_MODELS=1` or `--require-prewarm-all-models`.
The 4090 wrapper also writes asset and preflight reports under the same final or
dry-run output root by default; non-dry-run final audit now requires both reports
with `"ok": true`. Prepare reports follow the same output-root convention and
are required by final audit only when the matching prepare step is enabled.
It now writes `target_readiness.json` before the matrix starts, after asset
audit, preflight and optional prewarm have completed. Non-dry-run launches stop
there by default when `ready_for_matrix=false` (`FAIL_ON_TARGET_READINESS_BLOCKERS=0`
keeps the older diagnostic behavior), then refresh the same readiness report
after the matrix/final-audit path.
The target and local matrix configs now default to split accuracy tasks with
continue-on-error enabled; final validation still rejects any failed accuracy
tasks. Goal audit now surfaces these target accuracy controls under both
`target_4090_controls` and the accuracy requirement evidence.
Manual `run_edge_matrix.py --finalize` runs can now pass the same
`--require-prewarm-report` gate through to completion audit. The portable
handoff manifest generated by `tools/build_edge_handoff_manifest.py` now records
the same final manual commands with `--require-handoff-manifest`, so a migrated
4090 run cannot accidentally skip the handoff schema/code/artifact gate when
following `edge_handoff_manifest.json`. Its target-readiness command now also
requires non-rotational storage, the handoff manifest, optional backend reports
and quantized runtime reports, matching the strict preflight/finalization gate
set in the handoff manifest.
The strict final completion audit now also accepts `--require-target-readiness-report`.
When enabled, it rejects missing, stale or blocked `target_readiness.json`
outputs and requires the readiness report to point at the same final output root
and passing `completion_audit.json`.
The portable handoff manifest now exposes dataset cache prewarm and target
readiness as separate command groups, so target hosts run
`tools/prewarm_lm_eval_tasks.py` before the readiness command that requires
`edge_prewarm_report.json`.
It also exposes `prepare_precompressed` as a separate optional command group for
staging known public Qwen GPTQ/AWQ/SmoothQuant/OmniQuant checkpoints and writing
`edge_matrix_with_precompressed.json` before strict readiness or matrix runs.
The target-readiness `next_commands` list now includes the same
`tools/prepare_precompressed_assets.py --fail-on-error` command, so users who
follow readiness output get the same precompressed-config path as the handoff
manifest.
It also includes follow-up preflight, prewarm, readiness, matrix and final-audit
commands that use `edge_matrix_with_precompressed.json`, so a generated
`precompressed_paths` config is not silently ignored.
Those commands are also grouped under `next_command_groups` to distinguish the
baseline path, baseline readiness refresh, and generated precompressed-config
path for automation.
The generated precompressed-config readiness, matrix and final-audit commands
also include `--require-prepare-precompressed-report`, matching the wrapper's
strict gate. The `tools/run_edge_matrix.py --finalize` target-readiness step
now passes the same flag when finalization is run with
`--require-prepare-precompressed-report`, so wrapper and direct CLI runs enforce
the same precompressed-preparation evidence.
Readiness and handoff outputs now also include separate post-run audit groups
for the baseline config and generated precompressed config, so
`target_readiness.json` can be refreshed with the same config that produced the
matrix results.
Target readiness now propagates blockers from
`edge_prepare_precompressed_report.json`, so missing repo/checkpoint causes are
visible in `before_matrix_blockers`.
Target readiness now also reuses the structured final-audit gates for required
asset audit, preflight, prewarm, optional backend, quant runtime and
precompressed-preparation reports, so empty `{ok:true}` reports are blocked
before the long matrix starts.
Goal audit also summarizes those precompressed preparation blockers and failed
items under `final_artifacts.prepare_precompressed_report`.
Completion audit's simple report status now preserves report-level blockers and
failed items as well, including precompressed preparation failures.
The rendered Markdown report now includes a `Precompressed Preparation Blockers`
section when `edge_prepare_precompressed_report.json` contains blockers.
The goal-audit final artifact summary also surfaces the completion audit's
prewarm report gate status, missing attempted/succeeded pairs and violations
when final artifacts exist.
It also surfaces asset-audit and preflight-report gate status from completion
audit when those final artifacts exist. Preflight summaries include failed
required and optional steps so allow-blocked diagnostics remain visible.
Asset summaries include missing calibration/prompt files, low-disk findings and
required model or precompressed checkpoint readiness gaps.
Goal audit only marks final artifacts complete when those report gates are both
required and passing.
Final result collection now records compression source path, materialization
status and whether a row came from a precompressed checkpoint; validation and
completion audit check those provenance fields against each job's
`compression.json` manifest. The rendered Markdown report also includes these
compression provenance columns, asset/preflight status and a prewarm summary
when the corresponding reports are supplied.
Completion audit also requires the `## Prewarm` section when that report exists.
Final MMLU acceptance is now explicit in both validator and completion audit:
`matrix_results.json` must retain all configured `mmlu_*_score` fields for each
scoped model/method row, and `completion_audit.json` records
`mmlu_subtask_coverage_status` with per-row present/missing counts. The handoff
manifest's `completion_gates.coverage` also records the 57 required MMLU
subtasks, and handoff validation rejects stale manifests that omit or change
that list.
Final runtime acceptance now also requires CUDA availability to be explicit and
consistent with GPU metadata. `validate_edge_results.py` rejects final rows that
report a GPU name/count/memory while `runtime_cuda_available` is false, and
`audit_edge_completion.py` requires `runtime_cuda_available=true` whenever the
strict GPU/CUDA/cuDNN gates are enabled. `tools/run_smoke_checks.py` covers both
negative cases.
Final result validation has also been tightened for archived matrix tables:
`validate_edge_results.py` now rejects inconsistent MMLU/task aggregate scores,
wrong selected metric keys, non-integer count fields, unknown compression or
resource backend labels, wrong `compression_materialized` provenance,
missing method-specific compression parameters such as pruning sparsity,
quant bits/group size or calibration sample counts,
`accuracy_num_fewshot` values that do not match the matrix configuration,
deployment token counts above configured prompt/repeat/max-new-token bounds,
tokens/s values inconsistent with generated tokens and elapsed seconds,
prompt-source mismatches, HF cache path mismatches, attention backend mismatches
when configured, storage bytes/GB mismatches, sparse effective storage/FLOPs
mismatches, quantized effective storage bytes/GB/backend mismatches or
quantized-storage fields on non-quantized methods, and contradictory
`sparsity`/`nonzero_ratio` values. `tools/run_smoke_checks.py` has negative
fixtures for these final-table failures.
Current-goal audit matrix completion now also requires exact target pair
coverage, valid score metric provenance, non-limited/no-failed-task accuracy
runs, deployment memory within the configured 24GB limit, numeric resource
metrics with accepted backends and method-specific allowed compression backend
provenance including `compression_materialized` and
`compression_uses_precompressed` booleans. Rows using `external_command` must
also record non-empty `compression_external_command` and
`compression_external_returncode=0`.
The current-goal audit now also summarizes the `completion_audit.json`
`validator` sub-gate produced by `tools/validate_edge_results.py`. Final
artifacts remain incomplete unless that sub-gate reports `validator.ok=true`,
so a hand-written or stale `completion_audit.json` with only top-level
`ok=true` cannot bypass final result validation.
The same validator sub-gate is enforced by both the results-bundle verifier and
the final delivery verifier: `--require-complete` now requires
`completion_audit.json` to include `validator.ok=true`, and delivery verification
applies that check to the root output directory before comparing the bundle and
archive.
The results-bundle export and verifier now also require target readiness
completion. `export_edge_results_bundle.py --require-complete` and
`verify_edge_results_bundle.py --require-target-readiness-ok` both require
`target_readiness.json` to report `ok=true`, `completion_ok=true` and no
`before_matrix_blockers`.
The results-bundle verifier also performs direct result-table checks rather
than relying only on the root completion audit: duplicate manifest paths are
rejected; `matrix_results.json` and `matrix_results.csv` must agree by row,
pair, columns and values; score metric keys and MMLU/target aggregate scores
must be consistent; accuracy rows must not use lm-eval limits or failed tasks;
deployment tokens/s, peak memory and memory-limit fields must agree; resource
storage, sparse effective storage/FLOPs and quantized storage estimates must be
internally consistent; and compression backend/materialized/precompressed and
external-command provenance must match the selected compression method.
Delivery verification has also been tightened for portable handoff:
`tools/verify_edge_delivery.py` now checks the root output artifacts,
`edge_results_bundle/` directory and `edge_results_bundle.tar.gz` archive by
size and SHA256. It rejects stale root files, stale bundle directories and stale
tarballs before final results are considered portable.

### Qwen3-8B

Status: full checkpoint downloaded under:

```text
/opt/tiger/models/Qwen3-8B
```

Observed weight shards:

```text
model-00001-of-00005.safetensors 3996250744
model-00002-of-00005.safetensors 3993160032
model-00003-of-00005.safetensors 3959604768
model-00004-of-00005.safetensors 3187841392
model-00005-of-00005.safetensors 1244659840
```

Qwen GPTQ/AWQ precompressed checkpoints have also been staged for local H20
diagnostics:

```text
/opt/tiger/llmcbench_outputs/qwen3_precompressed/Qwen3-8B-GPTQ-Int4
/opt/tiger/llmcbench_outputs/qwen3_precompressed/Qwen3-8B-AWQ
/opt/tiger/llmcbench_outputs/qwen3_precompressed/Qwen3-8B-SmoothQuant-W8A8
/opt/tiger/llmcbench_outputs/qwen3_precompressed/Qwen3-8B-OmniQuant-4bit
```

Both contain `config.json`, `model.safetensors.index.json` and the expected two
or four weight shards. File-readiness passes for the local config's
`precompressed_paths`.
The staging workflow is now captured in `tools/prepare_precompressed_assets.py`,
which can download these known public Qwen GPTQ/AWQ/SmoothQuant/OmniQuant checkpoints from
`hf-mirror.com`, write an audit report, and emit a config copy with
`precompressed_paths` populated for matrix runs. Successful staging now writes
`.llmcbench_precompressed_source.json` in each checkpoint directory and refuses
to silently reuse a complete checkpoint when the marker names a different repo.
For the default Qwen3-8B GPTQ row it also detects the older GPTQModel metadata
path and reports `checkpoint_metadata_mismatch` with a `--force` remediation
instead of continuing with a checkpoint that needs the GPTQModel runtime.
Missing default repos, such as LLaMA3-8B OmniQuant, are now reported as
blocking `repo_not_configured` items and make the preparation report `ok=false`;
the wrapper passes `--fail-on-error` so those gaps stop before the long matrix.
Previous Python 3.9 deployment/accuracy loading was blocked by missing GPTQ/AWQ
runtime imports. The current shared Python now imports `optimum`,
`auto_gptq==0.7.1` and `awq==0.2.9`; target-readiness rows for Qwen3-8B and
LLaMA3-8B GPTQ/AWQ precompressed checkpoints now pass baseline import
readiness. The default Qwen3-8B GPTQ repo is now
`kaitchup/Qwen3-8B-autoround-4bit-gptq`; its metadata identifies
`Qwen3ForCausalLM`, GPTQ 4-bit, group size 128 and no GPTQModel-only quantizer,
so it is expected to use the stable GPTQ runtime (`optimum<2` plus
`auto-gptq`). Optimum 2.x routes GPTQ loading through GPTQModel symbols, so the
stable optional requirements pin `optimum<2`. GPTQModel-format override
checkpoints still need `gptqmodel`, which is part of the target optional
backend install set as `gptqmodel==4.0.0` for Python 3.10+ and is installed
separately with `--no-build-isolation`. This is still not final evidence
because the environment is Python 3.9/H20 rather than the required Python
3.10/RTX 4090 stack.
An isolated `PYTHONUSERBASE=/tmp/llmcbench_quant_userbase` probe with
`optimum`, `auto-gptq`, `autoawq`, `gekko` and `rouge` installed can import the
runtime packages without modifying the system Python. Under that probe:

- Qwen3-8B AWQ reaches `model_load` on `cuda:0`, then one-token generation
  fails with `AttributeError: module 'triton.language' has no attribute
  'interleave'` on the current Triton 2.1 stack.
- Qwen3-8B GPTQ previously stopped at the runtime-import gate when the staged
  checkpoint metadata advertised `gptqmodel:4.0.0`. The default repo has been
  switched to a public AutoRound GPTQ checkpoint without GPTQModel metadata;
  rerun `tools/prepare_precompressed_assets.py` and
  `tools/probe_quant_runtime.py` on the target runtime before treating this as
  final deployment evidence.

These checks are now reproducible through `tools/probe_quant_runtime.py` and
their JSON reports are stored under the Qwen GPTQ/AWQ precompressed output
directories. The same probe is integrated into `tools/run_edge_preflight.py`
through `--check-quant-runtime`; the 4090 wrapper exposes it with
`CHECK_QUANT_RUNTIME=1` and can optionally run one-token generation with
`QUANT_RUNTIME_GENERATE=1`.
The resource tool now records a config-only fallback for these checkpoints when
the quantized runtime is missing. Current diagnostics:

```text
Qwen3-8B GPTQ resource: storage_gb=5.695450127124786, flops=2.18 TFLOPS,
resource_backend=config_only_fallback
Qwen3-8B AWQ resource: storage_gb=5.690465065650642, flops=2.18 TFLOPS,
resource_backend=config_only_fallback
Qwen3-8B SmoothQuant resource: storage_gb=8.805205863900483, flops=2.18 TFLOPS,
resource_backend=config_only_fallback, missing_runtime=compressed_tensors
Qwen3-8B OmniQuant resource: storage_gb=15.463482476770878, flops=2.18 TFLOPS,
resource_backend=model_load
```

Additional runtime probes with the isolated quant runtime overlay show:

- Qwen3-8B SmoothQuant is stopped at the runtime-import gate before model load
  because `compressed_tensors==0.15.0.1` is not Python 3.9 compatible in this
  environment (`TypeError: unsupported operand type(s) for |: 'type' and
  'types.GenericAlias'`). The probe now reports
  `RuntimeError: missing required runtime imports: ['compressed_tensors']`. This
  should be rechecked on the target Python 3.10
  environment.
- Qwen3-8B OmniQuant loads as a Transformers model for resource accounting, but
  one-token generation exits with return code `-8` (SIGFPE) in the current
  Python 3.9 / torch 2.1 H20 environment. This is diagnostic-only and must be
  rechecked on the target 4090 stack.

### LLaMA3-8B

Status: blocked by gated HuggingFace access without an authorized token or local
checkpoint.

A current metadata-only probe wrote
`/data/outputs/llmcbench_edge_4090/edge_prepare_metadata_report.json` and
confirmed that `meta-llama/Meta-Llama-3-8B` returns HTTP 403 through
`https://hf-mirror.com` for `config.json` without authorization. The local
`/data/models/Meta-Llama-3-8B` and `/opt/tiger/models/Meta-Llama-3-8B`
directories are not runnable checkpoints: each is only about 20KB and contains
the HuggingFace cache placeholder tree, not `config.json` or weight files. The
current asset audit reports the configured LLaMA3 base path with
`checkpoint_ready=false`, `model_config_exists=false`,
`model_weights_exist=false` and `weight_status.missing=["weight_files"]`, so
LLaMA3 dense, Wanda and SparseGPT remain blocked until an authorized official
base checkpoint is placed at `/data/models/Meta-Llama-3-8B`.

As a mitigation path for precompressed-method testing, public LLaMA3-8B base
checkpoint candidates were found and metadata access was verified through
`tools/prepare_precompressed_assets.py --metadata-only`:

```text
GPTQ: yujiepan/Meta-Llama-3-8B-gptq-w4g64
AWQ: solidrust/Meta-Llama-3-8B-hf-AWQ
SmoothQuant: HRuiii/Meta-Llama-3-8B-smoothquant-w8a8
```

Their `config.json` and tokenizer metadata can be read from hf-mirror, and GPTQ
and AWQ expose Transformers quantization configs. They are not final substitute
results for the requested official LLaMA3-8B matrix; full weights, runtime
probes and the target 4090 audit still need to run before any LLaMA row can be
reported as final. No base-model OmniQuant default is configured because the
public candidates found during development were Instruct or derivative
checkpoints.

An additional diagnostic OmniQuant candidate was staged from
`compressa-ai/Saiga-Llama-3-8B-OmniQuant`:

```text
/data/models/precompressed/LLaMA3-8B-omniquant
/data/outputs/llmcbench_edge_4090/edge_prepare_llama_omniquant_candidate_report.json
/data/outputs/llmcbench_edge_4090/target_readiness_llama_omniquant_candidate.json
/data/outputs/llmcbench_edge_4090/llama3_omniquant_candidate_resource.json
```

The candidate contains `config.json`, `model.safetensors.index.json` and both
expected weight shards, and its isolated readiness row reports `ok=true` with
`uses_precompressed=true`. It is not accepted as a final official LLaMA3-8B
result because the repo is a Saiga-derived checkpoint; it is only a diagnostic
path for checking the OmniQuant loading and reporting flow. The resource
diagnostic records `resource_backend=config_only_fallback`,
`storage_gb=5.347812380641699`, quantized storage estimate
`4.1143798828125` GB and `36.24 TFLOPS`; model loading fails in the current
Python 3.9 stack with `NameError: name 'QuantizeConfig' is not defined`, so this
is not a final deployment/resource result.

The public LLaMA3-8B GPTQ/AWQ/SmoothQuant candidates have been fully staged
locally and are linked into `/data/models/precompressed`:

```text
/opt/tiger/llmcbench_outputs/llama3_precompressed/LLaMA3-8B-AWQ
/opt/tiger/llmcbench_outputs/llama3_precompressed/LLaMA3-8B-SmoothQuant-W8A8
/opt/tiger/llmcbench_outputs/llama3_precompressed/LLaMA3-8B-GPTQ-W4G64
/data/models/precompressed/LLaMA3-8B-awq
/data/models/precompressed/LLaMA3-8B-smoothquant
/data/models/precompressed/LLaMA3-8B-gptq
```

All three staged checkpoints contain `config.json`,
`model.safetensors.index.json` and the expected weight shards, and the target
precompressed matrix now references them. The latest target-readiness summary
shows GPTQ and AWQ at `ok=2, blocked=0` across LLaMA3-8B and Qwen3-8B.
SmoothQuant file readiness passes for both models but remains blocked by the
`target_python_3_10_required_for:compressed_tensors` runtime gate in the current
Python 3.9 diagnostic environment; it must be rechecked in the target Python
3.10 stack. An older AWQ
resource diagnostic was generated with `resource_backend=config_only_fallback`
because the system Python lacked `autoawq`; it reports
`storage_gb=5.343127744272351` and estimated `flops=2.14 TFLOPS`. The isolated
runtime probe reaches AWQ model load on `cuda:0`, then one-token generation
fails with `AttributeError: module 'triton.language' has no attribute
interleave'`, the
same Triton 2.1 issue observed for Qwen AWQ.

The SmoothQuant checkpoint contains `config.json`, `model.safetensors.index.json`
and all four expected weight shards. Resource evaluation reaches model load and
reports `storage_gb=14.966091142967343`, estimated `flops=2.14 TFLOPS`, and
`resource_backend=model_load`. Its generation probe exits with return code `-8`
(SIGFPE) in the current H20 / Python 3.9 / torch 2.1 environment and must be
rechecked on the target Python 3.10 / RTX 4090 stack.

The first single-file LLaMA GPTQ candidate was too slow to download, so the
diagnostic path switched to the two-shard `yujiepan/Meta-Llama-3-8B-gptq-w4g64`
candidate:

```text
/opt/tiger/llmcbench_outputs/llama3_precompressed/LLaMA3-8B-GPTQ-W4G64
```

It is now fully staged with tokenizer files copied from the verified LLaMA
metadata probe. Resource evaluation uses `resource_backend=config_only_fallback`
because the system Python lacks `optimum`; it reports
`storage_gb=5.477358779869974` and estimated `flops=2.14 TFLOPS`. The current
probe records default GPTQ runtime requirements (`optimum` plus
`auto_gptq|gptqmodel`) and still fails under the Python 3.9 diagnostics, so it
must be repeated on the target Python 3.10 / RTX 4090 stack.

## Real Smoke Result

The following command completed successfully for Qwen3-8B dense resource-only
smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m edge_benchmark pipeline \
  --model-path /opt/tiger/models/Qwen3-8B \
  --model-name Qwen3-8B \
  --compression-method dense \
  --compressed-output-dir /opt/tiger/llmcbench_outputs/qwen3_dense_smoke/compressed \
  --output-root /opt/tiger/llmcbench_outputs/qwen3_dense_smoke \
  --device cuda:0 \
  --device-map auto \
  --dtype fp16 \
  --seqlen 128 \
  --input-length 128 \
  --max-new-tokens 1 \
  --warmup 0 \
  --repeat 1 \
  --skip-accuracy \
  --skip-deploy \
  --min-free-gpu-memory-gb 20 \
  --overwrite-compressed
```

Result excerpt:

```json
{
  "model_name": "Qwen3-8B",
  "compression_method": "dense",
  "storage_gb": 15.261226058006287,
  "total_params": 8190735360,
  "flops": "2.18 TFLOPS",
  "macs": "1.09 TMACs",
  "compute_backend": "analytical_config_estimate"
}
```

This is not the final requested benchmark result because it skips accuracy and
deployment and covers only one model/method. It verifies that the real Qwen3-8B
checkpoint can be loaded and resource output can be generated in this
environment.

After installing `calflops`, a Qwen3-8B dense resource run with `seqlen=64`,
`fp32`, and `attn_implementation=eager` completed with the exact calflops
backend:

```json
{
  "storage_gb": 15.271327385678887,
  "total_params": 8190735360,
  "sparsity": 6.329590411769817e-06,
  "flops": "971.17 GFLOPS",
  "macs": "485.57 GMACs",
  "compute_backend": "calflops"
}
```

## Current Runtime Blocker

Qwen3-8B forward/generation on the current H20 + system Python 3.9 + torch 2.1
environment exits with a floating point exception after checkpoint loading. This
was reproduced with both `edge_benchmark deploy` and a direct forward-only
script, outside the benchmark CLI.

Implication: the checkpoint is present, but this host environment is not suitable
for final Qwen deployment/accuracy runs until the target Python 3.10 dependency
environment is installed and verified.

Use this preflight on the target machine before running the full matrix:

```bash
python tools/run_edge_preflight.py \
  --config configs/edge_4090_matrix.json \
  --check-hf-access \
  --check-model-forward \
  --forward-dtype fp16 \
  --finalize-dry-run
```

If forward fails, retry with `--forward-dtype bf16` and/or
`--forward-attn-implementation eager` to isolate dtype/backend issues.

### Deployment Diagnostic

Qwen3-8B dense deployment succeeds on the current H20 host with the same
memory-oriented dtype and max-memory controls used by the 4090 matrix:

```json
{
  "dtype": "fp16",
  "max_memory": "0:22GiB,cpu:120GiB",
  "tokens_per_second": 63.844070909714155,
  "peak_memory_allocated_gb": 15.311979293823242,
  "peak_memory_reserved_gb": 16.44140625,
  "deployment_memory_limit_gb": 24.0,
  "deployment_memory_limit_exceeded": false
}
```

This diagnostic confirms the checkpoint and deployment code path can run under
the 24GB deployment gate, but it is still not a final 4090 result because it was
measured on the H20/Python 3.9 development environment. The final 4090 run
should use the documented Python 3.10 environment and pass `--check-model-forward`.

### Accuracy Diagnostic

After installing `datasets` and `lm-eval` into the current user Python 3.9
environment, a Qwen3-8B dense PIQA smoke with `limit=1` completed:

```json
{
  "tasks": ["piqa"],
  "elapsed_seconds": 39.0309202671051,
  "results": {
    "piqa": {
      "acc,none": 1.0,
      "acc_norm,none": 1.0
    }
  }
}
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_CACHE=/opt/tiger/llm_eval_datasets \
python3 -m edge_benchmark accuracy \
  --model-path /opt/tiger/models/Qwen3-8B \
  --model-name Qwen3-8B \
  --compression-method dense \
  --dtype fp32 \
  --attn-implementation eager \
  --tasks piqa \
  --limit 1 \
  --batch-size 1 \
  --device cuda:0 \
  --cache-dir /opt/tiger/hf_cache \
  --data-dir /opt/tiger/llm_eval_datasets \
  --lm-eval-output /opt/tiger/llmcbench_outputs/qwen3_accuracy_smoke/lm_eval_raw \
  --output /opt/tiger/llmcbench_outputs/qwen3_accuracy_smoke/accuracy.json \
  --min-free-gpu-memory-gb 20
```

This validates the real accuracy path on one small sample. It is not the final
benchmark because it covers only Qwen3-8B dense on PIQA with `limit=1`, uses
fp32/eager for this host, and does not cover the full task/method matrix. The
temporary Python 3.9 install produced dependency conflict warnings, so final
results should still be generated in the documented isolated Python 3.10
environment.

### Initial Five-task Accuracy Smoke Attempt

A Qwen3-8B dense smoke over
`mmlu,arc_challenge,hellaswag,piqa,winogrande` with `limit=1` started
successfully and streamed lm-eval logs. The run failed while MMLU was expanding
and downloading many `cais/mmlu` parquet subsets because `hf-mirror.com`
returned HTTP 429 rate limits.

Implication: the five-task accuracy command path is wired, but full task runs
should be executed after dataset cache prewarming or with task subsets to avoid
resolver rate limits. The target 4090 environment should keep
`HF_DATASETS_CACHE` persistent across retries.
Use `tools/prewarm_lm_eval_tasks.py --config configs/edge_4090_matrix.json
--expand-mmlu` to warm these datasets one task at a time with sleeps/retries
before running the full matrix. `--expand-mmlu` avoids loading the full MMLU
group at once by prewarming all 57 MMLU subtasks incrementally. By default the
prewarm tool uses one selected/available model from the matrix config because
the dataset cache is shared; pass `--all-models` only when every model path
should be probed during prewarm.

### Five-task Split Accuracy Smoke

After cache warmup, a Qwen3-8B dense split accuracy pipeline over the full target
task list completed with `limit=1`:

```json
{
  "tasks": [
    "mmlu",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande"
  ],
  "split_accuracy_tasks": true,
  "results": {
    "mmlu": {
      "acc,none": 0.7894736842105263
    },
    "arc_challenge": {
      "acc,none": 1.0,
      "acc_norm,none": 1.0
    },
    "hellaswag": {
      "acc,none": 0.0,
      "acc_norm,none": 0.0
    },
    "piqa": {
      "acc,none": 1.0,
      "acc_norm,none": 1.0
    },
    "winogrande": {
      "acc,none": 0.0
    }
  },
  "failed_tasks": {}
}
```

This validates the real Qwen3 dense accuracy path for all five requested task
families, using split task execution and `limit=1`. It is still a smoke result,
not a final benchmark score.

### MMLU Subtask Accuracy Smoke

A Qwen3-8B dense smoke for `mmlu_abstract_algebra` with `limit=1` completed:

```json
{
  "mmlu_abstract_algebra": {
    "acc,none": 0.0
  }
}
```

This validates the MMLU task path on a real Qwen3 checkpoint. The full `mmlu`
group remains sensitive to hf-mirror resolver rate limits because it expands
many subtasks.

### Split Accuracy Smoke

A real Qwen3-8B dense pipeline using split accuracy mode completed for
`mmlu_abstract_algebra,piqa` with `limit=1`:

```json
{
  "tasks": [
    "mmlu_abstract_algebra",
    "piqa"
  ],
  "split_accuracy_tasks": true,
  "results": {
    "mmlu_abstract_algebra": {
      "acc,none": 0.0
    },
    "piqa": {
      "acc,none": 1.0,
      "acc_norm,none": 1.0
    }
  },
  "failed_tasks": {}
}
```

This validates that the pipeline can run lm-eval tasks one at a time and merge
the resulting `accuracy.json`. It also verifies that already cached MMLU
subtasks can be reused without rerunning the full MMLU group.

## Real Qwen3 Wanda Compression Smoke

A real Qwen3-8B Wanda compression smoke completed with a tiny calibration setup:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m edge_benchmark pipeline \
  --model-path /opt/tiger/models/Qwen3-8B \
  --model-name Qwen3-8B \
  --compression-method wanda \
  --compressed-output-dir /opt/tiger/llmcbench_outputs/qwen3_wanda_smoke/compressed \
  --output-root /opt/tiger/llmcbench_outputs/qwen3_wanda_smoke \
  --device cuda:0 \
  --device-map auto \
  --dtype fp16 \
  --attn-implementation eager \
  --seqlen 64 \
  --calibration-samples 2 \
  --calibration-batch-size 1 \
  --sparsity 0.5 \
  --skip-accuracy \
  --skip-deploy \
  --min-free-gpu-memory-gb 20 \
  --overwrite-compressed
```

Resource result excerpt:

```json
{
  "storage_gb": 15.26122615672648,
  "total_params": 8190735360,
  "sparsity": 0.4240079722219227,
  "flops": "1.09 TFLOPS",
  "macs": "543.55 GMACs",
  "compute_backend": "analytical_config_estimate"
}
```

The 16GB temporary compressed checkpoint was deleted after copying the JSON
records to `/opt/tiger/llmcbench_outputs/qwen3_wanda_smoke_records` to preserve
disk space. This smoke validates a real 8B pruning path but is not a final
benchmark because it uses only 2 calibration samples and skips accuracy/deploy.

## Real Qwen3 SparseGPT Compression Smoke

A real Qwen3-8B SparseGPT-style compression smoke completed with the same tiny
calibration setup:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m edge_benchmark pipeline \
  --model-path /opt/tiger/models/Qwen3-8B \
  --model-name Qwen3-8B \
  --compression-method sparsegpt \
  --compressed-output-dir /opt/tiger/llmcbench_outputs/qwen3_sparsegpt_smoke/compressed \
  --output-root /opt/tiger/llmcbench_outputs/qwen3_sparsegpt_smoke \
  --device cuda:0 \
  --device-map auto \
  --dtype fp16 \
  --attn-implementation eager \
  --seqlen 64 \
  --calibration-samples 2 \
  --calibration-batch-size 1 \
  --sparsity 0.5 \
  --skip-accuracy \
  --skip-deploy \
  --min-free-gpu-memory-gb 20 \
  --overwrite-compressed
```

Resource result excerpt:

```json
{
  "storage_gb": 15.261226198635995,
  "total_params": 8190735360,
  "sparsity": 0.42400796391985984,
  "flops": "1.09 TFLOPS",
  "macs": "543.55 GMACs",
  "compute_backend": "analytical_config_estimate"
}
```

The 16GB temporary compressed checkpoint was deleted after copying JSON records
to `/opt/tiger/llmcbench_outputs/qwen3_sparsegpt_smoke_records`.

## Partial Qwen Smoke Report

Current real Qwen smoke results were assembled into a partial report:

```text
/opt/tiger/llmcbench_outputs/qwen_smoke_report/matrix_results.json
/opt/tiger/llmcbench_outputs/qwen_smoke_report/edge_report.md
```

The report currently contains:

- Qwen3 dense split accuracy smoke over MMLU group, ARC-Challenge, HellaSwag,
  PIQA and WinoGrande with `limit=1`.
- Qwen3 dense deployment smoke with `fp16` and
  `max_memory=0:22GiB,cpu:120GiB` on H20. This passes the 24GB deployment gate
  and remains diagnostic only because it is not a 4090/Python 3.10 run.
- Qwen3 dense resource smoke with `calflops`.
- Qwen3 Wanda/SparseGPT resource smoke with tiny calibration settings.

This report is a partial development artifact, not the final required two-model
full-matrix result.
Goal audit now distinguishes these Wanda/SparseGPT resource smoke records from
deployable compressed checkpoints: the smoke JSON exists, but the referenced
compressed checkpoint directories are not present in this workspace.
The partial report uses the enriched matrix schema, including runtime metadata,
numeric resource fields, effective sparse storage/FLOPs and dense accuracy
retention fields. The newest dense deployment diagnostic passes the 24GB gate
under fp16/max-memory controls, but it must still not be treated as a final
4090-valid row because it was measured on H20/Python 3.9.

It can be regenerated from the local smoke outputs with:

```bash
python tools/build_qwen_smoke_report.py
```

A machine-readable current-goal audit snapshot is available at:

```text
/opt/tiger/llmcbench_outputs/current_goal_audit.json
```

It can be regenerated with:

```bash
python tools/build_goal_audit.py
```

The audit records the current git commit/remote, Qwen checkpoint shard sizes,
partial Qwen report coverage, dense five-task split accuracy scores, dense
deployment diagnostic memory/tokens/s, dense resource `calflops` metrics,
Wanda/SparseGPT resource smoke metrics, 4090 memory controls and method
readiness blockers. It also includes a `requirement_checklist` that maps the
requested accuracy, deployment and resource tools to current evidence and
remaining blockers, plus `final_artifacts` showing whether the expected 4090
output files exist, including `edge_prewarm_report.json`, whether final
`completion_audit.json` has `"ok": true`, and summaries of final matrix
coverage or completion-audit failures when those artifacts exist. It now also
reports `missing_required_artifacts` and top-level presence summaries for
`matrix_results.csv` and `edge_report.md`, so the final result table and report
deliverables are visible without digging through nested artifact maps. The
prompt-to-artifact checklist repeats those concrete missing final artifacts in
`final_two_model_results.missing_or_unverified`, including explicit
`matrix_results.csv` and `edge_report.md` entries when they are absent. It also
surfaces `final_artifacts.mmlu_subtask_coverage`, a compact summary of the
completion audit's final MMLU subtask coverage gate with required subtask count,
scoped row count and missing-subtask violations when final artifacts exist. It
now also exposes top-level `completion_state`, a stable handoff summary with
`status`, `complete`, incomplete prompt sections, missing final artifact count,
remaining blockers, final output root, `results_bundle_verify_ok`,
`results_bundle_verify_complete`, `delivery_verify_ok`,
`delivery_verify_substatus_ok`, `target_completion_ok` and target-readiness
command groups.
It also exposes top-level `target_path_access` and mirrors it into
`completion_state.target_path_access_ok`, so `/data` ownership or path
creatability blockers are visible before a target-side `edge_asset_audit.json`
exists. Automation should use `completion_state.complete`, not partial smoke
success, as the goal-level completion flag. That flag now remains false unless
`edge_results_bundle_verify.json` reports `"ok": true` and `"complete": true`,
and `edge_delivery_verify.json` reports `"ok": true` with every root,
bundle-directory and archive substatus check passing.

## Current Code And Audit Status

The code branch for this work is `edge-pipeline-updates` on:

```text
git@code.byted.org:data/llmcbench.git
```

The latest refreshed goal audit records whether local `HEAD` and
`origin/edge-pipeline-updates` are in sync for the current checkout. This is
code/documentation readiness only; the final benchmark remains incomplete until
the full 4090 matrix produces `completion_audit.json` with `"ok": true`.
Use `/opt/tiger/llmcbench_outputs/current_goal_audit.json` and
`/opt/tiger/llmcbench_outputs/edge_portable_bundle/portable_bundle_verify.json`
for the exact refreshed commit and portable-bundle provenance. The snapshot
records `scripts/init_edge_storage.sh` as `artifacts.target_storage_init` and
includes that script in the portable environment prompt-to-artifact checklist
and gate list, so fresh-target `/data` initialization is now visible in the
machine audit rather than only in the runbook.
Portable bundle verification now also hard-requires the migration-critical
tooling inside the exported code package: environment verification, external
method configuration, quantized-runtime probing, goal auditing, result
collection/rendering, final validation and result/delivery verification. This
keeps a copied bundle from being accepted if a required completion-gate script is
missing even when the top-level runbook and manifest still exist.
The verifier also has strict final-handoff provenance gates:
`--require-clean-code` rejects bundles exported from a dirty checkout, and
`--require-commit` rejects bundles whose portable manifest and handoff manifest
do not match the intended git commit.
Archive verification now reads the tarball contents and compares each archived
file's size and SHA256 against `PORTABLE_BUNDLE_MANIFEST.json`, so a stale
tarball with matching file names but old content is rejected. The archived
`PORTABLE_BUNDLE_MANIFEST.json` itself is also compared against the extracted
manifest file, and duplicate or unsafe archive paths are rejected.
Non-file/non-directory archive members, such as symbolic links or special files,
are also rejected so extraction cannot depend on platform-specific link or
device handling. The archive must also use the extracted bundle directory name
as its single top-level directory.
The current-goal audit now reads `portable_bundle_verify.json` and surfaces the
strict bundle evidence, including archive status, verified file count, dirty
flags, required commit, tarball verified-file count, archive top-level directory
names, duplicate archive path count, unsafe archive member count, unsupported
archive member count and archive size/SHA256 mismatch counts, while keeping the
overall goal incomplete until the real 4090 matrix and final result-bundle gates
pass.

Recent completion-audit hardening now rejects stale or incomplete final
artifacts beyond basic schema checks:

- `matrix_config_resolved.json` must preserve the source config path, selected
  model entries including paths, selected methods, source defaults, current
  output root, external command templates, precompressed paths and override maps.
- `matrix_run_report.json` must cover the expected scoped jobs, contain no
  unexpected or duplicate model/method jobs, keep job summary/log paths under
  the output root, and keep top-level status lists consistent with job details
  without missing or stale extra entries. `skipped_external_not_configured` is
  accepted only when final audit is run with `--allow-skipped-external`.
- `matrix_results.json` and `matrix_results.csv` must match by row, column and
  value; result rows must also match source `compression.json`, `accuracy.json`,
  `deployment.json`, `resource.json` and job `summary.json` artifacts.
- `matrix_run_report.json` jobs must be final statuses such as `completed` or
  validated `skipped_existing`; dry-run jobs and top-level dry-run reports are
  rejected as final evidence.
- Required auxiliary reports such as asset preparation, asset audit, preflight
  and quant runtime reports cannot be dry-run reports in final completion
  audit. The optional backend dependency-resolution report is allowed to remain
  a pip dry-run report because that is its expected scope, but it must include
  `mode=dry_run`, non-empty requested package/import summaries and
  `pip_returncode=0`. `gptqmodel_install` is checked only when GPTQModel is
  explicitly requested or checkpoint metadata requires it; the default Qwen3-8B
  AutoRound GPTQ path uses `optimum<2` plus `auto-gptq` and does not require
  GPTQModel requirements to pass. A Python-version-marked GPTQModel capability
  entry may still appear in optional backend dry-runs for override checkpoints;
  it is not a blocker for the default stable path. The metadata preparation
  report is allowed to remain `metadata_only=true`; full asset preparation
  reports are not.
- Required full `edge_prepare_assets_report.json` must contain non-empty model
  and file preparation items, and every item must pass; an empty `{ok:true}`
  report is rejected.
- Required `edge_prepare_metadata_report.json` must be `metadata_only=true`,
  contain non-empty model and file metadata-preparation items, and every item
  must pass; an empty `{ok:true}` report is rejected.
- Required `edge_asset_audit.json` must include model, precompressed-checkpoint
  and path sections covering `hf_cache_dir`, `data_dir`, `output_root`,
  `calibration_file` and `prompt_file`; an empty `{ok:true}` report is rejected.
  It now also reports directory writability/create checks through
  `unwritable_paths` and per-path `can_create`, and readiness/final storage
  gates reject non-creatable cache, dataset or output directories before the
  long matrix run starts.
- Required `edge_quant_runtime_report.json` must contain non-empty probe rows
  and every quant runtime row must pass; an empty `{ok:true}` report is
  rejected.
- Required `edge_prepare_precompressed_report.json` must contain non-empty
  successful preparation items, no blockers, and an existing valid
  `updated_config` JSON path that matches the config passed to readiness/final
  audit; an empty `{ok:true}` report is rejected.
- `edge_preflight_report.json` must include the expected preflight steps
  (`asset_audit`, `verify_env`, `method_readiness`, `matrix_dry_run`) and all
  required steps must pass; an empty `{ok:true}` report is rejected. When
  optional backend or quant runtime final reports are required, the preflight
  report must also include `optional_backend_resolution` and `quant_runtime`
  steps respectively.
- Final `edge_prewarm_report.json` cannot be a dry-run report; preflight
  dry-runs are written separately as `edge_prewarm_dry_run_report.json`.
- `edge_report.md` must be re-renderable from current `matrix_results.json` and
  `matrix_run_report.json`; stale report metadata or table content is rejected.
  Report rendering also rejects table row/header width mismatches, so added
  result columns cannot silently shift deployment, accuracy or resource tables.
- Source-artifact mismatch reports include `artifact_field_path`, so stale
  collected rows point directly at fields such as `manifest.backend` in
  `compression.json` or `tokens_per_second` in `deployment.json`.
- `matrix_finalize_report.json` must point at the standard final artifacts under
  the current output root. Finalize dry-runs now write the same report with
  `status=dry_run` steps so path and gate checks can be inspected before the
  long 4090 matrix starts, but completion audit rejects dry-run finalize steps
  as non-final evidence.

These gates are exercised by `tools/run_smoke_checks.py`, but they do not
replace the required full two-model 4090 run.

The dense row in the older partial report has complete accuracy/resource fields
when scoped to `Qwen3-8B + dense`, but that report still references the older
H20/fp32 deployment record and therefore does not pass the final validator:

```bash
python tools/validate_edge_results.py \
  --config configs/edge_local_h20_matrix.json \
  --results /opt/tiger/llmcbench_outputs/qwen_smoke_report/matrix_results.json \
  --only-model Qwen3-8B \
  --only-method dense \
  --allow-unexpected \
  --output /opt/tiger/llmcbench_outputs/qwen_smoke_report/validation_dense.json
```

Validation result:

```json
{
  "ok": false,
  "expected_rows": 1,
  "missing_fields": [],
  "memory_violations": [
    {
      "model": "Qwen3-8B",
      "method": "dense",
      "measured_peak_gb": 32.83984375,
      "source": "deployment_json"
    }
  ]
}
```

### Four-task Accuracy Smoke

A Qwen3-8B dense smoke over the non-MMLU tasks
`arc_challenge,hellaswag,piqa,winogrande` with `limit=1` completed:

```json
{
  "arc_challenge": {
    "acc,none": 1.0,
    "acc_norm,none": 1.0
  },
  "hellaswag": {
    "acc,none": 0.0,
    "acc_norm,none": 0.0
  },
  "piqa": {
    "acc,none": 1.0,
    "acc_norm,none": 1.0
  },
  "winogrande": {
    "acc,none": 0.0
  }
}
```

This validates four of the five requested accuracy task paths on the real Qwen3
checkpoint. MMLU remains the only target task smoke blocked by hf-mirror rate
limits in this environment.

### Deployable Qwen3 Wanda Diagnostic

A current deployable Qwen3-8B Wanda checkpoint was generated on H20 with tiny
calibration settings:

```json
{
  "compressed_model_path": "/opt/tiger/llmcbench_outputs/qwen3_wanda_deployable/compressed",
  "compression_method": "wanda",
  "dtype": "fp16",
  "seqlen": 128,
  "calibration_samples": 2,
  "calibration_batch_size": 1,
  "sparsity": 0.5
}
```

The generated checkpoint is complete and can be loaded by Transformers. A
deployment diagnostic with `max_memory=0:22GiB,cpu:120GiB` passed the 24GB gate:

```json
{
  "tokens_per_second": 64.38044126937777,
  "deployment_memory_measured_peak_gb": 15.333984375,
  "deployment_memory_limit_gb": 24.0,
  "deployment_memory_limit_exceeded": false
}
```

The resource diagnostic for this checkpoint reports:

```json
{
  "storage_gb": 15.261226277798414,
  "effective_storage_gb_sparse_approx": 8.79034468130857,
  "sparsity": 0.42400797148938774,
  "flops": "1.95 TFLOPS",
  "compute_backend": "calflops"
}
```

The compressed checkpoint also completed a four-task accuracy diagnostic with
`limit=1`:

```json
{
  "arc_challenge": 0.0,
  "hellaswag": 0.0,
  "piqa": 1.0,
  "winogrande": 0.0
}
```

It also completed an MMLU subtask smoke for `mmlu_abstract_algebra` with
`limit=1`:

```json
{
  "mmlu_abstract_algebra": 0.0
}
```

This is useful deployability evidence for the Qwen3 Wanda path, but remains a
development diagnostic because it uses tiny calibration and H20/Python 3.9
rather than the required full two-model 4090/Python 3.10 matrix.

### Deployable Qwen3 SparseGPT Diagnostic

A current deployable Qwen3-8B SparseGPT-style checkpoint was also generated on
H20 with the same tiny calibration settings:

```json
{
  "compressed_model_path": "/opt/tiger/llmcbench_outputs/qwen3_sparsegpt_deployable/compressed",
  "compression_method": "sparsegpt",
  "dtype": "fp16",
  "seqlen": 128,
  "calibration_samples": 2,
  "calibration_batch_size": 1,
  "sparsity": 0.5
}
```

The generated checkpoint is complete and can be loaded by Transformers. A
deployment diagnostic with `max_memory=0:22GiB,cpu:120GiB` passed the 24GB gate:

```json
{
  "tokens_per_second": 63.63371328021812,
  "deployment_memory_measured_peak_gb": 15.333984375,
  "deployment_memory_limit_gb": 24.0,
  "deployment_memory_limit_exceeded": false
}
```

The resource diagnostic for this checkpoint reports:

```json
{
  "storage_gb": 15.26122631970793,
  "effective_storage_gb_sparse_approx": 8.790344804199322,
  "sparsity": 0.4240079650186623,
  "flops": "1.95 TFLOPS",
  "compute_backend": "calflops"
}
```

The compressed checkpoint also completed a four-task accuracy diagnostic with
`limit=1`:

```json
{
  "arc_challenge": 0.0,
  "hellaswag": 0.0,
  "piqa": 1.0,
  "winogrande": 0.0
}
```

It also completed an MMLU subtask smoke for `mmlu_abstract_algebra` with
`limit=1`:

```json
{
  "mmlu_abstract_algebra": 0.0
}
```

This is useful deployability evidence for the Qwen3 SparseGPT path, but remains
a development diagnostic because it uses tiny calibration and H20/Python 3.9
rather than the required full two-model 4090/Python 3.10 matrix.
