# EXL3 verification record

Source checkpoint and recipe are pinned in the model card and manifest. All
three Modal campaigns have completed/stopped with zero tasks: one H100 at
a time, retries disabled, bounded conversion/worker waits, no speed benchmark.
The strict 8-bpw release is selected; strict 4 and mixed 4.49 fail their
respective frozen KL limits and are not distributed.

## Source and numerical checks

- `python -m compileall -q work/exl3-conversion` and `python -m py_compile`
  passed for the executed launchers/workers.
- Deterministic recipe checks compare all 142 projection allocations against
  the exact published descriptor. CPU preflight checks source SHA-256, model
  config/tokenizer and every expected projection shape before GPU conversion.
- Invalid converter bitrates 0/3/5/16 are rejected by the reused helper.
- `crosshair check verify_command.py --analysis_kind=asserts
  --per_condition_timeout=5 --report_all` reports confirmation for the real
  command helper over its two valid integer values, 4 and 8. An earlier
  unconstrained invocation encountered its expected ValueError at invalid
  bitrate 0; that diagnostic is retained locally. This is symbolic
  counterexample search over argument routing, not a tensor/GPU proof.
- The actual native KL helper passes the independent two-outcome oracle
  KL([0.5,0.5] || [0.25,0.75]) = 0.1438410362, identity, mismatched-shape
  and NaN rejection checks.
- Conversion audits check 142 projections, all expected source mappings,
  713 nonempty output tensors, finite floating payloads and the exact
  per-projection bitrate. ExLlama's unquantized NX control against official
  HF FP32 produces KL 0.00004372903205 over all 104 safe prompts.
- Each output file is downloaded through streaming Modal SDK reads and
  checked against its cloud SHA-256 before inference. An earlier CLI transfer
  of the rejected mixed model had the correct length but a wrong hash;
  the integrity check rejected it before any model execution.

## Native MLXL3 checks

The production engine archive v2.0.4-beta.1 (commit ead60d0323e9) is
SHA-256 f3a71e68139fdc9072b991582434148ca2a6c74ee25fb04f2118c5968ef74e9c.
All four runtime payloads match engine.json. Hardware: Apple M5, 24 GiB.

- `python work/exl3-conversion/native_check.py 8`: 104 complete pretokenized
  prompt forwards, 128,000 finite values per output, mean CUDA-to-MLXL3 KL
  0.00008576118, argmax agreement 99.04%.
- A separate 499-token bilingual case crosses the 256-token sliding window:
  runtime KL 0.00026417794. Both runtime checks use a predeclared 0.002 cap.
- Independent reset/replay of the checked prompt is bit-exact.
- The existing production `scripts/smoke-release.py` runs two real NDJSON
  chat turns and passes its CEDAR-42 recall oracle. Processes exit cleanly.
  This exercises the engine and Desktop bridge protocol; no GUI install or
  user registry was changed.
- Original HF FP32 -> native MLXL3 first-token KL over the same 104 benign
  prompts is 0.04389917965, below 0.05. Reusing the unchanged FP32 baseline
  collected during the four-bit campaign reproduces the eight-bit CUDA KL
  within 1e-6 before comparison to native logits.

## Scope and remaining gaps

Generation marker counts were measured on H100 ExLlamaV3, thinking disabled,
raw greedy, no repetition penalty, at most 96 tokens. The 104-row retest is
reused development evidence from source checkpoint selection, not an untouched
holdout. Native full-vocabulary KL is measured at the first output token;
these are not sequence-level KL or semantic task-success scores.

No EXL3-wide capability benchmark or proof of GPU computation was performed.
The CUDA reference and Apple native tests do not certify all other devices,
runtime versions, context lengths or user prompts. Existing BF16 capability
noninferiority remains inconclusive and is not inherited as an EXL3 guarantee.

The production `mlxl3 hub details ... --folder 8bpw` discovers the published
variant, but its preview marks LFM3 unsupported. The same flag appears for
all variants of the original published LFM3 EXL3 model. Native local loading,
forward and chat pass independently. The documented HF-download/local-folder
route is the verified path; no engine source changes or spoofed model type
were introduced to alter the preview. See validation/hub-preview.json.
