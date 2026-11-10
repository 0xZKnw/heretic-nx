# Experiment log

## 2026-11-10 — LFM3 1.7B residual-stream pilot

- Engine revision: `764fbb44e0edc577cdcf32adb6224459f30c7c61`.
- Source: official `LiquidAI/LFM3-1.7B`, revision
  `1da06bec6a819da10b54c67e5c90972b46d81871`; checkpoint SHA-256
  `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`.
- Hardware/runtime: Apple M5, 24 GiB unified memory, Metal/MPS, BF16,
  Python 3.12, PyTorch 2.15.0, Transformers 5.22.2. One model job at a time.
- Workload: 128 safe and 128 target geometry rows, 24 safe and 24 target
  group-disjoint training-selection rows, seed 1703, greedy decoding,
  96 new tokens, batch size 4, thinking disabled. No prompt truncation.
- Hypothesis: existing NX contrastive residual axes and norm-preserving weight
  edits can reduce LFM3 lexical refusals within a benign first-token KL cap of
  0.03. Compare raw axes with rank-4 benign-PCA protection; edit both output
  projections in blocks 4–23 and restore the original weights per candidate.
- Correctness: official checkpoint hash and all 48 hybrid projections checked;
  raw per-block hook collection; deterministic held-apart manifests; immutable
  weight snapshots; reloaded output evaluated separately. No numerical
  equivalence between original and edited weights is intended.

### Rejected initial collection fit

Fitting all 24 block axes failed closed: block 0 has exactly zero contrast at
the fixed assistant suffix because its convolution sees no user tokens there.
Measured target-minus-safe mean norm and safe variation are both zero at block
0. Restricting the fit, not only the edit, to blocks 4–23 resolves the failure.
The residual cache was reused because collection itself was correct. A
regression test covers constant early states.

### Comparable development measurements

The untouched base has 22/24 lexical markers. All rows and inference settings
are identical across candidates. These are refusal proxies, not semantic
success scores.

| Geometry | Scale | Markers / 24 | Mean benign first-token KL | Decision |
| --- | ---: | ---: | ---: | --- |
| Raw | 0.10 | 22 | 0.002759 | No target improvement |
| Raw | 0.25 | 22 | 0.009436 | No target improvement |
| Raw | 0.50 | 17 | 0.026827 | Selected under the predeclared cap |
| Raw | 1.00 | 5 | 0.069292 | Reject: exceeds KL cap |
| Protected rank 4 | 0.10 | 22 | 0.002500 | No target improvement |
| Protected rank 4 | 0.25 | 21 | 0.008614 | Less target improvement |
| Protected rank 4 | 0.50 | 14 | 0.032391 | Reject: exceeds KL cap |
| Protected rank 4 | 1.00 | 6 | 0.088101 | Reject: exceeds KL cap |

The stronger refusal effects at higher intensity do not justify silently
relaxing the drift criterion. The selected checkpoint is BF16; no quantization
or context/output-length change is used to improve these numbers.

This is a model-editing experiment, not a throughput optimization claim. The
tensor implementation is covered by tests, not a formal numerical proof.
CrossHair separately explored the production selection rule with two candidate
records, nonnegative counts and symbolic floating KL values; that scope does
not verify model inference or arbitrary-length candidate lists.

Independent test and paired capability results are recorded after the frozen
checkpoint is reloaded; they must not be used to tune this same holdout.

### Second wave — zero-development-refusal objective

The conservative build did not meet the required near-zero-refusal objective
and was archived locally. Its independent validation was interrupted after
the base target arm and during safe generation; no edited test result was used
for candidate tuning. The safe response workload was also bounded to the first
104 test rows, rather than generating the entire 6,265-row harmless test split.

The revised, explicit protocol requires zero markers on the same 24 training
selection prompts, rejects empty answers, and chooses minimum KL among the
successful candidates. The development KL ceiling is now 0.5, rather than
0.03; this is a disclosed objective change, not a claim of equal preservation.
Raw and protected directions are tested at scales 1.25, 1.5 and 2.0, preserving
BF16, context, output length, batch size, source weights and data partitions.
The actual independent KL and paired capability difference must be reported,
including any failed capability gate.

CI initially exposed another missing test dependency (`datasets`) in an
existing provenance import chain. CI now installs the existing `experiments`
extra as well as `dev` and `gguf`; no production credentials are needed.
