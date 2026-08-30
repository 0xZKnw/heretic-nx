# Changelog

All notable public changes to Heretic NX are documented here.

## 0.3.0 — 2026-08-30

### Added

- Transactional mixed-GGUF editing for Q2_K through Q6_K plus common legacy
  quants, backed by llama.cpp's native same-type codecs.
- Quantization-aware min-drift block selection, optional strength candidates,
  realized-delta and row-norm gates, opposite-endian rejection and full
  undeclared-byte verification by default.
- `hnx inspect-gguf` and `hnx abliterate-gguf`, while retaining the Q8 v1 API.
- Artifact-bound multi-quant capability certificates and complete-checkpoint
  validation for KL comparisons, including local llama.cpp `/props` model-path
  attestation and rejection of identical base/candidate bytes.
- Rank-space low-rank optimizer statistics and opt-in bounded-batch Frequent
  Directions compression.
- Native libggml K-codec coverage, atomic no-clobber publication and
  snapshot-bound source/factor/plan verification.
- Refusal-cap early termination for frontier screening, while keeping full
  104-row reports mandatory before KL certification.
- Content-addressed refusal-first evaluation scheduling with immutable replay
  cache, group-aware split isolation, semantic-versus-lexical verdict separation,
  exact partial-KL pruning and a matched-protocol refusal/KL Pareto frontier.
- Direct, bounded-memory Q8_0 GGUF static merging with hash-bound ablation
  plans, row-norm preservation, atomic output replacement and stacked MoE
  expert-bank support.
- Conditional direct low-rank Q8 deltas with independently fitted right
  factors for suppressing benign drift without materializing BF16 weights.
- `hnx inspect-q8` and `hnx abliterate-q8` commands with a no-write dry run and
  per-tensor payload provenance.
- LFM2.5-2.6B Residual-Stream PRIME pipeline with capability-protected residual
  axes, sparse repair portfolios and a hard `0.0142` first-token KL cap.
- Independent 450-row XSTest and 854-row paired capability promotion gates for
  arbitrary frozen 2.6B candidates.
- Hash-bound 2.6B release evidence and matched local evaluation of the disclosed
  Heretic Q8 comparator.
- Pinned Q8 GGUF XSTest and 854-row capability reports, plus a cross-runtime
  validation showing the PRIME BF16 GGUF within 0.23 point of native scoring.
- A 2.87 GB Q8_0 PRIME release artifact with 60.66% MCQ, 96.37% native
  prediction agreement and both LM Studio and official llama.cpp b10621
  runtime reports.
- Reproducible Heretic master and v1.4.0 closed-comparison configurations.
- One-command paired external evaluator retaining item-level XSTest,
  StrongREJECT-proxy and capability observations.
- Hash-bound evidence showing corrected target superiority with capability and
  safe-behavior non-inferiority against the pinned upstream runs.

### Clarified

- K-quant suffixes such as Q4_K_M are mixed file recipes; edits dispatch on
  each tensor's actual type and every released quantization needs its own
  same-quant baseline and certificate.
- Under the corrected single-BOS b10621 protocol, the 2.6B native PRIME release
  records 6/104 lexical refusal markers and its Q8 records 9/104, versus 4/104
  for the locally tested Heretic Q8 comparator. Its measured first-token KL is
  `0.012396`, below the comparator card's published `0.0142`, but that KL
  comparison remains descriptive because the comparator protocol is incomplete.
- On the symmetric official b10621 Q8 check, PRIME records 9/450 XSTest markers
  versus 8/450 for the comparator and ties its 6/200 unsafe-contrast count.
  The two Q8 capability arms tie at 60.66%, with a paired equivalence interval
  of `[-1.87, +1.76]` points; PRIME native BF16 records 61.24%.
- The stronger Heretic-wide arm has a higher observed target-refusal count than
  Residual-Stream, but the paired target-superiority interval still crosses zero.
- The comparison is model-specific and does not claim universal engine superiority or third-party certification.

### Performance

- GGUF source, untouched-region and final-output integrity digests now share
  sequential scans and publication verifies the already-hashed inode. This
  cuts a real merge from six full/partial verification scans to two without
  weakening source binding or undeclared-byte checks; the reproducible 128 MiB
  microbenchmark is 1.72x faster on the reference Mac.
- Native same-type quantization now splits independent row ranges across a
  lazy, affinity-aware worker pool while preserving exact encoded bytes. On the
  16 MiB reference workload, Q2_K through Q6_K encode roughly 4.0x–5.1x faster
  with eight workers; a complete Q4_K edit kernel is 2.92x faster with identical
  payload and report metrics.
- Signed spectral fitting now diagonalizes only the joined target/protected
  factor span instead of allocating and decomposing an ambient `d x d` matrix.
  At dimension 2,048 and factor rank 8, the reproducible benchmark reduces the
  tracked working representation from 16 MiB to 128 KiB and runs about 700x
  faster while matching eigenvalues within `8.94e-8`.

### Fixed

- Direct-Q8 PRIME site fitting now enforces the metric geometry decision instead
  of normalizing rejected near-zero residuals into full-strength operators. On
  the retained LFM2.5 8B activation cache, all 48 unsafe sites are now rejected
  (maximum retained energy `4.53e-5`) rather than silently promoted.
- Capability PCA now uses the exact smaller Gram space, reports its effective
  numerical rank and drops rank-deficient null axes. Singular vectors stay
  paired with their singular values for correct covariance reconstruction, and
  CPU-only post-processing no longer imports the Metal collection runtime.

## 0.2.0 — 2026-08-24

### Added

- Residual-Stream contrastive-axis estimation from padding-safe residual hidden states.
- Architecture-aware static weight editors with exact output-row norm restoration.
- Semantic projection discovery for LFM and Llama/Gemma-style decoder layouts.
- Spectral operator bounds and smooth, auditable layer-strength kernels.
- Full-sequence teacher-forced KL aggregation with left/right-padding correctness.
- Reproducible LFM2.5 selection, BF16 build, XSTest and paired capability scripts.
- Portable release evidence with model, comparator, dataset, report and GGUF hashes.
- BF16 Transformers and non-quantized BF16 GGUF publication with LM Studio validation.

### Changed

- Public naming now separates the Heretic NX engine, Residual-Stream algorithm profile and PRIME validation protocol.
- Experimental sweep scripts remain local; the public workflow exposes only the supported reproduction path.

### Validation

- 79 unit and integration tests pass locally.
- The LFM2.5 release matches the pinned Heretic refusal-marker count at 3/104 while approximately halving sequence KL.
- The paired 241-row MCQ slice passes the preregistered 3 percentage-point non-inferiority checks against both base and comparator.

## 0.1.0 — 2026-08-24

- Initial Heretic NX engine, PRIME research scaffold, static editors, runtime sidecars and closed-track benchmark primitives.
