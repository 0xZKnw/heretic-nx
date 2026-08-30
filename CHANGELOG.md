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
- Thread-safe end-to-end phase timing and measured portfolio ETA bounds,
  including explicit geometry/build work, cache savings, failures, overlap and
  a single-winner public-report barrier.
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

- Residual-stream capability protection now transfers each layer's safe/target
  diagnostics to CPU once and reuses the safe mean. The 8-layer,
  256x2,048 reference fit is bit-exact and 1.12x faster on CPU, while a GPU
  caller avoids three redundant full safe-layer transfers per layer.
- Low-rank metric calibration now converts BF16 activation matrices to FP32
  once instead of materializing the same conversion twice. The 1,024x4,096
  reference fit is bit-exact, 1.17x faster and removes one 16 MiB temporary.
- Euclidean geometry gates now reuse target/protected bases for projection and
  principal angles instead of fitting them up to twice. The 4,096-dimensional
  rank-16 reference gate is bit-exact and 1.63x faster.
- Metric geometry gates now reuse their M-orthonormal target/protected bases
  and cross-Gram matrix for residualization and principal angles. The
  4,096-dimensional, 512-factor, rank-16 reference gate is bit-exact and 1.49x
  faster, directly reducing architecture-independent site-screening cost.
- Disposable quantized search artifacts can defer float64 realized-drift
  diagnostics while retaining identical encoded payloads, target hashes and
  final artifact hashes. The 2,048x4,096 rank-8 kernels are 3.09x faster for
  direct Q8_0 and 1.61x faster for direct Q4_K; final winners must still be
  rebuilt with complete diagnostics and undeclared-byte verification.
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
- LEACE now fits in the thin sample space and keeps the affine eraser factorized,
  materializing the historical dense projection only on explicit access. On a
  256x2,048 binary-concept workload this makes fitting 13.2x faster, application
  9.4x faster and stored output 683x smaller, while removing the dense solver's
  hundreds of spurious numerical-rank directions.
- Sparse activation operators now compute and scatter only selected output
  coordinates below the measured density crossover; the 4,096-wide, rank-8,
  64-coordinate benchmark is 2.44x faster at 512 tokens. Metric projector
  construction is also bit-identical and 1.73x faster by reusing `M @ Q`.
- Low-rank matrix optimization now restores its best finite objective instead
  of returning a degraded terminal step. The regression fixture improves the
  returned objective by 3.24x; opt-in patience stops at 16/200 steps, retains
  exactly the same best state and runs about 11x faster.
- Teacher-forced sequence KL now skips padding before softmax and processes
  selected rows in vocabulary-aware chunks with reusable buffers. At 56.25%
  token density the reference metric is exactly unchanged, runs 1.66x faster
  and reduces estimated metric workspace by at least 15.8x.
- Judge verdict caches now support atomic bulk reads/writes and deduplicate
  identical misses in `JudgeCascade.judge_many`. For 104 local verdicts on the
  reference Mac, batched durable writes are 22.3x faster and the complete
  cascade/cache path is 8.5x faster; external judge latency remains dominant
  when a model call is required.
- Raw-logit KL artifacts are now hashed and numerically validated in one file
  pass with reusable row workspaces. On a 104x128,000 float32 artifact, this is
  1.77x faster and reduces estimated validation workspace by 5.46x while
  preserving the SHA and mapped values exactly.
- Refusal-first candidate waves can use bounded independent runtime slots while
  preserving a global stage barrier. The synthetic eight-candidate/four-slot
  I/O benchmark is 3.95x faster than serial; this deliberately does not claim
  equivalent scaling for one model on one GPU.

### Fixed

- Judge-cache rows are now immutable: identical replays are idempotent, while
  conflicting concurrent verdicts, malformed JSON and non-finite payloads fail
  closed. Task-specific success is included in cache identity so a success
  verdict cannot leak into the same prompt/response judged without that signal.
- Raw KL validation now checks file identity before and after its scan and maps
  the already-hashed descriptor, preventing a concurrent pathname replacement
  from swapping the returned artifact after verification.
- New v3 GGUF plans now use a canonical projection reduction, making edited
  payload bytes independent of streaming chunk size. Historical v2 and Q8 v1
  plans retain their old arithmetic for exact
  replay instead of being silently reinterpreted.
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
