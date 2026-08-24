# Changelog

All notable public changes to Heretic NX are documented here.

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
