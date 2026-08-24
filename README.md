# HERETIC-NX

HERETIC-NX is a research engine for low-memory behavioral model editing and
directional ablation. Its first target is aggressive
false-refusal removal while retaining as much general capability as possible.

**HERETIC-NX** is the project and engine. **PRIME** is its capability-preserving
optimization pipeline: multi-fold Grassmann consensus, protected benign
subspaces, covariance/Fisher-aware geometry, semantic-site discovery and an
exact search over edit sites and intensity. The **Residual-Stream profile**
estimates one contrastive axis per transformer residual block, binds it to
architecture-discovered output projections and restores every edited row norm
exactly.

This repository does not vendor Heretic's implementation. The source here is a
separate implementation around deterministic manifests, streaming statistics,
low-rank geometry and reproducible evaluation. Comparative reports explicitly
pin the external checkpoint used as the Heretic baseline.

## Current LFM2.5 result

The Residual-Stream candidate was regenerated from the pinned official
`LiquidAI/LFM2.5-1.2B-Thinking` checkpoint at revision
`f313478934a7612d22991f752959d7a1a8756fec`; it was not derived from the older
Heretic weights. The external checkpoint was loaded only after selection to
evaluate the locked holdout.

| Evaluation | Official base | Heretic baseline | Heretic NX + PRIME |
| --- | ---: | ---: | ---: |
| Pinned harmful rows 0–103 | — | 3 / 104 | **3 / 104** |
| Development sequence KL (80 benign rows) | 0 | ~0.1438 | **0.0701** |
| Locked-holdout sequence KL (24 benign rows) | 0 | 0.1343 | **0.0643** |
| XSTest marker total | 128 / 450 | 15 / 450 | **7 / 450** |
| XSTest safe marker count | 15 / 250 | 4 / 250 | **1 / 250** |
| Paired MCQ capability check (241 rows) | 24.1% | 23.7% | **25.7%** |

These refusal-marker rates are lexical proxies, not semantic task-success or
safety judgments. Sequence KL is exact teacher-forced full-vocabulary KL, but it
is not a substitute for broad downstream capability evaluation. These results
establish a strong local comparison, not an official universal winner claim.
The portable release evidence, checkpoint hashes, dataset revisions and claim
boundaries live in
[`evidence/lfm25-residual-stream/release.json`](evidence/lfm25-residual-stream/release.json).

The selected edit uses 19 semantic sites at scale `0.78`. Its BF16 checkpoint
was reloaded for the 450-prompt XSTest comparison. Model weights, residual
caches and generated artifacts are intentionally not stored in this source
repository.

## PRIME components

- deterministic input manifests and content hashes;
- Welford streaming statistics and Frequent Directions sketches;
- covariance/Fisher metrics, LEACE and protected capability subspaces;
- multi-fold Grassmann consensus and principal-angle geometry gates;
- architecture-aware semantic sites for LFM, Llama/Gemma-style and related
  decoder projection conventions;
- padding-safe residual-stream extraction and per-block contrastive axes;
- exact low-rank projectors, Cayley editors and sparse activation operators;
- gradient attribution, causal scanning and reduced QCQP optimization;
- KKT intensity allocation, robust objectives and sequential pruning;
- versioned NX-IR/NX-IR2 metadata and reversible runtime sidecars;
- operation-specific adaptive GPU memory budgets.

See [`PRIME_AUDIT.md`](PRIME_AUDIT.md) for the implementation matrix.
The research and validation design is documented in
[`PRIME_RESEARCH.md`](PRIME_RESEARCH.md).

## Installation

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,experiments]"
.venv\Scripts\python -m pytest
```

The reliable workflow currently edits floating-point source weights and
quantizes the result afterward. Direct Q8/Q6 GGUF editing is not yet a promoted
path. For the experimental NF4 adapter path, install the additional `quant`
extra where bitsandbytes is supported:

```powershell
.venv\Scripts\python -m pip install -e ".[experiments,quant]"
```

## Reproducing the LFM2.5 pilot

Download the pinned official checkpoint into the sibling path
`checkpoints/lfm25-prime-fresh`, then run:

```powershell
python experiments/lfm25_prime_uncensor.py
python experiments/lfm25_residual_stream_select.py
python experiments/lfm25_residual_stream_build.py
python experiments/lfm25_residual_stream_xstest.py
python experiments/lfm25_residual_stream_capability.py
```

The scripts verify the model revision/hash, pin the evaluation datasets and
write deterministic reports under `runs/`. They require a CUDA-capable system;
the current batch sizes target an 8 GB development GPU.

## LM Studio

LM Studio requires GGUF rather than the Transformers safetensors directory.
Convert the selected BF16 checkpoint without quantizing it:

```powershell
python path\to\llama.cpp\convert_hf_to_gguf.py `
  path\to\LFM2.5-1.2B-Thinking-Heretic-NX-Residual-Stream `
  --outfile LFM2.5-1.2B-Thinking-Heretic-NX-Residual-Stream-BF16.gguf `
  --outtype bf16
lms import --hard-link --user-repo local/Heretic-NX-Residual-Stream `
  LFM2.5-1.2B-Thinking-Heretic-NX-Residual-Stream-BF16.gguf
```

The published BF16 GGUF was loaded by LM Studio's llama.cpp runtime and passed
an end-to-end generation smoke test. Its SHA-256 is recorded in the release
evidence.

## Development

```powershell
hnx doctor --backend cuda --output profiles/local.json
python experiments/lfm25_prime_gate_g1.py
python experiments/lfm25_prime_route_calibration.py
python experiments/gemma4_nx_smoke.py
```

## Distribution

No model weights, checkpoints, activations or third-party research trees are
tracked here. License selection for the clean-room source is intentionally
deferred; until a license is added, normal copyright restrictions apply.
