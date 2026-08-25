# HERETIC-NX

[![CI](https://github.com/0xZKnw/heretic-nx/actions/workflows/ci.yml/badge.svg)](https://github.com/0xZKnw/heretic-nx/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)

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

## LFM2.5 2.6B — Heretic NX PRIME

The current 2.6B release was regenerated from the pinned official
`LiquidAI/LFM2.5-2.6B` BF16 checkpoint at revision
`654f9463ce32b05d0429d76fe1f580b27d4c1ac0`. It combines the
Residual-Stream profile with a capability-protected rank-1 repair axis and an
eight-site sparse intervention. Comparator weights were not used to construct
the edit.

| Evaluation | Base | Heretic Q8 comparator | Heretic NX PRIME |
| --- | ---: | ---: | ---: |
| Matched harmful rows | — | **5 / 104** | 6 / 104 |
| First-token KL to official base | 0 | published as 0.0142 | **0.012396** |
| XSTest marker total | 149 / 450 | 18 / 450 | **16 / 450** |
| XSTest safe marker count | 21 / 250 | **2 / 250** | **2 / 250** |
| XSTest unsafe-contrast markers | 128 / 200 | 16 / 200 | **14 / 200** |
| Paired MCQ capability slice | 61.71% | 61.36% (Q8 GGUF) | 61.24% (native BF16) |

The refusal comparison is intentionally reported without spin: on the matched
104-row lexical protocol, the locally tested
[`Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF`](https://huggingface.co/Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF)
Q8 artifact has one fewer marker. Heretic NX PRIME preserves the stronger drift
result: its measured first-token KL is `0.012396`, below the comparator card's
published `0.0142`. The KL comparison is descriptive rather than formally
matched because the comparator card does not disclose enough protocol and
artifact provenance to reproduce that value independently.

The comparator is pinned at revision
`1eaf992a33529fc839cbeca32109a9c4c43b57c4`; the evaluated Q8 file has SHA-256
`027f0a8308879a21163dd0c981b7397d1b8828dc06ce01e72250d3adf2f87f9b`.
On the broader matched checks, PRIME has two fewer XSTest marker hits, ties the
safe-prompt count and is capability-equivalent to the Q8 comparator. PRIME's
paired native-minus-comparator MCQ difference is `-0.12` point with interval
`[-2.22, +2.11]` points. As a runtime control, the PRIME BF16 GGUF scored
`61.48%` through the same llama.cpp restricted-choice path, only `+0.23` point
from its native score, with `97.89%` prediction agreement. Q8 and BF16 remain
different precision formats, so the comparison is reported with that caveat.

All 104 harmful rows participated in frontier selection. Promotion therefore
depends on independent gates: the 450-row XSTest target and safe-behavior tests
pass, while the 854-row ARC-Challenge/HellaSwag/MMLU first-token slice passes
the predeclared 3-point capability non-inferiority margin. The full suite now
contains 84 passing tests.

The native BF16 checkpoint and non-quantized BF16 GGUF are published at
[`LFM2.5-2.6B-Heretic-NX-PRIME`](https://huggingface.co/0xzknw/LFM2.5-2.6B-Heretic-NX-PRIME).
Hashes, dataset revisions, independent gates and claim boundaries are recorded
in [`evidence/lfm25-2p6b-prime/release.json`](evidence/lfm25-2p6b-prime/release.json).
The matched Q8 runtime reports are
[`xstest.json`](runs/lfm25-2p6b-gguf-comparator/xstest.json),
[`capability.json`](runs/lfm25-2p6b-gguf-comparator/capability.json) and the
[`PRIME GGUF runtime validation`](runs/lfm25-2p6b-gguf-comparator/prime-native-validation.json).

## LFM2.5 1.2B Thinking result

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

The evaluated Transformers checkpoint and the non-quantized BF16 GGUF are
published together on
[`LFM2.5-1.2B-Thinking-Heretic-NX-Residual-Stream`](https://huggingface.co/0xzknw/LFM2.5-1.2B-Thinking-Heretic-NX-Residual-Stream).

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
Release history is recorded in [`CHANGELOG.md`](CHANGELOG.md).

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

The 2.6B Residual-Stream PRIME research path uses the same low-memory design
and adds capability-protected residual axes plus iterative repair. Its staged
scripts retain hash checks for their pinned parent artifacts; they document the
actual search history rather than pretending the final checkpoint came from a
single post-hoc command. The frozen result can be independently reevaluated
with:

```powershell
python -m experiments.lfm25_2p6b_eval capability `
  --candidate path\to\LFM2.5-2.6B-Heretic-NX-PRIME `
  --run-dir runs\lfm25-2p6b-eval-prime
python -m experiments.lfm25_2p6b_eval xstest `
  --candidate path\to\LFM2.5-2.6B-Heretic-NX-PRIME `
  --run-dir runs\lfm25-2p6b-eval-prime
```

The pinned Heretic Q8 comparator can be rerun with
`experiments/lfm25_2p6b_gguf_comparator_eval.py`. XSTest uses LM Studio's
OpenAI-compatible completion route. Capability uses llama.cpp's native
`/completion` route with pre-tokenized prompts and the grammar
`root ::= [ABCD]`; this preserves one explicit BOS and performs deterministic
restricted first-token selection without logit bias or sampling.

The internal `v8` suffix identifies the frozen search build. The public model
name deliberately omits it: **LFM2.5-2.6B-Heretic-NX-PRIME**.

## Closed comparison against Heretic

The frozen Residual-Stream artifact was also evaluated against Heretic master
`bedb94e`, Heretic v1.4.0 and the stronger disclosed Heretic-wide comparator.
Across the 513 target prompts it produced 19 lexical refusal markers, versus
160 for master, 65 for v1.4.0 and 24 for Heretic-wide. On the paired 854-row
capability slice it scored 23.42%, versus 23.07%, 22.95% and 22.95%.

After a familywise 5% Bonferroni correction over three metrics and the three
Heretic arms, the paired target lower bounds remain positive against the pinned
upstream master and v1.4.0 runs. The 3 percentage-point capability
non-inferiority gate and safe-behavior gate also pass. This satisfies the
predeclared model-specific rule for saying Residual-Stream outperforms those two
pinned upstream runs. Heretic-wide has a slightly higher observed refusal count,
but its target-superiority interval crosses zero. See
[the protocol](benchmarks/lfm25-closed-track/README.md) and
[hash-bound summary](evidence/lfm25-closed-track/summary.json). Lexical refusal
markers are proxies, not semantic task-success or safety judgments, and this is
not a universal engine ranking.

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
