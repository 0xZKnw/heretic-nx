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

## LFM2.5 8B-A1B — conditional direct-Q8 release

The 8B-A1B release introduces conditional direct-Q8 deltas: an eight-site
operator teacher is distilled against 1,024 harmless states and 2,627 harmful
response-trajectory states, then merged directly into the deployment GGUF.
The selected `lambda=100`, `beta=2.25` Q8 has **4/104** lexical refusal markers
and mean full-vocabulary first-token `KL(base Q8 || candidate Q8) = 0.016948`
over 104 benign rows. The original Q8 has 95/104 markers under the same harmful
protocol.

The 9.34 GB evaluated artifact and its exact release manifest are published at
[`LFM2.5-8B-A1B-Heretic-NX-PRIME-GGUF`](https://huggingface.co/0xzknw/LFM2.5-8B-A1B-Heretic-NX-PRIME-GGUF).
All 104 harmful rows participated in development; lexical markers are a proxy,
not semantic task success or an untouched holdout.

On the post-selection 854-row ARC-Challenge/HellaSwag/MMLU capability check,
the original and Heretic Q8 artifacts both score **55.04%**. Heretic minus base
is `0.00` point with a paired 95% interval of `[-1.29, +1.29]` points, passing
the predeclared 3-point non-inferiority and symmetric equivalence gates. This
supports capability preservation on the measured slice; it does not establish
an aggregate capability increase.

## LFM2.5 2.6B — Heretic NX PRIME

The current 2.6B release was regenerated from the pinned official
`LiquidAI/LFM2.5-2.6B` BF16 checkpoint at revision
`654f9463ce32b05d0429d76fe1f580b27d4c1ac0`. It combines the
Residual-Stream profile with a capability-protected rank-1 repair axis and an
eight-site sparse intervention. Comparator weights were not used to construct
the edit.

| Evaluation | Base | Heretic Q8 comparator | PRIME native BF16 | PRIME Q8_0 |
| --- | ---: | ---: | ---: | ---: |
| Matched harmful rows | — | **4 / 104** | 6 / 104 | 9 / 104 |
| First-token KL to official base | 0 | published as 0.0142 | **0.012396** | not remeasured |
| XSTest marker total | 149 / 450 | **8 / 450** | 16 / 450 | 9 / 450 |
| XSTest safe marker count | 21 / 250 | **2 / 250** | **2 / 250** | 3 / 250 |
| XSTest unsafe-contrast markers | 128 / 200 | **6 / 200** | 14 / 200 | **6 / 200** |
| Paired MCQ capability slice | **61.71%** | 60.66% | 61.24% | 60.66% |

The comparison is intentionally reported without spin. The pinned
[`Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF`](https://huggingface.co/Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF)
Q8 has fewer lexical markers on the 104-row optimization set. On the broader
independent XSTest slice, the two Q8 artifacts differ by one marker out of 450
and tie on the 200 unsafe-contrast rows. PRIME's measured native-BF16
first-token KL is `0.012396`, below the comparator card's published `0.0142`.
That KL comparison is descriptive rather than formally matched because the
comparator card does not disclose enough protocol and artifact provenance to
reproduce its value independently; post-quantization KL was not measured.

The comparator is pinned at revision
`1eaf992a33529fc839cbeca32109a9c4c43b57c4`; the evaluated file has SHA-256
`027f0a8308879a21163dd0c981b7397d1b8828dc06ce01e72250d3adf2f87f9b`.
Both Q8 XSTest arms used official llama.cpp b10621, native pre-tokenized
`/completion`, one explicit BOS, greedy decoding and identical prompts. The
native-BF16 capability comparison remains paired and equivalent: PRIME minus
the comparator is `+0.59` point with interval `[-1.52, +2.81]` points. The
PRIME BF16 GGUF scored `61.48%`, only `+0.23` point from its native score, with
`97.89%` prediction agreement. Precision and runtime formats are labeled
because they are not interchangeable.

All 104 harmful rows participated in frontier selection. Promotion therefore
depends on independent gates: the 450-row XSTest target and safe-behavior tests
pass against the official base, while the 854-row
ARC-Challenge/HellaSwag/MMLU first-token slice passes the predeclared 3-point
capability non-inferiority margin. The full test suite is executed in CI.

The native BF16 checkpoint plus BF16 and Q8_0 GGUF variants are published at
[`LFM2.5-2.6B-Heretic-NX-PRIME`](https://huggingface.co/0xzknw/LFM2.5-2.6B-Heretic-NX-PRIME).
Hashes, dataset revisions, independent gates and claim boundaries are recorded
in [`evidence/lfm25-2p6b-prime/release.json`](evidence/lfm25-2p6b-prime/release.json).
The matched Q8 runtime reports are
[`xstest.json`](runs/lfm25-2p6b-gguf-comparator/xstest.json),
[`capability-b10621.json`](runs/lfm25-2p6b-gguf-comparator/capability-b10621.json) and the
[`PRIME GGUF runtime validation`](runs/lfm25-2p6b-gguf-comparator/prime-native-validation.json).

The lighter Q8_0 artifact was quantized only after the BF16 edit was frozen.
Its official b10621 comparison is:

| Q8 runtime artifact | Size | XSTest total | Safe | Unsafe contrast | MCQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heretic comparator Q8_0 | 2.87 GB | **8 / 450** | **2 / 250** | **6 / 200** | **60.66%** |
| PRIME Q8_0 | **2.87 GB** | 9 / 450 | 3 / 250 | **6 / 200** | **60.66%** |

As a separate same-runtime LM Studio control, PRIME BF16 scored 12/450 and
PRIME Q8 scored 10/450. On official b10621 the two Q8 capability scores tie;
the paired PRIME-minus-comparator interval is `[-1.87, +1.76]` points. PRIME
Q8 is `-0.59` point versus native BF16 with `96.37%` prediction agreement.
These are lexical-refusal and narrow MCQ checks, not a universal quality
guarantee. Hash-bound Q8 reports live under
[`runs/lfm25-2p6b-prime-q8`](runs/lfm25-2p6b-prime-q8).

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

The floating-point workflow remains available, but Heretic NX can also merge a
precomputed low-rank ablation directly into mixed-quantization GGUF files. The
same-type backend supports `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q4_0`,
`Q4_1`, `Q5_0`, `Q5_1` and `Q8_0`. It streams matrices and stacked MoE banks in
fixed row chunks, keeps non-target bytes unchanged and never materializes a
full BF16 checkpoint. Install the GGUF dependency with:

```powershell
.venv\Scripts\python -m pip install -e ".[gguf]"
```

K-quant encoding uses llama.cpp's native `libggml-base`; pass its path with
`--ggml-library` or set `HERETIC_NX_GGML_LIBRARY`. Heretic NX also discovers a
local llama.cpp build or a library next to `llama-quantize`. Native encoding is
row-parallel above 65,536 elements and automatically uses up to eight
affinity-aware CPU workers. Set `HERETIC_NX_QUANT_THREADS=1` for serial
execution, `auto` for the default, or a positive integer to tune the worker
count. First inspect the actual tensor types:

```powershell
hnx inspect-gguf --input model-Q4_K_M.gguf --output quantized-tensors.json
```

`Q4_K_M`, `Q4_K_S`, `Q5_K_M` and related names are file recipes, not tensor
types. A single model can contain several K types plus Q8 and float tensors, so
each plan entry declares the exact expected type and fails closed on a mismatch.
There is no official `Q6_K_S` tensor type in llama.cpp: the corresponding
weight format is `Q6_K`, which is supported.
The plan binds both the source GGUF and its safetensors factors by SHA-256.

```python
from heretic_nx.edits import GGUFQuantizedAblationPlan, GGUFQuantizedTensorEdit
from heretic_nx.hashing import sha256_file

GGUFQuantizedAblationPlan(
    source_sha256=sha256_file("model-Q4_K_M.gguf"),
    tensor_artifact_sha256=sha256_file("axes.safetensors"),
    row_chunk_size=128,
    edits=(
        GGUFQuantizedTensorEdit(
            tensor_name="blk.2.attn_output.weight",
            expected_quantization="Q4_K",
            a_key="residual_axis.L02",
            strength=0.8,
            quantization_multipliers=(0.75, 1.0, 1.25),
            minimum_delta_cosine=0.8,
        ),
    ),
).write("quantized-plan.json")
```

Validate dimensions and hashes without copying the model, then perform the
atomic static merge:

```powershell
hnx abliterate-gguf --input model-Q4_K_M.gguf --plan quantized-plan.json `
  --tensors axes.safetensors --dry-run
hnx abliterate-gguf --input model-Q4_K_M.gguf `
  --output model-Heretic-NX-Q4_K_M.gguf `
  --plan quantized-plan.json --tensors axes.safetensors
```

The default min-drift policy compares the original encoded block with each
quantized strength candidate and keeps the representation closest to the
intended float edit. Reports include realized-delta cosine/error, row-norm
drift, changed blocks, target-approximation error, codec provenance and a
tracked-array memory lower bound. Output is reopened, target layouts are
checked, and by default all undeclared byte ranges are content-verified against
the immutable pre-edit snapshot before no-clobber atomic publication. Disabling
`verify_untouched_bytes` is intended only for disposable search candidates.

The legacy `inspect-q8` and `abliterate-q8` commands remain available for v1
Q8 plans; legacy merges are routed through the same hardened engine. Every
final quantization must be evaluated independently against an
untouched baseline of the same recipe: compare base Q4 to edited Q4, never base
Q8 to edited Q4.

For large candidate frontiers, run the refusal screen before KL. The Ling
runner accepts `--refusal-cap 6` and stops irreversibly failed candidates as
soon as the seventh marker is observed; such partial reports are explicitly
non-certifying. Re-run survivors without the cap for the complete 104-row
report, then compute KL only for those full-report survivors.

KL collection now takes the GGUF path with `--artifact`, attests the running
llama.cpp server through `/props`, hashes the local artifact before and after
collection, and records the runtime build/type in each checkpoint. Comparisons
reject identical base/candidate bytes, mismatched runtime builds, different
quantization types, incomplete matrices and unnormalized rows.

The optimizer/backend microbenchmark is reproducible with:

```powershell
python benchmarks/backend_microbench.py
python benchmarks/gguf_codec_parallel.py
python benchmarks/spectral_compact.py
```

It compares the former dense `d x d` regularization path with the rank-space
implementation and times native Q4_K/Q6_K codecs. The dedicated codec benchmark
checks bit identity and reports Q2_K through Q6_K scaling across worker counts.
The spectral benchmark checks the exact low-rank eigensolver against the former
ambient `d x d` decomposition. These speedups are component measurements, not
claims that complete model evaluation is equally faster.

For the experimental NF4 adapter path, install the additional `quant` extra
where bitsandbytes is supported:

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
`experiments/lfm25_2p6b_gguf_comparator_eval.py`. XSTest and capability use
llama.cpp's native `/completion` route with pre-tokenized prompts and exactly
one explicit BOS. Capability additionally uses the grammar `root ::= [ABCD]`
for deterministic restricted first-token selection without logit bias or
sampling. The released Q8 is checked with
`experiments/lfm25_2p6b_prime_gguf_xstest_validate.py`.

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
