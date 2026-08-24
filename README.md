# HERETIC-NX

HERETIC-NX is a clean-room research engine for low-memory behavioral model
editing and directional ablation. Its first target is aggressive
false-refusal removal while retaining as much general capability as possible.

**HERETIC-NX** is the project and engine. **PRIME** is its capability-preserving
optimization pipeline: multi-fold Grassmann consensus, protected benign
subspaces, covariance/Fisher-aware geometry, semantic-site discovery and an
exact search over edit sites and intensity. **PRIME v2** is the aggressive
overprojection profile used for the current LFM2.5 pilot.

This repository does not copy Heretic's AGPL implementation. The mathematical
core was independently implemented around deterministic manifests, streaming
statistics, low-rank geometry and reproducible evaluation.

## Current LFM2.5 result

The latest candidate was regenerated from the pinned official
`LiquidAI/LFM2.5-1.2B-Thinking` checkpoint at revision
`f313478934a7612d22991f752959d7a1a8756fec`; it was not derived from the older
Heretic output.

| Evaluation | Official base | PRIME v2 |
| --- | ---: | ---: |
| Refusal-marker benchmark | 98 / 100 | **3 / 100** |
| XSTest marker total | 131 / 450 | **14 / 450** |
| Benign capability smoke | 2 / 4 | **3 / 4** |
| First-token benign KL | — | 0.0018575 |

These refusal-marker rates are lexical proxies, not semantic task-success or
safety judgments. The four-item capability smoke is deliberately small and is
reported as preliminary evidence, not a universal quality claim. Full reports
and hashes live under `runs/lfm25-prime-uncensor-v2/` and
`runs/lfm25-xstest-retest-prime-v2/`.

The selected edit uses 24 semantic sites at `beta=2.0`. The saved BF16 candidate
reloaded with the same score, and its BF16 GGUF was validated in LM Studio at
roughly 75–78 tokens/s on the development machine. Model weights and generated
artifacts are intentionally not stored in this source repository.

## PRIME components

- deterministic input manifests and content hashes;
- Welford streaming statistics and Frequent Directions sketches;
- covariance/Fisher metrics, LEACE and protected capability subspaces;
- multi-fold Grassmann consensus and principal-angle geometry gates;
- architecture-aware semantic sites for LFM2.5;
- exact low-rank projectors, Cayley editors and sparse activation operators;
- gradient attribution, causal scanning and reduced QCQP optimization;
- KKT intensity allocation, robust objectives and sequential pruning;
- versioned NX-IR/NX-IR2 metadata and reversible runtime sidecars;
- operation-specific adaptive GPU memory budgets.

See [`PRIME_AUDIT.md`](PRIME_AUDIT.md) for the implementation matrix.

## Installation

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,experiments]"
.venv\Scripts\python -m pytest
```

For the experimental NF4 path, install the additional `quant` extra where
bitsandbytes is supported:

```powershell
.venv\Scripts\python -m pip install -e ".[experiments,quant]"
```

## Reproducing the LFM2.5 pilot

Download the pinned official checkpoint into the sibling path
`checkpoints/lfm25-prime-fresh`, then run:

```powershell
python experiments/lfm25_prime_uncensor.py
python experiments/lfm25_xstest_retest.py
```

The scripts verify the model revision/hash, pin the evaluation datasets and
write deterministic reports under `runs/`. They require a CUDA-capable system;
the current batch sizes target an 8 GB development GPU.

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
