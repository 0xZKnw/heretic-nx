# HERETIC-NX PRIME implementation audit

Source reviewed in full: `HERETIC-NX-PRIME_rapport_recherche_final.pdf`, 25 pages,
including visual rendering of every page.

## Implemented

- architecture-aware LFM2.5 semantic-site registry and fail-closed 10 LIV / 6 GQA guard;
- frozen PRIME manifest schema and golden tensor fingerprints;
- Welford, Frequent Directions and mergeable paired cross-covariance;
- chordal Grassmann consensus with stability spectrum and learned rank;
- covariance/Fisher low-rank metric, M-orthogonal residualization and metric gate;
- closed-form LEACE baseline and linear leakage diagnostic;
- activation-native metric projector, input/output merge factors, sparse Atomic Units;
- NX-IR 2 sidecar schema, temporal gate and bounded PID research controller;
- fail-closed temporal logits controller, multi-site risk consensus and verified
  temporal-only NX-IR 2 loader;
- gradient AttrScan, exact central reliability check and top-K selection;
- Rademacher reduced Hessian, QCQP with separate H/C constraints, worst-case/CVaR;
- BSR/BRR/deflection metrics, J0-J3 judge cascade and content-addressed cache;
- anytime-valid confidence sequences and sequential promotion/pruning;
- per-operation batch/token budgets and CUDA soft/hard cap calculation;
- verified benign pairs, fail-closed oracle consensus, anti-leak splits and safe delta debugging.

## Measured gates (2026-08-24)

### G0 — pass

- 44 deterministic tests pass on the mathematical core and runtime contracts;
- the AttrScan mini-transformer test recovers the exact top-K intervention sites;
- the temporal controller is inert unless both risk and task gates pass, closes
  inside the configured 96+8 window, and draft sidecars fail closed at load time.

### G1 — existing merged Heretic rejected

The published merged BF16 checkpoint was compared with the pinned LiquidAI base
on CUDA BF16. This is a deliberately small pilot, so it can reject but cannot
certify broad quality.

- deterministic benign task success: base `2/4`, merged Heretic `1/4`;
- controlled thinking: raw runs did not close by 320 tokens; the runtime sidecar
  closed at token `104` and then completed the checked process-management task;
- all 4 harmful-response outputs remained J3-required under the conservative
  judge cascade, so no risk claim was inferred from lexical markers;
- 66/96 semantic site-position readings rejected the realized merged displacement
  under the metric H+C gate; maximum measured benign relative drift was `0.3891`;
- minimum measurable retention of harmfulness separation was `0.7496`.

Canonical report: `runs/lfm25-prime-g1/report.json` (`promoted=false`).

### Temporal route calibration — draft only

The base checkpoint was calibrated on all 450 pinned XSTest-v2 prompts with
grouped anti-leak train/public/secret splits. A public-perfect OR consensus over
GQA block L12 and LIV block L13 was selected without tuning on the secret split.

- public harmful controls: `40/40` blocked; benign routed: `17/47`;
- secret harmful controls: `39/39` blocked; benign routed: `17/49`;
- the 95% zero-miss upper error bound is still `0.07394`, so the absolute
  zero-tolerance risk gate is not statistically certified;
- `temporal-route-draft.nx-ir2.json` has no accepted-report hash and is therefore
  refused by the runtime unless research-only `allow_unaccepted=True` is explicit.

Canonical report: `runs/lfm25-prime-route/report.json`.

### GGUF runtime — usable, not PRIME-certified

The already-converted BF16 GGUF is not blocked. It was loaded directly by LM
Studio's CUDA llama.cpp runtime with full GPU offload and produced the expected
final answer with EOS:

- load time: `3.71 s`, loaded allocation: `2.18 GiB`;
- deterministic arithmetic final: `437`;
- throughput: `76.66 tok/s`, time-to-first-token: `0.317 s`;
- installed LM Studio file is bit-identical to the source GGUF, SHA-256
  `D913B88F41696A5FF8DB3B9C8D3768B83CB870B05BC8051CAE0491C454202654`.

This establishes GGUF/LM Studio compatibility only. It does not override the G1
rejection of the merged behavioral edit. Canonical smoke report:
`runs/lfm25-gguf-smoke/report.json`.

## Hardware-gated

- MLX 4/8-bit execution requires the user's M5 Mac and cannot be certified on Windows;
- GGUF runtime compatibility is smoke-tested; PRIME cross-backend behavioral
  parity and MLX promotion still start only after CUDA Gate G1 promotes a candidate;
- human J3 adjudication is intentionally external and cannot be replaced by an uncalibrated local judge.

## Explicitly experimental

- manifold-aware correction, causal-nullspace generalized eigenvectors and MPC remain
  ablations, matching the report's instruction not to put them on the PRIME-Core path
  before Gate G2 evidence.
