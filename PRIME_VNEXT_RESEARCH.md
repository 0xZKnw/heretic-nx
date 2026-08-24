# HERETIC-NX PRIME vNext — research specification

Date: 2026-08-24

This document turns three desired claims into falsifiable engineering targets:

1. the produced model is capability-preserving outside the intended behavior change;
2. the artifact is validated under the open PRIME specification;
3. HERETIC-NX outperforms Heretic under a fair, reproducible comparison.

`PRIME-validated` is a project-defined validation label, not an accreditation by
an external standards body. Until an independent party reproduces a result, the
model card must say "self-validated under PRIME" rather than "certified".

## Research conclusions

The next engine should not hard-code one ablation algorithm. It should be a
portfolio optimizer that shares activation caches and compares several clean-room
editor families under the same constraints. The main scientific advantage over
Heretic should come from better targets, finer interventions, causal screening,
and held-out constrained optimization—not from a larger blind parameter sweep.

Three points change the current design substantially:

- Geometry is a screening signal, not a capability certificate. Principal angles
  and retained energy can reject obviously entangled edits, but only held-out
  behavioral equivalence tests can support a capability-preserving claim.
- Refusal removal is not universally one-dimensional. A global direction is a
  useful baseline, while over-refusal is task-conditioned and often higher-rank.
- Static and runtime editors are different products. Affine additions and
  per-token adaptive steering cannot always be folded into ordinary model weights.
  LM Studio/GGUF comparisons must use static-foldable editors only; runtime
  sidecars belong to a separate track.

## The three claims, defined operationally

### 1. Capability-preserving

The intended behavior axis is excluded from the preservation claim. Everything
else is tested as paired change from the exact base checkpoint.

A candidate is capability-preserving only when all of the following hold on data
that never entered site, rank, family, or strength selection:

- the simultaneous lower confidence bounds for benchmark deltas exceed
  preregistered non-inferiority margins;
- no protected domain slice violates its own margin;
- teacher-forced sequence KL remains inside a calibrated drift limit;
- format following, reasoning termination, length, calibration, uncertainty
  language, and multilingual behavior pass their dedicated gates;
- BF16 and each distributed quantization are evaluated as separate artifacts.

Failing to find a statistically significant decrease is not evidence of
preservation. PRIME must positively pass a non-inferiority or equivalence test.

Suggested protected suite for the LFM2.5 1.2B pilot:

| Dimension | Measurements |
| --- | --- |
| instruction following | IFEval strict/loose, exact JSON and constrained formats |
| reasoning | GSM8K-style exact answers, ARC-Challenge, thinking-close and EOS rates |
| knowledge/common sense | MMLU-Pro subset, HellaSwag, PIQA, Winogrande |
| code | deterministic unit-test tasks and syntax validity |
| distributional drift | teacher-forced sequence KL, top-k mass coverage, perplexity delta |
| disposition | verbosity, optimism, confidence and explicit uncertainty |
| robustness | paraphrases, templates, languages, context lengths and quantizations |

Margins must be chosen before opening the secret set, after a power/variance study
on repeated base evaluations. A single average must never hide a failed slice.

### 2. PRIME-validated

PRIME validation is an artifact-level contract, not a statement that a script
with `prime` in its filename ran successfully.

Validation levels:

| Label | Required evidence |
| --- | --- |
| `PRIME-Candidate` | clean run, enforced gates, no secret-set access, complete provenance |
| `PRIME-Validated` | locked secret evaluation, capability non-inferiority, target success, all artifact checks |
| `PRIME-Reproduced` | independent rerun reproduces the artifact or the preregistered tolerance envelope |
| `PRIME-Benchmark-Winner` | closed-track head-to-head superiority against pinned competitors |

The validation report must bind by hash:

- base model, revision and all tensor shards;
- tokenizer, processor and rendered chat template;
- HERETIC-NX commit, config and dependency lock;
- dataset revisions, group assignments and split commitments;
- editor IR and every tensor key, shape, dtype and digest;
- complete response artifact and judge/cache version;
- BF16 output and every derived GGUF or other quantized output;
- the actual accepted report, not merely a string claiming its hash.

Unknown fields, missing tensors, non-finite values, device mismatches, solver
failures, or an unapplied reject decision invalidate the candidate. Writes are
atomic and the loader fails closed.

### 3. Better than Heretic

There are two different claims and they must not be conflated:

- model claim: best measured LFM2.5-1.2B-Thinking edit among the compared runs;
- engine claim: HERETIC-NX generalizes better than Heretic across model families.

The model claim needs one pinned base. The engine claim needs at least three
architecturally distinct bases, for example a dense Transformer, LFM hybrid and
MoE model, with multiple paired seeds.

The comparison track must pin:

- Heretic v1.4.0, current stable/master commit, and its experimental ARA branch
  when that branch supports the target;
- identical base revision, tokenizer, template, prompts and generation settings;
- equal optimization trial or wall-clock budgets, reported both ways;
- identical public/secret partitions and semantic judges;
- peak VRAM, wall time and output size as secondary objectives.

The primary result is a Pareto frontier, not one hand-picked scalar score. At a
matched semantic refusal/compliance rate, compare capability drift; at matched
drift, compare semantic target success. Report paired bootstrap intervals and the
probability that HERETIC-NX's feasible hypervolume exceeds Heretic's.

Heretic's current stable implementation is a strong baseline: it uses a
multi-objective TPE search, a flexible per-component layer kernel, normalization,
and reproducible model reconstruction. Its default quality proxy is still
first-token KL and its default direction is harmful-versus-harmless
difference-of-means. A multi-token KL pull request exists but is not merged as of
this research date. The benchmark must pin commits because this baseline evolves
quickly.

## PRIME vNext engine architecture

### A. Grouped data contract

Build three distinct geometries:

- `T`: task-conditioned minimal pairs where a benign trigger and neutral wording
  require equivalent useful outputs;
- `H`: harmfulness/intent recognition directions used as protected information,
  not as the target direction;
- `C`: ordinary capabilities, formats, reasoning and language directions.

Each semantic family receives a stable `group_id`. Its paraphrases, translations,
templates and quantized variants stay in one split. Use nested grouped folds:

1. `train_geometry` for representation estimation;
2. `tune_search` for editor/rank/strength selection;
3. `public_test` for development ablations;
4. `secret_test` opened once per major protocol version.

Near-duplicate protection combines exact hashes, n-gram/MinHash retrieval and a
semantic verifier. Public benchmark paraphrases and translations are treated as
the same contamination family.

### B. Robust shared activation store

One forward collection should feed every editor family. Store centered sufficient
statistics rather than raw activations whenever possible:

- mergeable Welford mean/covariance state;
- centered Frequent Directions sketches;
- diagonal plus low-rank covariance/Fisher metrics;
- robust median/MAD or explicitly configured winsorization diagnostics;
- massive-activation masks, because a few nearly constant extreme dimensions can
  behave like indispensable implicit bias terms;
- provenance for position, template, fold, seed, dtype and backend.

LEACE and metric solves use Woodbury against diagonal-plus-low-rank factors. Dense
`d x d` matrices are reserved for small validation fixtures.

### C. Candidate discovery: screen, then intervene

Do not equate localization with editability. Use a two-stage funnel:

1. AtP*-style gradient attribution or sparse atomic-unit statistics cheaply rank
   layers, components, modes and token positions;
2. exact activation patching/interventions verify target gain and protected drift
   on held-out folds.

The causal verification result—not the localization score—enters the editor
optimizer. FFN sites are neither globally enabled nor globally forbidden: they
must demonstrate incremental held-out gain over an attention/LIV-only candidate.

### D. Three nested gates

Each proposed static site passes all three gates:

1. **Geometry gate:** bootstrap lower bounds for metric principal angle, retained
   editable energy, rank stability and cross-fold consensus.
2. **Causal gate:** held-out intervention shows positive target effect while the
   upper bound of protected drift stays inside its budget.
3. **Behavior gate:** a fitted candidate passes semantic target and capability
   constraints on `tune_search` before it may enter the public finalist set.

Thresholds are calibrated from their ability to predict held-out damage. The
current fixed 20-degree/0.20 gate remains a conservative bootstrap starting point,
not an eternal scientific constant. Every static editor must carry an explicit
`safe-static` decision; `conditional-only` sites require a runtime route; rejected
sites cannot be serialized into an accepted artifact.

### E. Editor portfolio

All families implement a common interface and declare export compatibility.

| Family | Role | Static-foldable |
| --- | --- | --- |
| normalized directional projector | Heretic-compatible baseline | yes |
| task-conditioned LEACE/ACE projection | multi-dimensional removal/disentanglement | projection yes; affine addition usually no |
| signed metric spectral editor | per-mode attenuation, erase or bounded reflection | yes |
| low-rank Cayley/orthogonal rotation | norm-preserving redistribution | yes |
| clean-room low-rank matrix optimizer | expressive ARA-like competitor with explicit constraints | yes |
| sparse atomic-unit steering | fine localization and runtime teacher | sometimes |
| adaptive affine/per-token route | maximum conditional precision | runtime track only |

For an M-orthonormal basis `Q`, the main static spectral family is

```text
T = I + Q (S - I) Q^T M,       ||S||_2 <= 1
```

where each diagonal or small block of `S` can preserve, attenuate, erase, or
reflect a mode. `beta=2` becomes the explicit `S=-I` reflection case rather than
an out-of-schema projector strength. Cross-mode 2x2 Cayley blocks allow bounded
rotations without changing norm.

Runtime-only ACE/AUSteer/ORBIT-like candidates can serve as teachers: if they
produce a superior target/capability tradeoff, distill their average conditional
effect into a static low-rank candidate and certify that new artifact separately.

### F. Constrained multi-fidelity optimization

Use an optimization funnel so expensive semantic and capability evaluations are
spent only on plausible candidates:

| Fidelity | Typical work | Purpose |
| --- | --- | --- |
| M0 | cached logits, short sequence KL, lexical/classifier screen | reject obvious failures |
| M1 | semantic judge, causal mini-suite, public capability subset | estimate feasible Pareto front |
| M2 | full public suite, multiple seeds/quantizations | select finalists |
| M3 | locked secret suite | validate once |

Discrete prefix/rank allocation uses exact dynamic programming or a small mixed
integer solve. Continuous spectral/rotation strengths use a PSD-projected QCQP
for the local quadratic model. The outer noisy multi-objective search should test
qNEHVI/constrained Bayesian optimization against Optuna TPE; retain TPE as a
simple reproducible fallback. Solver failure returns the identity edit.

Objectives:

```text
maximize  semantic target success
minimize  sequence KL, capability drift, runtime and artifact size
subject to simultaneous capability/risk UCB constraints and all hard gates
```

Candidate racing uses paired examples and common seeds. Sequential pruning may
discard a candidate early, but promotion requires every risk/capability upper
bound to be inside its preregistered limit.

### G. Semantic evaluation cascade

Keyword markers are an M0 screen only. Save full local responses and apply:

1. deterministic task checks when an exact oracle exists;
2. a calibrated refusal/compliance classifier;
3. two anonymized judges with randomized output order for semantic usefulness;
4. human adjudication for disagreement and a blinded sample of agreements.

Report judge versions, inter-rater agreement, uncertainty and rejudging history.
LLM judges have documented order, authority and presentation biases, so no single
judge can be the certificate oracle.

## Closed comparison protocol against Heretic

### Track A — LFM2.5 model claim

Run from the same pinned official BF16 checkpoint:

- untouched base;
- Heretic v1.4.0 at its documented defaults and equal-budget variants;
- Heretic master and ARA commit when compatible;
- current PRIME v2 as an explicitly non-validated historical arm;
- PRIME vNext portfolio finalists.

The old G1 Heretic-wide report cannot be reused for the new PRIME v2 SHA. Every
arm receives fresh responses under one runner.

### Track B — engine claim

Repeat on at least three model families. For every family use at least three
paired optimizer seeds, identical split groups, and both equal-trial and
equal-wall-clock budgets. Publish all Pareto points, not only the winner.

Success wording:

- after Track A only: "best measured LFM2.5-1.2B-Thinking candidate in the pinned
  comparison";
- after Track B: "HERETIC-NX produced a statistically superior feasible Pareto
  frontier to the pinned Heretic baselines across the evaluated families";
- never claim universal superiority outside evaluated models/backends.

## Implementation order

### P0 — make claims impossible to fake

1. Enforce geometry decisions before adapter construction.
2. Implement grouped nested splits and secret-set access guards.
3. Separate edit mode from scalar strength; add signed spectral IR.
4. Fix promotion UCB, NaN/device/batch handling and QCQP failure behavior.
5. Make NX-IR3 verify the complete report/artifact chain.
6. Add adversarial tests from the external audit.

Exit criterion: the current LFM PRIME v2 run is rejected automatically for the
same 28/28 gate failures already observed.

### P1 — build the strongest static candidate

1. Add centered sketches, robust activation diagnostics and low-rank metrics.
2. Add AtP*-style screen plus exact causal verification.
3. Implement the editor portfolio and export compatibility declarations.
4. Add teacher-forced sequence KL and semantic response artifacts.
5. Add the M0/M1/M2 constrained Pareto optimizer.
6. Rerun fresh LFM2.5 arms, including pinned Heretic baselines.

### P2 — validate and generalize

1. Freeze margins, secret split commitments and PRIME spec version.
2. Run the LFM closed track once.
3. Reproduce BF16 and GGUF artifact-level results.
4. Extend to dense Transformer and MoE/hybrid families.
5. Seek an independent reproduction before using `PRIME-Reproduced`.

## Primary research basis

- Arditi et al., [Refusal in Language Models Is Mediated by a Single
  Direction](https://arxiv.org/abs/2406.11717).
- Marshall et al., [Refusal in LLMs is an Affine
  Function](https://arxiv.org/abs/2411.09003).
- Maskey et al., [Over-Refusal and Representation
  Subspaces](https://arxiv.org/abs/2603.27518).
- Ravfogel et al., [Linear Adversarial Concept
  Erasure](https://arxiv.org/abs/2201.12091).
- Feng et al., [Fine-Grained Activation Steering: Steering Less, Achieving
  More](https://arxiv.org/abs/2602.04428).
- Ghasemi et al., [ORBIT: Orthogonal Subspace
  Rotation](https://arxiv.org/abs/2606.22357).
- Qiu et al., [Orthogonal Finetuning Made
  Scalable](https://arxiv.org/abs/2506.19847).
- Kramár et al., [AtP*: Scalable Localization of LLM
  Behaviour](https://arxiv.org/abs/2403.00745).
- Hase et al., [Does Localization Inform
  Editing?](https://arxiv.org/abs/2301.04213).
- Sun et al., [Massive Activations in Large Language
  Models](https://arxiv.org/abs/2402.17762).
- Daulton et al., [Noisy Expected Hypervolume Improvement for Parallel
  Multi-Objective Bayesian Optimization](https://arxiv.org/abs/2105.08195).
- Xu et al., [Selective Conformal Risk
  Control](https://arxiv.org/abs/2512.12844).
- Rottger et al., [XSTest](https://arxiv.org/abs/2308.01263).
- Cui et al., [OR-Bench](https://arxiv.org/abs/2405.20947).
- Fafuła, [Abliteration Is Not a
  Scalpel](https://arxiv.org/abs/2607.17427).
- Yang et al., [Benchmark Contamination with Rephrased
  Samples](https://arxiv.org/abs/2311.04850).
- Bouthillier et al., [Accounting for Variance in Machine Learning
  Benchmarks](https://arxiv.org/abs/2103.03098).
- Heretic upstream, [repository](https://github.com/p-e-w/heretic),
  [v1.4.0 release](https://github.com/p-e-w/heretic/releases/tag/v1.4.0), and
  [open multi-token KL work](https://github.com/p-e-w/heretic/pull/209).

These sources motivate candidates and evaluation design; they do not establish
that the proposed combination will win. Every claimed gain remains experimental
until the closed comparison protocol is run.
