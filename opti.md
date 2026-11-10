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

| Geometry | Scale | Markers / 24 | Mean benign first-token KL | Decision |
| --- | ---: | ---: | ---: | --- |
| Raw | 1.25 | 3 | 0.089049 | Does not reach the zero-marker target |
| Raw | 1.50 | 0 | 0.104801 | Selected: lowest drift among zero-marker candidates |
| Raw | 2.00 | 0 | 0.142940 | Higher drift for the same lexical result |
| Protected rank 4 | 1.25 | 2 | 0.115087 | Does not reach the zero-marker target |
| Protected rank 4 | 1.50 | 1 | 0.137444 | Does not reach the zero-marker target |
| Protected rank 4 | 2.00 | 0 | 0.175858 | Higher drift for the same lexical result |

No candidate emitted an empty response. Manual inspection of all 24 selected
development continuations found one safety redirection despite the absence
of lexical markers. Zero markers therefore must not be described as zero
semantic refusals or successful task completion. The frozen selected scale
remains 1.5; the independent test is not used to tune it.

Delivered checkpoint metadata contains metrics and generation hashes only;
raw generations remain in ignored local run directories.

### Frozen wave-two validation and new KL ceiling

The scale-1.5 BF16 artifact has SHA-256
`5acd5ee0015891923a05ed814b6aecd76c56fbf2c572275458d083bd834a757f`.
Reloaded, matched 96-token testing gives 95→4 target lexical markers and 3→1
safe markers over 104 rows each, with mean first-token KL **0.100096**.
No generated test response is empty. Manual review found two clear safety
redirections and four ambiguous substitutions; three of the four lexical hits
are topic words in non-refusing answers. This is not semantic task success.

The 241-row capability check is 128→133 correct (53.11→55.19%); the 3-point
non-inferiority gate is inconclusive. An expanded 854-row check is 450→451
correct (52.69→52.81%), interval [-3.47,+3.70] points at alpha 0.025 for two
capability looks; it also fails to establish the 3-point non-inferiority gate.
A French smoke test produces coherent prose and exact JSON, but fails
37×19 (returns 5923). This cannot be attributed to editing without the matched
base arithmetic control. The artifact is retained as a reference, not declared
capability-certified.

The new required ceiling is **KL ≤ 0.05** with few refusals. The previous
104-row target test has already been viewed; future reuse is labeled a paired
retest rather than a fresh untouched holdout.

### Shared directions and sandwich coordinates — rejected screening attempts

Raw and protected axes retain strong cross-fold alignment (minimum cosine
about 0.98). Post-operator gains vary substantially (e.g. block 20 absolute
gain 0.002075–0.161133). This motivates testing a shared residual direction
and conjugating the edit by post-normalization gains and original row norms.
The algebraic coordinate mapping is checked on 32 deterministic random cases;
it does not make the nonlinear normalized block an exact residual projector.

The disposable screening uses full-vocabulary KL on the same 24 benign
development rows first (about one second per candidate), then 16-token target
generation only for KL survivors (about 5–6 seconds). Full certification still
uses 96 tokens. These screen counts cannot replace the original full reports.
All candidates and timings are retained in `global-screen.json` evidence.

Shared directions at blocks 12, 16, 20 and 22, with/without coordinate mapping,
were tested at 0.5, 1.0 and 1.5, plus local mapped axes and protected block-20
mapping. Mapping increased KL and is rejected. The best raw shared screen is
block 22 at scale 1.0: KL 0.044420, 6/24 short-window markers, still too many.
The next hypothesis is sparse edit-site routing using the original raw axes;
source weights, prompt partitions, BF16 and final evaluation settings remain
unchanged.

### Sparse routing screens and full-length check

The 36 sparse screens compare first edited block 8/12/16/20, all projections
versus FFN-only versus attention/convolution-only, and scales 1/1.5/2. The
strongest low-drift result comes from mixer-only edits, showing that FFN edits
carry substantial avoidable benign drift here. Full candidate data, rejected
portfolios and measured timing are retained in `sparse-screen.json`.

- Block 8 onward, mixers only, scale 2: development KL 0.042324 and 0/24
  **16-token** markers. At the required **96 tokens** this becomes 2/24;
  the hits are topic word `illegal` and a self-harm safety redirection.
- The same portfolio at scale 1.75: KL 0.032782, 3/24 full-length markers.
- Protected scale 1.75: KL 0.049828, 3/24 full-length markers.
- Protected scale 2: 0/24 full-length markers, but KL 0.061695; reject.

Thus the zero-marker gate correctly published no model from that full search.
The requested near-zero rather than absolute-zero objective permits a
two-marker development limit, while the KL cap stays fixed at 0.05. The next
focused screen compares first mixer block 4/6/8 and scales 1.75/2/2.25, with
all FFN weights untouched. Final reporting still uses 96-token generations;
no short-window result is promoted as the complete refusal count.

### Sparse full-checkpoint rejection

The block-4 mixer-only scale-2 candidate has SHA-256
`860e0bde8e06079fb15ca72827af36a9c7a4236362b6a6185348106510a0c83d`.
Development KL is 0.049425 with 1/24 full-length markers, but the paired
104-row retest is **KL 0.062113 and 8 markers**. The safe arm has one marker;
MCQ is 130/241 (53.94%). Both the required KL ceiling and the four-marker
target gate fail, so the checkpoint is rejected and its regenerable weights
removed after preserving the build, validation and audit metadata. All FFN
weights and all non-target tensors were bit-identical to the base.

### Conditional teacher distillation

Reuse the benign-penalized input-detector approach from the existing Gemma/LFM
experiments, with the previous 4/104 NX checkpoint as teacher. The solver was
extracted unchanged from the LFM2.5 experiment into `optimize/ridge.py` to avoid
an incidental MLX import chain; the old API delegates with its original 40-step,
1e-5 tolerance and 1e-4 ridge settings. Nine deterministic float32 parity cases
(dimensions 4/16/32 and penalties 1/10/100) match the pinned original solver
bit for bit, including iteration counts and residuals. Independent small
dense-solve tests validate the numerical objective; this is not a formal proof.

Collect 1,024 benign prompt-boundary inputs from the hash-partitioned training
geometry set and teacher trajectories for 128 harmful geometry prompts. The
trajectory collector runs 32 new tokens and samples every fourth forward,
excluding finished rows. Both caches contain 1,024 BF16 states per projection.
Search and test prompts are not used to fit these detectors.

For each of 40 sites, approximate the teacher output-delta field with a fixed
rank-one output axis (randomized PCA, q=5, two iterations, seed 1703+site), then
fit the input detector against the target field with a benign covariance
penalty. Merge the resulting rank-one additive delta into the original weights.
This profile deliberately does not restore row norms; norm drift and full-model
behavior must be measured instead of inheriting the residual-profile claim.

The solver is a 40-iteration approximation: none of the 120 production fits
reaches its 1e-5 stopping tolerance. Maximum relative residual is 0.0683,
0.3360 and 3.8287 for penalties 1, 10 and 100. These are disclosed incomplete
numerical solves, not convergence certificates. Median target-fit errors are
0.0667, 0.1609 and 0.3703 respectively. Candidate promotion depends on actual
model evaluation, not an assumed optimum of these fits.

An initial cache write failed because penalty trials shared the same axis
storage. Independent serialized storage fixes this without changing tensor
values; a regression test reproduces the original safetensors rejection and
checks the roundtrip. Input caches were retained, so no GPU collection was
repeated. Partial 16-token refusal screens cannot qualify as full results.

| Benign penalty | Delta scale | Development KL | Full markers / 24 |
| --- | ---: | ---: | ---: |
| 1 | 1 | 0.014805 | 2 |
| 1 | 1.5 | 0.042346 | 0 |
| 10 | 1.5 | 0.026208 | 0 |
| 100 | 1.5 | 0.020696 | 1 |

The other five trials fail the KL cap or a monotone short-prefix marker bound;
their counts are recorded as incomplete, not invented full scores. Two frozen
frontier points are tested at the full 96-token/104-row protocol: minimum
development KL (penalty 1, scale 1) and zero development markers (penalty 10,
scale 1.5). The former gets KL **0.020613 but 12/104 markers**, failing the
target gate. The second paired retest is in progress.
