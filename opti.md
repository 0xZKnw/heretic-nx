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
target gate. The second paired retest passes both proxy targets: **4/104 markers** (base
95), **KL 0.04501414299 ≤ 0.05**, and 2/104 safe markers (base 3), with no empty
responses. It is selected and saved as the final BF16 checkpoint, SHA-256
`875e4722c380b7b8b916fdc97c83ed1dd518db5df6ad12d9c029b18d5c6d5e40`.
The reused 104-row test participated in frontier selection, so this result is
not an untouched holdout; fitting still uses separate training geometry.

The 241-row MCQ result matches the base at 128/241 (53.11%). An expanded,
post-selection 854-row check gets 446/854 (52.22%) versus base 450/854 (52.69%),
difference -0.47 point, paired interval [-3.12, +2.19] points at alpha 0.025
(two capability looks). Three-point noninferiority is still inconclusive.
French prose and exact JSON smoke checks pass. Multiplication 37×19 fails in
both original and edited BF16 checkpoints; do not attribute all arithmetic
failure to the edit. Exactly 40 tensors change, shapes/dtypes remain intact,
and all other tensors are bit-identical. Conditional merging deliberately
leaves row norms unrestored; maximum relative row-norm drift is 5.81%.

### Native Q8_0 release validation

Convert the exact selected BF16 checkpoint and the official base independently
with llama.cpp b12088-d1044bc82, then quantize both with the native Q8_0 recipe
and four CPU threads. Final Q8 SHA-256 is
`5c979a09b41acb8e225a9f66c7c006ef732cf84e5571dd87a855d30048eb187a`
(1,846,237,824 bytes). Both BF16 GGUF conversion intermediates are removed
after hashing, retaining the actual release and base Q8 evaluation artifact.

Use the same one-BOS HF prompt token IDs, thinking disabled, greedy 96-token
native completions, context 4096, one slot, full GPU offload and F16 KV cache.
Attest model path/hash and runtime before and after each sequential arm. All
128,000 first-token IDs are required for KL; finite native softmax values are
normalized in float64. MCQ uses a greedy single-letter grammar, checked against
the full-logit A/B/C/D argmax on the first three rows per arm. No concurrent
GPU workloads. Initial driver assertions were corrected for the server's
reported greedy temperature of 0 when requesting -1; aborted probes produced
no promoted scores. The Q8 checkpoint passes the same proxy objectives:
**3/104 target markers** versus matched original Q8 **96/104**, safe markers
**1/104 versus 3/104**, no empty responses, first-token **KL 0.04633291462**.
The 287-tensor native audit finds exactly 40 intended changed projections;
all other tensor bytes, shapes and quantization types match the original Q8.
Expanded native MCQ scores match at 438/854 (51.29%) per arm. The paired
alpha-0.025 interval is [-3.18, +3.18] points; three-point noninferiority is
inconclusive despite equal point estimates. The native probability parser
passes a permuted-uniform roundtrip and rejects missing, duplicate,
nonfinite and unnormalized entries. These are sampled checks, not proofs.
Only compact reports and hashes are released; raw generations remain local.


### EXL3-2026-11-10 — Heretic NX conversion using the published MLXL3 recipe

Requested continuation: convert the published final Heretic NX checkpoint for
MLXL3, using one Modal H100 and the owner's already published recipe. Source
`0xzknw/LFM3-1.7B-Heretic-NX@56a40b05ac6affe579c9aa3c7def410c24f75b5c`,
weight SHA-256 `875e4722c380b7b8b916fdc97c83ed1dd518db5df6ad12d9c029b18d5c6d5e40`.
Recipe reference `0xzknw/LFM3-1.7B-EXL3@794219432967af4826d8276d370b2d1676d1f870`,
4.4892703863 projection bpw: 61 projections at 4 bits, 24 FFN down at 5,
57 attention/conv outputs at 6; embeddings/head at 16 bits, MCG, automatic
output scales, scaled Hessians, calibration 250 x 2048 bundled tokens. Reuse
ExLlamaV3 1.6.6/bd1f519 wheel and existing bounded conversion helpers.

Preflight checks source hash, model config and all 142 recipe projection
shapes on CPU before requesting the H100. One GPU worker, retries disabled,
conversion timeout 1800s, overall worker timeout 3600s. No speed benchmark.
Check descriptor allocation, all expected tensors and finite payloads; compare
first-token full-vocabulary logits on the same 104 safe prompts with original
and edited BF16 references. Re-measure target markers where the actual runtime
allows the same greedy 96-token protocol. Validate local MLXL3 Metal loading,
nonempty finite logits, reference parity and a real generation. Publish a
separate EXL3 directory/repo only with explicit format-specific reports;
existing Safetensors/GGUF weights remain unchanged. Status: preparation.

Local disk reached 112 MiB free during EXL3 preparation. Removed only
regenerable task intermediates: the matched original Q8 GGUF (hash retained
in gguf-build.json), the verified pinned source weight download (source code
and config retained), and fitted-detector activation caches (hashes retained
in build.json; fitted factors and final checkpoints preserved). No benchmark
or quantization was restarted. The first CrossHair invocation exposed the
helper's documented ValueError for invalid bitrate 0; invalid-bit rejection
is already tested. Repeat symbolic exploration over the actual valid 4/8-bit
helper domain; the initial invalid-domain diagnostic is retained.

The mixed 4.49-bpw recipe is the first requested candidate. If its measured
combined KL exceeds 0.05 or the target marker cap, retain the rejected report
and test the existing published 8-bpw recipe as a higher-fidelity fallback,
without relaxing either quality criterion. Do not silently present BF16 or
GGUF metrics as EXL3 measurements. This fallback is preregistered before the
first EXL3 quality results are available.

Mixed 4.49 bpw conversion completed on H100. All 142 projection allocations
and 713 output tensors pass structure/finiteness checks, and the 104-row
marker counts are target 3 / safe 1 with no empty outputs. However, measured
first-token KL is 0.21471471835 to original HF FP32 and 0.15222477184 to
Heretic HF FP32: rejected for the 0.05 drift ceiling. Reports and weights
retained privately. Before converting the published 8-bpw fallback, compare
unquantized ExLlama to the same Heretic HF FP32 reference on all 104 safe
rows; require KL <= 0.005 to exclude an engine incompatibility. Reuse those
reference arrays in the fallback measurement. Uniform 8 bits, head/embedding
16, MCG, auto scales, 250x2048 calibration, no scaled Hessian, matching the
existing published 8-bpw allocation. No simultaneous H100 workers.

Owner steering: drop 4.49 bpw and compare the existing strict 4/8-bpw
recipes instead. Mixed checkpoint will not be published. A local Modal CLI
file download had the right byte count but a mismatched SHA-256; integrity
check rejected it before any MLXL3 inference, and the unneeded local mixed
copy was removed. Reports and expected/actual hashes retained in task logs.
Use streaming Modal SDK reads with checksum verification for selected 4/8
weights. Unquantized ExLlama/HF FP32 control passed at KL 0.00004372903205
over 104 safe rows. Strict 8-bpw conversion is running. After it completes,
compare strict 4 bpw with its published unscaled-Hessian/MCG/auto-scales
recipe and unchanged 250x2048 calibration, head/embedding 16. Retain all
quality criteria, and publish only passing formats.

User revised only the strict 4-bpw KL budget during conversion, before any
4-bpw quality result: allow +0.05 nats relative to the measured 8-bpw KL
(0.043624878533), giving cap **0.093624878533**.
The 8-bpw cap remains 0.05; all other checks stay unchanged. Frozen budget
record: work/exl3-conversion/quality-budgets.json. The running remote worker
still records its original 0.05 gate; retain that raw report and apply the
newly authorized budget explicitly in the final acceptance report, without
rerunning or changing the measured logits/generations.

Strict 8 bpw completed on H100: target markers 3/104, safe markers 2/104,
no empty responses; original-FP32-to-EXL3 first-token KL
0.043624878533, NX-FP32-to-EXL3 KL
0.001415807885. All 142 projections and 713 tensors
pass integrity/finiteness checks. Streaming SDK download passes every file
SHA-256. Weight file 2,518,073,115 bytes, SHA-256
8a7c4200d8c4cb5a994ceb7107e53f7b6464be1e8c8f697dc9ac961a2b3489c5.

Published production MLXL3 engine v2.0.4-beta.1/ead60d0323e9 validates all
104 prompt logits: CUDA-EXL3-to-MLXL3 mean KL 0.00008576118, argmax
agreement 99.04%, finite 128,000-value outputs and bit-exact reset replay.
The 499-token SWA check passes with runtime KL 0.00026417794. Two-turn
production NDJSON bridge chat passes independent CEDAR-42 recall. No speed
benchmark was run. Reusing the unchanged original FP32 reference from the
4-bit job gives MLXL3 first-token model KL **0.043899179649 <= 0.05**; CUDA
recomputation matches the original 8-bit result within 1e-6.

Strict 4 bpw fails the revised cap: measured combined first-token KL
0.464836920468 > 0.093624878533; quantization
drift versus NX FP32 0.353066835600. A stop request
after the decisive KL failure was rejected by the CLI's interactive prompt
(no --yes flag); the job naturally finished before follow-up. Final 4-bit
counts are complete: target 4/104, safe 0/104, no empties. Preserve both the
pre-stop snapshot and full raw report; reject on KL and do not publish its
weights. All three Modal apps are now confirmed stopped with zero tasks.
Only strict 8 bpw is accepted for publication.

EXL3 8-bpw initial publication verified publicly at
0xzknw/LFM3-1.7B-Heretic-NX-EXL3@4ed8d37eb05d7373624f78064b4b20bb74148b83,
weight SHA and size match the tested artifact, all files present, selected
metadata downloaded byte-exact. Production Hub details discovers 8bpw but
its preview marks lfm3 unsupported, exactly as for every variant of the
owner's original LFM3-1.7B-EXL3 repo. Model metadata is correct; native
loading/104 forwards/long/reset/chat all pass. Document the verified
hf-download/local-folder path and this existing preview limitation rather
than spoofing the architecture or changing the unrelated runtime.

EXL3 final metadata publication verified at a7cff0c15b6b1ab26ea77e29bd7a4ba66b0a25b8; original eight-bit weight hash unchanged. All publication files verified, model card and Hub-preview workaround documentation match remote bytes.
