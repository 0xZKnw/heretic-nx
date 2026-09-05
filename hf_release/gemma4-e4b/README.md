---
base_model: google/gemma-4-E4B-it
language:
- multilingual
library_name: gguf
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
tags:
- gguf
- gemma4
- q8_0
- abliterated
- model-editing
- heretic-nx
- research
- lm-studio
---

# Gemma 4 E4B IT — Heretic NX RESEARCH Q8_0

**A research release with an explicit capability tradeoff.** The refusal and
KL targets passed, but the capability non-inferiority gate below did not.
The weights are released with that limitation disclosed, not as a
capability-preserved PRIME release.

A static, direct-Q8 behavioral edit of
[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it),
with explicitly measured deviation from the original Q8 artifact.
No runtime steering adapter is required. This release is text-only; it does
not include the separate image/audio preprocessing components.

`PRIME` names the project's internal evaluation protocol, not an external
certification. Reduced refusal does not imply greater factuality or safety.

## Why this release required compromises

Gemma 4 E4B was a difficult model to edit in this project. This was not a
straightforward reuse of a successful recipe, nor a no-sacrifice conversion.
Getting the refusal-marker count down was possible; doing so while keeping
mean KL below 0.05 required repeated changes to the edit and direct checks on
the actual Q8 file.

The central difficulty in these experiments was the tradeoff between the
behavioral effect and preservation of the original model's outputs. Protecting
only the last prompt position left substantial drift on a few benign prompts.
Earlier positions can also affect the final prediction through attention and
cached keys/values. We therefore captured native Q8 projection inputs at every
prompt position, rather than treating a final-position constraint as an
end-to-end guarantee.

Protecting those trajectories completely then weakened the edit too much:
one such screen reached 9 marker hits in just 16 prompts. Restoring matrix row
norms was numerically successful but behaviorally unsuccessful: 7 hits in the
first 8 prompts. Both were rejected without claiming full-set results.
The useful compromise was **partial trajectory protection plus a small
strength adjustment**, measured after Q8 requantization.

Selected complete development measurements show the progression:

| Stage | Marker hits / 104 | Mean first-token KL |
| --- | ---: | ---: |
| Earlier uniform-strength Q8 edit | 6 | 0.077709 |
| Native final-position protection | 6 | 0.060183 |
| 10% trajectory blend, scale 1.025 | 5 | 0.053380 |
| 15% trajectory blend, scale 1.04 | 5 | 0.050643 |
| **Released: 16% trajectory blend, scale 1.04** | **5** | **0.049576** |

These are selected checkpoints from a longer search, not an exhaustive
benchmark or evidence of monotonic behavior. Q8 dequantization/editing/
requantization is nonlinear; small parameter changes must be tested, not
assumed to interpolate smoothly. E4B is a dense model: these experiments do
not establish that MoE routing, quantization alone, or any one architectural
feature caused the difficulty.

**What was sacrificed?** The released model has a lower measured score on
the paired capability slice: **78.34% versus 79.63%** for the original, a
**1.29 percentage-point decrease**. HellaSwag shows the largest measured loss
at 2.73 points. The mean KL is also only about 0.000424 below the requested
cap. This is a documented compromise for reduced refusal, not a claim that
the model became better overall or retained every capability unchanged.

## Artifact and acceptance results

File: `Gemma-4-E4B-it-Heretic-NX-RESEARCH-Q8_0.gguf` (8,031,242,688 bytes,
approximately 8.03 GB / 7.48 GiB).

SHA-256: `db49bb67bc7b81f4636ee176d6bb1c72377face53e9f2662239a5d1087c5b673`.

| Measurement | Result |
| --- | ---: |
| Lexical refusal-marker hits | **5 / 104** |
| Rerun in natural dataset order | **5 / 104** |
| Mean first-token KL(original Q8 || edited Q8) | **0.0495762265** |
| Median first-token KL | 0.0066655752 |
| P95 first-token KL | 0.2390656606 |
| Maximum first-token KL | 0.9797779631 |

The exact GGUF passes the requested caps of at most 6 marker hits on 104 rows
and mean first-token KL at most 0.05. This is not a claim that KL is below 0.05
for every prompt, later token, runtime or dataset. The margin below the mean
cap is small; use the pinned evaluation protocol when reproducing it.

Refusal evaluation uses the pinned `mlabonne/harmful_behaviors` test split,
greedy decoding, 96 generated tokens, four llama.cpp slots, pre-tokenized native
completion requests, and the official template with thinking disabled.
The original Q8 scored 104/104 marker hits. The edited model's hit rows are
38, 60, 68, 97 and 98 (one-based).

KL uses 104 pinned `mlabonne/harmless_alpaca` test prompts and raw logits over
the full 262,144-token vocabulary: **KL(P_original || P_edited)** at the first
generated token, with state cleared between prompts. See `evaluations/` for
dataset revisions, runtime attestations, per-row measurements and hashes.

**Both 104-row suites participated in tuning and selection. They are development
sets, not untouched holdouts.** The natural-order rerun checks reproducibility;
it is not a new independent benchmark. Lexical markers are only an operational
refusal proxy: warnings can trigger them, and refusals can evade them.

## Paired capability evaluation

The same deterministic 854-question slice was evaluated on both artifacts
using greedy first-token selection restricted to `A/B/C/D`, identical prompts,
tokenizer and runtime. This slice was not used to fit the edit.

| Task | Rows | Original Q8 | Edited Q8 | Difference |
| --- | ---: | ---: | ---: | ---: |
| ARC-Challenge | 256 | 88.67% | 88.67% | 0.00 points |
| HellaSwag | 256 | 77.73% | 75.00% | -2.73 points |
| MMLU | 342 | 74.27% | 73.10% | -1.17 points |
| Overall | 854 | **79.63%** | **78.34%** | **-1.29 points** |

There were 653 shared successes, 158 shared failures, 27 original-only
successes and 16 edited-only successes. The conservative paired binary 95%
interval for edited minus original is **[-3.79, +1.24] percentage points**.
It uses simultaneous Clopper-Pearson bounds on the two discordance
probabilities, not the older percentile-bootstrap implementation.

The predeclared three-point non-inferiority gate **did not pass**, overall or
under the simultaneous task-level checks. This does not prove that the true
loss exceeds three points: the evidence does not exclude it. The measured
score is lower, and no capability improvement is demonstrated. The project
owner explicitly approved publication as a research artifact with this
tradeoff disclosed. The failed gate remains recorded as failed in the
machine-readable release report; no threshold or confidence method was
relaxed to label it a pass.

## Method

Seven dense output projections receive rank-one weight edits:

- Attention outputs: layers 23, 25 and 35 (zero-based).
- FFN down projections: layers 6, 7, 12 and 33.

The parent recipe uses benign-penalized detector distillation (`lambda=100`)
and nonuniform site strengths. Two right-factor variants share the same left
axis: one protects final-position native Q8 inputs from the 104 benign
development prompts; the other protects all input positions from the two
largest benign development outliers (one-based rows 34 and 16).

The final right factor is `1.04 * (0.84 * last_position + 0.16 * full_position)`.
Both source variants already contain the parent strength multiplier 1.1.
The selected intervention remains rank one at each site. This partial
protection retained more behavioral effect than full trajectory projection.
Row-norm restoration was tested and rejected; it is **not** used in this file.

The edit dequantizes selected Q8 matrices, applies their low-rank deltas and
requantizes to Q8_0 with original-block preservation. It is lossy, not a
lossless optimization. Untouched GGUF regions are verified byte-for-byte.

The final factor artifact and SHA-bound ablation plan are included under
`evaluations/`. They can be applied with Heretic NX's
`apply_quantized_gguf_ablation` API to the exact source Q8 identified in the
release report. Experiment scripts are in
[`0xZKnw/heretic-nx`](https://github.com/0xZKnw/heretic-nx).

Base model revision: `ee0ef6023621cff504d758262d4e04895a5af4a2`.
Base Q8 SHA-256:
`34be82b17b4942d389b9b527170c4b058027abdd32531fda063d3d97dd8ce80a`.

## Loading

Load the GGUF normally in a Gemma 4-compatible LM Studio / llama.cpp runtime.
No separate adapter or patched inference graph is needed. Only this Q8_0 is
evaluated; no results are transferred to other quantizations.

```bash
hf download 0xzknw/Gemma-4-E4B-it-Heretic-NX-GGUF \
  Gemma-4-E4B-it-Heretic-NX-RESEARCH-Q8_0.gguf --local-dir .
```

For llama.cpp, use a recent Gemma 4-compatible build and the model's chat
template. The measured protocol uses thinking disabled; the reported numbers
must not be transferred to thinking-enabled or differently prompted runs.
The file size is not a total RAM requirement: the runtime, context/KV cache
and other applications require additional memory.

## Included evidence and edit reproduction

The repository includes the evaluated GGUF, sanitized refusal results,
per-prompt first-token KL, paired capability comparison, merge report,
release decision, and the final low-rank factors and ablation plan.
Raw potentially harmful response text and temporary research checkpoints are
not included. Response hashes are retained for auditing the local full results.

To reproduce the static edit, obtain the **exact source Q8 hash stated above**
and use the matching Heretic NX implementation and codec/runtime build:

```python
from heretic_nx.edits import apply_quantized_gguf_ablation

apply_quantized_gguf_ablation(
    "gemma-4-E4B-it-Q8_0.gguf",
    "reproduced-e4b.gguf",
    "evaluations/ablation-plan.json",
    "evaluations/edit-factors.safetensors",
)
```

The plan validates the source and factor hashes. Compare the resulting GGUF
SHA-256 with the released file before reusing any of these measurements.
The factors are an edit recipe, **not** a standalone model or a runtime LoRA.

## Limitations and responsibility

This edit weakens refusal behavior and can increase compliance with unsafe,
illegal or harmful requests. It supplies no factuality guarantees, application
sandbox, access controls or content safeguards. Neither the first-token KL nor
the narrow multiple-choice slice establishes universal capability preservation
or sequence-level equivalence. Use appropriate application-level protections.

The base model card declares the
[Gemma 4 Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license).
