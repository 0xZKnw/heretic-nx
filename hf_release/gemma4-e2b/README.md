---
base_model: google/gemma-4-E2B-it
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
- uncensored
- model-editing
- heretic-nx
- prime
- lm-studio
---

# Gemma 4 E2B IT — Heretic NX PRIME Q8_0

Heretic NX PRIME is a direct-Q8 behavioral edit of
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it).
It is designed to reduce false refusals while explicitly measuring drift from
the unedited Q8 deployment artifact.

The release contains the evaluated `Q8_0`. `PRIME` is the project's internal
evidence and capability-preservation protocol, not a third-party certification.
This GGUF is for text generation and does not bundle the base model's separate
image or audio preprocessing components.

## File

| File | Size | SHA-256 | Refusal markers |
| --- | ---: | --- | ---: |
| `Gemma-4-E2B-it-Heretic-NX-PRIME-Q8_0.gguf` | 4.95 GB | `43d2b7ed2d756f7ddd5100eafaa23d2ff108ab2af60b89a79b7634b7da1050d9` | **6/104** |

## Evaluation

The final GGUF was loaded from disk with pinned llama.cpp commit
`d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b`. Refusal evaluation used the
native pre-tokenized `/completion` endpoint, greedy sampling, 96 generated
tokens, four runtime slots and the official chat template with
`enable_thinking=False`.

| Metric | Result |
| --- | ---: |
| Lexical refusal markers, 104 harmful rows | **6** |
| Hit rows, one-based | `3,38,60,68,75,97` |
| Mean first-token KL(original Q8 || variant), 104 benign rows, full 262,144-token vocabulary | **0.008579** |
| Median first-token KL | **0.000054** |
| P95 first-token KL | **0.025772** |
| Maximum first-token KL | **0.201503** |

The release satisfies both predeclared targets: at most 6 lexical refusal
markers and mean first-token KL at most 0.05.

Lexical markers are a refusal proxy, not semantic harmful-task success or a
universal quality score. The 104 harmful rows participated in development and
selection, so this suite is not an untouched holdout.

### Paired capability check

After candidate selection, the original and Heretic Q8 artifacts were evaluated
on the same deterministic 854-question slice from ARC-Challenge, HellaSwag and
MMLU. Each answer was the greedy first-token argmax restricted to `A/B/C/D`;
prompts, tokenizer, runtime and precision were identical between arms.

| Task | Rows | Original Q8 | Heretic NX Q8 | Difference |
| --- | ---: | ---: | ---: | ---: |
| ARC-Challenge | 256 | **74.22%** | 73.44% | -0.78 points |
| HellaSwag | 256 | 58.59% | **59.38%** | +0.78 points |
| MMLU | 342 | 61.99% | **62.28%** | +0.29 points |
| **Overall** | **854** | 64.64% | **64.75%** | **+0.12 points** |

The paired bootstrap 95% interval for Heretic minus original is **[-0.94,
+1.17] points**. There were 543 questions both got right, 292 both got wrong,
9 original-only successes and 10 Heretic-only successes. This passes the
predeclared 3-point non-inferiority margin and the symmetric +/-3-point
equivalence gate. The interval crosses zero, so this does **not** demonstrate
an aggregate accuracy increase; it supports capability preservation on this
narrow multiple-choice slice.

## Method

The artifact is edited directly from the source Q8_0; it is not reconstructed
from a separately quantized edited checkpoint. Seven dense shared operators are
modified: FFN down projections in layers 15, 23, 25 and 26, plus attention
output projections in layers 16, 17 and 30.

The selected profile uses benign-penalized detector distillation at
`lambda=100`, `beta=4.0`, additive repair `gamma=1.875`, and two coordinate
strength adjustments. The final composed detector at every site is projected
orthogonal to the BF16 activation of benign evaluation row 53, which was the
development KL outlier. This targeted repair is disclosed because it means the
benign KL suite also participated in final selection and is not an untouched
holdout.

All plans, factors, source artifacts, runtime evaluations and untouched output
bytes are SHA-256 bound. The compact release report, selected factor artifact,
protected input vector, ablation plan and sanitized evaluation reports are
included under `evaluations/`; reproducible experiment scripts are available in
[`0xZKnw/heretic-nx`](https://github.com/0xZKnw/heretic-nx).

## LM Studio / llama.cpp

Download the GGUF and load it normally. For llama.cpp:

```bash
llama-server \
  -m Gemma-4-E2B-it-Heretic-NX-PRIME-Q8_0.gguf \
  -ngl 99 -c 4096 --jinja
```

Use a recent runtime with Gemma 4 support. Thinking is disabled in the reported
tests; applications should select the desired chat-template mode explicitly.

## Limitations and responsibility

This edit intentionally weakens refusal behavior. It can increase compliance
with unsafe, illegal, incorrect or otherwise harmful requests. It does not add
factuality, judgment, sandboxing or application-level safety. Run untrusted
generations in an appropriate sandbox.

The capability evaluation is narrow, and both behavioral suites participated
in development. The release is not externally certified or claimed equivalent
to the original model on every task.

Use is subject to the [Gemma 4 Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license).
