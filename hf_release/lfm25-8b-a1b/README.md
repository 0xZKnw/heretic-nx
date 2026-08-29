---
base_model: LiquidAI/LFM2.5-8B-A1B
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
library_name: gguf
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
- gguf
- liquid
- lfm2.5
- moe
- heretic-nx
---

# LFM2.5 8B-A1B — Heretic NX PRIME GGUF

Heretic NX PRIME is a direct-GGUF behavioral edit of
[`LiquidAI/LFM2.5-8B-A1B`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B).
It is designed to reduce false refusals while explicitly measuring drift from
the unedited deployment artifact.

The release is the evaluated `UD-Q8_K_XL`. `PRIME` is the project's internal
evidence and capability-preservation protocol, not a third-party certification.

## Files

| File | Size | SHA-256 | Refusal markers |
| --- | ---: | --- | ---: |
| `LFM2.5-8B-A1B-Heretic-NX-PRIME-UD-Q8_K_XL.gguf` | 9.34 GB | `bea74de71f6f3cfd5f6d807ec011f743555c2c84f41501115808c502375af43d` | **4/104** |
| `Q8_RELEASE.json` | Exact Q8 provenance, protocols and claim limits | — | — |
| `evaluations/capability-*.json` | Reproducible paired capability reports | — | — |

## Evaluation

Both final files were loaded from disk with llama.cpp and evaluated over all
104 rows. Decoding used the native pre-tokenized `/completion` endpoint,
greedy sampling, 96 generated tokens, the closed-thinking chat template and
four runtime slots.

| Metric | Original Q8 | Heretic NX Q8 |
| --- | ---: | ---: |
| Lexical refusal markers, 104 harmful rows | 95 | **4** |
| Hit rows, one-based | — | `30,60,68,97` |
| Mean first-token KL(original Q8 || variant), 104 benign rows, full 128k vocabulary | 0 | **0.016948** |
| Median first-token KL | 0 | **0.004140** |

The release satisfies both targets: at most 6 refusal markers and mean KL at
most 0.05.

Lexical markers are a refusal proxy, not semantic task success or a universal
quality score. All 104 harmful rows participated in development and selection,
so this suite is not an untouched holdout.

### Paired capability check

After the Q8 candidate was frozen, the original and Heretic Q8 artifacts were
evaluated on the same deterministic 854-question slice from ARC-Challenge,
HellaSwag and MMLU. Each answer was the greedy first-token argmax restricted to
`A/B/C/D`; prompts, tokenizer, runtime and precision were identical between
arms.

| Task | Rows | Original Q8 | Heretic NX Q8 | Difference |
| --- | ---: | ---: | ---: | ---: |
| ARC-Challenge | 256 | 73.83% | **75.78%** | +1.95 points |
| HellaSwag | 256 | **36.33%** | 34.38% | -1.95 points |
| MMLU | 342 | 54.97% | 54.97% | 0.00 points |
| **Overall** | **854** | **55.04%** | **55.04%** | **0.00 points** |

The paired bootstrap 95% interval for Heretic minus original is **[-1.29,
+1.29] points**. There were 454 questions both got right, 368 both got wrong,
16 original-only successes and 16 Heretic-only successes. This passes the
predeclared 3-point non-inferiority margin and the symmetric ±3-point
equivalence gate. It does **not** demonstrate an aggregate accuracy increase;
it supports capability preservation on this narrow multiple-choice slice.

## Method

The selected edit is a benign-penalized distillation of an eight-site PRIME
teacher. It fits conditional rank-one right factors from 1,024 harmless states
and 2,627 response-trajectory states, then merges them directly into eight
Q8_0 operator-output tensors at `lambda=100` and `beta=2.25`.

No MoE expert-bank tensor is edited. The eight sites are two attention outputs
and six short-convolution outputs in layers 12, 14, 16, 17, 19, 21, 22 and 23.
Every plan, factor artifact and output is SHA-256 bound. The direct-Q8 backend,
tests and reproducible experiment scripts are available in
[`0xZKnw/heretic-nx`](https://github.com/0xZKnw/heretic-nx).

## LM Studio / llama.cpp

Download one GGUF and load it normally. For llama.cpp:

```bash
llama-server \
  -m LFM2.5-8B-A1B-Heretic-NX-PRIME-UD-Q8_K_XL.gguf \
  -ngl 99 -c 4096 --jinja
```

A recent runtime with `lfm2moe` support is required.

## Limitations and responsibility

This edit intentionally weakens refusal behavior. It can increase compliance
with unsafe, illegal, incorrect or otherwise harmful requests. It does not add
factuality, judgment, sandboxing or application-level safety. Run untrusted
generations in an appropriate sandbox.

The paired capability check is narrow and should not be read as a universal
quality guarantee. The release is not claimed to be a universal winner,
externally certified or equivalent to the original model on every task.

Use is subject to the included LFM Open License v1.0.
