# Qwen3.5-9B Q8 search

Baseline and infrastructure validated; candidate search is in progress.

- Exact baseline: SHA-256 809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4.
- Baseline: 104/104 lexical marker hits, 429.679 seconds, 96-token limit,
  official non-thinking template, explicit cache_prompt=false, four slots.
- The earlier `base-q8.partial.json` run was interrupted before completion to
  make cache_prompt=false explicit. It is not acceptance evidence.
- Baseline full-vocabulary logits: 104 x 248320, 46.052 seconds total,
  38.923 seconds inside the collector. No BF16 speed comparison was performed.
- Geometry: 64 safe and 64 target train-geometry rows, 64 native projection
  inputs, 41.351 seconds. No test rows used to fit initial directions.
- Capture-control logits for geometry rows 0 and 64 are bit-identical to
  independently collected reference logits; maximum absolute difference 0.
- All 128 geometry prompts have exact HF/native tokenizer ID parity.
- Protected sample-space FP64 solver checks residuals after FP32 export;
  small feature-space reference tests pass. This does not guarantee low
  inference-time KL or generalization from training geometry.
- First candidate: `l10-top4-b1`, rank-one edits at FFN layers 24, 25, 23, 21.
  SHA-256 d86c1c7396b1ebd5276bf72d8df3a9dc105e4575f938a6e80357bee172a6621b.
  Rejected: 8/8 lexical marker hits. Doubling beta was also rejected at 8/8.
- Eight middle mixer outputs, beta 2: rejected at 8/8.
- All 32 mixer outputs, beta 2: rejected at 7/32. This is a partial
  measurement, not an estimated full-104 result. KL was skipped.
- All 32 mixer outputs, beta 3: screening in progress.
- A shared residual-stream direction is an additional research branch.
  The named-node native capture recorded all 32 residual outputs on 128
  geometry rows with logits bit-identical to the projection-input collector
  on every row. A C++ matcher harness passed for named activations, ordinary
  matrix-input targets, unrelated nodes, and disabled callbacks.
- No winning candidate, capability claim, or external weight publication yet.

Reproduction entry points: `experiments/qwen35_9b_q8_geometry.py`
(`prepare`, `capture`, `control`, `fit`, `build`),
`experiments/qwen35_9b_q8_eval.py`, `experiments/qwen35_9b_q8_kl.py`, and
`experiments/qwen35_9b_q8_screen.py`. See CONTRACT.md for fixed data revisions
and thresholds. Capture and factor files are generated locally; retain their
hash-bound manifests. The 104-row refusal/KL sets become development evidence
when used to choose candidates.

Named capture can be built alongside the original collector, without replacing
the executable used in the earlier attested experiments:

```sh
clang++ -std=c++17 -O2 experiments/llama_capture_weight_inputs.cpp \
  -I references/llama.cpp/include -I references/llama.cpp/ggml/include \
  -L build/llama.cpp-native/bin -lllama -lggml -lggml-base \
  -Wl,-rpath,@loader_path -o build/llama.cpp-native/bin/llama_capture_named
```

The same compile arguments with `tests/native_capture_matcher.cpp` build the
matcher harness. The Qwen-only residual diagnostic is
`experiments/qwen35_9b_q8_residual.py`. The validated initial pipeline tests
also passed from a clean archive of Git commit eb1890d (35 scoped tests).
