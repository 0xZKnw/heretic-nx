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
  Full acceptance not established; refusal screening is in progress.
- No winning candidate, capability claim, or external weight publication yet.

Reproduction entry points: `experiments/qwen35_9b_q8_geometry.py`
(`prepare`, `capture`, `control`, `fit`, `build`),
`experiments/qwen35_9b_q8_eval.py`, `experiments/qwen35_9b_q8_kl.py`, and
`experiments/qwen35_9b_q8_screen.py`. See CONTRACT.md for fixed data revisions
and thresholds. Capture and factor files are generated locally; retain their
hash-bound manifests. The 104-row refusal/KL sets become development evidence
when used to choose candidates.
