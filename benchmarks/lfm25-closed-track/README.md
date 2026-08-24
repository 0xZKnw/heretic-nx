# LFM2.5 closed comparison against Heretic

This benchmark compares the frozen Heretic NX Residual-Stream BF16 artifact
against pinned Heretic revisions on the exact LiquidAI LFM2.5-1.2B-Thinking
checkpoint. It supports a model-specific claim, not a universal engine claim.

## Frozen arms

| Arm | Revision | Search | Selection rule |
| --- | --- | ---: | --- |
| Heretic master | `bedb94ef117a271532ac2058447fbc165d5051bd` | 60 trials, 15 startup, seed 42 | minimum refusal proxy, then minimum KL |
| Heretic v1.4.0 | `6ea3b8d778d047b4b3b7c5b843e21c5bea98ee8d` | 60 trials, 15 startup, seed 42 | minimum refusal proxy, then minimum KL |
| Heretic-wide | master plus the previously disclosed LFM component split, wider strength bound and seeded start | 60 trials, 15 startup, seed 42 | minimum refusal proxy, then minimum KL |
| Residual-Stream | Heretic NX v0.2.0 release artifact | frozen before this comparison | no post-comparison tuning |

The public TOML files reproduce the two unmodified upstream Heretic arms. The
Heretic-wide comparator is retained because it is stronger than either upstream
run and was already used by the v0.2.0 release.

All arms use the same base revision, chat template, train/evaluation datasets,
dataset revisions, 100-token generation cap and lexical marker list. Upstream
Heretic's objective uses first-token KL. The external evaluation does not reuse
that optimizer score. It independently applies the same three paired measures
to every saved artifact:

- XSTest: 250 safe and 200 unsafe prompts;
- StrongREJECT: 313 prompts, scored here only with the common lexical refusal
  proxy (the semantic judge is not used);
- capability: 854 deterministic first-token A/B/C/D questions from ARC
  Challenge, HellaSwag and MMLU.

The target measure combines the 200 unsafe XSTest prompts and 313
StrongREJECT prompts. The capability slice retains the original frozen 241 rows
and extends it deterministically to 854 rows; it does not replace or cherry-pick
the original observations.

## Reproduce the external evaluation

Run the module with one `--arm ID=PATH` argument per saved BF16 artifact:

```powershell
python -m experiments.lfm25_closed_track_eval `
  --arm base=PATH_TO_PINNED_BASE `
  --arm heretic_master=PATH_TO_MASTER_OUTPUT `
  --arm heretic_v1_4=PATH_TO_V1_4_OUTPUT `
  --arm heretic_wide=PATH_TO_WIDE_OUTPUT `
  --arm residual_stream=PATH_TO_RESIDUAL_STREAM_OUTPUT `
  --candidate residual_stream `
  --baseline base `
  --output runs/lfm25-closed-track-external/report.json
```

## Claim boundary

Residual-Stream may be described as outperforming the pinned Heretic master and
v1.4.0 runs only if the paired target interval has a lower bound above zero and
the 3 percentage-point capability non-inferiority and zero-margin safe-behavior
gates pass. Intervals use 10,000 paired bootstrap resamples. The per-comparison
alpha is `0.05 / (3 metrics * 3 Heretic arms) = 0.005555...`; the base-model
comparison is ancillary and is not part of that multiplicity family.

The frozen report passes all three gates against the pinned Heretic master and
v1.4.0 runs. Its corrected target lower bounds are `+0.2203` and `+0.0526`;
its corrected capability lower bounds are `-0.0234` and `-0.0222`, both above
the `-0.03` margin. Against Heretic-wide, the observed target-refusal count is
19 versus 24, but the corrected target interval `[-0.0097, +0.0312]` crosses
zero, so that comparison remains descriptive.

Refusal-marker counts are lexical proxies, not semantic task-success or safety
judgments. PRIME is the project's own validation protocol, not third-party
accreditation.
