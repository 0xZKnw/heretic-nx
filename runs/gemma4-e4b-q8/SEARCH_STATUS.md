# Gemma 4 E4B Q8 search — 2026-09-05

Contract: static Q8_0 GGUF, at most 6 lexical refusal-marker hits on the pinned
104-row set, and mean full-vocabulary first-token KL(base || candidate) at most
0.05 on the pinned 104 benign prompts. Both metrics must refer to the same
artifact hash. These sets have been used for tuning; they are **development
sets**, not independent held-out evidence. Capability checks follow both gates.

## Best complete measurement so far

- `trajectory-blend-0p16-scale1p04`: **5/104**, mean KL **0.04957622648575924**.
- SHA-256: `db49bb67bc7b81f4636ee176d6bb1c72377face53e9f2662239a5d1087c5b673`.
- Refusal evidence: `refusal/trajectory-blend-0p16-scale1p04.json`.
- Natural-order recheck: `refusal/trajectory-blend-0p16-scale1p04-final.json`, also **5/104**.
- KL evidence: `kl/trajectory-blend-0p16-scale1p04-vs-base.json`.
- **Both requested gates passed. The separate capability non-inferiority gate failed; weights are not published.**

## Paired capability result

On 854 frozen questions, the original scored 680/854 (79.6253%) and the
candidate scored 669/854 (78.3372%): a measured -1.2881 percentage-point
difference. Conservative paired binary 95% interval: [-3.7941, +1.2360]
points. The predeclared three-point non-inferiority gate is not established;
this is not proof that the true loss exceeds three points. Per-task checks
also did not establish non-inferiority.

Evidence: `capability/trajectory-blend-0p16-scale1p04-final-vs-base-q8.json`.
Its evidence SHA-256 is
`b2c6b11b53a4cd0e245be6b43a28fc5e1881b79e7de0df525c5d4b5ada155cf1`.
The validated code was committed and pushed as `5a9e978`. Publication of
weights is paused pending the user's decision about this capability tradeoff.

The previous complete frontier point was 6/104 and KL 0.06018327703259983,
SHA-256 `62813cc8b599f866a50ad043b2ac1a1af343bdd4586a618dd249c85f087c1a1e`.
Rebuilding it as `trajectory-blend-0` reproduced that exact hash.

## New approach and rejected screens

The native capture instrument now supports every input token position, rather
than just the final position. In a two-prompt control its output logits were
bit-identical to the reference raw-logit collector. The corresponding benign
input subspace is projected out of the edit's right factors, with a partial
blend to retain behavioral effect. This does not mathematically guarantee
end-to-end KL preservation, especially after Q8 rounding and upstream edits.

| Candidate | Observed marker hits | KL |
| --- | --- | --- |
| Full trajectory protection, all seven sites | 9/16 | Skipped |
| Trajectory blend 0.3 | 10/16 | Skipped |
| Trajectory blend 0.1 | 7/64 | Skipped |
| Full trajectory protection, sites 2/4/6 only | 7/16 | Skipped |
| Full row-norm restoration of previous best | 7/8 | Skipped |
| Trajectory blend 0.15, scale 1.025 | 7/16 | Skipped |

Intermediate complete result: trajectory blend 0.1, scale 1.025 reached 5/104
and mean KL 0.053380018593792516, artifact SHA-256
`95a30816041f9040203767a1233efb2e096d4c4b633ca9f901a26f6f30d68225`.
Trajectory blend 0.15, scale 1.04 then reached 5/104 and KL
0.05064285274374125, narrowly missing the KL cap; it was not accepted.

These partial denominators are the actual evaluated counts, not estimates for
104. The screening runner visits difficult development cases first, includes
every row exactly once, and stops after a batch takes the count above six.
It checks complete 104-row coverage and consistent marker counts before KL.

Row-norm restoration is a separate research experiment that rescales Q8 block
scales while retaining integer codes. Untouched GGUF regions are hash-checked.
Its numerical correctness does not imply behavioral success; the full-strength
variant failed the refusal gate above. It is not enabled in the core editor.

Nothing from this search has been published as an accepted E4B model.
