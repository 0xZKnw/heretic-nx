# Verification record

The final BF16 checkpoint was reloaded and evaluated on Apple M5/24 GiB
using Python 3.12, PyTorch 2.15.0, Transformers 5.22.2, MPS and BF16.
Only one GPU inference workload ran at a time.

- `.venv/bin/python -m pytest -q`: 423 passed, 51 skipped. Includes regression
  checks for structural discovery, selection ceilings, incomplete-screen
  exclusion, KL validation, EOS handling, hook cleanup, immutable originals,
  factor-cache aliasing, no-clobber publication, and the extracted CG solver.
- Ruff lint/format checks on the new LFM3 runner, distillation runner, shared
  ridge solver and regression tests passed; compileall and actionlint passed.
  The preexisting non-executable shebang in the old LFM2.5 experiment is
  excluded from EXE001 only for that file.
- CI passed on Python 3.11 and 3.13 for implementation commit 6869bf2.
- Built wheel includes the extracted `heretic_nx/optimize/ridge.py`.
- Two deliberately bad variants were detected by regression checks: permitting
  equal-to-baseline refusal counts and excluding the first eligible block.
- CrossHair searched the real candidate-selection gates for two-candidate
  lists with nonnegative marker counts and symbolic floating-point metrics.
  No counterexample was found. This is symbolic counterexample search, not
  a deductive proof or a proof about tensor computation.
- Nine deterministic float32 cases match the original extracted CG solver
  bit for bit, including diagnostics: dimensions 4/16/32 and penalties
  1/10/100, 40-iteration bound. Independent small dense-solve tests check the
  mathematical objective. Thirty-two sampled coordinate-conjugation cases
  check the sandwich-aware control algebra. These are sampled tests.
- Production detector fits reached the 40-iteration limit without reaching
  the 1e-5 residual tolerance. They are approximate solves; diagnostics are
  included in `heretic_nx/fit.json`.
- Full checkpoint audit: exactly 40 intended tensors changed, all remaining
  tensors bit-identical, shapes and dtypes preserved. Row norms are not
  restored in the conditional profile; maximum relative drift is 5.81%.

The BF16 validation records 4/104 lexical target markers, 2/104 safe markers
and first-token KL 0.04501414299. The reused 104-row retest selected a
frontier checkpoint, so it is not an untouched holdout. The expanded 854-row
MCQ check is inconclusive at the predeclared three-point noninferiority margin.
These checks do not prove universal refusal behavior, semantic task success,
sequence-level KL, or broad capability preservation. Full-checkpoint CPU/CUDA
paths and other llama.cpp versions have not been validated.

The GGUF build and separate native-runtime results are recorded in
HERETIC_NX_GGUF_BUILD.json and HERETIC_NX_GGUF_VALIDATION.json. Both arms use
the same Q8_0 recipe, executable, tokenized prompts and settings. Native
full-vocabulary probabilities are checked for all 128,000 unique IDs and
finite values, then normalized in float64 before KL. Model paths, sizes,
hashes and server identity are attested before and after evaluation. A parser
self-check verifies a permuted uniform distribution and rejects missing IDs,
duplicate IDs, nonfinite probabilities and unnormalized distributions.
Native greedy letter selection is checked against the full-logit argmax on
three capability rows per arm. The GGUF tensor audit finds 287 tensors,
exactly 40 intended changed projection tensors, matching shapes/types and
bit-identical bytes for all other tensors.

Final native Q8_0 results: 3/104 target markers versus original Q8 96/104,
1/104 safe markers versus 3/104, no empty responses, first-token
KL 0.04633291462 ≤ 0.05. Both Q8 arms score 438/854 (51.29%) on MCQ;
the nominal alpha-0.025 interval is [-3.18, +3.18] percentage points and
three-point noninferiority remains inconclusive. Identical row hashes bind
the format comparisons to the same pinned datasets.

The documented native server configuration passed an end-to-end French JSON
chat smoke check. `/apply-template` output equals the HF template, `/tokenize`
IDs match the HF tokenizer with exactly one BOS, and `/v1/chat/completions`
returns the requested JSON. See HERETIC_NX_CHAT_SMOKE.json.
