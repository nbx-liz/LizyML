# #270 re-measurement (2026-10-07, develop `1d41b66`)

Issue #270 said 179 of 1803 tests "make a behavioural claim and never execute an
operation capable of producing the effect they claim", measured in September at
`3abb6c4`. That list was never saved, and the suite has grown since, so the
population was measured again at the current head.

## What was run

1. `instruments/trace_plugin.py` over the default suite (`-m 'not slow'`):
   `trace_run.txt` (the pytest output), 9296 traced items. The trace file itself is not kept (large).
2. `instruments/d6_classify.py` → `d6_rows.jsonl`: 2335 test functions,
   CANDIDATE-HOLLOW 244 / SOUND 495 / STRUCTURAL 264 / CANNOT-TELL 1332.
   Its positive control (`test_feature_weights_changes_importance`, which
   trains) must come out SOUND with `lightgbm.train` traced, or the run exits
   non-zero; its row in `d6_rows.jsonl` is SOUND with that hit. The version
   that produced these rows printed this check inverted and never failed on
   it; the rows themselves do not depend on it.
3. `instruments/d6_kill_confirm.py` + `d6_tally.py` → `r6/`: with each
   candidate's producer set patched to raise, 231 pass (confirmed "never
   executes the producer") and 13 fail. "Confirmed" requires a `PASSED` line
   for every item of the id (`-rA`); an id with no result is UNRESOLVED and
   fails the tally (0 here). The control test
   (`TestFeatureWeightsE2E::test_feature_weights_applied`) passes normally and
   fails with `ProducerRan` under every mode — executed; its verdicts are the
   `control` entry of `r6/tally.json`. The `r6/` records were produced by the
   committed instruments run against an archive of `1d41b66` (the commit
   measured), so paths in the logs point at that scratch copy.
   `confirmed.json` lists the 231 with their kill mode and age relative to
   `3abb6c4` (176 unchanged, 54 new, 1 modified).

## What "confirmed" did and did not mean

The classifier maps nouns in a test's name and docstring to producers
("fit" → `lightgbm.train`, "prediction" → `Model`'s public members, ...).
"Confirmed" therefore means the test never runs the producer its *words*
matched, not that its claim is unmet. A refusal test named "... before
training" and a unit test of `wape` both match. So every one of the 231 was
read:

- `triage_A.json` / `triage_B.json`: two source-reading passes
  (`triage_capsule.md` is their brief).
- `codex/`: one critique of the whole classification and four re-check runs
  that between them opened every UNIT / NEGATIVE row (the follow-up runs
  `recheck31` / `recheck32` exist because `recheck1`'s transcript showed it had
  opened 46 of its 109 rows; they return one finding per row).

## Result — `dispositions.json`

| final | count | meaning |
|---|---:|---|
| UNIT | 167 | the claim is about the unit the test calls |
| NEGATIVE | 48 | the claim is a refusal before the producer; passing under the kill is the claim |
| HOLLOW | 6 | the claim needs the producer and nothing else asserts it there |
| OVERCLAIM | 5 | as HOLLOW, but another test asserts it at the boundary |
| WEAK | 5 | can pass vacuously or credits the wrong gate |

The 16 non-UNIT/NEGATIVE rows are repaired in the same PR; each row says how
(`action`, `repaired_at`). Acceptance:

- `instruments/i270_kill_repaired.py`: every repaired boundary test passes
  unkilled and fails with `ProducerRan` under the kill that made it hollow
  (exit 0 only if all do).
- `instruments/i270_mutations.py`: seven mutations of the code each repaired
  test claims to cover. For each, the repaired test must pass unmutated and
  fail (pytest's tests-failed exit, no errors) mutated, and the pre-repair
  version (from `1d41b66`) must pass mutated — except `filter_metrics`, where
  the old test also caught the mutated shape and is expected to fail, and
  `test_every_calibration_default_written_as_an_alias_reaches_training`, which
  is new and has no pre-repair version, so its base run is skipped. Exit 0
  only if every expectation holds.

Repairing `test_dict_form_objective_raises` exposed a real defect (a dict or
list `objective` raised a raw `TypeError`), fixed under H-0116.

## Corrections to earlier records

- `phase3-plan.md` §4 PR 2 says `test_fit_args_override_tune_best` was
  rewritten to drive `Model.fit(params=...)`. It was not: PR 2 added
  `test_fit_params_outrank_the_tuning_result` beside it. It is now named for the
  merge helper it tests (`test_merge_override_outranks_tune_best`).
- `phase3-plan.md` §5 says the convention "a test named for an effect at a
  boundary asserts at that boundary" is stated in `skills/testing/SKILL.md`. It
  was not, and that file lives under the git-ignored `.claude/`, so it cannot
  carry a shipped convention. This PR states it in `CONTRIBUTING.md` (Testing
  Requirements).
- `build_calibration_splitter` has no production caller. That is the H-0058
  deprecation shim, listed in `docs/DEPRECATIONS.md` and tracked by #148, not
  an unreachable feature.
