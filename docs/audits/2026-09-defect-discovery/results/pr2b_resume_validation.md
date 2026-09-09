# PR 2b resume validation — 2026-09-09

Base: `8dcf5bd6437bbba074f9c4b07574523724392bb8`. Candidate: H-0097 Revision 2,
including the fit-only boundary clarification. This report records local checks;
it is not external review, hosted CI, publication, or acceptance.

## Reproduction and corrections

- The initial 14-case regression file produced 8 failures and 6 passes on the
  baseline: four missing-origin cases, four direct duplicate cases, and six
  facade duplicates already correctly refused before adapter extraction.
- `test_seed_takes_priority_over_random_state` existed since `6619d7eb`, dated
  2026-03-07. The handoff's contrary claim was false. Direct duplicate rejection
  intentionally replaces that historical behavior; single-spelling aliases remain.
- A one-trial search overlay (`application=binary`) replacing a regression
  objective on a binary task succeeded at the base and failed in the first
  candidate. Restricting entrance validation to `fit()` preserves this behavior.
  `test_tuning_validates_after_sampled_overlay` now covers it.
- The refusal matrix detected an unregistered new validator. The matrix now
  covers three fit-input origins with executed cells, and states why the trial
  space and calibration paths are outside this merged-value gate.

## Final checks

- `ruff check .`: passed.
- `ruff format --check .`: passed (299 files).
- `mypy lizyml/`: passed (110 source files).
- Full suite through `instruments/run-exclusive.sh`: **7746 passed, 230 skipped,
  8 deselected**, 109.24 seconds. The eight deselected tests are the configured
  slow tests. Existing warnings remain (395); this is not a warnings-clean claim.
- Targeted provenance, matrix and facade tests: 71 passed before the final suite.
- `git diff --check`: passed.

Commands used the existing uv-managed Python environment with `PYTHONPATH`
pointing to the isolated candidate and `PYTHONDONTWRITEBYTECODE=1`. Tests were
not run against the original worktree by mistake. JUnit output is retained
outside the candidate at `/tmp/lizyml-pr2b-compatible-tests.xml`.

## Remaining boundaries

Freeze the candidate head for the deployed CLI review. No provider calls have
been made for this LizyML work. No GitHub mutation has been made. #286 remains
open; the six reproduced cases do not establish universal unreachability.
The original worktree and its two staged handoff files are preserved.
