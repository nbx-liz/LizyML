Review-kind: review
Review-round: 1

# PR 3c code review, round 1 — does the implementation meet its written acceptance criteria?

You are reviewing **code**. You have **read-only** access to the repository; change
nothing, make no network calls, open no pull request and edit no issue. You may run
read-only commands and the test suite (`uv run pytest <paths> -q -p no:cacheprovider`).

**Every path below is relative to the repository root `/home/rem/repos/LizyML`.**

## Exact head

- Branch `fix/phase3-pr3c-calibration-params`, head `f67f8c7`, base `develop` at `5ac725e`.
- The diff under review: `git diff 5ac725e...f67f8c7`.
- Pull request #296.

## Contract (decides; read it yourself)

- `HISTORY.md` **H-0100** (the proposal for this PR, decisions 1-6 and its rule positions).
- `BLUEPRINT.md` §12.2 and §15.4 as changed by this diff.
- The acceptance criteria written before this review:
  `docs/audits/2026-09-defect-discovery/results/pr3c_acceptance_criteria.md` (Japanese).
  §2 maps each criterion to a test. §3 records explicit dispositions. §4 declares in advance
  how findings are bucketed.
- The design: `docs/audits/2026-09-defect-discovery/results/pr3c_design.md` (revision 3).
- Earlier contracts the change must not break: H-0030, H-0031, H-0047, H-0058, H-0059,
  H-0090, H-0093, H-0094 decision 8, H-0095; `CLAUDE.md` §3 (persistence compatibility,
  leakage); `ARCHITECTURE.md` layer rules.

The maintainer decisions recorded in H-0100 (honour rather than refuse, move platt to
Platt's original method in this PR, own MLE, the overridable sets, legacy migration without
a format version bump) are **not** open questions. Judge whether the code implements them.

## Production entrypoints

- `Model.fit` / `Model.tune` → `lizyml/core/model.py::_run_calibration` →
  `lizyml/core/_model_factories.py::prepare_calibration_params` and
  `check_calibration_param_names` → `lizyml/calibration/cross_fit.py` →
  `lizyml/calibration/{platt,beta,isotonic}.py`, `lizyml/calibration/_optimizer.py`.
- `Model.export_code` → `lizyml/core/_model_persistence.py` →
  `lizyml/codegen/{generator,config_writer,templates}.py`.
- Loading an old artifact → `PlattCalibrator.__setstate__`.

## What to check

1. **The criteria table.** For each row of §2, does the named test actually establish the
   criterion against the production path, or could it pass while the criterion is false?
   Name any row whose evidence is weaker than its claim.
2. **Defect classes.** Hunt each explicitly in the diff:
   - DC1: a refused or malformed `calibration.params` value that passes silently, or a
     fallback that hides a failed optimisation;
   - DC2: loose matching of names or methods;
   - DC3: drift between the runtime calibrators and the generated fitters in
     `templates.py`, or between `_optimizer.py` and the template's own tables;
   - DC4: a params path that validates but never reaches a fitted calibrator
     (every fold and C_final, fit and tune, export);
   - DC6: a condition that never fires for real inputs (the rescaling threshold, the
     OptimizeWarning probe, the legacy branch);
   - DC7: a declared value no real input can satisfy (the method table against the real
     scipy, including scipy 1.10).
3. **Platt fidelity.** The fit, its gradient, the smoothed targets, the initial point, the
   rescaling of `x0` and `bounds`, and the export mapping `a = -A`, `b = -B` against the
   unchanged `predict` form.
4. **Legacy migration.** Does every old persisted form predict as before, and does a new
   form survive a round trip? Name any container or state the migration does not reach.
5. **Leakage and layering.** OOF-only cross-fit and outer split reuse are unchanged, and
   `lizyml/calibration/` imports nothing from `lizyml/estimators/`.

## Output

At most **1500 words**.

1. `VERDICT: APPROVE` or `VERDICT: REQUEST_CHANGES`.
2. **Findings**, most severe first. Each: severity (blocking / should-change / note), the
   §4 bucket (B1-B4) you believe it falls in, file and line, a **concrete reproduction**
   (input → observed vs expected; run it if you can and say whether you did), and the
   defect class if one applies.
3. **Criteria rows** whose evidence is weaker than the claim, if any.
4. **Bounds**: what you read, ran and did not verify.

Report only what you can reproduce or point to in the code. Do not propose new features.
