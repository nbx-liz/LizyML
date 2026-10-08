Review-kind: review
Review-round: 2
Monitor-mode: absolute
Monitor-verdict: DELIVERABLE-FOCUSED
Monitor-carrier: read-only fresh context, inheriting neither maker rationale nor reviewer conclusions
Monitor-evidence: docs/audits/2026-09-defect-discovery/results/pr3c_monitor_round1.md
Monitor-disposition: continue
Monitor-rationale: The monitor found round-1 remedies deliverable-focused and proportionate, and judged the Platt default and fitter replacement to be added product scope rather than apparatus, recommending redirect. Not adopted: the maintainer chose this PR over an offered separate-PR option, with the result change disclosed. Mitigation: the replacement is declared a separable split point, and the plan row for 3c is updated.

# PR 3c design review, round 2 — revision 2 against the original design

You are reviewing a **design**, not code. Nothing is implemented yet. You have
**read-only** access to the repository; change nothing, make no network calls,
open no pull request and edit no issue.

**Every path below is relative to the repository root `/home/rem/repos/LizyML`.**

## Exact head

- Branch `fix/phase3-pr3c-calibration-params`, branched from `develop` at `5ac725e`.
- The design under review, **revision 2**:
  `docs/audits/2026-09-defect-discovery/results/pr3c_design.md` (Japanese).

## What changed since round 1

Round 1 returned `ADHERES-WITH-CHANGES` with five findings
(`docs/audits/2026-09-defect-discovery/results/pr3c_design_review_round1.md`).
All five were adopted. After round 1, research on the Platt intercept
(`docs/audits/2026-09-defect-discovery/results/pr3c_platt_intercept_research.md`)
led the maintainer to decide, **in this PR**:

1. move the `platt` defaults to Platt's original method — no regularisation,
   smoothed targets `t+ = (N+ + 1)/(N+ + 2)`, `t- = 1/(N- + 2)`;
2. replace the sklearn `LogisticRegression` fitter with an own Platt maximum-likelihood
   fit using `scipy.optimize.minimize`, like `beta`;
3. make the `platt` overridable set `x0`, `method`, `bounds`, `tol`, `options`,
   `target_smoothing`;
4. migrate the old `PlattCalibrator` state when an old artifact is loaded, via
   `__setstate__`, keeping `FORMAT_VERSION` at 2.

These are maintainer decisions. Do not re-argue whether they should have been made;
judge whether the design **implements them faithfully and without contradicting the
original design or a higher-priority document**. The loop monitor's scope concern is
recorded in the monitor note and is not a question for this round.

## The questions

1. **Round-1 findings.** Does revision 2 correctly apply each of the five round-1
   findings? Name any that is applied incompletely.
2. **Faithfulness to Platt.** Does design section 3.1 describe Platt's method correctly —
   model form and sign convention, joint estimation of slope and intercept, the smoothed
   targets, the initial point, the rescaling of large scores — against Platt (1999) as
   cited in the research note and against `sklearn/calibration.py::_sigmoid_calibration`
   in the installed scikit-learn? Is the `export_params` mapping `a = -A`, `b = -B`
   consistent with the unchanged `predict` form `sigmoid(a*s + b)`?
3. **Persistence compatibility.** `CLAUDE.md` section 3 requires a breaking change to raise
   `format_version` with a migration policy. Does section 3.6 (a class-level
   `__setstate__` migration, `FORMAT_VERSION` unchanged) satisfy that rule? Is the claim
   that an old artifact predicts identically after migration correct? Name any case the
   migration does not cover — for example an old artifact whose calibrator is inside a
   different container than `FitResult.calibrator.c_final`, or a RefitResult.
4. **Dependencies and environments.** Is it correct that no new install requirement
   arises? Are the proposed `minimize` options valid across the supported range
   (`scikit-learn>=1.3`, `scipy>=1.10`) that the CI lowest-direct lane exercises? Is the
   generated `requirements.txt` change consistent with `BLUEPRINT.md` section 15.4 and
   the README claim the template comment refers to?
5. **Anything else in revision 2** that contradicts `BLUEPRINT.md`, `HISTORY.md`
   (H-0030, H-0031, H-0047, H-0058, H-0059, H-0090, H-0093, H-0094 decision 8, H-0095),
   `ARCHITECTURE.md` layer rules, or `.claude/skills/calibration/SKILL.md`.

## Output

At most **1200 words**.

1. `VERDICT: ADHERES`, `VERDICT: ADHERES-WITH-CHANGES`, or `VERDICT: DEVIATES`.
2. **Findings**, most severe first. Each: severity (blocking / should-change / note), the
   design section, the source it conflicts with or is missing (file and section or line),
   and the concrete change you recommend.
3. **Answers to questions 1-4**, briefly.
4. **Bounds**: what you read, what you did not read, what you did not verify.

Do not implement anything and do not grade code that does not exist yet.
