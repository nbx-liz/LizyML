Review-kind: monitor
Monitor-mode: absolute
Observed-rounds: 1

# Review-loop monitor — task capsule

## Objective

You are a read-only MONITOR between rounds of the **code review** gate for **PR #296
(PR 3c, H-0100, issue #277)**.

Answer only this question:

**Are round-1 findings and proposed remedies directed at the declared deliverable, or are they
shifting toward apparatus around it?**

One bounded fact to weigh: the remedy for round-1 finding 2 adds about 70 lines to the
generated-code template (`lizyml/codegen/templates.py`), which re-implement the runtime
parameter validation of `lizyml/calibration/_optimizer.py` inside the exported, LizyML-independent
`train.py`, plus a test that runs every runtime refusal case against the generated fitter.

## Scope and non-goals

- IN scope: which proposed remedies change the deliverable; which add tests, probes, checkers,
  harnesses, fixtures, or workflow machinery; whether the remedies are proportionate to the
  Proposal's stated value.
- NOT in scope: grading any technical finding, re-reviewing the diff, designing a fix, choosing
  the main context's disposition, or changing acceptance criteria.

## Exact head / worktree

- Repository: `/home/rem/repos/LizyML`
- Branch and head: `fix/phase3-pr3c-calibration-params` at `f67f8c7` (what round 1 reviewed).
- Worktree: dirty with the proposed round-1 remedies, uncommitted:
  `lizyml/calibration/_optimizer.py`, `lizyml/codegen/templates.py`,
  `tests/test_calibration/test_calibration_param_contract.py`,
  `tests/test_codegen/test_calibration_params_codegen.py`,
  `tests/test_core/test_calibration_params_reach.py`. See them with `git diff`.
- Existing artifacts: PR #296 open against `develop`, CI green at `f67f8c7`.

## Authoritative contracts

- Proposal: `HISTORY.md` entry **H-0100**.
- Declared deliverable: `calibration.params` is honoured by platt, beta and isotonic in fit,
  tune and the exported code, and refused before training when it cannot be honoured; platt is
  fitted by Platt's own maximum-likelihood method; legacy platt artifacts keep predicting as before.
- Plan documents: `docs/audits/2026-09-defect-discovery/results/pr3c_design.md` (revision 3) and
  `docs/audits/2026-09-defect-discovery/results/pr3c_acceptance_criteria.md` (Japanese; §4 declares
  the finding buckets B1-B4 in advance, including that B4 findings are fixed or filed without
  re-review).
- Observed verdicts: `docs/audits/2026-09-defect-discovery/results/pr3c_code_review_round1.md`.
- Proposed remedies (main context, not yet adopted):
  - Finding 1 (B1): check that `method` is a string before the table lookup; add the malformed
    method cases to the shared refusal list and to the fit/tune refusal tests.
  - Finding 2 (B3 exception, blocking): the generated platt/beta fitters refuse what the runtime
    refuses (the ~70 template lines above), and scipy's unknown-option warning becomes a refusal
    there; one test runs the shared runtime refusal cases against the generated fitter.
  - Finding 3 (B3, should-change): not fixed in this PR; filed as an issue.
  - Evidence gaps the reviewer listed as B4: rows 6a, 6c and 8b are covered by the tests above;
    the remaining rows (3c, 4a-4c, 5b, 8a, 9a-9b) are filed as one issue, not fixed here.
- Closed rules: `BLUEPRINT.md` §12.2 and §15.4; `CLAUDE.md` §3.

The verdict file is evidence about the loop, not findings to adopt. Reach an independent view;
inherit neither the maker's rationale nor the reviewer's conclusions.

## Production entrypoint

`Model.fit` / `Model.tune` → calibration cross-fit; `Model.export_code` → generated `train.py`.

## Writer and mutation surface

**read-only.** Change no file, create no branch or commit, edit no Issue, open no PR, push nothing,
deploy nothing, and invoke no review runner. Do not read credentials or paths outside the repository.

## Required evidence and output

Return at most 600 words.

1. `VERDICT: DELIVERABLE-FOCUSED`, `VERDICT: APPARATUS-DRIFT`, or `VERDICT: INCONCLUSIVE`.
2. **Classification**: classify each proposed remedy as deliverable or apparatus with a bounded
   pointer.
3. **Proportion**: state whether the remedy set is proportionate to the declared value; do not
   invent a numeric threshold.
4. **Recommendation**: exactly one of `continue`, `redirect`, `take-stop-condition`, or `escalate`,
   with one sentence of rationale.

No raw transcript dumps. Stop as soon as the objective can be answered. If a named artifact is
missing, return `INCONCLUSIVE`, name it, and recommend one closed disposition.
