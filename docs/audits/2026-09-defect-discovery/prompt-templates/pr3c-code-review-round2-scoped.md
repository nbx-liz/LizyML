Review-kind: review
Review-round: 2
Monitor-mode: absolute
Monitor-verdict: DELIVERABLE-FOCUSED
Monitor-carrier: Codex gpt-6-astra effort low, read-only fresh context inheriting neither maker rationale nor reviewer conclusions
Monitor-evidence: docs/audits/2026-09-defect-discovery/results/pr3c_code_monitor_round1.md
Monitor-disposition: continue
Monitor-rationale: The monitor classified both production remedies as deliverable changes with bounded supporting tests, and the issue filings as bounded tracking. Adopted. Per the pre-declared B1 rule, round 2 verifies only the two round-1 fixes; finding 3 is filed as #297 and the B4 evidence gaps as #298.

# PR 3c code review, round 2 — scoped to the round-1 fixes

You are verifying **two fixes**, not re-reviewing the pull request. You have **read-only**
access; change nothing, make no network calls, open no pull request and edit no issue. You may
run read-only commands. To run tests use
`/home/rem/repos/LizyML/.venv/bin/python -m pytest <paths> -q -p no:cacheprovider --no-cov`
(`uv run` cannot take its cache lock in your sandbox).

**Every path below is relative to the repository root `/home/rem/repos/LizyML`.**

## Exact head

- Branch `fix/phase3-pr3c-calibration-params`, head `{{HEAD}}`.
- Round 1 reviewed `f67f8c7`. The fixes under review are exactly `git diff f67f8c7 {{HEAD}} -- lizyml tests`.

## Round 1, as recorded

`docs/audits/2026-09-defect-discovery/results/pr3c_code_review_round1.md`, and its disposition in
`docs/audits/2026-09-defect-discovery/results/pr3c_acceptance_criteria.md` §5.

1. **Finding 1 (B1).** A list or dict `method` escaped as `TypeError` instead of `CONFIG_INVALID`.
   Fix: `lizyml/calibration/_optimizer.py::validate_optimizer_params` checks the type before the
   table lookup.
2. **Finding 2 (blocking B3 exception).** The generated platt/beta fitters read the names they knew
   and discarded the rest of an edited `config.json`. Fix: `lizyml/codegen/templates.py` adds
   `_check_cal_params` and `_run_minimize`. The generated fitters refuse what
   `validate_optimizer_params` refuses, and scipy's unknown-option `OptimizeWarning` becomes a
   refusal.

Finding 3 and the evidence-gap rows are **out of scope** for this round (filed as #297 and #298).

## The questions

1. **Does fix 1 close finding 1?** Is there any other value of a `calibration.params` setting,
   reachable after value normalisation, that still escapes `validate_optimizer_params` as something
   other than `LizyMLError(CONFIG_INVALID)`?
2. **Does fix 2 close finding 2?** For platt and beta, is there a `calibration_params` value that
   `config.json` can express (JSON: dict, list, str, number, bool, null) which the runtime refuses
   but the generated fitter accepts — or the reverse, a value the runtime accepts that the generated
   fitter now refuses? Run the generated code; the test
   `tests/test_codegen/test_calibration_params_codegen.py::test_generated_fitter_refuses_what_the_runtime_refuses`
   shows how to load it.
3. **Did either fix break anything it touches?** Consider the generated fitters on valid settings,
   warnings other than `OptimizeWarning` that `_run_minimize` re-emits, and the scipy 1.10 floor.

Do not report findings outside these three questions. If you notice one, list it in one line
under **Out of scope** with no severity.

## Output

At most **900 words**.

1. `VERDICT: APPROVE` or `VERDICT: REQUEST_CHANGES` — about the two fixes only.
2. **Findings** answering questions 1-3. Each gives the question number, file and line, and a
   concrete reproduction (input → observed vs expected; say whether you ran it) with a severity
   (blocking / should-change / note).
3. **Out of scope** (one line each, optional).
4. **Bounds**: what you read, ran and did not verify.
