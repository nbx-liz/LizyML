Review-kind: review
Review-round: 2
Monitor-mode: absolute
Monitor-verdict: DELIVERABLE-FOCUSED
Monitor-carrier: lizy-runner managed monitor (codex gpt-6.1-sol, medium), run h0147-proposal-review
Monitor-evidence: monitor round 2 attempt 1 on head 6fcc8d8 classified every round-1 remedy as deliverable or bounded evidence and recommended continue
Monitor-disposition: continue
Monitor-rationale: the remedies revise the Proposal contract (name rule, adapter enumeration, acceptance 3); the added probes are bounded evidence
Monitor-head: 6fcc8d8b5d1bdf487a32c392ab5f33ebc21f63e3

# Review round 2: Proposal H-0147 (HISTORY.md `## H-0147`) after the round-1 remedies (#406)

Review the Proposal text at this head (`6fcc8d8`; the remedy is `git diff 64eac3a HEAD`). Answer as findings with `file:line` locations and executed or quoted evidence. `APPROVE` when the Proposal is correct, complete and implementable as written; `CHANGES_REQUESTED` otherwise.

## Round 1 and its remedies (verify each, do not re-derive the rest)

1. **Line-break names** (major): the rule is now "empty, or holds `\n` or `\r`" with reasons `empty_feature_name` / `line_break_in_feature_name` (対応方針 1). Evidence: `docs/audits/2026-11-i406/booster-names.out` (21 names, first and last positions, 42 cases) and `linebreak-readback.out` (the line-break names that read back come back as `a`, and `"a\nobjective=regression"` changes predictions). Does the rule now match LightGBM's model text exactly as the evidence shows — every case that fails to read back or reads back with a different `feature_name()` is refused, and no other case is? Is refusing the last-column line-break names (which load today, altered) consistent with the stop criterion's revised wording "同じまま読み戻せる"?
2. **Fold boosters** (major): the check and the warning use a new `_saved_adapters(fit_result, refit_result)` listing `refit` and every `fit_result.models` entry as `fold k`, independent of `pipeline_state_per_fold` (対応方針 2–3). Does that cover every booster `fit_result.pkl` and `refit_model.pkl` hold? Are there other LightGBM boosters saved (for example a calibrator in `lizyml/calibration/`) whose feature names a user column can reach?
3. **Acceptance 3** (minor): the refused names are now excluded.

Also check that the revised 互換性, Firing rate (6/17 and 14/42 against the three `.out` files), 代替案 and acceptance criteria 1, 2, 5 and 6 are consistent with each other and with the code (`export_code` order: `project_column_names` admits a name with `\n`, so the new refusal reaches `export_code` for it).

## Decided (do not reopen)

The owner's rulings in 管理者の判断, and the choices the round-1 remedy made within them.

## Environment

The workspace is a digest-fixed, read-only copy of this head. Run what your environment allows; say what could not run. Never run the full suite.

## Stop criterion

The Proposal's own レビューの停止基準 applies. A finding must be a contract defect or a false factual claim; #409's names, rescuing 0.21.1 artifacts and other estimators are not findings.
