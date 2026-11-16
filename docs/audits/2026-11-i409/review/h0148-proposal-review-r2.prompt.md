Review-kind: review
Review-round: 2
Monitor-mode: absolute
Monitor-verdict: DELIVERABLE-FOCUSED
Monitor-carrier: lizy-runner managed monitor (codex gpt-6.1-sol, medium), run h0148-proposal-review
Monitor-evidence: monitor round 2 attempt 1 on head ca389a8 classified the round-1 remedies as Proposal changes with one bounded probe and recommended continue
Monitor-disposition: continue
Monitor-rationale: the remedies revise the duplicate-column rule, the compatibility text and the counts; the added probe is bounded evidence
Monitor-head: ca389a8d0dffcd70768996bb073a1bf9dd15b95d

# Review round 2: Proposal H-0148 (HISTORY.md `## H-0148`) after the round-1 remedies (#409)

Review the Proposal text at this head (`ca389a8`; the remedy is `git diff 31f0a41 HEAD`). Answer as findings with `file:line` locations and executed or quoted evidence. `APPROVE` when the Proposal is correct, complete and implementable as written; `CHANGES_REQUESTED` otherwise.

## Round 1 and its remedies (verify each)

1. **Duplicate target through the exclude exemption** (major) and 2. **duplicate group_col fitting today with kfold** (major): 対応方針 4 now calls a new `check_duplicate_columns(df, roles, exclude)` from `_prepare_training_data` before `dataframe_builder.build`. A label appearing more than once is refused when it is a configured role column (`data.target`, `data.time_col`, `data.group_col`, `split.blocks.col`, `split.groups.col`, `inner_valid.group_col`) regardless of `exclude` and of the split, refused when it is a feature (not a role, not excluded), and admitted when it is a non-role excluded column. 互換性 now names the newly refused inputs (a duplicated role column the split does not read). Evidence: `docs/audits/2026-11-i409/duplicate-roles.out`. Is the role list exactly the columns LizyML reads by name from the frame (check `lizyml/config/` for every column-name field, including inner-valid and blocked-split settings)? Does the rule now refuse every duplicate that crashes and admit every one that does not, except the disclosed role case?
3. **Counts** (minor): 17 cases, 5 collisions, Firing rate 13/17 and 6/7.

Also check that 互換性, the Firing rate line, and acceptance criteria 1-5 are consistent with each other and with the code.

## Decided (do not reopen)

The owner's rulings in 管理者の判断, and refusing a duplicated role column even when the split does not read it (disclosed in 互換性).

## Environment

The workspace is a digest-fixed, read-only copy of this head. Run what your environment allows; say what could not run. Never run the full suite.

## Stop criterion

The Proposal's own レビューの停止基準 applies. A finding must be a contract defect or a false factual claim; non-colliding space names, H-0147's save check, predict frames and other estimators are not findings.
