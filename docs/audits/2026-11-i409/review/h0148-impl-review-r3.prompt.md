Review-kind: review
Review-round: 3
Monitor-mode: relational
Monitor-verdict: CONVERGING
Monitor-carrier: lizy-runner managed monitor (codex gpt-6.1-sol, medium), run h0148-impl-review
Monitor-evidence: monitor round 3 attempt 1 on head a944a4c compared rounds 1-2: both remedies stay inside the duplicate-refusal contract, the probes are bounded to the grouping question; recommended continue
Monitor-disposition: continue
Monitor-rationale: converging on check_duplicate_columns; round 2's question was settled by an owner ruling
Monitor-head: a944a4c4172541b7bc90cfe12af6a0a22abfc665

# Review round 3: the implementation of H-0148 (#409) after the round-2 remedy

Review the implementation at this head (`a944a4c`; the remedy is the last commit, `git diff ebb50f6 HEAD`).

Round 2 found that `Index.factorize` merged None and NaN, which the pandas Index keeps apart. Measured (`docs/audits/2026-11-i409/missing-labels.out`): pandas' `duplicated` and `get_loc` keep None and NaN apart and LightGBM accepts that frame, but LizyML's `select_training_columns` (`df[list(columns)]`) turns None into NaN, so `["x", None, NaN]` fails in LightGBM today with `Feature (nan) appears more than one time`. Also measured (`label-grouping.out`, and in the remedy's tests): `factorize` merges `"a\x00b"` with `"a"`, which `duplicated` does not.

**Owner ruling (decided, do not reopen)**: missing labels (None, NaN) count as one label in the duplicate check.

The remedy maps every missing scalar label to one NaN key (`_grouping_key`), groups with `Index.duplicated(keep=False)` and `Index.get_loc` on those keys, and matches roles and `features.exclude` with `get_loc` on the original labels. Verify that grouping against pandas' own selection for non-missing labels (`0`/`False`, `1`/`1.0`, tuples, `"0"`/`0`, `"a\x00b"`/`"a"`), the one-label treatment of missing labels, the reviewer's round-2 examples, and that nothing outside `check_duplicate_columns` changed. Answer as findings with `file:line` and evidence; `APPROVE` when the implementation does what Proposal H-0148 and the ruling say, `CHANGES_REQUESTED` otherwise.

## Environment

Read-only copy of this head; run single test files with the workspace's Python if it has the dependencies, never the full suite. Say what could not run.

## Stop criterion

The Proposal's own レビューの停止基準 applies: contract defects or false statements in the diff only. The owner ruling above is not a finding.
