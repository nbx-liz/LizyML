# Deferred — the Phase 3 completion instrument (shipped in PR 8c)

**The instrument this directory held is shipped.** PR 8c rebuilt it and ships it
as `../phase3_gap.py`, `../phase3_manifest.json` and
`tests/test_docs/test_phase3_gap.py`. Its contract is
`../../results/pr8c_acceptance_criteria.md` §3, what it cannot read is §5 there,
and the measurements behind each rule are `../../results/pr8c_measurements.txt`.
The archived `phase3_gap.py`, `test_phase3_gap.py` and `check_derivations.py`
were deleted in PR 8c; they are in git history. Do not restore them.

## What was archived here, and what happened to it

PR 0 named four files built in `/tmp/lizyml-discovery-plan/`. That scratchpad
was lost, and they were recovered from the session transcript by
`../recover_from_transcript.py`, which replays `Write` and `Edit` tool calls but
not edits a script made to another file.

**Correction (PR 8c, measurement 7):** this README used to say four files were
archived here. Only three ever were. `phase3_manifest.json` matched the
repository's `.gitignore` rule `*.json`, so it was never committed; it existed
only in the working copy that wrote this README. PR 8c adds
`!docs/audits/**/*.json` to `.gitignore`, so the shipped manifest is tracked,
and `tests/test_docs/test_phase3_gap.py` reads the tracked file, so CI fails if it
is ever not.

The five stale items this README listed, and how PR 8c resolved each:

1. **#271's population grows with the run** (92 at `5712f41`, 110 at `33a3f6e`).
   Proposition 4 may now compare against `derived_from`, executed in the after
   worktree, instead of a literal (criteria §3, p4/p5). #271's own test does not
   exist yet; it is PR 9's deliverable, and the row is INCOMPLETE until then.
2. **#265's row named a test that does not exist.** Every row was rebuilt from
   the PR's own acceptance criteria and checked by collection at `33a3f6e`
   (criteria §2; measurement 4 counted 13 dead selectors in the archived rows).
3. **#266's row counted the wrong thing.** It now counts the 7 sites the test
   scans in the documents, so adding a declaration changes the collected count.
4. **Eight of the 29 unit tests were missing.** The test file was rewritten
   against the shipped contract; criteria §4 maps each criterion to its tests.
5. Rows naming future tests "by design" no longer exist: every row names tests
   that are on `develop`, except #271.

Proposition 6 as archived read `closedByPullRequestsReferences`, which is empty
for every issue in this run because merges go to `develop`, not the default
branch (DC7, measurement 2). It now reads a pinned closure comment.

## What stays declared rather than measured

Shipping the instrument closes "no shipped instrument". It does not make every
part of completion measured. `../../results/pr8c_acceptance_criteria.md` §5 lists
what the instrument cannot read and leaves to reviewed declarations: that a
pinned closure comment affirms the fix; that a `red_mutation` restores the
defect (or, for #288, the behaviour the missing tests would have missed); per-PR
RED for rows with two PRs; and populations a test declares itself. Phase 3
completion is decided by running the instrument after PR 9.
