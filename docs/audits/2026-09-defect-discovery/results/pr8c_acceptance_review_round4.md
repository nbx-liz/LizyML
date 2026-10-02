VERDICT: REQUEST_CHANGES

Reviewed `cbbd5bde6878753adc9a2185bc2a3ffc25ac3313` against `origin/develop` (`33a3f6e`). Two blockers remain. The repository and all 14 reused worktrees are clean; no GitHub mutations were made.

**Round-3 findings**

| # | Status | Evidence |
|---|---|---|
| 1 | Partly resolved | Original relative-import counterexample now returns INCOMPLETE. Explicit-package variants still manufacture RED and return COMPLETE; see blocker 1. |
| 2 | Resolved | Real pytest collected `[required]` but reported `[substitute]`; p3 now rejects both the missing and unexpected identity. |
| 3 | Resolved | #288 correctly describes a coverage gap. Its revised mutation adds zero failures to pre-fix tests and one to shipped tests. |
| 4 | Resolved | Real collection-error test passes. Removing #264’s mutation now produces INCOMPLETE through p2. |
| 5 | Resolved | Unhashable disposition/outcome and non-object manifests raise `ManifestError`. |
| 6 | Resolved | Recounted 25 staged helpers plus unstaged `conftest.py`, and 22 pinned comments. |

Evidence: [counterexamples](/tmp/pr8c-round4-counterexamples.txt), [historical RED replay](/tmp/pr8c-round4-red.txt), [seam replay](/tmp/pr8c-round4-seams.txt), [staging replay](/tmp/pr8c-round4-staging.txt).

**Criteria**

All 122 instrument tests passed. The statuses below also incorporate the independent counterexamples and real-data executions.

| # | Status | Evidence |
|---|---|---|
| 1 | MET | Deliverables tracked; deferred implementations removed; JSON ignore exception verified. |
| 2a | MET | All 26 malformed-row cases and non-object manifest tests passed. |
| 2b | MET | Valid manifest; 25 issues equal the plan-derived set. |
| 2c | MET | Both-column parsing and malformed-table tests passed. |
| 3 | MET | Closed node grammar, duplicate refusal, spaces and `::` cases passed. |
| 4a | MET | FAILED-node requirement, real collection errors and earliest-parent selection passed. |
| 4b | MET | Staging/restoration tests passed; reused worktrees remained clean. |
| 4c | **NOT MET** | Literal explicit-package imports bypass the guard; blocker 1. |
| 4d | MET | Real unused-new-module-import counterexample remains green, providing no RED evidence. |
| 4e | MET | Mutation provenance, failure, collection-error refusal and restoration tests passed. |
| 4f | MET | Real unrelated-test mutation counterexample remains green. |
| 4g | MET | Outcome tests passed; real same-count identity substitution refused. |
| 5 | MET | Population, derivation, failure and declared-count tests passed. |
| 6a | MET | Each missing GitHub condition is refused. |
| 6b | MET | Pinned-comment, acknowledgement, title and citation-boundary tests passed. |
| 6c | MET | Partial, not-planned, decision-only and missing-test/PR cases passed. |
| 7 | MET | Version copying, interpreter, derivation cwd/environment and wrong-head tests passed. |
| 8 | MET | Verdict arithmetic and separate reporting tests passed. |
| 9 | MET at frozen base | Reproduced trial at `33a3f6e`, exit 1; pre-merge rerun remains pending. |
| 10 | MET | All nine mutations caught with the specified classifications. |
| 11 | MET | Required documents updated; completion explicitly deferred until the post-PR-9 run. Separate factual error in §5 remains. |
| 12 | NOT VERIFIABLE in full | Ruff, formatting and mypy passed. Full suite has only the disclosed metadata failure; hosted CI remains pending. |

**Fact check**

| # | Claim | Result | Evidence / correction |
|---|---|---|---|
| 1 | 122 tests now, 108 previously; 12 newly effective RED cases and two negative guards | TRUE | Old instrument with current tests: **12 failed, 110 passed**. Old collection: 108. |
| 2 | #288 reports missing coverage; `22b11b3` is test-only | TRUE | Live issue inspected; commit changes tests and acceptance documentation only. |
| 3 | Item 11’s five seam rows | TRUE | Reproduced pre-fix/shipped newly-failed counts: **9/13, 1/4, 0/1, 0/3, 0/0**. |
| 4 | `_model_tuning.py:266` is a pre-study ownership check | TRUE | `resolved_model` feeds only `check_training_managed_overrides`; mutation adds no failures. |
| 5 | Current #288 mutation demonstrates pre-fix blindness | TRUE | Zero new pre-fix failures; shipped failure is `test_the_provider_fixed_seam_carries_one_spelling`. |
| 6 | Mutation provenance and counts | TRUE | Trial validates added-line provenance; #264 gives **96 failed/133 passed**, #288 **1 failed/228 passed**. |
| 7 | 19 rows, 13 before commits, zero references, 25 helpers | TRUE | Staging replay matches saved output; existing helpers and unstaged `conftest.py` are identical. |
| 8 | 22 affirmative, correctly timed pinned comments | TRUE | Read all 22 live pinned comments; full trial verified association and timing. |
| 9 | Trial 3 split and JUnit identity reconciliation | TRUE | **20 COMPLETE, 2 RED-BY-MUTATION, 1 PARTIAL, 1 NOT-PLANNED, 1 INCOMPLETE, 0 UNKNOWN**; all 23 executed rows reconcile. |
| 10 | Nine mutations caught | TRUE | (a)–(h) INCOMPLETE; (i) UNKNOWN; campaign exit 0. |
| 11 | Plan banner/row and MANIFEST describe shipped tooling and post-PR-9 completion | TRUE | Deliverables and counts verified; no production-code changes. |
| 12 | Deferred README’s archive correction and population growth | TRUE | No archived manifest in available Git history; shipped manifest tracked; derivation gives **92 → 110**. |
| 13 | `pr2_NEXT.md` describes pending merge and PR 9 work | TRUE | Instructions distinguish pending merge, #271 work and the subsequent completion run. |
| 14 | Guard covers the stated AST-visible relative imports and `fromlist` forms | **PARTLY** | Original forms work; explicit package arguments do not. |
| 15 | §5 says #262 and #265 both receive an aggregate p2 comparison | **FALSE** | #262 does; decision-only #265 skips p2 entirely. |

The [trial](/tmp/pr8c-round4-trial.txt), [mutation campaign](/tmp/pr8c-round4-mutations.txt), historical seam table and staging replay match their saved evidence bodies.

**Blocking findings**

1. **B1/B2 — Explicit-package literal imports still manufacture false RED.**

   [The guard](/home/rem/repos/LizyML/docs/audits/2026-09-defect-discovery/instruments/phase3_gap.py:300) resolves strings against the importing file’s package but does not combine an explicit package argument with the relative module name.

   Executed before-code in `lizyml/a.py`:

   ```python
   from importlib import import_module
   try:
       Y = import_module(".new", "lizyml.other").Y
   except ImportError:
       Y = 0
   ```

   With after-only `lizyml/other/new.py` containing `Y = 1`:

   - Guard references: `[]`.
   - Original before: **1 passed**.
   - Staged before: **1 failed**.
   - After: **1 passed**.
   - Instrument verdict: **COMPLETE**.

   A literal `__import__('lizyml.other', fromlist=['new']).new` variant produces the same result. Pytest execution was real; GitHub metadata was scripted valid metadata.

   These imports are AST-visible and therefore outside §5’s excluded paths. Resolve these literal call arguments—or conservatively refuse staging—and add both counterexamples to criterion 4c.

2. **B3 — §5 overstates the evidence collected for #265.**

   [The two-PR bound](/home/rem/repos/LizyML/docs/audits/2026-09-defect-discovery/results/pr8c_acceptance_criteria.md:151) says #262 and #265 receive a comparison from before the earliest PR to the after tree.

   Execution shows #265 is `decision-only`: **9 passed**, COMPLETE, with no before-tree execution or p2 result. The implementation and §3 exemption table agree.

   Correct the bound to say that #262 receives aggregate RED evidence, while #265 is exempt from p2. This requires no new disposition or per-PR RED requirement.

**Non-blocking**

1. #288 can retain `regression` under the agreed evidence-class contract. The revised mutation supports that classification; no concrete wrong verdict requires another disposition.
2. Reproducing the historical broad-mutation row required explicitly loading the mutation from `c26ea8a`; the shipped probe reads the current manifest.
3. `git diff --check` reports one trailing space in archived `pr8c_round2_probe.txt:54`.

**Not verified**

- Hosted CI and the develop-head rerun immediately before merge remain pending, as authorized.
- A completely green local suite was not obtained: **8,714 passed, 230 skipped, 13 xfailed, 8 deselected, 1 failed**. The sole failure is the disclosed installed-metadata/version mismatch. [Full output](/tmp/pr8c-round4-full-suite.txt).
- Historical elapsed-time claims were not independently certified.

Progress: 4/4 review milestones complete — plan v1. Acceptance remains blocked by findings 1–2; disposition returns to the maintainer.