VERDICT: REQUEST_CHANGES

Reviewed `1b4b35662afecae07bd021389d8c5814d1fe0ff0`. Both round-4 remedies work, but an additional IN-kind literal import still manufactures false RED. Repository files remain unchanged; all 15 reused worktrees are clean.

**Round-4 findings**

| # | Status | Evidence |
|---|---|---|
| 1 | Resolved for both reported counterexamples | Explicit-package and absolute-package `fromlist` examples now return INCOMPLETE. Both added tests fail against `cbbd5bd`. [Counterexamples](/tmp/pr8c-round5-counterexamples.txt), [historical RED](/tmp/pr8c-round5-red.txt). |
| 2 | Resolved | §5 correctly exempts #265 from p2. The real trial reports nine passed tests, COMPLETE, and no before/p2 execution. [Trial](/tmp/pr8c-round5-trial.txt). |

**Regression check**

| Criterion | Still met? | Evidence |
|---|---|---|
| 1–3 | Yes | Deliverables tracked, manifest covers 25 issues, validation and collection tests pass. |
| 4a–4b, 4d–4g | Yes | Instrument suite: **124 passed**. |
| 4c | **No** | New relative-`fromlist` counterexample below. |
| 5–8 | Yes | Population, closure, runner and reporting tests pass; real trial classifications match. |
| 9 | Yes, at `33a3f6e` | Exact saved-output match: **20 COMPLETE, 2 COMPLETE-RED-BY-MUTATION, 1 PARTIAL, 1 NOT-PLANNED, 1 INCOMPLETE, 0 UNKNOWN**; exit 1. |
| 10 | Yes | All nine mutations caught: (a)–(h) INCOMPLETE, (i) UNKNOWN. [Replay](/tmp/pr8c-round5-mutations.txt). |
| 11 | Yes | Required documents distinguish shipped tooling from post-PR-9 completion. |
| 12 | Partially verified | Ruff, formatting and mypy pass. Full suite: **8,716 passed, 230 skipped, 13 xfailed, 8 deselected, 1 failed**—the disclosed version-metadata mismatch. [Output](/tmp/pr8c-round5-full-suite.txt). |

**Fact check**

The archived round-4 report was assessed as a historical report about `cbbd5bd`.

| # | Claim (short) | Result | Evidence | Correct statement if not TRUE |
|---|---|---|---|---|
| 1 | 124 current tests; two new cases fail against `cbbd5bd` | TRUE | Executed current suite and historical instrument substitution. | — |
| 2 | Both round-4 literal-import examples are refused | TRUE | Real pytest probes return INCOMPLETE. | — |
| 3 | Expanded guard covers the stated literal `level`/`fromlist` boundary | **PARTLY** | Relative package plus literal `fromlist` bypasses it. | Explicit absolute-package examples work; relative package/fromlist combinations remain incomplete. |
| 4 | §5’s `pkg = "lizyml.other"` example is detected | TRUE | Executed probe returns the reference and INCOMPLETE. | — |
| 5 | Item 12: 19 rows, 13 before commits, zero references | TRUE | Fresh staging output exactly matches `pr8c_round4_staging.txt`. [Replay](/tmp/pr8c-round5-staging.txt). | — |
| 6 | Round-4 and round-3 staging row lines agree | TRUE | Compared saved rows and fresh replay. | — |
| 7 | 25 staged helpers; existing helpers and unstaged `conftest.py` identical | TRUE | Fresh staging replay reports no differing existing files. | — |
| 8 | #262 has aggregate p2; #265 skips p2 | TRUE | Executed trial and disposition logic agree with corrected §5. | — |
| 9 | Historical test counts: 108 previously; 122 with 12 effective RED cases | TRUE | Reproduced **12 failed / 110 passed** and **108 collected**. [Evidence](/tmp/pr8c-round5-historical-tests.txt). | — |
| 10 | Historical five-seam measurements and #288’s pre-fix blindness | TRUE | Exact saved-output match; current provider-fixed mutation adds **0 pre-fix / 1 shipped** failures. [Replay](/tmp/pr8c-round5-seams.txt). | — |
| 11 | #288 reports missing coverage; `22b11b3` is test-only | TRUE | Live issue and commit diff inspected; commit changes tests and acceptance documentation. | — |
| 12 | Mutation counts, trial split and nine mutation refusals | TRUE | Fresh trial and mutation outputs match their saved evidence bodies. | — |
| 13 | 22 affirmative pinned closure comments | TRUE | Re-read all 22 live comments; trial verifies association and timing. | — |
| 14 | Archive correction, population growth and documentation scope | TRUE | No archived manifest in available Git history; shipped manifest tracked; derivation reproduces **92 → 110**; no production-code diff. | — |
| 15 | Archived trailing whitespace was removed | TRUE | `git diff --check origin/develop...HEAD` passes. | — |

**Staging-guard boundary: open**

An executed IN-kind counterexample in `lizyml/a.py`:

```python
try:
    Y = __import__('other', globals(), None, ['new'], 1).new.Y
except (ImportError, AttributeError):
    Y = 0
```

With existing `lizyml/other/__init__.py`, after-only `lizyml/other/new.py` containing `Y = 1`, an after implementation containing `Y = 0`, and a test asserting `Y == 0`:

| Execution | Result |
|---|---|
| Guard references | `[]` |
| Original before | 1 passed |
| Before with staged files | 1 failed |
| After | 1 passed |
| Instrument verdict | **COMPLETE** |

The module names, `fromlist` element and import level are literals. No name is constructed or passed through a variable, so this falls within the stated boundary. Pytest execution was real; GitHub metadata was scripted valid metadata.

**Blocking findings**

1. **B1/B2/B3 — Relative package plus literal `fromlist` still manufactures false RED.** [The combination logic](/home/rem/repos/LizyML/docs/audits/2026-09-defect-discovery/instruments/phase3_gap.py:331) produces `other.new`, while separately resolving the literals produces `lizyml.other` and `lizyml.new`. It never produces the actual imported module, `lizyml.other.new`. Resolve the package relative to the importing file before combining its `fromlist`, or conservatively refuse staging. Add the executed counterexample as regression coverage. The broader coverage claim is PARTLY true and therefore blocking under the review contract.

**Non-blocking**

1. No real manifest row was over-refused.
2. The sole full-suite failure is the disclosed installed-metadata/version mismatch.
3. No additional blocker was found outside the staging guard.

**Not verified**

- Hosted CI, pushing, PR creation and merging remain pending.
- Criterion 9 must still be rerun at the develop head immediately before merge.
- Historical elapsed-time claims were not independently certified.

Progress: **4/4 review milestones complete — plan v1.** Acceptance remains blocked by finding 1.