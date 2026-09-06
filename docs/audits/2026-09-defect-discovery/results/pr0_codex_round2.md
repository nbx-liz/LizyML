# PR 0 — Codex review, round 2 (2026-09-07)

Round 1 is at `pr0_codex_round1.md`. Before writing this round's prompt, an
absolute loop monitor ran in a fresh context and returned `DELIVERABLE-FOCUSED`
with recommendation `continue`; the main context's disposition, including the one
scope growth the monitor named, is recorded in the run's workflow notes.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

### Findings

- **blocking** — `tests/test_docs/test_declared_versions.py:63`: the grammar
  accepts a numeric prefix within an invalid superstring. In a scratch copy,
  a declaration was changed to `format_version=2bogus` while seven matching
  sites were retained, then:

  ```
  PYTHONPATH=/home/rem/repos/LizyML .venv/bin/python -m pytest \
      -p no:cacheprovider /tmp/lizyml-review-r2-doc/tests/test_docs/test_declared_versions.py -q
  ```

  Result: **11 passed**. Because `(\d+)` has no trailing boundary, the gate reads
  the malformed declaration as version `2` and reports clean. This violates the
  required DC2 superstring check, so the drift gate cannot detect everything it
  claims.

### Checked and clean

- The round-1 production defect is fixed. The `build_inner_valid` production
  factory returned `BlockedGroupInnerValid(task="regression")`; six rows across
  three groups produced train `[0,1,2,3,4]`, valid `[5]`, and the `TimeHoldout`
  warning.
- Reverting the regression dispatch in a scratch copy made
  `TestRegressionFallbackIsTimeOrdered` red: **1 failed, 2 passed**. Replacing the
  complete source with the base version also made the group red.
- Making an existing scratch `ARCHITECTURE.md` contribute no version sites
  correctly failed both `MUST_CONTRIBUTE` and `MIN_SITES`.
- Changed tests passed: **38 passed**. Neighbouring inner-validation and
  blocked-group tests passed: **80 passed**.
- The five §10.3.3 signatures, split calculations, gap behaviour, error
  conditions, task-dependent fallback, and direct stratification refusal match
  the implementation.
- §§10.3.1, 10.3.3 and 10.6.2 agree. Automatic resolution inherits the outer gap;
  explicit `time_holdout` retains `gap=0`.
- Classification fallback and the `y is None` tail-holdout path remain
  operational.
- H-0092 accurately records both behaviour changes and includes purpose, scope,
  compatibility, alternatives, and acceptance criteria.
- The `task` branch is ordinary dispatch under `change-gate.md`, not conditional
  activation requiring firing-rate evidence.
- Exclusion reasons and existence assertions are sound. No additional DC1 or
  DC3–DC7 defects were found.
- `git diff --check`, Ruff, and formatting checks passed. The repository remained
  unchanged.
- GitHub issues #265/#266 were inaccessible because `gh` was unauthenticated in
  the reviewer's sandbox; their locally archived scope and the governing HISTORY
  entries were checked instead.

---

## Disposition (main context)

The finding is upheld and was reproduced directly: the old pattern reads
`format_version=2bogus` as `2`.

The repair is not a tighter value pattern — a trailing boundary alone would make
the malformed declaration stop matching, which converts a wrong answer into a
silent omission (DC1: "couldn't parse → skip"). The grammar is closed instead,
with an exhaustive three-way classification of what the declaration position can
hold:

| Outcome | Grammar | Disposition |
|---|---|---|
| value site | name, separator, bare integer | checked against the code's constant |
| type annotation | mermaid `+config_version: int`; admitted spellings enumerated, and each asserted to be in use | out of scope by construction |
| unreadable | anything else — `2bogus`, `2.5`, a separator with no value, a word | **test failure**, never a skip |

The value token is the maximal run of word characters and dots, so mermaid's
closing quote in `format_version=2"` is punctuation and stops the token at `2`,
while `2bogus` and `2.5` are captured whole and rejected.

Verified against the counter-example and its neighbours:

```
'...format_version=2"]'      -> site value=2
'    +FORMAT_VERSION = 2'    -> site value=2
'    +config_version: int'   -> type-annotation 'int'
'format_version=2bogus'      -> REJECTED raw='2bogus'
'format_version=2.5'         -> REJECTED raw='2.5'
'format_version='            -> REJECTED raw=''
'format_version: two'        -> REJECTED raw='two'
'- `format_version` を必須にする。'  -> []   (not a declaration)

old grammar on 'format_version=2bogus' -> 2
```

`tests/test_docs`: **13 passed** (two new guards —
`test_no_declaration_is_unreadable` and `test_named_type_annotations_are_used`).
H-0092's acceptance criteria now state the accept/reject matrix.

Note that finding `+config_version: int` was a consequence of closing the
grammar, not of the counter-example: it sits in the same name-separator-token
position and had been passing through the old pattern unnoticed because it holds
no digits.
