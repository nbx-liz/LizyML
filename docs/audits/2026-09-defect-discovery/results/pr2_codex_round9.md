# PR 2 — Codex review, round 9 (scope-limited, 2026-09-08)

Scoped by the rounds 7-8 relational monitor (`results/pr2_monitor_round78.md`,
**`DRIFTING` / `redirect`**) to the round-8 remedies and the three repairs made
after it, with rounds 1-6 out of scope and no new region admitted. The monitor
added one constraint, carried into the prompt: **a finding in existing apparatus
is repaired in place or that apparatus is deleted — round 9 may not answer a
finding by adding a new module, generator or scanner.**

## Verdict

```
VERDICT: REQUEST_CHANGES
```

Three blocking findings at head `0e20a43`. All three reproduced here before being
fixed. **Two were in the same two constructs round 8 found** — the value helper's
fallback and the exporting-test instrument.

### 1 — the printed-form fallback handed the decision back to the caller (DC5)

`lizyml/core/value_equality.py`, `_printed_forms_differ`. `repr` may return a
**subclass** of `str`, and a subclass may override the comparison. Comparing the
two texts with `!=` therefore ran caller code at the step that exists to escape
caller code:

```
values_differ(a, b) -> [False]     # a list, not a bool
facade  -> LizyMLError
adapter -> LizyMLError             # two identical printed forms, refused
```

The function's whole contract is a boolean decision, and both refusals on the
production path fired on a pair that printed the same.

### 2 — the exporting-population scan still lost a writer, silently (DC1, DC3, DC5)

`_writer_spellings`. Round 8 closed `writer = model.export` and
`getattr(model, "export")`. Round 9 opened `getattr(model, name)`:

```
spellings: set()          # the scan sees nothing
export called: True       # the test does export
selected: False           # so it is not in the population
population pin: PASS      # and nothing says so
```

The reviewer named the shape rather than the spelling: *adding another
recognised spelling would leave the underlying open-grammar hypothesis intact.*

### 3 — the probe confused the return path with the artifact (DC1, DC5)

`_probe` substituted the **methods**, which changed what they returned as well as
what they wrote. A target asserting only `model.export(path) == path` fails under
that substitution and was reported `noticed`, having inspected nothing:

```
return-path-only verdict: noticed
```

---

## The remedy

**Finding 1 — compare the characters, through `str.__eq__`.** That bypasses
subclass dispatch and cannot reach caller code. Anything other than a definite
"not equal" answers "the same", which is the direction the floor already takes.
One labelled case added; no behaviour axis added, because the fix is structural.

**Finding 2 — the scanner is deleted.** Not taught a fourth spelling. Python's
dispatch is an open grammar — after `getattr(model, name)` come
`operator.methodcaller`, `functools.partial` and `Model.__dict__` — so the claim
"every test that exports" cannot be delivered by any AST scan, and each round
refuted one more spelling of a hypothesis that was never closable. Both the
monitor (*"or that apparatus is deleted"*) and the reviewer (*"prefer deleting
the scanner's completeness claim"*) named this option.

In its place, `ARTIFACT_TESTS` names the four tests, says in the same breath that
it is hand-maintained, and states what that costs: **a new exporting test must be
added by hand, and nothing detects a failure to do so.** What is checked is what
is true — that every name exists, is unique, and takes the fixture this
instrument supplies. A bounded claim honestly stated cannot be reproduced
against; the unbounded one was reproduced against three times.

Removed with it: `_tests_that_export`, `_writer_spellings`, `WRITER_SPELLINGS`
and its eight-case grammar test, and the alias-refusal test. Their subject no
longer exists — the same reasoning that removed `_defines_own_truth` in round 8.

**Finding 3 — substitute the writing, not the method.** `_probe` now patches
`lizyml.persistence.exporter.export` and `lizyml.codegen.generator.generate_code`
beneath the methods, so `Model.export` and `Model.export_code` run for real,
resolve their paths and return them exactly as they would. The substitution
differs from the real thing in **nothing but the writing**, which is what makes
"failed after the writer was reached" evidence at all — and that inference is
now written in `_probe`'s docstring as a limit rather than left implicit. A
fourth synthetic target, asserting only on the returned path, pins the `green`
verdict the reviewer's reproduction demanded.

RED verified per finding: reverting the fallback to `!=` reddens the new case in
both directions; substituting the methods again reddens the probe test; the
reviewer's own three scripts now report `False`/`bool`, both callers accepting,
the scanner gone, and `green`.

## Checked and clean (round 9, from the reviewer)

- DC1: the repaired `AWKWARD_VALUES` consumers executed for all six factories,
  12 invocations; non-singleton pairs are distinct; the summarisation premise
  and case execute.
- DC2: all eight declared writer spellings passed their classifier assertions,
  including the docstring and longer-identifier negatives.
- DC3: the writer derivation returns exactly `export` and `export_code`, both
  exist on `Model`, and the 30-case cross product and its population checks pass.

Its stated bounds, kept: finding 1 concerns legal custom `__repr__` results, not
numpy or pandas representations; finding 2's injected source existed only in
memory and does not establish that one of today's four export tests is omitted;
finding 3 is a false positive in the instrument, not a defect in production
export.

## State handed to the maintainer

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3**. Nineteen findings, every one
reproduced and every one fixed.

Round 9's apparatus share was **2 of 3**, and both apparatus findings were in the
constructs round 8 had already found. The remedy this time **deleted** one of
them: the second contraction in two rounds, after round 8 removed the elementwise
step. Both removals took out a claim that no implementation could keep — what
iterating an arbitrary object yields, and what every spelling of a method call
looks like.

Full suite **2402 passed**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean. The rounds 8-9 relational monitor runs before round 10.
