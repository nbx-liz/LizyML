# PR 2 — Codex review, round 6 (scope-limited, 2026-09-07)

Rounds 1-5 are in the sibling files. Round 5 met a pre-registered stop
condition; PR 2 went to the maintainer as **D7**, and the maintainer directed
**one further round with the target narrowed to the remedies** — the same choice
they made on PR 1's D5.

The rounds 4-5 monitor (`results/pr2_monitor_round45.md`) judged that round
warranted on its merits and pointed it at the **value domain** of the two
equality comparisons rather than at the identity logic five rounds had covered.
It found the array-valued regression itself, minutes after the maker had found
the same defect in self-review; that fix was made and disclosed in the round-6
prompt.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

Three blocking findings, **all in the pre-round-6 fix**, and all three in the
same place the monitor pointed: what a parameter's *value* can be.

### 1 — a library scalar was refused as a different value (DC7)

`np.float64(0.5) == 0.5` yields `np.bool_`, which is not a `bool` and is not
iterable, so it fell through to comparing printed forms —
`"np.float64(0.5)"` against `"0.5"` — and refused:

```
{'learning_rate': np.float64(0.5)}                TRAINED
{'eta': 0.5}                                      TRAINED
{'learning_rate': np.float64(0.5), 'eta': 0.5}    LizyMLError [CONFIG_INVALID]
```

This is round 5's finding again in a narrower form: a gate refusing valid input.

### 2 — broadcasting hid two sequences of different lengths (DC1)

`np.array([1., 1.]) == np.array([1.])` broadcasts to all-true, so the refusal
called them equal and one of the two values was chosen silently. An empty array
agreed with anything through a vacuous `all()`.

### 3 — the declared exception-safe fallback did not exist (DC5)

The comparison itself ran **outside** the handler, so an `__eq__` that raises
propagated, and the reduction caught only `TypeError` while a nested array
raises `ValueError`. H-0094 claimed the function never raises. It did.

---

## The remedy

`values_differ` now decides in a fixed order, and the order is the content:

1. **Length, when both values have one** — elementwise comparison broadcasts,
   and an empty sequence agrees with anything by a vacuous `all()`. Finding 2.
2. **The comparison as a truth value** — `bool(equal)`, which covers ordinary
   values *and* the library scalars whose result is not a `bool` but converts to
   one. Finding 1.
3. **Elementwise**, requiring every element to be equal, for array-like results
   that cannot convert to a single truth value.
4. **The printed forms**, for anything that raised on the way — the comparison
   included. Finding 3. A weaker answer than equality, and the only one both
   values always have.

`tests/test_core/test_value_equality.py` is a table of **25 inputs**, each
labelled by the property it is about rather than by its literal, asserted in
both directions (the two refusals compare in opposite orders, so an asymmetric
answer would refuse a call on one path and accept it on the other), plus a case
asserting the whole table raises nothing.

RED-verified per finding, and each mutation fails a different set: dropping the
length guard reddens 4 cells, dropping the truth step 11, moving the comparison
outside the handler 3.

Two judgements are recorded as cases rather than left implicit: a list and an
equal tuple are **different** values, because a caller who wrote both wrote two
things; and `nan` differs from itself, which the docstring states.

Codex also corrected a wording claim: the module's "no imports" is literally
false — it imports `__future__` and `typing`. Reworded to "depends on nothing
but the standard library", which is what Layer 0 requires.

---

## Checked and clean (round 6)

- 89 passed on the override file, including the real array-valued fit (141 after
  this remedy, across both files).
- Equal pandas Series and equal dicts compare equal; unequal strings differ.
- **Layer inspection**: the adapter's dependency on the foundation respects the
  5-layer DAG.
- The deferrals remain explicit in BLUEPRINT, HISTORY and the CHANGELOG.
- DC1–DC7 checked within scope; no DC2/DC3/DC4/DC6 finding, and no
  `OUT OF SCOPE` section.
- `git diff --check` clean; the reviewer changed nothing.
- Setup limitations reported as such, not as findings: the sandbox blocked a
  heredoc temp file, and `AGENTS.md` is a broken symlink.

## State handed to the maintainer

Blocking findings per round: **1, 1, 2, 2, 1, 3**. Ten findings, every one
reproduced before it was accepted, every one on the path
`fit(params=)` → `lgb.train`, and every one fixed.

The pre-registration for this round said `APPROVE` merges and anything else goes
to the maintainer, with no round 7 either way. It was not an `APPROVE`. The
three findings were fixed rather than left standing, and no round 7 was opened.

**The shape has changed, and it is worth stating plainly.** Rounds 1-4 found
defects that had shipped; rounds 5 and 6 found defects the previous remedy
wrote, and round 6's were all in a 40-line helper written to fix round 5's. The
subject has narrowed to "how do you compare two values", which is a smaller
question than the one the PR started on, but the loop is no longer finding
defects that predate it.

Full suite **2334 passed**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean. The decision is **D7** again, updated with this round.
