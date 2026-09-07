# PR 2 — Codex review, round 8 (scope-limited, 2026-09-08)

Rounds 1-7 are in the sibling files. The maintainer **rescinded the round-7 stop
condition** and directed the loop to run until the external reviewer returns
`APPROVE`, on the standard applied to PR 1, with the stated reasoning that not
obtaining `APPROVE` is itself evidence that real problems remain in the fix code.

The rounds 6-7 relational monitor (`results/pr2_monitor_round67.md`) returned
**`CONVERGING` / `redirect`** and scoped this round to the round-7 remedies plus
the two instruments it prescribed, with rounds 1-6 surfaces out of scope and no
new region admitted.

---

## Before the round

One self-found defect, fixed at `3136187`: `hasattr(element, "__bool__")`
**invokes** the attribute, so an element whose `__bool__` is a property that
raises escaped a function declared not to raise. Round 7's finding one step
further along the same path. Two corrections to the instruments themselves went
with it (`1d9d5f2`).

## Verdict

```
VERDICT: REQUEST_CHANGES
```

**Four blocking findings — the most of any round in this run.** All four
reproduced here before being fixed. Two are in the instruments the previous
monitor prescribed; one is a production correctness defect.

### 1 — a `DataFrame` with integer labels was called equal to a different one (DC1)

`lizyml/core/value_equality.py`. Iterating a `DataFrame` comparison yields its
**column labels**. Round 6 found this with string labels and the guard became
"the element must be a string". Before round 8 that guard became "the element's
type must define `__bool__`" — and integer labels define `__bool__`.

```
>>> a = pd.DataFrame({1: [10, 20]}); b = pd.DataFrame({1: [30, 40]})
>>> a.equals(b), values_differ(a, b), values_differ(b, a)
(False, False, False)
```

Two different values reported as the same, in both directions, on the production
path. The reviewer noted the shape precisely: *defining a truth method does not
establish that an iterated element represents equality.*

### 2 — the export instrument counted a failure that never reached a writer (DC1)

`tests/test_core/test_fit_params_override.py`. The blanket handler read *any*
exception as "the test noticed the missing artifact". With every target replaced
by one that raises during setup, the instrument passed although neither
substituted writer was called. Reproduced with the reviewer's own script:
`ACCEPTED failures before export`.

### 3 — a writer named under an alias left the population in silence (DC1)

`_calls_a_writer` matched only a direct attribute call, so a test containing
`writer = model.export; writer(path)` was not selected — and the population pin
stayed green. A test that exports and is reviewed by nothing is exactly what the
instrument exists to prevent.

### 4 — "does not raise" exceeded the handlers (DC5)

Every handler catches `Exception`, so a caller value raising a direct
`BaseException` subclass propagated. The generated population's `raises`
behaviours all use `RuntimeError`, so passing them established nothing about the
unrestricted claim.

---

## The remedy

**Finding 1 — the elementwise step is removed, not repaired.** It was the only
step whose correctness depended on what iterating an arbitrary object yields,
and every version of it was a hypothesis about object structure that the next
round refuted: strings in round 6, a raising `__bool__` property before round 8,
integer labels in round 8. Comparisons that cannot be a truth value are now
decided by the printed forms, which is what already handled every array case in
the table. `values_differ` is now five steps, each resting on a single protocol
call on the values themselves.

**The cost is stated rather than hidden**, in the docstring: two arrays holding
equal numbers under different dtypes print differently and are reported as
differing, so such a caller is refused — legibly, with both spellings and both
values named. Two arrays that print alike are reported as the same, and the
callers keep the first spelling written. That is the direction the floor already
chose, since this function feeds refusals.

Every one of the 104 cases in the file passed unchanged after the removal, so no
case in the table depended on the step.

**Finding 2 — the probe is a function with three verdicts, and each is tested.**
The target must pass **unpatched** first, and a failure under substitution counts
only if `export.called or export_code.called`. Extracted so the decision rule can
be exercised on synthetic targets rather than only on the four tests that happen
to be selected.

**Finding 3 — the writer grammar is closed and refuses what it cannot classify.**
`_writer_spellings` returns `call`, `bound attribute` or `getattr`; anything but
`call` is an assertion failure naming the test, not a quiet omission. Eight
labelled spellings pin it, including the two negatives (a writer in a docstring,
a longer identifier).

**Finding 4 — the declaration is bounded, not widened.** Catching
`BaseException` would swallow `KeyboardInterrupt` and make a hung comparison
uninterruptible. The docstring now says "does not raise an `Exception`" and says
why the rest propagates on purpose; a test pins that a `KeyboardInterrupt` from a
caller's `__eq__` reaches the caller.

RED verified per finding: the integer-label case reddens if the elementwise step
returns; the reviewer's setup-failure script now stops at the control run; both
alias spellings are refused by name; the `KeyboardInterrupt` case is asserted
positively.

## Checked and clean (round 8, from the reviewer)

- DC2: exact attribute-name matching; longer identifiers, comments and
  docstrings do not match as writer calls.
- DC3: all 30 generated combinations, the per-axis pin and the identity
  assertion execute and pass; the cross product is derived from `_BEHAVIOURS`.
- DC4: both instruments and the scoped assertions are reachable and were
  executed.
- DC6: the scanner selects four tests and the instrument runs; not inactive.
- DC7: no permanently unsatisfiable declaration in the tested population.
- Identity precedes every caller-controlled operation, and all 30 generated
  values compare as identical to themselves.

Its stated bounds: head `690f2f6`, confined to the module and the instruments;
no full-suite run; merge, forwarding, persistence and export correctness not
reopened; *passing generated cases establish behaviour for those fixtures, not
closure over arbitrary Python values.*

## State handed to the maintainer

Blocking findings per round: **1, 1, 2, 2, 1, 3, 2, 4**. Sixteen findings, every
one reproduced and every one fixed.

**Round 8 produced the most findings of any round**, and two of them were in the
instruments the rounds 6-7 monitor prescribed. The monitor's own prediction —
that "run until `APPROVE`" manufactures its own next finding while each remedy
ships a declaration verified by a hand-written table — held for an eighth round,
including for the apparatus built to end that pattern.

What is different this time is the direction of the largest remedy: finding 1 was
answered by **deleting** the step that three rounds had each patched, rather than
by adding a case to it. Full suite **2403 passed**; `ruff check .`,
`ruff format --check .`, `mypy lizyml/` clean.

The relational monitor for rounds 7-8 runs before round 9, and must be told the
`4` without softening.
