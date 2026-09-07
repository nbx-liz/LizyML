# PR 2 — relational monitor, rounds 5-6 (2026-09-07)

Read-only, fresh context, spawned after the maintainer directed a further round
for the second time. **The first non-`CONVERGING` verdict in this run.**

```
VERDICT: DRIFTING
RECOMMENDATION: redirect
```

## Why, in its own terms

It rejected the series the main context had been reporting:

| R | blocking | self-authored |
|---|---|---|
| 1 | 1 | 0 (shipped) |
| 2 | 1 | 1 (round-1 code) |
| 3 | 2 | 0 (shipped; one deferred as #280) |
| 4 | 2 | 1 (round-3 code) + **the last shipped defect** |
| 5 | 1 | 1 (round-4 code) |
| 6 | 3 | 3 (all in the pre-round-6 fix) |

> **6 of 10 findings were written by the loop.** Self-authorship did not begin at
> round 5 — it appears at rounds 2 and 4 — and no round since **round 4** has
> found a defect that predates the PR. The counts 1,1,2,2,1,3 are the wrong
> series; this row is the trajectory.

Production lines: rounds 1-4 **478**, round 5 **64**, round 6 **66**. Since
round 4, `model.py`, `_merge_params` and `_build_params` — the declared
deliverable — are **untouched**; every production change is inside a 90-line
helper, with ~617 lines of tests and verdict prose around 130 lines of code.
"That is the exact shape `policy:loop-monitor` exists to catch."

**Is the round-6 remedy different in kind?** No.

> Round 5: "compare by `==`, else `repr`." Round 6: "length, then `==`, then
> elementwise, then `repr`." Both enumerate the value shapes someone thought of
> and terminate in `repr`.

And it demonstrated generation 4 already existed at HEAD, read-only:
`values_differ` raised on an object whose `__eq__` and `__repr__` both fail,
"present at HEAD, unfound … DC1 over an open grammar — any Python object — and
six rounds are the until-dry loop the defect-classes file warns to close before
entering."

## What it pointed at instead

> One shipped-surface axis: **persistence and export of an overridden fit**.
> `grep -n "save\|load\|export"` over the override test file returns **zero**
> hits across its 41 tests.

And on the helper: no review round can exhaust an open value domain — "each
round buys one more exotic value". What closes it is a **bound**, not a round.

## Main context's disposition — `redirect`, adopted

Three things done before round 7, all measured rather than argued:

**1. The value domain is bounded by construction, not by another case.**
`values_differ` now checks **identity first** — a value compared with itself is
the same value, which is the comparison the callers make most often and the one
that was raising — and its floor answers "the same" when even the printed forms
fail. It answers "same" rather than "different" on purpose: the refusal exists
to catch a parameter written twice, and refusing a value nothing can analyse
would block a legitimate call. The function no longer raises for any input.

**2. Two more self-authored defects found before the round, not by it.**
`_length_or_none` caught only `TypeError`, so a `__len__` that raised anything
else propagated; and iterating a DataFrame comparison yields **column labels**,
all truthy, so two different frames reduced to "equal". Both measured, both
fixed, both now cases in the table.

**3. The unexamined axis was measured, and it is correct.** An override reaches
the exported booster and the generated project's `lgbm_params`; it does **not**
survive `Model.load()`, and a re-fit after loading trains on the config value —
which is exactly what "applies to this call only" means, asserted across the
artifact boundary for the first time. Four tests now pin it. `export_code`'s
`train.py` carries a fixed example line mentioning `learning_rate`, identical
with and without an override, so the assertions read the generated config
instead.

## The stop condition for round 7, adopted verbatim

> any finding inside code written in rounds 5-7 stops the loop regardless of
> severity; only a finding in code predating round 5 justifies continuing.

Two pre-registrations have already been overridden by the maintainer, so this
one is stated as what the *run* will do: if it trips, PR 2 goes back to the
maintainer with the loop closed from this side, and reopening it is a decision
only they can make.
