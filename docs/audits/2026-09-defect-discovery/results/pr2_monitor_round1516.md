# PR 2 — relational monitor, rounds 15-16 (2026-09-08)

Spawned before round 17, at head `200f571`.

```
VERDICT: DRIFTING
RECOMMENDATION: escalate
```

**The first `DRIFTING` and the first `escalate` of the run.** Four monitors preceded
it, three returning `redirect` and all four `CONVERGING`.

## What it measured

`git diff --numstat`, production = `lizyml/`, tests = `tests/`:

| range | production | tests | other |
|---|---:|---:|---:|
| `92e3d51..bca3844` (round 15) | +250/−26 | +881/−0 | +1488/−30 |
| `bca3844..200f571` (round 16) | +134/−35 | +316/−0 | +248/−1 |
| `origin/develop..200f571` (whole change) | +1486/−47 | +4098/−36 | +5677/−3 |

It stated the bound on its own numbers: these are interval totals, not attribution to
a round's findings alone, and "other" includes shipped audit instruments, not only
prose.

## Its reasoning

> The decisive failure is relational: enumerating four readers caught inconsistent
> sources, but changing those sources introduced lifecycle inconsistency. **The run's
> remedy worked as discovery and failed as prevention.**

It weighted that above the round count, explicitly: sixteen rounds without `APPROVE` is
supporting context, and the stronger evidence is that *the predicted authorship failure
happened anyway, after the prescribed probe was executed.* The rounds 14-15 monitor
named the risk, the probe ran, the probe closed one half and missed the half the fix
itself introduced.

Its second point is that rounds 15-16 expanded obligations around **describing** the
fitted model and around **exceptional** equality inputs, which are connected to the
touched paths but are not another increment in ordinary override forwarding.

It credited the scope restraint of filing #281 and #282 rather than absorbing them.

## The main context's reconciliation

`policy:main-context-ownership` — a monitor's output is a finding to reconcile, never a
verdict to adopt. Three parts, and they are answered differently.

**1. The prevention failure: adopted without qualification.** This is correct and is
the sharpest thing any monitor has said about this run. Probe-before-round has been
treated here as *the* remedy for the universal-declaration failure mode, and round 16
is the case where it ran and did not prevent the defect — because a probe written by
the context that wrote the fix inherits that fix's assumption about which lifecycle
matters. Recorded in H-0094 decision 13 in that form.

**2. "Not another increment in ordinary override forwarding": partly declined, on the
record.** `export_code` generates a project meant to reproduce the training; a project
that trains a different model is the deliverable failing, not apparatus around it. And
round 16's finding 2 lives in the duplicate-identity comparison that every
`fit(params=)` call passes through — it made `fit()` raise where it had trained. Both
findings were on the shipped path. What the monitor is right about is the *shape*: two
consecutive rounds have found their defects in the reporting and comparison periphery
of the parameter path rather than in the parameter path itself.

**3. `escalate`: surfaced, not blocking.** The standing instruction is explicit and
persisted — run PR 2 to PR 1's standard, until `APPROVE`, on the maintainer's judgment
that the absence of `APPROVE` is itself evidence that problems remain in the fix code.
Rounds 11-16 have confirmed that premise by execution every time. The maintainer also
rescinded the pre-registered stop condition **before** round 16 and said the verdict is
decided on the code's merits. So this monitor's `escalate` does not stop the loop by
itself; it is reported to the maintainer, who can redirect, while the work continues.

## Its enumeration, adopted

> Four values (`early_stopping_rounds`, `validation_ratio`, `seed`, `num_boost_round`)
> × five declared lifecycles × two reporting surfaces = **40 cells**. Execute each
> lifecycle, capture what training actually consumed, and compare both reports; execute
> the generated training where the cell claims reproduction.

This is decision 13's own declared set, held to execution rather than to a table — the
same move that has twice turned a named surface into a fix instead of a finding. It
also stated its own bound: it checks decision 13's set, not every possible lifecycle,
and it declined to make any additional defect claim or to call an unexecuted candidate
dead.

## The enumeration, executed

`instruments/report_lifecycle_grid.py`, shipped so the table cannot go stale in
prose. Each cell compares a surface reading against **what the training consumed**,
captured while it ran — the inner-validation ratio from the factory that built the
strategy, and `seed` / `num_boost_round` from what reached `lgb.train`.

```
cells: 48 (40 from decision 13, 8 for the bound issue #281 names)
  agrees:      40
  known-bound:  2
  n/a:          6
```

No cell disagrees. The six `n/a` are `seed` × `params_table` across the lifecycles:
`params_summary` reports booster parameters by identity and `seed` is not among the
names it lists, so the table has no row to compare — a gap in the table, not a
disagreement, and stated rather than passed over.

**The grid grew by one lifecycle, and the reason is the point.** Decision 13's five
do not contain `tune -> fit -> export -> load`, which is the only lifecycle its own
stated bound bites in — so the instrument's `known-bound` branch was a branch no
input reached, the DC6 shape, inside the instrument whose job is to catch exactly
that. The sixth lifecycle was added and an assertion now fails the run if that
branch stops firing. Measured there: the run used `0.45`, both surfaces report
`0.2`, which is #281 and nothing new.

## The half the grid could not reach, executed separately

The monitor also asked for the **generated training** to be executed where a cell
claims reproduction. Reading the argument handed to `generate_code` is one step
short of the claim: the generated project is what a user runs. Nothing in the suite
executed the generated `train.py` — the existing subprocess tests run `predict.py`.

Executed after `tune -> fit`, configured patience 7 against the run's 2:

```
config.json early_stopping_rounds : 2
config.json validation_ratio      : 0.45
generated train.py                : holdout: 110 train / 90 valid
```

90 of 200 rows is the tuned `0.45`; the configured `0.2` would have held out 40.
Pinned by `test_the_generated_project_trains_at_the_patience_the_run_used`.

Full suite **2552 passed**; `ruff check .` and `ruff format --check .` clean.
