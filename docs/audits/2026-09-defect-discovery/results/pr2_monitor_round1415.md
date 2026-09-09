# PR 2 — relational monitor, rounds 14-15 (2026-09-08)

Spawned before round 16.

```
VERDICT: CONVERGING
RECOMMENDATION: redirect
```

The third consecutive `redirect`, and the most substantive monitor of the run.
Both of its enumerations were executed here before anything was acted on; one
was a defect and one was clean.

## What it measured, and the sentence that matters

Round 14 production +45/−2; round 15 production **+84/−12** across three files,
all on the declared path, all fixing reproduced defects. Periphery: round 15
tests +87/−1, docs +305/−27. It noted the 824-line declaration commit between
the rounds was the *previous* monitor's redirect, not a round finding.

Then it said the thing the verdict cannot:

> What the CONVERGING/DRIFTING binary misses, and is the actual reason fifteen
> rounds have produced no `APPROVE`: **the maker ships a universal declaration
> each round without executing it over its set, and the next round falsifies
> it.** Four consecutive now — the seam scan (r11-12), the value-equality class
> (r12-13), the grid cell (r15), and the one below, which I hold.

Adopted as the finding of record. It is a statement about this context's
practice, not about the code, and it is correct.

## Its first enumeration — a defect, executed and closed

Round 15's fix declared `effective_early_stopping_rounds` "the **single
definition**" and its docstring said the trainer and the gate "cannot disagree".
The monitor enumerated the readers of that question: **four members, and the
claim had been executed over two.**

| reader | before |
|---|---|
| `_model_factories.py` — the refusal | unified |
| `model.py` — the trainer | unified |
| `_model_persistence.py:239` — feeds `export_code` | **config alone** |
| `_model_tables.py:290` — feeds `params_table` | **config alone** |

Executed here, config patience 7 and a tuned patience of 2:

```
tuned patience  : 2
params_table    : 7
export_code     : 7
the run actually used: 2
```

**Worse than reporting.** `export_code` generates a project meant to reproduce
the training, and it was generating one that would train a **different model**.
Both readers now use the shared definition; all four agree.

The monitor also stated the consequence plainly rather than leaving it: the
claim is in commit `b737062`, so a round-16 finding there would fire D7's
authorship condition on round 15. Probing it before the round is what prevents
that, and is the precedent that worked at rounds 10-11 and 11-12.

## Its second enumeration — a named blind-spot class, executed and clean

> **A parameter that reaches `lgb.train` correctly, under one honoured name, and
> is then outranked by a channel that is not the params dict.** Every instrument
> here reads the dict. `adapter.py` also passes `num_boost_round=` as a keyword,
> the early-stopping callback, and `categorical_feature=` / `weight=` at Dataset
> construction. `TRAINING_MANAGED_PARAMS` holds two entries, so `num_iterations`
> and `categorical_feature` are channels with no refusal and no grid column.

It also said how to look: cross those channels against the alias table and
assert on **what the booster did**, not on `booster.params` — the dict being
exactly what would not show it.

Executed that way, and **clean**:

```
num_boost_round, all 7 spellings   -> booster grew 17 trees (asked 17, config 6)
                                      and the params dict was empty
categorical_feature, index form    -> [categorical_feature: 0] in the booster
categorical_feature, "name:" form  -> LightGBMError, loudly, naming the column
```

The `num_boost_round` result is rounds 1-2's fix still holding under every
spelling; the `name:` failure is loud, and loud is the acceptable half — the
class being hunted is silent defeat. Both are now pinned by tests so the class
cannot regress quietly, which is the durable half of a clean result.

It declined to claim a third instance (`lgb.Dataset` built without `params=`)
because `lgb.train` may push params into an unconstructed Dataset, which would
make it dead — and said so rather than asserting it. That restraint is why the
two claims it did make were both worth executing.

## Its recommendation, adopted

> `redirect` — execute the two finite enumerations before round 16, and leave
> round 16 **unscoped**, because probe-before-round has twice converted a
> monitor's named surface into a fix instead of a finding, while scoping a round
> has four times produced nothing.

Adopted in full, including the scoping part — which is the first time a
monitor's redirect has recommended leaving the round unscoped rather than
narrowing it, and it reaches that from the run's own record.

Full suite **2535 passed**.
