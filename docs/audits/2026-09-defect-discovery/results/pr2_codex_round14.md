# PR 2 — Codex review round 14 (2026-09-08)

Unscoped, head `995de29`, full suite 2498 passed at the time of review.

```
VERDICT: REQUEST_CHANGES
```

**One blocking finding, marked `[P1]`** — the lowest count since round 5, and
the first round since then with a single finding. Reproduced here before
anything was changed.

The round-14 prompt asked the reviewer one question it had not been asked
before: *where this diff claims a set, check whether it executed over that set.*
That question came from the rounds 12-13 monitor, which had just falsified a
set-claim for the second time in two rounds. **The single finding is an instance
of exactly that**, which is worth recording as a fact about how the prompt
worked rather than only about the code.

## Finding 1 — a study returns a result its own next step refuses (DC1 + DC4 + DC5)

`check_training_managed_overrides` runs inside `_merge_params`, and a comment
there claimed it "covers every input at once". That was true of the three inputs
that meet in `_merge_params` and **false of the fourth**: trial parameters
overlay afterwards, inside the tune objective.

So a `category: model` search dimension naming `seed` or `early_stopping_round`
was accepted, sampled, and trained on — and the `fit()` that follows then
refused the `best_model_params` the study had just produced.

Reproduced at head over **all seven spellings of both entries**:

```
random_seed            tuned 2 boosters, best={'random_seed': 12}            then fit REFUSED
random_state           tuned 2 boosters, best={'random_state': 12}           then fit REFUSED
seed                   tuned 2 boosters, best={'seed': 12}                   then fit REFUSED
early_stopping         tuned 2 boosters, best={'early_stopping': 12}         then fit REFUSED
early_stopping_round   tuned 2 boosters, best={'early_stopping_round': 12}   then fit REFUSED
early_stopping_rounds  tuned 2 boosters, best={'early_stopping_rounds': 12}  then fit REFUSED
n_iter_no_change       tuned 2 boosters, best={'n_iter_no_change': 12}       then fit REFUSED
```

Every one trained real boosters first. The defect is not that the refusal was
missing — it is that it arrived after the study had finished.

**Fix.** `check_training_managed_space` runs before the study starts, beside the
two space-level refusals already there (`check_param_names` and
`check_duplicate_space_dimensions`). The space is a layer like the others, and
this is the same wiring gap round 11 found for `check_duplicate_identities` and
the decision-8 addendum found for the space itself — third instance of one shape.

After the fix, all seven are refused with `train_calls = 0`.

**The false claim is corrected where it was made**, not only in the fix: the
comment in `model.py` now says which three inputs the merged-dict check covers
and names the one it does not, with a pointer to where that one is checked.

```
Firing rate: 0/70 of pre-existing configs carrying a category:model search space
             (7/77 including this change's own regression test, which iterates
             the seven spellings)
```

## What the reviewer checked and found clean

Bounded, and it said how:

- `fit(params={"eta": 0.5})` executed for binary, multiclass and regression;
  every CV booster **and** the full-data refit booster carried
  `[learning_rate: 0.5]`.
- All seven training-managed spellings on `model.params` **and** `fit(params=)`:
  **14/14 refused before any training**. The finding is the tuning layer only.
- Export-parameter extraction from trained adapters for `metric`, `metrics` and
  `metric_types`: all three retained the Brier evaluation metadata — round 13's
  finding 1 confirmed fixed by an independent execution.
- 379 selected tests passed under its own constraints.

**What it explicitly did not establish**, in its own words: it did not re-run the
full suite, lint or mypy; and disk export, generated-project execution, and a
real `Model.load()` followed by fit were not executed under its read-only
constraint. Those are run here instead — full suite **2500 passed**, `ruff`,
`ruff format --check`, `mypy` clean.

## The shape of the record

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3, 1.**

Fourteen rounds, no `APPROVE`. What changed in this one is not only the count:
the single finding was found by a question the loop had not been asking until
the previous monitor supplied it, and the reviewer's clean section independently
re-executed a previous round's fix rather than taking it.
