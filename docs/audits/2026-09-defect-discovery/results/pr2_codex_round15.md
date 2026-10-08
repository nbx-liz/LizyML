# PR 2 — Codex review round 15 (2026-09-08)

Unscoped, head `a989f0b`, full suite 2529 passed at the time of review.

```
VERDICT: REQUEST_CHANGES
```

Two blocking findings and one non-blocking, all reproduced here before anything
was changed. **The round-15 prompt asked the reviewer to attack the two new
executable declarations directly, and it did: finding 2 falsifies one of the
grid's own `n/a` rationales.** That is the declaration working as intended — a
table that can be shown wrong is worth more than prose that cannot.

## Finding 1 — a tuned early-stopping setting is invisible to the conflict gate (DC1)

`check_training_managed_overrides` decided whether to claim
`early_stopping_round` by reading `cfg.training.early_stopping.enabled`. But
`_build_train_components` takes the patience from
`best_training_params["early_stopping_rounds"]` **when a tuning result supplies
it, whether or not the config enables early stopping**. So a study can switch
early stopping on for a config that disables it, and the gate did not see that.

Reproduced at head — config disabled, a `category: training` dimension tuning
the patience to 2, and a `fit(params=)` override of 10:

```
tuned training params: {'early_stopping_rounds': 2, 'validation_ratio': 0.2}
(override, callback rounds, iterations): [(10, [2], 4), (10, [2], 4), (10, [2], 4)]
```

The override reached `lgb.train` on every call and every booster stopped at the
tuned 2. Same class as round 13's finding 3, one activation source over.

**Fix.** `effective_early_stopping_rounds(cfg, training_overrides)` is now the
single definition of "is early stopping on, and at what patience", used by
`_build_train_components` **and** by the gate. Two readings of that question is
precisely what the defect was, so the repair is to have one.

## Finding 2 — a restored `best_model_params` escapes the same-layer refusal (DC1 + DC4 + DC5)

`overlay_params` drops competing spellings from the layer it overlays and keeps
whatever the **overlay itself** carries. `_merge_params` never checked the
overlay's own internal duplicates.

Reproduced at head, with a restored tuning result:

```
best_model_params = {"learning_rate": 0.1, "eta": 0.8}
  at lgb.train: [(0.1, 0.8), (0.1, 0.8), (0.1, 0.8)]
  booster:      [learning_rate: 0.1]
```

Both spellings reached LightGBM; it kept the canonical one. The reviewer also
reproduced it through a real `export()` / `load()` round trip, and was careful
to say what it was **not** claiming: that current `tune()` produces such a
result. It does not — `check_duplicate_space_dimensions` prevents two dimensions
naming one parameter. The population is artifacts written before this PR.

**And the grid said this was fine.** The cell
`tuning best_model_params × check_duplicate_identities` read
`"n/a: overlaid by identity into a checked dict"`. That rationale is false: the
overlay is checked against the layer below it, not against itself. The cell is
now `wired`, with an executed input, and the harness that requires every `wired`
cell to have one is what keeps it honest.

**Fix.** `check_duplicate_identities` on `best_model_params` before the overlay.
`load()` still reads such an artifact — an artifact is the record of a fit that
happened, and refusing to read it helps nobody. The refusal belongs on the
re-fit.

## Non-blocking — `params_table()` under-reports after an alias override

`params_summary` read a hardcoded list of canonical names out of the booster
dict by literal spelling. Reproduced:

```
fit(params={"learning_rate": 0.5})  ->  table lists learning_rate: 0.5
fit(params={"eta": 0.5})            ->  table lists neither name
                                        (the booster trained at 0.5 either way)
```

**Fixed rather than deferred**, though the reviewer marked it non-blocking and
correctly bounded it as "reporting, not the trained value". It misreports the
run on the one path this whole change exists to make work, and it is the same
literal-read construct that cost the export defect in round 13. The read scan
shipped last round did not catch it — its `_DICT_NAMES` does not include
`booster_params`, and the key is a loop variable rather than a literal — which
the reviewer named, and which is exactly the limit that scan's docstring
declares.

## Firing rates

```
Finding 1: 1 config in the shipped suite carries a `category: training`
           `early_stopping_rounds` dimension (`test_tuner_extended.py:38`), and
           it names no model-layer early-stopping parameter, so 0 pre-existing
           configs are refused. Verified: that test still passes.
Finding 2: unmeasurable from this repository (the population is artifacts
           written by earlier versions, which it does not hold). Bounded
           instead: `tune()` cannot now produce a duplicate-spelling
           `best_model_params`, because two dimensions naming one parameter are
           refused before the study starts, so no newly written artifact can
           trip this refusal.
Non-blocking: no refusal added; a report that was empty is now populated.
```

## What the reviewer checked and found clean

- **Inspected exception tracebacks for all 12 `_CELL_INPUTS` fixtures** and
  confirmed each reached its named checker rather than failing for another
  reason. That is the check this context could not perform on its own grid.
- Executed a real `export()` / `load()` round trip with RAM-backed artifact I/O:
  re-fitting without an override restored the config's `learning_rate=0.001`,
  and a fresh `eta=0.7` override reached all three training calls. **This path
  had not been executed by a reviewer since round 7.**
- Executed `export_code()` **and the generated `train_lgbm()`** for `metric`,
  `metrics` and `metric_types`: each preserved the Brier evaluation and trained
  a booster carrying `learning_rate=0.5`.
- 400 selected tests passed.

Its stated bounds: disk I/O was replaced with memory-backed operations, so
filesystem behaviour was not verified; it did not re-run the full suite, lint or
mypy. Those are run here — **2533 passed**, `ruff`, `ruff format --check`,
`mypy` clean.

## The record

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3, 1, 2.**

Fifteen rounds, no `APPROVE`. The two findings are both on the deliverable path
and neither is in code round 14 wrote — finding 1 is in round 13's check
(a second defect in that function, from a different direction) and finding 2 is
in a merge that predates the PR, exposed by a declaration this PR added.
