# PR 2 — Codex review, round 16 (2026-09-08)

Unscoped, at head `bca3844`. The first round a monitor's `redirect` recommended
leaving unscoped rather than narrowing.

```
VERDICT: REQUEST_CHANGES
```

Two blocking findings. Both reproduced here at that head before anything was
changed; both fixed; both RED-verified.

Blocking per round so far: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3, 1, 2, 2**.

## D7's authorship condition fired, on commit `be2795a`

Finding 1 lands in code the **previous round's fix wrote**. Stated plainly,
because the run's own standard says to and because softening exactly this once
before is what the rounds 13-14 monitor caught:

> **D7 authorship condition — a defect found in round N+1 inside the code round
> N's fix wrote — fired. The commit is `be2795a`, decision 12's fix.**

The rounds 14-15 monitor named this consequence *before* the round, and said
probing before the round is what prevents it. The probe found and closed one
half (`export_code` and `params_table` were reading the config alone). It did
not find the half that the fix itself introduced, which is the harder half by
construction: a probe run by the context that wrote the fix inherits the fix's
own assumption about which lifecycle matters.

## Finding 1 — the reports follow the latest tuning result, not the fitted model

`_model_tables.py` / `_model_persistence.py`. **DC3, DC5.**

Decision 12 unified four readers of "what patience did this run use?" by making
`params_table()` and `export_code()` recompute it from the config plus the
model's tuning result. `tune()` replaces that tuning result and **leaves the
fitted adapters alone**, so after `fit -> tune` both surfaces answer for a model
that was never trained:

```
the fitted adapter trained with : 7
params_table before tune        : 7
same fitted adapter after tune  : True
params_table after tune         : 2
export_code after tune          : 2
```

`export_code` generates a project meant to reproduce the training. Decision 12
closed that exact defect coming from the config; this reopened it from the
other side.

### The set, enumerated and executed before the fix

The reviewer's direction was explicit — *"resolve it from retained training
state or the trained adapter through the provider"* — and the run's central
lesson says the set to execute over is **the lifecycles**, not the one ordering
that surfaced the defect. Every value `export_code` passes from config and
every row `params_table` builds from config, under five lifecycles (`fit`,
`tune -> fit`, `fit -> tune -> report`, `fit -> export -> load`,
`fit -> tune -> export -> load`):

| value | before | source now |
|---|---|---|
| `early_stopping_rounds` | wrong in lifecycles 3 and 5b | **the trained adapter**, via `ExportParams` |
| `validation_ratio` | **wrong in lifecycle 2 — present in `develop`, not introduced here** | the retained overlay, via `FitState.applied_training_params` |
| `seed` | correct | config alone; the trainer reads `cfg.training.seed` and nothing else |
| `num_boost_round` | correct | already adapter-sourced; confirmed by execution, not assumed |

**The patience comes from the adapter**, because that is the only surface that
survives both a later `tune()` and a `load()`. Measured: an artifact exported
after `fit -> tune` carries a tuning result no fit consumed, so a fix routed
through the tuning result would have been wrong in lifecycle 5b — the one
lifecycle that separates the two candidate routes. `ExportParams` gains
`early_stopping_rounds` with **no default**: a defaulted `None` would make "the
provider did not set it" and "early stopping was off" the same value, which is
the DC1 shape.

**The ratio comes from retained training state**, because the adapter does not
record it. Executed rather than inferred — the trainer's inner-validation
factory was called with the tuned ratio while both surfaces reported the
configured one:

```
inner-valid factory called with : [0.45]
params_table validation_ratio   : 0.2
```

`tuned_validation_ratio()` is now the one definition, with three readers: the
trainer chooses its inner-validation strategy from it, and the two reporting
surfaces report it.

**The bound, stated rather than left to be found.** After `load()` the retained
overlay is empty — the artifact records the tuning result but not which fit
consumed it — so a loaded model reports the configured ratio. Closing that
needs a `metadata.json` key, which is a Change Gate item and not this PR's
business. It is pinned by a test and filed.

## Finding 2 — the comma-form step can raise, against the module's declared bound

`lizyml/core/value_equality.py::_comma_form_matches`. **DC5, DC7.**

`float(element)` caught `TypeError` and `ValueError` only. Reproduced with a
`float` subclass whose `__float__` raises `RuntimeError`:

```
{'learning_rate': 0.5}                 TRAINED True
{'eta': '0.5'}                         TRAINED True
{'learning_rate': 0.5, 'eta': '0.5'}   RuntimeError conversion unavailable
```

The module declares **"this function does not raise an `Exception`"** and
"every expression that touches a caller's value is inside a `try`". `str(element)`,
one line down, was not inside one at all. Both are now guarded;
`BaseException` still propagates on purpose.

**The durable half is the cell this came out of.** The existing generated
population compares an awkward value against another awkward value, so `text` is
never a `str` and the comma-form step returns `None` before it reaches an
element. **A string on one side and hostile elements on the other** was a cell
the cross product could not produce. `_ELEMENT_BEHAVIOURS` (`__float__` × 3,
`__str__` × 2) adds it, with the `float`-subclass route kept as its own case
because it reaches the same expression differently.

## Filed, not fixed here

- **A loaded model's `validation_ratio`** — the bound above; needs a persistence
  format change. Filed as [#281](https://github.com/nbx-liz/LizyML/issues/281).
- **A `category: training` `seed` dimension is sampled and silently ignored.**
  Executed: the study accepted it, returned `{'seed': 123}` in
  `best_training_params`, and the trainer used `0`. Present in `develop`, and
  not on the `fit(params=)` path this PR exists to fix. Filed as
  [#282](https://github.com/nbx-liz/LizyML/issues/282).

## RED verification

Each fix reverted independently, against both regression files:

```
reverted F2 (comma-form guard)              -> 6 failed, 320 passed
reverted F1 (patience source)               -> 3 failed, 323 passed
reverted F1 twin (validation_ratio source)  -> 2 failed, 324 passed
restored control run                        -> 326 passed
```

Full suite **2551 passed**; `ruff check .`, `ruff format --check .`, `mypy lizyml/` clean.
