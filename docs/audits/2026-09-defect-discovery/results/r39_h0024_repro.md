# H-0024: two decided contracts that were never implemented

Batch B raised both as `blocking`. Both claim runtime behaviour, so both were
run rather than read. Scripts are in the session scratchpad
(`repro_direction.py`, `repro_merge.py`); each prints its own verdict.

These are **not** BLUEPRINT omissions. H-0024 decided a contract, BLUEPRINT
states a different one, and the implementation follows neither reliably. A
BLUEPRINT edit alone would ratify the current behaviour without anyone deciding
to.

---

## 1. Tuning minimizes a greater-is-better metric

**Decided** — `HISTORY.md:1592-1596`, the H-0024 table:

| task | metric | direction |
|---|---|---|
| regression | `evaluation.metrics[0]` or `rmse` | `minimize` |
| binary | `evaluation.metrics[0]` or `auc` | follows the metric's `greater_is_better` |
| multiclass | `evaluation.metrics[0]` or `logloss` | follows the metric's `greater_is_better` |

**Implemented** — nothing derives it. `grep -rn "direction" --include="*.py"
lizyml/` returns only `config/schema.py:498` (the config field, default
`"minimize"`), `tuning/tuner.py` (which receives it), `plots/tuning.py`,
`persistence/exporter.py` and `core/types/tuning_result.py` (which record it).
`core/_model_tuning.py:495` passes `optuna_cfg.direction` straight through, and
the objective at `:456-457` returns the raw metric value with no negation.

**Observed** — binary task, `evaluation.metrics: ["auc"]`, `tuning.optuna` left
at its defaults:

```
metric            : auc
direction         : minimize
trial AUCs        : [0.9341, 0.9362, 0.9362, 0.9367, 0.9367, 0.9465]
best_score chosen : 0.9341     <- the minimum
```

Tuning selected the **worst** AUC of the six trials it ran, silently. The same
holds for any greater-is-better metric a user puts first: `auc_pr`, `f1`,
`accuracy`, `precision_at_k`, `r2`.

**Why no test catches it** — `tests/_helpers.py:239-241` hardcodes
`"direction": "minimize"` in every tuning config it builds, and the default
first metric is minimize-correct for all three tasks
(`_model_metrics.py:19-23`: regression `rmse`, binary `logloss`, multiclass
`logloss`). The defect needs a user-chosen metric, which no fixture supplies.

Note also that H-0024's own table gives binary's default metric as `auc`, while
`_DEFAULT_METRICS["binary"]` is `["logloss", "auc"]` — first entry `logloss`.
A second, smaller drift on the same line.

---

## 2. A partial tuning space silently discards the default space

**Decided** — `HISTORY.md:1618`: 「デフォルト空間の個別次元を上書きしたい場合は、
`space` に該当キーを指定する（デフォルトとマージ）」 — naming selected keys merges
them over the remaining default dimensions.

**Specified** — `BLUEPRINT.md:741` says a supplied `space` is used as given.
BLUEPRINT and HISTORY disagree, and HISTORY is the later decision.

**Implemented** — `core/_model_tuning.py:329-331`:
```python
user_space = parse_space(cfg.tuning.optuna.space)
if user_space:
    space = user_space          # whole-space replacement
```

**Observed** — same data, two configs:

```
provider default_space (10): bagging_fraction, early_stopping_rounds,
    feature_fraction, learning_rate, max_depth, min_data_in_leaf_ratio,
    n_estimators, num_leaves_ratio, objective, validation_ratio
space={}   searched (10): ...all ten...
space={lr} searched  (1): learning_rate
```

Narrowing one dimension drops the other nine, with no warning. A user tightening
`learning_rate` gets a 1-dimensional search where they expected 10, and the run
looks normal.

There is a second consequence in the same block: `:337-339` sets
`fixed = provider.default_fixed_params(cfg.task) if used_default else {}`, so a
partial space also drops the per-task fixed parameters.

---

## Disposition

Both are Change Gate matters, not documentation. Each needs a proposal that
picks a side before anything is written down:

- **direction** — implement the metric-derived direction H-0024 decided, or
  supersede it and keep the user-configured field. If the field stays, a
  greater-is-better metric under `direction: "minimize"` should refuse or warn;
  silently optimizing backwards is the DC1 shape.
- **space merge** — implement the merge H-0024 decided, or supersede it with
  replacement. If replacement stays, a partial space that drops nine dimensions
  and the fixed params should say so.

Either way the fix carries a regression test, per the global rule on
user-observable defects.

**Defect-class: DC4** for both — the decided mechanism was never wired; nothing
reads the metric's `greater_is_better` and nothing merges the space. The
direction case additionally presents as DC1 at runtime: the backwards search
reports a normal result.
