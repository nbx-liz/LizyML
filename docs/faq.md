# FAQ / Troubleshooting

---

## Training and Metrics

### Why do my OOF metrics differ from test metrics?

OOF (out-of-fold) metrics are computed on held-out validation rows from
cross-validation — they are a reliable estimate of generalization error on
the training distribution.

Test metrics are computed on a completely separate holdout that was never
seen during training or model selection. Differences are expected and normal:

- **OOF is optimistic** when the training distribution and test distribution
  differ (e.g. temporal drift, domain shift).
- **Test is noisier** for small holdout sets.
- **OOF coverage < 1.0** means some rows were excluded from OOF computation
  (see [What is oof_coverage?](#what-is-oof_coverage-and-when-is-it--10)).

If the gap is large and consistent, investigate distribution shift between
your training and test data.

---

### Why does `tune()` + `fit()` give different scores than `fit()` alone?

This was a known bug fixed in **v0.7.2–v0.7.3**.

In older versions, `tune()` used one data preparation path and `fit()` used
another, so the split seed and preprocessing were not identical. The best
params from `tune()` were also not always forwarded correctly to `fit()`.

After v0.7.3, `tune()` and `fit()` share the same `_build_train_components`
path and the tuning result is automatically applied on the next `fit()` call.
If you still see a discrepancy, ensure you are on v0.7.3 or later.

---

### When should I use `purged_time_series` vs `time_series`?

| Splitter | Use when |
|----------|----------|
| `time_series` | Observations are ordered by time but rows are independent (e.g. daily aggregates, no look-ahead risk from adjacent rows). |
| `purged_time_series` | Rows within a window of each other share information (e.g. rolling features, overlapping label windows). A purge gap is inserted between train and validation folds to prevent leakage. |

As a rule of thumb: if your features involve any rolling computation or if
the target at time _t_ depends on data at time _t+k_, use `purged_time_series`
and set the `gap` parameter to at least the maximum look-ahead in your features.

---

### My categorical column is being treated as numeric — why?

LizyML uses the DataFrame dtype to decide encoding:

- `object` or `pd.StringDtype` columns → categorical encoding.
- `pd.CategoricalDtype` columns → categorical encoding.
- Integer or float columns → numeric (even if the values happen to be 0/1/2).

Common causes:

1. **The column was read as integer.** Cast it explicitly:
   ```python
   df["my_col"] = df["my_col"].astype("category")
   ```
2. **You are listing it under `features.numeric` in config.** Move it to
   `features.categorical` or remove it from the explicit list and let
   auto-detection use the dtype.
3. **The column has mixed types that pandas resolved to float.** Fill `NaN`
   with a sentinel string before casting.

---

### How do I use a custom LightGBM objective?

Pass it through `model.params` in the config:

```yaml
model:
  type: lgbm
  params:
    objective: "binary"   # or any LightGBM built-in string
```

For a fully custom Python objective function, define it and pass it via
`fit(params={"objective": my_fn})`. LizyML forwards all `model.params`
values directly to LightGBM; no additional wrappers are needed.

Note that custom objectives may require a corresponding `feval` function for
evaluation during training. See `evaluation.metrics` in config and the
metric bridge documentation.

---

## OOF Coverage

### What is `oof_coverage` and when is it < 1.0?

`oof_coverage` is the fraction of training rows that received an OOF
prediction. It is exposed as:

- `evaluate()["raw"]["oof_coverage"]`
- `evaluate_table().attrs["oof_coverage"]`

Coverage is **< 1.0** when:

- **Time-series splitters with expanding windows:** the earliest folds have
  no corresponding validation fold, so those rows are never held out.
- **Purged gaps:** rows within the purge window adjacent to the validation
  fold boundary are excluded from OOF to prevent leakage.
- **Group-based splitters:** when all samples of a group appear in the
  training set across all folds (rare, but possible with very large groups
  and few splits).

OOF metrics are computed only on covered rows. Rows that are not covered
are not extrapolated or imputed.

---

## Errors

### `ValueError: Inner validation would consume all N sample(s)`

Inner validation (early stopping) requires a minimum number of samples to
form both a training split and a validation split. This error means the
training fold in one outer CV split is too small to satisfy the configured
`validation_ratio`.

Solutions:

- Increase the dataset size.
- Reduce `early_stopping.validation_ratio` (e.g. `0.1` instead of `0.2`).
- Reduce `split.n_splits` so each outer fold has more training rows.
- Disable early stopping (`early_stopping.enabled: false`) if sample count
  is genuinely too small for validation.

---

### `LizyMLError: CONFIG_INVALID`

The config is missing a required field, has an invalid value, or `tune()` was
called without a `tuning` section. Check `e.context` for the specific field:

```python
except LizyMLError as e:
    if e.code == ErrorCode.CONFIG_INVALID:
        print(e.context)   # {"field": "split.n_splits", ...}
```

---

### `LizyMLError: LEAKAGE_SUSPECTED`

LizyML detected a split or calibration condition that could indicate data
leakage. Common causes:

- The same row index appears in both the training and validation fold
  (can happen with non-unique indices).
- A time-based constraint was violated (validation timestamps overlap with
  training timestamps).
- Calibration tried to train on rows that are also in the OOF set.

Reset the DataFrame index to a unique RangeIndex before passing it to
`fit()` if you suspect index collisions.

---

### `LizyMLError: MODEL_NOT_FIT`

A method that requires a trained model was called before `fit()`. Call chain
must be: `Model(config)` → `fit(data)` → any diagnostic method.

After `Model.load(path)`, methods that need `analysis_context` (such as
`confusion_matrix()`, `importance(kind="shap")`, `residuals()`) will also
raise this error if the artifact was exported before `analysis_context`
support was added. Re-export with the current version to restore access.

---

## Calibration

### How does calibration avoid leakage?

LizyML's calibration uses the outer CV splits (the same splits used for OOF
prediction). The calibrator is fitted in a cross-fit loop:

1. For each outer fold, the calibrator is trained on OOF predictions from
   the **other** folds (hold-one-out).
2. The final calibrator (`c_final`) is trained on **all** OOF raw scores
   after CV completes.

This means no row's calibrated probability is computed using a calibrator
that was trained on that row's own label. See BLUEPRINT §10.5 for the
full invariant specification.

`CalibrationConfig.n_splits` is deprecated as of v0.7.x — the outer split
count is now reused automatically.

---

## Export

### What is the difference between `export()` and `export_code()`?

| | `export()` | `export_code()` |
|-|------------|-----------------|
| **Output** | Pickle/joblib artifacts + metadata JSON | Pure Python files + JSON config |
| **Requires LizyML to load** | Yes | No |
| **Use case** | Save and restore a `Model` instance via `Model.load()` | Deploy to an environment without LizyML installed |
| **Calibrator** | Bundled inside `fit_result.pkl` | Serialized separately in `artifacts/` |
| **Generated files** | `fit_result.pkl`, `refit_model.pkl`, `metadata.json`, `analysis_context.pkl` | `train.py`, `predict.py`, `test_equivalence.py`, `config.json`, `requirements.txt`, `artifacts/` |

Use `export()` for experiment tracking and continued analysis within
LizyML. Use `export_code()` when you need a self-contained deployment
package that runs without any LizyML dependency.
