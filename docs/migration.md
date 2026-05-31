# Version Migration Guide

This document covers breaking changes and migration steps for each LizyML
release from v0.5.0 onward.

---

## v0.15.0

### `LGBMConfig.params["objective"]` is now respected when task-compatible

**Impact:** Tuning / training with a non-default LightGBM `objective` (H-0079).

Before v0.15.0 a user/Optuna-supplied `objective` was silently stripped and
replaced with the task default. From v0.15.0:

- A **same-task** objective string flows through to `lgb.train` — e.g.
  `objective: "fair"` on a regression task now actually trains with Fair loss.
  Re-running `tune()` over `default_space` may therefore yield different
  `best_params` / metrics than pre-0.15 runs.
- A **cross-task** objective now raises `LizyMLError(CONFIG_INVALID)` instead of
  being silently demoted (contract unchanged — still rejected).
- A **callable** objective is rejected with `CONFIG_INVALID` (unsupported).

No config edits are required; review any explicit non-default `objective`.

---

## v0.10.0

### Persistence `format_version` bumped to 2

**Impact:** Saved `model.lizyml` artifacts (H-0070).

Auto-encoding of non-numeric classification targets added a `TargetEncoder` to
the artifact, bumping `FORMAT_VERSION` from 1 to 2. `Model.load()` accepts
**both** versions: v1 artifacts are migrated in memory by injecting a no-op
`TargetEncoder`, so existing saved models keep loading with no user action. New
saves are written as v2.

---

## v0.8.x

### ECE formula corrected

**Impact:** Binary calibration metrics only.

The Expected Calibration Error (ECE) computation was updated to use
`fraction_of_positives` (the true positive rate per bin) instead of the
old formula that was using `mean_predicted_value` for the accuracy term.
Existing saved metrics will show numerically different ECE values after
recomputing with v0.8.x.

**Action:** Re-run `fit()` to get updated ECE values. No config changes required.

---

### `confusion_matrix_table` excludes NaN rows

**Impact:** Binary and multiclass tasks with missing target values.

Rows where the target is `NaN` are now excluded from confusion matrix
computation. Previously they were included and could produce incorrect
totals. The change aligns `confusion_matrix()` output with `evaluate()`
which already excluded NaN rows.

**Action:** No config change required. If you were relying on the old row
count for validation, update your assertions.

---

### Calibrated metrics now include `oof_per_fold`

**Impact:** `evaluate()` output structure for binary tasks with calibration.

The `"calibrated"` key now contains `oof_per_fold` in addition to `oof`:

```python
# v0.8.x and later
result["calibrated"] == {
    "oof": {"roc_auc": ..., ...},
    "oof_per_fold": [{"roc_auc": ...}, ...],
}
```

Code that previously assumed `"calibrated"` had only one key will still
work — the additional key is additive.

---

### `RefitTrainer` pipeline boundary change

**Impact:** Prediction pipeline behaviour.

The pipeline `fit` boundary in `RefitTrainer` now matches the CVTrainer
boundary exactly: the pipeline is fitted on the full training set `X_train`
with no inner validation split. This ensures that feature statistics (e.g.
target encodings) use all available data uniformly.

**Action:** No config change required. Predictions may differ slightly from
v0.7.x artifacts when using target encoding transformers.

---

### Inner validation raises on empty training split

**Impact:** Configurations with very small datasets or high `validation_ratio`.

`InnerValid` now raises `ValueError` (with a descriptive message) instead of
silently training on zero rows when the inner training split is empty.

**Action:** Reduce `early_stopping.validation_ratio` or disable early stopping
for small datasets.

---

## v0.8.0

### Codegen feval support (H-0066)

`export_code()` now includes custom `feval` metric functions in the generated
`train.py` when the config specifies custom evaluation metrics. The generated
code is self-contained and no longer requires LizyML to be installed.

**Action:** Re-export with `export_code()` to get feval-aware generated code.

---

### New estimator implementation guide

A step-by-step guide for adding new estimator adapters is available at
`docs/add-estimator-guide.md`. No API changes.

---

## v0.7.x

### Tune-fit identity fixes (v0.7.2–v0.7.3)

**Impact:** Any workflow using `tune()` followed by `fit()`.

Before v0.7.2, `tune()` and `fit()` used different data preparation
paths, causing inconsistent splits and preprocessing. The best params from
`tune()` were also not always forwarded correctly.

After v0.7.3:
- Both use the unified `_build_train_components` path.
- `TuningResult.best_params` is automatically applied on the next `fit()`.
- Split seed is consistent between tuning and fitting.

**Action:** No code changes required. Update to v0.7.3+ to get correct
tune-then-fit behaviour.

---

### OOF coverage (H-0057, v0.7.x)

`evaluate()` now returns `oof_coverage` under `raw`:

```python
result["raw"]["oof_coverage"]  # float in [0, 1]
```

OOF metrics are computed only on covered rows (rows that appeared in a
validation fold). Previously all rows were included, which could bias
metrics for time-series splitters where early rows are never held out.

`evaluate_table()` exposes coverage via `df.attrs["oof_coverage"]`.

**Action:** No config change required. Metrics may differ numerically from
v0.6.x when coverage < 1.0 (time-series or purged splits).

---

### Outer split reuse for calibration (H-0058, v0.7.x)

Calibration now reuses the outer CV splits instead of creating a separate
set of calibration splits. This makes calibration deterministic and removes
the `calibration.n_splits` parameter.

```yaml
# v0.6.x (still accepted but deprecated)
calibration:
  method: isotonic
  n_splits: 5   # deprecated — emits UserWarning

# v0.7.x and later
calibration:
  method: isotonic
  # n_splits is no longer needed
```

`build_calibration_splitter()` is deprecated and emits `DeprecationWarning`.

**Action:** Remove `calibration.n_splits` from your config. If you call
`build_calibration_splitter()` directly, migrate to passing the outer splits
from `FitResult.splits.outer`.

---

## v0.6.0

### Blocked Group KFold (H-0060)

A new `BlockedGroupKFoldConfig` split type was added for 2-axis
cross-validation (time blocks × group folds):

```yaml
split:
  type: blocked_group_kfold
  blocks:
    col: date
    cutoffs: ["2023-01-01", "2023-07-01"]
    mode: expanding   # or "sliding"
  groups:
    col: customer_id
    n_splits: 5
    stratify: true
```

**Action:** No breaking changes. Existing configs using `time_series` or
`kfold` are unaffected.

---

## v0.5.0

### EstimatorProvider and 5-layer architecture (H-0051/H-0052/H-0053)

v0.5.0 introduced a major internal refactor to a 5-layer category
architecture. Public API is unchanged, but several internal modules were
reorganized:

- `lizyml/estimators/lgbm.py` → `lizyml/estimators/lgbm/` sub-package
  (`adapter.py`, `smart_params.py`, `defaults.py`, `provider.py`).
- `EstimatorProvider` protocol introduced (`lizyml/estimators/provider.py`).
- Default search space and fixed params moved from `tuning/search_space.py`
  to `estimators/lgbm/defaults.py`.
- `DataFingerprint` moved from a utility module to
  `lizyml/core/types/artifacts.py`.

**Action:** If you import from internal LizyML modules (not recommended),
update your import paths. All public API imports via `from lizyml import Model`
are unaffected.

---

## Checking Your Version

```python
import lizyml
print(lizyml.__version__)
```

Or via uv:

```bash
uv run python -c "import lizyml; print(lizyml.__version__)"
```
