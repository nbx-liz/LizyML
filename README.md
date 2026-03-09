# LizyML

LizyML is an analytics library designed to run machine learning workflows in a unified, config-driven way for regression, binary classification, and multiclass classification.

Its goal is to provide a single `Model` interface that handles the following with stable contracts:

- `tune`: hyperparameter optimization
- `fit`: training (CV / refit / early stopping)
- `evaluate`: evaluation (IF / OOF, before-and-after calibration comparison)
- `predict`: inference (column mismatch detection, optional explainability)
- `export`: save and reuse a `Model Artifact`

## Features

- Centralized training control through configuration
- Reproducibility-first design with saved `seed / split / params / versions / schema`
- Leakage-aware OOF and calibration workflow
- Stable contracts for `FitResult / PredictionResult / Artifacts`
- `Model.load()` support for restoring a saved `Model Artifact`, including training-time evaluation metadata

## Current Scope

- LightGBM is the highest-priority backend for the initial phase.
- Interfaces and responsibility boundaries are fixed first so the library can later expand to `sklearn` and DNNs (Torch).
- Distributed training platforms (Ray / Dask) and large-scale Auto Feature Engineering are not current priorities.

## Basic Usage

```python
from lizyml import Model

model = Model(config=config)

model.tune()
fit_result = model.fit()
eval_result = model.evaluate()

pred_result = model.predict(X_test, return_shap=True)

model.export("artifacts/run_001")
```

Saved `Model Artifact`s can be restored later with `Model.load()`.

```python
loaded_model = Model.load("artifacts/run_001")

eval_result = loaded_model.evaluate()
pred_result = loaded_model.predict(X_new)
```

This design lets you inspect not only predictions, but also how well the model performed and which settings were used during training.

## Config Example

```python
config = {
    "config_version": 1,
    "task": "regression",
    "data": {
        "path": "data.csv",
        "target": "y",
    },
    "features": {
        "exclude": ["id"],
        "auto_categorical": True,
        "categorical": ["cat_feature1", "cat_feature2"],
    },
    "split": {
        "method": "kfold",
        "n_splits": 5,
        "random_state": 1120,
    },
    "model": {
        "lgbm": {
            "params": {
                "n_estimators": 1000,
                "learning_rate": 0.05,
            }
        }
    },
    "training": {
        "early_stopping": {
            "enabled": True,
            "inner_valid": {
                "method": "holdout",
                "ratio": 0.1,
                "random_state": 1120,
            },
        }
    },
    "tuning": {
        "optuna": {
            "params": {
                "n_trials": 50,
                "direction": "minimize",
            },
            "space": {
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.1,
                    "log": True,
                    "category": "model",
                },
                "num_leaves_ratio": {
                    "type": "float",
                    "low": 0.5,
                    "high": 1.0,
                    "category": "smart",
                },
            },
        }
    },
    "evaluation": {
        "metrics": ["rmse", "mae"],
    },
}
```

## Config Reference (All Keys)

This section documents all config keys currently supported by the implemented schema (`config_version=1`).

### Top-Level Keys

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `config_version` | `int` | Yes | - | Only `1` is currently supported. |
| `task` | `"regression" \| "binary" \| "multiclass"` | Yes | - | Task type used across split/metrics/training behavior. |
| `data` | `object` | Yes | - | Data source and target definition. |
| `features` | `object` | No | `{}` | Uses `exclude=[]`, `auto_categorical=True`, `categorical=[]`. |
| `split` | `object` | No (loader fills if missing) | task-dependent | `binary/multiclass` -> stratified default, `regression` -> kfold default. |
| `model` | `object` | Yes | - | LightGBM only in current scope. |
| `training` | `object` | No | `{}` | Uses `seed=42` and default early stopping config. |
| `tuning` | `object \| null` | No | `null` | Required only if you call `model.tune()`. |
| `evaluation` | `object` | No | `{}` | Uses `metrics=[]` (runtime fallback applies). |
| `calibration` | `object \| null` | No | `null` | Binary-only feature at runtime. |

### `data`

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `path` | `str \| null` | No | `null` | CSV/Parquet path used when `fit()` is called without `data=`. |
| `target` | `str` | Yes | - | Target column name. |
| `time_col` | `str \| null` | No | `null` | Time column for chronological workflows (`time_series`, `purged_time_series`, `group_time_series` require it). |
| `group_col` | `str \| null` | No | `null` | Group column for group-aware split/validation. |

### `features`

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `exclude` | `list[str]` | No | `[]` | Columns excluded from training features. |
| `auto_categorical` | `bool` | No | `True` | Automatically treats suitable columns as categorical. |
| `categorical` | `list[str]` | No | `[]` | Explicit categorical feature names. |

### `split`

`split.method` is one of:

- `kfold`
- `stratified_kfold`
- `group_kfold`
- `time_series`
- `purged_time_series`
- `group_time_series`

Supported aliases are normalized automatically:

- `k-fold` -> `kfold`
- `stratified-kfold` / `stratifiedkfold` -> `stratified_kfold`
- `group-kfold` / `groupkfold` -> `group_kfold`
- `time-series` / `timeseries` -> `time_series`
- `purged-time-series` / `purgedtimeseries` -> `purged_time_series`
- `group-time-series` / `grouptimeseries` -> `group_time_series`

Method-specific keys:

| method | Keys |
|---|---|
| `kfold` | `n_splits=5`, `random_state=42`, `shuffle=True` |
| `stratified_kfold` | `n_splits=5`, `random_state=42` |
| `group_kfold` | `n_splits=5` |
| `time_series` | `n_splits=5`, `gap=0`, `train_size_max=null`, `test_size_max=null` |
| `purged_time_series` | `n_splits=5`, `purge_gap=0`, `embargo=0`, `train_size_max=null`, `test_size_max=null` |
| `group_time_series` | `n_splits=5`, `gap=0`, `train_size_max=null`, `test_size_max=null` |

Default when `split` is omitted:

- `task in {"binary", "multiclass"}` -> `{"method": "stratified_kfold", "n_splits": 5, "random_state": 42}`
- `task == "regression"` -> `{"method": "kfold", "n_splits": 5, "random_state": 42, "shuffle": True}`

Time-series notes:

- `time_series`, `purged_time_series`, and `group_time_series` all sort rows by `data.time_col` in ascending order before fold generation.
- `train_size_max` and `test_size_max` are shared across all three methods and cap training/validation window sizes.
- `purged_time_series` uses `embargo` as the canonical key (`embargo_pct` is accepted only as a legacy alias during migration).

### TimeSeries CV Guide (3 Methods)

All three methods enforce chronological splitting. The difference is how strictly each method blocks potentially leaky rows around the validation window.

Shared index-building rules:

1. Sort rows by `data.time_col` in ascending order.
2. Build each fold in chronological order (`train` always before `valid`).
3. Apply method-specific exclusion (`gap` / `purge_gap` / `embargo`).
4. Apply `train_size_max` / `test_size_max` caps when configured.

Quick comparison:

| method | boundary key | extra exclusion key | group-safe split | typical use |
|---|---|---|---|---|
| `time_series` | `gap` | - | No | Standard forward CV |
| `purged_time_series` | `purge_gap` | `embargo` | No | Leakage-sensitive time labels/features |
| `group_time_series` | `gap` | - | Yes | Entity blocks + chronology |

#### 1) `time_series`

Use this when regular forward-chaining CV is enough.

```text
time ---> older ........................................ newer

Fold k:
[           train           ][gap][    valid    ]
                (optional train_size_max / test_size_max caps)
```

- Validation always comes after training in time.
- `gap` removes rows right before validation.

#### 2) `purged_time_series`

Use this when labels/features can leak across nearby timestamps and you need stronger exclusion.

```text
time ---> older ........................................ newer

Fold k:
[      candidate train region      ][purge_gap][ valid ][embargo]
         \____________ train kept _______________/   \__ excluded __/
```

- `purge_gap` separates train and validation.
- `embargo` additionally excludes rows adjacent to the validation window.
- In migration periods, `embargo_pct` is normalized to `embargo`.

#### 3) `group_time_series`

Use this when samples must be split by group blocks while still respecting chronological order.

```text
time-sorted groups:   G1   G1   G2   G2   G3   G3   G4   G4

Fold k:
[       train groups        ][gap][ valid groups ]
```

- Group boundaries are preserved (no train/valid overlap on the same group).
- Group ordering follows chronological order from `time_col`.

### `model`

Current backend is LightGBM only.

Accepted input styles:

- BLUEPRINT style: `{"model": {"lgbm": {...}}}`
- Normalized style: `{"model": {"name": "lgbm", ...}}`

`model.lgbm` / normalized `model` keys:

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | `"lgbm"` | Yes (normalized style) | - | Automatically derived from BLUEPRINT style. |
| `params` | `dict[str, Any]` | No | `{}` | Passed to LightGBM adapter. |
| `auto_num_leaves` | `bool` | No | `True` | Auto-resolves `num_leaves` from depth logic. |
| `num_leaves_ratio` | `float` | No | `1.0` | Must satisfy `0 < ratio <= 1`. |
| `min_data_in_leaf_ratio` | `float \| null` | No | `0.01` | Must satisfy `0 < ratio < 1` if set. |
| `min_data_in_bin_ratio` | `float \| null` | No | `0.01` | Must satisfy `0 < ratio < 1` if set. |
| `feature_weights` | `dict[str, float] \| null` | No | `null` | All values must be `> 0`. |
| `balanced` | `bool \| null` | No | `null` | `null`=auto (regression→false, binary/multiclass→true). Classification only. |

Validation constraints:

- `auto_num_leaves=True` and `params.num_leaves` cannot be specified together.
- `min_data_in_leaf_ratio` and `params.min_data_in_leaf` cannot be specified together.
- `min_data_in_bin_ratio` and `params.min_data_in_bin` cannot be specified together.

Default LightGBM params applied when not overridden in `model.params`:

Task-specific defaults:

| task | `objective` | `metric` |
|---|---|---|
| `regression` | `huber` | `["huber", "mae", "mape"]` |
| `binary` | `binary` | `["auc", "binary_logloss"]` |
| `multiclass` | `multiclass` | `["auc_mu", "multi_logloss"]` |

Common defaults:

| param | default |
|---|---|
| `boosting` | `gbdt` |
| `n_estimators` | `1500` |
| `learning_rate` | `0.001` |
| `max_depth` | `5` |
| `max_bin` | `511` |
| `feature_fraction` | `0.7` |
| `bagging_fraction` | `0.7` |
| `bagging_freq` | `10` |
| `lambda_l1` | `0.0` |
| `lambda_l2` | `0.000001` |
| `first_metric_only` | `False` |
| `verbose` | `-1` |

Runtime-injected default:

- `random_state`: uses `training.seed` (default `42`)

### `training`

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `seed` | `int` | No | `42` | Global training seed. |
| `early_stopping` | `object` | No | `{}` | Early stopping behavior. |

`training.early_stopping`:

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `enabled` | `bool` | No | `True` | Disable to skip inner validation strategy. |
| `rounds` | `int` | No | `150` | Early stopping rounds passed to adapter. |
| `validation_ratio` | `float \| null` | No | `0.1` | Shorthand for inner validation ratio. |
| `inner_valid` | `object \| null` | No | `null` (auto-resolved) | Explicit strategy config. |

`training.early_stopping.inner_valid.method` variants:

| method | Keys |
|---|---|
| `holdout` | `ratio=0.1`, `stratify=False`, `random_state=42` |
| `group_holdout` | `ratio=0.1`, `random_state=42` |
| `time_holdout` | `ratio=0.1` |

Resolution rules:

- If `inner_valid` is not explicitly set, method is auto-resolved from `split.method`:
  - `stratified_kfold` -> `holdout(stratify=True)`
  - `group_kfold` -> `group_holdout`
  - `time_series` -> `time_holdout`
  - `purged_time_series` -> `time_holdout`
  - `group_time_series` -> `group_holdout`
  - otherwise -> `holdout(stratify=False)`
- `validation_ratio` and `inner_valid` should not be explicitly set together (except round-trip-equivalent holdout dump values).

### `tuning`

`tuning` is optional. If present:

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `optuna.params.n_trials` | `int` | No | `50` | Number of optimization trials. |
| `optuna.params.direction` | `"minimize" \| "maximize"` | No | `"minimize"` | Optimization direction. |
| `optuna.params.timeout` | `float \| null` | No | `null` | Optional timeout in seconds. |
| `optuna.space` | `dict[str, Any]` | No | `{}` | Empty dict triggers task-specific default search space. |

`optuna.space` entry format:

```python
"space": {
    "learning_rate": {
        "type": "float",          # "float" | "int" | "categorical"
        "low": 0.0001,            # for float/int
        "high": 0.1,              # for float/int
        "log": True,              # optional for float/int
        "category": "model",      # optional: "model" | "smart" | "training"
    },
    "validation_ratio": {
        "type": "float",
        "low": 0.1,
        "high": 0.3,
        "category": "training",
    },
}
```

Default search space used when `optuna.space = {}`:

| Param | Type | Range / Choices | Log | Category |
|---|---|---|---|---|
| `objective` | `categorical` | Task-specific (see below) | - | `model` |
| `n_estimators` | `int` | `600 .. 2500` | `False` | `model` |
| `early_stopping_rounds` | `int` | `40 .. 240` | `False` | `training` |
| `validation_ratio` | `float` | `0.1 .. 0.3` | `False` | `training` |
| `learning_rate` | `float` | `0.0001 .. 0.1` | `True` | `model` |
| `max_depth` | `int` | `3 .. 12` | `False` | `model` |
| `feature_fraction` | `float` | `0.5 .. 1.0` | `False` | `model` |
| `bagging_fraction` | `float` | `0.5 .. 1.0` | `False` | `model` |
| `num_leaves_ratio` | `float` | `0.5 .. 1.0` | `False` | `smart` |
| `min_data_in_leaf_ratio` | `float` | `0.01 .. 0.2` | `False` | `smart` |

Task-specific default `objective` choices:

- `regression`: `["huber", "fair"]`
- `binary`: `["binary"]`
- `multiclass`: `["multiclass", "multiclassova"]`

When using the default space, these fixed params are also applied to every trial:

- `auto_num_leaves=True`
- `first_metric_only=True`
- `metric` is task-specific:
- for `regression`: `["huber", "mae", "mape"]`
- for `binary`: `["auc", "binary_logloss"]`
- for `multiclass`: `["auc_mu", "multi_logloss"]`

### `evaluation`

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `metrics` | `list[str]` | No | `[]` | Metric names validated per task. |

If `metrics` is empty, runtime defaults are:

- `regression`: `["rmse", "mae"]`
- `binary`: `["logloss", "auc"]`
- `multiclass`: `["logloss", "f1", "accuracy"]`

Supported metric names by task:

| task | metrics |
|---|---|
| `regression` | `rmse`, `mae`, `r2`, `rmsle`, `mape`, `huber` |
| `binary` | `logloss`, `auc`, `auc_pr`, `f1`, `accuracy`, `brier`, `ece`, `precision_at_k` |
| `multiclass` | `logloss`, `f1`, `accuracy`, `auc`, `auc_pr`, `brier` |

### `calibration`

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `method` | `"platt" \| "isotonic" \| "beta"` | No | `"platt"` | All methods are implemented. `beta` requires `scipy`. |
| `n_splits` | `int` | No | `5` | Number of folds for calibration cross-fit. |

Runtime notes:

- Calibration is supported only for `task="binary"`.
- `method="beta"` is supported (install optional dependency: `pip install 'lizyml[calibration]'`).
- Calibration cross-fit splits inherit `split.method` and its parameters
  (e.g. `gap`, `purge_gap`, `embargo`, group boundaries). Only the fold
  count is overridden by `calibration.n_splits`.

### Loader/Override Behavior

- Config source can be `dict`, `.json`, `.yaml`, or `.yml`.
- Environment-variable overrides use `LIZYML__` prefix and `__` nesting separators.
- Example: `LIZYML__training__seed=999`
- Example: `LIZYML__model__lgbm__params__learning_rate=0.01`

The config system is designed around the following assumptions:

- unified loading from `dict / JSON / YAML`
- strict validation with `pydantic` (`extra="forbid"`)
- CLI and environment-variable overrides
- normalization rules for aliases and deprecated keys

## Returned Objects and Saved Outputs

### `FitResult`

Training results are expected to follow a fixed schema, including:

- OOF predictions
- IF predictions
- metrics (with a stable structure for raw and calibrated comparisons)
- per-fold models
- training history (`learning_curve`, `best_iteration`, and similar metadata)
- split indices
- data fingerprint
- `FeaturePipeline` state

### `PredictionResult`

Prediction results are expected to include:

- `pred`
- `proba` (for binary classification)
- `shap_values` (when requested)
- `used_features`
- `warnings`

### `Model Artifact`

The artifact produced by `export` is expected to include at least:

- trained models (fold ensemble and/or refit model)
- `FeaturePipeline` state
- schema (column names / dtypes / categorical handling)
- calibrator
- metrics / history / fit summary
- `config_normalized`
- `versions / format_version`

## Core Design Priorities

### Reproducibility

Given the same config and seed, the library should reproduce the same split behavior and the same primary outputs.

### Leakage Prevention

- OOF predictions must come only from models that did not train on the target row.
- Calibration is OOF-only by design, and the calibrator itself is evaluated with cross-fit.
- Time and group constraints must be preserved across outer / inner / calibration splits.

### Separation of Responsibilities

- `Model` acts as a facade that orchestrates components.
- `Splitter` is limited to returning indices only.
- Plotting uses existing `FitResult` data and must not rely on inference or recomputation.

## Typical Use Cases

- Reproducing training conditions from config
- Managing CV / OOF / early stopping in a consistent way
- Applying binary calibration safely
- Saving trained models and reloading them later with evaluation metadata intact
- Using stable result contracts with predictable shapes and meanings

## Future Expansion

- broader `sklearn` estimator support
- optional DNN (Torch) support
- multiclass calibration
- ranking tasks
- additional export formats such as `ONNX / TorchScript / Booster text`

## Documentation

- Full implementation spec: [`BLUEPRINT.md`](BLUEPRINT.md)
- Proposal and decision history: [`HISTORY.md`](HISTORY.md)

`BLUEPRINT.md` is the source of truth for implementation rules. This README is a user-facing summary of that specification.
