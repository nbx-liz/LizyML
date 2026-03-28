# Config Reference

> Full reference for all `config_version=1` keys.
> See [README](../README.md) for quick start and installation.

## Top-Level Keys

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

## `data`

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `path` | `str \| null` | No | `null` | CSV/Parquet path used when `fit()` is called without `data=`. |
| `target` | `str` | Yes | - | Target column name. |
| `time_col` | `str \| null` | No | `null` | Time column for chronological workflows (`time_series`, `purged_time_series`, `group_time_series` require it). |
| `group_col` | `str \| null` | No | `null` | Group column for group-aware split/validation. |

## `features`

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `exclude` | `list[str]` | No | `[]` | Columns excluded from training features. |
| `auto_categorical` | `bool` | No | `True` | Automatically treats suitable columns as categorical. |
| `categorical` | `list[str]` | No | `[]` | Explicit categorical feature names. |

## `split`

`split.method` is one of:

- `kfold`
- `stratified_kfold`
- `group_kfold`
- `time_series`
- `purged_time_series`
- `group_time_series`
- `blocked_group_kfold`

Supported aliases are normalized automatically:

- `k-fold` -> `kfold`
- `stratified-kfold` / `stratifiedkfold` -> `stratified_kfold`
- `group-kfold` / `groupkfold` -> `group_kfold`
- `time-series` / `timeseries` -> `time_series`
- `purged-time-series` / `purgedtimeseries` -> `purged_time_series`
- `group-time-series` / `grouptimeseries` -> `group_time_series`
- `blocked-group-kfold` / `blockedgroupkfold` -> `blocked_group_kfold`

Method-specific keys:

| method | Keys |
|---|---|
| `kfold` | `n_splits=5`, `random_state=42`, `shuffle=True` |
| `stratified_kfold` | `n_splits=5`, `random_state=42` |
| `group_kfold` | `n_splits=5` |
| `time_series` | `n_splits=5`, `gap=0`, `train_size_max=null`, `test_size_max=null` |
| `purged_time_series` | `n_splits=5`, `purge_gap=0`, `embargo=0`, `train_size_max=null`, `test_size_max=null` |
| `group_time_series` | `n_splits=5`, `gap=0`, `train_size_max=null`, `test_size_max=null` |
| `blocked_group_kfold` | `blocks={col, cutoffs, mode, train_window}`, `groups={col, n_splits, stratify, shuffle}`, `min_train_rows=10`, `min_valid_rows=5` |

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

### 2-Axis CV: `blocked_group_kfold`

Use this when you need both **time-period blocking** and **group isolation** (e.g., train on past months with separate users for train/valid).

Each fold = (time block) x (group fold). Total folds = `len(cutoffs) x groups.n_splits`.

```yaml
split:
  method: blocked_group_kfold
  blocks:                          # Period axis: what defines train/valid periods
    col: date                      #   Column to split on (must be orderable)
    cutoffs: ["2025-03"]           #   Boundary values (valid period starts here)
    mode: expanding                #   expanding | sliding
  groups:                          # Group axis: how to cross-validate entities
    col: user_id                   #   Column to split groups on
    n_splits: 3                    #   Number of group folds
    stratify: auto                 #   auto (task-dependent) | true | false
    shuffle: true
  min_train_rows: 10               # Skip fold if train < N rows
  min_valid_rows: 5                # Skip fold if valid < N rows
```

```text
cutoffs=["2025-03"], mode=expanding, groups.n_splits=2

Time block 0:  Train period = before Mar,  Valid period = Mar onward

  Sub-fold 0-0:  Train users={A,B}  Valid users={C,D}
    Train = (before Mar) & {A,B}     Valid = (Mar onward) & {C,D}

  Sub-fold 0-1:  Train users={C,D}  Valid users={A,B}
    Train = (before Mar) & {C,D}     Valid = (Mar onward) & {A,B}

Excluded rows: (train period x valid users) + (valid period x train users)
-> No user appears in both train and valid within any fold.
```

Key behaviors:
- `blocks.col` and `groups.col` must be different columns.
- Data is sorted by `blocks.col` before splitting.
- For classification tasks, `stratify: auto` balances target distribution across group folds using each group's majority-class label.
- Inner validation (early stopping) uses `BlockedGroupInnerValid`: group-isolated, time-ordered, stratified (classification). Falls back to row-level split when fewer than 4 groups.

## `model`

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
| `balanced` | `bool \| null` | No | `null` | `null`=auto (regression->false, binary/multiclass->true). Classification only. |

Validation constraints:

- `auto_num_leaves=True` and `params.num_leaves` cannot be specified together.
- `min_data_in_leaf_ratio` and `params.min_data_in_leaf` cannot be specified together.
- `min_data_in_bin_ratio` and `params.min_data_in_bin` cannot be specified together.

### Training metric vs evaluation metric

LizyML has two separate metric systems:

| | Training metric (`model.params.metric`) | Evaluation metric (`evaluation.metrics`) |
|---|---|---|
| **Purpose** | Used by LightGBM during training (early stopping, validation monitoring) | Used by LizyML after training (OOF/IF evaluation, results table) |
| **Where** | Passed to `lgb.train(params={"metric": ...})` | Computed by `Evaluator` on FitResult |
| **Naming** | LightGBM native names (`binary_logloss`) or LizyML names (`logloss`) | LizyML names only (`logloss`) |

You can use LizyML metric names in `model.params.metric` — they are automatically translated
to LightGBM equivalents (e.g. `"logloss"` → `"binary_logloss"` for binary tasks).

### Training metric reference (`model.params.metric`)

Supported values for `model.params.metric` by task:

**LightGBM native metrics:**

| task | metrics |
|---|---|
| `regression` | `l1` (`mae`), `l2` (`mse`), `rmse`, `quantile`, `mape`, `huber`, `fair`, `poisson`, `gamma`, `gamma_deviance`, `tweedie`, `r2` |
| `binary` | `binary_logloss`, `binary_error`, `auc`, `average_precision`, `cross_entropy`, `cross_entropy_lambda`, `kullback_leibler` |
| `multiclass` | `multi_logloss`, `multi_error`, `auc`, `auc_mu` |

**LizyML name auto-translation:**

| LizyML name | translates to | task |
|---|---|---|
| `logloss` | `binary_logloss` / `multi_logloss` | binary / multiclass |
| `auc_pr` | `average_precision` | binary / multiclass |

**Custom feval metrics** (injected via `lgb.train(feval=...)`):

| metric | regression | binary | multiclass |
|---|:---:|:---:|:---:|
| `rmsle` | ✅ | | |
| `f1` | | ✅ | ✅ |
| `brier` | | ✅ | ✅ |
| `ece` | | ✅ | |
| `precision_at_k` | | ✅ | |
| `accuracy` | | ✅ | ✅ |

Native and custom metrics can be mixed: `params={"metric": ["auc", "f1"]}`.

Invalid metric names are rejected before training with a clear error message listing valid options.

### Default LightGBM params

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

## `training`

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
  - `blocked_group_kfold` -> `blocked_group_inner_valid` (group-isolated + time-ordered + stratified for classification; falls back to row-level split when < 4 groups)
  - otherwise -> `holdout(stratify=False)`
- `validation_ratio` and `inner_valid` should not be explicitly set together (except round-trip-equivalent holdout dump values).

## `tuning`

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

## `evaluation`

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

Metric details:

| metric | description | `needs_proba` | `greater_is_better` |
|---|---|:---:|:---:|
| `rmse` | Root Mean Squared Error | No | No |
| `mae` | Mean Absolute Error | No | No |
| `r2` | R² (Coefficient of Determination) | No | Yes |
| `rmsle` | Root Mean Squared Logarithmic Error (requires non-negative values) | No | No |
| `mape` | Mean Absolute Percentage Error (undefined when y_true contains zeros) | No | No |
| `huber` | Huber Loss (delta=1.0) | No | No |
| `logloss` | Log Loss (binary cross-entropy / multi-class cross-entropy) | Yes | No |
| `auc` | Area Under the ROC Curve (binary or multiclass OvR macro) | Yes | Yes |
| `auc_pr` | Area Under the Precision-Recall Curve (macro average for multiclass) | Yes | Yes |
| `f1` | F1 Score (threshold=0.5 for binary, macro average for multiclass) | No | Yes |
| `accuracy` | Classification Accuracy (threshold=0.5 for binary) | No | Yes |
| `brier` | Brier Score (mean squared probability error, macro average for multiclass) | Yes | No |
| `ece` | Expected Calibration Error (equal-width bins, M=10) | Yes | No |
| `precision_at_k` | Precision at top-K% (default K=10) | Yes | Yes |

## `calibration`

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `method` | `"platt" \| "isotonic" \| "beta"` | No | `"platt"` | All methods are implemented. `beta` requires `scipy`. |
| `n_splits` | `int` | No | `5` | **Deprecated (H-0058)**: ignored. Calibration cross-fit reuses outer CV splits. Non-default values emit `UserWarning`. |

Runtime notes:

- Calibration is supported only for `task="binary"`.
- `method="beta"` is supported (install optional dependency: `pip install 'lizyml[calibration]'`).
- Calibration cross-fit reuses outer CV split indices directly (H-0058). The fold count and split boundaries (group / time / purge / embargo) are inherited from the outer CV configuration.

## Loader/Override Behavior

- Config source can be `dict`, `.json`, `.yaml`, or `.yml`.
- Environment-variable overrides use `LIZYML__` prefix and `__` nesting separators.
  - Example: `LIZYML__training__seed=999`
  - Example: `LIZYML__model__lgbm__params__learning_rate=0.01`

The config system uses:
- unified loading from `dict / JSON / YAML`
- strict validation with `pydantic` (`extra="forbid"`)
- CLI and environment-variable overrides
- normalization rules for aliases and deprecated keys
