# Public API Reference

This document covers the `Model` class — LizyML's primary public interface.

## Model

```python
from lizyml import Model
```

`Model` is the config-driven facade for training, tuning, predicting, evaluating,
and exporting. It inherits diagnostic helpers from `ModelTablesMixin`,
`ModelPlotsMixin`, and `ModelPersistenceMixin`.

---

### Constructor

```python
Model(
    config: dict | str | Path | LizyMLConfig,
    *,
    data: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `dict \| str \| Path \| LizyMLConfig` | Config source: a dict, a YAML/JSON file path, or a `LizyMLConfig` instance. |
| `data` | `pd.DataFrame \| None` | Optional training DataFrame. Overrides `data.path` in config. |
| `output_dir` | `str \| Path \| None` | Root directory for run outputs. Overrides `output_dir` in config. |

---

### Core Methods

#### `fit`

```python
def fit(
    self,
    data: pd.DataFrame | None = None,
    params: dict | None = None,
) -> FitResult
```

Runs cross-validation training and (when configured) a full-data refit for
prediction. When `tune()` was called beforehand, the best hyperparameters are
automatically applied.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame \| None` | Training DataFrame. Overrides any data from construction time or `data.path` in config. |
| `params` | `dict \| None` | Model parameters that override `model.params` in config. |

**Returns:** [`FitResult`](#fitresult)

**Raises:**
- `LizyMLError(CONFIG_INVALID)` — missing required config fields.
- `LizyMLError(DATA_SCHEMA_INVALID)` — target or feature columns not found.
- `LizyMLError(LEAKAGE_SUSPECTED)` — split/calibration invariant violated.

---

#### `tune`

```python
def tune(
    self,
    data: pd.DataFrame | None = None,
    *,
    resume: bool = False,
    n_trials: int | None = None,
    expand_boundary: bool | None = None,
    boundary_threshold: float = 0.05,
    progress_callback: TuneProgressCallback | None = None,
) -> TuningResult
```

Runs Optuna-based hyperparameter search. Best params are stored internally and
applied automatically on the next `fit()` call.

Call with `resume=True` after an initial `tune()` to add trials to the existing
Optuna study. The TPE sampler reuses knowledge from previous trials, and the
previous best params are enqueued as a warm-start trial. When `expand_boundary`
is enabled, dimensions whose best params are near the search space edge are
automatically expanded in the promising direction (H-0068).

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame \| None` | Training DataFrame. |
| `resume` | `bool` | If `True`, resume from the previous Study and add trials. Requires a prior `tune()` call. |
| `n_trials` | `int \| None` | Number of trials. `None` uses the config value. |
| `expand_boundary` | `bool \| None` | Auto-expand dims near boundary. `None` means `True` for default space, `False` for user-specified space. |
| `boundary_threshold` | `float` | Edge detection threshold (0.0–0.5). Best values within this fraction of the range from either edge trigger expansion. |
| `progress_callback` | `TuneProgressCallback \| None` | Called after each trial with a `TuneProgressInfo`. Exceptions inside the callback are caught and emitted as `RuntimeWarning`; tuning is never aborted. |

**Returns:** [`TuningResult`](#tuningresult)

**Raises:**
- `LizyMLError(CONFIG_INVALID)` — no `tuning` section in config, or `boundary_threshold` out of range.
- `LizyMLError(OPTIONAL_DEP_MISSING)` — `optuna` not installed.
- `LizyMLError(TUNING_FAILED)` — `resume=True` without prior `tune()` call.
- `LizyMLError(TUNING_FAILED)` — study failure.

---

#### `predict`

```python
def predict(
    self,
    X: pd.DataFrame,
    *,
    return_shap: bool = False,
) -> PredictionResult
```

Generates predictions using the full-data model trained by `RefitTrainer`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `pd.DataFrame` | Feature DataFrame with the same columns as training data. |
| `return_shap` | `bool` | When `True`, compute SHAP values (requires `pip install 'lizyml[explain]'`). |

**Returns:** [`PredictionResult`](#predictionresult)

**Raises:**
- `LizyMLError(MODEL_NOT_FIT)` — called before `fit()`.
- `LizyMLError(OPTIONAL_DEP_MISSING)` — `return_shap=True` and `shap` not installed.

---

#### `evaluate`

```python
def evaluate(
    self,
    metrics: list[str | dict] | None = None,
) -> dict
```

Returns structured evaluation metrics computed during `fit()`. When `metrics` is
provided, the pre-computed result is filtered to that subset — no recomputation
occurs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | `list[str \| dict] \| None` | Metric names or parameterised `MetricEntry` dicts to filter. `None` returns all metrics. |

**Returns:** Nested dict:

```python
{
    "raw": {
        "oof": {"metric_name": float, ...},
        "oof_per_fold": [{"metric_name": float}, ...],
        "if_mean": {"metric_name": float, ...},
        "if_per_fold": [{"metric_name": float}, ...],
        "oof_coverage": float,          # fraction of rows covered by OOF
    },
    "calibrated": {                     # binary task + calibrator only
        "oof": {"metric_name": float, ...},
        "oof_per_fold": [{"metric_name": float}, ...],
        # NOTE: if_mean / if_per_fold are intentionally absent
        #       (computing them on in-fold data is a leakage risk)
    },
}
```

**Raises:**
- `LizyMLError(MODEL_NOT_FIT)` — called before `fit()`.
- `LizyMLError(UNSUPPORTED_METRIC)` — unknown or task-incompatible metric name.

---

#### `evaluate_table`

```python
def evaluate_table(self) -> pd.DataFrame
```

Returns evaluation metrics as a formatted DataFrame. Rows are metric names;
columns are `if_mean`, `oof`, `fold_0` … `fold_N-1`, and `cal_oof` when
calibrated. `df.attrs["oof_coverage"]` contains the coverage fraction.

**Raises:** `LizyMLError(MODEL_NOT_FIT)` — called before `fit()`.

---

#### `confusion_matrix`

```python
def confusion_matrix(self, threshold: float = 0.5) -> dict[str, pd.DataFrame]
```

Returns IS/OOS confusion matrices for binary and multiclass tasks.

| Parameter | Type | Description |
|-----------|------|-------------|
| `threshold` | `float` | Binary decision boundary (default `0.5`). Ignored for multiclass. |

**Returns:** `{"is": pd.DataFrame, "oos": pd.DataFrame}`

**Raises:**
- `LizyMLError(MODEL_NOT_FIT)` — called before `fit()` or after `load()` without `analysis_context`.
- `LizyMLError(UNSUPPORTED_TASK)` — called on a regression model.

---

#### `importance`

```python
def importance(self, kind: str = "split") -> dict[str, float]
```

Returns averaged feature importance across all CV fold models.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kind` | `str` | `"split"`, `"gain"`, or `"shap"`. `"shap"` computes `mean(|SHAP|)` per feature across folds. |

**Returns:** `{feature_name: importance_score}`

**Raises:**
- `LizyMLError(MODEL_NOT_FIT)` — called before `fit()` or (for `"shap"`) after `load()` without `analysis_context`.
- `LizyMLError(OPTIONAL_DEP_MISSING)` — `kind="shap"` and `shap` not installed.

---

#### `export`

```python
def export(self, path: str | Path | None = None) -> Path
```

Saves LizyML artifacts to a directory for later `load()`. Writes:
`fit_result.pkl`, `refit_model.pkl`, `metadata.json`, `analysis_context.pkl`.

Path resolution (first match wins):
1. Explicit `path` argument.
2. `{run_dir}/export` when a run directory exists from `fit()` / `tune()`.
3. New run directory under `output_dir` if configured.
4. `LizyMLError(SERIALIZATION_FAILED)` — no destination available.

**Returns:** Resolved export directory path.

**Raises:**
- `LizyMLError(MODEL_NOT_FIT)` — called before `fit()`.
- `LizyMLError(SERIALIZATION_FAILED)` — I/O error or unresolvable path.

> **Security note:** The `.pkl` files use joblib/pickle. Only load artifacts
> from trusted sources.

---

#### `export_code`

```python
def export_code(self, path: str | Path) -> Path
```

Generates LizyML-independent Python code for training and inference. Output:
`train.py`, `predict.py`, `test_equivalence.py`, `config.json`,
`requirements.txt`, `artifacts/`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Output directory (created if absent). |

**Returns:** Resolved output directory path.

**Raises:**
- `LizyMLError(MODEL_NOT_FIT)` — called before `fit()`.
- `LizyMLError(UNSUPPORTED_TASK)` — non-LGBM estimator (not yet supported).

---

#### `load` (classmethod)

```python
@classmethod
def load(cls, path: str | Path) -> Model
```

Restores a `Model` from a directory created by `export()`. The returned
instance supports `predict()`, `evaluate()`, and (when `analysis_context` was
saved) `confusion_matrix()`, `importance()`, and `residuals()`.

**Raises:** `LizyMLError(DESERIALIZATION_FAILED)` — validation or I/O error.

> **Security note:** Only load from trusted sources — joblib uses pickle internally.

---

### Diagnostic Methods

These methods require `fit()` to have been called (or `load()` with a compatible
artifact).

| Method | Task | Description |
|--------|------|-------------|
| `residuals()` | regression | OOF residuals `(y_true - oof_pred)`, shape `(n_samples,)`. |
| `split_summary()` | all | Per-fold split sizes as a DataFrame. |
| `params_table()` | all | Resolved model parameters as a single-column DataFrame. |
| `tuning_table()` | all | All tuning trial results with `round` and `state` columns; requires `tune()` first. |
| `boundary_table()` | all | Boundary detection results per dimension; requires `tune(resume=True, expand_boundary=True)` (H-0068). |
| `residuals_plot()` | regression | Plotly residual diagnostic plots. |
| `roc_curve_plot()` | binary/multiclass | OOF ROC curve. |
| `calibration_plot()` | binary | Calibration reliability diagram. |
| `probability_histogram_plot()` | binary/multiclass | OOF probability distribution. |
| `importance_plot()` | all | Top-N feature importance bar chart. |
| `plot_learning_curve()` | all | Training/validation loss curves per fold. |
| `plot_oof_distribution()` | all | OOF prediction distribution. |
| `tuning_plot()` | all | Optuna optimization history. |

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `fit_result` | `FitResult` | Read-only access to the CV training result. Raises `MODEL_NOT_FIT` if `fit()` was not called. |

---

## Result Types

### FitResult

Complete output of a CV training run. All fields are populated; only
`calibrator` and `oof_raw_scores` may be `None`.

| Field | Type | Description |
|-------|------|-------------|
| `oof_pred` | `NDArray[float64]` | OOF predictions. Shape `(n_samples,)` for regression/binary; `(n_samples, n_classes)` for multiclass. |
| `if_pred_per_fold` | `list[NDArray[float64]]` | In-fold predictions, one array per fold. |
| `metrics` | `dict` | Nested metrics dict (same structure as `evaluate()` output). |
| `models` | `list` | Trained model adapters, one per CV fold. |
| `history` | `list[dict]` | Per-fold training history. Each dict has `"eval_history"` and `"best_iteration"`. |
| `feature_names` | `list[str]` | Ordered list of feature column names used during training. |
| `dtypes` | `dict[str, str]` | Feature name → dtype string mapping. |
| `categorical_features` | `list[str]` | Feature names encoded as categorical. |
| `splits` | `SplitIndices` | Full index record for outer/inner/calibration splits. |
| `data_fingerprint` | `DataFingerprint` | Fingerprint of the training dataset. |
| `pipeline_state` | `Any` | Serializable state of the `FeaturePipeline`. |
| `calibrator` | `CalibrationResult \| None` | Fitted calibrator; `None` when calibration is disabled. |
| `run_meta` | `RunMeta` | Version and config metadata captured at fit time. |
| `oof_raw_scores` | `NDArray[float64] \| None` | OOF raw logit scores for calibration. `None` when calibration is not enabled. |

---

### PredictionResult

Output of a single `predict()` call.

| Field | Type | Description |
|-------|------|-------------|
| `pred` | `NDArray[float64]` | Point predictions, shape `(n_samples,)`. |
| `proba` | `NDArray[float64] \| None` | Class probabilities. Shape `(n_samples,)` for binary; `(n_samples, n_classes)` for multiclass. `None` for regression. |
| `shap_values` | `NDArray[float64] \| None` | SHAP values, shape `(n_samples, n_features)`. `None` when `return_shap=False`. |
| `used_features` | `list[str]` | Feature names that were present and used for prediction. |
| `warnings` | `list[str]` | Human-readable messages about column drift or corrections applied. |

---

### TuningResult

Result of a full hyperparameter search.

| Field | Type | Description |
|-------|------|-------------|
| `best_params` | `dict` | Flat view of all best parameters (model + smart + training). Convenience property. |
| `best_model_params` | `dict` | Best booster/model parameters. |
| `best_smart_params` | `dict` | Best smart parameters (e.g. `pos_weight_ratio`). |
| `best_training_params` | `dict` | Best training parameters (e.g. `num_boost_round`). |
| `best_score` | `float` | Best OOF score achieved. |
| `trials` | `list[TrialResult]` | All trial results (number, params, score, state, round). |
| `metric_name` | `str` | Name of the metric used for optimization. |
| `direction` | `str` | `"minimize"` or `"maximize"`. |
| `rounds` | `tuple[RoundSummary, ...]` | Per-round tuning history (H-0068). Empty tuple for single-round tuning. |
| `boundary_report` | `BoundaryReport \| None` | Boundary detection results (H-0068). Set only when `resume=True` with `expand_boundary` enabled. |

### TrialResult

| Field | Type | Description |
|-------|------|-------------|
| `number` | `int` | Trial index (0-based). |
| `params` | `dict` | Parameters sampled in this trial. |
| `score` | `float` | Objective value (NaN if failed). |
| `state` | `str` | `"complete"`, `"pruned"`, or `"fail"`. |
| `round` | `int` | Which re-tune round this trial belongs to (1-indexed, H-0068). |

### RoundSummary

Summary of a single tuning round (H-0068).

| Field | Type | Description |
|-------|------|-------------|
| `round` | `int` | Round number (1-indexed). |
| `n_trials` | `int` | Number of trials in this round. |
| `best_score_before` | `float \| None` | Best score at start of round (`None` for round 1). |
| `best_score_after` | `float` | Best score at end of round. |
| `expanded_dims` | `tuple[str, ...]` | Names of dimensions expanded before this round. |
| `space_snapshot` | `tuple[SearchDim, ...]` | Search space used in this round. |

### BoundaryReport

Boundary detection results for all dimensions (H-0068).

| Field | Type | Description |
|-------|------|-------------|
| `dims` | `tuple[BoundaryDimStatus, ...]` | Per-dimension boundary analysis. |
| `expanded_names` | `tuple[str, ...]` | Names of dimensions that were expanded. |

### BoundaryDimStatus

Per-dimension boundary analysis (H-0068).

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Dimension name. |
| `best_value` | `float \| int \| str \| None` | Best parameter value found. |
| `low` / `high` | `float \| int \| None` | Search range bounds (`None` for categorical). |
| `position_pct` | `float \| None` | Relative position of best in [0.0, 1.0] (`None` for categorical). |
| `edge` | `str` | `"lower"`, `"upper"`, or `"none"`. |
| `expanded` | `bool` | Whether this dim was expanded. |
| `new_low` / `new_high` | `float \| int \| None` | New bounds after expansion (`None` if not expanded). |

### TuneProgressInfo

Progress information emitted after each tuning trial (H-0048, H-0068).

| Field | Type | Description |
|-------|------|-------------|
| `current_trial` | `int` | Current trial number in this round (1-indexed). |
| `total_trials` | `int` | Total trials in this round. |
| `elapsed_seconds` | `float` | Time elapsed since `tune()` started. |
| `best_score` | `float \| None` | Best score so far. |
| `latest_score` | `float \| None` | Score of the latest trial (`None` if fail/pruned). |
| `latest_state` | `str` | `"complete"`, `"pruned"`, or `"fail"`. |
| `round` | `int` | Current round number (1-indexed, H-0068). |
| `cumulative_trials` | `int` | Total trials across all rounds (H-0068). |
| `expanded_dims` | `tuple[str, ...]` | Dimensions expanded in this round (H-0068). |

---

## Error Codes

All LizyML errors are instances of `LizyMLError` and carry a structured `code`
field for programmatic handling.

```python
from lizyml.core.exceptions import LizyMLError, ErrorCode

try:
    model.fit(data=df)
except LizyMLError as e:
    print(e.code)          # ErrorCode.CONFIG_INVALID
    print(e.user_message)  # human-readable description
    print(e.context)       # structured debug info dict
```

| Code | When raised |
|------|-------------|
| `CONFIG_INVALID` | Missing required config fields; `tuning` section absent when `tune()` is called. |
| `CONFIG_VERSION_UNSUPPORTED` | `config_version` value is not supported. |
| `DATA_SCHEMA_INVALID` | Target or feature columns not found in the DataFrame. |
| `DATA_FINGERPRINT_MISMATCH` | DataFrame does not match the fingerprint recorded at fit time. |
| `LEAKAGE_SUSPECTED` | A split or calibration invariant that could indicate leakage was violated. |
| `LEAKAGE_CONFIRMED` | A confirmed leakage condition (e.g. same row in train and validation). |
| `OPTIONAL_DEP_MISSING` | An optional dependency (`shap`, `optuna`) is not installed. |
| `MODEL_NOT_FIT` | A method requiring a trained model was called before `fit()`. |
| `INCOMPATIBLE_COLUMNS` | Prediction data columns do not match training columns. |
| `UNSUPPORTED_TASK` | A method is not applicable to the configured task type. |
| `UNSUPPORTED_METRIC` | An unknown or task-incompatible metric name was provided. |
| `TUNING_FAILED` | The Optuna study encountered an unrecoverable failure. |
| `CALIBRATION_NOT_SUPPORTED` | Calibration was requested for a non-binary task or unsupported config. |
| `SERIALIZATION_FAILED` | `export()` encountered an I/O error or could not resolve a path. |
| `DESERIALIZATION_FAILED` | `load()` encountered a validation or I/O error. |
