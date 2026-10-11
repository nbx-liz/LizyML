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
    storage: str | BaseStorage | None = None,
    study_name: str | None = None,
    space: Mapping[str, Any] | None = None,
) -> TuningResult
```

Runs one round of Optuna-based hyperparameter search. Best params are stored
internally and applied automatically on the next `fit()` call.

**Rounds and the objective (H-0122).** Each call runs one *round* of trials.
With `resume=True` the round is added to an existing Optuna study, and the TPE
sampler learns from all of its trials. Every trial of one study observes the
same objective: when a study is created, LizyML records an identity of the
data (column names and order, dtypes, categories, values and row order), of
the config outside the search space (task, data and feature settings, `model`,
the effective fixed parameters, `training`, `split`) and of the first metric.
A resume whose identity differs is refused with `CONFIG_INVALID`, and the
error's context names the parts that changed. The library versions are
recorded too; a different version only warns.

Every trial trains on the config's parameters, the study's fixed parameters
(set by `space_mode: merge`) and the trial's sampled values. A resumed round
does not re-evaluate the previous best: each of its `n_trials` trials is a new
TPE sample. The result's best is the best trial of the whole study, which may
come from an earlier round, from a range that was moved since, or from a
dimension that was removed since; `fit()` then trains with that trial's
values. When `expand_boundary` is enabled, dimensions whose best value lies
near an edge of the range are expanded in that direction (H-0068). Expansion
never narrows or inverts a range. A dimension whose declaration changed in
this round, or whose best value lies outside its range, is not expanded.

**Changing the search range.** Pass `space=` to a `resume=True` call. The
config's `tuning.optuna.space` declares the space of round 1 only; editing it
and resuming is refused.

**Storage.** With `storage` and `study_name`, the trials and LizyML's records
(`lizyml:` study attributes: the objective identity, and for each round its
declared and effective space, budget and boundary report) are written to the
storage as the round runs. To continue a stored study, from a new process or a
new `Model` as well, call `tune(resume=True, storage=..., study_name=...)`: it
reattaches to the study and continues its rounds. When the storage has no
study of that name, `resume=True` creates the study and starts round 1 (logged
at INFO). If the last round was interrupted, the call finishes that round with
its recorded space and remaining budget instead of starting a new one, and
marks the trials a dead process left RUNNING as FAIL, counting them in the
budget. `resume=False` always starts a new study, so its `study_name` must be
new in the storage. One process adds trials to a study at a time; this is
assumed, not enforced. To inspect a stored study without adding to it, use
[`read_study_lineage`](#study-lineage-lizymltuning); to get a closed round's
result back without adding trials, use [`restore_tuning`](#restore_tuning).

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame \| None` | Training DataFrame. |
| `resume` | `bool` | If `True`, add a round to a study: the one this Model holds or, when it holds none, the stored `(storage, study_name)` study, which is reattached, or created when the storage lacks it. With a held study, a `storage` / `study_name` passed along must be the ones it was created or reattached with (an equal URL string, or the same `BaseStorage` object). |
| `n_trials` | `int \| None` | Number of trials. `None` uses the config value. An interrupted round is finished with its recorded budget and accepts only that value. |
| `expand_boundary` | `bool \| None` | Auto-expand dims near boundary. `None` means `True` for merge mode when this round's declared space has no user dimension, otherwise `False`. An interrupted round accepts only its recorded value. |
| `boundary_threshold` | `float` | Edge detection threshold (0.0–0.5). Best values within this fraction of the range from either edge trigger expansion. |
| `progress_callback` | `TuneProgressCallback \| None` | Called after each trial with a `TuneProgressInfo`. Exceptions inside the callback are caught and emitted as `RuntimeWarning`; tuning is never aborted. |
| `storage` | `str \| BaseStorage \| None` | Optional Optuna storage URL or `BaseStorage` for resumable tuning (H-0072). `None` keeps the in-memory behavior. Requires `study_name` when set. |
| `study_name` | `str \| None` | Study identifier, used together with `storage` and only with it. With `resume=False` the name must be new in the storage; continue a stored study with `resume=True`. |
| `space` | `Mapping[str, Any] \| None` | This round's search space, in the form of `tuning.optuna.space`, resolved by the study's `space_mode`: `merge` lays it over the provider's default dimensions (a default dimension `space=` leaves out keeps its default declaration, even if the config had overridden it; a dimension only the config declared is no longer searched unless `space=` declares it again), `replace` declares exactly it. Only for a round after the first (`resume=True` on a study that has a round). `None` keeps the previous round's space. A range (`low` / `high`) may change; a dimension's distribution kind, `log`, choices and category may not. A dimension declared as before keeps its expanded range; a changed or new one starts from the given range. A dimension left out is no longer searched, and later trials train on its config value. A range outside the provider's bounds is accepted and not clamped. An interrupted round accepts only its recorded space. |

**Returns:** [`TuningResult`](#tuningresult)

**Raises:**
- `LizyMLError(CONFIG_INVALID)`:
  - no `tuning` section in config, or `boundary_threshold` outside (0.0, 0.5);
  - `storage` without `study_name`, or `study_name` without `storage`, whatever `resume` is;
  - `resume=False` when the storage already holds a study of that name (continue it with `resume=True`, or pick a new `study_name`);
  - `resume=True` with a `storage` / `study_name` that is not the study this Model holds;
  - `space=` on a call that starts round 1 of a study: `resume=False`, a `resume=True` whose storage lacks the study, or a stored study with no round yet (a study's first space is `tuning.optuna.space`);
  - the call's objective differs from the study's: its data, config outside the search space, split or first metric;
  - the study has trials but no LizyML record: it was created before v0.19.0 or outside LizyML, so whether its trials observe this objective cannot be checked;
  - the study's LizyML record cannot be read (it comes from a newer LizyML, or was edited);
  - the study's record and its trials disagree;
  - `tuning.optuna.space_mode` differs from the study's;
  - `tuning.optuna.space` differs from the space the study was created with (change the range with `space=` instead);
  - `space=` is malformed, names an unknown parameter or one parameter twice, or changes a dimension's distribution kind (`float` / `int` / `categorical`), `log` or choices;
  - `space=` moves a dimension to another category (`model` / `smart` / `training`);
  - finishing an interrupted round with an `n_trials`, `expand_boundary` or `space=` other than the recorded one (omit them, or pass the recorded value);
  - for each of these, start a new study (another `study_name`) when the change is intended. Nothing is written to the study and the Model is unchanged.
- `LizyMLError(OPTIONAL_DEP_MISSING)` — `optuna` not installed.
- `LizyMLError(TUNING_FAILED)` — `resume=True` when the Model holds neither a study nor `storage` (also after `Model.load`); a round that ends with no COMPLETE trial in the study (it is recorded `failed`, and the next `resume=True` starts a new round); study failure.
- `UserWarning` — the study was created with other library versions (the objective identity matches, so it continues); or an interrupted round is finished while `tuning.optuna.params.n_trials` differs from its recorded budget (the recorded budget is used).

---

#### `restore_tuning`

```python
def restore_tuning(
    self,
    storage: str | BaseStorage,
    study_name: str,
    data: pd.DataFrame | None = None,
) -> TuningResult
```

Returns the result of a stored study's last round without adding to the study
(H-0122). This is for a tune whose round closed but whose caller never received
the result, for example because the process died in between: the study's last
round is `completed`, and `tune(resume=True)` would start another round. The
returned `TuningResult` equals the one that tune returned: the best parameters
split into model, smart and training parameters, `best_score`, the trials with
their rounds, `rounds`, and the last round's `boundary_report`.

On success the Model holds the study, its last round's effective space, its
rounds and the study's fixed parameters, as the tune left them. `fit()` then
trains with the best trial's parameters, and a later `tune(resume=True)` starts
the next round of this study. A fitted Model keeps its fit; only the tuning
result is replaced. `restore_tuning` writes nothing to the study: no attribute,
no trial, and a trial left RUNNING stays RUNNING.

| Parameter | Type | Description |
|-----------|------|-------------|
| `storage` | `str \| BaseStorage` | Optuna storage URL or `BaseStorage` holding the study. |
| `study_name` | `str` | The study's name in `storage`. |
| `data` | `pd.DataFrame \| None` | Training DataFrame, with the same priority as in `tune`. It must be the data the study observes. |

**Returns:** [`TuningResult`](#tuningresult)

**Raises:**
- `LizyMLError(TUNING_FAILED)` — the storage has no study of that name, or the study has no round that closed `completed`: its creation was interrupted, it has no round yet, or its last round closed `failed`. Checked before the objective, so changed data does not change this error.
- `LizyMLError(CONFIG_INVALID)` — no `tuning` section in config; `storage` or `study_name` missing; the study has no LizyML record, an unreadable one, or one that disagrees with its trials; the last round is open (interrupted: finish it with `tune(resume=True)`, a part-finished round's result is not restored); the study's direction, objective or config space differs from this Model's.
- `LizyMLError(OPTIONAL_DEP_MISSING)` — `optuna` not installed; raised before data is loaded or the storage is contacted.
- `UserWarning` — the study was created with other library versions.

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
| `kind` | `str` | `"split"`, `"gain"`, or `"shap"`. `"shap"` computes `mean(|SHAP|)` per feature across folds, explaining each fold model on its validation rows encoded by that fold's own feature pipeline (H-0114). |

**Returns:** `{feature_name: importance_score}`

**Raises:**
- `LizyMLError(MODEL_NOT_FIT)` — called before `fit()`; or, for `"shap"`, when the `FitResult` has no per-fold pipeline states (an artifact written before H-0114, or a `FitResult` constructed without `pipeline_state_per_fold`; `context["missing"] == "pipeline_state_per_fold"`, checked first; call `fit()` again), or after `load()` without `analysis_context`.
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

Run on the same data saved as parquet, the generated `train.py` retrains the
refit model: its uncalibrated predictions match the LizyML refit model's
uncalibrated predictions at `rtol=1e-7`, given the same machine and the same
LightGBM, numpy, pandas and scikit-learn versions (H-0120). With calibration
configured, `Model.predict` returns calibrated probabilities, and those are not
part of the promise: the calibrator is rebuilt, not reproduced. BLUEPRINT §15.4
states the premises and the CSV condition.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Output directory (created if absent). |

**Returns:** Resolved output directory path.

**Raises:**
- `LizyMLError(MODEL_NOT_FIT)` — called before `fit()`.
- `LizyMLError(UNSUPPORTED_TASK)` — non-LGBM estimator (not yet supported).
- `LizyMLError(SERIALIZATION_FAILED)` — a category, mode or target label that
  JSON cannot return with its type (for example a `tuple`, `bytes`, a timestamp
  or a `Decimal`). Nothing is written.

---

#### `load` (classmethod)

```python
@classmethod
def load(cls, path: str | Path) -> Model
```

Restores a `Model` from a directory created by `export()`. The returned
instance supports `predict()`, `evaluate()`, and (when `analysis_context` was
saved) `confusion_matrix()`, `importance()`, and `residuals()`. An artifact
written before H-0114 has no per-fold pipeline states, so on it
`importance(kind="shap")` raises `MODEL_NOT_FIT`; the other kinds still work.

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

Complete output of a CV training run. All fields are populated by training;
only `calibrator`, `oof_raw_scores` and `pipeline_state_per_fold` may be `None`.

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
| `pipeline_state` | `Any` | Serializable state of the last CV fold's `FeaturePipeline`. |
| `calibrator` | `CalibrationResult \| None` | Fitted calibrator; `None` when calibration is disabled. |
| `run_meta` | `RunMeta` | Version and config metadata captured at fit time. |
| `oof_raw_scores` | `NDArray[float64] \| None` | OOF raw logit scores for calibration. `None` when calibration is not enabled. |
| `target_encoder` | `TargetEncoder` | Maps non-numeric classification labels to integer codes and back. A no-op (`needs_encoding=False`) for numeric targets and regression; otherwise `classes_` holds the original labels sorted lexicographically by their string form (H-0070). |
| `pipeline_state_per_fold` | `list \| None` | Serializable state of each CV fold's `FeaturePipeline`, in fold order; the last equals `pipeline_state`. `None` when unavailable (an artifact written before H-0114, or a `FitResult` constructed without it). Shared by reference across copies (H-0114). |

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
| `rounds` | `tuple[RoundSummary, ...]` | Per-round tuning history (H-0068): one summary per round of the study that closed `completed`, in round order. A round closed `failed` (it ended with no COMPLETE trial in the study) is omitted, so `rounds` can start at round 2, and the next round's `best_score_before` is `None`. |
| `boundary_report` | `BoundaryReport \| None` | Boundary detection results of this round (H-0068): set when the round analysed boundaries (a round after the first, with `expand_boundary` resolved to `True` and a COMPLETE trial in the study), otherwise `None`. `restore_tuning` returns the last round's. |

### TrialResult

| Field | Type | Description |
|-------|------|-------------|
| `number` | `int` | Trial index (0-based). |
| `params` | `dict` | Parameters sampled in this trial. |
| `score` | `float` | Objective value (NaN if failed). |
| `state` | `str` | `"complete"`, `"pruned"`, or `"fail"`. |
| `round` | `int` | Which round this trial belongs to (1-indexed, H-0068), decided by the trial's number from the study's round records. |

### RoundSummary

Summary of a single tuning round (H-0068).

| Field | Type | Description |
|-------|------|-------------|
| `round` | `int` | Round number (1-indexed). |
| `n_trials` | `int` | Number of trials this round ran (its actual count, which a `timeout` can make smaller than the requested budget). |
| `best_score_before` | `float \| None` | Best score at start of round (`None` when the study had no COMPLETE trial yet: round 1, or the round after a `failed` one). |
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
| `current_trial` | `int` | The trial's position in this round (1-indexed). An interrupted round finished after a reattach continues the count: with K of its trials already finished, the first new trial is K+1, with or without `storage`. |
| `total_trials` | `int` | The round's requested `n_trials` (the recorded budget for an interrupted round). |
| `elapsed_seconds` | `float` | Time elapsed since `tune()` started. |
| `best_score` | `float \| None` | Best score so far. |
| `latest_score` | `float \| None` | Score of the latest trial (`None` if fail/pruned). |
| `latest_state` | `str` | `"complete"`, `"pruned"`, or `"fail"`. |
| `round` | `int` | The trial's round number (1-indexed, H-0068). It continues the study's rounds after a reattach. |
| `cumulative_trials` | `int` | Trials in the whole study so far, `trial.number + 1` (H-0068). |
| `expanded_dims` | `tuple[str, ...]` | Dimensions expanded in this round (H-0068). |

---

## Study lineage (`lizyml.tuning`)

A read-only view of a stored study's state and rounds (H-0122), for a caller
that decides between `tune(resume=True)`, `restore_tuning` and a new study, for
example after a pause or a crash. Import it from `lizyml.tuning`; it is not in
the top-level `lizyml` namespace.

```python
from lizyml.tuning import read_study_lineage

lineage = read_study_lineage("sqlite:///optuna.db", "my-study")
if lineage.state == "open":
    model.tune(resume=True, storage="sqlite:///optuna.db", study_name="my-study")
elif lineage.state == "completed":
    model.restore_tuning("sqlite:///optuna.db", "my-study")
```

### `read_study_lineage`

```python
def read_study_lineage(storage: str | BaseStorage, study_name: str) -> StudyLineage
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `storage` | `str \| BaseStorage` | Optuna storage URL or `BaseStorage`, as for `Model.tune`. |
| `study_name` | `str` | The study's name in `storage`. |

It is read-only: it writes no study attribute, leaves a trial that a dead
process left RUNNING as RUNNING (only `tune(resume=True)` marks it FAIL), and
does not check the objective identity, since it takes no data and no config.
The one side effect on the storage is Optuna's: listing a SQLite storage's
studies creates the file and its schema when they do not exist yet. No study is
created.

**Raises:**
- `LizyMLError(CONFIG_INVALID)` — `storage` or `study_name` missing.
- `LizyMLError(OPTIONAL_DEP_MISSING)` — `optuna` not installed.

### StudyLineage

A frozen dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `state` | `str` | The study's state; one of the values in the table below. |
| `rounds` | `tuple[RoundRecord, ...]` | One record per recorded round, `failed` rounds included (unlike `TuningResult.rounds`). Empty unless `state` is `"open"`, `"completed"` or `"failed"`. |
| `open_round` | `int \| None` | The open round's number when `state` is `"open"`, otherwise `None`. |

---

### RoundRecord

A frozen dataclass: one recorded round of a study.

| Field | Type | Description |
|-------|------|-------------|
| `round` | `int` | The round number, from 1. |
| `status` | `str` | `"completed"` or `"failed"` from the round's end record; `"open"` when the round has none (it was interrupted). |
| `n_trials` | `int` | The trial budget the round asked for. |
| `n_finished` | `int` | The round's COMPLETE, FAIL and PRUNED trials; a RUNNING trial is not counted. |
| `best_score_before` | `float \| None` | The study's best value when the round started; `None` when the study had no COMPLETE trial. |
| `best_score_after` | `float \| None` | The study's best value when the round closed; `None` for a `failed` or `open` round. |

---

### Study states

A study is classified by the first row that applies, from `"legacy"` down.
`tune(resume=False)` onto any study that exists raises `CONFIG_INVALID`.

| `state` | Meaning | `tune(resume=True)` | `restore_tuning` |
|---------|---------|---------------------|------------------|
| `"absent"` | The storage has no study of that name. | Creates it and starts round 1. | `TUNING_FAILED` |
| `"legacy"` | Trials but no LizyML record: created before v0.19.0 or outside LizyML. | `CONFIG_INVALID` | `CONFIG_INVALID` |
| `"uninitialized"` | No LizyML record and no trial: its creation was interrupted. | Initialises it and starts round 1. | `TUNING_FAILED` |
| `"unreadable"` | A LizyML record this LizyML cannot read (from a newer LizyML, or edited). | `CONFIG_INVALID` | `CONFIG_INVALID` |
| `"no_round"` | A LizyML record, but no round and no trial. | Starts round 1. | `TUNING_FAILED` |
| `"inconsistent"` | LizyML's records and the trials disagree. | `CONFIG_INVALID` | `CONFIG_INVALID` |
| `"open"` | The last round has no end record: it was interrupted. | Finishes that round with its recorded space and remaining budget. | `CONFIG_INVALID` |
| `"completed"` | The last round closed `completed`. | Starts the next round. | Restores the result. |
| `"failed"` | The last round closed `failed`: the study had no COMPLETE trial. | Starts the next round, without boundary expansion. | `TUNING_FAILED` |

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
| `CONFIG_INVALID` | Missing required config fields; `tuning` section absent when `tune()` is called; a `tune` or `restore_tuning` call its study refuses (another objective, a study created before v0.19.0, `resume=False` onto a stored study, a `space=` the study cannot take; see [`tune`](#tune)). |
| `CONFIG_VERSION_UNSUPPORTED` | `config_version` value is not supported, on every path a config reaches `Model` (dict, file, `LizyMLConfig` instance, environment override). |
| `DATA_SCHEMA_INVALID` | Target or feature columns not found in the DataFrame. Also raised when `validate_no_target_leakage` / `validate_time_series_order` is given a `target` / `time_col` that is not in the frame (`context` has `missing_columns` / `available_columns`), or when `validate_no_target_leakage` cannot compare a column with the target (`context` names `column` and `target`, `cause` is the original error). |
| `LEAKAGE_SUSPECTED` | A split or calibration invariant that could indicate leakage was violated. |
| `LEAKAGE_CONFIRMED` | A confirmed leakage condition (e.g. same row in train and validation). |
| `OPTIONAL_DEP_MISSING` | An optional dependency (`shap`, `optuna`) is not installed. |
| `MODEL_NOT_FIT` | A method requiring a trained model was called before `fit()`, or a diagnostic needs state the loaded model lacks (`analysis_context`, or per-fold pipeline states for SHAP importance with `context["missing"]`). |
| `INCOMPATIBLE_COLUMNS` | A column that was numeric (or bool) at fit arrives at `predict()` with a non-numeric dtype (string, object, category, datetime, ...). `context["columns"]` names each one with its fit and predict dtypes. A missing column raises `DATA_SCHEMA_INVALID`; an extra column is dropped with a warning. |
| `UNSUPPORTED_TASK` | A method is not applicable to the configured task type. |
| `UNSUPPORTED_METRIC` | An unknown or task-incompatible metric name was provided. |
| `METRIC_REQUIRES_PROBA` | A probability metric (`logloss`, `auc`, `auc_pr`, `brier`, `ece`, `precision_at_k`) received values that are not probabilities: non-numeric, non-finite, outside [0, 1], or 1-D for more than two classes. |
| `TUNING_FAILED` | The Optuna study encountered an unrecoverable failure; `tune(resume=True)` with no study to continue; a round ended with no COMPLETE trial in the study; `restore_tuning` with no closed round to restore. |
| `EVALUATION_FAILED` | OOF predictions contain NaN in covered rows, or feval construction failed. |
| `CALIBRATION_NOT_SUPPORTED` | Calibration was requested for a non-binary task or unsupported config. |
| `CALIBRATION_NOT_FITTED` | A calibrator's `predict()` / `export_params()` was called before `fit()`. |
| `CALIBRATION_FAILED` | The Platt or Beta calibrator's `minimize` did not converge (`success` false), so its coefficients are not used. `context` has `calibrator`, `method`, `message`, `status` and `nit`, plus `stage` (`"cross_fit"` with `fold`, or `"c_final"`) when raised from `Model.fit`. |
| `SERIALIZATION_FAILED` | `export()` encountered an I/O error or could not resolve a path. |
| `DESERIALIZATION_FAILED` | `load()` encountered a validation or I/O error. |
| `TARGET_NOT_NUMERIC` | `task: regression` with a non-numeric target column. |
| `TARGET_UNSEEN_LABEL` | The target encoder is given a classification label it was not fitted on, or a class code outside the fitted classes to decode. |

## Leakage validators (`lizyml.data`)

Optional, explicitly-called leakage checks (they are **not** auto-run by
`Model.fit`). Each returns a list of warning messages, or raises
`LizyMLError(LEAKAGE_SUSPECTED / LEAKAGE_CONFIRMED)` when `raise_on_violation=True`
(the default).

`validate_no_target_leakage` checks the columns in order. If a column cannot be
compared with the target (for example an extension array whose comparison
raises), it raises `LizyMLError(DATA_SCHEMA_INVALID)` naming the column, whatever
`raise_on_violation` is.

Both `validate_no_target_leakage` and `validate_time_series_order` first check
that the column they are named for (`target` / `time_col`) is in the frame. If
it is not, they raise `LizyMLError(DATA_SCHEMA_INVALID)` before checking
anything, whatever `raise_on_violation` is. `context` names the column under
`target` / `time_col`, together with `missing_columns` and `available_columns`
(#311). So a returned list (`[]` or warnings) always means the check ran to the
end.

```python
from lizyml.data import (
    validate_time_series_order,
    validate_no_target_leakage,
    validate_group_split,
)

validate_time_series_order(df, time_col="date")        # non-decreasing time order
validate_no_target_leakage(df, target="y")             # no column ≡ target
validate_group_split(groups, train_idx, valid_idx)     # no group in both folds
```
