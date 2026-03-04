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
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 64, 128],
            },
        }
    },
    "evaluation": {
        "metrics": ["rmse", "mae"],
    },
}
```

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
