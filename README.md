# LizyML

[![CI](https://github.com/nbx-liz/LizyML/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nbx-liz/LizyML/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/lizyml.svg)](https://pypi.org/project/lizyml/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Config-driven ML library that unifies **tune / fit / predict / evaluate / export** for regression, binary classification, and multiclass classification.

## Key Features

- **One config, full pipeline** -- A single dict/YAML/JSON drives splitting, training, tuning, evaluation, and export. No boilerplate orchestration code.
- **Reproducibility by default** -- Seed, split indices, params, library versions, and data fingerprint are captured automatically in every run.
- **Leakage-aware CV and calibration** -- OOF predictions never see their own training rows. Calibration uses cross-fit on the same outer splits. Time and group constraints propagate to inner validation.
- **8 CV strategies** -- KFold, Stratified, Group, StratifiedGroup, TimeSeries, Purged TimeSeries, Group TimeSeries, and 2-axis Blocked Group KFold.
- **Stable result contracts** -- `FitResult`, `PredictionResult`, and artifact formats have fixed schemas. Downstream code never breaks on shape changes.
- **Codegen export** -- Generate standalone `train.py` + `predict.py` that run without LizyML installed.
- **Optional extras** -- Tuning (Optuna), SHAP explanations, Plotly visualizations, and Beta calibration (scipy) are all opt-in.

## Installation

```bash
pip install lizyml
```

### Extras

```bash
pip install 'lizyml[tuning]'        # Optuna hyperparameter search
pip install 'lizyml[explain]'        # SHAP explanations
pip install 'lizyml[plots]'          # Plotly visualizations
pip install 'lizyml[calibration]'    # Beta calibrator (scipy)
pip install 'lizyml[tuning,explain,plots,calibration]'  # all extras
```

### System Requirements

- Python 3.10+
- LightGBM native library (`libgomp` on Linux: `apt-get install libgomp1`)

### Development install

```bash
git clone https://github.com/nbx-liz/LizyML.git
cd LizyML
uv sync --group dev
```

## Quick Start

```python
import numpy as np
import pandas as pd
from lizyml import Model

# --- Synthetic data ---
rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "feat_a": rng.normal(size=n),
    "feat_b": rng.normal(size=n),
    "cat_col": rng.choice(["x", "y", "z"], size=n),
    "target": rng.normal(size=n),
})

# --- Config ---
config = {
    "config_version": 1,
    "task": "regression",
    "data": {"target": "target"},
    "features": {"categorical": ["cat_col"]},
    "split": {"method": "kfold", "n_splits": 5},
    "model": {"name": "lgbm"},
    "evaluation": {"metrics": ["rmse", "mae"]},
}

# --- Train, evaluate, predict ---
model = Model(config=config)
fit_result = model.fit(data=df)
metrics = model.evaluate()
print(metrics)  # {"raw": {"oof": {"rmse": ..., "mae": ...}, ...}}

pred = model.predict(df.drop(columns=["target"]))
print(pred.pred[:5])

# --- Save and reload ---
model.export("my_model")
loaded = Model.load("my_model")
loaded.predict(df.drop(columns=["target"]))
```

## Configuration

LizyML accepts configs as Python dicts, JSON files, or YAML files. Environment variables override any key using the `LIZYML__` prefix (e.g., `LIZYML__training__seed=99`).

See **[Config Reference](docs/config-reference.md)** for all keys, defaults, split method guides, and tuning space definitions.

## Codegen Export

Generate LizyML-free scripts for production deployment:

```python
model.export_code("deploy/my_model")
```

Output:
- `train.py` -- retrain on new data with `python train.py data.csv`
- `predict.py` -- run inference with `python predict.py test.csv -o out.csv`
- `config.json` -- all hyperparameters and feature definitions
- `test_equivalence.py` -- verify codegen matches `Model.predict()`
- `artifacts/` -- model files in human-readable formats

Dependencies: `lightgbm`, `numpy`, `pandas`, `scikit-learn` (plus `scipy` when the model uses beta calibration). The generated `requirements.txt` lists exactly what is needed.

## Architecture

LizyML uses a 5-layer architecture where dependencies flow strictly downward:

```
Layer 4  Facade       Model (orchestration only, no logic)
           |
Layer 3  Optional     explain / plots / persistence / codegen
           |
Layer 2  Composition  training / evaluation / tuning
           |
Layer 1  Leaf         config / data / splitters / features / estimators / metrics / calibration
           |
Layer 0  Foundation   types (FitResult, PredictionResult, ...) / exceptions / logging
```

Key rules:
- **Downward-only** dependencies (no circular imports)
- Layer 2 references Layer 1 through **abstract interfaces only**
- Only the Facade (Layer 4) assembles concrete classes

See [ARCHITECTURE.md](ARCHITECTURE.md) for full diagrams and module layout.

## Design Priorities

**Reproducibility** -- Same config + seed = same splits, same OOF predictions, same metrics, **within a fixed `(num_threads, CPU)` environment**. (LightGBM's histogram construction is thread-count sensitive, so bit-identical results across machines with different CPU/thread counts are out of scope.) Every run captures seed, split indices, params, library versions, and a data fingerprint.

**Leakage prevention** -- OOF rows are never seen during training. Calibration cross-fit reuses outer CV splits. Time and group constraints propagate to inner validation (early stopping) and calibration.

**Contract stability** -- `FitResult`, `PredictionResult`, and artifact formats have fixed schemas. Breaking changes require a `format_version` bump and migration path.

## Result Objects

| Object | Key fields |
|---|---|
| `FitResult` | `oof_pred`, `if_pred_per_fold`, `metrics`, `models`, `splits`, `run_meta` |
| `PredictionResult` | `pred`, `proba` (binary), `shap_values` (optional), `warnings` |
| `Model Artifact` | Trained models, pipeline state, calibrator, config, `format_version` |

`model.evaluate()` returns structured metrics:

```python
{
    "raw": {
        "oof": {"rmse": 0.42, "mae": 0.33},
        "if_mean": {"rmse": 0.40, "mae": 0.31},
        "if_per_fold": [...],
        "oof_coverage": 1.0,
    },
    "calibrated": {                             # binary only
        "oof": {"logloss": 0.35},
        "oof_per_fold": [{"logloss": 0.36}, {"logloss": 0.34}, ...],
    },
}
```

See [BLUEPRINT.md](BLUEPRINT.md) for full schemas and invariants.

## Roadmap

- Broader scikit-learn estimator support
- DNN backend (PyTorch)
- Multiclass calibration
- Ranking tasks
- Additional export formats (ONNX, TorchScript)

## Documentation

- [API Reference](docs/api.md) -- public Model API, result objects, and error codes
- [Config Reference](docs/config-reference.md) -- all config keys, defaults, and split guides
- [Examples & Tutorials](docs/examples.md) -- Jupyter notebook index
- [FAQ / Troubleshooting](docs/faq.md) -- common questions and error resolution
- [Migration Guide](docs/migration.md) -- upgrading between versions
- [BLUEPRINT.md](BLUEPRINT.md) -- implementation specification (source of truth)
- [ARCHITECTURE.md](ARCHITECTURE.md) -- 5-layer architecture diagrams
- [CHANGELOG.md](CHANGELOG.md) -- release history
- [HISTORY.md](HISTORY.md) -- proposal and decision records

## Contributing

1. Fork the repo and create a branch from `develop`
2. Run quality gates: `uv run ruff check . && uv run mypy lizyml/ && uv run pytest`
3. Open a PR against `develop`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Language Policy

- **Specs** (`BLUEPRINT.md`, `HISTORY.md`, `ARCHITECTURE.md`): Japanese
- **Code, docstrings, commit messages, PRs, user-facing docs**: English

## License

[MIT](LICENSE)
