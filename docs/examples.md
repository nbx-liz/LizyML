# Notebook Index

All notebooks are located in the `notebooks/` directory. They can be run
with any Jupyter-compatible environment. Install the required extras before
running (see the table below).

## Available Notebooks

### `tutorial_regression_lgbm.ipynb`

End-to-end regression walkthrough: config definition, `fit()`, `evaluate()`,
`importance()`, `residuals_plot()`, and `export()`. Good starting point if
you are new to LizyML.

**Extras required:** none (base install)

---

### `tutorial_binary_lgbm.ipynb`

Binary classification with LightGBM: ROC curve, confusion matrix, probability
histogram, and OOF coverage interpretation. Covers the full `evaluate()`
output including the `"calibrated"` section.

**Extras required:** none (base install)

---

### `tutorial_multiclass_lgbm.ipynb`

Multiclass classification: per-class metrics, `confusion_matrix()`,
`roc_curve_plot()`, and `importance_plot()`. Demonstrates how to interpret
`oof_per_fold` across multiple classes.

**Extras required:** none (base install)

---

### `tutorial_regression_tuning_lgbm.ipynb`

Hyperparameter tuning with Optuna: `tune()` → `fit()` workflow,
`tuning_table()`, `tuning_plot()`, and `params_table()`. Includes a
`TuneProgressCallback` example for tracking trial progress.

**Extras required:** `pip install 'lizyml[tuning]'`

---

### `tutorial_time_series_lgbm.ipynb`

Time-series cross-validation: `time_series` and `purged_time_series`
splitters, `split_summary()`, `oof_coverage` interpretation, and
`plot_learning_curve()`. Demonstrates expanding-window CV and how to
detect temporal leakage with OOF coverage.

**Extras required:** none (base install)

---

## Installing Extras

```bash
# Tuning support (Optuna)
pip install 'lizyml[tuning]'

# SHAP explanations
pip install 'lizyml[explain]'

# All extras
pip install 'lizyml[tuning,explain]'
```

---

## Planned Notebooks

The following notebooks are planned for future releases:

- **`tutorial_shap_explain.ipynb`** — SHAP value computation via
  `predict(return_shap=True)` and `importance(kind="shap")`.
  Requires `lizyml[explain]`.

- **`tutorial_calibration.ipynb`** — Isotonic and Platt calibration for
  binary classification: calibration plot, ECE metric, and how
  `oof_coverage` interacts with calibrated OOF metrics.

- **`tutorial_export_code.ipynb`** — `export_code()` walkthrough: generating
  LizyML-independent `train.py` / `predict.py` and verifying equivalence with
  `test_equivalence.py`.
