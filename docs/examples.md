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

### `tutorial_shap_explanations.ipynb`

SHAP value computation and interpretation: `predict(return_shap=True)` for
per-sample explanations, `importance_plot(kind="shap")` for global
feature importance, and comparison of split vs gain vs SHAP rankings.

**Extras required:** `pip install 'lizyml[explain]'`

---

### `tutorial_calibration.ipynb`

Probability calibration for binary classification: Platt, Isotonic, and
Beta methods. Compares raw vs calibrated metrics (logloss, brier, ece),
visualizes with `calibration_plot()` and `probability_histogram_plot()`.

**Extras required:** `pip install 'lizyml[calibration]'` (for Beta method)

---

### `tutorial_codegen_export.ipynb`

Codegen export walkthrough: `export_code()` generates standalone
`train.py` + `predict.py` + `config.json` that run without LizyML.
Shows generated file structure and equivalence verification with
`test_equivalence.py`.

**Extras required:** none (base install)

---

## Installing Extras

```bash
# Tuning support (Optuna)
pip install 'lizyml[tuning]'

# SHAP explanations
pip install 'lizyml[explain]'

# Calibration (Beta method requires scipy)
pip install 'lizyml[calibration]'

# All extras
pip install 'lizyml[tuning,explain,plots,calibration]'
```
