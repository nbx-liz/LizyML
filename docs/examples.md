# Notebook Index

<!-- Generated from each notebook's metadata.lizyml.index by scripts/examples_index.py. Do not edit this region by hand. rows=8 sha256=6ed9207d0f7a2932e1b3b5b6aa38876478af9cc0050cb94b3d87ef75f64feb14 -->

| Notebook | Demonstrates | Extras required |
|---|---|---|
| `tutorial_binary_lgbm.ipynb` | `calibration_plot()`, `confusion_matrix()`, `evaluate_table()`, `importance_plot()`, `probability_histogram_plot()`, `roc_curve_plot()` | `pip install 'lizyml[explain,plots]'` |
| `tutorial_calibration.ipynb` | `calibration_plot()`, `evaluate()`, `evaluate_table()`, `probability_histogram_plot()` | `pip install 'lizyml[plots]'` |
| `tutorial_codegen_export.ipynb` | `export_code()`, `predict()` | none (base install) |
| `tutorial_multiclass_lgbm.ipynb` | `confusion_matrix()`, `evaluate_table()`, `importance_plot()`, `plot_learning_curve()`, `roc_curve_plot()` | `pip install 'lizyml[explain,plots]'` |
| `tutorial_regression_lgbm.ipynb` | `evaluate_table()`, `fit()`, `importance()`, `importance_plot()`, `params_table()`, `plot_learning_curve()`, `residuals_plot()` | `pip install 'lizyml[explain,plots]'` |
| `tutorial_regression_tuning_lgbm.ipynb` | `boundary_table()`, `fit()`, `params_table()`, `tune()`, `tuning_plot()`, `tuning_table()` | `pip install 'lizyml[plots,tuning]'` |
| `tutorial_shap_explanations.ipynb` | `importance_plot()`, `predict()` | `pip install 'lizyml[explain,plots]'` |
| `tutorial_time_series_lgbm.ipynb` | `evaluate_table()`, `plot_learning_curve()`, `predict()`, `split_summary()` | `pip install 'lizyml[plots]'` |

<!-- index:end -->

All notebooks are located in the `notebooks/` directory. They can be run
with any Jupyter-compatible environment. Install the extras each notebook
lists before running it (see [Installing Extras](#installing-extras)).

## Available Notebooks

### `tutorial_regression_lgbm.ipynb`

End-to-end regression walkthrough: config definition, `fit()`,
`evaluate_table()`, `plot_learning_curve()`, `residuals_plot()`,
`importance()` / `importance_plot()` by split, gain and SHAP, and
`params_table()`. Good starting point if you are new to LizyML.

---

### `tutorial_binary_lgbm.ipynb`

Binary classification with LightGBM and isotonic calibration:
`evaluate_table()` with raw and calibrated metrics, `roc_curve_plot()`,
`confusion_matrix()`, `probability_histogram_plot()`, `calibration_plot()`,
and feature importance.

---

### `tutorial_multiclass_lgbm.ipynb`

Multiclass classification with stratified CV: `evaluate_table()`,
`confusion_matrix()`, `roc_curve_plot()` (one-vs-rest, per-class AUC),
`importance_plot()`, and `plot_learning_curve()`.

---

### `tutorial_regression_tuning_lgbm.ipynb`

Hyperparameter tuning with Optuna: `tune()` → `fit()` workflow,
`tuning_table()`, `tuning_plot()`, `boundary_table()` and `params_table()`.
Includes a `progress_callback` example (`TuneProgressInfo`) for tracking
trial progress.

---

### `tutorial_time_series_lgbm.ipynb`

Time-series cross-validation: the `time_series` splitter (expanding-window
CV) and `purged_time_series` with `purge_gap`, `split_summary()`,
`evaluate_table()`, `plot_learning_curve()`, and `predict()` on the last 100
rows of the frame.

---

### `tutorial_shap_explanations.ipynb`

SHAP value computation and interpretation: `predict(return_shap=True)` for
per-sample explanations, `importance_plot(kind="shap")` for global
feature importance, and comparison of split vs gain vs SHAP rankings.

---

### `tutorial_calibration.ipynb`

Probability calibration for binary classification with Platt and Isotonic
(Beta is listed as a third option and selected the same way). Compares raw
vs calibrated metrics with `evaluate()` / `evaluate_table()`, and visualizes
them with `calibration_plot()` and `probability_histogram_plot()`.

---

### `tutorial_codegen_export.ipynb`

Codegen export walkthrough: `export_code()` generates standalone
`train.py` + `predict.py` + `config.json` that run without LizyML.
Shows the generated file structure, runs `predict.py`, and checks it with
`test_equivalence.py` against `predict()` reference predictions.

---

## Installing Extras

```bash
# Plots (plotly): every *_plot method
pip install 'lizyml[plots]'

# SHAP: importance(kind="shap") and predict(return_shap=True)
pip install 'lizyml[explain]'

# Tuning (Optuna): tune()
pip install 'lizyml[tuning]'

# All extras
pip install 'lizyml[tuning,explain,plots,calibration]'
```

Platt and Beta calibration are fitted with scipy, which the base install
already brings in through scikit-learn; the `calibration` extra declares it
explicitly.
