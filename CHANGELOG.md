# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.5] - 2026-03-15

### Fixed

- Calibration cross-fit OOF array now NaN-initialized instead of `np.empty` — prevents silent garbage values for time-series splitters
- `GroupTimeSeriesSplitter` last fold now extends to include all trailing groups (previously silently dropped)
- `ECE` metric last bin is now right-inclusive (`y_pred == 1.0` no longer excluded)
- `RMSLE` raises `LizyMLError` for negative predictions/targets instead of producing NaN
- `FitResult` post-construction mutation replaced with `dataclasses.replace()`
- `_prepare_training_data` no longer mutates `DataFrameComponents` in-place
- `evaluate()` bare `assert` replaced with proper `LizyMLError`
- `_filter_metrics` removes empty branches after filtering
- Task-locked `objective`/`metric` can no longer be overridden by user search space params
- `LGBMAdapter.update_params` creates new dict instead of mutating in-place
- `compute_shap_importance` handles empty models list without `ZeroDivisionError`
- QQ plots raise `LizyMLError(OPTIONAL_DEP_MISSING)` instead of bare `ImportError` when scipy is missing
- Tuner trial failures now logged via warning callback; catch tuple narrowed from `Exception` to specific types
- `TuningResult`/`TrialResult` deep-copy mutable `dict`/`list` fields in `__post_init__`
- `HoldoutInnerValid` `n_valid` uses `ceil` to match `HoldoutSplitter` rounding
- All timestamps now include UTC timezone info
- `params_table` guards against empty models list

### Changed

- CI test matrix now includes Python 3.11
- Added `[tool.coverage.run]` and `[tool.coverage.report]` configuration to `pyproject.toml`
- `PredictionResult.proba` docstring corrected for multiclass shape
- `cross_fit_calibrate` docstring notes raw score (logit) support

## [0.1.4] - 2026-03-14

### Fixed

- Multiclass OVA (`multiclassova`) predictions now correctly pass `roc_auc_score` validation; row-wise normalization applied only to simplex-required metrics (AUC, LogLoss) (H-0049)

### Added

- `BaseMetric.needs_simplex` property (default `False`) to distinguish metrics requiring probability distributions (sum=1) from per-class OvR metrics (H-0049)
- `AUC` and `LogLoss` override `needs_simplex=True`; per-class metrics (`AUCPR`, `Brier`) keep raw predictions (H-0049)

## [0.1.3] - 2026-03-14

### Added

- `IsotonicCalibrator` migrated to LightGBM native Booster API with early stopping and internal validation split (H-0047)
- `TuneProgressInfo` / `TuneProgressCallback` for `Model.tune(progress_callback=fn)` (H-0048)

### Fixed

- Remove double-sigmoid in `IsotonicCalibrator.predict()` — `Booster.predict()` already returns probabilities (H-0047)

## [0.1.2] - 2026-03-10

### Changed

- Calibration cross-fit splits now inherit `split.method` and its parameters (group/time/purge/embargo boundaries); only fold count is overridden by `calibration.n_splits` (H-0044)
- `evaluate()` now returns `raw.oof_per_fold` metrics computed on each outer fold's valid indices; `evaluate_table()` fold columns changed from IF to OOF-per-fold (H-0045)
- Calibration split failure now raises `LizyMLError(CONFIG_INVALID)` with `split_method`, `calibration_n_splits`, `n_samples`, and `n_groups` (when applicable) in context
- BLUEPRINT §13.4: IF/OOF classification for diagnostic vs generalization monitoring APIs (H-0046)

### Added

- Contract tests for `purged_time_series` calibration splits (purge_gap + embargo boundary verification)
- Contract tests for `group_time_series` calibration splits (group disjointness + temporal ordering)
- Golden test coverage for `oof_per_fold` in metrics structure
- README and notebook documentation for calibration split.method inheritance contract

## [0.1.1] - 2026-03-08

### Changed

- Decompose Model facade into mixins: ModelPlotsMixin, ModelTablesMixin, ModelPersistenceMixin, factory functions (H-0042)
- Consolidate test helpers into `tests/_helpers.py`; remove ~40 duplicated definitions (H-0043)
- Enhance pytest parametrize usage for common task-agnostic tests (H-0043)
- CI now runs on develop branch PRs; slow tests excluded for develop, included for main (H-0043)
- Default `pytest` run skips slow tests via `addopts`; use `-m ""` for all tests (H-0043)
- Add `--cov-fail-under=95` coverage threshold to CI (H-0043)

## [0.1.0] - 2026-03-07

### Added

- Config-driven ML pipeline for regression, binary, and multiclass classification
- LightGBM estimator adapter using native Booster API
- Cross-validation training with OOF/IF predictions
- Inner validation (early stopping) support
- Feature pipeline with leakage prevention
- Splitters: KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, Holdout
- Calibration: Platt, Isotonic, Beta (cross-fit, OOF-only)
- Evaluation with pre-computed metrics (raw + calibrated)
- SHAP explanations (optional dependency)
- Optuna-based tuning with unified search space (optional dependency)
- Plotly-based visualizations: learning curve, importance, OOF distribution, residuals (optional dependency)
- Export/load with format_version=1 and metadata
- Simulate (bootstrap prediction distributions)
- YAML/JSON config loading with pydantic validation
