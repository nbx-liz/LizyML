# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
