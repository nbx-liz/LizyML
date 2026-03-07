# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
