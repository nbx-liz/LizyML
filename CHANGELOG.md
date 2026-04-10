# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.8.1] - 2026-04-11

### Fixed

- **ECE formula corrected** (H-0067) — ECE per-bin accuracy now uses `mean(y_true)` (fraction of positives) instead of binarized-prediction accuracy. The old formula systematically overestimated ECE for well-calibrated models. Same fix applied to codegen templates.
- **Confusion matrix NaN exclusion** (H-0067) — `confusion_matrix_table()` now applies `compute_oof_valid_mask()` to exclude structurally uncovered rows (e.g., TimeSeriesCV first period) from the OOS matrix. Previously, NaN predictions were silently treated as class 0.
- **Leakage validator eval order** (H-0067) — `validate_no_target_leakage()` now checks NaN positions (`isna().equals()`) before `np.allclose()`, preventing a silent `ValueError` swallow when columns have NaN at different positions.
- **Isotonic calibrator log suppression** (H-0067) — Changed `lgbm.log_evaluation(period=0)` to `period=-1` for well-defined behavior in LightGBM 4.x.
- **RefitTrainer pipeline leakage boundary** (H-0067) — Pipeline is now fitted on inner-train rows only (consistent with CVTrainer). A second pipeline is fitted on all data for the final `pipeline_state` used at inference. `categorical_features` sourced from the full-data pipeline.
- **Cross-fit calibration NaN guard** (H-0067) — `cross_fit_calibrate()` now guards against NaN in validation indices. Finite rows go to `cal.predict()`, NaN rows fall back to uncalibrated OOF predictions.
- **Calibrated metrics include oof_per_fold** (H-0067) — `metrics["calibrated"]` now includes `oof_per_fold` in addition to `oof`. IF metrics remain excluded (leakage risk).
- **Inner validation empty train guard** (H-0067) — `HoldoutInnerValid` and `TimeHoldoutInnerValid` now raise `ValueError` when `n_valid >= n_samples` instead of producing an empty training set.

## [0.8.0] - 2026-04-03

### Added

- **Codegen feval metric support** (H-0066) — `export_code()` now preserves feval metrics (f1, brier, ece, precision_at_k, accuracy, rmsle, r2) in generated code. Previously, feval metrics were silently dropped during code generation, causing `metric="None"` in config.json and incorrect early stopping behavior.
  - `config.json` gains a `feval_metrics` field with metric metadata (name, params, greater_is_better, needs_proba)
  - `train.py` template includes pure numpy/sklearn feval implementations and a `build_feval_from_config()` factory
  - Backward compatible: empty `feval_metrics` produces identical output to previous versions
- **New estimator implementation guide** — `docs/add-estimator-guide.md` documents all requirements for adding non-LightGBM estimators: adapter, provider, config, metric bridge, codegen, and test checklist

## [0.7.3] - 2026-04-02

### Fixed

- **Tune → Fit exact identity** — Unified tune objective and fit code paths so both go through `_build_train_components(training_overrides=...)`. Previously, the tune objective rebuilt the estimator factory with pre-smart-resolution params (`merged_model`) when `early_stopping_rounds` was in the search space, causing `num_leaves` and `scale_pos_weight` to be missing. Tune and fit now produce bit-for-bit identical OOF scores.

## [0.7.2] - 2026-04-02

### Fixed

- **Tune → Fit parameter identity** (#76) — `default_fixed_params()` was leaking `auto_num_leaves` (a smart param) into model params during tuning. Additionally, `best_training_params` (`early_stopping_rounds`, `validation_ratio`) from tuning were not applied during subsequent `fit()`. Both issues caused score divergence between tune best trial and fit OOF. After fix, tune and fit produce identical model params and near-identical OOF scores (within LightGBM floating-point tolerance).

## [0.7.1] - 2026-04-02

### Fixed

- **Categorical search space validation** — `parse_space()` now rejects non-scalar choices (e.g. nested lists from YAML `- [auc, binary_logloss]`) with a clear `CONFIG_INVALID` error and a hint for the correct YAML format. Previously, such values passed through to Optuna's `suggest_categorical()` causing repeated warnings during tuning.

## [0.7.0] - 2026-03-28

### Added

- **Parameterised MetricEntry** (H-0065) — `precision_at_k` `k` parameter is now user-configurable via dict form in both `evaluation.metrics` and `model.lgbm.params.metric`
  - `metrics: ["auc", {precision_at_k: {k: 20}}]` sets top-K% cutoff
  - Evaluation and Model Params support independent `k` values
  - `params_summary()` displays feval metric parameters (e.g. `precision_at_k (k=20)`)
  - Learning curve subplot titles show parameterised metric names
  - Plain string `"precision_at_k"` continues to use default `k=10` (backward compatible)
  - Invalid metric parameters now raise `LizyMLError(CONFIG_INVALID)` instead of raw `ValueError`

## [0.6.1] - 2026-03-28

### Fixed

- **`r2` metric with early stopping** — `r2` was listed as a LightGBM native metric but is not implemented in LightGBM 4.6.0 (only in unreleased master). Passing `metric: "r2"` silently produced empty eval results, breaking early stopping. Moved `r2` from native whitelist to feval (custom function) so it works correctly with early stopping and learning curves.

## [0.6.0] - 2026-03-28

### Added

- **Metric bridge** (`metric_bridge.py`) — unified metric handling for LightGBM training (H-0064, #57, #58, #59)
  - LizyML metric names auto-translate to LightGBM equivalents (`logloss` → `binary_logloss`, `auc_pr` → `average_precision`)
  - Per-task whitelist validation before `lgb.train()` with clear error messages
  - Custom feval support for LizyML-only metrics: `rmsle`, `f1`, `brier`, `ece`, `precision_at_k`, `accuracy`
  - Native + feval metrics can be mixed (e.g. `params={"metric": ["auc", "f1"]}`)
- 64 new tests: metric mapping, whitelist validation, feval numerical correctness, behavioral training tests
- `docs/config-reference.md`: training metric reference, metric details table, two-system explanation

### Changed

- `_build_params()` now returns `(params, num_boost_round, feval_list)` — 3-element tuple
- `fit()` passes feval callables to `lgb.train(feval=...)` when custom metrics are specified
- Invalid metric names are now rejected at `_build_params()` time (pre-validation) instead of relying on LightGBM post-hoc detection

## [0.5.0] - 2026-03-28

### Added

- `LGBMConfig.params.metric` — user-specified LightGBM evaluation metric override with task-default fallback (H-0061, #50, #51)
- `plot_learning_curve(*, metrics=None)` — filter displayed subplots by metric name (H-0062, #52)
- `Model.plot_learning_curve(*, metrics=None)` — pass-through for metrics filter
- `params_summary()` now includes `metric` in output rows
- Silent invalid metric detection: `UserWarning` when user metric produces no eval results
- 70 new tests: metric override, learning curve filter, Config propagation + behavioral effect (H-0063)

### Fixed

- Variable shadowing bug in `plot_learning_curve()` — loop variable `metrics` overwrote the function parameter (pre-existing, exposed by H-0062)
- Empty string metric (`[""]`) now correctly falls back to task default

### Changed

- `_build_params()` no longer strips user-specified `metric` from params dict
- Error handling split: `LightGBMError` (metric keyword) and `ValueError` (eval metric) caught separately for precise diagnostics

### Removed

- Duplicate `_FIXED_METRIC` dict in `defaults.py` — consolidated into `_TASK_METRIC`

## [0.4.2] - 2026-03-21

### Added

- `CONTRIBUTING.md` — development workflow, quality gates, and spec-first process
- `SECURITY.md` — vulnerability reporting policy
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- `Makefile` — unified development commands (`make ci`, `make test`, etc.)
- `.editorconfig` — cross-editor formatting consistency
- `.github/dependabot.yml` — automated dependency updates (pip + GitHub Actions)
- `.github/PULL_REQUEST_TEMPLATE.md` — PR checklist template
- `.github/ISSUE_TEMPLATE/` — bug report and feature request templates

## [0.4.1] - 2026-03-21

### Changed

- Rewrote README: 620 → 190 lines with badges, installation, quick start, and architecture diagram
- Extracted Config Reference to `docs/config-reference.md` (384 lines)

### Added

- `scripts/release.py` — automated release script (CHANGELOG validation, commit, push, PR creation)
- `.github/workflows/auto-release.yml` — auto-tag and GitHub Release on merge to main
- GitHub Releases for all past versions (v0.1.0–v0.4.0)
- `CONTRIBUTING.md` — development workflow, quality gates, and spec-first process
- `SECURITY.md` — vulnerability reporting policy
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- `Makefile` — unified development commands (`make ci`, `make test`, etc.)
- `.editorconfig` — cross-editor formatting consistency
- `.github/dependabot.yml` — automated dependency updates (pip + GitHub Actions)
- `.github/PULL_REQUEST_TEMPLATE.md` — PR checklist template
- `.github/ISSUE_TEMPLATE/` — bug report and feature request templates

## [0.4.0] - 2026-03-21

### Added

- `blocked_group_kfold` split method — 2-axis cross-validation combining period-block splitting with group KFold (H-0060)
- `BlockedGroupKFoldSplitter` — new splitter with expanding/sliding window modes and cutoff-based period boundaries
- `BlockedGroupKFoldConfig` with nested `blocks` (col/cutoffs/mode/train_window) and `groups` (col/n_splits/stratify/shuffle) sections
- `BlockedGroupInnerValid` — group-isolated, time-ordered, stratified inner validation for early stopping
- `StratifiedTimeHoldoutInnerValid` — per-class tail selection fallback for inner validation when fewer than 4 groups
- 62 new tests (config, splitter, inner valid, factory, E2E) with 100% splitter coverage

## [0.3.0] - 2026-03-20

### Added

- `Model.export_code(path)` — generate LizyML-independent training and prediction scripts (H-0059)
- `lizyml/codegen/` package — config_writer, artifact_writer, templates, generator modules
- Exported output: `train.py`, `predict.py`, `test_equivalence.py`, `config.json`, `requirements.txt`, `artifacts/`
- Supports all task types (regression, binary, multiclass) and all calibrators (Platt, Beta, Isotonic)
- `BaseCalibratorAdapter.export_params()` — abstract method for calibrator parameter export
- `PlattCalibrator.export_params()`, `BetaCalibrator.export_params()`, `IsotonicCalibrator.export_params()` + `save_model_text()`
- `LGBMAdapter.save_model_text()` — export Booster to human-readable text format
- `NativeFeaturePipeline.export_state_json()` — export pipeline state to JSON
- 73 new codegen tests including E2E equivalence verification (5 patterns)

## [0.2.0] - 2026-03-17

### Added

- `StratifiedGroupKFoldSplitter` — new split method combining stratification with group boundaries (`split.method: "stratified_group_kfold"`) (H-0055)
- `metrics["raw"]["oof_coverage"]` — float (0.0–1.0) indicating the fraction of rows covered by OOF validation folds; `evaluate_table().attrs["oof_coverage"]` for programmatic access (H-0057)
- `compute_oof_valid_mask()` — derives OOF coverage mask from split indices, not NaN detection; NaN in covered rows raises `ValueError` for bug detection (H-0057)

### Changed

- Calibration cross-fit now reuses outer CV split indices directly instead of generating independent splits (H-0058)
- `CalibrationConfig.n_splits` is deprecated — non-default values emit `UserWarning` and are ignored (H-0058)
- `build_calibration_splitter()` is deprecated with `DeprecationWarning` (H-0058)
- OOF metrics are computed on covered rows only; TimeSeriesCV first-period rows (never validated) are excluded instead of propagating NaN (H-0057)
- `cross_fit_calibrate()` handles NaN training rows with identity fallback to `oof_pred` probabilities (H-0058)

### Internal

- 5-layer DAG architecture migration: dead code removal (H-0051), layer dependency purification (H-0052), `EstimatorProvider` protocol introduction (H-0053)
- `TrainComponents` frozen dataclass, `resolve_smart_params` dict unification, `TuningResult` 3-way category split (H-0050)
- `EstimatorProvider` extensibility: `params_summary`, `set_categorical_features`, provider-level factory dispatch (H-0054)
- Systematic test reinforcement: 92 new tests across 5 categories — config propagation, facade orchestration, provider invariants, artifact compatibility, tuning reproducibility, dtype boundaries, pairwise parameters (H-0056)

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
