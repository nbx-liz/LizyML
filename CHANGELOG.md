# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Leakage validators are now public API on `lizyml.data`** (H-0087, [#216](https://github.com/nbx-liz/LizyML/issues/216)). `validate_time_series_order`, `validate_no_target_leakage`, and `validate_group_split` (previously dead code with no call sites) are re-exported from `lizyml.data` and documented, so users can run explicit time-order / target-leakage / group-overlap checks in line with the leakage-first charter. They are not auto-wired into `Model.fit` (that behavior change is deferred to a future proposal). The empty, unused `lizyml.utils` package was removed. Additive; no behavior change.
- **Contract types and the unified exception are now importable from the top-level package** (H-0086, [#213](https://github.com/nbx-liz/LizyML/issues/213)). `FitResult`, `PredictionResult`, `TuningResult`, `LizyMLError`, `ErrorCode`, `load_config`, and `TaskType` are re-exported from `lizyml` (previously only reachable via the private-looking `lizyml.core.*` paths), and `DataFingerprint` is re-exported from `lizyml.core.types`. Users can now write `from lizyml import FitResult, LizyMLError` for type annotations and `except LizyMLError` handling. Purely additive; a golden test pins the top-level `__all__`.
- **Tuned parameters are now persisted and restored across `export()` / `Model.load()`** (H-0086, [#215](https://github.com/nbx-liz/LizyML/issues/215)). `export()` records the tuned-param overlay (`best_model_params` / `best_smart_params` / `best_training_params` + score / metric / direction) under a `tuning` block in `metadata.json`, and `Model.load()` restores it into `_tuning_result`. A re-`fit()` after `load()` now reproduces the tuned params instead of silently reverting to config defaults. Additive and back-compatible — non-tuned and pre-#215 artifacts have no `tuning` block and load with `_tuning_result = None`; `format_version` stays `2`. (Restoring the full optuna study for a complete `tune(resume=True)` from a loaded model remains a follow-up.)

### Changed (potentially breaking)

- **Unified the inner-valid (early-stopping) pipeline fit boundary** (H-0085, [#208](https://github.com/nbx-liz/LizyML/issues/208)). `RefitTrainer` now fits the feature pipeline **once on the full dataset** — the same outer-train boundary `CVTrainer` already uses — instead of fitting on the inner-train slice and then refitting a second pipeline on all data. This removes the double fit and aligns `best_iteration` selection with the CV folds. OOF predictions are unaffected (the y-free `NativeFeaturePipeline` never let outer-valid rows into the fit); however, a model's `best_iteration` — and therefore the refit model — can change for configs that use early stopping. Resolves a BLUEPRINT self-contradiction (§6.2 / §10.3.2). `format_version` unchanged.
- **`purge_gap` / `embargo` / `gap` now propagate to the early-stopping (inner-valid) boundary** (H-0085, [#212](https://github.com/nbx-liz/LizyML/issues/212)). For an auto-resolved inner-valid split, `purged_time_series` purges `purge_gap + embargo` rows and `time_series` purges `gap` rows between inner-train and inner-valid, instead of placing them directly adjacent (zero gap). This closes a look-ahead leak that biased `best_iteration` on every fold when the target is constructed from future windows. OOF predictions are unaffected. Because the early-stopping split changes, `best_iteration` — and therefore the trained model — can change for `time_series` / `purged_time_series` configs with a non-zero gap/purge/embargo and early stopping enabled. `format_version` unchanged.

### Fixed

- **Bare `ValueError` / `RuntimeError` in user-reachable paths are now unified `LizyMLError`s** ([#214](https://github.com/nbx-liz/LizyML/issues/214)). The NaN-in-covered-OOF guard in `Evaluator` (reachable from `Model.fit()`) now raises `LizyMLError(EVALUATION_FAILED)` with `nan_count` / `nan_indices` context; the calibrators' not-fitted guards (`platt` / `isotonic` / `beta`, reachable via `fit_result.calibrator`) raise `LizyMLError(CALIBRATION_NOT_FITTED)` with a `calibrator` tag; and the LightGBM feval-construction guards raise `LizyMLError(EVALUATION_FAILED` / `CONFIG_INVALID)`. `except LizyMLError` now catches all of them. Two additive `ErrorCode` members (`EVALUATION_FAILED`, `CALIBRATION_NOT_FITTED`) were added. Callers that caught the bare `ValueError` / `RuntimeError` should switch to `LizyMLError`.
- **An explicit `inner_valid` strategy now survives a config round-trip** (H-0086, [#203](https://github.com/nbx-liz/LizyML/issues/203)). `model_dump()` always re-emits the computed `validation_ratio`, which previously flipped the explicitness heuristic on reload so an explicit `inner_valid` (e.g. `time_holdout` / `group_holdout`) was silently replaced by split-derived auto-resolution on every `dump -> reload` and `export -> load -> fit` — leakage-relevant for time/group data. `EarlyStoppingConfig` now emits a round-trip-safe `inner_valid_explicit` marker (popped on re-validation like `validation_ratio`) that preserves the user's explicit choice. Additive: pure-legacy `validation_ratio` input keeps its auto-resolve semantics, and configs dumped before this change fall back to the previous heuristic. No new settable input field.
- **`fit()` and `load()` no longer expose the internal `FitResult` by reference** (H-0086, [#204](https://github.com/nbx-liz/LizyML/issues/204)). `fit()` — the primary access path — now returns a selective deep copy (the same isolation the `fit_result` property already applied under H-0082), and `Model.load()` deep-copies the restored metrics dict. Mutating a returned result can no longer corrupt internal state or contaminate a later `export()`'s `metadata.json`. Trained estimators (`models` / `calibrator` / `pipeline_state`) stay shared by reference as before; the return type is unchanged.
- **Config validation hardening** (H-0085, [#210](https://github.com/nbx-liz/LizyML/issues/210)):
  - **`config_version` string bypass** — a string value (e.g. `"999"`) previously skipped the supported-version gate and was then lax-coerced by pydantic, loading an unsupported version silently. The gate now coerces to `int` before the check, so unsupported versions are rejected as `CONFIG_VERSION_UNSUPPORTED` regardless of input type.
  - **Legacy `embargo_pct` / `gap` truncation** — a fractional legacy value (e.g. `embargo_pct=0.05`) was migrated via `int()` and silently collapsed to `0`, removing the leakage guard. Fractional legacy values are now rejected with `CONFIG_INVALID` and guidance to supply an integer observation count; integer-valued inputs still migrate.
  - **Shuffled `inner_valid` under a time-ordered outer split** — an explicit `inner_valid.method="holdout"` (shuffled) combined with a `time_series` / `purged_time_series` outer split now emits a `UserWarning` (the temporally-leaked early-stopping split is otherwise silent). Behavior is unchanged — the explicit choice is still honored.
- **A NaN in a numeric / regression target is now rejected** (H-0085, [#207](https://github.com/nbx-liz/LizyML/issues/207)). Previously the `Model.fit` contract was undefined for a NaN numeric target (only string classification targets were validated), so it was silently accepted and could corrupt training. It now raises `LizyMLError(DATA_SCHEMA_INVALID)` with a `nan_count` context, symmetric with the existing string-target check.

### Internal

- **Fail-closed leakage tests** (H-0085, [#207](https://github.com/nbx-liz/LizyML/issues/207)) — replaced a tautological OOF leakage assertion with a trap that records each fold estimator's training rows and asserts disjointness from its validation rows; added traps for per-fold calibration boundaries (calibrator fit on the train slice, not the scored valid rows) and inner-index containment (relative to each outer train fold).

### Internal

- **Plot optional-dependency guard de-duplicated** ([#218](https://github.com/nbx-liz/LizyML/issues/218)). The three per-module `_require_plotly` / `_check_plotly` guards (two names, three message strings) are replaced by a single `lizyml/plots/_deps.py::require_plotly(...)`. Same `OPTIONAL_DEP_MISSING` behavior. Also corrected the README's generated-code dependency note to include `scipy` (required for beta calibration).
- **Facade slimming — extracted accumulated logic out of `core/model.py`** ([#209](https://github.com/nbx-liz/LizyML/issues/209)). Round-summary assembly and per-trial round renumbering moved to `lizyml/tuning/rounds.py` (`assemble_round_result`); split-driven data ordering/extraction moved to `lizyml/data/dataframe_builder.py` (`prepare_for_split` / `sort_components`). `core/model.py` dropped from 1355 to 1244 lines. Pure refactor — no behavior change, no public-API/`format_version` change. Relocating the ~471-line `tune()` orchestration into a mixin is tracked as a follow-up (it needs an invariant decision vs the H-0077 read-only-mixin rule).

### Internal

- **Generated `test_equivalence.py` now imports `predict.py`** ([#217](https://github.com/nbx-liz/LizyML/issues/217)) — the codegen equivalence checker previously inlined a second copy of the transform / calibration logic that had already diverged from `predict.py` (e.g. missing column-drift validation), so it could pass while validating a different code path than users run. It now calls `predict.predict()` directly, and a template guard test asserts no second prediction implementation is emitted.

## [0.16.1] - 2026-06-30

### Fixed

- **Generated `train.py` / `predict.py` / `test_equivalence.py` open JSON artifacts as UTF-8** ([#192](https://github.com/nbx-liz/LizyML/issues/192)) — every `open()` in the exported scripts now pins `encoding="utf-8"`. A Windows (`cp1252`) or `C`-locale end-user running the generated scripts over data with a non-ASCII categorical value previously hit a `UnicodeEncodeError` / `UnicodeDecodeError` when writing/reading `pipeline_state.json`. (Completes [#180](https://github.com/nbx-liz/LizyML/issues/180), which fixed only the LizyML-side codegen writes.)

## [0.16.0] - 2026-06-01

Quality-audit remediation release: resolves the v0.15.0 comprehensive audit
(issues [#167](https://github.com/nbx-liz/LizyML/issues/167)–[#180](https://github.com/nbx-liz/LizyML/issues/180), all except the v1.0 deprecation-removal tracker #148).

### Changed (potentially breaking)

- **`training.seed` now propagates to the outer splitter and the isotonic calibrator** (H-0080, [#169](https://github.com/nbx-liz/LizyML/issues/169)). `split.random_state` defaults to a sentinel `None` meaning "inherit `training.seed`", and the loader no longer hard-codes `42`. For configs that set `training.seed` to a non-`42` value **without** an explicit `split.random_state`, CV fold composition — and therefore OOF predictions, metrics, and saved split indices — now reflects `training.seed` (previously folds were silently fixed at `42`). Explicit `split.random_state` is still honored and unchanged; all-default configs are unaffected (the effective seed stays `42`).

### Added

- **Artifact integrity binding (SHA-256)** (H-0083, [#179](https://github.com/nbx-liz/LizyML/issues/179)). `export()` records the SHA-256 of each `.pkl` (`fit_result` / `refit_model` / `analysis_context`) in `metadata.json` under a `checksums` field; `Model.load()` verifies the digest before `joblib.load` and raises `DESERIALIZATION_FAILED` on a mismatch, detecting tampering or corruption. Additive and back-compatible — artifacts without the field still load and `format_version` stays `2`. (Does not make pickle safe against a trusted-but-malicious producer; the trusted-source contract is unchanged.)

### Fixed

- **multiclass OvR metrics (AUC / AUCPR / Brier) no longer fail on a class-missing CV fold** ([#167](https://github.com/nbx-liz/LizyML/issues/167)) — they now macro-average only over the classes present in `y_true` instead of raising an unwrapped `ValueError` / silently degrading.
- **cross-fit calibration tolerates a single-class training fold** ([#168](https://github.com/nbx-liz/LizyML/issues/168)) — it falls back gracefully instead of raising an unwrapped `ValueError`.
- **`evaluate(None)` and the `fit_result` property return independent copies** (H-0082, [#174](https://github.com/nbx-liz/LizyML/issues/174)) — they no longer hand out live internal references, so mutating a returned metrics dict / `FitResult` can no longer corrupt internal state or contaminate a later `export()`. Trained estimators reachable via `fit_result` (`models` / `calibrator` / `pipeline_state`) are shared by reference (read-only by convention) to preserve LightGBM Booster fidelity.
- **`export_code()` writes generated files as UTF-8** ([#180](https://github.com/nbx-liz/LizyML/issues/180)) — fixes a `UnicodeEncodeError` when exporting on Windows, where the default `cp1252` codec could not encode non-ASCII template characters.

### Internal

- **Facade `predict()` extracted** ([#172](https://github.com/nbx-liz/LizyML/issues/172)) — estimator/calibration/SHAP branching moved to `core/_model_predict.py`; `Model.predict()` is now assembly-only.
- **`FitState` / `TuningState` moved out of Layer-0** (H-0084, [#171](https://github.com/nbx-liz/LizyML/issues/171)) — relocated from `core/types/` to facade-adjacent `core/_model_state.py`, restoring the Layer-0 "dependency-free" invariant (the sole DAG back-edge removed). Internal types; no public-API change.
- **Forward-compat CI lane + OS portability matrix** ([#180](https://github.com/nbx-liz/LizyML/issues/180)) — a non-blocking latest-deps lane (`uv sync --upgrade`) surfaces upstream breakage early, and an ubuntu/windows/macos smoke matrix catches path/newline portability issues. The no-upper-bound dependency policy is documented in CONTRIBUTING and BLUEPRINT §18.2.
- **Branch coverage enabled** ([#173](https://github.com/nbx-liz/LizyML/issues/173)) — `branch = true`; `--cov-fail-under` re-baselined to a branch-inclusive `96` (measured ~97%).
- **Forward-looking test hardening** ([#178](https://github.com/nbx-liz/LizyML/issues/178)) — codegen real-subprocess equivalence, a feature-pipeline leakage trap, reproducibility bit-equality + seed-sensitivity, README-block execution, inner-valid ratio guards across all strategies, and notebook coverage. Executing every tutorial in CI surfaced two previously-unreferenced, broken notebooks — the `calibration` and `SHAP` tutorials now run end-to-end (fixed outdated config/result-access against the current API); notebook execution also skips gracefully on a remote-dataset network outage instead of failing the release gate.
- **Documentation reconciliation** ([#170](https://github.com/nbx-liz/LizyML/issues/170), [#175](https://github.com/nbx-liz/LizyML/issues/175), [#176](https://github.com/nbx-liz/LizyML/issues/176), [#177](https://github.com/nbx-liz/LizyML/issues/177)) — corrected the FAQ custom-objective example, `api.md` `tune()` storage/study_name, `migration.md`, several docstrings, documented `oof_coverage`, scoped the bit-identical reproducibility guarantee to a fixed `(num_threads, CPU)` environment, and corrected the erroneous `simulate` CHANGELOG entry.

## [0.15.0] - 2026-05-10

### Changed (potentially breaking)

- **`LGBMConfig.params["objective"]` is now respected when task-compatible** (H-0079 Phase 1, [#159](https://github.com/nbx-liz/LizyML/issues/159)) — pre-0.15 `LGBMAdapter._build_params()` silently stripped any user/Optuna-supplied `objective` and force-set `_TASK_OBJECTIVE[task]`, so `default_space("regression")` trials sampling `"fair"` actually trained with `"huber"`. From this release: same-task values flow through to `lgb.train` (e.g. `objective="fair"` for regression now actually uses Fair loss); cross-task values raise `LizyMLError(CONFIG_INVALID)` instead of being silently demoted. Users who relied on the silent strip to suppress accidental cross-task injection see no behaviour change at the contract level (still rejected), but **same-task non-default objectives may produce different metrics than pre-0.15 runs**. Re-running tune over `default_space` may yield a different `best_params` because `"fair"` is now genuinely evaluated.
- **`LGBMAdapter._build_params()` enforces an objective invariant assertion** (H-0079 L5) — at the end of `_build_params()`, an `assert` validates that any user-supplied `objective` survives the build. Active in dev / test / CI; suppressed under `python -O`. Fail-fast guard against future regressions to the silent-strip pattern.

### Added

- **`TASK_COMPATIBLE_OBJECTIVES` whitelist in `lizyml.estimators.lgbm.defaults`** (H-0079 Phase 1) — public mapping `dict[str, frozenset[str]]` enumerating the canonical LightGBM objective names valid per task (regression: 9, binary: 3, multiclass: 2). Used by `_build_params()` for cross-task validation and exposed for downstream integrations until the Provider-level API ships in Phase 2.
- **`EstimatorProvider.objective_choices(task) -> tuple[str, ...]`** (H-0079 Phase 2, [#159](https://github.com/nbx-liz/LizyML/issues/159)) — new Protocol method returning canonical objective names valid for *task* in deterministic order. `LGBMProvider.objective_choices(task)` ships ordered tuples (regression: 9, binary: 3, multiclass: 2) sourced from the same whitelist as `TASK_COMPATIBLE_OBJECTIVES`. Drift between the two surfaces is caught by a load-time invariant. Used by `default_space()` and downstream UIs (LizyStudio) so the canonical list lives in one place.
- **`EstimatorProvider.metric_choices(task) -> dict[Literal["native", "feval"], tuple[str, ...]]`** (H-0079 Phase 2, [#159](https://github.com/nbx-liz/LizyML/issues/159)) — new Protocol method returning per-task valid metrics split by source. `"native"` lists LightGBM-evaluated metrics composed straight into `params["metric"]`; `"feval"` lists LizyML custom metrics wired as feval callables. Canonical names only (aliases like `l1` / `l2` / `mse` are still accepted at config-input time but not surfaced).
- **`MetricChoices` type alias in `lizyml.estimators.provider`** (H-0079 Phase 2) — `dict[Literal["native", "feval"], tuple[str, ...]]`. Forward-compatible: future estimators may add new keys (e.g. `"sklearn"` for scikit-learn-backed metrics) without breaking existing consumers.
- **`default_space(task, provider=None)` accepts an optional `EstimatorProvider`** (H-0079 Phase 2) — when supplied, the `objective` `CategoricalDim` is built from `provider.objective_choices(task)` so default-space and user-supplied provider stay aligned. Existing call sites unchanged (provider defaults to `None` → conservative tune-safe subset).

### Fixed

- **`metric_bridge._LGBM_NATIVE_METRICS["multiclass"]` incorrectly listed `auc`** (H-0079 Phase 3, surfaced by L4 drift test) — LightGBM 4.x raises `LightGBMError("Multiclass objective and metrics don't match")` when `auc` reaches multiclass `params["metric"]`. The whitelist accepted the name pre-validation, so users got a cryptic LightGBM-side error instead of a clear LizyML rejection. The whitelist now omits `auc` for multiclass; users requesting AUC on multiclass should use `Model.evaluate(metrics=["auc"])` (sklearn OvR, computed Python-side post-fit) or `auc_mu` for fit-time evaluation.

### Internal

- **`_OBJECTIVE_CHOICES` retired** (H-0079 Phase 3) — replaced by a conservative `_DEFAULT_TUNE_OBJECTIVES` table in `lizyml/estimators/lgbm/defaults.py` whose values intentionally exclude `gamma` / `poisson` / `tweedie` / `mape` (target-distribution-restricted). The full canonical set is exposed via `LGBMProvider().objective_choices(task)` for downstream UIs and explicit user-supplied search spaces.
- **`_LGBM_OBJECTIVE_CHOICES` self-validates against `TASK_COMPATIBLE_OBJECTIVES`** at module load time (H-0079 Phase 2/3) so the two sources of truth cannot drift.
- **L4 MetricRegistry coverage drift test** (`tests/test_estimators/test_metric_choices_registry_coverage.py`, H-0079 Phase 3) — every fit-time-reachable metric in `MetricRegistry._TASK_METRICS` is now asserted to appear in `LGBMProvider.metric_choices()` after alias translation. Caught the multiclass `auc` omission above.
- **`_validate_metric_consistency()` load-time guard** (H-0079 follow-up, [#164](https://github.com/nbx-liz/LizyML/pull/164)) — parallel to `_validate_objective_consistency()`. Asserts at module load that every name surfaced via `_LGBM_NATIVE_METRIC_CHOICES` / `_LGBM_FEVAL_METRIC_CHOICES` is reachable via `metric_bridge` whitelists. Drift would offer a metric to a downstream UI that the library would later reject; fail-fast prevents that.
- **H-0079 follow-up coverage tests** (`tests/test_estimators/test_h0079_followup.py`, [#164](https://github.com/nbx-liz/LizyML/pull/164)) — 11 tests pinning previously-untested integration boundaries: codegen export with non-default objective (3), save/load round-trip with non-default objective (2), `_check_objective_compatible` edge inputs (4), and Platt calibration on top of binary `cross_entropy` objective (2).
- **`LGBMProvider.build_export_params` docstring** (H-0079 follow-up) gains a `Note:` block documenting the intentional same-package private call into `LGBMAdapter._build_params()` so future refactors update both call sites together.

## [0.14.0] - 2026-05-10

### Added

- **`EstimatorProvider.parameter_bounds(task)`** (H-0078, [#152](https://github.com/nbx-liz/LizyML/issues/152)) — new Protocol method returning per-parameter meaningful bounds (`{"min": ..., "max": ...}`) for boundary expansion. `LGBMProvider.parameter_bounds(task)` ships a static map for 15 LightGBM parameters (e.g. `learning_rate ∈ [1e-8, 1.0]`, `feature_fraction ∈ [1e-3, 1.0]`, `validation_ratio ∈ [0.05, 0.5]`, `max_depth ∈ [-1, 30]`). Used by `Model.tune` and downstream UIs (LizyStudio) to constrain user input. Third-party providers may return `{}` for unbounded behaviour.
- **`SearchDim.min_allowed` / `max_allowed`** (H-0078) — optional bounds on `FloatDim` / `IntDim` (default `None`). Boundary expansion clamps to these limits when set.
- **`BoundaryDimStatus.clamped_to_bound: bool`** (H-0078) — flags dims whose expansion hit the parameter-meaningful bound, so downstream UIs can badge "max reached" dims. Defaults `False`.
- **`attach_bounds(dims, bounds)` helper in `lizyml.tuning.search_space`** (H-0078) — injects `min_allowed` / `max_allowed` onto matching dims by name. Called from `Model._resolve_search_space` so default-space and user-supplied dims both pick up provider bounds automatically.

### Changed

- **`parse_space()` rejects degenerate / inverted ranges and log-with-non-positive-low** (H-0078, [#152](https://github.com/nbx-liz/LizyML/issues/152)) — `low >= high` and `log=True ∧ low <= 0` now raise `LizyMLError(CONFIG_INVALID)` at parse time instead of letting Optuna raise a generic error mid-trial. Strictly better failure mode (earlier and clearer).
- **`expand_dims` propagates `min_allowed` / `max_allowed`** (H-0078) — re-tune over multiple rounds preserves provider-supplied bounds, preventing the original `learning_rate` drift (`0.1 → 0.3 → 0.9 → 2.7`) reported in #152. 5- and 10-round regression tests guard the contract.

### Internal

- **`_expand_range` is bounds-aware** (H-0078) — keyword-only `min_allowed` / `max_allowed` arguments + 3-tuple return `(low, high, clamped)`. Internal signature change; only `detect_boundary` is a caller.

## [0.13.0] - 2026-05-10

### Added

- **`EstimatorProvider.build_export_params()`** (H-0073, [#109](https://github.com/nbx-liz/LizyML/issues/109), [#126](https://github.com/nbx-liz/LizyML/issues/126)) — codegen-relevant booster params and feval metadata are now retrieved through the `EstimatorProvider` Protocol, so `Model.export_code()` is fully estimator-agnostic. Adding a new estimator no longer requires editing `lizyml/core/_model_persistence.py`. The new method also unifies `BlockedGroupKFold` `n_splits` resolution between persistence and factories.
- **`FitState` / `TuningState` frozen dataclasses + `Model._get_fit_state()` / `_get_tuning_state()`** (H-0074 Phase 1 + H-0077 Phase 2, [#112](https://github.com/nbx-liz/LizyML/issues/112)) — `ModelPlotsMixin` / `ModelTablesMixin` / `ModelPersistenceMixin` now read state exclusively through these snapshots. Direct `self._<private>` access is forbidden inside Mixin bodies and enforced by a static guard test (`tests/test_core/test_mixin_state_isolation.py`). Mixins become unit-testable with synthetic state. Public API unchanged.
- **`docs/DEPRECATIONS.md` central deprecation registry** (H-0076, [#120](https://github.com/nbx-liz/LizyML/issues/120), [#121](https://github.com/nbx-liz/LizyML/issues/121)) — single source of truth for every deprecated public surface and its removal target version. Every `DeprecationWarning` LizyML raises now contains "Will be removed in vX.Y." (currently "v1.0"); `tests/test_core/test_deprecation_registry.py` enforces this contract in CI.

### Changed

- **`TaskType` Literal centralised + propagated to all dispatch sites** (H-0075, [#122](https://github.com/nbx-liz/LizyML/issues/122)) — `lizyml/core/types/task.py` exposes `TaskType = Literal["regression", "binary", "multiclass"]`. Every branch on task now uses an exhaustive dispatch table, eliminating string-comparison divergence between modules. Public API unchanged; internal type-safety only.
- **`DeprecationWarning` messages now state the removal target version** (H-0076) — users see "Will be removed in v1.0." in every deprecation message so migration deadlines are explicit.
- **Inner-valid membership checks vectorised with numpy** (H-0065 follow-up, [#135](https://github.com/nbx-liz/LizyML/issues/135)) — measurable speedup on large CV folds; behaviour unchanged.
- **`Model.tune()` decomposed into 5 testable helpers** (H-0040 follow-up, [#114](https://github.com/nbx-liz/LizyML/issues/114)) — orchestrator + per-step helpers, no behaviour change.
- **`LizyMLError.context` enriched at 23 sites** ([#118](https://github.com/nbx-liz/LizyML/issues/118)) — fold index / config path / method name now consistently carried; regression guard added.
- **`storage` parameter type tightened to `str | BaseStorage | None`** ([#136](https://github.com/nbx-liz/LizyML/issues/136)) — was `Any`; aligns with H-0072 docstring.
- **Plot theme deduplicated via `apply_default_layout`** ([#134](https://github.com/nbx-liz/LizyML/issues/134)) — `lizyml/plots/_theme.py` is now the single source for default layout settings shared across every plot module.
- **`StratifiedTimeHoldoutInnerValid` tail-holdout fallback inlined** ([#133](https://github.com/nbx-liz/LizyML/issues/133)) — readability improvement, behaviour unchanged.
- **`TargetEncoder` lexicographic class ordering documented with example** ([#132](https://github.com/nbx-liz/LizyML/issues/132)).

### Fixed

- **Tuning progress callback warning now includes exception type and message** ([#128](https://github.com/nbx-liz/LizyML/issues/128)) — debugging callback failures no longer requires re-running with logging tweaks.
- **`FloatDim` linear lower expansion clamped at zero** ([#129](https://github.com/nbx-liz/LizyML/issues/129)) — boundary-detection re-tune (H-0068) no longer drives lower bounds below zero on naturally non-negative parameters.
- **Legacy top-level `validation_ratio` input emits `DeprecationWarning`** ([#130](https://github.com/nbx-liz/LizyML/issues/130)) — previously a YAML with only `early_stopping.validation_ratio` (and no `inner_valid:` block) was silent. Now the deprecation contract is enforced on input, not just output.
- **Unhandled `SplitConfig` variants raise `LizyMLError(CONFIG_INVALID)`** ([#131](https://github.com/nbx-liz/LizyML/issues/131)) — previously a silent fallback could mask schema regressions.
- **Code-review HIGH issues + `#124` test gap closed** ([#141](https://github.com/nbx-liz/LizyML/issues/141)) — Sprint 1+2 follow-up batch.
- **MEDIUM / LOW code-review batch** ([#142](https://github.com/nbx-liz/LizyML/issues/142)) — accumulated cleanup landed in one PR.

### Internal

- **`TrainComponents` rebuild path extracted to `_sort_and_rebuild_components()`** ([#137](https://github.com/nbx-liz/LizyML/issues/137)) — refactor only.
- **Mixin source files contain zero direct `self._<private>` access** (H-0077 Phase 2, enforced by `tests/test_core/test_mixin_state_isolation.py`).

## [0.12.0] - 2026-05-06

### Added

- **Resumable tuning via Optuna persistent storage** (H-0072, [#105](https://github.com/nbx-liz/LizyML/issues/105)) — `Tuner` and `Model.tune()` now accept `storage` (Optuna URL such as `sqlite:///path/to.db` or a `BaseStorage` instance) and `study_name`. When set, trial state is persisted to disk after each trial completes; re-invoking `Model.tune(storage=..., study_name=...)` with the same identifiers re-attaches via `load_if_exists=True` so completed trials are not re-run. `storage=None` (default) preserves the in-memory behavior with no disk I/O. Designed for long-running tune jobs that must survive process kill, server restart, or network outage. No new dependencies (uses Optuna's built-in storage backends).

## [0.11.0] - 2026-05-05

### Added

- **sMAPE / WAPE — zero-tolerant percentage-style regression metrics** (H-0071, [#101](https://github.com/nbx-liz/LizyML/issues/101)) — `lizyml.metrics.SMAPE` and `lizyml.metrics.WAPE` are now available for `task=regression` and registered in `MetricRegistry` under `"smape"` / `"wape"`. Both close the gap left by MAPE on datasets where `y_true` may be `0` (sales / demand / count regressions). Wired into the LightGBM metric bridge so `params={"metric": ["smape", "wape"]}` produces feval-driven entries in `eval_history` / learning curves, and into the codegen exporter so `Model.export_code()` reproduces the same values offline. Authoritative formulas and edge-case conventions are documented in [`docs/config-reference.md` § Metric formula reference](docs/config-reference.md#metric-formula-reference).
- **Metric formula reference** — `docs/config-reference.md` now documents authoritative formulas, ranges, and edge-case conventions for every regression and classification metric LizyML ships, plus a MAPE / sMAPE / WAPE selection guide.

## [0.10.0] - 2026-05-04

### Added

- **Auto-encode non-numeric classification targets** (H-0070, [#98](https://github.com/nbx-liz/LizyML/issues/98)) — `task ∈ {binary, multiclass}` now accepts non-numeric `y` (object / `pd.StringDtype` / category / bool). LizyML applies a `TargetEncoder` automatically and `Model.predict()` returns predictions in the **original label dtype** (e.g. `"Adelie"` instead of `2`). The new `FitResult.target_encoder` carries `classes_` so consumers (incl. `export_code()`-generated `train.py` / `predict.py`) can map int codes back to the original labels. Calibration / tuning paths work transparently.
- **New error codes**: `TARGET_NOT_NUMERIC`, `TARGET_UNSEEN_LABEL`.

### Changed

- **`task=regression` × non-numeric `y` now raises `TARGET_NOT_NUMERIC` before model training starts** (H-0070) — previously fit failed with an unclear error from the LightGBM layer.
- **Codegen `predict.py` output dtype**: when the original target was non-numeric, generated predictions now decode int codes back to the original labels via a `target_encoder.classes` array baked into `config.json`.
- **Persistence `FORMAT_VERSION` bumped to 2** (H-0070) — `Model.load()` accepts both `format_version=1` (old) and `2` (current). v1 artifacts are migrated in-memory by injecting a no-op `TargetEncoder`, so existing saved models continue to load without user action.

## [0.9.1] - 2026-05-02

### Fixed

- **`Model.load()` fails for non-holdout `inner_valid`** (H-0069, [#95](https://github.com/nbx-liz/LizyML/issues/95)) — Saving a model fit with `inner_valid.method ∈ {group_holdout, time_holdout}` produced an artifact that could not be re-loaded (`CONFIG_INVALID`). Root cause: `validation_ratio` and `inner_valid.ratio` were two mutable fields whose only synchronization was a one-way validator branch that ignored `group_holdout` / `time_holdout`. `model_dump()` always emitted both keys, so the round-trip silently broke.

### Changed

- **`EarlyStoppingConfig.validation_ratio` is now a read-only computed field** (H-0069) — `validation_ratio` mirrors `inner_valid.ratio` automatically, eliminating the dual-write inconsistency at its source. Existing YAML inputs (`validation_ratio: 0.1` only, or `inner_valid: {...}` only) are fully backward compatible. Existing `model.lizyml` artifacts load without migration. Side effect: codegen `export_code()` now uses the correct holdout fraction when `inner_valid.ratio` differs from the default 0.1 (previously a silent ratio mismatch).

## [0.9.0] - 2026-04-12

### Added

- **Re-tune: Study Resume + Boundary Expansion** (H-0068)
  - `Model.tune(resume=True)` resumes from the previous Optuna Study with additional trials; TPE sampler reuses knowledge from prior trials and previous best params are enqueued as a warm-start trial
  - Automatic boundary detection identifies dimensions where best params are near the search space edge
  - Asymmetric space expansion extends promising directions only (linear: 2× range, log: 3× in log space)
  - `TuningResult.rounds` tracks per-round history (`RoundSummary` with scores, expanded dims, space snapshots)
  - `TuningResult.boundary_report` provides dimension-by-dimension boundary analysis (`BoundaryReport` / `BoundaryDimStatus`)
  - `Model.boundary_table()` returns boundary detection results as a DataFrame
  - `TuneProgressInfo` gains `round`, `cumulative_trials`, `expanded_dims` for real-time progress monitoring
  - `TrialResult.round` indicates which re-tune round each trial belongs to
  - `tuning_table()` includes `round` and `state` columns
  - `plot_tuning_history()` shows round boundary separators with expanded dimension annotations
  - New public types exported from `lizyml`: `BoundaryReport`, `BoundaryDimStatus`, `RoundSummary`
  - Fully backward compatible: `tune()` with no new parameters behaves identically to previous versions

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
- ~~Simulate (bootstrap prediction distributions)~~ — listed in error; this
  feature was never shipped (no `simulate` API has existed in any release).
  The CHANGELOG/code reconciliation was completed in
  [#177](https://github.com/nbx-liz/LizyML/issues/177); implementing the feature
  is tracked separately in [#194](https://github.com/nbx-liz/LizyML/issues/194).
- YAML/JSON config loading with pydantic validation
