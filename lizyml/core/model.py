"""Model — the public-facing facade for LizyML.

Model's responsibilities: assembly and delegation only.

What Model does:
1. Validate Config → derive Specs.
2. Load data via DataSource → DataFrameBuilder.
3. Select components via Registry.
4. Delegate to CVTrainer / Evaluator / RefitTrainer.
5. Store FitResult and RefitResult; expose them via evaluate / predict.

What Model does NOT contain:
- OOF/IF generation logic  → evaluation/oof.py
- Metric computation        → evaluation/evaluator.py
- LGBM-specific processing  → estimators/lgbm.py
- Plot implementations      → plots/*
- Persistence details       → persistence/*

Mixin decomposition (H-0042):
- Plot methods      → _model_plots.py (ModelPlotsMixin)
- Table/accessors   → _model_tables.py (ModelTablesMixin)
- Export/load       → _model_persistence.py (ModelPersistenceMixin)
- Splitter/IV build → _model_factories.py (module-level functions)
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml import __version__
from lizyml.config.loader import load_config
from lizyml.config.schema import (
    LGBMConfig,
    LizyMLConfig,
)
from lizyml.core._model_factories import (
    build_inner_valid,
    build_splitter,
    make_inner_valid_factory,
)
from lizyml.core._model_persistence import ModelPersistenceMixin
from lizyml.core._model_plots import ModelPlotsMixin
from lizyml.core._model_tables import ModelTablesMixin
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import generate_run_id, get_logger
from lizyml.core.specs.feature_spec import FeatureSpec
from lizyml.core.specs.problem_spec import ProblemSpec
from lizyml.core.types.artifacts import RunMeta
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.predict_result import PredictionResult
from lizyml.core.types.tuning_result import TuningResult
from lizyml.data import dataframe_builder, datasource
from lizyml.data.dataframe_builder import DataFrameComponents
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.estimators.lgbm import (
    _COMMON_DEFAULTS,
    LGBMAdapter,
    resolve_ratio_params,
    resolve_smart_params,
)
from lizyml.evaluation.evaluator import Evaluator
from lizyml.features.pipelines_native import NativeFeaturePipeline
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.refit_trainer import RefitResult, RefitTrainer
from lizyml.tuning.search_space import default_fixed_params, default_space, parse_space
from lizyml.tuning.tuner import Tuner

_log = get_logger("model")

TaskType = Literal["regression", "binary", "multiclass"]

# Default metrics per task when none are specified in config.
_DEFAULT_METRICS: dict[str, list[str]] = {
    "regression": ["rmse", "mae"],
    "binary": ["logloss", "auc"],
    "multiclass": ["logloss", "f1", "accuracy"],
}

_TS_METHODS = frozenset({"time_series", "purged_time_series", "group_time_series"})


class Model(ModelPlotsMixin, ModelTablesMixin, ModelPersistenceMixin):
    """Public-facing facade for LizyML.

    Args:
        config: Config source — a dict, a YAML/JSON file path, or a
            :class:`~lizyml.config.schema.LizyMLConfig` instance.
        data: Optional training DataFrame.  When provided, overrides any
            ``data.path`` from the config (useful for in-memory workflows).

    Example::

        model = Model({"config_version": 1, "task": "regression", ...})
        result = model.fit(data=df)
        metrics = model.evaluate()
        predictions = model.predict(X_new)
    """

    def __init__(
        self,
        config: dict[str, Any] | str | Path | LizyMLConfig,
        *,
        data: pd.DataFrame | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        if isinstance(config, LizyMLConfig):
            self._cfg = config
        else:
            self._cfg = load_config(config)

        self._data: pd.DataFrame | None = data
        # Constructor arg takes priority over Config (BLUEPRINT §17)
        resolved_dir = output_dir or getattr(self._cfg, "output_dir", None)
        self._output_dir: str | Path | None = resolved_dir
        self._run_dir: Path | None = None
        self._fit_result: FitResult | None = None
        self._refit_result: RefitResult | None = None
        self._metrics: dict[str, Any] | None = None
        self._best_params: dict[str, Any] | None = None
        self._tuning_result: TuningResult | None = None
        self._y: pd.Series | None = None  # transient; not persisted
        self._X: pd.DataFrame | None = None  # transient; not persisted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame | None = None,
        params: dict[str, Any] | None = None,
    ) -> FitResult:
        """Run CV training and (optionally) full-data refit.

        Args:
            data: Training DataFrame.  Overrides any ``data`` passed at
                construction time and the ``data.path`` from config.
            params: LightGBM parameters to override the config ``model.params``.

        Returns:
            The :class:`~lizyml.core.types.fit_result.FitResult` from CV.
        """
        cfg = self._cfg
        run_id = generate_run_id()
        run_meta = self._build_run_meta(run_id)

        # --- Output directory setup (H-0034) ---------------------------------
        if self._output_dir is not None:
            from lizyml.core.logging import setup_output_dir

            self._run_dir = setup_output_dir(self._output_dir, run_id)

        _log.info("event='fit.start' run_id=%s task=%s", run_id, cfg.task)

        # --- Load & prepare data ---------------------------------------------
        X, y, groups, components = self._prepare_training_data(data)
        fingerprint = fp_compute(X, file_path=None)

        # --- Build components ------------------------------------------------
        splitter = build_splitter(cfg)
        inner_valid = build_inner_valid(cfg)
        lgbm_params = {
            **_get_lgbm_params(cfg),
            **(self._best_params or {}),
            **(params or {}),
        }
        n_classes = int(y.nunique()) if cfg.task == "multiclass" else None

        # Resolve smart parameters (H-0021)
        sample_weight: npt.NDArray[np.float64] | None = None
        model_cfg = cfg.model
        if isinstance(model_cfg, LGBMConfig):
            effective = {**_COMMON_DEFAULTS, **lgbm_params}
            smart_resolved, sample_weight = resolve_smart_params(
                config=model_cfg,
                effective_params=effective,
                n_rows=len(X),
                feature_names=list(X.columns),
                y=y,
                task=cfg.task,
            )
            lgbm_params.update(smart_resolved)

        # Build per-fold ratio param resolver (H-0036)
        ratio_resolver: Callable[[int], dict[str, Any]] | None = None
        if isinstance(model_cfg, LGBMConfig) and (
            model_cfg.min_data_in_leaf_ratio is not None
            or model_cfg.min_data_in_bin_ratio is not None
        ):
            _leaf_ratio = model_cfg.min_data_in_leaf_ratio
            _bin_ratio = model_cfg.min_data_in_bin_ratio

            def ratio_resolver(n: int) -> dict[str, int]:
                return resolve_ratio_params(
                    min_data_in_leaf_ratio=_leaf_ratio,
                    min_data_in_bin_ratio=_bin_ratio,
                    n_rows=n,
                )

        def make_pipeline() -> NativeFeaturePipeline:
            return NativeFeaturePipeline()

        def make_estimator() -> LGBMAdapter:
            return LGBMAdapter(
                task=cfg.task,
                params=lgbm_params,
                num_class=n_classes,
                early_stopping_rounds=(
                    cfg.training.early_stopping.rounds
                    if cfg.training.early_stopping.enabled
                    else None
                ),
                random_state=cfg.training.seed,
            )

        # --- CV training -----------------------------------------------------
        cv_trainer = CVTrainer(
            outer_splitter=splitter,
            inner_valid=inner_valid,
            pipeline_factory=make_pipeline,
            estimator_factory=make_estimator,
            task=cfg.task,
            n_classes=n_classes,
            ratio_param_resolver=ratio_resolver,
            collect_raw_scores=(cfg.calibration is not None),
        )
        time_values = components.time_col if components.time_col is not None else None
        fit_result = cv_trainer.fit(
            X,
            y,
            groups,
            data_fingerprint=fingerprint,
            run_meta=run_meta,
            sample_weight=sample_weight,
            time_values=time_values,
        )

        # --- Calibration (binary only) ---------------------------------------
        if cfg.calibration is not None:
            if cfg.task != "binary":
                raise LizyMLError(
                    code=ErrorCode.CALIBRATION_NOT_SUPPORTED,
                    user_message=(
                        f"Calibration is only supported for binary classification. "
                        f"Got task='{cfg.task}'."
                    ),
                    context={"task": cfg.task},
                )
            from lizyml.calibration.cross_fit import cross_fit_calibrate
            from lizyml.calibration.registry import get_calibrator

            method = cfg.calibration.method
            cal_params = cfg.calibration.params or None
            # Use raw scores (logits) for calibration (H-0030)
            cal_scores = (
                fit_result.oof_raw_scores
                if fit_result.oof_raw_scores is not None
                else fit_result.oof_pred
            )
            calibration_result = cross_fit_calibrate(
                oof_scores=cal_scores,
                y=y.to_numpy(),
                calibrator_factory=lambda: get_calibrator(method, params=cal_params),
                n_splits=cfg.calibration.n_splits,
                random_state=cfg.training.seed,
            )
            fit_result.calibrator = calibration_result
            # Store calibration split indices for reproducibility / audit
            fit_result.splits.calibration = calibration_result.split_indices

        # --- Evaluation -------------------------------------------------------
        metric_names = cfg.evaluation.metrics or _DEFAULT_METRICS[cfg.task]
        evaluator = Evaluator(task=cfg.task)
        metrics = evaluator.evaluate(fit_result, y, metric_names)
        fit_result.metrics.update(metrics)
        self._metrics = metrics

        # --- Full-data refit (for predict) -----------------------------------
        refit_trainer = RefitTrainer(
            inner_valid=inner_valid,
            pipeline_factory=make_pipeline,
            estimator_factory=make_estimator,
            task=cfg.task,
            ratio_param_resolver=ratio_resolver,
        )
        self._refit_result = refit_trainer.fit(X, y, groups)

        self._fit_result = fit_result
        _log.info("event='fit.done' run_id=%s", run_id)
        return fit_result

    def evaluate(
        self,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return structured evaluation metrics from the last ``fit``.

        Args:
            metrics: Metric names to compute.  When ``None`` uses defaults
                or config-defined metrics (already computed during ``fit``).

        Returns:
            Structured dict: ``{"raw": {"oof": ..., "if_mean": ...,
            "if_per_fold": ...}}``.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit``.
        """
        self._require_fit()

        if metrics is None:
            assert self._metrics is not None  # noqa: S101 — set by fit()
            return self._metrics

        # Validate task compatibility first (raises UNSUPPORTED_METRIC if invalid)
        from lizyml.metrics.registry import get_metrics_for_task

        get_metrics_for_task(metrics, self._cfg.task)  # raises on unknown/incompatible

        # Filter the pre-computed metrics dict to the requested subset
        assert self._metrics is not None  # noqa: S101
        return _filter_metrics(self._metrics, set(metrics))

    def predict(
        self,
        X: pd.DataFrame,
        *,
        return_shap: bool = False,
    ) -> PredictionResult:
        """Generate predictions for new data.

        Uses the final model trained on the full dataset (via RefitTrainer).

        Args:
            X: Feature DataFrame with the same columns as training data.
            return_shap: When ``True``, compute SHAP values and populate
                ``PredictionResult.shap_values`` with shape
                ``(n_samples, n_features)``.  Requires ``shap`` to be
                installed (``pip install 'lizyml[explain]'``).

        Returns:
            :class:`~lizyml.core.types.predict_result.PredictionResult`.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit``.
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``OPTIONAL_DEP_MISSING`` when ``return_shap=True`` and shap
            is not installed.
        """
        self._require_fit()
        refit = self._require_refit()

        # Restore pipeline from saved state
        pipeline = NativeFeaturePipeline()
        pipeline.load_state(refit.pipeline_state)

        X_t, warnings = pipeline.transform_with_warnings(X)

        model = refit.model
        task = self._cfg.task

        pred: npt.NDArray[np.float64]
        proba: npt.NDArray[np.float64] | None = None

        fit = self._fit_result  # non-None guaranteed by _require_fit()

        if task == "regression":
            pred = model.predict(X_t)
        elif task == "binary":
            proba_2d = model.predict_proba(X_t)
            proba = proba_2d[:, 1]
            # Apply C_final calibrator when available (H-0030: raw score input)
            if fit is not None and fit.calibrator is not None:
                from lizyml.calibration.cross_fit import CalibrationResult

                if isinstance(fit.calibrator, CalibrationResult):
                    if fit.oof_raw_scores is not None:
                        raw_scores: npt.NDArray[np.float64] = model.predict_raw(X_t)
                        proba = fit.calibrator.c_final.predict(raw_scores)
                    else:
                        # Backward compat: old artifact trained on probabilities
                        proba = fit.calibrator.c_final.predict(proba)
            pred = (proba >= 0.5).astype(int)
        else:  # multiclass
            proba = model.predict_proba(X_t)
            pred = proba.argmax(axis=1)

        shap_values: npt.NDArray[np.float64] | None = None
        if return_shap:
            from lizyml.explain.shap_explainer import compute_shap_values

            shap_values = compute_shap_values(refit.model, X_t, task)

        return PredictionResult(
            pred=pred,
            proba=proba,
            shap_values=shap_values,
            used_features=refit.feature_names,
            warnings=warnings,
        )

    def tune(
        self,
        data: pd.DataFrame | None = None,
    ) -> TuningResult:
        """Run hyperparameter search with optuna.

        Requires ``tuning`` section in the config.  Best params are stored
        internally and used automatically in the next ``fit()`` call.

        Args:
            data: Training DataFrame.  Overrides any data from construction
                or ``data.path`` in config.

        Returns:
            :class:`~lizyml.core.types.tuning_result.TuningResult` with
            best params, best score, and full trial history.

        Raises:
            LizyMLError with CONFIG_INVALID when no ``tuning`` config is set.
            LizyMLError with OPTIONAL_DEP_MISSING when optuna is not installed.
            LizyMLError with TUNING_FAILED on study failure.
        """
        cfg = self._cfg
        if cfg.tuning is None:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    "No tuning configuration found. "
                    "Add a 'tuning' section to the config to enable tuning."
                ),
                context={},
            )

        _log.info("event='tune.start' task=%s", cfg.task)

        # --- Output directory setup (H-0034) ---------------------------------
        if self._output_dir is not None:
            from lizyml.core.logging import setup_output_dir

            tune_run_id = generate_run_id()
            self._run_dir = setup_output_dir(self._output_dir, tune_run_id)

        X, y, groups, _ = self._prepare_training_data(data)

        n_classes = int(y.nunique()) if cfg.task == "multiclass" else None
        splitter = build_splitter(cfg)
        inner_valid = build_inner_valid(cfg)
        base_params = _get_lgbm_params(cfg)
        user_space = parse_space(cfg.tuning.optuna.space)
        if user_space:
            space = user_space
            fixed: dict[str, Any] = {}
        else:
            space = default_space(cfg.task)
            fixed = default_fixed_params(cfg.task)

        optuna_cfg = cfg.tuning.optuna.params
        metric_names = cfg.evaluation.metrics or _DEFAULT_METRICS[cfg.task]
        metric_name = metric_names[0]

        def make_trial_estimator(trial_params: dict[str, Any]) -> LGBMAdapter:
            merged = {**base_params, **trial_params}
            return LGBMAdapter(
                task=cfg.task,
                params=merged,
                num_class=n_classes,
                early_stopping_rounds=(
                    cfg.training.early_stopping.rounds
                    if cfg.training.early_stopping.enabled
                    else None
                ),
                random_state=cfg.training.seed,
            )

        tuner = Tuner(
            task=cfg.task,
            outer_splitter=splitter,
            inner_valid=inner_valid,
            pipeline_factory=NativeFeaturePipeline,
            estimator_factory=make_trial_estimator,
            dims=space,
            n_trials=optuna_cfg.n_trials,
            direction=optuna_cfg.direction,
            timeout=optuna_cfg.timeout,
            metric_name=metric_name,
            n_classes=n_classes,
            seed=cfg.training.seed,
            inner_valid_factory=make_inner_valid_factory(cfg),
            n_rows=len(X),
            fixed_params=fixed,
        )

        result = tuner.tune(X, y, groups)
        self._best_params = result.best_params
        self._tuning_result = result
        _log.info("event='tune.done' best_params=%s", result.best_params)
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fit_result(self) -> FitResult:
        """Read-only access to the CV training result.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when ``fit()`` has not been called.
        """
        return self._require_fit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_data(self, data: pd.DataFrame | None) -> pd.DataFrame:
        """Resolve data from argument, constructor, or config path."""
        if data is not None:
            return data
        if self._data is not None:
            return self._data
        if self._cfg.data.path:
            return datasource.read(self._cfg.data.path)
        raise LizyMLError(
            code=ErrorCode.DATA_SCHEMA_INVALID,
            user_message=(
                "No data provided. Pass a DataFrame to fit(data=df) or "
                "set data.path in the config."
            ),
            context={},
        )

    def _prepare_training_data(
        self, data: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.Series, npt.NDArray[Any] | None, DataFrameComponents]:
        """Load data, build specs, and prepare X/y/groups for training.

        Handles time-series sorting when the split method requires it.
        Sets ``self._X`` and ``self._y`` as a side effect.
        """
        cfg = self._cfg
        df = self._load_data(data)

        problem_spec = ProblemSpec(
            task=cfg.task,
            target=cfg.data.target,
            time_col=cfg.data.time_col,
            group_col=cfg.data.group_col,
            data_path=cfg.data.path,
        )
        feature_spec = FeatureSpec(
            exclude=tuple(cfg.features.exclude),
            auto_categorical=cfg.features.auto_categorical,
            categorical=tuple(cfg.features.categorical),
        )

        components = dataframe_builder.build(df, problem_spec, feature_spec)
        X, y = components.X, components.y
        groups: npt.NDArray[Any] | None = (
            components.group_col.to_numpy()
            if components.group_col is not None
            else None
        )

        if cfg.split.method in _TS_METHODS:
            if cfg.data.time_col is None:
                raise LizyMLError(
                    code=ErrorCode.CONFIG_INVALID,
                    user_message=(
                        f"split.method='{cfg.split.method}' requires "
                        "data.time_col to be set."
                    ),
                    context={"split_method": cfg.split.method},
                )
            tc = components.time_col
            assert tc is not None  # noqa: S101
            sort_order = tc.argsort()
            X = X.iloc[sort_order].reset_index(drop=True)
            y = y.iloc[sort_order].reset_index(drop=True)
            if groups is not None:
                groups = groups[sort_order]
            components.X = X
            components.y = y
            components.time_col = tc.iloc[sort_order].reset_index(drop=True)
            if components.group_col is not None:
                gc = components.group_col
                components.group_col = gc.iloc[sort_order].reset_index(drop=True)

        self._X = X
        self._y = y
        return X, y, groups, components

    def _build_run_meta(self, run_id: str) -> RunMeta:
        def _ver(pkg: str) -> str:
            try:
                return pkg_version(pkg)
            except Exception:
                return "unknown"

        return RunMeta(
            lizyml_version=__version__,
            python_version=sys.version,
            deps_versions={
                "lightgbm": _ver("lightgbm"),
                "pandas": _ver("pandas"),
                "numpy": _ver("numpy"),
                "scikit-learn": _ver("scikit-learn"),
            },
            config_normalized=self._cfg.model_dump(),
            config_version=self._cfg.config_version,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
        )

    def _require_fit(self) -> FitResult:
        if self._fit_result is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="Model has not been fitted. Call fit() first.",
                context={},
            )
        return self._fit_result

    def _require_refit(self) -> RefitResult:
        if self._refit_result is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="Model has not been fitted. Call fit() first.",
                context={},
            )
        return self._refit_result


def _get_lgbm_params(cfg: LizyMLConfig) -> dict[str, Any]:
    """Extract LightGBM params from model config."""
    model_cfg = cfg.model
    if isinstance(model_cfg, LGBMConfig):
        return dict(model_cfg.params)
    return {}


def _filter_metrics(metrics_dict: dict[str, Any], keep: set[str]) -> dict[str, Any]:
    """Return a copy of *metrics_dict* with only *keep* metric names retained.

    Works recursively on the nested
    ``{"raw": {"oof": {...}, ...}, "calibrated": {...}}``
    structure produced by :class:`~lizyml.evaluation.evaluator.Evaluator`.
    """
    result: dict[str, Any] = {}
    for top_key, top_val in metrics_dict.items():
        if not isinstance(top_val, dict):
            result[top_key] = top_val
            continue
        filtered_top: dict[str, Any] = {}
        for sub_key, sub_val in top_val.items():
            if sub_key == "if_per_fold":
                # List of per-fold dicts
                filtered_top[sub_key] = [
                    {m: v for m, v in fold.items() if m in keep} for fold in sub_val
                ]
            elif isinstance(sub_val, dict):
                filtered_top[sub_key] = {m: v for m, v in sub_val.items() if m in keep}
            else:
                filtered_top[sub_key] = sub_val
        result[top_key] = filtered_top
    return result
