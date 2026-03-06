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
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import generate_run_id, get_logger
from lizyml.core.specs.feature_spec import FeatureSpec
from lizyml.core.specs.problem_spec import ProblemSpec
from lizyml.core.types.artifacts import RunMeta
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.predict_result import PredictionResult
from lizyml.core.types.tuning_result import TuningResult
from lizyml.data import dataframe_builder, datasource
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.estimators.lgbm import (
    _COMMON_DEFAULTS,
    LGBMAdapter,
    resolve_ratio_params,
    resolve_smart_params,
)
from lizyml.evaluation.evaluator import Evaluator
from lizyml.features.pipelines_native import NativeFeaturePipeline
from lizyml.splitters.base import BaseSplitter
from lizyml.splitters.group_kfold import GroupKFoldSplitter
from lizyml.splitters.kfold import KFoldSplitter, StratifiedKFoldSplitter
from lizyml.splitters.time_series import TimeSeriesSplitter
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.inner_valid import (
    GroupHoldoutInnerValid,
    HoldoutInnerValid,
    NoInnerValid,
    TimeHoldoutInnerValid,
)
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


class Model:
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
    ) -> None:
        if isinstance(config, LizyMLConfig):
            self._cfg = config
        else:
            self._cfg = load_config(config)

        self._data: pd.DataFrame | None = data
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

        _log.info("event='fit.start' run_id=%s task=%s", run_id, cfg.task)

        # --- Load data -------------------------------------------------------
        df = self._load_data(data)

        # --- Build specs -----------------------------------------------------
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

        # --- Build X, y ------------------------------------------------------
        components = dataframe_builder.build(df, problem_spec, feature_spec)
        X, y = components.X, components.y
        self._X = X
        self._y = y
        groups = (
            components.group_col.to_numpy()
            if components.group_col is not None
            else None
        )

        fingerprint = fp_compute(X, file_path=None)

        # --- Build components ------------------------------------------------
        splitter = self._build_splitter()
        inner_valid = self._build_inner_valid()
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
        )
        fit_result = cv_trainer.fit(
            X,
            y,
            groups,
            data_fingerprint=fingerprint,
            run_meta=run_meta,
            sample_weight=sample_weight,
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
            calibration_result = cross_fit_calibrate(
                oof_scores=fit_result.oof_pred,
                y=y.to_numpy(),
                calibrator_factory=lambda: get_calibrator(method),
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

    def evaluate_table(self) -> pd.DataFrame:
        """Return evaluation metrics as a formatted DataFrame.

        Rows are metric names, columns are ``oof``, ``if_mean``,
        ``fold_0`` … ``fold_N-1``, and ``cal_oof`` when calibrated.

        Returns:
            :class:`pd.DataFrame` with metric values.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit``.
        """
        self._require_fit()
        from lizyml.evaluation.table_formatter import format_metrics_table

        assert self._metrics is not None  # noqa: S101 — set by fit()
        return format_metrics_table(self._metrics)

    def residuals(self) -> npt.NDArray[np.float64]:
        """Return OOF residuals ``(y_true - oof_pred)``.  Regression only.

        Returns:
            1-D array of shape ``(n_samples,)``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for non-regression tasks.
        """
        fit_result = self._require_fit()
        if self._cfg.task != "regression":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message=(
                    "residuals() is only supported for regression tasks. "
                    f"Got task='{self._cfg.task}'."
                ),
                context={"task": self._cfg.task},
            )
        if self._y is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    "Target values not available. "
                    "Re-export the model with the latest version to enable "
                    "diagnostic APIs after Model.load()."
                ),
                context={},
            )
        result: npt.NDArray[np.float64] = np.asarray(self._y) - fit_result.oof_pred
        return result

    def residuals_plot(self, *, kind: str = "all") -> Any:
        """Plot residual analysis.  Regression only.

        Args:
            kind: Which plot to render.
                ``"scatter"`` — residuals vs predicted (IS + OOS overlay).
                ``"histogram"`` — residual distribution (IS + OOS overlay).
                ``"qq"`` — QQ plot of OOS residuals.
                ``"all"`` — all three panels in one figure (default).

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for non-regression tasks.
            LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
            LizyMLError with ``CONFIG_INVALID`` for an unknown ``kind`` value.
        """
        # Validate state (raises MODEL_NOT_FIT / UNSUPPORTED_TASK as needed)
        self.residuals()
        fit_result = self._require_fit()
        from lizyml.plots.residuals import plot_residuals

        return plot_residuals(fit_result, np.asarray(self._y), kind=kind)

    def roc_curve_plot(self) -> Any:
        """Plot ROC Curve. Binary: IS/OOS overlay. Multiclass: OvR subplots.

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for regression.
            LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
        """
        fit_result = self._require_fit()
        if self._y is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    "Target values not available. "
                    "Re-export the model with the latest version "
                    "to enable diagnostic APIs after Model.load()."
                ),
                context={},
            )
        if self._cfg.task == "regression":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="roc_curve_plot() requires a binary or multiclass task.",
                context={"task": self._cfg.task},
            )
        from lizyml.plots.classification import plot_roc_curve

        return plot_roc_curve(fit_result, np.asarray(self._y), task=self._cfg.task)

    def confusion_matrix(self, threshold: float = 0.5) -> dict[str, pd.DataFrame]:
        """Return IS/OOS confusion matrices.

        Args:
            threshold: Binary decision boundary (binary only).

        Returns:
            ``{"is": DataFrame, "oos": DataFrame}``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for regression.
        """
        fit_result = self._require_fit()
        if self._y is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    "Target values not available. "
                    "Re-export the model with the latest version "
                    "to enable diagnostic APIs after Model.load()."
                ),
                context={},
            )
        if self._cfg.task == "regression":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="confusion_matrix() requires a binary or multiclass task.",
                context={"task": self._cfg.task},
            )
        from lizyml.evaluation.confusion import confusion_matrix_table

        return confusion_matrix_table(
            fit_result,
            np.asarray(self._y),
            threshold=threshold,
            task=self._cfg.task,
        )

    def calibration_plot(self) -> Any:
        """Plot calibration reliability diagram. Binary + calibration only.

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for non-binary tasks.
            LizyMLError with ``CALIBRATION_NOT_SUPPORTED`` if calibration
                is not enabled.
            LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
        """
        fit_result = self._require_fit()
        if self._y is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    "Target values not available. "
                    "Re-export the model with the latest version "
                    "to enable diagnostic APIs after Model.load()."
                ),
                context={},
            )
        if self._cfg.task != "binary":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="calibration_plot() requires a binary task.",
                context={"task": self._cfg.task},
            )
        from lizyml.plots.calibration import plot_calibration_curve

        return plot_calibration_curve(fit_result, np.asarray(self._y))

    def probability_histogram_plot(self) -> Any:
        """Plot raw vs calibrated probability histogram. Binary + calibration only.

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for non-binary tasks.
            LizyMLError with ``CALIBRATION_NOT_SUPPORTED`` if calibration
                is not enabled.
            LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
        """
        self._require_fit()
        if self._y is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    "Target values not available. "
                    "Re-export the model with the latest version "
                    "to enable diagnostic APIs after Model.load()."
                ),
                context={},
            )
        if self._cfg.task != "binary":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="probability_histogram_plot() requires a binary task.",
                context={"task": self._cfg.task},
            )
        fit_result = self._require_fit()
        from lizyml.plots.calibration import plot_probability_histogram

        return plot_probability_histogram(fit_result)

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
            # Apply C_final calibrator when available
            if fit is not None and fit.calibrator is not None:
                from lizyml.calibration.cross_fit import CalibrationResult

                if isinstance(fit.calibrator, CalibrationResult):
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

    def importance(self, kind: str = "split") -> dict[str, float]:
        """Return averaged feature importance across CV fold models.

        Args:
            kind: ``"split"``, ``"gain"``, or ``"shap"``.
                ``"shap"`` computes mean(|SHAP|) per feature across folds.
                Requires ``shap`` to be installed and training data to be
                available (or ``analysis_context`` to be restored after load).

        Returns:
            Dict mapping feature name → importance score.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit`` or (for ``"shap"``)
            when loaded artifacts lack ``analysis_context``.
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``OPTIONAL_DEP_MISSING`` when ``kind="shap"`` and shap is
            not installed.
        """
        fit_result = self._require_fit()

        if kind == "shap":
            if self._X is None:
                raise LizyMLError(
                    code=ErrorCode.MODEL_NOT_FIT,
                    user_message=(
                        "Training data not available. "
                        "Re-export the model with the latest version to enable "
                        "diagnostic APIs after Model.load()."
                    ),
                    context={},
                )
            from lizyml.explain.shap_explainer import compute_shap_importance

            return compute_shap_importance(
                models=fit_result.models,
                X=self._X,
                splits_outer=fit_result.splits.outer,
                task=self._cfg.task,
                feature_names=fit_result.feature_names,
                pipeline_state=fit_result.pipeline_state,
            )

        models = fit_result.models
        if not models:
            return {}

        agg: dict[str, float] = {}
        for m in models:
            for feat, val in m.importance(kind=kind).items():
                agg[feat] = agg.get(feat, 0.0) + val / len(models)
        return agg

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
        groups = (
            components.group_col.to_numpy()
            if components.group_col is not None
            else None
        )

        n_classes = int(y.nunique()) if cfg.task == "multiclass" else None
        splitter = self._build_splitter()
        inner_valid = self._build_inner_valid()
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
            inner_valid_factory=self._make_inner_valid_factory(),
            n_rows=len(X),
            fixed_params=fixed,
        )

        result = tuner.tune(X, y, groups)
        self._best_params = result.best_params
        self._tuning_result = result
        _log.info("event='tune.done' best_params=%s", result.best_params)
        return result

    def tuning_table(self) -> pd.DataFrame:
        """Return a DataFrame of all tuning trial results.

        Columns: ``trial``, metric name, and each searched parameter name.

        Returns:
            DataFrame with one row per trial.

        Raises:
            LizyMLError with MODEL_NOT_FIT when ``tune()`` has not been called.
        """
        if self._tuning_result is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="tune() has not been called yet.",
                context={},
            )
        tr = self._tuning_result
        rows = []
        for t in tr.trials:
            row: dict[str, Any] = {
                "trial": t.number,
                tr.metric_name: t.score,
                **t.params,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def tuning_plot(self) -> Any:
        """Plot tuning history. Requires ``tune()`` to have been called.

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when ``tune()`` has not been called.
            LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
        """
        if self._tuning_result is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="tune() has not been called yet.",
                context={},
            )
        from lizyml.plots.tuning import plot_tuning_history

        return plot_tuning_history(self._tuning_result)

    def importance_plot(self, kind: str = "split", top_n: int | None = 20) -> Any:
        """Plot fold-averaged feature importances as a horizontal bar chart.

        Args:
            kind: ``"split"``, ``"gain"``, or ``"shap"``.
            top_n: Maximum number of features to show.

        Returns:
            A ``plotly.graph_objects.Figure`` object.

        Raises:
            LizyMLError with MODEL_NOT_FIT when called before fit.
            LizyMLError with OPTIONAL_DEP_MISSING when plotly (or shap for
                ``kind="shap"``) is not installed.
        """
        if kind == "shap":
            imp = self.importance(kind="shap")
            from lizyml.plots.importance import plot_importance_from_dict

            return plot_importance_from_dict(imp, top_n=top_n)

        fit_result = self._require_fit()
        from lizyml.plots.importance import plot_importance

        return plot_importance(fit_result, kind=kind, top_n=top_n)

    def plot_learning_curve(self) -> Any:
        """Plot per-fold training/validation loss vs iteration.

        Returns:
            A ``plotly.graph_objects.Figure`` object.

        Raises:
            LizyMLError with MODEL_NOT_FIT when called before fit or when
                no evaluation history is available.
            LizyMLError with OPTIONAL_DEP_MISSING when plotly is not installed.
        """
        fit_result = self._require_fit()
        from lizyml.plots.learning_curve import plot_learning_curve

        return plot_learning_curve(fit_result)

    def plot_oof_distribution(self) -> Any:
        """Plot the distribution of out-of-fold predictions.

        Returns:
            A ``plotly.graph_objects.Figure`` object.

        Raises:
            LizyMLError with MODEL_NOT_FIT when called before fit.
            LizyMLError with OPTIONAL_DEP_MISSING when plotly is not installed.
        """
        fit_result = self._require_fit()
        from lizyml.plots.oof_distribution import plot_oof_distribution

        return plot_oof_distribution(fit_result)

    def export(self, path: str | Path) -> None:
        """Export Model artifacts to a directory.

        Saves ``fit_result.pkl``, ``refit_model.pkl``, ``metadata.json``,
        and ``analysis_context.pkl`` under *path*.  The saved model can be
        restored with :meth:`load`, including diagnostic API support.

        Args:
            path: Output directory (created if absent).

        Raises:
            LizyMLError with MODEL_NOT_FIT when called before ``fit``.
            LizyMLError with SERIALIZATION_FAILED on I/O errors.

        Warning:
            The ``.pkl`` files use joblib/pickle.  Only load artifacts from
            trusted sources.
        """
        fit_result = self._require_fit()
        refit_result = self._require_refit()
        from lizyml.persistence.exporter import AnalysisContext
        from lizyml.persistence.exporter import export as _export

        ctx: AnalysisContext | None = None
        if self._y is not None and self._X is not None:
            ctx = AnalysisContext(y_true=self._y, X_for_explain=self._X)

        _export(
            path=path,
            fit_result=fit_result,
            refit_result=refit_result,
            config=self._cfg.model_dump(),
            task=self._cfg.task,
            analysis_context=ctx,
        )
        _log.info("event='export.done' path=%s", path)

    @classmethod
    def load(cls, path: str | Path) -> Model:
        """Restore a Model from a directory created by :meth:`export`.

        Args:
            path: Directory containing ``metadata.json``, ``fit_result.pkl``,
                and ``refit_model.pkl``.

        Returns:
            A :class:`Model` instance ready for ``predict`` and ``evaluate``.

        Raises:
            LizyMLError with DESERIALIZATION_FAILED on validation or I/O errors.

        Warning:
            Only load from trusted sources — joblib uses pickle internally.
        """
        from lizyml.persistence.loader import load as _load

        fit_result, refit_result, metadata, analysis_context = _load(path)
        config = metadata["config"]
        instance = cls(config)
        instance._fit_result = fit_result
        instance._refit_result = refit_result
        instance._metrics = fit_result.metrics
        if analysis_context is not None:
            instance._y = analysis_context.y_true
            instance._X = analysis_context.X_for_explain
        _log.info("event='load.done' path=%s run_id=%s", path, metadata.get("run_id"))
        return instance

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

    def _build_splitter(self) -> BaseSplitter:
        """Instantiate splitter from config."""
        import warnings

        split_cfg = self._cfg.split
        method = split_cfg.method
        n_splits = split_cfg.n_splits
        random_state = getattr(split_cfg, "random_state", 42)
        shuffle = getattr(split_cfg, "shuffle", True)

        # Warn if classification task explicitly uses kfold (H-0013)
        if method == "kfold" and self._cfg.task in ("binary", "multiclass"):
            warnings.warn(
                f"task='{self._cfg.task}' with split.method='kfold' does not "
                "preserve class distribution. Consider using 'stratified_kfold' "
                "instead.",
                UserWarning,
                stacklevel=2,
            )

        if method == "stratified_kfold":
            return StratifiedKFoldSplitter(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
        if method == "group_kfold":
            return GroupKFoldSplitter(n_splits=n_splits)
        if method == "time_series":
            gap = getattr(split_cfg, "gap", 0)
            return TimeSeriesSplitter(n_splits=n_splits, gap=gap)
        # Default: kfold
        return KFoldSplitter(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def _build_inner_valid(
        self,
    ) -> (
        HoldoutInnerValid
        | GroupHoldoutInnerValid
        | TimeHoldoutInnerValid
        | NoInnerValid
    ):
        """Instantiate inner validation strategy from training config.

        When early stopping is enabled but ``inner_valid`` is not explicitly
        set, the strategy is auto-resolved based on the outer split method:

        - ``stratified_kfold`` → ``HoldoutInnerValid(stratify=True)``
        - ``group_kfold`` → ``GroupHoldoutInnerValid``
        - ``time_series`` → ``TimeHoldoutInnerValid``
        - ``kfold`` (or other) → ``HoldoutInnerValid(stratify=False)``
        """
        es = self._cfg.training.early_stopping
        if not es.enabled:
            return NoInnerValid()

        iv_cfg = es.inner_valid
        split_method = self._cfg.split.method

        # Auto-resolve when inner_valid is not explicitly set (includes
        # the case where it was auto-created from validation_ratio default)
        if iv_cfg is None or not es._inner_valid_explicit:
            ratio = iv_cfg.ratio if iv_cfg is not None else 0.1
            seed = self._cfg.training.seed
            if split_method == "stratified_kfold":
                return HoldoutInnerValid(ratio=ratio, random_state=seed, stratify=True)
            if split_method == "group_kfold":
                return GroupHoldoutInnerValid(ratio=ratio, random_state=seed)
            if split_method == "time_series":
                return TimeHoldoutInnerValid(ratio=ratio)
            return HoldoutInnerValid(ratio=ratio, random_state=seed, stratify=False)

        # Explicit config — use getattr for fields not common to all variants
        method = iv_cfg.method
        if method == "holdout":
            return HoldoutInnerValid(
                ratio=iv_cfg.ratio,
                random_state=getattr(iv_cfg, "random_state", 42),
                stratify=getattr(iv_cfg, "stratify", False),
            )
        if method == "group_holdout":
            return GroupHoldoutInnerValid(
                ratio=iv_cfg.ratio,
                random_state=getattr(iv_cfg, "random_state", 42),
            )
        if method == "time_holdout":
            return TimeHoldoutInnerValid(ratio=iv_cfg.ratio)
        return NoInnerValid()

    def _make_inner_valid_factory(
        self,
    ) -> Callable[
        [float],
        HoldoutInnerValid | GroupHoldoutInnerValid | TimeHoldoutInnerValid,
    ]:
        """Return a factory that produces InnerValidStrategy for a given ratio.

        Used by the Tuner when ``validation_ratio`` is a search dimension.
        """
        split_method = self._cfg.split.method
        seed = self._cfg.training.seed

        def factory(
            ratio: float,
        ) -> HoldoutInnerValid | GroupHoldoutInnerValid | TimeHoldoutInnerValid:
            if split_method == "stratified_kfold":
                return HoldoutInnerValid(ratio=ratio, random_state=seed, stratify=True)
            if split_method == "group_kfold":
                return GroupHoldoutInnerValid(ratio=ratio, random_state=seed)
            if split_method == "time_series":
                return TimeHoldoutInnerValid(ratio=ratio)
            return HoldoutInnerValid(ratio=ratio, random_state=seed, stratify=False)

        return factory

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

    @property
    def fit_result(self) -> FitResult:
        """Read-only access to the CV training result.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when ``fit()`` has not been called.
        """
        return self._require_fit()

    def params_table(self) -> pd.DataFrame:
        """Return resolved parameters as a single-column DataFrame.

        Merges Config smart params, training settings, resolved booster
        params (fold 0), and per-fold ``best_iteration`` into one table.

        Returns:
            :class:`pd.DataFrame` with index ``parameter`` and column ``value``.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit``.
        """
        fr = self._require_fit()
        model_cfg = self._cfg.model

        rows: list[dict[str, Any]] = []

        # --- Config smart params ---
        if isinstance(model_cfg, LGBMConfig):
            smart = {
                "auto_num_leaves": model_cfg.auto_num_leaves,
                "num_leaves_ratio": model_cfg.num_leaves_ratio,
                "min_data_in_leaf_ratio": model_cfg.min_data_in_leaf_ratio,
                "min_data_in_bin_ratio": model_cfg.min_data_in_bin_ratio,
                "balanced": model_cfg.balanced,
                "feature_weights": model_cfg.feature_weights,
            }
            for k, v in smart.items():
                rows.append({"parameter": k, "value": v})

        # --- Config training params ---
        es = self._cfg.training.early_stopping
        if es is not None:
            rows.append({"parameter": "early_stopping_rounds", "value": es.rounds})
            rows.append({"parameter": "validation_ratio", "value": es.validation_ratio})

        # --- Resolved booster params (fold 0) ---
        booster = fr.models[0].get_native_model().booster_
        for k in [
            "objective",
            "learning_rate",
            "max_depth",
            "num_leaves",
            "min_data_in_leaf",
            "min_data_in_bin",
            "max_bin",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "num_iterations",
        ]:
            v = booster.params.get(k)
            if v is not None:
                rows.append({"parameter": k, "value": v})

        # task-specific params
        for k in ["scale_pos_weight", "num_class"]:
            v = booster.params.get(k)
            if v is not None:
                rows.append({"parameter": k, "value": v})

        # --- Best iteration per fold ---
        for i, m in enumerate(fr.models):
            rows.append({"parameter": f"best_iteration_{i}", "value": m.best_iteration})

        df = pd.DataFrame(rows)
        return df.set_index("parameter")

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
