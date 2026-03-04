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
from datetime import datetime
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Literal

import numpy as np
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
from lizyml.data import dataframe_builder, datasource
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.evaluation.evaluator import Evaluator
from lizyml.features.pipelines_native import NativeFeaturePipeline
from lizyml.splitters.kfold import KFoldSplitter, StratifiedKFoldSplitter
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.inner_valid import HoldoutInnerValid, NoInnerValid
from lizyml.training.refit_trainer import RefitResult, RefitTrainer
from lizyml.tuning.search_space import parse_space
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
        )
        fit_result = cv_trainer.fit(
            X, y, groups, data_fingerprint=fingerprint, run_meta=run_meta
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

        if metrics is None and self._metrics is not None:
            return self._metrics

        # Recompute with specified metrics
        from lizyml.data.dataframe_builder import DataFrameComponents  # noqa: F401

        raise LizyMLError(
            ErrorCode.MODEL_NOT_FIT,
            user_message=(
                "evaluate() with custom metrics requires access to the training "
                "labels. Call fit(data=df) first and pass the same data here, "
                "or use the pre-computed metrics from fit() without arguments."
            ),
            context={},
        )

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
            return_shap: Not yet implemented (Phase 15).

        Returns:
            :class:`~lizyml.core.types.predict_result.PredictionResult`.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit``.
        """
        self._require_fit()
        refit = self._require_refit()

        # Restore pipeline from saved state
        pipeline = NativeFeaturePipeline()
        pipeline.load_state(refit.pipeline_state)

        X_t, warnings = pipeline.transform_with_warnings(X)

        model = refit.model
        task = self._cfg.task

        pred: np.ndarray
        proba: np.ndarray | None = None

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

        return PredictionResult(
            pred=pred,
            proba=proba,
            shap_values=None,
            used_features=refit.feature_names,
            warnings=warnings,
        )

    def importance(self, kind: str = "split") -> dict[str, float]:
        """Return averaged feature importance across CV fold models.

        Args:
            kind: ``"split"`` or ``"gain"``.

        Returns:
            Dict mapping feature name → importance score.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit``.
        """
        fit_result = self._require_fit()
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
    ) -> dict[str, Any]:
        """Run hyperparameter search with optuna.

        Requires ``tuning`` section in the config.  Best params are stored
        internally and used automatically in the next ``fit()`` call.

        Args:
            data: Training DataFrame.  Overrides any data from construction
                or ``data.path`` in config.

        Returns:
            Dict of best hyperparameter values found by the search.

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
        space = parse_space(cfg.tuning.optuna.space)
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
        )

        best_params = tuner.tune(X, y, groups)
        self._best_params = best_params
        _log.info("event='tune.done' best_params=%s", best_params)
        return best_params

    def export(self, path: str | Path) -> None:
        """Not yet implemented — Phase 14."""
        raise NotImplementedError("Export will be implemented in Phase 14.")

    @classmethod
    def load(cls, path: str | Path) -> Model:
        """Not yet implemented — Phase 14."""
        raise NotImplementedError("Persistence will be implemented in Phase 14.")

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

    def _build_splitter(self) -> KFoldSplitter | StratifiedKFoldSplitter:
        """Instantiate splitter from config."""
        split_cfg = self._cfg.split
        method = split_cfg.method
        n_splits = split_cfg.n_splits
        random_state = getattr(split_cfg, "random_state", 42)
        shuffle = getattr(split_cfg, "shuffle", True)

        if method == "stratified_kfold":
            return StratifiedKFoldSplitter(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
        # Default: kfold (also fallback for unsupported methods in Phase 11)
        return KFoldSplitter(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def _build_inner_valid(self) -> HoldoutInnerValid | NoInnerValid:
        """Instantiate inner validation strategy from training config."""
        es = self._cfg.training.early_stopping
        if not es.enabled or es.inner_valid is None:
            return NoInnerValid()
        return HoldoutInnerValid(
            ratio=es.inner_valid.ratio,
            random_state=es.inner_valid.random_state,
        )

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
