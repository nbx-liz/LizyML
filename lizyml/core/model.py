"""Model — the public-facing facade for LizyML.

Model's responsibilities: assembly and delegation only.

What Model does:
1. Validate Config → derive Specs.
2. Load data via DataSource → DataFrameBuilder.
3. Select components via Registry.
4. Delegate to CVTrainer / Evaluator / RefitTrainer.
5. Store FitResult and RefitResult; expose them via evaluate / predict.

What Model does NOT contain:
- OOF/IF generation logic     → training/oof_assembly.py
- Metric computation          → evaluation/evaluator.py
- Estimator-specific logic    → estimators/<name>/provider.py (via EstimatorProvider)
- Plot implementations        → plots/*
- Persistence details         → persistence/*

Mixin decomposition (H-0042):
- Plot methods      → _model_plots.py (ModelPlotsMixin)        [read-only]
- Table/accessors   → _model_tables.py (ModelTablesMixin)      [read-only]
- Export/load       → _model_persistence.py (ModelPersistenceMixin) [read-only]
- Tuning orchestr.  → _model_tuning.py (ModelTuningMixin)      [writer; H-0091]
- Splitter/IV build → _model_factories.py (module-level functions)

Diagnostic mixins (the three read-only ones) read state exclusively through
``_get_fit_state()`` / ``_get_tuning_state()`` (H-0077). ``ModelTuningMixin`` is
the writer-exempt orchestrator (runs during the mutating ``tune()`` lifecycle).
"""

from __future__ import annotations

import dataclasses
import sys
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    pass

from lizyml import __version__
from lizyml.config.loader import load_config
from lizyml.config.schema import (
    BlockedGroupKFoldConfig,
    LizyMLConfig,
)
from lizyml.core._model_factories import (
    build_inner_valid,
    build_splitter,
    get_provider,
    make_inner_valid_factory,
)
from lizyml.core._model_metrics import (
    _DEFAULT_METRICS,
    assemble_calibrated_metrics,
    filter_metrics,
)
from lizyml.core._model_persistence import ModelPersistenceMixin
from lizyml.core._model_plots import ModelPlotsMixin
from lizyml.core._model_predict import run_predict
from lizyml.core._model_state import FitState
from lizyml.core._model_tables import ModelTablesMixin
from lizyml.core._model_tuning import ModelTuningMixin
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import generate_run_id, get_logger
from lizyml.core.specs.feature_spec import FeatureSpec
from lizyml.core.specs.problem_spec import ProblemSpec
from lizyml.core.train_components import TrainComponents
from lizyml.core.types.artifacts import RunMeta
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.predict_result import PredictionResult
from lizyml.core.types.tuning_result import (
    RoundSummary,
    TuningResult,
)
from lizyml.data import dataframe_builder, datasource
from lizyml.data.dataframe_builder import DataFrameComponents
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.estimators.provider import EstimatorProvider
from lizyml.evaluation.evaluator import Evaluator
from lizyml.metrics.registry import (
    get_metrics_for_task,
    parse_metric_entries,
)
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.inner_valid import BaseInnerValidStrategy
from lizyml.training.refit_trainer import RefitResult, RefitTrainer
from lizyml.tuning.search_space import (
    parse_space,
)

_log = get_logger("model")

_TS_METHODS = frozenset({"time_series", "purged_time_series", "group_time_series"})
_BLOCK_METHODS = frozenset({"blocked_group_kfold"})


class Model(ModelPlotsMixin, ModelTablesMixin, ModelPersistenceMixin, ModelTuningMixin):
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
        self._tuning_result: TuningResult | None = None
        self._y: pd.Series | None = None  # transient; not persisted
        self._X: pd.DataFrame | None = None  # transient; not persisted
        self._provider: EstimatorProvider | None = None  # set by fit/tune
        self._block_values: npt.NDArray[Any] | None = None  # _prepare_training_data
        # H-0068: re-tune state
        self._study: Any = None  # Optuna study for resume
        self._round_number: int = 0  # completed rounds count
        self._rounds: list[RoundSummary] = []  # round history
        self._space: list[Any] | None = None  # last search space used
        self._used_default_space: bool = False  # track for expand_boundary default

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
            params: Model parameters to override the config ``model.params``.

        Returns:
            The :class:`~lizyml.core.types.fit_result.FitResult` from CV.
        """
        cfg = self._cfg
        run_id = generate_run_id()

        self._ensure_run_dir(run_id)
        _log.info("event='fit.start' run_id=%s task=%s", run_id, cfg.task)

        # A fit() after tune() reuses the identical deterministic CV splits used
        # to select the params, so the reported OOF metrics are optimistically
        # biased (documented user-side policy, BLUEPRINT §11.6). Surface it once
        # per fit so it is not silently unenforced (#218).
        if self._tuning_result is not None:
            _log.warning(
                "event='fit.post_tune' run_id=%s "
                "msg='fit() after tune() reuses the tuning CV splits; reported "
                "OOF metrics are optimistically biased (BLUEPRINT 11.6). Use a "
                "held-out set for an unbiased estimate.'",
                run_id,
            )

        # --- Load & prepare data ---------------------------------------------
        X, y, groups, components = self._prepare_training_data(data)
        self._X, self._y = X, y
        fingerprint = fp_compute(X, file_path=None)

        # --- Build components (H-0050/H-0053: provider-based) ----------------
        provider = get_provider(cfg.model)
        self._provider = provider
        run_meta = self._build_run_meta(run_id)
        model_params, smart_params = self._merge_params(provider)
        training_overrides = (
            self._tuning_result.best_training_params
            if self._tuning_result is not None
            else {}
        )
        tc = self._build_train_components(
            X,
            y,
            provider=provider,
            model_params=model_params,
            smart_params=smart_params,
            training_overrides=training_overrides,
        )
        splitter = build_splitter(
            cfg,
            block_values=self._block_values,
            task=cfg.task,
            seed=cfg.training.seed,
        )
        n_classes = int(y.nunique()) if cfg.task == "multiclass" else None
        pipeline_factory = provider.build_pipeline_factory()

        # --- CV training -----------------------------------------------------
        cv_trainer = CVTrainer(
            outer_splitter=splitter,
            inner_valid=tc.inner_valid,
            pipeline_factory=pipeline_factory,
            estimator_factory=tc.estimator_factory,
            task=cfg.task,
            n_classes=n_classes,
            ratio_param_resolver=tc.ratio_resolver,
            collect_raw_scores=(cfg.calibration is not None),
        )
        time_values = components.time_col if components.time_col is not None else None
        fit_result = cv_trainer.fit(
            X,
            y,
            groups,
            data_fingerprint=fingerprint,
            run_meta=run_meta,
            sample_weight=tc.sample_weight,
            time_values=time_values,
        )

        # H-0070: attach target encoder so predict() / persistence / codegen
        # can map predicted int codes back to the original labels.
        fit_result = dataclasses.replace(
            fit_result, target_encoder=components.target_encoder
        )

        # --- Calibration (binary only) ---------------------------------------
        fit_result = self._run_calibration(cfg, fit_result, y, groups)

        # --- Evaluation -------------------------------------------------------
        metric_entries = cfg.evaluation.metrics or _DEFAULT_METRICS[cfg.task]
        evaluator = Evaluator(task=cfg.task)
        metrics = evaluator.evaluate(fit_result, y, metric_entries)
        metrics = assemble_calibrated_metrics(
            fit_result, y, metric_entries, evaluator, metrics
        )

        fit_result = dataclasses.replace(
            fit_result, metrics={**fit_result.metrics, **metrics}
        )
        self._metrics = metrics

        # --- Full-data refit (for predict) -----------------------------------
        refit_trainer = RefitTrainer(
            inner_valid=tc.inner_valid,
            pipeline_factory=pipeline_factory,
            estimator_factory=tc.estimator_factory,
            task=cfg.task,
            ratio_param_resolver=tc.ratio_resolver,
        )
        self._refit_result = refit_trainer.fit(X, y, groups)

        self._fit_result = fit_result
        _log.info("event='fit.done' run_id=%s", run_id)
        # Return a selective deep copy (FitResult.__deepcopy__): mutating the
        # primary return path must not corrupt internal state or a later
        # export() — the same H-0082 defense as the ``fit_result`` property
        # (#204 / H-0086). Trained estimators stay shared by reference.
        return deepcopy(fit_result)

    def evaluate(
        self,
        metrics: list[str | dict[str, dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Return structured evaluation metrics from the last ``fit``.

        Args:
            metrics: Metric names or parameterised ``MetricEntry`` dicts to
                filter.  When ``None`` uses defaults or config-defined metrics
                (already computed during ``fit``).

        Returns:
            Structured dict: ``{"raw": {"oof": ..., "oof_per_fold": ...,
            "if_mean": ..., "if_per_fold": ..., "oof_coverage": float}}``.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit``.
        """
        self._require_fit()

        if self._metrics is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="Metrics not computed. Call fit() first.",
                context={"task": self._cfg.task, "fit_called": False},
            )

        if metrics is None:
            # Return an independent copy so callers cannot mutate internal
            # state (and thereby contaminate a later export()) — H-0082.
            return deepcopy(self._metrics)

        # Validate task compatibility first (raises UNSUPPORTED_METRIC if invalid)
        get_metrics_for_task(metrics, self._cfg.task)  # raises on unknown/incompatible

        # Extract metric names for filtering (H-0065)
        names = {name for name, _kwargs in parse_metric_entries(metrics)}

        # Filter the pre-computed metrics dict to the requested subset
        return filter_metrics(self._metrics, names)

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
        fit = self._require_fit()
        refit = self._require_refit()
        provider = get_provider(self._cfg.model)

        # Estimator/calibration/SHAP branching lives behind this seam (#172);
        # the facade only resolves state and delegates.
        return run_predict(
            task=self._cfg.task,
            fit_result=fit,
            refit_result=refit,
            provider=provider,
            X=X,
            return_shap=return_shap,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fit_result(self) -> FitResult:
        """Read-only access to the CV training result.

        Returns an independent copy each call: mutable data fields are
        deep-copied so mutating the result cannot corrupt internal state (or a
        later ``export()``). Trained estimators (``models`` / ``calibrator`` /
        ``pipeline_state``) are shared by reference and must be treated as
        read-only — see :meth:`FitResult.__deepcopy__` (H-0082).

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when ``fit()`` has not been called.
        """
        # Selective deep copy (FitResult.__deepcopy__): mutable data copied,
        # trained estimators shared by reference to preserve Booster fidelity.
        return deepcopy(self._require_fit())

    # ------------------------------------------------------------------
    # Internal helpers — TrainComponents (H-0050)
    # ------------------------------------------------------------------

    def _merge_params(
        self,
        provider: Any,
        override: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Merge model and smart params with priority:
        Config defaults < tune best < fit() args.

        Args:
            provider: EstimatorProvider instance.
            override: Optional fit() arg overrides (highest priority).

        Returns:
            (model_params, smart_params) tuple.
        """
        cfg = self._cfg
        model_cfg = cfg.model

        # --- model_params / smart_params: Config defaults via provider ---
        model_params = provider.extract_model_params(model_cfg)
        smart_params = provider.extract_smart_params(model_cfg)

        # --- Overlay tune best ---
        if self._tuning_result is not None:
            # Apply default fixed params when default space was used (#76).
            # cfg.tuning is always set when _tuning_result exists (tune() sets
            # both), but guard defensively for unit tests that inject
            # _tuning_result directly.
            used_default_space = cfg.tuning is not None and not parse_space(
                cfg.tuning.optuna.space
            )
            if used_default_space:
                fixed = provider.default_fixed_params(cfg.task)
                model_params = {**model_params, **fixed}

            model_params = {
                **model_params,
                **self._tuning_result.best_model_params,
            }
            if self._tuning_result.best_smart_params:
                smart_params = {
                    **smart_params,
                    **self._tuning_result.best_smart_params,
                }

        # --- Overlay fit() args (highest priority) ---
        if override:
            model_params = {**model_params, **override}

        return model_params, smart_params

    def _build_train_components(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        provider: Any,
        model_params: dict[str, Any],
        smart_params: dict[str, Any],
        training_overrides: dict[str, Any] | None = None,
    ) -> TrainComponents:
        """Build shared training components for CVTrainer and RefitTrainer.

        Delegates estimator-specific logic to the provider (H-0053).

        Args:
            X: Feature DataFrame.
            y: Target Series.
            provider: EstimatorProvider instance.
            model_params: Merged model params (from ``_merge_params``).
            smart_params: Merged smart params (from ``_merge_params``).
            training_overrides: Optional training param overrides from tuning
                (``early_stopping_rounds``, ``validation_ratio``).  (#76)

        Returns:
            :class:`TrainComponents` ready to pass to both trainers.
        """
        cfg = self._cfg
        n_classes = int(y.nunique()) if cfg.task == "multiclass" else None
        tp = training_overrides or {}

        # --- Resolve smart params (Stage 1: data-size independent) ---
        sample_weight: npt.NDArray[np.float64] | None = None
        resolved_model = dict(model_params)

        if smart_params:
            smart_resolved, sample_weight = provider.resolve_smart_params(
                smart=smart_params,
                effective_params=resolved_model,
                n_rows=len(X),
                feature_names=list(X.columns),
                y=y,
                task=cfg.task,
            )
            resolved_model = {**resolved_model, **smart_resolved}

        # --- Build per-fold ratio resolver (Stage 2: n_rows dependent) ---
        ratio_resolver = provider.build_ratio_resolver(smart_params)

        # --- Resolve early stopping rounds (config < tune override) ---
        esr: int | None
        if "early_stopping_rounds" in tp:
            esr = int(tp["early_stopping_rounds"])
        elif cfg.training.early_stopping.enabled:
            esr = cfg.training.early_stopping.rounds
        else:
            esr = None

        # --- Build estimator factory ---
        estimator_factory = provider.build_estimator_factory(
            task=cfg.task,
            params=resolved_model,
            n_classes=n_classes,
            early_stopping_rounds=esr,
            seed=cfg.training.seed,
        )

        # --- Inner validation (config < tune override) ---
        inner_valid: BaseInnerValidStrategy
        if "validation_ratio" in tp:
            iv_factory = make_inner_valid_factory(cfg)
            inner_valid = iv_factory(tp["validation_ratio"])
        else:
            inner_valid = build_inner_valid(cfg)

        return TrainComponents(
            estimator_factory=estimator_factory,
            sample_weight=sample_weight,
            ratio_resolver=ratio_resolver,
            inner_valid=inner_valid,
        )

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
            context={
                # Defensive: only the boolean is logged so that user file
                # paths never leak through error logs / repr (cf. #118 audit).
                "cfg_data_path_set": bool(self._cfg.data.path),
                "constructor_data": self._data is not None,
            },
        )

    def _prepare_training_data(
        self, data: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.Series, npt.NDArray[Any] | None, DataFrameComponents]:
        """Load data, build specs, and prepare X/y/groups for training.

        Handles time-series sorting when the split method requires it.
        """
        cfg = self._cfg
        self._block_values = None  # reset per-call transient state
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
        groups: npt.NDArray[Any] | None = (
            components.group_col.to_numpy()
            if components.group_col is not None
            else None
        )

        # Determine split-driven ordering/extraction (config concerns stay
        # here; the pure data transforms live in dataframe_builder — #209).
        blocked: tuple[str, str] | None = None
        if cfg.split.method in _BLOCK_METHODS:
            if not isinstance(cfg.split, BlockedGroupKFoldConfig):
                raise LizyMLError(
                    code=ErrorCode.CONFIG_INVALID,
                    user_message=(
                        "Internal error: expected BlockedGroupKFoldConfig "
                        f"for method '{cfg.split.method}'."
                    ),
                    context={"split_method": cfg.split.method},
                )
            blocked = (cfg.split.blocks.col, cfg.split.groups.col)

        components, groups, self._block_values = dataframe_builder.prepare_for_split(
            df,
            components,
            groups,
            time_series=cfg.split.method in _TS_METHODS,
            method_name=cfg.split.method,
            blocked=blocked,
        )
        return components.X, components.y, groups, components

    def _build_run_meta(self, run_id: str) -> RunMeta:
        def _ver(pkg: str) -> str:
            try:
                return pkg_version(pkg)
            except Exception:
                return "unknown"

        # Common deps + estimator-specific deps via provider (H-0054)
        deps: dict[str, str] = {
            "pandas": _ver("pandas"),
            "numpy": _ver("numpy"),
            "scikit-learn": _ver("scikit-learn"),
        }
        if self._provider is not None:
            deps.update(self._provider.runtime_deps())

        return RunMeta(
            lizyml_version=__version__,
            python_version=sys.version,
            deps_versions=deps,
            config_normalized=self._cfg.model_dump(),
            config_version=self._cfg.config_version,
            run_id=run_id,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

    def _ensure_run_dir(self, run_id: str) -> None:
        """Set up the output directory for a run if configured."""
        if self._output_dir is not None:
            from lizyml.core.logging import setup_output_dir

            self._run_dir = setup_output_dir(self._output_dir, run_id)

    def _run_calibration(
        self,
        cfg: LizyMLConfig,
        fit_result: FitResult,
        y: pd.Series,
        groups: npt.NDArray[Any] | None,
    ) -> FitResult:
        """Apply cross-fit calibration if configured. Returns updated FitResult."""
        if cfg.calibration is None:
            return fit_result

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
        # Inherit training.seed for isotonic's internal validation split when
        # no explicit calibration seed is given (H-0080). Other calibrators
        # (platt / beta) do not use a seed, so leave their params untouched.
        cal_params_dict = dict(cfg.calibration.params or {})
        if method == "isotonic":
            cal_params_dict.setdefault("seed", cfg.training.seed)
        cal_params = cal_params_dict or None
        # Use raw scores (logits) for calibration (H-0030)
        cal_scores = (
            fit_result.oof_raw_scores
            if fit_result.oof_raw_scores is not None
            else fit_result.oof_pred
        )
        y_arr = y.to_numpy()
        # Reuse outer CV splits for calibration cross-fit (H-0058).
        # Calibration input is (oof_scores, y) only — no X leakage.
        cal_split_indices = fit_result.splits.outer
        calibration_result = cross_fit_calibrate(
            oof_scores=cal_scores,
            y=y_arr,
            calibrator_factory=lambda: get_calibrator(method, params=cal_params),
            split_indices=cal_split_indices,
            oof_pred=fit_result.oof_pred,
        )
        new_splits = dataclasses.replace(
            fit_result.splits,
            calibration=cal_split_indices,
        )
        return dataclasses.replace(
            fit_result,
            calibrator=calibration_result,
            splits=new_splits,
        )

    def _require_fit(self) -> FitResult:
        if self._fit_result is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="Model has not been fitted. Call fit() first.",
                context={"task": self._cfg.task, "method": "fit"},
            )
        return self._fit_result

    def _require_refit(self) -> RefitResult:
        if self._refit_result is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="Model has not been fitted. Call fit() first.",
                context={
                    "task": self._cfg.task,
                    "method": "refit",
                    "fit_done": self._fit_result is not None,
                },
            )
        return self._refit_result

    def _get_fit_state(self) -> FitState:
        """Return a frozen snapshot of post-fit state for Mixin methods (#112).

        Single read path for ``ModelPlotsMixin`` / ``ModelTablesMixin`` /
        ``ModelPersistenceMixin``. After H-0077 (Phase 2) Mixin methods read
        state exclusively from the returned ``FitState`` — direct ``self._*``
        access from Mixin bodies is forbidden.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit()`` (delegated via
            :meth:`_require_fit`).
        """
        fit_result = self._require_fit()
        if self._provider is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    "Provider is not initialised. Call fit() before "
                    "diagnostic / export APIs."
                ),
                context={"method": "_get_fit_state"},
            )
        return FitState(
            cfg=self._cfg,
            fit_result=fit_result,
            refit_result=self._refit_result,
            tuning_result=self._tuning_result,
            provider=self._provider,
            metrics=self._metrics,
            y=self._y,
            X=self._X,
            run_dir=self._run_dir,
            output_dir=self._output_dir,
        )

    def _resolve_export_path(self, path: str | Path | None) -> Path:
        """Resolve the export destination directory (H-0077: moved from Mixin).

        Path resolution (first match wins):

        1. Explicit *path* argument.
        2. ``{run_dir}/export`` when a run directory exists from
           ``fit()`` / ``tune()``.
        3. New run directory under ``output_dir`` if configured.
        4. Error — no destination available.

        This method writes to ``self._run_dir`` when allocating a new run
        directory in case 3, which is why it lives on the Model facade rather
        than on the (frozen-state) Mixin.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``SERIALIZATION_FAILED`` when no destination can be resolved.
        """
        if path is not None:
            return Path(path)
        if self._run_dir is not None:
            return Path(self._run_dir) / "export"
        if self._output_dir is not None:
            from lizyml.core.logging import setup_output_dir

            export_run_id = generate_run_id()
            self._run_dir = setup_output_dir(self._output_dir, export_run_id)
            return Path(self._run_dir) / "export"
        raise LizyMLError(
            ErrorCode.SERIALIZATION_FAILED,
            user_message=(
                "No export path provided and no output_dir configured. "
                "Pass an explicit path or set output_dir in Config / constructor."
            ),
        )
