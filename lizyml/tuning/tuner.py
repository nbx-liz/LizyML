"""Tuner — optuna-backed hyperparameter optimisation."""

from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import numpy.typing as npt
import pandas as pd

from lizyml import __version__
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import generate_run_id, get_logger
from lizyml.core.types.artifacts import RunMeta
from lizyml.core.types.tuning_result import (
    TrialResult,
    TuneProgressCallback,
    TuneProgressInfo,
    TuningResult,
)
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.estimators.lgbm import (
    _COMMON_DEFAULTS,
    resolve_ratio_params,
    resolve_smart_params,
)
from lizyml.evaluation.evaluator import Evaluator
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.inner_valid import BaseInnerValidStrategy
from lizyml.tuning.search_space import SearchDim, split_by_category, suggest_params

if TYPE_CHECKING:
    from lizyml.estimators.lgbm import LGBMAdapter
    from lizyml.features.pipelines_native import NativeFeaturePipeline
    from lizyml.splitters.base import BaseSplitter

_optuna: Any = None
try:
    import optuna

    _optuna = optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
except ImportError:  # pragma: no cover
    pass

_log = get_logger("tuner")

TaskType = Literal["regression", "binary", "multiclass"]


class Tuner:
    """Optuna-backed hyperparameter tuner.

    Uses the same CV splitter as Model.fit for directly comparable OOF scores.

    Args:
        task: ML task type.
        outer_splitter: CV splitter (same as used in Model.fit).
        inner_valid: Inner validation strategy (default for all trials).
        pipeline_factory: Factory producing NativeFeaturePipeline instances.
        estimator_factory: Factory accepting a trial params dict → LGBMAdapter.
        dims: Parsed search space dimensions.
        n_trials: Number of optuna trials.
        direction: Optimisation direction (minimize or maximize).
        timeout: Optional timeout in seconds.
        metric_name: Metric to optimise (OOF score).
        n_classes: Number of classes for multiclass tasks.
        seed: Random seed for the optuna sampler.
        inner_valid_factory: Optional factory to rebuild inner valid per trial
            when ``validation_ratio`` is a search dim.
        n_rows: Number of training rows (for smart param resolution).
        fixed_params: Fixed params applied to every trial's model params.
        progress_callback: Optional callback invoked after each trial with
            a :class:`TuneProgressInfo` (H-0048).
    """

    def __init__(
        self,
        task: TaskType,
        outer_splitter: BaseSplitter,
        inner_valid: BaseInnerValidStrategy,
        pipeline_factory: Callable[[], NativeFeaturePipeline],
        estimator_factory: Callable[[dict[str, Any]], LGBMAdapter],
        dims: list[SearchDim],
        n_trials: int = 50,
        direction: Literal["minimize", "maximize"] = "minimize",
        timeout: float | None = None,
        metric_name: str = "rmse",
        n_classes: int | None = None,
        seed: int = 42,
        *,
        inner_valid_factory: Callable[[float], BaseInnerValidStrategy] | None = None,
        n_rows: int | None = None,
        fixed_params: dict[str, Any] | None = None,
        progress_callback: TuneProgressCallback | None = None,
    ) -> None:
        self.task = task
        self.outer_splitter = outer_splitter
        self.inner_valid = inner_valid
        self.pipeline_factory = pipeline_factory
        self.estimator_factory = estimator_factory
        self.dims = dims
        self.n_trials = n_trials
        self.direction = direction
        self.timeout = timeout
        self.metric_name = metric_name
        self.n_classes = n_classes
        self.seed = seed
        self.inner_valid_factory = inner_valid_factory
        self.n_rows = n_rows
        self.fixed_params = fixed_params
        self.progress_callback = progress_callback

    # ------------------------------------------------------------------
    # Private helpers (extracted from objective closure)
    # ------------------------------------------------------------------

    def _resolve_trial_params(
        self,
        model_p: dict[str, Any],
        smart_p: dict[str, Any],
    ) -> tuple[dict[str, Any], Callable[[int], dict[str, Any]] | None]:
        """Merge fixed/model/smart params and build a ratio resolver."""
        merged = {**(self.fixed_params or {}), **model_p}

        # Resolve n_rows-independent smart params (num_leaves)
        if smart_p and self.n_rows is not None:
            effective = {**_COMMON_DEFAULTS, **merged}
            smart_resolved, _ = resolve_smart_params(
                smart=smart_p,
                effective_params=effective,
                n_rows=self.n_rows,
                feature_names=[],
                y=pd.Series(dtype=float),
                task=self.task,
            )
            merged = {**merged, **smart_resolved}

        # Build per-fold ratio resolver for n_rows-dependent params (H-0036)
        leaf_ratio = smart_p.get("min_data_in_leaf_ratio") if smart_p else None
        bin_ratio = smart_p.get("min_data_in_bin_ratio") if smart_p else None
        ratio_resolver: Callable[[int], dict[str, Any]] | None = None
        if leaf_ratio is not None or bin_ratio is not None:
            ratio_resolver = lambda n: resolve_ratio_params(  # noqa: E731
                leaf_ratio, bin_ratio, n
            )
        return merged, ratio_resolver

    def _build_trial_trainer(
        self,
        merged_model: dict[str, Any],
        training_p: dict[str, Any],
        ratio_resolver: Callable[[int], dict[str, Any]] | None,
    ) -> CVTrainer:
        """Build a CVTrainer for a single trial."""
        estimator = self.estimator_factory(merged_model)

        if "early_stopping_rounds" in training_p:
            estimator.early_stopping_rounds = int(training_p["early_stopping_rounds"])

        trial_inner_valid = self.inner_valid
        if "validation_ratio" in training_p and self.inner_valid_factory is not None:
            trial_inner_valid = self.inner_valid_factory(training_p["validation_ratio"])

        return CVTrainer(
            outer_splitter=self.outer_splitter,
            inner_valid=trial_inner_valid,
            pipeline_factory=self.pipeline_factory,
            estimator_factory=lambda: estimator,
            task=self.task,
            n_classes=self.n_classes,
            ratio_param_resolver=ratio_resolver,
        )

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: npt.NDArray[Any] | None = None,
    ) -> TuningResult:
        """Run hyperparameter search and return a TuningResult.

        Args:
            X: Feature DataFrame (already preprocessed by pipeline).
            y: Target Series.
            groups: Optional group array for group-based splitters.

        Returns:
            TuningResult with best params, best score, and trial history.

        Raises:
            LizyMLError with OPTIONAL_DEP_MISSING if optuna is not installed.
            LizyMLError with TUNING_FAILED on study failure.
        """
        if _optuna is None:  # pragma: no cover
            raise LizyMLError(
                code=ErrorCode.OPTIONAL_DEP_MISSING,
                user_message=(
                    "Optuna is required for tuning. "
                    "Install with: pip install 'lizyml[tuning]'"
                ),
                context={"package": "optuna"},
            )

        dims = self.dims
        evaluator = Evaluator(task=self.task)

        # Pre-compute fingerprint and run_meta once — reused across all trials
        fingerprint = fp_compute(X, file_path=None)
        run_meta = RunMeta(
            lizyml_version=__version__,
            python_version=sys.version,
            deps_versions={},
            config_normalized={},
            config_version=1,
            run_id=generate_run_id(),
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

        def objective(trial: Any) -> float:
            trial_params = suggest_params(trial, dims)
            model_p, smart_p, training_p = split_by_category(trial_params, dims)

            merged_model, ratio_resolver = self._resolve_trial_params(model_p, smart_p)
            cv_trainer = self._build_trial_trainer(
                merged_model, training_p, ratio_resolver
            )
            fit_result = cv_trainer.fit(
                X,
                y,
                groups,
                data_fingerprint=fingerprint,
                run_meta=run_meta,
                sample_weight=None,
            )
            metrics = evaluator.evaluate(fit_result, y, [self.metric_name])
            score: float = metrics["raw"]["oof"][self.metric_name]
            return score

        sampler = _optuna.samplers.TPESampler(seed=self.seed)
        study = _optuna.create_study(direction=self.direction, sampler=sampler)

        # Build optuna callback for progress reporting (H-0048)
        optuna_callbacks: list[Any] = []
        if self.progress_callback is not None:
            t0 = time.monotonic()
            user_cb = self.progress_callback
            n_total = self.n_trials

            def _progress_cb(study_: Any, trial: Any) -> None:
                state_name: str = trial.state.name.lower()
                # Only report score for completed trials (spec: H-0048)
                is_complete = trial.state == _optuna.trial.TrialState.COMPLETE
                latest_score = trial.value if is_complete else None
                try:
                    best = study_.best_value
                except ValueError:
                    best = None
                info = TuneProgressInfo(
                    current_trial=trial.number + 1,
                    total_trials=n_total,
                    elapsed_seconds=time.monotonic() - t0,
                    best_score=best,
                    latest_score=latest_score,
                    latest_state=state_name,
                )
                try:
                    user_cb(info)
                except Exception:
                    warnings.warn(
                        "progress_callback raised an exception; ignoring.",
                        RuntimeWarning,
                        stacklevel=1,
                    )

            optuna_callbacks.append(_progress_cb)

        # Add trial failure logging callback
        def _log_trial_failure(study_: Any, trial: Any) -> None:
            if trial.state != _optuna.trial.TrialState.COMPLETE:
                reason = trial.system_attrs.get("fail_reason", "unknown")
                _log.warning(
                    "event='trial.failed' trial=%d state=%s reason=%s",
                    trial.number,
                    trial.state.name,
                    reason,
                )

        all_callbacks = [*optuna_callbacks, _log_trial_failure]

        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False,
                catch=(LizyMLError, ValueError, RuntimeError),
                callbacks=all_callbacks,
            )
        except Exception as exc:
            raise LizyMLError(
                code=ErrorCode.TUNING_FAILED,
                user_message=f"Optuna study failed: {exc}",
                context={"n_trials": self.n_trials},
                cause=exc,
            ) from exc

        completed = [
            t for t in study.trials if t.state == _optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            raise LizyMLError(
                code=ErrorCode.TUNING_FAILED,
                user_message="All tuning trials failed. Check parameter ranges.",
                context={"n_trials": self.n_trials},
            )

        _log.info(
            "event='tune.done' best_value=%.4f n_trials=%d",
            study.best_value,
            len(study.trials),
        )

        trials = [
            TrialResult(
                number=t.number,
                params=dict(t.params),
                score=t.value if t.value is not None else float("nan"),
                state=t.state.name.lower(),
            )
            for t in study.trials
        ]
        best_flat = dict(study.best_params)
        best_model, best_smart, best_training = split_by_category(best_flat, dims)
        return TuningResult(
            best_model_params=best_model,
            best_smart_params=best_smart,
            best_training_params=best_training,
            best_score=study.best_value,
            trials=trials,
            metric_name=self.metric_name,
            direction=self.direction,
        )
