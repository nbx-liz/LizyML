"""Tuner — optuna-backed hyperparameter optimisation."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy.typing as npt
import pandas as pd

from lizyml import __version__
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import generate_run_id, get_logger
from lizyml.core.types.artifacts import RunMeta
from lizyml.core.types.tuning_result import TrialResult, TuningResult
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.estimators.lgbm import (
    _COMMON_DEFAULTS,
    resolve_ratio_params,
    resolve_smart_params_from_dict,
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
            timestamp=datetime.now().isoformat(),
        )

        def objective(trial: Any) -> float:
            trial_params = suggest_params(trial, dims)
            model_p, smart_p, training_p = split_by_category(trial_params, dims)

            # Apply fixed params as base, then trial's model params on top
            merged_model = {**(self.fixed_params or {}), **model_p}

            # Resolve n_rows-independent smart params (num_leaves only)
            sample_weight: npt.NDArray[Any] | None = None
            if smart_p and self.n_rows is not None:
                effective = {**_COMMON_DEFAULTS, **merged_model}
                smart_resolved = resolve_smart_params_from_dict(
                    smart_params=smart_p,
                    effective_params=effective,
                    n_rows=self.n_rows,
                )
                merged_model.update(smart_resolved)

            # Build per-fold ratio resolver for n_rows-dependent params (H-0036)
            leaf_ratio = smart_p.get("min_data_in_leaf_ratio") if smart_p else None
            bin_ratio = smart_p.get("min_data_in_bin_ratio") if smart_p else None
            trial_ratio_resolver: Callable[[int], dict[str, Any]] | None = None
            if leaf_ratio is not None or bin_ratio is not None:

                def _make_resolver(
                    lr: float | None, br: float | None
                ) -> Callable[[int], dict[str, Any]]:
                    return lambda n: resolve_ratio_params(lr, br, n)

                trial_ratio_resolver = _make_resolver(leaf_ratio, bin_ratio)

            # Build estimator with merged model params
            estimator = self.estimator_factory(merged_model)

            # Handle training params
            if "early_stopping_rounds" in training_p:
                estimator.early_stopping_rounds = int(
                    training_p["early_stopping_rounds"]
                )

            # Resolve inner_valid for this trial
            trial_inner_valid = self.inner_valid
            if (
                "validation_ratio" in training_p
                and self.inner_valid_factory is not None
            ):
                trial_inner_valid = self.inner_valid_factory(
                    training_p["validation_ratio"]
                )

            cv_trainer = CVTrainer(
                outer_splitter=self.outer_splitter,
                inner_valid=trial_inner_valid,
                pipeline_factory=self.pipeline_factory,
                estimator_factory=lambda: estimator,
                task=self.task,
                n_classes=self.n_classes,
                ratio_param_resolver=trial_ratio_resolver,
            )
            fit_result = cv_trainer.fit(
                X,
                y,
                groups,
                data_fingerprint=fingerprint,
                run_meta=run_meta,
                sample_weight=sample_weight,
            )
            metrics = evaluator.evaluate(fit_result, y, [self.metric_name])
            score: float = metrics["raw"]["oof"][self.metric_name]
            return score

        sampler = _optuna.samplers.TPESampler(seed=self.seed)
        study = _optuna.create_study(direction=self.direction, sampler=sampler)
        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False,
                catch=(Exception,),
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
        return TuningResult(
            best_params=dict(study.best_params),
            best_score=study.best_value,
            trials=trials,
            metric_name=self.metric_name,
            direction=self.direction,
        )
