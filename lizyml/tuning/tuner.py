"""Tuner — optuna-backed hyperparameter optimisation."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from lizyml import __version__
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import generate_run_id, get_logger
from lizyml.core.types.artifacts import RunMeta
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.evaluation.evaluator import Evaluator
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.inner_valid import BaseInnerValidStrategy
from lizyml.tuning.search_space import SearchDim, suggest_params

if TYPE_CHECKING:
    from lizyml.estimators.lgbm import LGBMAdapter
    from lizyml.features.pipelines_native import NativeFeaturePipeline
    from lizyml.splitters.base import BaseSplitter

_optuna: Any = None
try:
    import optuna  # type: ignore[import-not-found]

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
        inner_valid: Inner validation strategy.
        pipeline_factory: Factory producing NativeFeaturePipeline instances.
        estimator_factory: Factory accepting a trial params dict → LGBMAdapter.
        dims: Parsed search space dimensions.
        n_trials: Number of optuna trials.
        direction: Optimisation direction (minimize or maximize).
        timeout: Optional timeout in seconds.
        metric_name: Metric to optimise (OOF score).
        n_classes: Number of classes for multiclass tasks.
        seed: Random seed for the optuna sampler.
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

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run hyperparameter search and return best params.

        Args:
            X: Feature DataFrame (already preprocessed by pipeline).
            y: Target Series.
            groups: Optional group array for group-based splitters.

        Returns:
            Dict of best hyperparameter values found by optuna.

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
            estimator = self.estimator_factory(trial_params)
            cv_trainer = CVTrainer(
                outer_splitter=self.outer_splitter,
                inner_valid=self.inner_valid,
                pipeline_factory=self.pipeline_factory,
                estimator_factory=lambda: estimator,
                task=self.task,
                n_classes=self.n_classes,
            )
            fit_result = cv_trainer.fit(
                X, y, groups, data_fingerprint=fingerprint, run_meta=run_meta
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
            )
        except Exception as exc:
            raise LizyMLError(
                code=ErrorCode.TUNING_FAILED,
                user_message=f"Optuna study failed: {exc}",
                context={"n_trials": self.n_trials},
                cause=exc,
            ) from exc

        _log.info(
            "event='tune.done' best_value=%.4f n_trials=%d",
            study.best_value,
            len(study.trials),
        )
        return dict(study.best_params)
