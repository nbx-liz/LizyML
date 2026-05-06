"""Tuner — optuna-backed hyperparameter optimisation (H-0050, H-0068).

The Tuner is responsible for:
- Creating and running an Optuna study
- Managing progress callbacks and failure logging
- Collecting trial results into a TuningResult
- Accepting an existing study for resume (H-0068)

The Tuner does NOT know about LightGBM, smart params, CVTrainer, or any
model-specific logic.  The objective closure is built externally (by Model)
and passed into ``tune()``.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Literal

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import get_logger
from lizyml.core.types.tuning_result import (
    TrialResult,
    TuneProgressCallback,
    TuneProgressInfo,
    TuningResult,
)
from lizyml.tuning.search_space import SearchDim, split_by_category

_optuna: Any = None
try:
    import optuna

    _optuna = optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
except ImportError:  # pragma: no cover
    pass

_log = get_logger("tuner")


class Tuner:
    """Optuna-backed hyperparameter tuner (study management only).

    Args:
        dims: Parsed search space dimensions.
        n_trials: Number of optuna trials.
        direction: Optimisation direction (minimize or maximize).
        timeout: Optional timeout in seconds.
        seed: Random seed for the optuna sampler.
        progress_callback: Optional callback invoked after each trial.
        storage: Optional Optuna storage URL (e.g. ``sqlite:///path/to.db``)
            or ``BaseStorage`` instance for resumable tuning (H-0072).
            ``None`` (default) keeps the in-memory study behavior. When set,
            ``study_name`` is required.
        study_name: Optional study identifier used together with ``storage``
            (H-0072). Required when ``storage`` is given. Re-using the same
            ``(storage, study_name)`` pair re-attaches to an existing study
            via ``load_if_exists=True`` so completed trials are not re-run.
    """

    def __init__(
        self,
        dims: list[SearchDim],
        n_trials: int = 50,
        direction: Literal["minimize", "maximize"] = "minimize",
        timeout: float | None = None,
        seed: int = 42,
        *,
        progress_callback: TuneProgressCallback | None = None,
        storage: str | Any | None = None,
        study_name: str | None = None,
    ) -> None:
        if storage is not None and study_name is None:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    "study_name is required when storage is provided. "
                    "Pass a unique identifier so the study can be re-attached."
                ),
                context={"storage": str(storage)},
            )
        self.dims = dims
        self.n_trials = n_trials
        self.direction = direction
        self.timeout = timeout
        self.seed = seed
        self.progress_callback = progress_callback
        self.storage = storage
        self.study_name = study_name

    def tune(
        self,
        objective: Any,
        metric_name: str = "rmse",
        *,
        study: Any | None = None,
        enqueue_params: dict[str, Any] | None = None,
        round_number: int = 1,
        prior_trials: int = 0,
        expanded_dims: tuple[str, ...] = (),
    ) -> tuple[TuningResult, Any]:
        """Run hyperparameter search and return a TuningResult + study.

        Args:
            objective: Callable ``(optuna.Trial) -> float`` that evaluates a
                single trial.  Built externally by Model.tune().
            metric_name: Name of the metric being optimised (stored in
                TuningResult for downstream display/logging).
            study: Optional existing Optuna study to resume (H-0068).
                If None, a new study is created.
            enqueue_params: Optional params dict to enqueue as initial trial
                (H-0068). Typically the previous best params.
            round_number: Current round number (1-indexed, H-0068).
            prior_trials: Number of trials from previous rounds (H-0068).
            expanded_dims: Dimensions expanded in this round (H-0068).

        Returns:
            Tuple of (TuningResult, study). The study is returned so that
            Model can hold it for future resume calls.

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

        if study is None:
            sampler = _optuna.samplers.TPESampler(seed=self.seed)
            if self.storage is None:
                study = _optuna.create_study(direction=self.direction, sampler=sampler)
            else:
                study = _optuna.create_study(
                    direction=self.direction,
                    sampler=sampler,
                    storage=self.storage,
                    study_name=self.study_name,
                    load_if_exists=True,
                )

        # Enqueue previous best as initial trial (H-0068)
        if enqueue_params is not None:
            study.enqueue_trial(enqueue_params)

        # Build optuna callbacks for progress reporting (H-0048, H-0068)
        optuna_callbacks: list[Any] = []

        if self.progress_callback is not None:
            t0 = time.monotonic()
            user_cb = self.progress_callback
            n_total = self.n_trials
            rnd = round_number
            exp_dims = expanded_dims

            def _progress_cb(study_: Any, trial: Any) -> None:
                state_name: str = trial.state.name.lower()
                is_complete = trial.state == _optuna.trial.TrialState.COMPLETE
                latest_score = trial.value if is_complete else None
                try:
                    best = study_.best_value
                except ValueError:
                    best = None
                # trial.number is 0-indexed across the whole study
                current_in_round = trial.number - prior_trials + 1
                info = TuneProgressInfo(
                    current_trial=current_in_round,
                    total_trials=n_total,
                    elapsed_seconds=time.monotonic() - t0,
                    best_score=best,
                    latest_score=latest_score,
                    latest_state=state_name,
                    round=rnd,
                    cumulative_trials=trial.number + 1,
                    expanded_dims=exp_dims,
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

        # Trial failure logging callback
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
            "event='tune.done' best_value=%.4f n_trials=%d round=%d",
            study.best_value,
            len(study.trials),
            round_number,
        )

        dims = self.dims
        trials = [
            TrialResult(
                number=t.number,
                params=dict(t.params),
                score=t.value if t.value is not None else float("nan"),
                state=t.state.name.lower(),
                # Prior-round trials get sentinel round=1; Model.tune() reassigns
                # correct round numbers via dataclasses.replace().
                round=round_number if t.number >= prior_trials else 1,
            )
            for t in study.trials
        ]
        best_flat = dict(study.best_params)
        best_model, best_smart, best_training = split_by_category(best_flat, dims)
        result = TuningResult(
            best_model_params=best_model,
            best_smart_params=best_smart,
            best_training_params=best_training,
            best_score=study.best_value,
            trials=trials,
            metric_name=metric_name,
            direction=self.direction,
        )
        return result, study
