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
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import get_logger

if TYPE_CHECKING:
    from optuna.storages import BaseStorage
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


def _require_optuna() -> Any:
    if _optuna is None:  # pragma: no cover
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                "Optuna is required for tuning. "
                "Install with: pip install 'lizyml[tuning]'"
            ),
            context={"package": "optuna"},
        )
    return _optuna


def require_optuna() -> Any:
    """Return the ``optuna`` module, or raise ``OPTIONAL_DEP_MISSING``."""
    return _require_optuna()


def precheck_study_args(
    storage: str | BaseStorage | None, study_name: str | None
) -> None:
    """Refuse ``storage`` without ``study_name`` and the reverse (H-0122 §3).

    Shared by ``Model.tune`` and the lineage reader, so it lives with the
    storage helpers in Layer 2.

    Checked first, whatever ``resume`` is: before this, a ``resume=True`` on a
    Model without a study raised ``TUNING_FAILED`` for the first, and the
    second was ignored without notice.
    """
    if storage is not None and study_name is None:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                "study_name is required when storage is provided. "
                "Pass a unique identifier so the study can be re-attached."
            ),
            context={"surface": "tune(storage=)", "study_name_given": False},
        )
    if study_name is not None and storage is None:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"study_name={study_name!r} names a study in a storage, but no "
                "storage was given. Pass storage as well, or drop study_name for "
                "an in-memory study."
            ),
            context={
                "surface": "tune(study_name=)",
                "study_name": study_name,
                "storage_given": False,
            },
        )


def stored_study_exists(storage: str | BaseStorage, study_name: str | None) -> bool:
    """Whether *storage* already holds a study named *study_name* (H-0121).

    Reads only. Asked before the study is created; a second process creating
    the same name in between is not considered (one writer per study).
    """
    if study_name is None:
        return False
    names: list[str] = _require_optuna().get_all_study_names(storage)
    return study_name in names


# ---------------------------------------------------------------------------
# Study I/O (H-0122). The Model decides what to write; these only do it.
# ---------------------------------------------------------------------------


def load_stored_study(
    storage: str | BaseStorage, study_name: str, seed: int
) -> Any | None:
    """Return the stored study named *study_name*, or ``None`` when absent.

    Loading writes nothing to the study (Optuna 4.7, verified). The sampler is
    a fresh ``TPESampler(seed)``, as for a created study.
    """
    optuna_ = _require_optuna()
    if not stored_study_exists(storage, study_name):
        return None
    return optuna_.load_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna_.samplers.TPESampler(seed=seed),
    )


def create_study(
    direction: Literal["minimize", "maximize"],
    seed: int,
    storage: str | BaseStorage | None = None,
    study_name: str | None = None,
) -> Any:
    """Create a new study; never reattaches (``load_if_exists=False``).

    Calls ``optuna.create_study`` through the module at call time, so a test
    patching ``optuna.create_study`` sees every creation.
    """
    optuna_ = _require_optuna()
    sampler = optuna_.samplers.TPESampler(seed=seed)
    if storage is None:
        return optuna_.create_study(direction=direction, sampler=sampler)
    return optuna_.create_study(
        direction=direction,
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=False,
    )


def write_study_attr(study: Any, key: str, value: Any) -> None:
    """Write one study attribute. Every ``lizyml:`` write goes through here.

    Callers look this up on the module at call time
    (``lizyml.tuning.tuner.write_study_attr``), so it is the single place a
    test can stop a tune between two write positions (H-0122 §7.1).
    """
    study.set_user_attr(key, value)


def fail_running_trials(study: Any, numbers: Iterable[int]) -> None:
    """Mark the trials a dead process left RUNNING as FAIL (H-0122 §7.2)."""
    optuna_ = _require_optuna()
    for number in numbers:
        study.tell(number, state=optuna_.trial.TrialState.FAIL)


def check_study_direction(
    study: Any, direction: Literal["minimize", "maximize"]
) -> None:
    """Refuse a study whose direction is not the objective's (H-0099 decision 2).

    A persisted name or an explicitly supplied study can already carry another
    direction. Never enqueue or evaluate against that objective.
    """
    if len(study.directions) != 1 or study.directions[0].name.lower() != direction:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                "tuning study direction disagrees with the objective. "
                "Use a new study with the correct direction."
            ),
            context={
                "surface": "tuning.study",
                "expected_direction": direction,
                "study_directions": [item.name.lower() for item in study.directions],
            },
        )


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
        storage: str | BaseStorage | None = None,
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

        For direct callers. ``Model.tune`` does not call this: it creates or
        loads the study itself, writes the lineage records, and calls
        :meth:`run` (H-0122).

        Args:
            objective: Callable ``(optuna.Trial) -> float`` that evaluates a
                single trial.
            metric_name: Name of the metric being optimised (stored in
                TuningResult for downstream display/logging).
            study: Optional existing Optuna study to resume (H-0068).
                If None, a new study is created, or a stored one of the same
                ``(storage, study_name)`` re-attached (``load_if_exists``).
            enqueue_params: Optional params dict to enqueue as initial trial
                (H-0068).
            round_number: Current round number (1-indexed, H-0068).
            prior_trials: Number of trials from previous rounds (H-0068).
            expanded_dims: Dimensions expanded in this round (H-0068).

        Returns:
            Tuple of (TuningResult, study).

        Raises:
            LizyMLError with OPTIONAL_DEP_MISSING if optuna is not installed.
            LizyMLError with TUNING_FAILED on study failure.
        """
        optuna_ = _require_optuna()
        if study is None:
            sampler = optuna_.samplers.TPESampler(seed=self.seed)
            if self.storage is None:
                study = optuna_.create_study(direction=self.direction, sampler=sampler)
            else:
                study = optuna_.create_study(
                    direction=self.direction,
                    sampler=sampler,
                    storage=self.storage,
                    study_name=self.study_name,
                    load_if_exists=True,
                )

        check_study_direction(study, self.direction)

        # Enqueue previous best as initial trial (H-0068).
        if enqueue_params is not None:
            study.enqueue_trial(enqueue_params)

        self.run(
            study,
            objective,
            round_number=round_number,
            round_start=prior_trials,
            expanded_dims=expanded_dims,
        )

        completed = [
            t for t in study.trials if t.state == optuna_.trial.TrialState.COMPLETE
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
                # Prior-round trials get sentinel round=1.
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

    def run(
        self,
        study: Any,
        objective: Any,
        *,
        round_number: int,
        round_start: int,
        progress_total: int | None = None,
        expanded_dims: tuple[str, ...] = (),
    ) -> None:
        """Run ``optimize`` on *study* with this tuner's budget and callbacks.

        Only the optimize step: no study creation, no attribute, no result.

        Args:
            study: The study to add trials to.
            objective: Callable ``(optuna.Trial) -> float``.
            round_number: The round the trials belong to (progress ``round``).
            round_start: Number of the round's first trial, so progress
                ``current_trial`` is ``trial.number - round_start + 1``. A
                round finished on reattach passes its recorded start, so the
                count continues at K+1 (H-0122 §6.5).
            progress_total: Progress ``total_trials``; the round's requested
                ``n_trials``. Defaults to ``self.n_trials``.
            expanded_dims: Dimensions expanded for this round.

        Raises:
            LizyMLError with TUNING_FAILED when ``optimize`` raises an
            ``Exception``. A ``BaseException`` propagates unchanged.
        """
        optuna_ = _require_optuna()
        optuna_callbacks: list[Any] = []

        if self.progress_callback is not None:
            t0 = time.monotonic()
            user_cb = self.progress_callback
            n_total = self.n_trials if progress_total is None else progress_total
            rnd = round_number
            exp_dims = expanded_dims

            def _progress_cb(study_: Any, trial: Any) -> None:
                state_name: str = trial.state.name.lower()
                is_complete = trial.state == optuna_.trial.TrialState.COMPLETE
                latest_score = trial.value if is_complete else None
                try:
                    best = study_.best_value
                except ValueError:
                    best = None
                # trial.number is 0-indexed across the whole study
                current_in_round = trial.number - round_start + 1
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
                except Exception as exc:
                    # stacklevel=1 (this site) is intentional: by the time
                    # ``warnings.warn`` runs, ``user_cb`` has already raised
                    # and unwound, so its frame is not on the stack. The
                    # exception type+message in the warning text is the
                    # actionable signal for the user (#117).
                    warnings.warn(
                        f"progress_callback raised an exception; ignoring. "
                        f"{type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=1,
                    )

            optuna_callbacks.append(_progress_cb)

        # Trial failure logging callback
        def _log_trial_failure(study_: Any, trial: Any) -> None:
            if trial.state != optuna_.trial.TrialState.COMPLETE:
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
