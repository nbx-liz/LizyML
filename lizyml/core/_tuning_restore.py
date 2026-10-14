"""Take a tune's result back from a stored study (H-0122 §7.7).

A round whose ``:end`` was written leaves its study in state C even when the
process died before the caller received the result. ``Model.restore_tuning``
returns that result without adding a trial; this module holds its pure parts:

* :func:`refuse_unrestorable` -- the §7.7 state rules, in the order of design
  R-6: nothing to restore (``TUNING_FAILED``) wins over a study that would be
  refused anyway.
* :func:`restore_from_study` -- the result and the Model fields, assembled from
  the lineage exactly as the closing tune assembled them, so the restored
  result equals the one the tune returned: the same function builds it, and
  the boundary report is the last round's record, not a recomputation.

Nothing here writes to the study or to a Model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from lizyml.core._smart_values import smart_value_violation
from lizyml.core._tuning_plan import _uses_default
from lizyml.core._tuning_resume import judge_existing, objective_identity, refuse_state
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import SearchDim
from lizyml.core.types.tuning_result import TuningResult
from lizyml.tuning import tuner as _tuner
from lizyml.tuning.lineage import Lineage, classify_study, split_by_routes
from lizyml.tuning.rounds import (
    ScoredTrial,
    assemble_tuning_result,
    recorded_trial_params,
)

if TYPE_CHECKING:
    from optuna.storages import BaseStorage

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core.types.task import TaskType
    from lizyml.estimators.provider import EstimatorProvider

#: States with no closed, completed round to restore (§7.7).
_NOTHING_TO_RESTORE = frozenset({"uninitialized", "no_round", "failed"})


@dataclass(frozen=True)
class Restored:
    """What ``restore_tuning`` commits to the Model (§7.7)."""

    result: TuningResult
    round_number: int
    effective: tuple[SearchDim, ...]
    used_default: bool
    fixed: Mapping[str, Any]
    study: Any = None


def _nothing_to_restore(state: str, study_name: str, detail: str) -> LizyMLError:
    return LizyMLError(
        code=ErrorCode.TUNING_FAILED,
        user_message=(
            f"study '{study_name}' has no tuning result to restore: {detail} "
            "restore_tuning returns the result of a round that closed "
            "completed."
        ),
        context={"surface": "tuning.study", "state": state, "study_name": study_name},
    )


def refuse_unrestorable(
    lineage: Lineage | None, study_name: str, n_trials: int
) -> None:
    """Refuse every state but C, in the design R-6 order.

    ``None`` (the storage has no such study) and the states E, N and F raise
    ``TUNING_FAILED``; L, S and X raise ``CONFIG_INVALID`` as for
    ``resume=True``; O raises ``CONFIG_INVALID``: its round is finished with
    ``tune(resume=True)``, and a part-finished round's result is not restored.
    """
    if lineage is None:
        raise _nothing_to_restore(
            "absent", study_name, "the storage holds no study of that name."
        )
    refuse_state(lineage, study_name, n_trials)
    if lineage.state == "open":
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"round {lineage.open_round} of study '{study_name}' is open: it "
                "was interrupted. Finish it with tune(resume=True); a part-"
                "finished round's result is not restored."
            ),
            context={
                "surface": "tuning.study",
                "state": "open",
                "round": lineage.open_round,
                "study_name": study_name,
            },
        )
    if lineage.state in _NOTHING_TO_RESTORE:
        last = lineage.last_round.round if lineage.last_round is not None else "?"
        detail = {
            "uninitialized": "it was created but never initialised by LizyML.",
            "no_round": "it has no round yet.",
            # H-0128: the last round may follow completed rounds.
            "failed": (
                f"its last round ({last}) closed failed: no trial of that round "
                "completed. The results of earlier rounds are not restored; "
                "start a new round with tune(resume=True) and a corrected space=."
            ),
        }[lineage.state]
        raise _nothing_to_restore(lineage.state, study_name, detail)


def restore_from_study(
    lineage: Lineage,
    trials: Sequence[ScoredTrial],
    best_params: Mapping[str, Any],
    best_value: float,
    *,
    metric_name: str,
    direction: str,
) -> Restored:
    """The result and Model fields of a study in state C (pure).

    The result is built by the function the closing tune used, with its last
    round as the current round, so it carries that round's recorded boundary
    report (and threshold) rather than a recomputed one.

    Raises:
        LizyMLError: ``TUNING_FAILED`` when *lineage* is not ``completed``;
            :func:`refuse_unrestorable` refuses those first, so this is an
            internal error, never a silent result.
    """
    last = lineage.last_round
    if lineage.state != "completed" or last is None or lineage.study is None:
        raise LizyMLError(
            code=ErrorCode.TUNING_FAILED,
            user_message=(
                f"Internal error: a study in state '{lineage.state}' reached "
                "restore_from_study, which needs a completed round."
            ),
            context={"state": lineage.state},
        )
    result = assemble_tuning_result(
        lineage,
        trials,
        best_params,
        best_value,
        metric_name=metric_name,
        direction=direction,
        current_round=last.round,
    )
    return Restored(
        result=result,
        round_number=last.round,
        effective=tuple(last.effective),
        used_default=_uses_default(lineage.study.space_mode, last.provenance),
        fixed=dict(lineage.study.fixed),
    )


def read_restorable(
    cfg: LizyMLConfig,
    storage: str | BaseStorage,
    study_name: str,
    *,
    data_summary: Mapping[str, str],
    provider: EstimatorProvider,
    direction: Literal["minimize", "maximize"],
    metric_name: str,
) -> Restored:
    """Load the stored study, judge it, and restore its last round (§7.7).

    Reads only: loading writes nothing to the study, and a RUNNING trial stays
    RUNNING. In order: the state (:func:`refuse_unrestorable`), then for a
    ``completed`` study its direction, config space, objective identity and
    versions, which only warn (``judge_existing``, as for ``resume=True``).
    """
    study = _tuner.load_stored_study(storage, study_name, cfg.training.seed)
    trials: list[Any] = list(study.trials) if study is not None else []
    lineage = classify_study(study.user_attrs, trials) if study is not None else None
    refuse_unrestorable(lineage, study_name, len(trials))
    assert study is not None and lineage is not None  # noqa: S101 -- refused above
    identity, versions, _fixed = objective_identity(
        cfg, data_summary, provider, direction
    )
    judge_existing(
        study,
        lineage,
        cfg,
        identity=identity,
        versions=versions,
        direction=direction,
        study_name=study_name,
    )
    refuse_study_smart_values(lineage, trials, provider, cfg.task, study_name)
    best = study.best_trial
    restored = restore_from_study(
        lineage,
        trials,
        recorded_trial_params(lineage, best),
        best.value,
        metric_name=metric_name,
        direction=direction,
    )
    refuse_recorded_smart_values(restored.result, provider, cfg.task, study_name)
    return replace(restored, study=study)


def refuse_recorded_smart_values(
    result: TuningResult,
    provider: EstimatorProvider,
    task: TaskType,
    study_name: str,
) -> None:
    """Refuse a restored result whose smart values break the rule (H-0121 am. 2).

    Policy 5: a study tuned before the smart value rule existed can record a
    value no trial trained as written (``"false"`` trained with the parameter
    on). ``tune(resume=True)`` refuses such a study before writing
    (:func:`refuse_study_smart_values`); the result is refused here, before
    it reaches the Model or a ``fit``.
    """
    _refuse_smart_violation(
        smart_value_violation(provider, result.best_smart_params, task), study_name
    )


def refuse_study_smart_values(
    lineage: Lineage | None,
    trials: Sequence[Any],
    provider: EstimatorProvider,
    task: TaskType,
    study_name: str | None,
) -> None:
    """Refuse a stored study any of whose trials breaks the smart value rule.

    H-0121 amendment 2, policies 2 and 5: the round admission checks the new
    round's dimensions, but a stored study's earlier rounds can hold trials
    tuned before the rule existed (``"false"`` trained with the parameter on).
    Each one reaches the tune's result -- the best, the ``fit`` overlay and
    the export when it is the best, ``TuningResult.trials`` and the tables in
    any case -- even after a ``tune(space=...)`` removed its dimension. So
    every trial's recorded smart values (its recorded values split by the
    lineage routes) are checked, before any study write on resume and before
    a ``restore_tuning`` result reaches the Model.
    """
    if lineage is None:
        return
    for trial in trials:
        if not trial.params:
            continue
        _, smart, _ = split_by_routes(
            recorded_trial_params(lineage, trial), lineage.routes
        )
        found = smart_value_violation(provider, smart, task)
        if found is not None:
            _refuse_smart_violation(found, study_name, trial_number=trial.number)


def _refuse_smart_violation(
    found: Any, study_name: str | None, *, trial_number: int | None = None
) -> None:
    if found is None:
        return
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            f"study '{study_name}' records the smart parameter "
            f"'{found.name}'={found.value!r}"
            + (
                f" in trial {trial_number}"
                if trial_number is not None
                else " as its best"
            )
            + f", which '{found.name}' "
            f"cannot hold: it takes {found.expected}. Its trials did not train "
            "on that value as written, so the result is not restored. Tune "
            "again in a new study (another study_name)."
        ),
        context={
            "surface": "tuning.study",
            "study_name": study_name,
            "name": found.name,
            "value": found.value,
            "reason": found.reason,
        },
    )
