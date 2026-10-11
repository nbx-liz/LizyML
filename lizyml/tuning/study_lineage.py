"""Read a stored study's state and lineage without adding to it (H-0122 §7.6).

:func:`read_study_lineage` is the public, read-only view of the records
``Model.tune`` keeps in a study's ``lizyml:`` attributes: which §7.4 state
the study is in, and one :class:`RoundRecord` per recorded round. It is what a
caller asks before deciding between ``tune(resume=True)``,
``Model.restore_tuning`` and a new study.

It writes nothing to the study: no attribute, and a RUNNING trial left by a
dead process stays RUNNING (only ``tune(resume=True)`` FAILs it, §7.2). It
does not check the objective identity either; it takes no data and no config.
The one side effect on the storage is Optuna's: listing a SQLite storage's
studies creates the file and its schema when they do not exist yet (H-0122
design D-c). No study is created.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.tuning import tuner as _tuner
from lizyml.tuning.lineage import Lineage, TrialLike, classify_study, trials_of_round

if TYPE_CHECKING:
    from optuna.storages import BaseStorage

__all__ = [
    "RoundRecord",
    "StudyLineage",
    "StudyState",
    "lineage_view",
    "read_study_lineage",
    "require_stored_study_args",
]

StudyState = Literal[
    "absent",
    "legacy",
    "uninitialized",
    "unreadable",
    "no_round",
    "inconsistent",
    "open",
    "completed",
    "failed",
]
"""The §7.4 states, plus ``absent`` for a storage without the study."""

RoundStatus = Literal["completed", "failed", "open"]

#: Trials that have finished: a RUNNING or WAITING trial has not.
_FINISHED = frozenset({"COMPLETE", "FAIL", "PRUNED"})


@dataclass(frozen=True)
class RoundRecord:
    """One recorded round of a study (``lizyml:round:<k>`` and its end).

    Attributes:
        round: The round number, from 1.
        status: ``"completed"`` or ``"failed"`` from the round's end record,
            ``"open"`` when the round has none (it was interrupted).
        n_trials: The trial budget the round asked for.
        n_finished: The round's COMPLETE, FAIL and PRUNED trials. A RUNNING
            trial is not counted.
        best_score_before: The study's best value when the round started,
            ``None`` when the study had no COMPLETE trial.
        best_score_after: The study's best value when the round closed;
            ``None`` for a ``failed`` or ``open`` round.
    """

    round: int
    status: RoundStatus
    n_trials: int
    n_finished: int
    best_score_before: float | None
    best_score_after: float | None


@dataclass(frozen=True)
class StudyLineage:
    """A study's state and recorded rounds (H-0122 §7.6).

    Attributes:
        state: The study's §7.4 state, or ``"absent"``.
        rounds: One :class:`RoundRecord` per recorded round, ``failed``
            rounds included. Empty unless ``state`` is ``"open"``,
            ``"completed"`` or ``"failed"``.
        open_round: The open round's number in state ``"open"``, else
            ``None``.
    """

    state: StudyState
    rounds: tuple[RoundRecord, ...] = ()
    open_round: int | None = None


def lineage_view(lineage: Lineage | None, trials: Sequence[TrialLike]) -> StudyLineage:
    """The public view of a classified study; ``None`` means ``absent``.

    Pure: reads *lineage* and *trials* only.
    """
    if lineage is None:
        return StudyLineage(state="absent")
    records = []
    for record, end in zip(lineage.rounds, lineage.ends, strict=True):
        finished = sum(
            1
            for t in trials_of_round(lineage, record.round, trials)
            if t.state.name in _FINISHED
        )
        records.append(
            RoundRecord(
                round=record.round,
                status="open" if end is None else end.status,
                n_trials=record.n_trials,
                n_finished=finished,
                best_score_before=record.best_before,
                best_score_after=end.best_after if end is not None else None,
            )
        )
    return StudyLineage(
        state=lineage.state, rounds=tuple(records), open_round=lineage.open_round
    )


def require_stored_study_args(
    storage: str | BaseStorage | None, study_name: str | None, *, caller: str
) -> None:
    """The §3 precheck, plus: a caller that reads a stored study needs both.

    ``storage`` without ``study_name`` and the reverse are refused as in
    ``Model.tune``; neither given is refused too, since there is no study to
    read (H-0122 design R-5).
    """
    _tuner.precheck_study_args(storage, study_name)
    if storage is None:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"{caller} reads a stored study: pass storage and study_name."
            ),
            context={
                "surface": f"{caller}(storage=, study_name=)",
                "storage_given": False,
            },
        )


def read_study_lineage(storage: str | BaseStorage, study_name: str) -> StudyLineage:
    """Read a stored study's state and rounds without changing it (§7.6).

    Args:
        storage: An Optuna storage URL or ``BaseStorage``, as for
            ``Model.tune``.
        study_name: The study's name in *storage*.

    Returns:
        The study's :class:`StudyLineage`; ``state="absent"`` when *storage*
        holds no study of that name.

    Raises:
        LizyMLError: ``CONFIG_INVALID`` when ``storage`` or ``study_name`` is
            missing; ``OPTIONAL_DEP_MISSING`` without optuna.
    """
    require_stored_study_args(storage, study_name, caller="read_study_lineage")
    _tuner.require_optuna()
    study = _tuner.load_stored_study(storage, study_name, seed=0)
    if study is None:
        return lineage_view(None, ())
    trials = study.trials
    return lineage_view(classify_study(study.user_attrs, trials), trials)
