"""Assemble a ``TuningResult`` from a study's lineage (H-0068, #209, H-0122).

One function serves a round continued in-process and a round finished after a
reattach: both read the lineage the study carries, so the result's rounds,
trial rounds and best-parameter routing cannot depend on how the Model got
there (INV-R5). Pure: no ``Model`` state and no Optuna import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from lizyml.core.types.tuning_result import RoundSummary, TrialResult, TuningResult
from lizyml.tuning._step_grid import recorded_params
from lizyml.tuning.lineage import (
    Lineage,
    TrialLike,
    round_of,
    split_by_routes,
    trials_of_round,
)


class ScoredTrial(TrialLike, Protocol):
    """A trial with its objective value (``optuna.trial.FrozenTrial``)."""

    @property
    def value(self) -> float | None: ...


def recorded_trial_params(lineage: Lineage, trial: TrialLike) -> dict[str, Any]:
    """The values *trial* trained on, read with its own round's dims (H-0124).

    The one reader of a trial's raw Optuna values for anything LizyML reports,
    plans or trains from (policy 4): a stepped dimension's value is the grid
    point its raw sample rounds to under the ``effective`` space of the round
    the trial belongs to (:func:`round_of`), never the current round's. A
    dimension without ``step`` reads as the raw value.

    Raises:
        ValueError: The trial precedes every round, or holds a name its round
            does not declare; ``classify_study`` judges such a study
            ``inconsistent`` first.
    """
    by_round = {record.round: record for record in lineage.rounds}
    own = by_round[round_of(lineage.rounds, trial.number)]
    return recorded_params(trial.params, own.effective)


def round_summaries(
    lineage: Lineage, trials: Sequence[TrialLike]
) -> tuple[RoundSummary, ...]:
    """One ``RoundSummary`` per ``completed`` round, in round order.

    ``failed`` rounds are omitted (§7.2): ``RoundSummary`` holds a numeric best,
    which a failed round has not. ``n_trials`` is the round's actual trial
    count in every state (criterion 31), not the requested budget.
    """
    summaries: list[RoundSummary] = []
    for record, end in zip(lineage.rounds, lineage.ends, strict=True):
        if end is None or end.status != "completed":
            continue
        assert end.best_after is not None  # noqa: S101 -- decoder guarantees
        report = record.boundary_report
        summaries.append(
            RoundSummary(
                round=record.round,
                n_trials=len(trials_of_round(lineage, record.round, trials)),
                best_score_before=record.best_before,
                best_score_after=end.best_after,
                expanded_dims=report.expanded_names if report is not None else (),
                space_snapshot=tuple(record.effective),
            )
        )
    return tuple(summaries)


def assemble_tuning_result(
    lineage: Lineage,
    trials: Sequence[ScoredTrial],
    best_params: Mapping[str, Any],
    best_score: float,
    *,
    metric_name: str,
    direction: str,
    current_round: int,
) -> TuningResult:
    """Build the ``TuningResult`` of a study whose last round is closed.

    Args:
        lineage: The study's lineage; its last round has a ``completed`` end.
        trials: Every trial of the study.
        best_params: The study's best trial's recorded params
            (:func:`recorded_trial_params`), under their raw names.
        best_score: The study's best value.
        metric_name: The first metric's name.
        direction: The study's direction.
        current_round: The round this call ran; its record's boundary report
            is the result's, so a round finished on reattach reports what it
            decided when it started.

    Returns:
        The result: best params split by the lineage's routes (§6.4), every
        trial with the round its number falls in (§7.1), and the summaries
        of the ``completed`` rounds.
    """
    best_model, best_smart, best_training = split_by_routes(best_params, lineage.routes)
    by_round = {record.round: record for record in lineage.rounds}
    return TuningResult(
        best_model_params=best_model,
        best_smart_params=best_smart,
        best_training_params=best_training,
        best_score=best_score,
        trials=[
            TrialResult(
                number=t.number,
                params=recorded_trial_params(lineage, t),
                score=t.value if t.value is not None else float("nan"),
                state=t.state.name.lower(),
                round=round_of(lineage.rounds, t.number),
            )
            for t in trials
        ],
        metric_name=metric_name,
        direction=direction,
        rounds=round_summaries(lineage, trials),
        boundary_report=by_round[current_round].boundary_report,
    )
