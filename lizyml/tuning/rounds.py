"""Round-summary assembly and per-trial round renumbering for re-tune (#209).

Extracted verbatim from the ``Model`` facade (H-0068 introduced this in
``core/model.py``). This is ``TuningResult`` domain logic — building a
``RoundSummary``, computing cumulative round boundaries, and renumbering
per-trial ``round`` fields — so per the category contract it belongs in
``tuning/`` (Layer 2), not the Facade. Pure function: no ``Model`` state.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from lizyml.core.types.tuning_result import BoundaryReport, RoundSummary, TuningResult


def assemble_round_result(
    raw_result: TuningResult,
    *,
    round_number: int,
    actual_n_trials: int,
    best_score_before: float | None,
    expanded_names: tuple[str, ...],
    space: list[Any],
    boundary_report: BoundaryReport | None,
    prior_trials: int,
    prior_rounds: tuple[RoundSummary, ...],
) -> tuple[TuningResult, tuple[RoundSummary, ...]]:
    """Build the round summary, fix per-trial round numbers, and assemble the
    final :class:`TuningResult`.

    Args:
        raw_result: The ``TuningResult`` returned by the just-completed round.
        round_number: 1-based number of the just-completed round.
        actual_n_trials: Trials executed in this round.
        best_score_before: Best score prior to this round (``None`` on round 1).
        expanded_names: Dim names whose boundary was expanded this round.
        space: The search space used this round (snapshotted into the summary).
        boundary_report: Boundary report for this round (``None`` when disabled).
        prior_trials: Number of trials that existed before this round (for the
            resume case; trials with ``number >= prior_trials`` are new).
        prior_rounds: The completed rounds before this one (``Model._rounds``).

    Returns:
        ``(final_result, all_rounds)`` — ``all_rounds`` is the complete tuple
        including the just-completed round, ready to persist.
    """
    round_summary = RoundSummary(
        round=round_number,
        n_trials=actual_n_trials,
        best_score_before=best_score_before,
        best_score_after=raw_result.best_score,
        expanded_dims=expanded_names,
        space_snapshot=tuple(space),
    )
    all_rounds = tuple(prior_rounds) + (round_summary,)

    # Fix up trial round numbers: trials from previous rounds keep their
    # original round, new trials get the current round_number. Pre-compute
    # (cumulative_count, round) so the per-trial lookup is a small linear scan
    # over rounds (O(n + m) total) instead of O(n × m).
    round_boundaries: list[tuple[int, int]] = []
    running = 0
    for rs in prior_rounds:
        running += rs.n_trials
        round_boundaries.append((running, rs.round))

    fixed_trials = []
    for t in raw_result.trials:
        if t.number >= prior_trials:
            fixed_trials.append(dataclasses.replace(t, round=round_number))
            continue
        trial_round = 1
        for cumulative, rs_round in round_boundaries:
            if t.number < cumulative:
                trial_round = rs_round
                break
        fixed_trials.append(dataclasses.replace(t, round=trial_round))

    final_result = TuningResult(
        best_model_params=raw_result.best_model_params,
        best_smart_params=raw_result.best_smart_params,
        best_training_params=raw_result.best_training_params,
        best_score=raw_result.best_score,
        trials=fixed_trials,
        metric_name=raw_result.metric_name,
        direction=raw_result.direction,
        rounds=all_rounds,
        boundary_report=boundary_report,
    )
    return final_result, all_rounds
