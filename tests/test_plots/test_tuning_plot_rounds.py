"""H-0122 criterion 16a: round boundaries on the reporting surfaces (§6.5).

``tuning_plot``'s separators and round annotations, and ``tuning_table``'s
``round`` column, follow the trials' rounds (decided by trial number from the
study's round records), not a running sum of ``RoundSummary.n_trials``: a
``failed`` round has no summary, so the sum skips its trials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lizyml import Model
from lizyml.core.types.tuning_result import RoundSummary, TrialResult, TuningResult
from lizyml.plots.tuning import plot_tuning_history
from lizyml.tuning.lineage import classify_study, round_of
from tests.test_tuning._resume_helpers import (
    edge_config,
    failed_round_1,
    fast_config,
    frame,
    load,
    url,
)


def _vlines(fig: Any) -> list[float]:
    return [shape.x0 for shape in fig.layout.shapes or ()]


def _labels(fig: Any) -> dict[str, float]:
    return {a.text: a.x for a in fig.layout.annotations or ()}


def _first_numbers(result: TuningResult) -> dict[int, int]:
    """The first trial number of each round, from the trials' rounds."""
    first: dict[int, int] = {}
    for t in sorted(result.trials, key=lambda t: t.number):
        first.setdefault(t.round, t.number)
    return first


def _assert_follows_trial_rounds(result: TuningResult) -> None:
    fig = plot_tuning_history(result)
    first = _first_numbers(result)
    rounds = sorted(first)
    assert len(rounds) > 1
    assert _vlines(fig) == [first[k] - 0.5 for k in rounds[1:]]
    labels = _labels(fig)
    for k in rounds:
        numbers = [t.number for t in result.trials if t.round == k]
        (label,) = [text for text in labels if text.split(" (")[0] == f"Round {k}"]
        assert labels[label] == (min(numbers) + max(numbers)) / 2


def test_separators_follow_trial_rounds_failed_round_1(tmp_path: Path) -> None:
    """Round 1 closed ``failed`` (no summary); round 2 completed."""
    storage = url(tmp_path)
    failed_round_1(storage)
    model = Model(edge_config(n_trials=2), data=frame())
    result = model.tune(resume=True, storage=storage, study_name="s")
    assert [r.round for r in result.rounds] == [2]
    assert [t.round for t in result.trials] == [1, 2, 2]
    _assert_follows_trial_rounds(result)
    assert _vlines(model.tuning_plot()) == [0.5]
    assert set(_labels(model.tuning_plot())) == {"Round 1", "Round 2"}


def test_separators_follow_trial_rounds_timeout_round() -> None:
    """Round 1 stops on its timeout well short of its ``n_trials``."""
    cfg = fast_config(n_trials=1000)
    cfg["tuning"]["optuna"]["params"]["timeout"] = 0.5
    model = Model(cfg, data=frame())
    first = model.tune()
    n1 = len(first.trials)
    assert 0 < n1 < 1000
    result = model.tune(resume=True, n_trials=2)
    assert [t.round for t in result.trials][n1:] == [2] * (len(result.trials) - n1)
    _assert_follows_trial_rounds(result)
    assert _vlines(model.tuning_plot()) == [n1 - 0.5]


def _summary(k: int, n: int, expanded: tuple[str, ...] = ()) -> RoundSummary:
    return RoundSummary(
        round=k,
        n_trials=n,
        best_score_before=None if k == 1 else 0.5,
        best_score_after=0.5,
        expanded_dims=expanded,
        space_snapshot=(),
    )


def _trial(number: int, k: int, state: str = "complete") -> TrialResult:
    return TrialResult(
        number=number, params={"x": number}, score=0.5, state=state, round=k
    )


def test_cumulative_summaries_disagree_unit() -> None:
    """Summaries whose ``n_trials`` do not add up to the trials' rounds.

    A running sum of ``n_trials`` (2, then 3) would put the separators at 1.5
    and 4.5; the trials put them at 2.5 and 3.5. Round 2 has no summary (it
    closed ``failed``) and gets a plain label (design R-10).
    """
    result = TuningResult(
        best_model_params={},
        best_smart_params={},
        best_training_params={},
        best_score=0.5,
        trials=[
            _trial(0, 1),
            _trial(1, 1),
            _trial(2, 1),
            _trial(3, 2, "fail"),
            _trial(4, 3),
            _trial(5, 3),
        ],
        metric_name="rmse",
        direction="minimize",
        rounds=(_summary(1, 2), _summary(3, 3, ("lr",))),
    )
    fig = plot_tuning_history(result)
    assert _vlines(fig) == [2.5, 3.5]
    assert _labels(fig) == {"Round 1": 1.0, "Round 2": 3.0, "Round 3 (lr)": 4.5}


def test_one_round_draws_no_separator() -> None:
    result = TuningResult(
        best_model_params={},
        best_smart_params={},
        best_training_params={},
        best_score=0.5,
        trials=[_trial(0, 2), _trial(1, 2)],
        metric_name="rmse",
        direction="minimize",
        rounds=(_summary(2, 2),),
    )
    fig = plot_tuning_history(result)
    assert _vlines(fig) == []
    assert _labels(fig) == {}


def test_no_trials_draws_nothing() -> None:
    """After ``Model.load`` the result keeps no trials (H-0086)."""
    result = TuningResult(
        best_model_params={},
        best_smart_params={},
        best_training_params={},
        best_score=0.5,
        trials=[],
        metric_name="rmse",
        direction="minimize",
        rounds=(_summary(1, 2), _summary(2, 2)),
    )
    fig = plot_tuning_history(result)
    assert _vlines(fig) == []
    assert _labels(fig) == {}


@pytest.mark.parametrize("scenario", ["failed_round_1", "two_rounds"])
def test_tuning_table_round_column(tmp_path: Path, scenario: str) -> None:
    storage = url(tmp_path)
    if scenario == "failed_round_1":
        failed_round_1(storage)
        model = Model(edge_config(n_trials=2), data=frame())
        result = model.tune(resume=True, storage=storage, study_name="s")
    else:
        model = Model(fast_config(), data=frame())
        model.tune(storage=storage, study_name="s")
        result = model.tune(resume=True)
    table = model.tuning_table()
    assert table["round"].tolist() == [t.round for t in result.trials]
    study = load(storage, "s")
    lineage = classify_study(study.user_attrs, study.trials)
    assert table["round"].tolist() == [
        round_of(lineage.rounds, t.number) for t in study.trials
    ]
