"""Unit tests for the public lineage view (H-0122 §7.6, criterion 35).

``lineage_view`` turns the internal :class:`Lineage` into the public
:class:`StudyLineage`. The lineages are built from synthetic attributes and
trial stand-ins, so every §7.4 state is pinned without running a tune; the
storage-backed reader is covered by ``test_read_study_lineage.py``.
"""

from __future__ import annotations

import typing
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from lizyml.tuning.lineage import LineageState, classify_study
from lizyml.tuning.study_lineage import (
    RoundRecord,
    StudyLineage,
    StudyState,
    lineage_view,
)
from tests.test_tuning._lineage_fixtures import (
    FakeTrial,
    St,
    attrs_of,
    make_trials,
    stored_end,
    stored_round,
    stored_study,
    two_round_attrs,
)


def _view(attrs: dict[str, Any], trials: list[FakeTrial]) -> StudyLineage:
    return lineage_view(classify_study(attrs, trials), trials)


def test_states_are_the_lineage_states_plus_absent() -> None:
    """The public literal is §7.4's closed table plus ``absent``, one-to-one."""
    assert set(typing.get_args(StudyState)) == {
        "absent",
        *typing.get_args(LineageState),
    }


def test_absent() -> None:
    assert lineage_view(None, ()) == StudyLineage(
        state="absent", rounds=(), open_round=None
    )


@pytest.mark.parametrize(
    ("attrs", "n_trials", "state"),
    [
        ({}, 2, "legacy"),
        ({}, 0, "uninitialized"),
        ({"lizyml:study": {"schema": 999}}, 0, "unreadable"),
        (attrs_of(stored_study()), 0, "no_round"),
        (attrs_of(stored_study()), 2, "inconsistent"),
    ],
)
def test_states_without_rounds(
    attrs: dict[str, Any], n_trials: int, state: str
) -> None:
    """No round summary is shown for a study whose rounds cannot be trusted."""
    view = _view(attrs, make_trials(n_trials))
    assert (view.state, view.rounds, view.open_round) == (state, (), None)


def test_completed_two_rounds() -> None:
    trials = make_trials(6)
    trials[1] = FakeTrial(number=1, state=St.FAIL, value=None)
    trials[4] = FakeTrial(number=4, state=St.PRUNED)
    view = _view(two_round_attrs(), trials)
    assert view.state == "completed"
    assert view.open_round is None
    assert view.rounds == (
        RoundRecord(1, "completed", 3, 3, None, 1.0),
        RoundRecord(2, "completed", 3, 3, 1.0, 1.0),
    )


def test_open_round_excludes_running_trials() -> None:
    """``n_finished`` counts COMPLETE, FAIL and PRUNED; RUNNING is not finished."""
    trials = make_trials(5)
    trials[4] = FakeTrial(number=4, state=St.RUNNING, value=None, seconds=None)
    view = _view(two_round_attrs(end2=False), trials)
    assert view.state == "open"
    assert view.open_round == 2
    assert view.rounds[-1] == RoundRecord(2, "open", 3, 1, 1.0, None)


def test_waiting_trials_are_not_finished() -> None:
    trials = make_trials(5)
    trials[4] = FakeTrial(number=4, state=St.WAITING, value=None, seconds=None)
    view = _view(two_round_attrs(end2=False), trials)
    assert view.rounds[-1].n_finished == 1


def test_failed_rounds_are_included() -> None:
    """Unlike ``TuningResult.rounds``, a ``failed`` round is listed (§7.6)."""
    attrs = attrs_of(
        stored_study(),
        (stored_round(1, 0, n_trials=1), stored_round(2, 1, best_before=None)),
        {1: stored_end("failed"), 2: stored_end("failed")},
    )
    trials = [FakeTrial(number=i, state=St.FAIL, value=None) for i in range(4)]
    view = _view(attrs, trials)
    assert view.state == "failed"
    assert view.rounds == (
        RoundRecord(1, "failed", 1, 1, None, None),
        RoundRecord(2, "failed", 3, 3, None, None),
    )


def test_failed_round_then_completed_round() -> None:
    attrs = attrs_of(
        stored_study(),
        (stored_round(1, 0, n_trials=1), stored_round(2, 1, best_before=None)),
        {1: stored_end("failed"), 2: stored_end("completed", 0.5)},
    )
    trials = [FakeTrial(number=0, state=St.FAIL, value=None), *make_trials(4)[1:]]
    view = _view(attrs, trials)
    assert view.state == "completed"
    assert [(r.round, r.status) for r in view.rounds] == [
        (1, "failed"),
        (2, "completed"),
    ]
    assert view.rounds[1].best_score_after == 0.5


def test_frozen_and_tuple() -> None:
    view = _view(two_round_attrs(), make_trials(6))
    assert isinstance(view.rounds, tuple)
    with pytest.raises(FrozenInstanceError):
        view.state = "absent"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        view.rounds[0].n_trials = 9  # type: ignore[misc]
