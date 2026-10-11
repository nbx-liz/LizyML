"""H-0122 criterion 35: ``lizyml.tuning.read_study_lineage`` (§7.6).

Every §7.4 state is built in a real SQLite storage by running (and killing)
real tunes, then read through the public import path. Each read is checked to
leave the study as it was: the trial count, every trial's state (a RUNNING
trial stays RUNNING) and every ``lizyml:`` attribute.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import optuna
import pytest

import lizyml
from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.tuning import RoundRecord, StudyLineage, read_study_lineage
from lizyml.tuning.lineage import STUDY_KEY, round_key
from tests.test_tuning._resume_helpers import (
    INCONSISTENT,
    ROUND_1_TRIALS,
    ROUND_2_TRIALS,
    ProcessDeath,
    crash_round_2,
    delete_attrs,
    die_at_attr,
    die_in_objective,
    edge_config,
    fast_config,
    frame,
    load,
    set_attr,
    snapshot_study,
    url,
)


@pytest.fixture(scope="module")
def templates(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """``c``: two completed rounds; ``o``: round 2 open with a RUNNING trial;
    ``f``: round 1 closed ``failed``."""
    root = tmp_path_factory.mktemp("lineage")
    for key in ("c", "o", "f"):
        (root / key).mkdir()
    completed = url(root / "c")
    model = Model(fast_config(), data=frame())
    model.tune(storage=completed, study_name="s")
    model.tune(resume=True)
    with pytest.MonkeyPatch.context() as mp:
        crash_round_2(mp, Model(edge_config(), data=frame()), url(root / "o"), "s")
    failed = url(root / "f")
    with pytest.MonkeyPatch.context() as mp:
        first = Model(edge_config(n_trials=1), data=frame())
        die_in_objective(mp, first, at_call=1)
        with pytest.raises(ProcessDeath):
            first.tune(storage=failed, study_name="s")
    with pytest.raises(LizyMLError):
        Model(edge_config(n_trials=1), data=frame()).tune(
            resume=True, storage=failed, study_name="s"
        )
    return {key: root / key / "s.db" for key in ("c", "o", "f")}


def _copy(templates: dict[str, Path], key: str, tmp_path: Path) -> str:
    shutil.copy(templates[key], tmp_path / "s.db")
    return url(tmp_path)


def _absent(templates: dict[str, Path], tmp_path: Path) -> str:
    return url(tmp_path)


def _legacy(templates: dict[str, Path], tmp_path: Path) -> str:
    storage = _copy(templates, "c", tmp_path)
    assert delete_attrs(storage, "lizyml:%") > 0
    return storage


def _uninitialized(templates: dict[str, Path], tmp_path: Path) -> str:
    storage = url(tmp_path)
    optuna.create_study(storage=storage, study_name="s")
    return storage


def _unreadable(templates: dict[str, Path], tmp_path: Path) -> str:
    storage = _copy(templates, "c", tmp_path)
    set_attr(storage, "s", STUDY_KEY, {"schema": 999})
    return storage


def _no_round(templates: dict[str, Path], tmp_path: Path) -> str:
    storage = url(tmp_path)
    with pytest.MonkeyPatch.context() as mp:
        die_at_attr(mp, round_key(1))
        with pytest.raises(ProcessDeath):
            Model(fast_config(), data=frame()).tune(storage=storage, study_name="s")
    return storage


def _inconsistent(templates: dict[str, Path], tmp_path: Path) -> str:
    storage = _copy(templates, "c", tmp_path)
    INCONSISTENT["gap"](storage)
    return storage


BUILDERS: dict[str, Callable[[dict[str, Path], Path], str]] = {
    "absent": _absent,
    "legacy": _legacy,
    "uninitialized": _uninitialized,
    "unreadable": _unreadable,
    "no_round": _no_round,
    "inconsistent": _inconsistent,
    "open": lambda t, p: _copy(t, "o", p),
    "completed": lambda t, p: _copy(t, "c", p),
    "failed": lambda t, p: _copy(t, "f", p),
}


def _rounds_shape(view: StudyLineage) -> list[tuple[Any, ...]]:
    return [(r.round, r.status, r.n_trials, r.n_finished) for r in view.rounds]


#: ``(rounds as (round, status, n_trials, n_finished), open_round)`` per state.
EXPECTED: dict[str, tuple[list[tuple[Any, ...]], int | None]] = {
    "absent": ([], None),
    "legacy": ([], None),
    "uninitialized": ([], None),
    "unreadable": ([], None),
    "no_round": ([], None),
    "inconsistent": ([], None),
    "open": (
        [
            (1, "completed", ROUND_1_TRIALS, ROUND_1_TRIALS),
            # trial #2 COMPLETE, trial #3 RUNNING (not finished)
            (2, "open", ROUND_2_TRIALS, 1),
        ],
        2,
    ),
    "completed": ([(1, "completed", 2, 2), (2, "completed", 2, 2)], None),
    "failed": ([(1, "failed", 1, 1)], None),
}


@pytest.mark.parametrize("state", list(BUILDERS))
def test_state(templates: dict[str, Path], tmp_path: Path, state: str) -> None:
    storage = BUILDERS[state](templates, tmp_path)
    view = read_study_lineage(storage, "s")
    assert view.state == state
    rounds, open_round = EXPECTED[state]
    assert _rounds_shape(view) == rounds
    assert view.open_round == open_round


@pytest.mark.parametrize("state", list(BUILDERS))
def test_reads_only(templates: dict[str, Path], tmp_path: Path, state: str) -> None:
    storage = BUILDERS[state](templates, tmp_path)
    if state == "absent":
        read_study_lineage(storage, "s")
        assert "s" not in optuna.get_all_study_names(storage)
        return
    before = snapshot_study(storage, "s")
    read_study_lineage(storage, "s")
    assert snapshot_study(storage, "s") == before
    if state == "open":
        assert load(storage, "s").trials[-1].state.name == "RUNNING"


def test_scores_follow_the_records(templates: dict[str, Path], tmp_path: Path) -> None:
    storage = _copy(templates, "c", tmp_path)
    first, second = read_study_lineage(storage, "s").rounds
    assert first.best_score_before is None
    assert second.best_score_before == first.best_score_after
    assert second.best_score_after == load(storage, "s").best_value
    (tmp_path / "f").mkdir()
    failed = read_study_lineage(_copy(templates, "f", tmp_path / "f"), "s")
    assert failed.rounds[0].best_score_after is None


def test_storage_object_is_accepted(tmp_path: Path, templates: dict[str, Path]) -> None:
    storage = _copy(templates, "c", tmp_path)
    view = read_study_lineage(optuna.storages.RDBStorage(storage), "s")
    assert view.state == "completed"


@pytest.mark.parametrize(
    ("kwargs", "surface"),
    [
        ({"storage": "<url>", "study_name": None}, "tune(storage=)"),
        ({"storage": None, "study_name": "s"}, "tune(study_name=)"),
        (
            {"storage": None, "study_name": None},
            "read_study_lineage(storage=, study_name=)",
        ),
    ],
    ids=["storage_only", "name_only", "neither"],
)
def test_precheck(tmp_path: Path, kwargs: dict[str, Any], surface: str) -> None:
    """Refused before the storage is contacted: no SQLite file is created.

    The ``storage=``/``study_name=`` surfaces are ``Model.tune``'s: the
    precheck is shared (§7.6, "the same precheck as tune").
    """
    if kwargs["storage"] == "<url>":
        kwargs = {**kwargs, "storage": url(tmp_path)}
    with pytest.raises(LizyMLError) as caught:
        read_study_lineage(**kwargs)
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert caught.value.context["surface"] == surface
    assert not (tmp_path / "s.db").exists()


def test_frozen_and_tuple(templates: dict[str, Path], tmp_path: Path) -> None:
    view = read_study_lineage(_copy(templates, "c", tmp_path), "s")
    assert isinstance(view.rounds, tuple)
    assert all(isinstance(r, RoundRecord) for r in view.rounds)
    with pytest.raises(FrozenInstanceError):
        view.open_round = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        view.rounds[0].status = "open"  # type: ignore[misc]


def test_exports() -> None:
    assert {"read_study_lineage", "StudyLineage", "RoundRecord"} <= set(
        lizyml.tuning.__all__
    )
    assert not {"read_study_lineage", "StudyLineage", "RoundRecord"} & set(
        lizyml.__all__
    )
