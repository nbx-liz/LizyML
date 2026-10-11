"""Infrastructure of the criterion-22 table (``test_resume_refusal_invariance``).

The study templates and the per-test environment, the :class:`Case` a row
builds, the building blocks rows share, and :func:`run_case`, which makes the
call and checks that the refusal changed neither the study nor the Model
(INV-R4).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.tuning.lineage import classify_study
from tests.test_tuning._resume_helpers import (
    OPEN_SPACE,
    crash_round_2,
    edge_config,
    failed_round_1,
    fast_config,
    frame,
    load,
    open_with_space,
    snapshot_model,
    snapshot_study,
    url,
)

CI = ErrorCode.CONFIG_INVALID


# ---------------------------------------------------------------------------
# Templates and the per-test environment
# ---------------------------------------------------------------------------


_TEMPLATES = ("c", "o", "os", "f")


@pytest.fixture(scope="module")
def templates(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """``c``: state C, two rounds. ``o``: state O, round 2 with a RUNNING trial.
    ``os``: the same, round 2 started with ``space=OPEN_SPACE``. ``f``: state
    F, round 1 closed ``failed``."""
    root = tmp_path_factory.mktemp("invariance")
    for key in _TEMPLATES:
        (root / key).mkdir()
    completed = url(root / "c")
    Model(fast_config(), data=frame()).tune(storage=completed, study_name="s")
    Model(fast_config(), data=frame()).tune(
        resume=True, storage=completed, study_name="s"
    )
    with pytest.MonkeyPatch.context() as mp:
        crash_round_2(mp, Model(edge_config(), data=frame()), url(root / "o"), "s")
    open_with_space(root / "os", OPEN_SPACE)
    failed_round_1(url(root / "f"))
    return {key: root / key / "s.db" for key in _TEMPLATES}


@dataclass
class Env:
    tmp_path: Path
    c: str  # a copy of the C template
    o: str  # a copy of the O template
    os: str  # a copy of the O template whose round 2 was started with space=
    f: str  # a copy of the F template
    spare: str  # an empty storage

    def state(self, storage: str, name: str = "s") -> str:
        study = load(storage, name)
        return classify_study(study.user_attrs, study.trials).state


@pytest.fixture
def env(templates: dict[str, Path], tmp_path: Path) -> Env:
    for key in _TEMPLATES:
        (tmp_path / key).mkdir()
        shutil.copy(templates[key], tmp_path / key / "s.db")
    (tmp_path / "spare").mkdir()
    return Env(
        tmp_path,
        url(tmp_path / "c"),
        url(tmp_path / "o"),
        url(tmp_path / "os"),
        url(tmp_path / "f"),
        url(tmp_path / "spare"),
    )


@dataclass
class Case:
    """One refusal: the Model, where its study is, the call, and the reason.

    The call is ``getattr(model, method)(*args, **kwargs)``. ``extra`` may
    hold ``no_file`` (a storage URL whose file must not exist afterwards),
    ``absent_study`` (``(storage, name)`` that must not have been created) and
    ``also`` (a second study object that must not change).
    """

    model: Model
    #: ``(storage, name)`` re-read from the storage, or a study object.
    study: Any
    kwargs: dict[str, Any]
    code: ErrorCode
    reason: Callable[[LizyMLError], bool]
    extra: dict[str, Any] = field(default_factory=dict)
    method: str = "tune"
    args: tuple[Any, ...] = ()


def _snapshot(case: Case) -> Any:
    if isinstance(case.study, tuple):
        return snapshot_study(*case.study), snapshot_model(case.model)
    return snapshot_study(case.study), snapshot_model(case.model)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _memory_tuned() -> Model:
    model = Model(fast_config(), data=frame())
    model.tune()
    return model


def _holding_c(env: Env) -> Model:
    """A Model that reattached to the C study (it now holds it, round 3)."""
    model = Model(fast_config(), data=frame())
    model.tune(resume=True, storage=env.c, study_name="s")
    return model


def _changed_frame() -> pd.DataFrame:
    df = frame()
    df.loc[0, "feat_a"] = float(df.loc[0, "feat_a"]) + 1.0
    return df


def _empty_study(env: Env, direction: str = "minimize") -> None:
    optuna.create_study(storage=env.spare, study_name="s", direction=direction)


def _surface(name: str) -> Callable[[LizyMLError], bool]:
    return lambda e: e.context.get("surface") == name


def _state_is(state: str) -> Callable[[LizyMLError], bool]:
    return lambda e: e.context.get("state") == state


def _mismatched(element: str) -> Callable[[LizyMLError], bool]:
    return lambda e: element in e.context.get("mismatched", ())


def _message(fragment: str) -> Callable[[LizyMLError], bool]:
    return lambda e: fragment in e.user_message


def run_case(case: Case) -> None:
    """Call, check the code and the reason, and compare the snapshots."""
    also = case.extra.get("also")
    before = (
        _snapshot(case) if case.study is not None else snapshot_model(case.model),
        snapshot_study(also) if also is not None else None,
    )
    with pytest.raises(LizyMLError) as caught:
        getattr(case.model, case.method)(*case.args, **case.kwargs)
    error = caught.value
    assert error.code is case.code, error
    assert case.reason(error), (error.user_message, error.context)
    after = (
        _snapshot(case) if case.study is not None else snapshot_model(case.model),
        snapshot_study(also) if also is not None else None,
    )
    assert after == before
    if "no_file" in case.extra:
        assert not Path(case.extra["no_file"][len("sqlite:///") :]).exists()
    if "absent_study" in case.extra:
        storage, name = case.extra["absent_study"]
        assert name not in optuna.get_all_study_names(storage)
