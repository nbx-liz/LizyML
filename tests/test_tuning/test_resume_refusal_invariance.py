"""H-0122 criterion 22: a refusal changes nothing (INV-R4).

One table over every PR 1 refusal row (design §6.4, spec §8): each row builds
its state, snapshots the study (trial count, states, values, params and every
``lizyml:`` attribute; a stored study is re-read from the storage) and the
Model's tune-owned state, calls ``tune``, checks the error code and the reason
(so a row cannot pass on an unrelated refusal), and compares both snapshots.

Rows that refuse a Model holding a study use a Model that has tuned, so the
snapshot covers real state, not only ``None`` fields.
"""

from __future__ import annotations

import copy
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import pytest
from optuna.storages import RDBStorage

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.tuning.lineage import STUDY_KEY, classify_study
from tests.test_tuning._resume_helpers import (
    INCONSISTENT,
    RATIO,
    admission_refused_config,
    crash_round_2,
    delete_attrs,
    edge_config,
    fast_config,
    frame,
    load,
    set_attr,
    snapshot_model,
    snapshot_study,
    study_record_for,
    url,
)

CI = ErrorCode.CONFIG_INVALID


# ---------------------------------------------------------------------------
# Templates and the per-test environment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def templates(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """``c``: state C, two rounds. ``o``: state O, round 2 with a RUNNING trial."""
    root = tmp_path_factory.mktemp("invariance")
    (root / "c").mkdir()
    (root / "o").mkdir()
    completed = url(root / "c")
    Model(fast_config(), data=frame()).tune(storage=completed, study_name="s")
    Model(fast_config(), data=frame()).tune(
        resume=True, storage=completed, study_name="s"
    )
    with pytest.MonkeyPatch.context() as mp:
        crash_round_2(mp, Model(edge_config(), data=frame()), url(root / "o"), "s")
    return {"c": root / "c" / "s.db", "o": root / "o" / "s.db"}


@dataclass
class Env:
    tmp_path: Path
    c: str  # a copy of the C template
    o: str  # a copy of the O template
    spare: str  # an empty storage

    def state(self, storage: str, name: str = "s") -> str:
        study = load(storage, name)
        return classify_study(study.user_attrs, study.trials).state


@pytest.fixture
def env(templates: dict[str, Path], tmp_path: Path) -> Env:
    for key in ("c", "o"):
        (tmp_path / key).mkdir()
        shutil.copy(templates[key], tmp_path / key / "s.db")
    (tmp_path / "spare").mkdir()
    return Env(
        tmp_path,
        url(tmp_path / "c"),
        url(tmp_path / "o"),
        url(tmp_path / "spare"),
    )


@dataclass
class Case:
    """One refusal: the Model, where its study is, the call, and the reason."""

    model: Model
    #: ``(storage, name)`` re-read from the storage, or a study object.
    study: Any
    kwargs: dict[str, Any]
    code: ErrorCode
    reason: Callable[[LizyMLError], bool]
    extra: dict[str, Any] = field(default_factory=dict)


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


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------


def _storage_no_name(env: Env, resume: bool) -> Case:
    model = _memory_tuned()
    return Case(
        model,
        model._study,
        {"resume": resume, "storage": env.spare},
        CI,
        _surface("tune(storage=)"),
        {"no_file": env.spare},
    )


def _name_no_storage(env: Env, resume: bool) -> Case:
    model = _memory_tuned()
    return Case(
        model,
        model._study,
        {"resume": resume, "study_name": "s"},
        CI,
        _surface("tune(study_name=)"),
    )


def _resume_false_existing_c(env: Env) -> Case:
    model = _holding_c(env)
    return Case(
        model,
        (env.c, "s"),
        {"storage": env.c, "study_name": "s"},
        CI,
        _message("already exists"),
    )


def _resume_false_existing_o(env: Env) -> Case:
    assert env.state(env.o) == "open"
    return Case(
        _memory_tuned(),
        (env.o, "s"),
        {"storage": env.o, "study_name": "s"},
        CI,
        _message("already exists"),
    )


def _resume_false_existing_e(env: Env) -> Case:
    _empty_study(env)
    return Case(
        _memory_tuned(),
        (env.spare, "s"),
        {"storage": env.spare, "study_name": "s"},
        CI,
        _message("already exists"),
    )


def _other_name(env: Env) -> Case:
    return Case(
        _holding_c(env),
        (env.c, "s"),
        {"resume": True, "storage": env.c, "study_name": "other"},
        CI,
        _message("cannot be shown to name it"),
    )


def _other_url_string(env: Env) -> Case:
    path = Path(env.c[len("sqlite:///") :])
    spelled = "sqlite:///./" + os.path.relpath(path)
    assert spelled != env.c
    return Case(
        _holding_c(env),
        (env.c, "s"),
        {"resume": True, "storage": spelled, "study_name": "s"},
        CI,
        _message("cannot be shown to name it"),
    )


def _other_storage_object(env: Env) -> Case:
    model = Model(fast_config(), data=frame())
    model.tune(storage=RDBStorage(env.spare), study_name="x")
    return Case(
        model,
        (env.spare, "x"),
        {"resume": True, "storage": RDBStorage(env.spare), "study_name": "x"},
        CI,
        _message("cannot be shown to name it"),
    )


def _memory_plus_storage(env: Env) -> Case:
    model = _memory_tuned()
    return Case(
        model,
        (env.c, "s"),
        {"resume": True, "storage": env.c, "study_name": "s"},
        CI,
        _message("cannot be shown to name it"),
        {"also": model._study},
    )


def _identity_data(env: Env) -> Case:
    return Case(
        _holding_c(env),
        (env.c, "s"),
        {"resume": True, "data": _changed_frame()},
        CI,
        _mismatched("data.values"),
    )


def _identity_config(env: Env) -> Case:
    model = _holding_c(env)
    model._cfg.training.seed = model._cfg.training.seed + 1
    return Case(
        model, (env.c, "s"), {"resume": True}, CI, _mismatched("config.training")
    )


def _identity_metric(env: Env) -> Case:
    model = _holding_c(env)
    model._cfg.evaluation.metrics = ["mae"]
    return Case(model, (env.c, "s"), {"resume": True}, CI, _mismatched("metric"))


def _legacy(env: Env) -> Case:
    assert delete_attrs(env.c, "lizyml:%") > 0
    return Case(
        Model(fast_config(), data=frame()),
        (env.c, "s"),
        {"resume": True, "storage": env.c, "study_name": "s"},
        CI,
        _state_is("legacy"),
    )


def _unreadable(env: Env) -> Case:
    set_attr(env.c, "s", STUDY_KEY, {"schema": 999})
    return Case(
        Model(fast_config(), data=frame()),
        (env.c, "s"),
        {"resume": True, "storage": env.c, "study_name": "s"},
        CI,
        _state_is("unreadable"),
    )


def _unreadable_empty(env: Env) -> Case:
    _empty_study(env)
    set_attr(env.spare, "s", STUDY_KEY, {"schema": 999})
    return Case(
        Model(fast_config(), data=frame()),
        (env.spare, "s"),
        {"resume": True, "storage": env.spare, "study_name": "s"},
        CI,
        _state_is("unreadable"),
    )


def _inconsistent(corruption: str) -> Callable[[Env], Case]:
    def build(env: Env) -> Case:
        INCONSISTENT[corruption](env.c)
        assert env.state(env.c) == "inconsistent"
        return Case(
            Model(fast_config(), data=frame()),
            (env.c, "s"),
            {"resume": True, "storage": env.c, "study_name": "s"},
            CI,
            _state_is("inconsistent"),
        )

    return build


def _config_space(env: Env) -> Case:
    model = _holding_c(env)
    assert model._cfg.tuning is not None
    space = copy.deepcopy(model._cfg.tuning.optuna.space)
    space["num_leaves"]["high"] = 9
    model._cfg.tuning.optuna.space = space
    return Case(
        model, (env.c, "s"), {"resume": True}, CI, _surface("tuning.optuna.space")
    )


def _config_space_mode(env: Env) -> Case:
    model = _holding_c(env)
    assert model._cfg.tuning is not None
    model._cfg.tuning.optuna.space_mode = "merge"
    return Case(
        model, (env.c, "s"), {"resume": True}, CI, _surface("tuning.optuna.space_mode")
    )


def _open(kwargs: dict[str, Any], reason: Callable[[LizyMLError], bool]) -> Any:
    def build(env: Env) -> Case:
        assert load(env.o, "s").trials[3].state.name == "RUNNING"
        return Case(
            Model(edge_config(), data=frame()),
            (env.o, "s"),
            {"resume": True, "storage": env.o, "study_name": "s", **kwargs},
            CI,
            reason,
        )

    return build


def _no_round(env: Env, cfg: dict[str, Any], data: pd.DataFrame) -> None:
    _empty_study(env)
    set_attr(env.spare, "s", STUDY_KEY, study_record_for(cfg, data))
    assert env.state(env.spare) == "no_round"


def _no_round_identity(env: Env) -> Case:
    _no_round(env, fast_config(), frame())
    return Case(
        Model(fast_config(), data=_changed_frame()),
        (env.spare, "s"),
        {"resume": True, "storage": env.spare, "study_name": "s"},
        CI,
        _mismatched("data.values"),
    )


def _uninitialised_admission(env: Env) -> Case:
    _empty_study(env)
    return Case(
        Model(admission_refused_config(), data=frame("binary")),
        (env.spare, "s"),
        {"resume": True, "storage": env.spare, "study_name": "s"},
        CI,
        _message(RATIO),
    )


def _no_round_admission(env: Env) -> Case:
    _no_round(env, admission_refused_config(), frame("binary"))
    return Case(
        Model(admission_refused_config(), data=frame("binary")),
        (env.spare, "s"),
        {"resume": True, "storage": env.spare, "study_name": "s"},
        CI,
        _message(RATIO),
    )


def _direction_mismatch(env: Env) -> Case:
    _empty_study(env, direction="maximize")
    return Case(
        Model(fast_config(), data=frame()),
        (env.spare, "s"),
        {"resume": True, "storage": env.spare, "study_name": "s"},
        CI,
        _message("direction disagrees"),
    )


def _no_study_no_storage(env: Env) -> Case:
    return Case(
        Model(fast_config(), data=frame()),
        None,
        {"resume": True},
        ErrorCode.TUNING_FAILED,
        lambda e: e.context.get("storage_given") is False,
    )


ROWS: dict[str, Callable[[Env], Case]] = {
    "storage_no_name[resume=False]": lambda env: _storage_no_name(env, False),
    "storage_no_name[resume=True]": lambda env: _storage_no_name(env, True),
    "name_no_storage[resume=False]": lambda env: _name_no_storage(env, False),
    "name_no_storage[resume=True]": lambda env: _name_no_storage(env, True),
    "resume_false_existing[C]": _resume_false_existing_c,
    "resume_false_existing[O]": _resume_false_existing_o,
    "resume_false_existing[E]": _resume_false_existing_e,
    "other_target[name]": _other_name,
    "other_target[url_string]": _other_url_string,
    "other_target[storage_object]": _other_storage_object,
    "other_target[memory_plus_storage]": _memory_plus_storage,
    "identity[data]": _identity_data,
    "identity[config]": _identity_config,
    "identity[metric]": _identity_metric,
    "legacy": _legacy,
    "unreadable": _unreadable,
    "unreadable_empty": _unreadable_empty,
    **{f"inconsistent[{name}]": _inconsistent(name) for name in sorted(INCONSISTENT)},
    "config_space": _config_space,
    "config_space_mode": _config_space_mode,
    "open_n_trials": _open({"n_trials": 4}, _surface("tune(n_trials=)")),
    "open_expand": _open(
        {"expand_boundary": False}, _surface("tune(expand_boundary=)")
    ),
    "open_identity": _open({"data": _changed_frame()}, _mismatched("data.values")),
    "no_round_identity": _no_round_identity,
    "uninitialised_admission": _uninitialised_admission,
    "no_round_admission": _no_round_admission,
    "direction_mismatch": _direction_mismatch,
    "no_study_no_storage": _no_study_no_storage,
}


@pytest.mark.parametrize("row", list(ROWS))
def test_refusal_changes_nothing(env: Env, row: str) -> None:
    case = ROWS[row](env)
    also = case.extra.get("also")
    before = (
        _snapshot(case) if case.study is not None else snapshot_model(case.model),
        snapshot_study(also) if also is not None else None,
    )
    with pytest.raises(LizyMLError) as caught:
        case.model.tune(**case.kwargs)
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
