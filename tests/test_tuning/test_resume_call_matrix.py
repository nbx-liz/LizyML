"""H-0122 §3: the ``tune()`` call matrix and its pre-checks.

Criteria 9, 10, 12 (argument and schema parts), 14 (PR 1 rows, and the
``space=`` rows PR 2 adds at the end),
24, 25 and 33. Every refusal is also checked to change neither the study nor
the Model (INV-R4).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import optuna
import pytest
from optuna.storages import RDBStorage

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.tuning.lineage import STUDY_KEY, round_key
from tests.test_tuning._resume_helpers import (
    fast_config,
    frame,
    lizyml_attrs,
    load,
    set_attr,
    snapshot_model,
    snapshot_study,
    url,
)


def _tuned(storage: Any = None, name: str | None = None) -> Model:
    model = Model(fast_config(), data=frame())
    model.tune(storage=storage, study_name=name)
    return model


def _refused(
    model: Model, code: ErrorCode = ErrorCode.CONFIG_INVALID, **kwargs: Any
) -> LizyMLError:
    with pytest.raises(LizyMLError) as caught:
        model.tune(**kwargs)
    assert caught.value.code is code, caught.value
    return caught.value


# ---------------------------------------------------------------------------
# Pre-checks (criterion 12)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("given", ["storage_only", "name_only"])
def test_storage_only_or_name_only_refused(
    tmp_path: Path, resume: bool, given: str
) -> None:
    model = _tuned()
    before = (snapshot_study(model._study), snapshot_model(model))
    kwargs: dict[str, Any] = (
        {"storage": url(tmp_path)} if given == "storage_only" else {"study_name": "s"}
    )
    error = _refused(model, resume=resume, **kwargs)
    assert "study_name" in error.user_message
    assert (snapshot_study(model._study), snapshot_model(model)) == before
    assert not (tmp_path / "s.db").exists()


def test_storage_without_name_is_config_invalid_before_tuning_failed(
    tmp_path: Path,
) -> None:
    """Before: a fresh Model raised TUNING_FAILED for this call (§3, §8)."""
    _refused(Model(fast_config(), data=frame()), resume=True, storage=url(tmp_path))


def test_unreadable_schema_refused(tmp_path: Path) -> None:
    storage = url(tmp_path)
    _tuned(storage, "s")
    set_attr(storage, "s", STUDY_KEY, {"schema": 999})
    before = snapshot_study(storage, "s")
    model = Model(fast_config(), data=frame())
    error = _refused(model, resume=True, storage=storage, study_name="s")
    assert error.context["state"] == "unreadable"
    assert snapshot_study(storage, "s") == before


def test_unreadable_and_empty_is_unreadable(tmp_path: Path) -> None:
    """Criterion 33: S and N both match; S wins, so no round 1 is started."""
    storage = url(tmp_path)
    optuna.create_study(storage=storage, study_name="s", direction="minimize")
    set_attr(storage, "s", STUDY_KEY, {"schema": 999})
    before = snapshot_study(storage, "s")
    model = Model(fast_config(), data=frame())
    error = _refused(model, resume=True, storage=storage, study_name="s")
    assert error.context["state"] == "unreadable"
    assert snapshot_study(storage, "s") == before
    assert model._study is None


# ---------------------------------------------------------------------------
# resume=False onto an existing stored study (criterion 9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["completed", "uninitialized"])
def test_resume_false_onto_existing_refused(tmp_path: Path, state: str) -> None:
    storage = url(tmp_path)
    if state == "completed":
        _tuned(storage, "s")
    else:
        optuna.create_study(storage=storage, study_name="s", direction="minimize")
    before = snapshot_study(storage, "s")
    model = Model(fast_config(), data=frame())
    model_before = snapshot_model(model)
    error = _refused(model, storage=storage, study_name="s")
    assert "resume=True" in error.user_message
    assert snapshot_study(storage, "s") == before
    assert snapshot_model(model) == model_before


# ---------------------------------------------------------------------------
# resume=True with a study in memory (criteria 10, 14)
# ---------------------------------------------------------------------------


def test_resume_other_storage_refused(tmp_path: Path) -> None:
    storage = url(tmp_path)
    model = _tuned(storage, "s")
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    other = url(tmp_path, "other.db")
    _refused(model, resume=True, storage=other, study_name="s")
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before
    assert not (tmp_path / "other.db").exists()


@pytest.mark.parametrize("spelling", ["str", "same_object", "omitted"])
def test_same_storage_continues(tmp_path: Path, spelling: str) -> None:
    storage: Any = url(tmp_path)
    if spelling == "same_object":
        storage = RDBStorage(storage)
    model = _tuned(storage, "s")
    kwargs: dict[str, Any] = (
        {} if spelling == "omitted" else {"storage": storage, "study_name": "s"}
    )
    result = model.tune(resume=True, **kwargs)
    assert [r.round for r in result.rounds] == [1, 2]
    assert len(load(storage, "s").trials) == 4
    assert model._tuning_storage is storage
    assert model._tuning_study_name == "s"


def _relative_spelling(storage: str) -> str:
    """The same file as *storage*, spelled relative to the working directory."""
    path = storage[len("sqlite:///") :]
    return "sqlite:///" + os.path.relpath(path)


@pytest.mark.parametrize(
    "target",
    ["study_name", "url_string", "other_storage_object", "memory_plus_storage"],
)
def test_different_target_refused(tmp_path: Path, target: str) -> None:
    storage: Any = url(tmp_path)
    if target == "other_storage_object":
        storage = RDBStorage(storage)
    model = _tuned() if target == "memory_plus_storage" else _tuned(storage, "s")
    before = (snapshot_study(model._study), snapshot_model(model))
    kwargs: dict[str, Any] = {
        "study_name": {"study_name": "other"}.get(target, "s"),
        "storage": {
            "url_string": _relative_spelling(url(tmp_path)),
            "other_storage_object": RDBStorage(url(tmp_path)),
            "memory_plus_storage": url(tmp_path),
        }.get(target, storage),
    }
    error = _refused(model, resume=True, **kwargs)
    assert error.context["surface"] == "tuning.study"
    assert error.context["held_study_known"] is True
    assert (snapshot_study(model._study), snapshot_model(model)) == before


def test_pre_record_model_target_is_unknown_not_in_memory(tmp_path: Path) -> None:
    """Row 5 on a Model pickled before H-0122: it never remembered its target.

    Such a Model lacks ``_tuning_storage`` / ``_tuning_study_name``; the refusal
    says the held study is unknown instead of calling it in-memory.
    """
    storage = url(tmp_path)
    model = _tuned(storage, "s")
    del model._tuning_storage
    del model._tuning_study_name
    error = _refused(model, resume=True, storage=storage, study_name="s")
    assert error.context["surface"] == "tuning.study"
    assert error.context["held_study_known"] is False
    assert error.context["held_study_name"] is None
    assert "<in-memory>" not in error.user_message
    assert "predates" in error.user_message


# ---------------------------------------------------------------------------
# resume=True with no study in memory (criterion 14)
# ---------------------------------------------------------------------------


def test_reattach_existing(tmp_path: Path) -> None:
    storage = url(tmp_path)
    _tuned(storage, "s")
    fresh = Model(fast_config(), data=frame())
    result = fresh.tune(resume=True, storage=storage, study_name="s")
    study = load(storage, "s")
    assert len(study.trials) == 4
    assert [r.round for r in result.rounds] == [1, 2]
    assert [t.round for t in result.trials] == [1, 1, 2, 2]
    assert round_key(2) in study.user_attrs
    assert fresh._round_number == 2


def test_resume_creates_missing_study_with_info_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """U3: the study is created, announced at INFO, and starts at round 1."""
    storage = url(tmp_path)
    model = Model(fast_config(), data=frame())
    with caplog.at_level(logging.INFO, logger="lizyml"):
        result = model.tune(resume=True, storage=storage, study_name="new")
    created = [r for r in caplog.records if "tune.study_created" in r.getMessage()]
    assert len(created) == 1
    assert "study_name=new" in created[0].getMessage()
    attrs = lizyml_attrs(load(storage, "new"))
    assert STUDY_KEY in attrs
    assert round_key(1) in attrs
    assert [r.round for r in result.rounds] == [1]


def test_resume_without_study_or_storage_tuning_failed() -> None:
    model = Model(fast_config(), data=frame())
    error = _refused(model, ErrorCode.TUNING_FAILED, resume=True)
    assert error.context == {"resume": True, "round_number": 0, "storage_given": False}
    assert model._study is None


def test_after_load_resume_without_storage_tuning_failed(tmp_path: Path) -> None:
    """Criterion 24: a loaded Model carries no study, so there is nothing to resume."""
    model = _tuned()
    model.fit()
    loaded = Model.load(model.export(tmp_path / "export"))
    _refused(loaded, ErrorCode.TUNING_FAILED, data=frame(), resume=True)


def test_reattached_same_target_continues_and_name_change_refused(
    tmp_path: Path,
) -> None:
    """Criterion 25: a reattach remembers the target it reattached to."""
    storage = url(tmp_path)
    _tuned(storage, "s")
    fresh = Model(fast_config(), data=frame())
    fresh.tune(resume=True, storage=storage, study_name="s")
    result = fresh.tune(resume=True, storage=storage, study_name="s")
    assert [r.round for r in result.rounds] == [1, 2, 3]
    before = (snapshot_study(storage, "s"), snapshot_model(fresh))
    _refused(fresh, resume=True, storage=storage, study_name="renamed")
    assert (snapshot_study(storage, "s"), snapshot_model(fresh)) == before


# ---------------------------------------------------------------------------
# Criterion 14, the space= rows (PR 2): a call that creates a study
# ---------------------------------------------------------------------------

SPACE_ARG = {"learning_rate": {"type": "float", "low": 0.05, "high": 0.2}}


@pytest.mark.parametrize("where", ["memory", "storage_absent"])
def test_space_on_create_refused(tmp_path: Path, where: str) -> None:
    """``resume=False`` + ``space=``: refused before any storage contact.

    The tuned Model shows its state is untouched; the storage file is never
    created, because the refusal comes before the storage is read (D-c).
    """
    model = _tuned()
    before = (snapshot_study(model._study), snapshot_model(model))
    db = tmp_path / "s.db"
    kwargs: dict[str, Any] = {"space": SPACE_ARG}
    if where == "storage_absent":
        kwargs.update(storage=url(tmp_path), study_name="new")
    error = _refused(model, **kwargs)
    assert error.context["surface"] == "tune(space=)"
    assert error.context["reason"] == "resume_false"
    assert (snapshot_study(model._study), snapshot_model(model)) == before
    assert not db.exists()


def test_space_on_u3_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """U3 + ``space=``: the study is not created and nothing is announced.

    Locating the study lists the storage's studies, which may create the
    SQLite schema (design D-c, R7): asserted on the study names, not the file.
    """
    storage = url(tmp_path)
    _tuned(storage, "other")
    model = Model(fast_config(), data=frame())
    before = (snapshot_study(storage, "other"), snapshot_model(model))
    with caplog.at_level(logging.INFO, logger="lizyml"):
        error = _refused(
            model, resume=True, storage=storage, study_name="new", space=SPACE_ARG
        )
    assert error.context["reason"] == "study_absent"
    assert "new" not in optuna.get_all_study_names(storage)
    assert "tune.study_created" not in caplog.text
    assert (snapshot_study(storage, "other"), snapshot_model(model)) == before
