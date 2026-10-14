"""H-0122 §5.3, §6, §7.1: the lineage a study carries, and what reads it.

Criteria 5 (an unchanged declaration inherits the expanded range), 15 (no
enqueued re-evaluation of the previous best), 23 (a new study restarts the
lineage), 30.2 (expansion reads the study's best, not the Model's result) and
30.3 (``fixed`` comes from the study, not the Model field).

The §7.4 states a stop between two write positions leaves, and what
``resume=True`` does with each: criterion 15 (stopped before / after
``lizyml:round:<k>``), 19 without ``space=`` (E, N, X, an objective crash) and
32 (the ``:end`` record). A crash inside an open round is in
``test_resume_open_round.py``.
"""

from __future__ import annotations

import dataclasses
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import pytest

from lizyml import Model
from lizyml.core._tuning_plan import expand_for_next_round
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import FloatDim, IntDim
from lizyml.tuning.lineage import STUDY_KEY, classify_study, end_key, round_key
from tests.test_tuning._resume_helpers import (
    INCONSISTENT,
    RATIO,
    ProcessDeath,
    admission_refused_config,
    die_at_attr,
    die_in_objective,
    fast_config,
    frame,
    lizyml_attrs,
    load,
    snapshot_model,
    snapshot_study,
    study_record_for,
    url,
)


def _edge_config(mode: str = "replace") -> dict[str, Any]:
    """``num_leaves`` in [4, 5]: every best sits on an edge, so it expands."""
    cfg = fast_config(mode=mode)
    cfg["tuning"]["optuna"]["space"]["num_leaves"] = {
        "type": "int",
        "low": 4,
        "high": 5,
    }
    return cfg


def _records(study: Any) -> Any:
    lineage = classify_study(study.user_attrs, study.trials)
    assert lineage.state == "completed", lineage
    return {record.round: record for record in lineage.rounds}


def _dim(dims: Any, name: str) -> Any:
    return next(d for d in dims if d.name == name)


# ---------------------------------------------------------------------------
# Criterion 5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["memory", "reattach"])
def test_unchanged_declaration_inherits_expanded_range(
    tmp_path: Path, route: str
) -> None:
    storage = url(tmp_path) if route == "reattach" else None
    name = "s" if route == "reattach" else None
    model = Model(_edge_config(), data=frame())
    model.tune(storage=storage, study_name=name)
    model.tune(resume=True, expand_boundary=True)
    if route == "reattach":
        model = Model(_edge_config(), data=frame())
        model.tune(resume=True, storage=storage, study_name=name, expand_boundary=False)
    else:
        model.tune(resume=True, expand_boundary=False)
    study = load(storage, "s") if route == "reattach" else model._study
    records = _records(study)
    assert "num_leaves" in records[2].boundary_report.expanded_names
    leaves_1 = _dim(records[1].effective, "num_leaves")
    leaves_2 = _dim(records[2].effective, "num_leaves")
    assert (leaves_2.low, leaves_2.high) != (leaves_1.low, leaves_1.high)
    # Round 3 declared nothing new, so it starts from round 2's effective space.
    assert records[3].effective == records[2].effective
    assert records[3].declared == records[1].declared
    assert model._space == list(records[2].effective)


# ---------------------------------------------------------------------------
# Criterion 15 (the part without `space=`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["memory", "reattach"])
def test_resume_round_has_no_enqueued_trial(tmp_path: Path, route: str) -> None:
    storage = url(tmp_path) if route == "reattach" else None
    name = "s" if route == "reattach" else None
    model = Model(fast_config(n_trials=2), data=frame())
    model.tune(storage=storage, study_name=name)
    if route == "reattach":
        model = Model(fast_config(n_trials=2), data=frame())
    result = model.tune(resume=True, storage=storage, study_name=name, n_trials=3)
    study = model._study
    round_2 = [t for t in study.trials if t.number >= 2]
    assert len(round_2) == 3
    # Optuna marks an enqueued trial with `fixed_params` in its system attrs.
    assert not [t.number for t in round_2 if "fixed_params" in t.system_attrs]
    assert [t.round for t in result.trials] == [1, 1, 2, 2, 2]
    assert result.best_score == study.best_value
    assert result.best_params == study.best_params


# ---------------------------------------------------------------------------
# Criterion 23
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["memory", "new_name"])
def test_resume_false_new_study_restarts_lineage(tmp_path: Path, target: str) -> None:
    storage = url(tmp_path)
    model = Model(fast_config(), data=frame())
    model.tune(storage=storage, study_name="first")
    model.tune(resume=True)
    assert model._round_number == 2
    kwargs = {} if target == "memory" else {"storage": storage, "study_name": "second"}
    result = model.tune(**kwargs)
    assert [r.round for r in result.rounds] == [1]
    assert {t.round for t in result.trials} == {1}
    assert len(result.trials) == 2
    assert model._round_number == 1
    assert round_key(1) in model._study.user_attrs
    assert round_key(2) not in model._study.user_attrs
    held = (model._tuning_storage, model._tuning_study_name)
    assert held == ((None, None) if target == "memory" else (storage, "second"))


# ---------------------------------------------------------------------------
# Criterion 30.2
# ---------------------------------------------------------------------------


def test_loaded_model_expands_from_study_best(tmp_path: Path) -> None:
    """A loaded Model's tuning result belongs to another study; expansion
    reads the reattached study's best COMPLETE trial instead (§5.3)."""
    storage = url(tmp_path)
    Model(_edge_config(), data=frame()).tune(storage=storage, study_name="A")
    best_a = load(storage, "A").best_params

    other = Model(_edge_config(), data=frame())
    other.tune()
    other.fit()
    loaded = Model.load(other.export(tmp_path / "export"))
    # Make "another study's result" unmistakable: off-range values that would
    # expand nothing, or something else, if they were read.
    assert loaded._tuning_result is not None
    loaded._tuning_result = dataclasses.replace(
        loaded._tuning_result,
        best_model_params={"num_leaves": 1000, "learning_rate": 0.17},
    )
    result = loaded.tune(
        data=frame(), resume=True, storage=storage, study_name="A", expand_boundary=True
    )
    report = result.boundary_report
    assert report is not None
    assert {s.name: s.best_value for s in report.dims} == best_a
    assert "num_leaves" in report.expanded_names


# ---------------------------------------------------------------------------
# Criterion 30.3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["cleared", "overwritten", "widget_transfer"])
def test_fixed_from_study_not_model_field(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    cfg = fast_config(mode="merge")
    model = Model(cfg, data=frame())
    model.tune()
    study_fixed = dict(model._study.user_attrs["lizyml:study"]["fixed"])
    assert study_fixed["first_metric_only"] is True
    if variant == "cleared":
        model._tuning_fixed_params = None
    elif variant == "overwritten":
        model._tuning_fixed_params = {"first_metric_only": False}
    else:
        # LizyML-Widget carries these across processes, not the fixed field.
        carried = pickle.loads(
            pickle.dumps(
                (
                    model._study,
                    model._space,
                    model._used_default_space,
                    model._round_number,
                    model._rounds,
                )
            )
        )
        model = Model(cfg, data=frame())
        (
            model._study,
            model._space,
            model._used_default_space,
            model._round_number,
            model._rounds,
        ) = carried
    seen: list[Any] = []
    real = model._build_train_components

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["model_params"].get("first_metric_only"))
        return real(*args, **kwargs)

    monkeypatch.setattr(model, "_build_train_components", spy)
    model.tune(resume=True)
    assert seen and all(value is True for value in seen), seen
    assert model._tuning_fixed_params == study_fixed


# ---------------------------------------------------------------------------
# The §5.3 hook PR 2 fills (design R12)
# ---------------------------------------------------------------------------


def test_frozen_dimensions_are_reported_unexpanded() -> None:
    previous = (
        IntDim("num_leaves", 4, 5, min_allowed=2, max_allowed=131072),
        FloatDim("learning_rate", 0.05, 0.3, log=True),
    )
    best = {"num_leaves": 5, "learning_rate": 0.299}
    open_space, open_report = expand_for_next_round(previous, best, 0.05)
    assert set(open_report.expanded_names) == {"num_leaves", "learning_rate"}
    effective, report = expand_for_next_round(
        previous, best, 0.05, frozen_names=frozenset({"num_leaves"})
    )
    assert report.expanded_names == ("learning_rate",)
    frozen = next(s for s in report.dims if s.name == "num_leaves")
    assert (frozen.expanded, frozen.new_low, frozen.new_high) == (False, None, None)
    assert frozen.edge == "upper"
    assert effective[0] == previous[0]
    assert effective[1] == open_space[1] != previous[1]


# ---------------------------------------------------------------------------
# §7.4: what a stop between two write positions leaves
# ---------------------------------------------------------------------------


def _state(storage: str) -> str:
    study = load(storage, "s")
    return classify_study(study.user_attrs, study.trials).state


def _summary(result: Any) -> tuple[Any, Any]:
    return (
        [(r.round, r.n_trials) for r in result.rounds],
        [t.round for t in result.trials],
    )


def _reattach(cfg: dict[str, Any], storage: str, **kwargs: Any) -> Any:
    return Model(cfg, data=frame()).tune(
        resume=True, storage=storage, study_name="s", **kwargs
    )


# Criterion 15 (the stop part)


def test_stop_before_round_record_is_plain_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    model = Model(fast_config(), data=frame())
    model.tune(storage=storage, study_name="s")
    die_at_attr(monkeypatch, round_key(2))
    with pytest.raises(ProcessDeath):
        model.tune(resume=True, n_trials=3)
    monkeypatch.undo()
    assert round_key(2) not in load(storage, "s").user_attrs
    assert len(load(storage, "s").trials) == 2
    assert _state(storage) == "completed"
    # Nothing of round 2 was written, so it starts afresh: any n_trials.
    result = _reattach(fast_config(), storage, n_trials=4)
    assert _summary(result) == ([(1, 2), (2, 4)], [1, 1, 2, 2, 2, 2])


def test_stop_after_round_record_is_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    model = Model(fast_config(), data=frame())
    model.tune(storage=storage, study_name="s")

    def die(*args: Any, **kwargs: Any) -> None:
        raise ProcessDeath("died before the round's first trial")

    monkeypatch.setattr(model, "_run_tune_round", die)
    with pytest.raises(ProcessDeath):
        model.tune(resume=True, n_trials=3)
    monkeypatch.undo()
    stored = load(storage, "s")
    assert round_key(2) in stored.user_attrs
    assert end_key(2) not in stored.user_attrs
    assert len(stored.trials) == 2
    assert _state(storage) == "open"
    # The record's budget (3), not the config's (2).
    with pytest.warns(UserWarning, match="n_trials"):
        result = _reattach(fast_config(), storage)
    assert _summary(result) == ([(1, 2), (2, 3)], [1, 1, 2, 2, 2])


# Criterion 19 (without `space=`)


def test_stop_before_study_record_is_uninitialised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    die_at_attr(monkeypatch, STUDY_KEY)
    with pytest.raises(ProcessDeath):
        Model(fast_config(), data=frame()).tune(storage=storage, study_name="s")
    monkeypatch.undo()
    stored = load(storage, "s")
    assert stored.trials == []
    assert lizyml_attrs(stored) == {}
    assert _state(storage) == "uninitialized"
    result = _reattach(fast_config(), storage)
    attrs = lizyml_attrs(load(storage, "s"))
    assert {STUDY_KEY, round_key(1), end_key(1)} <= set(attrs)
    assert _summary(result) == ([(1, 2)], [1, 1])


def test_no_round_starts_round1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    die_at_attr(monkeypatch, round_key(1))
    with pytest.raises(ProcessDeath):
        Model(fast_config(), data=frame()).tune(storage=storage, study_name="s")
    monkeypatch.undo()
    stored = load(storage, "s")
    assert stored.trials == []
    assert set(lizyml_attrs(stored)) == {STUDY_KEY}
    assert _state(storage) == "no_round"
    record = stored.user_attrs[STUDY_KEY]
    result = _reattach(fast_config(), storage)
    stored = load(storage, "s")
    assert stored.user_attrs[STUDY_KEY] == record  # written once, not again
    assert round_key(1) in stored.user_attrs
    assert _summary(result) == ([(1, 2)], [1, 1])


@pytest.mark.parametrize(
    ("key", "state"),
    [(STUDY_KEY, "uninitialized"), (round_key(1), "no_round")],
    ids=["uninitialised", "no_round"],
)
def test_space_on_e_or_n_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, state: str
) -> None:
    """Criterion 19 (``space=``): states E and N start round 1 from the
    config, so ``space=`` is refused, and neither ``lizyml:study`` (E) nor a
    round record (N) is written (INV-R4)."""
    storage = url(tmp_path)
    die_at_attr(monkeypatch, key)
    with pytest.raises(ProcessDeath):
        Model(fast_config(), data=frame()).tune(storage=storage, study_name="s")
    monkeypatch.undo()
    assert _state(storage) == state
    model = Model(fast_config(), data=frame())
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with pytest.raises(LizyMLError) as caught:
        model.tune(
            resume=True,
            storage=storage,
            study_name="s",
            space={"learning_rate": {"type": "float", "low": 0.05, "high": 0.2}},
        )
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert caught.value.context["surface"] == "tune(space=)"
    assert caught.value.context["reason"] == state
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before
    assert _state(storage) == state


@pytest.fixture(scope="module")
def two_rounds(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A stored study in state C with two rounds."""
    root = tmp_path_factory.mktemp("two_rounds")
    storage = url(root)
    Model(fast_config(), data=frame()).tune(storage=storage, study_name="s")
    _reattach(fast_config(), storage)
    return root / "s.db"


@pytest.mark.parametrize("corruption", sorted(INCONSISTENT))
def test_inconsistent_refused(
    two_rounds: Path, tmp_path: Path, corruption: str
) -> None:
    shutil.copy(two_rounds, tmp_path / "s.db")
    storage = url(tmp_path)
    INCONSISTENT[corruption](storage)
    assert _state(storage) == "inconsistent"
    model = Model(fast_config(), data=frame())
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, storage=storage, study_name="s")
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert caught.value.context["state"] == "inconsistent"
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before


@pytest.mark.parametrize("state", ["uninitialized", "no_round"])
def test_admission_failure_writes_nothing(tmp_path: Path, state: str) -> None:
    """E and N: the H-0121 refusal comes before ``lizyml:study`` / round 1."""
    storage = url(tmp_path)
    cfg = admission_refused_config()
    data = frame("binary")
    study = optuna.create_study(storage=storage, study_name="s", direction="minimize")
    if state == "no_round":
        study.set_user_attr(STUDY_KEY, study_record_for(cfg, data))
    assert _state(storage) == state
    model = Model(cfg, data=data)
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, storage=storage, study_name="s")
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert RATIO in caught.value.user_message
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before
    assert _state(storage) == state


def test_objective_crash_is_open_and_counts_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Optuna creates the trial RUNNING before it calls the objective."""
    storage = url(tmp_path)
    model = Model(fast_config(n_trials=3), data=frame())
    die_in_objective(monkeypatch, model, at_call=2)
    with pytest.raises(ProcessDeath):
        model.tune(storage=storage, study_name="s")
    monkeypatch.undo()
    assert [t.state.name for t in load(storage, "s").trials] == [
        "COMPLETE",
        "RUNNING",
    ]
    assert _state(storage) == "open"
    result = _reattach(fast_config(n_trials=3), storage)
    assert _summary(result) == ([(1, 3)], [1, 1, 1])
    assert [t.state for t in result.trials] == ["complete", "fail", "complete"]


# ---------------------------------------------------------------------------
# Criterion 32: the `:end` record
# ---------------------------------------------------------------------------


def test_end_record_contents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = url(tmp_path)
    Model(fast_config(), data=frame()).tune(storage=storage, study_name="ok")
    study = load(storage, "ok")
    end = study.user_attrs[end_key(1)]
    assert end["status"] == "completed"
    assert end["best_after"] == ["float", study.best_value]
    assert datetime.fromisoformat(end["ended_at"]).tzinfo is not None

    model = Model(fast_config(), data=frame())

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("every trial fails")

    monkeypatch.setattr(model, "_build_train_components", broken)
    with pytest.raises(LizyMLError) as caught:
        model.tune(storage=storage, study_name="bad")
    assert caught.value.code is ErrorCode.TUNING_FAILED
    study = load(storage, "bad")
    assert [t.state.name for t in study.trials] == ["FAIL", "FAIL"]
    end = study.user_attrs[end_key(1)]
    assert end["status"] == "failed"
    assert end["best_after"] == ["none", None]
    assert datetime.fromisoformat(end["ended_at"]).tzinfo is not None
