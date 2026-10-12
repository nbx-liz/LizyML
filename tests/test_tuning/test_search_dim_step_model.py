"""H-0124 at the Model level: every reader reports the value a trial trained on.

Policy 4 (criteria 3 and 5): the tune's best parameters and the ``fit`` that
overlays them, ``restore_tuning``, an exported and loaded Model, the next
round's plan (``best_value``) and the trial list (``TrialResult.params``,
``tuning_table``) read a trial's raw Optuna value through its own round's
dimensions. Policy 5 (criterion 6): an expanded stepped dimension is recorded
with its ``step``, and a round whose space is off the grid is refused before
its record is written. Policy 7 (criteria 7 and 8): an ``int`` log dimension's
"step > 1" flag cannot change within a study, other step changes can, and
``space_summary`` shows ``step``.

The trials are steered, not left to the sampler: ``Tuner.run`` enqueues raw
values chosen per round, and the objective still trains every trial (so what
LightGBM saw can be recorded) but scores it by a predicate on the raw values,
so the best trial is known. A log dimension's raw sample is continuous, so a
reader that skips ``recorded_params`` shows ``0.41`` instead of ``0.4``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lizyml import Model
from lizyml.core import _tuning_plan
from lizyml.core._tuning_space import refuse_unchangeable, space_summary
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import FloatDim, IntDim
from lizyml.tuning.tuner import Tuner
from tests._train_spy import record_lightgbm_calls
from tests.test_tuning._lineage_fixtures import stored_round
from tests.test_tuning._resume_helpers import (
    fast_config,
    frame,
    load,
    round_record,
    snapshot_study,
    url,
)

CI = ErrorCode.CONFIG_INVALID

#: Round 1: a log grid of step 0.2. A raw 0.41 trains on 0.4.
LR_LOG_02 = {"type": "float", "low": 0.2, "high": 1.0, "log": True, "step": 0.2}
#: Round 2: the same dimension with step 0.25. A raw 0.41 would be 0.5 here.
LR_LOG_025 = {"type": "float", "low": 0.25, "high": 1.0, "log": True, "step": 0.25}
LEAVES = {"type": "int", "low": 4, "high": 8}


def _config(n_trials: int = 3, **space: Any) -> dict[str, Any]:
    cfg = fast_config(n_trials=n_trials)
    cfg["tuning"]["optuna"]["space"] = {"num_leaves": dict(LEAVES), **space}
    return cfg


def steer(
    monkeypatch: pytest.MonkeyPatch,
    enqueue: dict[int, list[dict[str, Any]]],
    is_best: Callable[[dict[str, Any]], bool],
) -> None:
    """Enqueue *enqueue[k]* in round k; score 0 for raw params *is_best* likes.

    The real objective still runs (every trial trains); only its score is
    replaced, so the study's best trial is the one *is_best* picks (the
    regression metric is minimised).
    """
    real_run = Tuner.run
    real_build = Model._build_tune_objective

    def run(self: Tuner, study: Any, objective: Any, **kwargs: Any) -> None:
        for params in enqueue.get(kwargs["round_number"], []):
            study.enqueue_trial(params)
        real_run(self, study, objective, **kwargs)

    def build(self: Model, **kwargs: Any) -> Callable[[Any], float]:
        objective = real_build(self, **kwargs)

        def steered(trial: Any) -> float:
            objective(trial)
            return 0.0 if is_best(trial.params) else 1.0

        return steered

    monkeypatch.setattr(Tuner, "run", run)
    monkeypatch.setattr(Model, "_build_tune_objective", build)


def _lr_is(value: float) -> Callable[[dict[str, Any]], bool]:
    return lambda params: params.get("learning_rate") == value


def _tune_041(
    monkeypatch: pytest.MonkeyPatch, storage: str | None
) -> tuple[Model, dict[str, Any]]:
    """Round 1 on ``LR_LOG_02`` whose best trial is trial 0, raw lr 0.41."""
    steer(monkeypatch, {1: [{"learning_rate": 0.41}]}, _lr_is(0.41))
    model = Model(_config(learning_rate=dict(LR_LOG_02)), data=frame())
    kwargs: dict[str, Any] = {}
    if storage is not None:
        kwargs = {"storage": storage, "study_name": "s"}
    model.tune(**kwargs)
    return model, kwargs


def _raw_lr(model: Model, number: int) -> Any:
    return model._study.trials[number].params["learning_rate"]


# ---------------------------------------------------------------------------
# Criterion 3: trained == recorded == TrialResult.params == best
# ---------------------------------------------------------------------------


def test_linear_step_reports_the_decimal_grid_point_everywhere() -> None:
    raw = 0.30000000000000004  # Optuna's own step-0.1 sample of 0.3
    spec = {"type": "float", "low": 0.1, "high": 0.7, "step": 0.1}
    model = Model(_config(learning_rate=spec), data=frame())
    with pytest.MonkeyPatch.context() as mp:
        steer(mp, {1: [{"learning_rate": raw}]}, _lr_is(raw))
        with record_lightgbm_calls() as calls:
            result = model.tune()
    assert _raw_lr(model, 0) == raw  # the study holds the binary neighbour
    trained = {p["learning_rate"] for p in calls["train_params"]}
    assert 0.3 in trained and raw not in trained
    assert result.trials[0].params["learning_rate"] == 0.3
    assert repr(result.best_model_params["learning_rate"]) == "0.3"
    assert model.tuning_table()["learning_rate"].iloc[0] == 0.3


# ---------------------------------------------------------------------------
# Criterion 5: every reader, with and without storage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stored", [True, False], ids=["storage", "memory"])
def test_tune_reports_recorded_best_and_trials(
    stored: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with record_lightgbm_calls() as calls:
        model, _ = _tune_041(monkeypatch, url(tmp_path) if stored else None)
    result = model._tuning_result
    assert _raw_lr(model, 0) == 0.41
    assert 0.4 in {p["learning_rate"] for p in calls["train_params"]}
    assert result.best_model_params["learning_rate"] == 0.4
    assert result.best_params["learning_rate"] == 0.4
    assert result.trials[0].params["learning_rate"] == 0.4
    for trial in result.trials:
        lr = trial.params["learning_rate"]
        assert lr == round(lr, 1) and round(lr / 0.2, 9) % 1 == 0, lr
    assert model.tuning_table()["learning_rate"].iloc[0] == 0.4


def test_fit_and_export_use_the_recorded_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = _tune_041(monkeypatch, None)
    with record_lightgbm_calls() as calls:
        model.fit()
    assert {p["learning_rate"] for p in calls["train_params"]} == {0.4}
    loaded = Model.load(model.export(tmp_path / "export"))
    assert loaded._tuning_result.best_model_params["learning_rate"] == 0.4


def test_restore_tuning_reports_the_recorded_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    _tune_041(monkeypatch, storage)
    other = Model(_config(learning_rate=dict(LR_LOG_02)), data=frame())
    restored = other.restore_tuning(storage, "s")
    assert restored.best_model_params["learning_rate"] == 0.4
    assert restored.trials[0].params["learning_rate"] == 0.4


def _two_rounds(
    monkeypatch: pytest.MonkeyPatch, storage: str | None
) -> tuple[Model, dict[str, Any]]:
    """Round 1 step 0.2 (best raw 0.41), round 2 the same dim with step 0.25."""
    model, kwargs = _tune_041(monkeypatch, storage)
    space = {"num_leaves": dict(LEAVES), "learning_rate": dict(LR_LOG_025)}
    model.tune(resume=True, space=space, n_trials=2, expand_boundary=False, **kwargs)
    return model, kwargs


@pytest.mark.parametrize("stored", [True, False], ids=["storage", "memory"])
def test_an_earlier_round_best_is_read_with_its_own_round_step(
    stored: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path) if stored else None
    model, kwargs = _two_rounds(monkeypatch, storage)
    result = model._tuning_result
    assert model._study.best_trial.number == 0
    # Round 1's step 0.2 makes raw 0.41 the value 0.4; round 2's 0.25 would
    # make it 0.5.
    assert result.best_model_params["learning_rate"] == 0.4
    assert result.trials[0].params["learning_rate"] == 0.4
    for trial in result.trials:
        if trial.round == 2:
            lr = trial.params["learning_rate"]
            assert lr in {0.25, 0.5, 0.75, 1.0}, lr
    # The next round's plan reads the same best: its boundary report holds
    # 0.4 as the best value of the dimension.
    model.tune(resume=True, n_trials=1, expand_boundary=True, **kwargs)
    report = model._tuning_result.boundary_report
    assert report is not None
    best_values = {s.name: s.best_value for s in report.dims}
    assert best_values["learning_rate"] == 0.4
    if storage is not None:
        assert round_record(storage, 3).best_before == 0.0
        restored = Model(_config(learning_rate=dict(LR_LOG_02)), data=frame())
        again = restored.restore_tuning(storage, "s")
        assert again.best_model_params["learning_rate"] == 0.4
        assert again.trials[0].params["learning_rate"] == 0.4


# ---------------------------------------------------------------------------
# Criterion 4 at the Model level: an int log step is reported as an int
# ---------------------------------------------------------------------------


def test_int_log_step_is_reported_as_an_int_grid_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaves = {"type": "int", "low": 2, "high": 64, "log": True, "step": 2}
    cfg = fast_config(n_trials=3)
    cfg["tuning"]["optuna"]["space"] = {"num_leaves": leaves}
    steer(monkeypatch, {1: [{"num_leaves": 2.9}]}, lambda p: p["num_leaves"] == 2.9)
    model = Model(cfg, data=frame())
    with record_lightgbm_calls() as calls:
        result = model.tune()
    assert model._study.trials[0].params["num_leaves"] == 2.9
    assert 2 in {p["num_leaves"] for p in calls["train_params"]}
    assert result.best_model_params["num_leaves"] == 2
    assert type(result.best_model_params["num_leaves"]) is int
    for trial in result.trials:
        n = trial.params["num_leaves"]
        assert type(n) is int and n % 2 == 0, n


# ---------------------------------------------------------------------------
# Criterion 6: expansion keeps the step; the round is checked before its record
# ---------------------------------------------------------------------------


def test_expanded_dim_is_recorded_with_its_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    spec = {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}
    steer(monkeypatch, {1: [{"learning_rate": 0.5}]}, _lr_is(0.5))
    model = Model(_config(learning_rate=spec), data=frame())
    model.tune(storage=storage, study_name="s")
    model.tune(
        resume=True, storage=storage, study_name="s", n_trials=1, expand_boundary=True
    )
    (lr,) = [d for d in round_record(storage, 2).effective if d.name == "learning_rate"]
    assert lr.step == 0.1
    assert (lr.low, lr.high) == (0.1, 0.9)


def test_off_grid_expansion_is_refused_before_the_round_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    spec = {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}
    steer(monkeypatch, {1: [{"learning_rate": 0.5}]}, _lr_is(0.5))
    model = Model(_config(learning_rate=spec), data=frame())
    model.tune(storage=storage, study_name="s")
    real = _tuning_plan.expand_for_next_round

    def off_grid(*args: Any, **kwargs: Any) -> Any:
        dims, report = real(*args, **kwargs)
        return (
            tuple(
                dataclasses.replace(d, high=0.95) if d.name == "learning_rate" else d
                for d in dims
            ),
            report,
        )

    monkeypatch.setattr(_tuning_plan, "expand_for_next_round", off_grid)
    before = snapshot_study(storage, "s")
    with pytest.raises(LizyMLError) as caught:
        model.tune(
            resume=True,
            storage=storage,
            study_name="s",
            n_trials=1,
            expand_boundary=True,
        )
    assert caught.value.code is CI
    assert caught.value.context["reason"] == "bound_off_grid"
    assert caught.value.context["dim"] == "learning_rate"
    assert snapshot_study(storage, "s") == before
    assert "lizyml:round:2" not in load(storage, "s").user_attrs


# ---------------------------------------------------------------------------
# Criterion 8: the int log "step > 1" flag cannot change; other steps can
# ---------------------------------------------------------------------------


def _leaves_log(step: int | None) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": "int", "low": 4, "high": 64, "log": True}
    if step is not None:
        spec["step"] = step
    return spec


@pytest.mark.parametrize(
    ("before", "after"),
    [(1, 2), (2, 1), (None, 2), (2, None)],
    ids=["1_to_2", "2_to_1", "none_to_2", "2_to_none"],
)
def test_int_log_step_flag_change_is_refused_before_the_round(
    before: int | None, after: int | None, tmp_path: Path
) -> None:
    storage = url(tmp_path)
    cfg = fast_config(n_trials=1)
    cfg["tuning"]["optuna"]["space"] = {"num_leaves": _leaves_log(before)}
    model = Model(cfg, data=frame())
    model.tune(storage=storage, study_name="s")
    snapshot = snapshot_study(storage, "s")
    with pytest.raises(LizyMLError) as caught:
        model.tune(
            resume=True,
            storage=storage,
            study_name="s",
            space={"num_leaves": _leaves_log(after)},
        )
    assert caught.value.code is CI
    assert caught.value.context["changed"] == "step"
    assert "step" in caught.value.user_message
    assert snapshot_study(storage, "s") == snapshot


@pytest.mark.parametrize(("before", "after"), [(None, 1), (1, None)])
def test_int_log_step_one_is_no_step(
    before: int | None, after: int | None, tmp_path: Path
) -> None:
    storage = url(tmp_path)
    cfg = fast_config(n_trials=1)
    cfg["tuning"]["optuna"]["space"] = {"num_leaves": _leaves_log(before)}
    model = Model(cfg, data=frame())
    model.tune(storage=storage, study_name="s")
    model.tune(
        resume=True,
        storage=storage,
        study_name="s",
        n_trials=1,
        space={"num_leaves": _leaves_log(after)},
    )
    assert model._round_number == 2


def test_linear_step_change_is_accepted_and_read_per_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = 0.30000000000000004
    spec = {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}
    steer(monkeypatch, {1: [{"learning_rate": raw}]}, _lr_is(raw))
    model = Model(_config(learning_rate=spec), data=frame())
    model.tune()
    changed = {"type": "float", "low": 0.25, "high": 0.5, "step": 0.25}
    result = model.tune(
        resume=True,
        n_trials=2,
        expand_boundary=False,
        space={"num_leaves": dict(LEAVES), "learning_rate": changed},
    )
    # Round 1's step 0.1 reads the raw value as 0.3; round 2's 0.25 would
    # read it as 0.25.
    assert result.trials[0].params["learning_rate"] == 0.3
    assert result.best_model_params["learning_rate"] == 0.3
    for trial in result.trials:
        if trial.round == 2:
            assert trial.params["learning_rate"] in {0.25, 0.5}


def test_refuse_unchangeable_names_step_for_the_int_log_flag() -> None:
    recorded = IntDim("num_leaves", 4, 64, log=True, step=1)
    record = _stored_round(recorded)
    given = IntDim("num_leaves", 4, 64, log=True, step=2)
    with pytest.raises(LizyMLError) as caught:
        refuse_unchangeable([given], (record,), {})
    assert caught.value.context["changed"] == "step"
    assert "step=1" in caught.value.user_message
    assert "step=2" in caught.value.user_message
    for accepted in (
        IntDim("num_leaves", 4, 64, log=True),
        IntDim("num_leaves", 4, 64, log=True, step=1),
    ):
        refuse_unchangeable([accepted], (record,), {})
    linear = _stored_round(IntDim("n", 2, 64, step=2))
    refuse_unchangeable([IntDim("n", 4, 64, step=4)], (linear,), {})
    flog = _stored_round(FloatDim("lr", 0.2, 1.0, log=True, step=0.2))
    refuse_unchangeable([FloatDim("lr", 0.25, 1.0, log=True, step=0.25)], (flog,), {})


def _stored_round(dim: Any) -> Any:
    return dataclasses.replace(stored_round(1, 0), declared=(dim,), effective=(dim,))


# ---------------------------------------------------------------------------
# Criterion 7: a step-only U2 difference reads as two different spaces
# ---------------------------------------------------------------------------


def test_space_summary_shows_the_step() -> None:
    a = space_summary([FloatDim("lr", 0.1, 1.0, step=0.1)], {})
    b = space_summary([FloatDim("lr", 0.1, 1.0, step=0.2)], {})
    c = space_summary([FloatDim("lr", 0.1, 1.0)], {})
    assert len({tuple(a), tuple(b), tuple(c)}) == 3
    assert "step" in a[0] and "step" not in c[0]
