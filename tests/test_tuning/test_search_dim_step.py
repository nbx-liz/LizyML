"""H-0124: a search dimension's ``step`` -- format, grid, sampling, rounding.

Criteria 1 (format, both surfaces, the ``bound_off_grid`` message range),
2 (the exact grid judgement and the ``step`` types), 3 (non-log sampling hands
training the grid value), 4 (log sampling: one rounding, half to even, and the
Optuna distribution kinds), 9 (``Tuner.tune`` refusals), and the pure reader
``recorded_params`` of policy 4. The Model-level readers of policy 4 (best
params, ``restore_tuning``, the next round's plan, ``tuning_table``) are wired
in the Model layer and tested there.
"""

from __future__ import annotations

import decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import optuna
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.tuning._step_grid import grid_value, recorded_params
from lizyml.tuning.search_space import parse_space, suggest_params
from lizyml.tuning.tuner import Tuner
from tests._train_spy import record_lightgbm_calls
from tests.test_tuning._resume_helpers import (
    fast_config,
    frame,
    round_record,
    url,
)

CI = ErrorCode.CONFIG_INVALID


def _refusal(spec: dict[str, Any], name: str = "d") -> LizyMLError:
    with pytest.raises(LizyMLError) as caught:
        parse_space({name: spec})
    assert caught.value.code is CI
    return caught.value


# ---------------------------------------------------------------------------
# Criterion 1: the format accepts step on both surfaces
# ---------------------------------------------------------------------------


def test_config_space_accepts_step() -> None:
    cfg = fast_config()
    cfg["tuning"]["optuna"]["space"]["learning_rate"] = {
        "type": "float",
        "low": 0.1,
        "high": 0.3,
        "step": 0.1,
    }
    cfg["tuning"]["optuna"]["space"]["num_leaves"] = {
        "type": "int",
        "low": 4,
        "high": 8,
        "step": 2,
    }
    model = Model(cfg, data=frame())
    model.tune()
    declared = {d.name: d for d in round_record(model._study, 1).declared}
    assert declared["learning_rate"] == FloatDim("learning_rate", 0.1, 0.3, step=0.1)
    assert declared["num_leaves"] == IntDim("num_leaves", 4, 8, step=2)


def test_space_arg_accepts_step() -> None:
    model = Model(fast_config(), data=frame())
    model.tune()
    space = {
        "num_leaves": {"type": "int", "low": 4, "high": 8, "step": 2},
        "learning_rate": {
            "type": "float",
            "low": 0.05,
            "high": 0.3,
            "log": True,
            "step": 0.05,
        },
    }
    model.tune(resume=True, space=space)
    declared = {d.name: d for d in round_record(model._study, 2).declared}
    assert declared["learning_rate"].step == 0.05  # type: ignore[union-attr]
    assert declared["num_leaves"].step == 2  # type: ignore[union-attr]


#: (spec, key, the range the message shows). Criterion 1's six cases.
OFF_GRID_MESSAGES = {
    "inner": (
        {"type": "float", "low": 0.15, "high": 1.0, "step": 0.1},
        "low",
        "[0.2, 1.0]",
    ),
    "inner_log": (
        {"type": "float", "low": 0.0001, "high": 0.01, "step": 0.001, "log": True},
        "low",
        "[0.001, 0.01]",
    ),
    "no_point": (
        {"type": "float", "low": 0.15, "high": 0.16, "step": 0.1},
        "low",
        "[0.1, 0.2]",
    ),
    "one_point": (
        {"type": "float", "low": 0.15, "high": 0.25, "step": 0.1},
        "low",
        "[0.1, 0.3]",
    ),
    "int_outer": ({"type": "int", "low": 3, "high": 4, "step": 5}, "low", "[0, 5]"),
    "log_outer_widened": (
        {"type": "float", "low": 0.0005, "high": 0.0008, "step": 0.001, "log": True},
        "low",
        "[0.001, 0.002]",
    ),
}


@pytest.mark.parametrize("case", list(OFF_GRID_MESSAGES))
def test_off_grid_message_shows_a_range_on_the_grid(case: str) -> None:
    spec, key, shown = OFF_GRID_MESSAGES[case]
    err = _refusal(spec)
    assert err.context["reason"] == "bound_off_grid"
    assert (err.context["dim"], err.context["key"]) == ("d", key)
    assert err.context["surface"] == "tuning.optuna.space"
    assert shown in err.user_message
    inside = case.startswith("inner")
    assert ("inside the declared range" in err.user_message) is inside
    assert ("encloses the declared range" in err.user_message) is not inside


def test_off_grid_high_names_high() -> None:
    err = _refusal({"type": "int", "low": 4, "high": 9, "step": 2})
    assert (err.context["reason"], err.context["key"]) == ("bound_off_grid", "high")
    assert "[4, 8]" in err.user_message


# ---------------------------------------------------------------------------
# Criterion 2: the grid judgement is exact; the step types
# ---------------------------------------------------------------------------

ON_GRID = {
    "tenths": ({"type": "float", "low": 0.1, "high": 0.3, "step": 0.1}, 0.1),
    "twentieths": ({"type": "float", "low": 0.5, "high": 1.0, "step": 0.05}, 0.05),
    "int_hundreds": ({"type": "int", "low": 500, "high": 2000, "step": 100}, 100),
    "numpy_float_step": (
        {"type": "float", "low": 0.5, "high": 1.0, "step": np.float64(0.05)},
        0.05,
    ),
    "str_float_step": (
        {"type": "float", "low": 0.5, "high": 1.0, "step": "0.05"},
        0.05,
    ),
    "numpy_int_step": (
        {"type": "int", "low": 500, "high": 2000, "step": np.int64(100)},
        100,
    ),
    "str_exponent_int_step": (
        {"type": "int", "low": 500, "high": 2000, "step": "1e2"},
        100,
    ),
    "log_int_step_one": (
        {"type": "int", "low": 2, "high": 64, "log": True, "step": 1},
        1,
    ),
}


@pytest.mark.parametrize("case", list(ON_GRID))
def test_on_grid_spec_is_accepted_with_a_plain_step(case: str) -> None:
    spec, step = ON_GRID[case]
    (dim,) = parse_space({"d": spec})
    assert isinstance(dim, (FloatDim, IntDim))
    assert dim.step == step
    assert type(dim.step) is (float if spec["type"] == "float" else int)


def test_off_grid_tenths_are_refused_with_the_inner_range() -> None:
    err = _refusal({"type": "float", "low": 0.15, "high": 1.0, "step": 0.1})
    assert "0.2" in err.user_message


def test_off_grid_int_is_refused() -> None:
    err = _refusal({"type": "int", "low": 3, "high": 10, "step": 2})
    assert (err.context["reason"], err.context["key"]) == ("bound_off_grid", "low")


#: (spec, reason). Each step the format refuses; key is always "step".
BAD_STEPS = {
    "bool": (
        {"type": "float", "low": 0.1, "high": 1.0, "step": True},
        "bound_not_a_number",
    ),
    "word": (
        {"type": "float", "low": 0.1, "high": 1.0, "step": "abc"},
        "bound_not_a_number",
    ),
    "nan": (
        {"type": "float", "low": 0.1, "high": 1.0, "step": float("nan")},
        "bound_not_finite",
    ),
    "zero": (
        {"type": "float", "low": 0.1, "high": 1.0, "step": 0},
        "step_not_positive",
    ),
    "negative": (
        {"type": "float", "low": 0.1, "high": 1.0, "step": -0.1},
        "step_not_positive",
    ),
    "int_fraction": (
        {"type": "int", "low": 0, "high": 9, "step": 1.5},
        "bound_not_integral",
    ),
    "int_zero": ({"type": "int", "low": 0, "high": 9, "step": 0}, "step_not_positive"),
    "int_bool": (
        {"type": "int", "low": 0, "high": 9, "step": True},
        "bound_not_a_number",
    ),
}


@pytest.mark.parametrize("case", list(BAD_STEPS))
def test_bad_step_is_refused(case: str) -> None:
    spec, reason = BAD_STEPS[case]
    err = _refusal(spec)
    assert (err.context["reason"], err.context["key"]) == (reason, "step")
    assert "step" in err.user_message


def test_large_bounds_with_a_small_step_are_exact_under_any_decimal_context() -> None:
    spec = {"type": "float", "low": 1e9, "high": 2e9, "step": 0.1, "log": True}
    (expected,) = parse_space({"d": spec})
    with decimal.localcontext() as ctx:
        ctx.prec = 6
        (dim,) = parse_space({"d": spec})
        off = spec | {"low": 1e9 + 0.05}
        err = _refusal(off)
    assert dim == expected == FloatDim("d", 1e9, 2e9, log=True, step=0.1)
    assert err.context["reason"] == "bound_off_grid"


# ---------------------------------------------------------------------------
# Rounding: grid_value and recorded_params (policies 3 and 4)
# ---------------------------------------------------------------------------


def test_grid_value_rounds_half_to_even() -> None:
    dim = FloatDim("x", 2.0, 8.0, log=True, step=2.0)
    assert grid_value(dim, 5.0) == 4.0  # 2.5 -> 2, not 3
    assert grid_value(dim, 7.0) == 8.0  # 3.5 -> 4


def test_grid_value_is_the_decimal_point_not_the_binary_neighbour() -> None:
    dim = FloatDim("x", 0.1, 1.0, step=0.1)
    assert grid_value(dim, 0.30000000000000004) == 0.3
    assert grid_value(dim, 0.7000000000000001) == 0.7


def test_grid_value_clamps_to_the_range() -> None:
    dim = FloatDim("x", 0.001, 0.1, log=True, step=0.001)
    assert grid_value(dim, 0.1004) == 0.1
    assert grid_value(dim, 0.001) == 0.001
    assert grid_value(IntDim("n", 2, 64, log=True, step=2), 64.9) == 64


def test_grid_value_of_an_int_dim_rounds_once() -> None:
    value = grid_value(IntDim("n", 2, 64, log=True, step=2), 2.9)
    assert value == 2  # 2.9 / 2 = 1.45 -> 1; rounding to 3 first would give 4
    assert type(value) is int


def test_recorded_params_rounds_stepped_dims_only() -> None:
    dims = [
        FloatDim("lr", 0.1, 1.0, step=0.1),
        FloatDim("ff", 0.1, 1.0),
        IntDim("n", 2, 64, log=True, step=2),
        CategoricalDim("b", ("gbdt", "dart")),
    ]
    raw = {"lr": 0.30000000000000004, "ff": 0.123, "n": 7.4, "b": "dart"}
    assert recorded_params(raw, dims) == {"lr": 0.3, "ff": 0.123, "n": 8, "b": "dart"}


def test_recorded_params_reads_a_subset_of_the_dims() -> None:
    """A failed trial may hold only some of its round's parameters."""
    dims = [FloatDim("lr", 0.1, 1.0, step=0.1), IntDim("n", 2, 8)]
    assert recorded_params({"lr": 0.2}, dims) == {"lr": 0.2}


def test_recorded_params_refuses_a_name_without_a_dimension() -> None:
    with pytest.raises(ValueError, match="'y'"):
        recorded_params({"x": 0.1, "y": 0.41}, [FloatDim("x", 0.0, 1.0, step=0.2)])


# ---------------------------------------------------------------------------
# Criterion 3: non-log sampling uses Optuna's step, training gets the grid point
# ---------------------------------------------------------------------------


def test_suggest_float_passes_step_and_returns_the_grid_value() -> None:
    trial = MagicMock()
    trial.suggest_float.return_value = 0.30000000000000004
    params = suggest_params(trial, [FloatDim("x", 0.1, 1.0, step=0.1)])
    trial.suggest_float.assert_called_once_with("x", 0.1, 1.0, log=False, step=0.1)
    assert params == {"x": 0.3}


def test_suggest_int_passes_step() -> None:
    trial = MagicMock()
    trial.suggest_int.return_value = 700
    params = suggest_params(trial, [IntDim("n", 500, 2000, step=100)])
    trial.suggest_int.assert_called_once_with("n", 500, 2000, log=False, step=100)
    assert params == {"n": 700}


def test_suggest_without_step_passes_no_step() -> None:
    trial = MagicMock()
    trial.suggest_float.return_value = 0.123
    trial.suggest_int.return_value = 7
    params = suggest_params(trial, [FloatDim("x", 0.1, 1.0), IntDim("n", 2, 8)])
    trial.suggest_float.assert_called_once_with("x", 0.1, 1.0, log=False)
    trial.suggest_int.assert_called_once_with("n", 2, 8, log=False)
    assert params == {"x": 0.123, "n": 7}


def test_model_trains_on_the_grid_points() -> None:
    cfg = fast_config(n_trials=6)
    cfg["tuning"]["optuna"]["space"]["learning_rate"] = {
        "type": "float",
        "low": 0.1,
        "high": 0.7,
        "step": 0.1,
    }
    grid = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}
    model = Model(cfg, data=frame())
    with record_lightgbm_calls() as calls:
        model.tune()
    trained = {p["learning_rate"] for p in calls["train_params"]}
    assert trained, "no training call was recorded"
    assert trained <= grid


# ---------------------------------------------------------------------------
# Criterion 4: log sampling, one rounding, Optuna distribution kinds
# ---------------------------------------------------------------------------


def _run(dims: list[Any], enqueued: list[dict[str, Any]], n: int | None = None) -> Any:
    """Run a study whose objective records what ``suggest_params`` returned."""
    seen: list[dict[str, Any]] = []
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    for params in enqueued:
        study.enqueue_trial(params)

    def objective(trial: Any) -> float:
        seen.append(suggest_params(trial, dims))
        return 0.0

    study.optimize(objective, n_trials=n if n is not None else len(enqueued))
    return study, seen


def test_float_log_step_rounds_half_to_even() -> None:
    dims = [FloatDim("x", 2.0, 8.0, log=True, step=2.0)]
    study, seen = _run(dims, [{"x": 5.0}])
    assert seen == [{"x": 4.0}]  # rounding up would give 6.0
    assert study.trials[0].distributions["x"] == optuna.distributions.FloatDistribution(
        2.0, 8.0, log=True
    )


def test_int_log_step_samples_a_float_and_rounds_once() -> None:
    dims = [IntDim("n", 2, 64, log=True, step=2)]
    study, seen = _run(dims, [{"n": 2.9}])
    assert seen == [{"n": 2}]  # an int sample (3) rounded again would be 4
    assert type(seen[0]["n"]) is int
    assert study.trials[0].distributions["n"] == optuna.distributions.FloatDistribution(
        2.0, 64.0, log=True
    )


def test_int_log_step_one_keeps_the_int_distribution() -> None:
    dims = [IntDim("n", 2, 64, log=True, step=1)]
    study, seen = _run(dims, [], n=3)
    assert all(type(p["n"]) is int for p in seen)
    assert study.trials[0].distributions["n"] == optuna.distributions.IntDistribution(
        2, 64, log=True
    )


def test_one_grid_point_sampled_twice_is_evaluated_twice() -> None:
    dims = [FloatDim("lr", 0.001, 0.1, log=True, step=0.001)]
    study, seen = _run(dims, [{"lr": 0.0011}, {"lr": 0.0012}])
    assert seen == [{"lr": 0.001}, {"lr": 0.001}]
    assert len(study.trials) == 2


def test_log_step_samples_are_grid_points_in_range() -> None:
    dims = [
        FloatDim("lr", 0.001, 0.1, log=True, step=0.001),
        IntDim("n", 2, 64, log=True, step=2),
    ]
    _, seen = _run(dims, [], n=25)
    for params in seen:
        lr = params["lr"]
        assert 0.001 <= lr <= 0.1
        assert lr == round(lr, 3)
        assert params["n"] % 2 == 0 and 2 <= params["n"] <= 64


def test_model_log_step_trains_on_grid_points() -> None:
    cfg = fast_config(n_trials=5)
    cfg["tuning"]["optuna"]["space"]["learning_rate"] = {
        "type": "float",
        "low": 0.001,
        "high": 0.1,
        "step": 0.001,
        "log": True,
    }
    model = Model(cfg, data=frame())
    with record_lightgbm_calls() as calls:
        model.tune()
    trained = [p["learning_rate"] for p in calls["train_params"]]
    assert trained
    assert all(0.001 <= v <= 0.1 and v == round(v, 3) for v in trained), trained


# ---------------------------------------------------------------------------
# Criterion 9: Tuner.tune refuses before any study exists
# ---------------------------------------------------------------------------

TUNER_REFUSED: dict[str, tuple[Any, str]] = {
    "float_log_step": (
        FloatDim("x", 0.001, 0.1, log=True, step=0.001),
        "log_with_step",
    ),
    "int_log_step_one": (IntDim("n", 2, 64, log=True, step=1), "log_with_step"),
    "int_log_step_two": (IntDim("n", 2, 64, log=True, step=2), "log_with_step"),
    "off_grid": (FloatDim("x", 0.15, 1.0, step=0.1), "bound_off_grid"),
    "step_zero": (FloatDim("x", 0, 1, step=0), "step_not_positive"),
    "step_nan": (FloatDim("x", 0, 1, step=float("nan")), "bound_not_finite"),
    "step_negative": (FloatDim("x", 0, 1, step=-0.1), "step_not_positive"),
    "step_bool": (FloatDim("x", 0, 1, step=True), "bound_not_a_number"),
    "int_step_fraction": (IntDim("n", 0, 10, step=1.5), "bound_not_integral"),  # type: ignore[arg-type]
}


def _never_called(trial: Any) -> float:
    raise AssertionError("the objective ran")


@pytest.mark.parametrize("case", list(TUNER_REFUSED))
def test_tuner_tune_refuses_before_creating_a_study(case: str, tmp_path: Path) -> None:
    dim, reason = TUNER_REFUSED[case]
    storage = url(tmp_path)
    tuner = Tuner(dims=[dim], n_trials=1, storage=storage, study_name="s")
    with pytest.raises(LizyMLError) as caught:
        tuner.tune(_never_called)
    err = caught.value
    assert err.code is CI
    assert err.context["reason"] == reason
    key = "low" if reason == "bound_off_grid" else "step"
    assert (err.context["dim"], err.context["key"]) == (dim.name, key)
    assert err.context["surface"] == "Tuner.tune"
    assert optuna.get_all_study_summaries(storage) == []


def test_tuner_init_accepts_a_log_step_dim_that_model_tune_runs() -> None:
    """Policy 6: the refusal is in ``tune``; ``Model.tune`` builds a Tuner too."""
    Tuner(dims=[IntDim("num_leaves", 4, 8, log=True, step=1)])
    cfg = fast_config()
    cfg["tuning"]["optuna"]["space"]["num_leaves"] = {
        "type": "int",
        "low": 4,
        "high": 8,
        "log": True,
        "step": 1,
    }
    cfg["tuning"]["optuna"]["space"]["learning_rate"] = {
        "type": "float",
        "low": 0.01,
        "high": 0.3,
        "log": True,
        "step": 0.01,
    }
    model = Model(cfg, data=frame())
    model.tune()
    assert model._study is not None


def test_tuner_tune_returns_the_raw_values_the_objective_sampled() -> None:
    dims = [FloatDim("x", 0.1, 1.0, step=0.1)]

    def objective(trial: Any) -> float:
        return float(trial.suggest_float("x", 0.1, 1.0))

    result, study = Tuner(dims=dims, n_trials=4, seed=1).tune(objective)
    raw = [t.params["x"] for t in study.trials]
    assert [t.params["x"] for t in result.trials] == raw
    assert result.best_model_params == {"x": study.best_params["x"]}
    assert any(v != round(v, 1) for v in raw)
