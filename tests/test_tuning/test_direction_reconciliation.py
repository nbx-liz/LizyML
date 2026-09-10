"""Objective orientation must govern new and resumed optimization studies."""

from unittest.mock import MagicMock, patch

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.metrics.registry import _TASK_METRICS, get_metric
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

PAIRS = [
    (task, metric)
    for task, metrics in _TASK_METRICS.items()
    for metric in sorted(metrics)
]


def inputs(task, metric):
    config = make_config(task, n_estimators=5, n_splits=2, tuning_n_trials=2)
    config["evaluation"] = {"metrics": [metric]}
    config["tuning"]["optuna"]["params"].pop("direction", None)
    config["tuning"]["optuna"]["space"] = {
        "learning_rate": {"type": "float", "low": 0.03, "high": 0.2}
    }
    maker = {
        "regression": make_regression_df,
        "binary": make_binary_df,
        "multiclass": make_multiclass_df,
    }[task]
    frame = maker(n=90)
    if task == "regression":
        frame["target"] = frame["target"].abs() + 1
    return config, frame


@pytest.mark.parametrize(("task", "metric"), PAIRS)
def test_inferred_direction_selects_correct_extremum(task, metric):
    config, frame = inputs(task, metric)
    model = Model(config, data=frame)
    result = model.tune()
    greater = get_metric(metric).greater_is_better
    expected = "maximize" if greater else "minimize"
    assert result.direction == expected
    assert model._study.direction.name.lower() == expected
    scores = [trial.value for trial in model._study.trials]
    assert result.best_score == (max(scores) if greater else min(scores))


@pytest.mark.parametrize(("task", "metric"), PAIRS)
def test_contradictory_direction_refused_before_study(task, metric):
    config, frame = inputs(task, metric)
    config["tuning"]["optuna"]["params"]["direction"] = (
        "minimize" if get_metric(metric).greater_is_better else "maximize"
    )
    with patch("optuna.create_study") as create:
        with pytest.raises(LizyMLError) as failure:
            Model(config, data=frame).tune()
        assert failure.value.code == ErrorCode.CONFIG_INVALID
        create.assert_not_called()


@pytest.mark.parametrize("task", list(_TASK_METRICS))
def test_default_metric_direction_and_dump_round_trip(task):
    from lizyml.config.schema import LizyMLConfig

    config, frame = inputs(task, "rmse" if task == "regression" else "logloss")
    config.pop("evaluation")
    parsed = LizyMLConfig.model_validate(config)
    dumped = parsed.model_dump()
    assert dumped["tuning"]["optuna"]["params"]["direction"] is None
    restored = LizyMLConfig.model_validate(dumped)
    assert Model(restored, data=frame).tune().direction == "minimize"


@pytest.mark.parametrize("requested", [None, "maximize"])
def test_parameterized_metric_round_trip_and_matching_direction(requested):
    from lizyml.config.schema import LizyMLConfig

    config, frame = inputs("binary", "auc")
    config["evaluation"]["metrics"] = [{"precision_at_k": {"k": 10}}]
    config["tuning"]["optuna"]["params"]["direction"] = requested
    parsed = LizyMLConfig.model_validate(config)
    restored = LizyMLConfig.model_validate_json(parsed.model_dump_json())
    assert Model(restored, data=frame).tune().direction == "maximize"


@pytest.mark.parametrize("persisted", [False, True])
def test_wrong_existing_study_refused_before_enqueue_or_objective(tmp_path, persisted):
    import optuna

    from lizyml.tuning.tuner import Tuner

    storage = f"sqlite:///{tmp_path / 'study.db'}" if persisted else None
    study = optuna.create_study(direction="minimize", storage=storage, study_name="old")
    tuner = Tuner(
        dims=[],
        n_trials=1,
        direction="maximize",
        storage=storage,
        study_name="old" if persisted else None,
    )
    called = MagicMock()
    with pytest.raises(LizyMLError) as failure:
        tuner.tune(
            called,
            metric_name="auc",
            study=None if persisted else study,
            enqueue_params={"x": 1},
        )
    assert failure.value.code == ErrorCode.CONFIG_INVALID
    called.assert_not_called()
    if persisted:
        study = optuna.load_study(storage=storage, study_name="old")
    assert study.trials == []
