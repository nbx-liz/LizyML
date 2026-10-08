"""Refusals identify the winning input before LightGBM trains."""

from typing import Any
from unittest.mock import patch

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TuningResult
from lizyml.estimators.lgbm.adapter import LGBMAdapter, _pop_by_identity
from tests._helpers import make_binary_df, make_config


@pytest.mark.parametrize(
    "surface", ["model.params", "fit(params=)", "tuning best_model_params"]
)
@pytest.mark.parametrize(
    "name,value", [("application", "regression"), ("metrics", "rmse")]
)
def test_refusal_names_winning_input(surface: str, name: str, value: str) -> None:
    cfg = make_config("binary", n_estimators=2, n_splits=2)
    supplied = {name: value}
    if surface == "model.params":
        cfg["model"]["params"].update(supplied)
    model = Model(cfg, data=make_binary_df(n=60))
    if surface == "tuning best_model_params":
        model._tuning_result = TuningResult(
            best_model_params=supplied,
            best_smart_params={},
            best_training_params={},
            best_score=0.0,
            metric_name="auc",
            direction="maximize",
            trials=(),
            rounds=(),
        )
    with patch("lightgbm.train") as train, pytest.raises(LizyMLError) as caught:
        model.fit(params=supplied if surface == "fit(params=)" else None)
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert surface in caught.value.user_message
    assert caught.value.context["surface"] == surface
    assert caught.value.context["parameter"] == name
    train.assert_not_called()


@pytest.mark.parametrize("surface", ["model.params", "fit(params=)"])
@pytest.mark.parametrize(
    "pair",
    [
        {"objective": "binary", "application": "binary"},
        {"metric": "auc", "metrics": "auc"},
        {"n_estimators": 2, "num_iterations": 2},
    ],
)
def test_duplicate_is_refused_before_adapter(
    surface: str, pair: dict[str, Any]
) -> None:
    cfg = make_config("binary", n_estimators=2, n_splits=2)
    if surface == "model.params":
        cfg["model"]["params"].update(pair)
    model = Model(cfg, data=make_binary_df(n=60))
    with (
        patch(
            "lizyml.estimators.lgbm.adapter._pop_by_identity", wraps=_pop_by_identity
        ) as pop,
        pytest.raises(LizyMLError) as caught,
    ):
        model.fit(params=pair if surface == "fit(params=)" else None)
    assert surface in caught.value.user_message
    pop.assert_not_called()


@pytest.mark.parametrize(
    "pair",
    [
        {"seed": 7, "random_state": 7},
        {"seed": 7, "random_seed": 9},
        {"verbosity": -1, "verbose": -1},
        {"verbosity": 0, "verbose": -1},
    ],
)
def test_direct_adapter_refuses_duplicate_spellings(pair: dict[str, Any]) -> None:
    adapter = LGBMAdapter(task="binary", params=pair)
    with pytest.raises(LizyMLError) as caught:
        adapter._build_params()
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert "surface" not in caught.value.context


def test_valid_override_replaces_invalid_config_before_validation() -> None:
    cfg = make_config(
        "binary", n_estimators=2, n_splits=2, objective="regression", metric="rmse"
    )
    model = Model(cfg, data=make_binary_df(n=60))
    model.fit(params={"application": "binary", "metrics": "auc"})
    assert model.fit_result.models[0].get_native_model().params["objective"] == "binary"


def test_mixed_origins_do_not_blame_unrelated_override() -> None:
    cfg = make_config("binary", n_estimators=2, n_splits=2, metrics="rmse")
    model = Model(cfg, data=make_binary_df(n=60))
    with pytest.raises(LizyMLError) as caught:
        model.fit(params={"eta": 0.2})
    assert caught.value.context["surface"] == "model.params"
    assert caught.value.context["parameter"] == "metrics"


def test_tuning_validates_after_sampled_overlay() -> None:
    """A later valid trial objective must still replace an invalid base value."""
    cfg = make_config(
        "binary",
        n_estimators=2,
        n_splits=2,
        objective="regression",
        tuning_n_trials=1,
    )
    cfg["tuning"]["optuna"]["space_mode"] = "replace"
    cfg["tuning"]["optuna"]["space"] = {
        "application": {
            "type": "categorical",
            "choices": ["binary"],
            "category": "model",
        }
    }
    result = Model(cfg, data=make_binary_df(n=60)).tune()
    assert result.best_model_params == {"application": "binary"}


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("sentinel"),
        LizyMLError(ErrorCode.UNSUPPORTED_METRIC, "sentinel"),
    ],
)
def test_other_errors_propagate_unchanged(error: Exception) -> None:
    cfg = make_config("binary", n_estimators=2, n_splits=2, objective="binary")
    model = Model(cfg, data=make_binary_df(n=60))
    with (
        patch(
            "lizyml.estimators.lgbm.param_validation.check_objective_compatible",
            side_effect=error,
        ),
        pytest.raises(type(error)) as caught,
    ):
        model.fit()
    assert caught.value is error


def test_config_error_preserves_details_and_cause() -> None:
    cause = ValueError("root")
    error = LizyMLError(
        ErrorCode.CONFIG_INVALID,
        "sentinel",
        debug_message="debug",
        context={"task": "binary"},
        cause=cause,
    )
    cfg = make_config("binary", n_estimators=2, n_splits=2, objective="binary")
    model = Model(cfg, data=make_binary_df(n=60))
    with (
        patch(
            "lizyml.estimators.lgbm.param_validation.check_objective_compatible",
            side_effect=error,
        ),
        pytest.raises(LizyMLError) as caught,
    ):
        model.fit()
    assert caught.value.context["task"] == "binary"
    assert caught.value.debug_message == "debug"
    assert caught.value.__cause__ is error
    assert caught.value.cause is error
    assert caught.value.cause.cause is cause
