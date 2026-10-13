"""H-0125 policies 1-2 (#307): ``cross_entropy_lambda`` is not a binary objective.

LightGBM's ``cross_entropy_lambda`` predicts λ; the probability is
``1 - exp(-λ)`` and is not what LightGBM returns. LizyML used to accept it and
return λ as ``predict().proba``. It is refused as an objective with its own
reason, on every spelling of the parameter and on every layer that reaches a
booster (criterion 1); every objective still accepted for binary predicts a
probability (criterion 2). As a *metric* ``cross_entropy_lambda`` stays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.core.types.tuning_result import TuningResult
from lizyml.estimators.lgbm.adapter import LGBMAdapter
from lizyml.estimators.lgbm.defaults import TASK_COMPATIBLE_OBJECTIVES
from lizyml.estimators.lgbm.param_validation import check_objective_compatible
from lizyml.estimators.lgbm.provider import LGBMProvider
from tests._helpers import make_binary_df, make_config
from tests._train_spy import record_lightgbm_calls

#: Every LightGBM spelling of the ``objective`` parameter (LightGBM 4.x aliases).
SPELLINGS = ("objective", "objective_type", "app", "application", "loss")
REASON = "not_a_probability"


def _assert_refused(error: LizyMLError) -> None:
    assert error.code == ErrorCode.CONFIG_INVALID
    assert error.context["reason"] == REASON
    message = error.user_message
    assert "cross_entropy_lambda" in message
    assert "not a probability" in message
    assert "'binary'" in message and "'cross_entropy'" in message


def _learnable_binary(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = (2 * X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Policy 2: the rule and its message
# ---------------------------------------------------------------------------


def test_the_rule_names_the_reason_and_the_alternatives() -> None:
    with pytest.raises(LizyMLError) as caught:
        check_objective_compatible("binary", "cross_entropy_lambda")
    _assert_refused(caught.value)
    assert caught.value.context["objective"] == "cross_entropy_lambda"
    assert caught.value.context["valid_objectives"] == ["binary", "cross_entropy"]


def test_the_alias_value_keeps_the_generic_refusal() -> None:
    """``xentlambda`` was never in the table; its refusal is unchanged."""
    with pytest.raises(LizyMLError) as caught:
        check_objective_compatible("binary", "xentlambda")
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert "reason" not in caught.value.context
    assert "is not compatible with task 'binary'" in caught.value.user_message


def test_other_tasks_keep_the_generic_refusal() -> None:
    with pytest.raises(LizyMLError) as caught:
        check_objective_compatible("regression", "cross_entropy_lambda")
    assert "reason" not in caught.value.context
    assert "is not compatible with task 'regression'" in caught.value.user_message


def test_a_str_subclass_runs_none_of_its_methods() -> None:
    """The guarantee of ``check_objective_compatible`` holds for the new branch."""

    class Hostile(str):
        def _boom(self, *args: object) -> object:
            raise RuntimeError("a str subclass method ran")

        __eq__ = __hash__ = __repr__ = __str__ = __format__ = _boom  # type: ignore[assignment]

    with pytest.raises(LizyMLError) as caught:
        check_objective_compatible("binary", Hostile("cross_entropy_lambda"))
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert caught.value.context["reason"] == REASON
    assert "'cross_entropy_lambda'" in caught.value.user_message


# ---------------------------------------------------------------------------
# Criterion 1: every spelling, every layer, before training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_model_params_refuses_before_training(spelling: str) -> None:
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    cfg["model"]["params"][spelling] = "cross_entropy_lambda"
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        Model(cfg).fit(data=make_binary_df(n=80))
    _assert_refused(caught.value)
    assert caught.value.context["surface"] == "model.params"
    assert caught.value.context["parameter"] == spelling
    assert seen["train_params"] == []


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_fit_params_refuses_before_training(spelling: str) -> None:
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        Model(cfg).fit(
            data=make_binary_df(n=80), params={spelling: "cross_entropy_lambda"}
        )
    _assert_refused(caught.value)
    assert caught.value.context["surface"] == "fit(params=)"
    assert seen["train_params"] == []


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_tuning_result_layer_refuses_before_training(spelling: str) -> None:
    model = Model(make_config("binary", n_estimators=3, n_splits=2))
    model._tuning_result = TuningResult(
        best_model_params={spelling: "cross_entropy_lambda"},
        best_smart_params={},
        best_training_params={},
        best_score=0.5,
        trials=[],
        metric_name="logloss",
        direction="minimize",
    )
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        model.fit(data=make_binary_df(n=80))
    _assert_refused(caught.value)
    assert caught.value.context["surface"] == "tuning best_model_params"
    assert seen["train_params"] == []


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_direct_adapter_refuses_before_training(spelling: str) -> None:
    df = make_binary_df(n=80)
    adapter = LGBMAdapter(
        task="binary", params={spelling: "cross_entropy_lambda", "n_estimators": 3}
    )
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        adapter.fit(df.drop(columns=["target"]), df["target"])
    _assert_refused(caught.value)
    assert seen["train_params"] == []


def test_the_alias_value_is_refused_at_fit() -> None:
    cfg = make_config("binary", n_estimators=3, n_splits=2, objective="xentlambda")
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        Model(cfg).fit(data=make_binary_df(n=80))
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert seen["train_params"] == []


# ---------------------------------------------------------------------------
# Criterion 2: what stays accepted predicts probabilities
# ---------------------------------------------------------------------------


def test_the_binary_choices_drop_cross_entropy_lambda() -> None:
    assert LGBMProvider().objective_choices("binary") == ("binary", "cross_entropy")
    assert TASK_COMPATIBLE_OBJECTIVES["binary"] == frozenset(
        {"binary", "cross_entropy"}
    )


@pytest.mark.parametrize("objective", sorted(TASK_COMPATIBLE_OBJECTIVES["binary"]))
def test_every_accepted_binary_objective_predicts_probabilities(
    objective: str,
) -> None:
    """Read from the table (#307 DoD): proba in [0, 1] and OOF metrics computed."""
    df = _learnable_binary()
    cfg = make_config(
        "binary", n_estimators=200, n_splits=3, objective=objective, learning_rate=0.1
    )
    cfg["evaluation"] = {"metrics": ["logloss", "auc", "brier"]}
    model = Model(cfg)
    fit_result = model.fit(data=df)
    proba = model.predict(df.drop(columns=["target"])).proba
    assert proba is not None
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    # Learnable data: the probabilities are not one constant.
    assert float(np.ptp(proba)) > 0.1
    oof = fit_result.metrics["raw"]["oof"]
    assert set(oof) == {"logloss", "auc", "brier"}
    assert all(np.isfinite(value) for value in oof.values())


def test_cross_entropy_lambda_is_still_a_metric() -> None:
    cfg = make_config(
        "binary", n_estimators=5, n_splits=2, metric=["cross_entropy_lambda"]
    )
    fit_result = Model(cfg).fit(data=make_binary_df(n=120))
    curve: dict[str, Any] = fit_result.history[0]["eval_history"]
    assert any("cross_entropy_lambda" in metrics for metrics in curve.values())
