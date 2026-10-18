"""MASE (H-0132, criterion 1): definition, binding and refusals."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.metrics import MASE
from lizyml.metrics.base import BaseMetric
from lizyml.metrics.registry import get_metric, get_metrics_for_task
from lizyml.metrics.regression import RMSE

# Training targets whose mean is 4; their mean absolute deviation is 2.
Y_TRAIN = np.array([2.0, 6.0, 2.0, 6.0])
SCALE = 2.0


def test_value_is_mae_over_the_training_mean_deviation() -> None:
    y_true = np.array([0.0, 10.0, 5.0])
    y_pred = np.array([1.0, 7.0, 5.0])
    mae = np.mean(np.abs(y_true - y_pred))  # 4/3
    assert MASE().bind_train(Y_TRAIN)(y_true, y_pred) == pytest.approx(mae / SCALE)


def test_the_scale_comes_from_the_training_targets_not_the_scored_rows() -> None:
    """The evaluated rows have mean deviation 10; the training rows 2."""
    y_true = np.array([0.0, 20.0])
    y_pred = np.array([1.0, 19.0])
    value = MASE().bind_train(Y_TRAIN)(y_true, y_pred)
    assert value == pytest.approx(1.0 / SCALE)
    assert value != pytest.approx(1.0 / 10.0)


def test_targets_containing_zero_give_a_finite_value() -> None:
    value = MASE().bind_train(np.array([0.0, 0.0, 4.0]))(
        np.array([0.0, 0.0, 1.0]), np.array([0.5, 0.0, 1.0])
    )
    assert np.isfinite(value)


@pytest.mark.parametrize(
    "y_train",
    [np.array([3.0, 3.0, 3.0]), np.array([1.0, np.inf]), np.array([1.0, np.nan])],
    ids=["constant", "infinite", "nan"],
)
def test_a_zero_or_non_finite_scale_is_refused(y_train: np.ndarray) -> None:
    with pytest.raises(LizyMLError) as caught:
        MASE().bind_train(y_train)
    assert caught.value.code == ErrorCode.UNSUPPORTED_METRIC
    assert caught.value.context["metric"] == "mase"
    assert caught.value.context["n_train"] == len(y_train)


@pytest.mark.parametrize(
    "y_train",
    [np.array([1.0, np.inf]), np.array([-np.inf, np.inf]), np.array([1e308, -1e308])],
    ids=["infinite", "both-infinities", "overflow"],
)
def test_a_non_finite_scale_is_refused_without_a_numpy_warning(
    y_train: np.ndarray,
) -> None:
    """Implementation review round 2: the refusal came after a RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(LizyMLError):
            MASE().bind_train(y_train)


def test_an_unbound_call_is_refused() -> None:
    with pytest.raises(LizyMLError) as caught:
        MASE()(np.array([1.0]), np.array([1.0]))
    assert caught.value.code == ErrorCode.UNSUPPORTED_METRIC
    assert "training" in caught.value.user_message


def test_binding_returns_a_new_instance_and_leaves_the_original_unbound() -> None:
    metric = MASE()
    bound = metric.bind_train(Y_TRAIN)
    assert bound is not metric
    with pytest.raises(LizyMLError):
        metric(np.array([1.0]), np.array([1.0]))


def test_the_metric_declares_its_contract() -> None:
    metric = get_metric("mase")
    assert isinstance(metric, MASE)
    assert metric.name == "mase"
    assert metric.greater_is_better is False
    assert metric.needs_proba is False
    assert metric.needs_train_target is True


def test_mase_is_a_regression_metric_only() -> None:
    assert [m.name for m in get_metrics_for_task(["mase"], "regression")] == ["mase"]
    for task in ("binary", "multiclass"):
        with pytest.raises(LizyMLError) as caught:
            get_metrics_for_task(["mase"], task)
        assert caught.value.code == ErrorCode.UNSUPPORTED_METRIC


def test_parameters_are_refused() -> None:
    with pytest.raises(LizyMLError) as caught:
        get_metric("mase", x=1)
    assert caught.value.code == ErrorCode.CONFIG_INVALID


def test_other_metrics_do_not_need_training_targets_and_refuse_binding() -> None:
    metric = RMSE()
    assert metric.needs_train_target is False
    with pytest.raises(LizyMLError) as caught:
        metric.bind_train(Y_TRAIN)
    assert caught.value.code == ErrorCode.UNSUPPORTED_METRIC


def test_an_existing_subclass_keeps_working_unchanged() -> None:
    """A third-party subclass that predates H-0132 is not made abstract."""

    class Custom(BaseMetric):
        @property
        def name(self) -> str:
            return "custom"

        @property
        def needs_proba(self) -> bool:
            return False

        @property
        def greater_is_better(self) -> bool:
            return False

        def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.mean(y_true - y_pred))

    metric = Custom()
    assert metric.needs_train_target is False
    assert metric(np.array([2.0]), np.array([1.0])) == 1.0
