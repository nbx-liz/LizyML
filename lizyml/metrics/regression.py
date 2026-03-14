"""Regression metrics: RMSE, MAE, R2, RMSLE, MAPE, HuberLoss."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.registries import MetricRegistry
from lizyml.metrics.base import BaseMetric


def _validate_shapes(
    y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any], name: str
) -> None:
    if y_true.shape != y_pred.shape:
        raise LizyMLError(
            code=ErrorCode.UNSUPPORTED_METRIC,
            user_message=(
                f"Metric '{name}' requires y_true and y_pred to have the same shape. "
                f"Got {y_true.shape} vs {y_pred.shape}."
            ),
            context={
                "metric": name,
                "y_true_shape": y_true.shape,
                "y_pred_shape": y_pred.shape,
            },
        )


@MetricRegistry.register("rmse")
class RMSE(BaseMetric):
    """Root Mean Squared Error."""

    @property
    def name(self) -> str:
        return "rmse"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _validate_shapes(y_true, y_pred, self.name)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@MetricRegistry.register("mae")
class MAE(BaseMetric):
    """Mean Absolute Error."""

    @property
    def name(self) -> str:
        return "mae"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _validate_shapes(y_true, y_pred, self.name)
        return float(np.mean(np.abs(y_true - y_pred)))


@MetricRegistry.register("r2")
class R2(BaseMetric):
    """Coefficient of Determination (R²)."""

    @property
    def name(self) -> str:
        return "r2"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return True

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _validate_shapes(y_true, y_pred, self.name)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot


@MetricRegistry.register("rmsle")
class RMSLE(BaseMetric):
    """Root Mean Squared Logarithmic Error.

    Requires non-negative predictions and targets.
    """

    @property
    def name(self) -> str:
        return "rmsle"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _validate_shapes(y_true, y_pred, self.name)
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_METRIC,
                user_message="RMSLE requires non-negative predictions and targets.",
                context={"metric": self.name},
            )
        return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


@MetricRegistry.register("mape")
class MAPE(BaseMetric):
    """Mean Absolute Percentage Error.

    Raises :class:`~lizyml.core.exceptions.LizyMLError` with
    ``UNSUPPORTED_METRIC`` when *y_true* contains zeros.
    """

    @property
    def name(self) -> str:
        return "mape"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _validate_shapes(y_true, y_pred, self.name)
        if np.any(y_true == 0):
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_METRIC,
                user_message="MAPE is undefined when y_true contains zeros.",
                context={"metric": self.name},
            )
        return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


@MetricRegistry.register("huber")
class HuberLoss(BaseMetric):
    """Huber Loss with configurable delta.

    For |error| <= delta: 0.5 * error²
    For |error| >  delta: delta * (|error| − 0.5 * delta)

    Args:
        delta: Threshold between squared and linear loss regions. Default 1.0.
    """

    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta

    @property
    def name(self) -> str:
        return "huber"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _validate_shapes(y_true, y_pred, self.name)
        e = y_true - y_pred
        abs_e = np.abs(e)
        loss: npt.NDArray[Any] = np.where(
            abs_e <= self.delta,
            0.5 * e**2,
            self.delta * (abs_e - 0.5 * self.delta),
        )
        return float(np.mean(loss))
