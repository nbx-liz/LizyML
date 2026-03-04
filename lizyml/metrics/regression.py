"""Regression metrics: RMSE, MAE, R2, RMSLE."""

from __future__ import annotations

import numpy as np

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.registries import MetricRegistry
from lizyml.metrics.base import BaseMetric


def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        _validate_shapes(y_true, y_pred, self.name)
        return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))
