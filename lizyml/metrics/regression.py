"""Regression metrics: RMSE, MAE, R2, RMSLE, MAPE, HuberLoss, SMAPE, WAPE, MASE."""

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


@MetricRegistry.register("smape")
class SMAPE(BaseMetric):
    """Symmetric Mean Absolute Percentage Error (range [0, 200]).

    Formula::

        sMAPE = mean( 2 * |y_true - y_pred| / (|y_true| + |y_pred|) ) * 100

    Tolerates ``y_true == 0`` (unlike :class:`MAPE`). Convention: when
    ``|y_true| + |y_pred| == 0`` for a row (i.e. both are zero), the
    row's contribution is 0 (perfect prediction).

    Use sMAPE instead of MAPE when the target may take the value 0
    on a per-row basis (e.g. demand / count regressions).
    """

    @property
    def name(self) -> str:
        return "smape"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _validate_shapes(y_true, y_pred, self.name)
        denom = np.abs(y_true) + np.abs(y_pred)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                denom == 0,
                0.0,
                2.0 * np.abs(y_true - y_pred) / denom,
            )
        return float(np.mean(terms) * 100.0)


@MetricRegistry.register("wape")
class WAPE(BaseMetric):
    """Weighted Absolute Percentage Error.

    Formula::

        WAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100
             = MAE / mean(|y_true|) * 100

    Defined whenever ``sum(|y_true|) > 0`` — much weaker than MAPE's
    per-row guard. Particularly useful for imbalanced-magnitude
    regressions where one large-volume series would otherwise dominate
    MAE noise.

    Raises :class:`~lizyml.core.exceptions.LizyMLError` with
    ``UNSUPPORTED_METRIC`` only when ``sum(|y_true|) == 0``.
    """

    @property
    def name(self) -> str:
        return "wape"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _validate_shapes(y_true, y_pred, self.name)
        denom = float(np.sum(np.abs(y_true)))
        if denom == 0.0:
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_METRIC,
                user_message="WAPE is undefined when sum(|y_true|) is zero.",
                context={"metric": self.name},
            )
        return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


@MetricRegistry.register("mase")
class MASE(BaseMetric):
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006; H-0132).

    Formula::

        MASE = mean(|y_true - y_pred|) / scale
        scale = mean(|y_train - mean(y_train)|)

    ``scale`` is the MAE of the training-mean forecast on the **training-side
    targets**, so MASE is defined when ``y = 0`` and is free of the target's
    unit. ``MASE < 1`` means better than predicting the training mean. The
    time-series naive and seasonal scales and per-series scales are out of
    scope (H-0132).

    The metric needs the training-side targets (:attr:`needs_train_target`):
    callers bind it with :meth:`bind_train` before calling it. An unbound
    instance answers its name and properties for configuration checks but
    refuses to compute, so the scale never falls back to the scored rows.
    """

    def __init__(self) -> None:
        self._scale: float | None = None

    @property
    def name(self) -> str:
        return "mase"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return False

    @property
    def needs_train_target(self) -> bool:
        return True

    def bind_train(self, y_train: npt.NDArray[Any]) -> MASE:
        """Return a new MASE whose scale comes from *y_train*.

        Raises:
            LizyMLError: ``UNSUPPORTED_METRIC`` when the scale is zero (every
                training target equal) or not finite.
        """
        y = np.asarray(y_train, dtype=np.float64)
        scale = float(np.mean(np.abs(y - np.mean(y)))) if len(y) else float("nan")
        if not np.isfinite(scale) or scale == 0.0:
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_METRIC,
                user_message=(
                    "MASE is undefined when the training targets' mean absolute "
                    f"deviation is zero or not finite (got {scale})."
                ),
                context={"metric": self.name, "scale": scale, "n_train": int(len(y))},
            )
        bound = MASE()
        bound._scale = scale
        return bound

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        if self._scale is None:
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_METRIC,
                user_message=(
                    "MASE needs the training targets: bind it with "
                    "bind_train(y_train) before calling it."
                ),
                context={"metric": self.name},
            )
        _validate_shapes(y_true, y_pred, self.name)
        return float(np.mean(np.abs(y_true - y_pred)) / self._scale)
