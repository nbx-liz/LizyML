"""LizyML metrics package.

Import from here to ensure all metrics are registered into ``MetricRegistry``.
"""

from lizyml.metrics.base import BaseMetric
from lizyml.metrics.classification import (
    AUC,
    AUCPR,
    ECE,
    F1,
    Accuracy,
    Brier,
    LogLoss,
)
from lizyml.metrics.registry import get_metric, get_metrics_for_task
from lizyml.metrics.regression import MAE, MAPE, R2, RMSE, RMSLE, HuberLoss

__all__ = [
    "BaseMetric",
    # regression
    "RMSE",
    "MAE",
    "R2",
    "RMSLE",
    "MAPE",
    "HuberLoss",
    # classification
    "LogLoss",
    "AUC",
    "AUCPR",
    "F1",
    "Accuracy",
    "Brier",
    "ECE",
    # helpers
    "get_metric",
    "get_metrics_for_task",
]
