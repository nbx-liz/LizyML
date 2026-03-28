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
    PrecisionAtK,
)
from lizyml.metrics.registry import (
    MetricEntry,
    get_metric,
    get_metrics_for_task,
    parse_metric_entries,
    parse_metric_entry,
)
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
    "PrecisionAtK",
    # helpers
    "MetricEntry",
    "get_metric",
    "get_metrics_for_task",
    "parse_metric_entry",
    "parse_metric_entries",
]
