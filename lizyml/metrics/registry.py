"""Metric registry helpers: task-aware lookup and validation."""

from __future__ import annotations

from typing import Literal

import lizyml.metrics.classification  # noqa: F401

# Import side-effect: registers all metrics into MetricRegistry
import lizyml.metrics.regression  # noqa: F401
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.registries import MetricRegistry
from lizyml.metrics.base import BaseMetric

TaskType = Literal["regression", "binary", "multiclass"]

# Metrics that are valid per task type
_TASK_METRICS: dict[TaskType, frozenset[str]] = {
    "regression": frozenset(["rmse", "mae", "r2", "rmsle", "mape", "huber"]),
    "binary": frozenset(["logloss", "auc", "auc_pr", "f1", "accuracy", "brier", "ece"]),
    "multiclass": frozenset(["logloss", "f1", "accuracy"]),
}


def get_metric(name: str) -> BaseMetric:
    """Return an instantiated metric by name.

    Args:
        name: Registered metric key (e.g. ``"rmse"``, ``"auc"``).

    Returns:
        Instantiated :class:`BaseMetric`.

    Raises:
        :class:`~lizyml.core.exceptions.LizyMLError` with
        ``UNSUPPORTED_METRIC`` when *name* is not registered.
    """
    try:
        cls = MetricRegistry.get(name)
    except KeyError:
        raise LizyMLError(
            code=ErrorCode.UNSUPPORTED_METRIC,
            user_message=(
                f"Metric '{name}' is not registered. Available: {MetricRegistry.keys()}"
            ),
            context={"metric": name},
        ) from None
    instance: BaseMetric = cls()
    return instance


def get_metrics_for_task(names: list[str], task: TaskType) -> list[BaseMetric]:
    """Return instantiated metrics, validating task compatibility.

    Args:
        names: List of metric keys.
        task: ML task type.

    Returns:
        List of :class:`BaseMetric` instances.

    Raises:
        :class:`~lizyml.core.exceptions.LizyMLError` with
        ``UNSUPPORTED_METRIC`` for unknown names or task-incompatible metrics.
    """
    valid_for_task = _TASK_METRICS.get(task, frozenset())
    metrics: list[BaseMetric] = []
    for name in names:
        metric = get_metric(name)
        if name not in valid_for_task:
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_METRIC,
                user_message=(
                    f"Metric '{name}' is not compatible with task '{task}'. "
                    f"Valid metrics for '{task}': {sorted(valid_for_task)}"
                ),
                context={"metric": name, "task": task},
            )
        metrics.append(metric)
    return metrics
