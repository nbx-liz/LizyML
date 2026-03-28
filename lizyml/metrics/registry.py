"""Metric registry helpers: task-aware lookup and validation.

Supports ``MetricEntry`` — either a plain string or a dict mapping a
metric name to its keyword arguments (H-0065).
"""

from __future__ import annotations

from typing import Any, Literal

import lizyml.metrics.classification  # noqa: F401

# Import side-effect: registers all metrics into MetricRegistry
import lizyml.metrics.regression  # noqa: F401
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.registries import MetricRegistry
from lizyml.metrics.base import BaseMetric

TaskType = Literal["regression", "binary", "multiclass"]

# H-0065: A metric entry is either a plain name or {name: {param: value}}.
MetricEntry = str | dict[str, dict[str, Any]]

# Metrics that are valid per task type
_TASK_METRICS: dict[TaskType, frozenset[str]] = {
    "regression": frozenset(["rmse", "mae", "r2", "rmsle", "mape", "huber"]),
    "binary": frozenset(
        ["logloss", "auc", "auc_pr", "f1", "accuracy", "brier", "ece", "precision_at_k"]
    ),
    "multiclass": frozenset(["logloss", "f1", "accuracy", "auc", "auc_pr", "brier"]),
}


# ---------------------------------------------------------------------------
# MetricEntry parsing (H-0065)
# ---------------------------------------------------------------------------


def parse_metric_entry(entry: MetricEntry) -> tuple[str, dict[str, Any]]:
    """Normalise a single MetricEntry to ``(name, kwargs)``.

    Args:
        entry: ``"rmse"`` or ``{"precision_at_k": {"k": 20}}``.

    Returns:
        ``(metric_name, kwargs_dict)`` — kwargs is empty for plain strings.

    Raises:
        :class:`~lizyml.core.exceptions.LizyMLError` with ``CONFIG_INVALID``
        when the dict form is malformed.
    """
    if isinstance(entry, str):
        return entry, {}

    if not isinstance(entry, dict):
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"MetricEntry must be a str or dict, got {type(entry).__name__}."
            ),
            context={"entry": entry},
        )

    if len(entry) != 1:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                "MetricEntry dict must have exactly one key (the metric name). "
                f"Got {len(entry)} keys: {sorted(entry.keys())}."
            ),
            context={"entry": entry},
        )

    name = next(iter(entry))
    kwargs = entry[name]

    if not isinstance(kwargs, dict):
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"MetricEntry value for '{name}' must be a dict of parameters, "
                f"got {type(kwargs).__name__}."
            ),
            context={"entry": entry},
        )

    return name, kwargs


def parse_metric_entries(
    entries: list[MetricEntry],
) -> list[tuple[str, dict[str, Any]]]:
    """Normalise a list of MetricEntry to ``[(name, kwargs), ...]``."""
    return [parse_metric_entry(e) for e in entries]


# ---------------------------------------------------------------------------
# Metric instantiation
# ---------------------------------------------------------------------------


def get_metric(name: str, **kwargs: Any) -> BaseMetric:
    """Return an instantiated metric by name.

    Args:
        name: Registered metric key (e.g. ``"rmse"``, ``"auc"``).
        **kwargs: Optional keyword arguments forwarded to the metric
            constructor (e.g. ``k=20`` for ``PrecisionAtK``).

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
    try:
        instance: BaseMetric = cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(f"Invalid parameters for metric '{name}': {exc}"),
            context={"metric": name, "kwargs": kwargs},
        ) from exc
    return instance


def get_metrics_for_task(
    entries: list[MetricEntry],
    task: TaskType,
) -> list[BaseMetric]:
    """Return instantiated metrics, validating task compatibility.

    Accepts both plain strings and ``MetricEntry`` dicts (H-0065).

    Args:
        entries: List of metric keys or ``{name: {param: value}}`` dicts.
        task: ML task type.

    Returns:
        List of :class:`BaseMetric` instances.

    Raises:
        :class:`~lizyml.core.exceptions.LizyMLError` with
        ``UNSUPPORTED_METRIC`` for unknown names or task-incompatible metrics.
    """
    valid_for_task = _TASK_METRICS.get(task, frozenset())
    metrics: list[BaseMetric] = []
    for entry in entries:
        name, kwargs = parse_metric_entry(entry)
        metric = get_metric(name, **kwargs)
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
