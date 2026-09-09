"""Shared objective and metric rules for merged inputs and direct adapters."""

from typing import Any

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.task import TaskType
from lizyml.estimators.lgbm.defaults import TASK_COMPATIBLE_OBJECTIVES
from lizyml.estimators.lgbm.metric_bridge import resolve_metrics


def check_objective_compatible(task: str, objective: str) -> None:
    """Reject an objective incompatible with the task."""
    valid = TASK_COMPATIBLE_OBJECTIVES.get(task, frozenset())
    if objective not in valid:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"objective '{objective}' is not compatible with task "
                f"'{task}'. Valid objectives: {sorted(valid)}."
            ),
            context={
                "task": task,
                "objective": objective,
                "valid_objectives": sorted(valid),
            },
        )


def resolve_user_metric(
    value: Any,
    task: TaskType,
    num_class: int | None = None,
) -> tuple[list[str], list[Any], list[str]] | None:
    """Apply the adapter's empty-value fallback and resolve explicit metrics."""
    if not value:
        return None
    entries = [value] if isinstance(value, (str, dict)) else value
    entries = [entry for entry in entries if entry]
    if not entries:
        return None
    return resolve_metrics(entries, task, num_class=num_class)
