"""Shared objective and metric rules for merged inputs and direct adapters."""

from typing import Any

import lightgbm as lgb

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.task import TaskType
from lizyml.estimators.lgbm.defaults import TASK_COMPATIBLE_OBJECTIVES
from lizyml.estimators.lgbm.metric_bridge import resolve_metrics

#: LightGBM objectives of a task whose output is not a probability although
#: LizyML would return it as ``predict().proba`` (H-0125, #307). Refused with
#: the reason ``not_a_probability`` and, when a stored booster records one,
#: by ``Model.load``. None of them is in ``TASK_COMPATIBLE_OBJECTIVES``.
NOT_A_PROBABILITY_OBJECTIVES: dict[str, frozenset[str]] = {
    "binary": frozenset({"cross_entropy_lambda"}),
}


def check_objective_compatible(task: str, objective: Any) -> None:
    """Reject an objective incompatible with the task.

    Every accepted objective is a string, so any other value is refused before
    the membership test: a dict or list there raised a raw ``TypeError``
    (unhashable) instead of ``CONFIG_INVALID`` (H-0116). ``None`` never
    reaches here: both callers treat it as "no override" and skip the check.

    The membership test and the message do not format or hash the rejected
    value: a non-string is described by its type name, and a string is read
    through a plain ``str`` copy, so a ``str`` subclass's overridden
    ``__hash__``, ``__eq__``, ``__format__``, ``__str__`` or ``__repr__`` is
    not used. That is the whole guarantee. A value whose type lookups raise
    (a ``__class__`` that ``isinstance`` reads, a metaclass that raises on
    ``__name__``) is outside it, and so is
    rendering the error afterwards: ``context["objective"]`` keeps the value
    as written, so ``repr(error)`` runs that value's ``__repr__``.
    """
    valid = TASK_COMPATIBLE_OBJECTIVES.get(task, frozenset())
    if isinstance(objective, str):
        name = str.__str__(objective)  # an exact str: no subclass override is used
        shown = f"'{name}'"
        compatible = name in valid
    else:
        name = None
        shown = f"of type '{type(objective).__name__}'"
        compatible = False
    if not compatible and name in NOT_A_PROBABILITY_OBJECTIVES.get(task, ()):
        # H-0125 (#307): a LightGBM objective of this task whose output LizyML
        # would return as a probability although it is not one.
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"objective {shown} is not accepted for task '{task}': its "
                "output is not a probability (LightGBM predicts lambda; the "
                f"probability is 1 - exp(-lambda)), and LizyML returns a {task} "
                "model's output as predict().proba. Use "
                + " or ".join(f"'{value}'" for value in sorted(valid))
                + "."
            ),
            context={
                "task": task,
                "objective": objective,
                "valid_objectives": sorted(valid),
                "reason": "not_a_probability",
            },
        )
    if not compatible:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"objective {shown} is not compatible with task "
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


def recorded_objective(booster: lgb.Booster) -> str | None:
    """The objective LightGBM recorded inside *booster* (H-0125 policy 5).

    Read from the ``objective=`` line of the booster's model string, whose
    first token is the canonical objective (``binary sigmoid:1`` records
    ``binary``; the alias ``xentlambda`` records ``cross_entropy_lambda``).
    That is what the booster was trained with, however it was specified.
    ``booster.params`` is not read: a booster restored from its model string
    has none, and they can be written to after training.

    Only the first iteration is serialised. The header that carries the line
    is the same for any iteration count, so the cost does not grow with the
    number of trees.

    Returns:
        The objective's first token, or ``None`` when the model string has no
        ``objective=`` line (LightGBM writes none for a booster trained with a
        custom objective function, which LizyML never does).
    """
    text = booster.model_to_string(num_iteration=1)
    for line in text.splitlines():
        if line.startswith("objective="):
            tokens = line[len("objective=") :].split()
            return tokens[0] if tokens else None
        if line.startswith("Tree="):
            break
    return None
