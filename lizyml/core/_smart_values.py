"""The values a smart parameter takes in a tune (H-0121 amendment 2, #347).

A ``category: smart`` dimension's sampled value becomes the trial's smart
parameter as it is, and smart resolution reads it by truthiness and ``None``:
``["false"]`` trained with ``auto_num_leaves`` on, a ratio of ``0`` silently
became another value. The provider declares, per smart parameter, the values
it can hold (``EstimatorProvider.smart_value_rules``); this module holds them
to it:

* :func:`check_smart_dimension_values` -- every value a round's smart
  dimensions can take (policy 2), before the H-0121 owner rule reads them by
  truthiness and before the round admission's other checks;
* :func:`smart_value_violation` -- a recorded ``best_smart_params``, for
  ``Model.load`` (policy 4) and ``restore_tuning`` (policy 5), which choose
  their own error code.

Pure: reads the provider and the values, writes nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.search_dim import SearchDim
    from lizyml.core.types.task import TaskType
    from lizyml.estimators.provider import EstimatorProvider, SmartValueRule


@dataclass(frozen=True)
class SmartValueViolation:
    """A recorded smart value outside its rule."""

    name: str
    value: Any
    reason: str
    expected: str


def _rule_for(
    provider: EstimatorProvider, rules: Mapping[str, SmartValueRule], name: str
) -> SmartValueRule:
    rule = rules.get(name)
    if rule is None:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"Internal error: the estimator provider "
                f"{type(provider).__name__} declares the smart parameter "
                f"'{name}' without a value rule (smart_value_rules)."
            ),
            context={"name": name, "provider": type(provider).__name__},
        )
    return rule


def check_smart_dimension_values(
    provider: EstimatorProvider,
    space: Iterable[SearchDim],
    task: TaskType,
    *,
    surface: str,
) -> None:
    """Refuse a ``category: smart`` dimension that can take a value outside its rule.

    Every value counts: a numeric dimension's ``low`` and ``high`` (the range
    between them lies inside any interval both are in), a categorical one's
    choices. A name that is not a smart parameter is left to the undeclared-
    name refusal of ``validate_tuning_dimensions``.

    Args:
        provider: Declares the smart parameters and their rules.
        space: The dimensions to check; only ``category: smart`` ones are read.
        task: The ML task (``balanced=True`` is refused on regression).
        surface: ``tuning.optuna.space`` or ``tune(space=)``.

    Raises:
        LizyMLError: ``CONFIG_INVALID`` with ``reason`` ``smart_value_invalid``
            or ``smart_value_unsupported_task``, and ``surface``, ``dim``,
            ``key`` (``low`` / ``high`` / ``choices``) and ``value``.
    """
    declared = provider.smart_param_names()
    rules = provider.smart_value_rules()
    for dim in space:
        if dim.category != "smart" or dim.name not in declared:
            continue
        rule = _rule_for(provider, rules, dim.name)
        found = rule.check_dimension(dim, task)
        if found is None:
            continue
        key, value, reason = found
        hint = (
            " Values are read by their type: write true / false or a number, "
            "not a string."
            if isinstance(value, str)
            else ""
        )
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"{surface}: category: smart dimension '{dim.name}' can take "
                f"{value!r} (its {key}), which the smart parameter '{dim.name}' "
                f"cannot hold: it takes {rule.describe(task)}.{hint}"
            ),
            context={
                "surface": surface,
                "dim": dim.name,
                "key": key,
                "value": value,
                "reason": reason,
            },
        )


def smart_value_violation(
    provider: EstimatorProvider, values: Mapping[str, Any], task: TaskType
) -> SmartValueViolation | None:
    """The first recorded smart value outside its rule, or ``None``.

    *values* is a tune result's ``best_smart_params``: the sampled values of
    the round's smart dimensions. A name the provider does not declare is a
    violation too: no tune records one (such a dimension is refused before
    the study), so the record is not LizyML's own.
    """
    rules = provider.smart_value_rules()
    declared = provider.smart_param_names()
    for name, value in values.items():
        if name not in declared:
            return SmartValueViolation(
                name=name,
                value=value,
                reason="smart_value_invalid",
                expected="no value: it is not a smart parameter of this estimator",
            )
        rule = _rule_for(provider, rules, name)
        reason = rule.check_value(value, task)
        if reason is not None:
            return SmartValueViolation(
                name=name, value=value, reason=reason, expected=rule.describe(task)
            )
    return None
