"""The smart values a tune's trials can take, and the layers checked over them.

H-0121 amendment 1 (#341). A ``category: smart`` dimension overlays its
sampled value on every trial, so a smart parameter the config leaves inactive
can be active inside a trial and manage a name a parameter layer sets. The
checks therefore read every combination of the values the round's smart
dimensions can take, reduced by the provider contract
(``EstimatorProvider.smart_managed_param_names``): the result depends on each
smart value only through whether it is ``None`` and its truthiness. One
representative per state is enough, so the product has at most ``3 ** k``
members for ``k`` smart dimensions.

Both checks use :func:`smart_value_combinations`: the layers
(:func:`check_layers_over_smart_values`, from ``check_round_space``) and the
``category: model`` dimensions (``validate_tuning_dimensions``).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from lizyml.core._model_factories import check_smart_managed_overrides
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim, SearchDim

if TYPE_CHECKING:
    from lizyml.core.types.task import TaskType
    from lizyml.estimators.provider import EstimatorProvider

#: How a refusal names the provider's default fixed parameters. The same
#: string as ``lizyml.core.model._FIXED_LAYER``, which this module cannot
#: import (``model`` imports the tuning modules that import this one); a test
#: pins the two equal.
FIXED_LAYER = "provider default fixed params"

#: The states the provider contract distinguishes, in the order the
#: representatives are returned.
_STATES: tuple[bool | None, ...] = (None, False, True)


def _state(value: Any) -> bool | None:
    """``None`` for ``None``, else the value's truthiness."""
    return None if value is None else bool(value)


def smart_value_candidates(dim: SearchDim) -> tuple[Any, ...]:
    """One value per state a ``category: smart`` dimension can take.

    The values a trial can sample: a categorical dimension's choices; a
    numeric one's two ends, plus ``0`` when the range spans it (``low <= 0 <=
    high``) -- the only falsy number a range can hold, and one neither end
    shows when both are non-zero. Then one representative per state: the
    first value of each state in that order, returned as ``None``, falsy,
    truthy (states the dimension cannot take are left out). Under the
    provider contract each state gives one result, so the representatives
    cover every result the dimension can produce.
    """
    if isinstance(dim, CategoricalDim):
        values: Iterable[Any] = dim.choices
    else:
        zero = (0,) if dim.low <= 0 <= dim.high else ()
        values = (dim.low, dim.high, *zero)
    first: dict[bool | None, Any] = {}
    for value in values:
        first.setdefault(_state(value), value)
    return tuple(first[state] for state in _STATES if state in first)


def smart_dimensions(space: Iterable[SearchDim]) -> list[SearchDim]:
    """The ``category: smart`` dimensions of *space*, in space order."""
    return [dim for dim in space if dim.category == "smart"]


def smart_value_combinations(
    space: Iterable[SearchDim], base_smart: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Every smart parameter dict a trial of this space can resolve with.

    The product of :func:`smart_value_candidates` over the smart dimensions,
    each laid over *base_smart* (a trial overlays every dimension's sample,
    so a dimension's base value never reaches a trial; a smart parameter
    without a dimension keeps its base value). With no smart dimension the
    result is ``[base_smart]``.
    """
    dims = smart_dimensions(space)
    candidates = [smart_value_candidates(dim) for dim in dims]
    return [
        {
            **base_smart,
            **{dim.name: value for dim, value in zip(dims, combo, strict=True)},
        }
        for combo in itertools.product(*candidates)
    ]


def _activating_dimensions(
    provider: EstimatorProvider,
    name: str,
    combination: Mapping[str, Any],
    base_smart: Mapping[str, Any],
    dims: Sequence[SearchDim],
    task: TaskType,
) -> list[str]:
    """The smart dimensions whose value in *combination* makes *name* managed.

    A dimension counts when putting its base value back clears *name*. When
    no single dimension does (several can each activate it), every dimension
    whose value differs in state from its base value is named; when none does
    either, every smart dimension is.
    """
    necessary = [
        dim.name
        for dim in dims
        if name
        not in provider.smart_managed_param_names(
            {**combination, dim.name: base_smart.get(dim.name)}, task
        )
    ]
    if necessary:
        return necessary
    differing = [
        dim.name
        for dim in dims
        if _state(combination[dim.name]) != _state(base_smart.get(dim.name))
    ]
    return differing or [dim.name for dim in dims]


def check_layer_over_smart_values(
    provider: EstimatorProvider,
    layer: Mapping[str, Any],
    combinations: Sequence[Mapping[str, Any]],
    space: Iterable[SearchDim],
    base_smart: Mapping[str, Any],
    task: TaskType,
    *,
    surface: str,
    space_surface: str,
) -> None:
    """Refuse a name of *layer* some combination's smart parameters manage.

    ``check_smart_managed_overrides`` decides, once per combination. Its
    refusal is re-raised with the smart dimension(s) that make the managing
    smart parameter active and the space they were declared in (H-0121
    amendment 1, policy 1).

    Args:
        provider: The estimator provider.
        layer: The names checked (a parameter layer, or the space's
            ``category: model`` dimensions).
        combinations: :func:`smart_value_combinations` of *space*.
        space: The round's effective space.
        base_smart: The effective smart values the round starts from.
        task: ML task type.
        surface: How to name *layer* in the message.
        space_surface: ``tuning.optuna.space`` or ``tune(space=)``.

    Raises:
        LizyMLError: ``CONFIG_INVALID``.
    """
    if not layer:
        return
    dims = smart_dimensions(space)
    names = dict(layer)
    for combination in combinations:
        try:
            check_smart_managed_overrides(
                provider, names, dict(combination), task, surface=surface
            )
        except LizyMLError as exc:
            # Without a smart dimension, or without the managed names the
            # message is rebuilt from, the base refusal stands as it is.
            if not dims or not exc.context.get("managed"):
                raise
            raise _with_dimensions(
                exc, provider, combination, base_smart, dims, task, space_surface
            ) from exc


def _with_dimensions(
    exc: LizyMLError,
    provider: EstimatorProvider,
    combination: Mapping[str, Any],
    base_smart: Mapping[str, Any],
    dims: Sequence[SearchDim],
    task: TaskType,
    space_surface: str,
) -> LizyMLError:
    """*exc*, naming the smart dimension(s) that activate each managed name.

    The message is rebuilt from *exc*'s context rather than extended: the
    base refusal advises disabling the smart parameter in the config, which a
    dimension that switches it on in some trials makes the wrong advice.
    """
    managed = []
    lines = []
    for hit in exc.context.get("managed", []):
        activating = _activating_dimensions(
            provider, hit["name"], combination, base_smart, dims, task
        )
        entry = {**hit, "smart_dimensions": activating, "space_surface": space_surface}
        managed.append(entry)
        name, canonical, layer = hit["name"], hit["canonical"], hit["surface"]
        subject = (
            f"'{name}'"
            if name == canonical
            else f"'{name}', which names '{canonical}',"
        )
        values = ", ".join(f"'{dim}'={combination[dim]!r}" for dim in activating)
        lines.append(
            f"  {layer}: {subject} is resolved from the smart parameter "
            f"'{hit['smart_param']}' in the trials where the category: smart "
            f"dimension(s) of {space_surface} take {values}, and would be "
            f"replaced there, so setting it here would have no effect in those "
            f"trials. Remove '{name}' from {layer}, or declare the dimension(s) "
            "without the value(s) that activate it."
        )
    return LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            "Parameter name(s) a smart parameter manages in some trials of this "
            "tune:\n" + "\n".join(lines)
        ),
        cause=exc,
        context={
            **exc.context,
            "managed": managed,
            "space_surface": space_surface,
            "smart_values": {dim.name: combination[dim.name] for dim in dims},
        },
    )


def check_layers_over_smart_values(
    provider: EstimatorProvider,
    layers: Sequence[tuple[str, Mapping[str, Any]]],
    space: Iterable[SearchDim],
    base_smart: Mapping[str, Any],
    task: TaskType,
    *,
    space_surface: str,
) -> None:
    """Check each ``(surface, layer)`` over every trial's smart values.

    The parameter layers a trial trains on -- ``model.params`` and the
    round's fixed parameters -- against :func:`smart_value_combinations`
    (H-0121 amendment 1, policy 1). Policy 1 of H-0121 already checked them
    against the base smart values (``Model._merge_params``).
    """
    space = list(space)
    if not smart_dimensions(space):
        return
    combinations = smart_value_combinations(space, base_smart)
    for surface, layer in layers:
        check_layer_over_smart_values(
            provider,
            layer,
            combinations,
            space,
            base_smart,
            task,
            surface=surface,
            space_surface=space_surface,
        )
