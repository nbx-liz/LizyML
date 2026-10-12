"""Tuning admission checks over effective metrics and resolved dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from lizyml.core import _smart_values
from lizyml.core._model_factories import check_smart_managed_overrides
from lizyml.core._model_metrics import _DEFAULT_METRICS
from lizyml.core._smart_combinations import (
    check_layer_over_smart_values,
    smart_value_combinations,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim, SearchDim
from lizyml.core.types.task import TaskType
from lizyml.metrics.registry import get_metric, parse_metric_entry

if TYPE_CHECKING:
    from lizyml.config.schema import LizyMLConfig
    from lizyml.estimators.provider import EstimatorProvider

#: The ``category: training`` dimensions training consumes. Read here, to refuse
#: any other name in a search space, and by ``Model.load()``, to refuse any
#: other name in an artifact's record of the overlay a fit applied (H-0109).
TRAINING_DIMENSION_NAMES = frozenset({"early_stopping_rounds", "validation_ratio"})


def resolve_tuning_direction(cfg: LizyMLConfig) -> Literal["minimize", "maximize"]:
    """Resolve automatic orientation, refusing contradictory explicit input."""
    assert cfg.tuning is not None
    entry = (cfg.evaluation.metrics or _DEFAULT_METRICS[cfg.task])[0]
    name, kwargs = parse_metric_entry(entry)
    expected: Literal["minimize", "maximize"] = (
        "maximize" if get_metric(name, **kwargs).greater_is_better else "minimize"
    )
    requested = cfg.tuning.optuna.params.direction
    if requested is not None and requested != expected:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"tuning.optuna.params.direction: metric '{name}' requires "
                f"'{expected}', not '{requested}'. "
                "Omit direction for automatic selection."
            ),
            context={
                "surface": "tuning.optuna.params.direction",
                "metric": name,
                "direction": requested,
                "expected_direction": expected,
            },
        )
    return expected


def validate_tuning_dimensions(
    provider: EstimatorProvider,
    space: list[SearchDim],
    smart: dict[str, Any],
    task: TaskType,
    *,
    surface: str = "tuning.optuna.space",
) -> None:
    """Refuse model/training dimensions that cannot be applied as declared.

    A ``category: model`` dimension a smart parameter manages is refused
    against the base smart values and against every combination of the values
    the smart dimensions can take (H-0121 amendment 1, policy 2;
    ``lizyml.core._smart_combinations``). The provider contract -- activation
    depends on each smart value only through ``None`` and truthiness -- lets
    one value per state stand for a dimension; the provider remains the
    authority for the affected native names and their aliases.
    ``surface`` names where the dimensions were declared: the config's
    ``tuning.optuna.space`` or a resumed round's ``tune(space=)`` (H-0122).
    """
    unsupported = sorted(
        dim.name
        for dim in space
        if dim.category == "training" and dim.name not in TRAINING_DIMENSION_NAMES
    )
    if unsupported:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"{surface}: training does not consume dimension(s): "
                + ", ".join(unsupported)
            ),
            context={
                "surface": surface,
                "names": unsupported,
                "category": "training",
            },
        )
    _refuse_undeclared_smart_dimensions(provider, space, surface=surface)
    # H-0121 amendment 2, policy 2: every value the round's smart dimensions
    # can take -- the config's, `space=`'s, the defaults and the expanded
    # ones alike -- before the checks below read them by truthiness.
    _smart_values.check_smart_dimension_values(provider, space, task, surface=surface)
    model_names = dict.fromkeys(dim.name for dim in space if dim.category == "model")
    check_smart_managed_overrides(provider, model_names, smart, task, surface=surface)
    # H-0121 amendment 1, policy 2: against the smart values the trials can
    # take, as one product -- a provider may activate a name only for a
    # combination of smart values -- with one representative per state of
    # each dimension (the same reduction ``check_round_space`` uses for the
    # parameter layers).
    check_layer_over_smart_values(
        provider,
        model_names,
        smart_value_combinations(space, smart),
        space,
        smart,
        task,
        surface=surface,
        space_surface=surface,
    )


def _refuse_undeclared_smart_dimensions(
    provider: EstimatorProvider,
    space: list[SearchDim],
    *,
    surface: str = "tuning.optuna.space",
) -> None:
    """Refuse a ``category: smart`` dimension that names no smart parameter.

    Smart resolution reads only the smart parameters the provider declares, so
    any other name is sampled, recorded in ``best_params`` and read by nothing
    (#299, H-0121 decision 2). A native estimator name is the likely intent, so
    the message points it at ``category: model``.
    """
    declared = provider.smart_param_names()
    undeclared = [
        dim.name
        for dim in space
        if dim.category == "smart" and dim.name not in declared
    ]
    if not undeclared:
        return
    native = provider.accepted_model_param_names()
    lines = []
    for name in undeclared:
        if name in native:
            lines.append(
                f"  '{name}' is a native model parameter, not a smart "
                "parameter. Declare it with category: model."
            )
        else:
            lines.append(f"  '{name}' is not a smart parameter.")
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            f"{surface}: category: smart dimension(s) that no smart "
            "parameter reads, so they would have had no effect:\n" + "\n".join(lines)
        ),
        context={
            "surface": surface,
            "names": undeclared,
            "category": "smart",
        },
    )


def _owner_can_be_true(
    owner: str, smart_dims: dict[str, SearchDim], smart: dict[str, Any]
) -> bool:
    """Whether *owner* can be true in some trial of this tune (H-0121).

    A dimension on the owner overlays its sampled value on every trial, so its
    possible values decide; otherwise the effective smart value does.
    """
    dim = smart_dims.get(owner)
    if dim is None:
        return bool(smart.get(owner))
    if isinstance(dim, CategoricalDim):
        return any(bool(choice) for choice in dim.choices)
    return not (dim.low == 0 and dim.high == 0)


def inactive_smart_dimensions(
    provider: EstimatorProvider,
    space: list[SearchDim],
    smart: dict[str, Any],
) -> list[SearchDim]:
    """Return the smart dimensions whose owner cannot be true (H-0121).

    Provider-independent: the provider declares only which smart parameter
    owns which (``EstimatorProvider.smart_param_owners``).

    Args:
        provider: Declares the owners.
        space: The resolved search space.
        smart: The effective smart parameter values the tune starts from.

    Returns:
        The dimensions, in space order, that no trial would read.
    """
    owners = provider.smart_param_owners()
    smart_dims = {dim.name: dim for dim in space if dim.category == "smart"}
    return [
        dim
        for dim in space
        if dim.category == "smart"
        and dim.name in owners
        and not _owner_can_be_true(owners[dim.name], smart_dims, smart)
    ]


def admit_smart_dimensions(
    provider: EstimatorProvider,
    space: list[SearchDim],
    smart: dict[str, Any],
    default_dims: frozenset[tuple[str, str]],
    *,
    surface: str = "tuning.optuna.space",
) -> tuple[list[SearchDim], list[SearchDim]]:
    """Refuse or drop the smart dimensions nothing would read (H-0121).

    For a tune that creates a new study only. A dimension the user wrote is
    refused (decision 3): the user asked for it and would otherwise not learn
    that it did nothing. One that came from the provider's default space is
    dropped (decision 4): ``auto_num_leaves: false`` is the documented way to
    tune native leaves, and refusing would break that configuration.

    Args:
        provider: Declares the owners.
        space: The resolved search space.
        smart: The effective smart parameter values the tune starts from.
        default_dims: ``(category, name)`` of the dimensions that came from
            the default space and were not overridden by the user.
        surface: Where the user declared the space, quoted in the refusal:
            the config, or ``tune(space=)`` for a later round (H-0122).

    Returns:
        ``(space without the dropped dimensions, the dropped dimensions)``.

    Raises:
        LizyMLError: ``CONFIG_INVALID`` naming each user-written dimension
            and its owner.
    """
    inactive = inactive_smart_dimensions(provider, space, smart)
    if not inactive:
        return space, []
    owners = provider.smart_param_owners()
    written = [dim for dim in inactive if (dim.category, dim.name) not in default_dims]
    if written:
        lines = []
        for dim in written:
            owner = owners[dim.name]
            lines.append(
                f"  '{dim.name}' is read only while '{owner}' is true, and "
                f"'{owner}' cannot be true in this tune. Set 'model.{owner}: "
                f"true', give a category: smart '{owner}' dimension a true "
                f"choice, or remove '{dim.name}'."
            )
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"{surface}: smart dimension(s) that nothing would "
                "read, so they would have had no effect:\n" + "\n".join(lines)
            ),
            context={
                "surface": surface,
                "names": [dim.name for dim in written],
                "owners": {dim.name: owners[dim.name] for dim in written},
            },
        )
    return [dim for dim in space if dim not in inactive], inactive
