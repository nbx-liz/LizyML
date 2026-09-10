"""Tuning admission checks over effective metrics and resolved dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from lizyml.core._model_factories import check_smart_managed_overrides
from lizyml.core._model_metrics import _DEFAULT_METRICS
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim, SearchDim
from lizyml.core.types.task import TaskType
from lizyml.metrics.registry import get_metric, parse_metric_entry

if TYPE_CHECKING:
    from lizyml.config.schema import LizyMLConfig
    from lizyml.estimators.provider import EstimatorProvider


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
) -> None:
    """Refuse model/training dimensions that cannot be applied as declared.

    The shipped provider's activation rules use truthiness or presence. Numeric
    bounds and categorical choices cover those activation states; the provider
    remains the authority for the affected native names and their aliases.
    """
    unsupported = sorted(
        dim.name
        for dim in space
        if dim.category == "training"
        and dim.name not in {"early_stopping_rounds", "validation_ratio"}
    )
    if unsupported:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                "tuning.optuna.space: training does not consume dimension(s): "
                + ", ".join(unsupported)
            ),
            context={
                "surface": "tuning.optuna.space",
                "names": unsupported,
                "category": "training",
            },
        )
    model_names = dict.fromkeys(dim.name for dim in space if dim.category == "model")
    check_smart_managed_overrides(
        provider, model_names, smart, task, surface="tuning.optuna.space"
    )
    for dim in space:
        if dim.category != "smart":
            continue
        values = dim.choices if isinstance(dim, CategoricalDim) else (dim.low, dim.high)
        for value in values:
            check_smart_managed_overrides(
                provider,
                model_names,
                {**smart, dim.name: value},
                task,
                surface="tuning.optuna.space",
            )
