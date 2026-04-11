"""SearchSpace — optuna-independent representation of hyperparameter search spaces.

Type definitions (``SearchDim``, ``FloatDim``, ``IntDim``, ``CategoricalDim``,
``DimCategory``) live in ``core/types/search_dim.py`` (Foundation layer) and are
re-exported here for backward compatibility.
"""

from __future__ import annotations

import math
from typing import Any

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import (
    CategoricalDim,
    DimCategory,
    FloatDim,
    IntDim,
    SearchDim,
)
from lizyml.core.types.tuning_result import BoundaryDimStatus, BoundaryReport

__all__ = [
    "CategoricalDim",
    "FloatDim",
    "IntDim",
    "SearchDim",
    "detect_boundary",
    "expand_dims",
    "parse_space",
    "split_by_category",
    "suggest_params",
]


_ALLOWED_CHOICE_TYPES = (type(None), bool, int, float, str)


def _validate_categorical_choices(name: str, choices: list[Any]) -> None:
    """Validate that every element in *choices* is a scalar type.

    Optuna's ``CategoricalDistribution`` requires each choice to be
    ``None | bool | int | float | str``.  Non-scalar values (e.g. a nested
    list produced by YAML ``- [a, b]``) are rejected early with a clear
    error message.
    """
    for i, val in enumerate(choices):
        if not isinstance(val, _ALLOWED_CHOICE_TYPES):
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    f"Categorical dim '{name}' has invalid choice at index {i}: "
                    f"got {val!r} (type={type(val).__name__}). "
                    f"Each choice must be a scalar (str, int, float, bool, or None). "
                    f"Hint: flatten nested lists in your YAML config "
                    f'— use "- value" instead of "- [value1, value2]".'
                ),
                context={"param": name, "index": i, "bad_value": str(val)},
            )


def parse_space(space: dict[str, Any]) -> list[SearchDim]:
    """Parse a config-style space dict into typed SearchDim instances.

    Space dict format::

        {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 16, "high": 256},
            "subsample": {"type": "categorical", "choices": [0.6, 0.8, 1.0]},
        }

    Args:
        space: Raw search space dict from config.

    Returns:
        List of typed SearchDim instances.

    Raises:
        LizyMLError with CONFIG_INVALID for unknown types or missing keys.
    """
    dims: list[SearchDim] = []
    for name, spec in space.items():
        dim_type: str = spec.get("type", "")
        category: DimCategory = spec.get("category", "model")
        if dim_type == "float":
            dims.append(
                FloatDim(
                    name=name,
                    low=float(spec["low"]),
                    high=float(spec["high"]),
                    log=bool(spec.get("log", False)),
                    category=category,
                )
            )
        elif dim_type == "int":
            dims.append(
                IntDim(
                    name=name,
                    low=int(spec["low"]),
                    high=int(spec["high"]),
                    log=bool(spec.get("log", False)),
                    category=category,
                )
            )
        elif dim_type == "categorical":
            choices = spec.get("choices")
            if not choices:
                raise LizyMLError(
                    code=ErrorCode.CONFIG_INVALID,
                    user_message=(
                        f"Categorical dim '{name}' requires non-empty 'choices'."
                    ),
                    context={"param": name},
                )
            _validate_categorical_choices(name, choices)
            dims.append(
                CategoricalDim(name=name, choices=tuple(choices), category=category)
            )
        else:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    f"Unknown search space type '{dim_type}' for param '{name}'. "
                    f"Expected 'float', 'int', or 'categorical'."
                ),
                context={"param": name, "type": dim_type},
            )
    return dims


def split_by_category(
    trial_params: dict[str, Any],
    dims: list[SearchDim],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split trial params into (model, smart, training) dicts by dim category.

    Args:
        trial_params: Sampled parameters from a trial.
        dims: Search dimensions with category attributes.

    Returns:
        Tuple of (model_params, smart_params, training_params).
    """
    cat_map = {d.name: d.category for d in dims}
    model_p: dict[str, Any] = {}
    smart_p: dict[str, Any] = {}
    training_p: dict[str, Any] = {}
    for name, val in trial_params.items():
        cat = cat_map.get(name, "model")
        if cat == "smart":
            smart_p[name] = val
        elif cat == "training":
            training_p[name] = val
        else:
            model_p[name] = val
    return model_p, smart_p, training_p


def suggest_params(trial: Any, dims: list[SearchDim]) -> dict[str, Any]:
    """Sample a parameter dict from an optuna trial.

    Args:
        trial: An ``optuna.Trial`` instance.
        dims: List of search dimensions.

    Returns:
        Dict of sampled hyperparameter values.
    """
    params: dict[str, Any] = {}
    for dim in dims:
        if isinstance(dim, FloatDim):
            params[dim.name] = trial.suggest_float(
                dim.name, dim.low, dim.high, log=dim.log
            )
        elif isinstance(dim, IntDim):
            params[dim.name] = trial.suggest_int(
                dim.name, dim.low, dim.high, log=dim.log
            )
        else:  # CategoricalDim
            params[dim.name] = trial.suggest_categorical(dim.name, dim.choices)
    return params


# ---------------------------------------------------------------------------
# Boundary detection and expansion (H-0068)
# ---------------------------------------------------------------------------


_LOG_EXPANSION_FACTOR = 3.0
"""Expansion factor for log-scale dimensions (applied in log space)."""


def _position_pct(best: float, low: float, high: float, *, log: bool) -> float:
    """Compute relative position of *best* within [low, high] as 0.0–1.0."""
    if log and low > 0 and high > 0 and best > 0:
        log_low, log_high, log_best = math.log(low), math.log(high), math.log(best)
        span = log_high - log_low
        if span <= 0:
            return 0.5
        return (log_best - log_low) / span
    span = high - low
    if span <= 0:
        return 0.5
    return (best - low) / span


def _detect_edge(position: float, threshold: float) -> str:
    """Return 'lower', 'upper', or 'none' based on position and threshold."""
    if position < threshold:
        return "lower"
    if position > (1.0 - threshold):
        return "upper"
    return "none"


def _expand_range(
    low: float,
    high: float,
    edge: str,
    *,
    log: bool,
) -> tuple[float, float]:
    """Expand *low*/*high* asymmetrically toward *edge*."""
    if edge == "lower":
        new_low = low / _LOG_EXPANSION_FACTOR if log and low > 0 else low - (high - low)
        return new_low, high
    if edge == "upper":
        new_high = (
            high * _LOG_EXPANSION_FACTOR if log and high > 0 else high + (high - low)
        )
        return low, new_high
    return low, high


def detect_boundary(
    dims: list[SearchDim],
    best_params: dict[str, Any],
    threshold: float = 0.05,
) -> BoundaryReport:
    """Detect which dimensions have best params near their boundary (H-0068).

    Args:
        dims: Current search space dimensions.
        best_params: Best parameter values from the previous tuning round.
        threshold: Edge detection threshold (0.0–1.0). A best value within
            this fraction of the range from either edge is considered near
            the boundary.

    Returns:
        BoundaryReport with per-dimension analysis.
    """
    statuses: list[BoundaryDimStatus] = []
    expanded_names: list[str] = []

    for dim in dims:
        best_val = best_params.get(dim.name)

        if isinstance(dim, CategoricalDim):
            statuses.append(
                BoundaryDimStatus(
                    name=dim.name,
                    best_value=best_val,
                    low=None,
                    high=None,
                    position_pct=None,
                    edge="none",
                    expanded=False,
                    new_low=None,
                    new_high=None,
                )
            )
            continue

        # FloatDim or IntDim
        low: float = float(dim.low)
        high: float = float(dim.high)
        is_log = dim.log
        best_num = float(best_val) if best_val is not None else (low + high) / 2

        position = _position_pct(best_num, low, high, log=is_log)
        edge = _detect_edge(position, threshold)
        should_expand = edge != "none"

        new_low: float | int | None = None
        new_high: float | int | None = None
        if should_expand:
            nl, nh = _expand_range(low, high, edge, log=is_log)
            if isinstance(dim, IntDim):
                nl = max(1, int(math.floor(nl)))
                nh = int(math.ceil(nh))
            new_low = nl
            new_high = nh
            expanded_names.append(dim.name)

        statuses.append(
            BoundaryDimStatus(
                name=dim.name,
                best_value=best_val,
                low=dim.low,
                high=dim.high,
                position_pct=round(position, 4),
                edge=edge,
                expanded=should_expand,
                new_low=new_low,
                new_high=new_high,
            )
        )

    return BoundaryReport(
        dims=tuple(statuses),
        expanded_names=tuple(expanded_names),
    )


def expand_dims(
    dims: list[SearchDim],
    report: BoundaryReport,
) -> list[SearchDim]:
    """Return a new list of SearchDim with boundary-expanded ranges (H-0068).

    Only dimensions flagged ``expanded=True`` in *report* are modified.
    Non-expanded dimensions are returned unchanged.

    Args:
        dims: Original search space dimensions.
        report: Boundary report from :func:`detect_boundary`.

    Returns:
        New list of SearchDim with expanded ranges where applicable.
    """
    expansion_map: dict[str, BoundaryDimStatus] = {
        s.name: s for s in report.dims if s.expanded
    }
    new_dims: list[SearchDim] = []
    for dim in dims:
        status = expansion_map.get(dim.name)
        if status is None:
            new_dims.append(dim)
            continue

        if (
            isinstance(dim, FloatDim)
            and status.new_low is not None
            and status.new_high is not None
        ):
            new_dims.append(
                FloatDim(
                    name=dim.name,
                    low=float(status.new_low),
                    high=float(status.new_high),
                    log=dim.log,
                    category=dim.category,
                )
            )
        elif (
            isinstance(dim, IntDim)
            and status.new_low is not None
            and status.new_high is not None
        ):
            new_dims.append(
                IntDim(
                    name=dim.name,
                    low=int(status.new_low),
                    high=int(status.new_high),
                    log=dim.log,
                    category=dim.category,
                )
            )
        else:
            new_dims.append(dim)

    return new_dims
