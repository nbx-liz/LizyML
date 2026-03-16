"""SearchSpace — optuna-independent representation of hyperparameter search spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from lizyml.core.exceptions import ErrorCode, LizyMLError

DimCategory = Literal["model", "smart", "training"]


@dataclass(frozen=True)
class FloatDim:
    """A continuous float hyperparameter dimension."""

    name: str
    low: float
    high: float
    log: bool = False
    category: DimCategory = "model"


@dataclass(frozen=True)
class IntDim:
    """An integer hyperparameter dimension."""

    name: str
    low: int
    high: int
    log: bool = False
    category: DimCategory = "model"


@dataclass(frozen=True)
class CategoricalDim:
    """A categorical hyperparameter dimension."""

    name: str
    choices: tuple[Any, ...]
    category: DimCategory = "model"


SearchDim = FloatDim | IntDim | CategoricalDim


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
