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


_OBJECTIVE_CHOICES: dict[str, tuple[str, ...]] = {
    "regression": ("huber", "fair"),
    "binary": ("binary",),
    "multiclass": ("multiclass", "multiclassova"),
}

_FIXED_METRIC: dict[str, list[str]] = {
    "regression": ["huber", "mae", "mape"],
    "binary": ["auc", "binary_logloss"],
    "multiclass": ["auc_mu", "multi_logloss"],
}


def default_space(task: str) -> list[SearchDim]:
    """Return the PLAN-specified default search space for LightGBM.

    Args:
        task: ML task type (``"regression"``, ``"binary"``, ``"multiclass"``).

    Returns:
        List of 10 SearchDim across model / smart / training categories.
    """
    dims: list[SearchDim] = [
        # -- model --
        CategoricalDim(
            "objective",
            _OBJECTIVE_CHOICES.get(task, ("huber",)),
            category="model",
        ),
        IntDim("n_estimators", 600, 2500, category="model"),
        FloatDim("learning_rate", 0.0001, 0.1, log=True, category="model"),
        IntDim("max_depth", 3, 12, category="model"),
        FloatDim("feature_fraction", 0.5, 1.0, category="model"),
        FloatDim("bagging_fraction", 0.5, 1.0, category="model"),
        # -- smart --
        FloatDim("num_leaves_ratio", 0.5, 1.0, category="smart"),
        FloatDim("min_data_in_leaf_ratio", 0.01, 0.2, category="smart"),
        # -- training --
        IntDim("early_stopping_rounds", 40, 240, category="training"),
        FloatDim("validation_ratio", 0.1, 0.3, category="training"),
    ]
    return dims


def default_fixed_params(task: str) -> dict[str, Any]:
    """Return fixed parameters applied to every trial when using default space.

    Args:
        task: ML task type.

    Returns:
        Dict with ``auto_num_leaves``, ``first_metric_only``, and ``metric``.
    """
    return {
        "auto_num_leaves": True,
        "first_metric_only": True,
        "metric": _FIXED_METRIC.get(task, ["huber", "mae", "mape"]),
    }


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
