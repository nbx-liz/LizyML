"""LightGBM defaults — task mappings, common defaults, and default search space."""

from __future__ import annotations

from typing import Any

from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim, SearchDim

TaskType = str

# Maps task → objective
_TASK_OBJECTIVE: dict[str, str] = {
    "regression": "huber",
    "binary": "binary",
    "multiclass": "multiclass",
}

# Maps task → eval_metric list
_TASK_METRIC: dict[str, list[str]] = {
    "regression": ["huber", "mae", "mape"],
    "binary": ["auc", "binary_logloss"],
    "multiclass": ["auc_mu", "multi_logloss"],
}

# Common LightGBM defaults (also used by resolve_smart_params)
_COMMON_DEFAULTS: dict[str, Any] = {
    "boosting": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.001,
    "max_depth": 5,
    "max_bin": 511,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 10,
    "lambda_l1": 0.0,
    "lambda_l2": 0.000001,
    "first_metric_only": False,
}

_OBJECTIVE_CHOICES: dict[str, tuple[str, ...]] = {
    "regression": ("huber", "fair"),
    "binary": ("binary",),
    "multiclass": ("multiclass", "multiclassova"),
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

    Only model-level LightGBM parameters belong here.  Smart params
    (``auto_num_leaves`` etc.) are handled via ``base_smart_params`` in the
    tune objective and must **not** leak into model params (#76).

    Args:
        task: ML task type.

    Returns:
        Dict with ``first_metric_only`` and ``metric``.
    """
    return {
        "first_metric_only": True,
        "metric": _TASK_METRIC.get(task, ["huber", "mae", "mape"]),
    }
