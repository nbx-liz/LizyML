"""Metric bridge — mapping, validation, and feval generation for LightGBM.

Bridges the gap between LizyML evaluation metric names and LightGBM
training metric names (H-0064).

Responsibilities:
- Translate LizyML metric names → LightGBM native names
- Validate metric names against per-task whitelist
- Generate feval callables for LizyML-only metrics
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.task import TaskType
from lizyml.metrics.base import BaseMetric
from lizyml.metrics.registry import MetricEntry, get_metric, parse_metric_entries

# ---------------------------------------------------------------------------
# Phase 1: LizyML → LightGBM name mapping
# ---------------------------------------------------------------------------

# Maps LizyML metric name → {task: LightGBM name}
_LIZYML_TO_LGBM: dict[str, dict[str, str]] = {
    "logloss": {"binary": "binary_logloss", "multiclass": "multi_logloss"},
    "auc_pr": {"binary": "average_precision", "multiclass": "average_precision"},
}


def translate_metric(name: str, task: TaskType) -> str:
    """Translate a LizyML metric name to its LightGBM equivalent.

    If no mapping exists, returns the name unchanged.

    Args:
        name: Metric name (LizyML or LightGBM).
        task: ML task type (``"regression"``, ``"binary"``, ``"multiclass"``).

    Returns:
        LightGBM-compatible metric name.
    """
    task_map = _LIZYML_TO_LGBM.get(name)
    if task_map is not None:
        return task_map.get(task, name)
    return name


# ---------------------------------------------------------------------------
# Phase 2: Whitelist validation
# ---------------------------------------------------------------------------

# LightGBM native metrics valid per task (LightGBM 4.x)
_LGBM_NATIVE_METRICS: dict[str, frozenset[str]] = {
    "regression": frozenset(
        [
            "l1",
            "mae",
            "mean_absolute_error",
            "regression_l1",
            "l2",
            "mse",
            "mean_squared_error",
            "regression_l2",
            "regression",
            "rmse",
            "root_mean_squared_error",
            "l2_root",
            "quantile",
            "mape",
            "mean_absolute_percentage_error",
            "huber",
            "fair",
            "poisson",
            "gamma",
            "gamma_deviance",
            "tweedie",
        ]
    ),
    "binary": frozenset(
        [
            "binary_logloss",
            "binary",
            "binary_error",
            "auc",
            "average_precision",
            "cross_entropy",
            "xentropy",
            "cross_entropy_lambda",
            "xentlambda",
            "kullback_leibler",
            "kldiv",
        ]
    ),
    "multiclass": frozenset(
        [
            "multi_logloss",
            "multiclass",
            "softmax",
            "multiclassova",
            "multiclass_ova",
            "ova",
            "ovr",
            "multi_error",
            # H-0079 Phase 3: ``auc`` was incorrectly listed here, but
            # LightGBM 4.x raises "Multiclass objective and metrics don't
            # match" when ``auc`` reaches multiclass ``params["metric"]``.
            # The post-fit ``MetricRegistry`` still computes multiclass
            # AUC via sklearn OvR (``Model.evaluate(metrics=["auc"])`` keeps
            # working); only the lgb.train passthrough is rejected.
            "auc_mu",
        ]
    ),
}

# Metrics that require feval (no LightGBM native equivalent) per task.
# NOTE: r2 is documented in LightGBM master but not shipped in 4.6.0 binary.
# Kept here until the upstream release is confirmed and minimum version bumped.
_FEVAL_METRICS: dict[str, frozenset[str]] = {
    "regression": frozenset(["rmsle", "r2", "smape", "wape"]),
    "binary": frozenset(["f1", "brier", "ece", "precision_at_k", "accuracy"]),
    "multiclass": frozenset(["f1", "brier", "accuracy"]),
}

# All feval metric names across all tasks
_ALL_FEVAL_NAMES: frozenset[str] = frozenset().union(*_FEVAL_METRICS.values())


def validate_lgbm_metrics(
    metrics: list[str],
    task: TaskType,
    *,
    feval_names: frozenset[str],
) -> None:
    """Validate metric names against the LightGBM native whitelist.

    Metrics listed in *feval_names* bypass validation (they will be
    handled as custom feval functions).

    Args:
        metrics: List of metric names to validate.
        task: ML task type.
        feval_names: Set of metric names handled via feval (bypass).

    Raises:
        :class:`~lizyml.core.exceptions.LizyMLError` with
        ``CONFIG_INVALID`` when a metric is not valid for the task.
    """
    valid = _LGBM_NATIVE_METRICS.get(task, frozenset())
    for m in metrics:
        if m in feval_names:
            continue
        if m not in valid:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    f"Metric '{m}' is not a valid LightGBM metric for task "
                    f"'{task}'. Valid native metrics: {sorted(valid)}. "
                    f"Valid custom (feval) metrics: "
                    f"{sorted(_FEVAL_METRICS.get(task, frozenset()))}."
                ),
                context={"metric": m, "task": task},
            )


# ---------------------------------------------------------------------------
# Phase 3: feval custom function generation
# ---------------------------------------------------------------------------


def _sigmoid(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Numerically stable sigmoid using clip to avoid overflow warnings."""
    clipped = np.clip(x, -500, 500)
    result: npt.NDArray[Any] = 1.0 / (1.0 + np.exp(-clipped))
    return result


def _softmax(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Row-wise softmax for 2D array."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    result: npt.NDArray[Any] = e_x / e_x.sum(axis=1, keepdims=True)
    return result


def _metric_display_name(metric: BaseMetric, kwargs: dict[str, Any]) -> str:
    """Build a display name for a metric, appending params if present.

    Examples::

        _metric_display_name(PrecisionAtK(k=20), {"k": 20})
        # -> "precision_at_k (k=20)"

        _metric_display_name(RMSE(), {})
        # -> "rmse"
    """
    if not kwargs:
        return metric.name
    params_str = ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return f"{metric.name} ({params_str})"


def _build_feval(
    metric: BaseMetric,
    task: TaskType,
    num_class: int | None = None,
    *,
    display_name: str | None = None,
) -> Callable[..., tuple[str, float, bool]]:
    """Create a LightGBM feval callable from a BaseMetric.

    The callable transforms raw LightGBM predictions (logits for binary,
    flattened logits for multiclass) into probabilities before delegating
    to the metric's ``__call__``.

    Args:
        metric: A LizyML BaseMetric instance.
        task: ML task type.
        num_class: Number of classes (required for multiclass).
        display_name: Override name returned in the feval tuple.  When
            ``None``, falls back to ``metric.name``.

    Returns:
        A callable with signature
        ``(y_pred: ndarray, dataset: lgb.Dataset) -> (name, value, is_higher_better)``.
    """
    import lightgbm as lgb

    feval_name = display_name or metric.name

    def feval_fn(
        y_pred: npt.NDArray[Any], dataset: lgb.Dataset
    ) -> tuple[str, float, bool]:
        y_true = dataset.get_label()
        if y_true is None:  # pragma: no cover
            raise RuntimeError(
                "feval received a Dataset with no label — "
                "ensure the Dataset was constructed with label data."
            )

        if task == "binary":
            proba = _sigmoid(y_pred)
        elif task == "multiclass":
            if num_class is None:  # pragma: no cover
                raise RuntimeError("num_class is required for multiclass feval.")
            proba = y_pred.reshape(-1, num_class)
            if proba.shape[0] != len(y_true):
                raise RuntimeError(
                    f"feval reshape mismatch: expected ({len(y_true)}, "
                    f"{num_class}), got {proba.shape}. "
                    f"num_class may be incorrect."
                )
            proba = _softmax(proba)
        else:
            # regression: predictions are direct values
            proba = y_pred

        # For metrics that don't need probabilities, convert to labels
        if not metric.needs_proba and task in ("binary", "multiclass"):
            if proba.ndim == 2:
                pred = proba.argmax(axis=1).astype(np.int64)
            else:
                pred = (proba >= 0.5).astype(np.int64)
        else:
            pred = proba

        value = metric(np.asarray(y_true), np.asarray(pred))
        return (feval_name, value, metric.greater_is_better)

    return feval_fn


def resolve_metrics(
    metrics: list[MetricEntry],
    task: TaskType,
    num_class: int | None = None,
) -> tuple[list[str], list[Callable[..., tuple[str, float, bool]]], list[str]]:
    """Split metrics into native LightGBM names and feval callables.

    Accepts both plain strings and ``MetricEntry`` dicts (H-0065).

    Steps:
    1. Translate LizyML names → LightGBM names
    2. Classify each as native or feval
    3. Validate all metrics (native against whitelist, feval against task)
    4. Build feval callables for non-native metrics

    Args:
        metrics: List of metric names or ``{name: {param: value}}`` dicts.
        task: ML task type.
        num_class: Number of classes (needed for multiclass feval).

    Returns:
        ``(native_metrics, feval_callables, feval_display_names)`` tuple.

    Raises:
        :class:`~lizyml.core.exceptions.LizyMLError` with
        ``CONFIG_INVALID`` for unknown or task-incompatible metrics.
    """
    parsed = parse_metric_entries(metrics)
    feval_for_task = _FEVAL_METRICS.get(task, frozenset())
    native: list[str] = []
    fevals: list[Callable[..., tuple[str, float, bool]]] = []
    feval_display_names: list[str] = []
    feval_names_for_validation: set[str] = set()

    # Reject duplicate metric names
    seen: set[str] = set()
    for name, _kwargs in parsed:
        if name in seen:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=f"Duplicate metric '{name}' in metrics list.",
                context={"metric": name},
            )
        seen.add(name)

    for raw_name, kwargs in parsed:
        # Check if the raw name is a feval metric BEFORE translation
        if raw_name in feval_for_task:
            feval_names_for_validation.add(raw_name)
            metric_obj = get_metric(raw_name, **kwargs)
            display = _metric_display_name(metric_obj, kwargs)
            fevals.append(
                _build_feval(metric_obj, task, num_class, display_name=display)
            )
            feval_display_names.append(display)
        elif raw_name in _ALL_FEVAL_NAMES and raw_name not in feval_for_task:
            # feval metric but wrong task
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    f"Metric '{raw_name}' is not valid for task '{task}'. "
                    f"Valid custom metrics for '{task}': {sorted(feval_for_task)}."
                ),
                context={"metric": raw_name, "task": task},
            )
        else:
            translated = translate_metric(raw_name, task)
            native.append(translated)

    # Validate native metrics against whitelist
    validate_lgbm_metrics(
        native, task, feval_names=frozenset(feval_names_for_validation)
    )

    return native, fevals, feval_display_names
