"""LGBMProvider — EstimatorProvider implementation for LightGBM (H-0053).

Bridges the Facade (model.py) with LightGBM-specific logic so that
model.py has zero LightGBM imports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import SearchDim
from lizyml.core.types.task import TaskType
from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.estimators.lgbm.adapter import LGBMAdapter
from lizyml.estimators.lgbm.defaults import (
    _COMMON_DEFAULTS,
    TASK_COMPATIBLE_OBJECTIVES,
    default_fixed_params,
    default_space,
)
from lizyml.estimators.lgbm.smart_params import (
    resolve_ratio_params,
    resolve_smart_params,
)
from lizyml.estimators.provider import ExportParams, MetricChoices
from lizyml.features.pipeline_base import BaseFeaturePipeline
from lizyml.features.pipelines_native import NativeFeaturePipeline

# H-0078: LightGBM-meaningful per-parameter bounds for boundary expansion.
# Values reflect LightGBM's documented limits and physically-meaningful
# ranges; they keep ``expand_dims`` from drifting into ranges that crash
# the booster or have no effect.
_LGBM_PARAMETER_BOUNDS: dict[str, dict[str, float | int]] = {
    "learning_rate": {"min": 1e-8, "max": 1.0},
    "feature_fraction": {"min": 1e-3, "max": 1.0},
    "bagging_fraction": {"min": 1e-3, "max": 1.0},
    "num_leaves_ratio": {"min": 0.1, "max": 2.0},
    "min_data_in_leaf_ratio": {"min": 1e-4, "max": 0.5},
    "min_data_in_bin_ratio": {"min": 1e-4, "max": 0.5},
    "validation_ratio": {"min": 0.05, "max": 0.5},
    "lambda_l1": {"min": 0.0, "max": 100.0},
    "lambda_l2": {"min": 0.0, "max": 100.0},
    "n_estimators": {"min": 10, "max": 10000},
    "max_depth": {"min": -1, "max": 30},
    "max_bin": {"min": 2, "max": 8192},
    "bagging_freq": {"min": 0, "max": 100},
    "early_stopping_rounds": {"min": 1, "max": 5000},
    "seed": {"min": 0, "max": 2**31 - 1},
}

# H-0079: canonical objective tuples per task, ordered for stable UI display.
# Source of truth lives in ``defaults.TASK_COMPATIBLE_OBJECTIVES`` (the
# whitelist used by ``LGBMAdapter._build_params`` for cross-task validation).
# This module only adds an explicit ordering for surface APIs.
_LGBM_OBJECTIVE_CHOICES: dict[str, tuple[str, ...]] = {
    "regression": (
        "regression",
        "regression_l1",
        "huber",
        "fair",
        "poisson",
        "quantile",
        "mape",
        "gamma",
        "tweedie",
    ),
    "binary": (
        "binary",
        "cross_entropy",
        "cross_entropy_lambda",
    ),
    "multiclass": (
        "multiclass",
        "multiclassova",
    ),
}

# H-0079: canonical metric tuples per task / source. Aliases such as
# LightGBM's ``l1`` / ``l2`` / ``mse`` / ``mean_absolute_error`` /
# ``regression_l1`` / ``ova`` / ``ovr`` are still **accepted** at config
# input time by ``metric_bridge.validate_lgbm_metrics``; the choice tables
# below intentionally surface only the canonical short form so downstream
# UIs render a single picker per metric.
_LGBM_NATIVE_METRIC_CHOICES: dict[str, tuple[str, ...]] = {
    "regression": (
        "rmse",
        "mae",
        "mape",
        "huber",
        "fair",
        "poisson",
        "quantile",
        "gamma",
        "gamma_deviance",
        "tweedie",
    ),
    "binary": (
        "binary_logloss",
        "binary_error",
        "auc",
        "average_precision",
        "cross_entropy",
        "cross_entropy_lambda",
        "kullback_leibler",
    ),
    "multiclass": (
        "multi_logloss",
        "multi_error",
        "auc_mu",
        "multiclassova",
    ),
}

_LGBM_FEVAL_METRIC_CHOICES: dict[str, tuple[str, ...]] = {
    "regression": ("rmsle", "r2", "smape", "wape"),
    "binary": ("f1", "brier", "ece", "precision_at_k", "accuracy"),
    "multiclass": ("f1", "brier", "accuracy"),
}


def _validate_objective_consistency() -> None:
    """Module-load self-check: per-task surface tuples must match the
    Phase-1 whitelist exactly. Catches drift between the two sources of
    truth before they reach a user (Phase 3 collapses these into one,
    but during Phase 2 both must agree).

    Raises ``RuntimeError`` at module load time, which renders LizyML
    un-importable. Intentional fail-fast: a drift means
    ``LGBMProvider.objective_choices`` could surface a value that
    ``LGBMAdapter._build_params`` rejects (or vice versa), producing
    impossible-to-debug user reports. Better to fail at process start.
    """
    for task, surface in _LGBM_OBJECTIVE_CHOICES.items():
        whitelist = TASK_COMPATIBLE_OBJECTIVES[task]
        if set(surface) != set(whitelist):
            raise RuntimeError(  # pragma: no cover — load-time invariant
                f"H-0079 drift: _LGBM_OBJECTIVE_CHOICES[{task!r}]="
                f"{sorted(surface)} differs from TASK_COMPATIBLE_OBJECTIVES="
                f"{sorted(whitelist)}."
            )


def _validate_metric_consistency() -> None:
    """Module-load self-check: every metric surfaced via
    ``metric_choices()`` must be reachable at fit-time (H-0079 follow-up).

    Mirrors ``_validate_objective_consistency`` for the metric side:

    - Each ``_LGBM_NATIVE_METRIC_CHOICES[task]`` entry must be in
      ``metric_bridge._LGBM_NATIVE_METRICS[task]`` so that
      ``validate_lgbm_metrics`` accepts it before training.
    - Each ``_LGBM_FEVAL_METRIC_CHOICES[task]`` entry must be in
      ``metric_bridge._FEVAL_METRICS[task]`` so that ``resolve_metrics``
      wires it as a feval callable.

    Drift here would mean a downstream UI offers a metric that the
    library subsequently rejects — exactly the failure mode the
    ``metric_choices`` API was introduced to eliminate.
    """
    # Local import to avoid circular dependency at module top.
    from lizyml.estimators.lgbm.metric_bridge import (
        _FEVAL_METRICS,
        _LGBM_NATIVE_METRICS,
    )

    for task, surface in _LGBM_NATIVE_METRIC_CHOICES.items():
        whitelist = _LGBM_NATIVE_METRICS.get(task, frozenset())
        unsupported = set(surface) - set(whitelist)
        if unsupported:
            raise RuntimeError(  # pragma: no cover — load-time invariant
                f"H-0079 drift: _LGBM_NATIVE_METRIC_CHOICES[{task!r}] "
                f"includes {sorted(unsupported)} which are not in "
                f"metric_bridge._LGBM_NATIVE_METRICS[{task!r}]."
            )
    for task, surface in _LGBM_FEVAL_METRIC_CHOICES.items():
        whitelist = _FEVAL_METRICS.get(task, frozenset())
        unsupported = set(surface) - set(whitelist)
        if unsupported:
            raise RuntimeError(  # pragma: no cover — load-time invariant
                f"H-0079 drift: _LGBM_FEVAL_METRIC_CHOICES[{task!r}] "
                f"includes {sorted(unsupported)} which are not in "
                f"metric_bridge._FEVAL_METRICS[{task!r}]."
            )


_validate_objective_consistency()
_validate_metric_consistency()


class LGBMProvider:
    """EstimatorProvider implementation for LightGBM.

    Encapsulates all LightGBM-specific knowledge that the Facade needs:
    config extraction, smart param resolution, estimator/pipeline factories,
    and default search space.
    """

    def extract_model_params(self, model_cfg: Any) -> dict[str, Any]:
        """Extract native model parameters from LGBMConfig."""
        return dict(model_cfg.params)

    def extract_smart_params(self, model_cfg: Any) -> dict[str, Any]:
        """Extract smart parameter fields from LGBMConfig as a plain dict."""
        return {
            "auto_num_leaves": model_cfg.auto_num_leaves,
            "num_leaves_ratio": model_cfg.num_leaves_ratio,
            "min_data_in_leaf_ratio": model_cfg.min_data_in_leaf_ratio,
            "min_data_in_bin_ratio": model_cfg.min_data_in_bin_ratio,
            "feature_weights": model_cfg.feature_weights,
            "balanced": model_cfg.balanced,
        }

    def resolve_smart_params(
        self,
        smart: dict[str, Any],
        effective_params: dict[str, Any],
        n_rows: int,
        feature_names: list[str],
        y: pd.Series,
        task: TaskType,
    ) -> tuple[dict[str, Any], npt.NDArray[np.float64] | None]:
        """Resolve smart parameters to native LightGBM parameters.

        Merges ``_COMMON_DEFAULTS`` into ``effective_params`` before resolution.
        """
        effective = {**_COMMON_DEFAULTS, **effective_params}
        return resolve_smart_params(
            smart=smart,
            effective_params=effective,
            n_rows=n_rows,
            feature_names=feature_names,
            y=y,
            task=task,
        )

    def build_ratio_resolver(
        self,
        smart: dict[str, Any],
    ) -> Callable[[int], dict[str, Any]] | None:
        """Build a per-fold ratio resolver from smart params."""
        leaf_ratio = smart.get("min_data_in_leaf_ratio")
        bin_ratio = smart.get("min_data_in_bin_ratio")
        if leaf_ratio is None and bin_ratio is None:
            return None
        return lambda n: resolve_ratio_params(leaf_ratio, bin_ratio, n)  # noqa: E731

    def build_estimator_factory(
        self,
        task: TaskType,
        params: dict[str, Any],
        n_classes: int | None,
        early_stopping_rounds: int | None,
        seed: int,
    ) -> Callable[[], BaseEstimatorAdapter]:
        """Return a factory that creates a configured LGBMAdapter."""
        final_params = params

        def make_estimator() -> LGBMAdapter:
            return LGBMAdapter(
                task=task,
                params=final_params,
                num_class=n_classes,
                early_stopping_rounds=early_stopping_rounds,
                random_state=seed,
            )

        return make_estimator

    def build_pipeline_factory(self) -> Callable[[], BaseFeaturePipeline]:
        """Return a factory that creates NativeFeaturePipeline."""
        return NativeFeaturePipeline

    def default_space(self, task: TaskType) -> list[SearchDim]:
        """Return the default LightGBM search space."""
        return default_space(task)

    def default_fixed_params(self, task: TaskType) -> dict[str, Any]:
        """Return fixed params for default search space."""
        return default_fixed_params(task)

    def runtime_deps(self) -> dict[str, str]:
        """Return LightGBM package version."""
        try:
            from importlib.metadata import version as pkg_version

            ver = pkg_version("lightgbm")
        except Exception:
            ver = "unknown"
        return {"lightgbm": ver}

    def params_summary(
        self,
        model: BaseEstimatorAdapter,
        model_cfg: Any,
    ) -> list[dict[str, Any]]:
        """Return parameter rows for params_table().

        Includes smart params, resolved booster params, and task-specific params.
        """
        rows: list[dict[str, Any]] = []

        # Smart params from config
        smart = self.extract_smart_params(model_cfg)
        for k, v in smart.items():
            rows.append({"parameter": k, "value": v})

        # Resolved booster params (from fold 0)
        native = model.get_native_model()
        booster_params = getattr(native, "params", {})
        for k in [
            "objective",
            "metric",
            "learning_rate",
            "max_depth",
            "num_leaves",
            "min_data_in_leaf",
            "min_data_in_bin",
            "max_bin",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "num_iterations",
        ]:
            v = booster_params.get(k)
            if v is not None:
                rows.append({"parameter": k, "value": v})

        # Task-specific params
        for k in ["scale_pos_weight", "num_class"]:
            v = booster_params.get(k)
            if v is not None:
                rows.append({"parameter": k, "value": v})

        # Feval metric display names (H-0065: shows k params etc.)
        if isinstance(model, LGBMAdapter) and model._feval_display_names:
            rows.append(
                {
                    "parameter": "feval_metrics",
                    "value": ", ".join(model._feval_display_names),
                }
            )

        return rows

    def parameter_bounds(self, task: TaskType) -> dict[str, dict[str, float | int]]:
        """Return LightGBM-meaningful bounds for boundary expansion (H-0078).

        These limits constrain ``expand_dims`` so that ``re_tune=True``
        cannot grow ``learning_rate`` past 1.0, ``feature_fraction`` past
        1.0, ``validation_ratio`` below ~0, etc. Bounds are not
        task-dependent for LightGBM.
        """
        del task  # bounds are identical across tasks for LightGBM
        return _LGBM_PARAMETER_BOUNDS

    def objective_choices(self, task: TaskType) -> tuple[str, ...]:
        """Canonical LightGBM objective names valid for *task* (H-0079).

        Returns the same set of values that ``LGBMAdapter._build_params``
        accepts (see ``TASK_COMPATIBLE_OBJECTIVES``), but as an ordered
        tuple suitable for downstream UI rendering. Unknown tasks return
        the empty tuple.
        """
        return _LGBM_OBJECTIVE_CHOICES.get(task, ())

    def metric_choices(self, task: TaskType) -> MetricChoices:
        """Canonical LightGBM metric names per task, split by source (H-0079).

        Returns ``{"native": (...), "feval": (...)}`` with deterministic
        ordering and no duplicates across the two keys. Aliases such as
        ``l1`` / ``l2`` are still accepted at config-input time but are
        not surfaced here.
        """
        return {
            "native": _LGBM_NATIVE_METRIC_CHOICES.get(task, ()),
            "feval": _LGBM_FEVAL_METRIC_CHOICES.get(task, ()),
        }

    def build_export_params(self, adapter: BaseEstimatorAdapter) -> ExportParams:
        """Build codegen-relevant params from a fitted ``LGBMAdapter`` (H-0073).

        Wraps the LGBM-private ``_build_params()`` and the feval metadata
        extraction so that ``_model_persistence.py`` does not need to import
        ``LGBMAdapter`` or call private methods.

        Note:
            This method intentionally calls ``adapter._build_params()`` —
            an attribute of the same ``lgbm/`` subpackage. Treat the
            (provider, adapter) pair as a unit when refactoring; the
            access is co-located by package boundary, but type checkers
            cannot enforce it. If ``LGBMAdapter._build_params()`` ever
            grows new return values, update both this call site and
            ``LGBMAdapter`` in the same change.

        Raises:
            LizyMLError with ``UNSUPPORTED_TASK`` when the supplied adapter
            is not an ``LGBMAdapter``.
        """
        if not isinstance(adapter, LGBMAdapter):
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message=(
                    "LGBMProvider.build_export_params() requires an "
                    "LGBMAdapter instance."
                ),
                context={"adapter_type": type(adapter).__name__},
            )
        params, num_boost_round, _, _ = adapter._build_params()
        feval_metadata = _extract_feval_metadata(adapter)
        return ExportParams(
            params=params,
            num_boost_round=num_boost_round,
            feval_metadata=feval_metadata,
        )


def _extract_feval_metadata(
    adapter: LGBMAdapter,
) -> list[dict[str, Any]]:
    """Extract feval metric metadata from adapter params (H-0066/H-0073).

    Reads the user-specified ``metric`` from the adapter's params dict,
    identifies which are feval metrics (not LightGBM native), and returns
    serializable metadata for each.

    Moved from ``_model_persistence.py`` so that the persistence layer
    no longer reaches into ``LGBMAdapter`` internals.
    """
    from lizyml.estimators.lgbm.metric_bridge import _FEVAL_METRICS
    from lizyml.metrics.registry import get_metric, parse_metric_entries

    user_metric = adapter.params.get("metric")
    if not user_metric:
        return []

    if isinstance(user_metric, (str, dict)):
        user_metric = [user_metric]
    user_metric = [m for m in user_metric if m]
    if not user_metric:
        return []

    feval_for_task = _FEVAL_METRICS.get(adapter.task, frozenset())
    parsed = parse_metric_entries(user_metric)

    result: list[dict[str, Any]] = []
    for name, kwargs in parsed:
        if name in feval_for_task:
            metric_obj = get_metric(name, **kwargs)
            result.append(
                {
                    "name": name,
                    "params": kwargs,
                    "greater_is_better": metric_obj.greater_is_better,
                    "needs_proba": metric_obj.needs_proba,
                }
            )
    return result
