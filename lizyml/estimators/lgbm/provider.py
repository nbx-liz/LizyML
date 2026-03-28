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

from lizyml.core.types.search_dim import SearchDim
from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.estimators.lgbm.adapter import LGBMAdapter, TaskType
from lizyml.estimators.lgbm.defaults import (
    _COMMON_DEFAULTS,
    default_fixed_params,
    default_space,
)
from lizyml.estimators.lgbm.smart_params import (
    resolve_ratio_params,
    resolve_smart_params,
)
from lizyml.features.pipeline_base import BaseFeaturePipeline
from lizyml.features.pipelines_native import NativeFeaturePipeline


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
        task: str,
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
        task: str,
        params: dict[str, Any],
        n_classes: int | None,
        early_stopping_rounds: int | None,
        seed: int,
    ) -> Callable[[], BaseEstimatorAdapter]:
        """Return a factory that creates a configured LGBMAdapter."""
        final_params = params
        lgbm_task: TaskType = task  # type: ignore[assignment]

        def make_estimator() -> LGBMAdapter:
            return LGBMAdapter(
                task=lgbm_task,
                params=final_params,
                num_class=n_classes,
                early_stopping_rounds=early_stopping_rounds,
                random_state=seed,
            )

        return make_estimator

    def build_pipeline_factory(self) -> Callable[[], BaseFeaturePipeline]:
        """Return a factory that creates NativeFeaturePipeline."""
        return NativeFeaturePipeline

    def default_space(self, task: str) -> list[SearchDim]:
        """Return the default LightGBM search space."""
        return default_space(task)

    def default_fixed_params(self, task: str) -> dict[str, Any]:
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
