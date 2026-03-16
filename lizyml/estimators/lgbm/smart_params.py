"""Smart parameter resolution for LightGBM (H-0021)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError


def _compute_num_leaves(max_depth: int | None, ratio: float) -> int:
    """Compute num_leaves from max_depth and ratio."""
    base = 131072 if max_depth is None or max_depth < 0 else 2**max_depth
    return max(8, min(131072, math.ceil(base * ratio)))


def _compute_ratio_param(n_rows: int, ratio: float) -> int:
    """Convert a ratio to an absolute count (min 1)."""
    return max(1, math.ceil(n_rows * ratio))


def resolve_smart_params(
    smart: dict[str, Any],
    effective_params: dict[str, Any],
    n_rows: int,
    feature_names: list[str],
    y: pd.Series,
    task: str,
) -> tuple[dict[str, Any], npt.NDArray[np.float64] | None]:
    """Resolve smart parameters to native LightGBM parameters.

    Unified function used by both ``fit()`` and ``tune()`` (H-0050).
    The *smart* dict is typically produced by ``extract_smart_params()``
    and optionally merged with tuning best_smart_params overrides.

    Args:
        smart: Dict of smart parameter values (from Config or tuning).
        effective_params: Merged params (defaults + user + best_params).
        n_rows: Number of training rows.
        feature_names: List of feature column names.
        y: Target series.
        task: ML task type.

    Returns:
        Tuple of (resolved native params dict, sample_weight array or None).
    """
    resolved: dict[str, Any] = {}
    sample_weight: npt.NDArray[np.float64] | None = None

    # auto_num_leaves
    if smart.get("auto_num_leaves", False):
        ratio = smart.get("num_leaves_ratio", 1.0)
        resolved["num_leaves"] = _compute_num_leaves(
            effective_params.get("max_depth"), ratio
        )

    # NOTE: ratio params (min_data_in_leaf_ratio, min_data_in_bin_ratio) are
    # resolved per-fold via resolve_ratio_params() using inner_train size (H-0036).

    # feature_weights
    fw = smart.get("feature_weights")
    if fw is not None:
        unknown = set(fw) - set(feature_names)
        if unknown:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=f"Unknown features in feature_weights: {sorted(unknown)}",
                context={"unknown_features": sorted(unknown)},
            )
        weights = [fw.get(f, 1.0) for f in feature_names]
        resolved["feature_weights"] = weights
        resolved["feature_pre_filter"] = False

    # balanced — None means auto (True for binary/multiclass, False for regression)
    effective_balanced = smart.get("balanced")
    if effective_balanced is None:
        effective_balanced = task != "regression"
    if effective_balanced:
        if task == "regression":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="'balanced' is not supported for regression tasks.",
                context={"task": task},
            )
        if task == "binary":
            neg = int((y == 0).sum())
            pos = int((y == 1).sum())
            resolved["scale_pos_weight"] = neg / pos if pos > 0 else 1.0
        else:  # multiclass
            from sklearn.utils.class_weight import compute_sample_weight

            sw: npt.NDArray[np.float64] = compute_sample_weight("balanced", y)
            sample_weight = sw

    return resolved, sample_weight


def resolve_ratio_params(
    min_data_in_leaf_ratio: float | None,
    min_data_in_bin_ratio: float | None,
    n_rows: int,
) -> dict[str, int]:
    """Resolve n_rows-dependent ratio params to native LightGBM values.

    Called per-fold with inner_train size (after inner_valid split) to ensure
    ratio params reflect the actual training data size (H-0036).

    Args:
        min_data_in_leaf_ratio: Ratio for min_data_in_leaf (None to skip).
        min_data_in_bin_ratio: Ratio for min_data_in_bin (None to skip).
        n_rows: Number of inner-train rows (after inner_valid split).

    Returns:
        Dict of resolved native LightGBM parameters.
    """
    resolved: dict[str, int] = {}
    if min_data_in_leaf_ratio is not None:
        resolved["min_data_in_leaf"] = _compute_ratio_param(
            n_rows, min_data_in_leaf_ratio
        )
    if min_data_in_bin_ratio is not None:
        resolved["min_data_in_bin"] = _compute_ratio_param(
            n_rows, min_data_in_bin_ratio
        )
    return resolved
