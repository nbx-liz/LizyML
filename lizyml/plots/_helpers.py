"""Shared data-collection helpers for plot modules (#218).

``collect_is_data`` replaces the near-duplicate ``_collect_is_data``
(classification) and ``_build_is_data`` (residuals): both concatenate the
per-fold in-sample (IF) predictions and the matching training targets across
the outer splits. Hoisting the single implementation keeps the fold-iteration
contract declared once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult


def collect_is_data(
    fit_result: FitResult,
    y_true: npt.NDArray[Any],
    *,
    dtype: Any = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Return ``(is_pred_all, is_y_all)`` concatenated across the outer folds.

    Args:
        fit_result: Fit result carrying ``splits.outer`` and ``if_pred_per_fold``.
        y_true: Full-length target array indexed by the outer train indices.
        dtype: When given, each fold's arrays are cast to this dtype (residual
            plots need ``float64``); when ``None`` the source dtype is preserved
            (classification ROC).

    Returns:
        Concatenated in-sample predictions and targets. Empty arrays (of
        ``dtype`` or ``y_true``'s dtype) when there are no folds.
    """
    is_preds: list[npt.NDArray[Any]] = []
    is_y: list[npt.NDArray[Any]] = []
    for (train_idx, _), if_pred in zip(
        fit_result.splits.outer, fit_result.if_pred_per_fold, strict=True
    ):
        yt = y_true[train_idx]
        if dtype is not None:
            if_pred = if_pred.astype(dtype)
            yt = yt.astype(dtype)
        is_preds.append(if_pred)
        is_y.append(yt)
    if not is_preds:
        empty: npt.NDArray[Any] = np.array([], dtype=dtype or y_true.dtype)
        return empty, empty
    return np.concatenate(is_preds), np.concatenate(is_y)
