"""SHAP-based explanation utilities.

Computes SHAP values using TreeExplainer.  ``shap`` is an optional
dependency — install with ``pip install 'lizyml[explain]'``.

Shape contract (per H-0002):
    ``shap_values`` is always ``(n_samples, n_features)`` regardless of task.

    - Regression / binary:  TreeExplainer returns ``(n, p)`` directly.
    - Multiclass:            TreeExplainer returns a list of ``k`` arrays
                             each ``(n, p)``; we return mean-absolute across
                             classes → ``(n, p)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.estimators.base import BaseEstimatorAdapter

_shap: Any = None
try:
    import shap

    _shap = shap
except ImportError:  # pragma: no cover
    pass


def compute_shap_values(
    model: BaseEstimatorAdapter,
    X: pd.DataFrame,
    task: str,
) -> np.ndarray:
    """Compute SHAP values for *X* using *model*.

    Args:
        model: A fitted estimator adapter exposing ``get_native_model()``.
        X: Feature DataFrame (post-pipeline transform).
        task: ML task — ``"regression"``, ``"binary"``, or ``"multiclass"``.

    Returns:
        SHAP values array of shape ``(n_samples, n_features)``.

    Raises:
        LizyMLError with ``OPTIONAL_DEP_MISSING`` when shap is not installed.
    """
    if _shap is None:
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                "shap is required for SHAP explanations. "
                "Install with: pip install 'lizyml[explain]'"
            ),
            context={"package": "shap"},
        )

    native = model.get_native_model()
    explainer = _shap.TreeExplainer(native)
    raw = explainer.shap_values(X)

    if isinstance(raw, np.ndarray):
        if raw.ndim == 3:
            # Multiclass: (n_samples, n_features, n_classes) — reduce to (n, p)
            reduced: np.ndarray = np.mean(np.abs(raw), axis=2)
            return reduced
        # Regression or binary: (n_samples, n_features)
        return raw

    # Legacy list format from older SHAP versions
    if isinstance(raw, list):
        if task == "binary" and len(raw) == 2:
            result: np.ndarray = raw[1]
            return result
        # Multiclass: list of k arrays each (n_samples, n_features)
        stacked: np.ndarray = np.stack(raw, axis=0)  # (k, n, p)
        mean_abs: np.ndarray = np.mean(np.abs(stacked), axis=0)  # (n, p)
        return mean_abs

    arr: np.ndarray = np.asarray(raw)
    return arr
