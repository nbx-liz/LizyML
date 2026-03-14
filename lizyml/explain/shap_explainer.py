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
import numpy.typing as npt
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
) -> npt.NDArray[np.float64]:
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
            reduced: npt.NDArray[np.float64] = np.mean(np.abs(raw), axis=2)
            return reduced
        # Regression or binary: (n_samples, n_features)
        return raw

    # Legacy list format from older SHAP versions
    if isinstance(raw, list):
        if task == "binary" and len(raw) == 2:
            result: npt.NDArray[np.float64] = raw[1]
            return result
        # Multiclass: list of k arrays each (n_samples, n_features)
        stacked: npt.NDArray[np.float64] = np.stack(raw, axis=0)  # (k, n, p)
        mean_abs: npt.NDArray[np.float64] = np.mean(np.abs(stacked), axis=0)  # (n, p)
        return mean_abs

    arr: npt.NDArray[np.float64] = np.asarray(raw)
    return arr


def compute_shap_importance(
    models: list[Any],
    X: pd.DataFrame,
    splits_outer: list[tuple[npt.NDArray[Any], npt.NDArray[Any]]],
    task: str,
    feature_names: list[str],
    pipeline_state: Any,
) -> dict[str, float]:
    """Compute fold-averaged SHAP-based feature importance.

    For each CV fold, SHAP values are computed on the validation subset.
    The per-feature importance is ``mean(|SHAP|)`` averaged across folds.

    Args:
        models: List of fitted estimator adapters (one per fold).
        X: Raw feature DataFrame (pre-pipeline).
        splits_outer: Outer CV split indices ``(train_idx, valid_idx)`` per fold.
        task: ML task type.
        feature_names: Ordered feature names from training.
        pipeline_state: Serialized FeaturePipeline state for transformation.

    Returns:
        Dict mapping feature name → importance score.

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

    n_features = len(feature_names)
    n_folds = len(models)
    if n_folds == 0:
        return {name: 0.0 for name in feature_names}

    from lizyml.features.pipelines_native import NativeFeaturePipeline

    # Reconstruct pipeline and transform X
    pipeline = NativeFeaturePipeline()
    pipeline.load_state(pipeline_state)
    X_t, _ = pipeline.transform_with_warnings(X)

    agg = np.zeros(n_features)

    for fold_idx, model in enumerate(models):
        _, valid_idx = splits_outer[fold_idx]
        X_valid = X_t.iloc[valid_idx]
        shap_vals = compute_shap_values(model, X_valid, task)
        # mean(|SHAP|) per feature for this fold
        fold_importance: npt.NDArray[np.float64] = np.mean(np.abs(shap_vals), axis=0)
        agg += fold_importance

    agg /= n_folds
    return {name: float(agg[i]) for i, name in enumerate(feature_names)}
