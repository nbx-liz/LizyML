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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.task import TaskType

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
    task: TaskType,
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
    return _normalize_shap_output(raw, task)


def _normalize_shap_output(raw: Any, task: TaskType) -> npt.NDArray[np.float64]:
    """Normalize a raw ``TreeExplainer.shap_values`` return to ``(n, p)``.

    Handles the three shapes SHAP may return, per the H-0002 contract:

    - ``ndarray`` of ``ndim == 3`` — multiclass ``(n, p, k)`` → mean-abs over ``k``.
    - ``ndarray`` of ``ndim <= 2`` — regression/binary ``(n, p)`` → returned as-is.
    - ``list`` — legacy per-class format: binary keeps the positive class,
      multiclass reduces via mean-abs across classes.
    - anything else — coerced with ``np.asarray`` as a last resort.

    Pure (no SHAP call), so the normalization branches are unit-testable with
    plain numpy inputs.
    """
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
    task: TaskType,
    feature_names: list[str],
    pipeline_state: Any,
    pipeline_factory: Callable[[], Any] | None = None,
    pipeline_state_per_fold: list[Any] | None = None,
) -> dict[str, float]:
    """Compute fold-averaged SHAP-based feature importance.

    For each CV fold, SHAP values are computed on the validation subset.
    The per-feature importance is ``mean(|SHAP|)`` averaged across folds.

    With ``pipeline_state_per_fold``, fold k's validation rows are encoded by
    fold k's own pipeline state, as they were when that fold's model was
    trained and predicted its OOF rows (H-0114). Without it, every row is
    encoded by ``pipeline_state``.

    Args:
        models: List of fitted estimator adapters (one per fold).
        X: Raw feature DataFrame (pre-pipeline).
        splits_outer: Outer CV split indices ``(train_idx, valid_idx)`` per fold.
        task: ML task type.
        feature_names: Ordered feature names from training.
        pipeline_state: Serialized FeaturePipeline state, used for every fold
            when ``pipeline_state_per_fold`` is not given.
        pipeline_factory: Optional factory to create the pipeline (H-0054).
            Falls back to ``NativeFeaturePipeline`` when not provided.
        pipeline_state_per_fold: Each fold's serialized pipeline state, in
            fold order (``FitResult.pipeline_state_per_fold``).

    Returns:
        Dict mapping feature name → importance score.

    Raises:
        LizyMLError with ``OPTIONAL_DEP_MISSING`` when shap is not installed.
        ValueError: When ``pipeline_state_per_fold`` and ``models`` differ in
            length.
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
    # Checked before the zero-fold return so a mismatch is never answered
    # with an all-zero importance.
    if pipeline_state_per_fold is not None and len(pipeline_state_per_fold) != n_folds:
        raise ValueError(
            f"pipeline_state_per_fold has {len(pipeline_state_per_fold)} states "
            f"for {n_folds} fold models."
        )
    if n_folds == 0:
        return {name: 0.0 for name in feature_names}

    def load_pipeline(state: Any) -> Any:
        # H-0054: use the provider's factory when given.
        if pipeline_factory is not None:
            pipeline = pipeline_factory()
        else:
            from lizyml.features.pipelines_native import NativeFeaturePipeline

            pipeline = NativeFeaturePipeline()
        pipeline.load_state(state)
        return pipeline

    X_t: pd.DataFrame | None = None
    if pipeline_state_per_fold is None:
        X_t, _ = load_pipeline(pipeline_state).transform_with_warnings(X)

    agg = np.zeros(n_features)

    for fold_idx, model in enumerate(models):
        _, valid_idx = splits_outer[fold_idx]
        if pipeline_state_per_fold is None:
            assert X_t is not None  # noqa: S101
            X_valid = X_t.iloc[valid_idx]
        else:
            fold_pipeline = load_pipeline(pipeline_state_per_fold[fold_idx])
            X_valid, _ = fold_pipeline.transform_with_warnings(X.iloc[valid_idx])
        shap_vals = compute_shap_values(model, X_valid, task)
        # mean(|SHAP|) per feature for this fold
        fold_importance: npt.NDArray[np.float64] = np.mean(np.abs(shap_vals), axis=0)
        agg += fold_importance

    agg /= n_folds
    return {name: float(agg[i]) for i, name in enumerate(feature_names)}
