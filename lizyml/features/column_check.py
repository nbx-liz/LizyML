"""Prediction-time column check, independent of the feature pipeline (H-0104).

The prediction facade runs this before handing data to any pipeline, so a
pipeline that does not check its input -- or overrides the method that would --
cannot let a missing column through. ``NativeFeaturePipeline`` calls the same
function when it is used on its own; this is the only implementation.
"""

from __future__ import annotations

import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError


def select_training_columns(
    X: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Return *X* reduced to the training columns, in training order.

    Args:
        X: Feature DataFrame supplied for prediction.
        feature_names: Feature columns the model was fitted on, in order.

    Returns:
        Tuple of ``(selected, warnings)``. ``warnings`` names any extra
        columns, which are dropped.

    Raises:
        LizyMLError: With ``DATA_SCHEMA_INVALID`` when a training column is
            missing from *X*.
    """
    present = set(X.columns)
    missing = sorted(set(feature_names) - present)
    if missing:
        raise LizyMLError(
            ErrorCode.DATA_SCHEMA_INVALID,
            user_message=f"Required feature columns missing: {missing}",
            context={"missing_columns": missing},
        )

    warnings: list[str] = []
    extra = sorted(present - set(feature_names))
    if extra:
        warnings.append(f"Extra columns ignored during transform: {extra}")
    return X[list(feature_names)].copy(), warnings
