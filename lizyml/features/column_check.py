"""Prediction-time column check, independent of the feature pipeline (H-0104).

The prediction facade runs this before handing data to any pipeline, so a
pipeline that does not check its input -- or overrides the method that would --
cannot let a missing column through. ``NativeFeaturePipeline`` calls the same
function when it is used on its own; this is the only implementation.

The facade also passes the fit-time dtypes, and a column that was numeric at
fit must arrive numeric (H-0106). That is an input contract for every
pipeline, checked before any of them runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError


def is_numeric_dtype(dtype: Any) -> bool:
    """True when *dtype*'s scalar type is a numpy integer, floating or bool type.

    ``timedelta64`` (an integer subclass) and ``longdouble`` are excluded. This
    is the set LightGBM accepts for a pandas column; pandas'
    ``is_numeric_dtype`` differs on the pyarrow numeric dtypes and on
    ``longdouble`` (4 of 33 dtypes measured, H-0106).
    """
    scalar = getattr(dtype, "type", None)
    return (
        isinstance(scalar, type)
        and issubclass(scalar, (np.integer, np.floating, np.bool_))
        and not issubclass(scalar, (np.timedelta64, np.longdouble))
    )


def _numeric_at_fit(recorded: str) -> bool:
    """Whether a recorded ``FitResult.dtypes`` string names a numeric dtype.

    A string ``pandas_dtype`` cannot read is an explicit exemption: the column
    is not checked and prediction proceeds as it did before this check
    (H-0106 decision 2). None of the 23 dtypes LightGBM can fit records one;
    a test keeps that population empty.
    """
    try:
        dtype = pd.api.types.pandas_dtype(recorded)
    except (TypeError, ValueError):
        return False
    return is_numeric_dtype(dtype)


def select_training_columns(
    X: pd.DataFrame,
    feature_names: list[str],
    dtypes: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return *X* reduced to the training columns, in training order.

    Args:
        X: Feature DataFrame supplied for prediction.
        feature_names: Feature columns the model was fitted on, in order.
        dtypes: Fit-time dtype of each feature (``FitResult.dtypes``). When
            given, a column numeric at fit must arrive with a numeric dtype.
            Columns categorical at fit are not checked: their values go to the
            encoder.

    Returns:
        Tuple of ``(selected, warnings)``. ``warnings`` names any extra
        columns, which are dropped.

    Raises:
        LizyMLError: With ``DATA_SCHEMA_INVALID`` when a training column is
            missing from *X* (checked first), and with ``INCOMPATIBLE_COLUMNS``
            naming every column numeric at fit that arrives non-numeric.
    """
    present = set(X.columns)
    missing = sorted(set(feature_names) - present)
    if missing:
        raise LizyMLError(
            ErrorCode.DATA_SCHEMA_INVALID,
            user_message=f"Required feature columns missing: {missing}",
            context={"missing_columns": missing},
        )

    if dtypes:
        offending = [
            {
                "column": col,
                "fit_dtype": dtypes[col],
                "predict_dtype": str(X[col].dtype),
            }
            for col in feature_names
            if col in dtypes
            and _numeric_at_fit(dtypes[col])
            and not is_numeric_dtype(X[col].dtype)
        ]
        if offending:
            raise LizyMLError(
                ErrorCode.INCOMPATIBLE_COLUMNS,
                user_message=(
                    "Columns that were numeric at fit arrived with a non-numeric "
                    f"dtype: {[c['column'] for c in offending]}"
                ),
                context={"columns": offending},
            )

    warnings: list[str] = []
    extra = sorted(present - set(feature_names))
    if extra:
        warnings.append(f"Extra columns ignored during transform: {extra}")
    return X[list(feature_names)].copy(), warnings
