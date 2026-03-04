"""Data validators: detect time series violations, group leakage, and target leakage."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError


def validate_time_series_order(
    df: pd.DataFrame,
    time_col: str,
    *,
    raise_on_violation: bool = True,
) -> list[str]:
    """Validate that the time column is sorted in non-decreasing order.

    Args:
        df: DataFrame to validate.
        time_col: Name of the time column.
        raise_on_violation: If True, raises on violation. If False, returns warnings.

    Returns:
        List of warning messages (empty if no violations).

    Raises:
        LizyMLError: With ``LEAKAGE_SUSPECTED`` when ``raise_on_violation=True``
            and the time column is not sorted.
    """
    if time_col not in df.columns:
        return []
    col = df[time_col]
    is_sorted = col.is_monotonic_increasing
    if not is_sorted:
        msg = (
            f"Time column '{time_col}' is not sorted in non-decreasing order. "
            "This may indicate future information leakage in time-series splits."
        )
        if raise_on_violation:
            raise LizyMLError(
                ErrorCode.LEAKAGE_SUSPECTED,
                user_message=msg,
                context={"time_col": time_col},
            )
        return [msg]
    return []


def validate_no_target_leakage(
    df: pd.DataFrame,
    target: str,
    *,
    raise_on_violation: bool = True,
) -> list[str]:
    """Check for columns perfectly correlated with the target (potential leakage).

    A column that is perfectly correlated with the target almost certainly leaks
    label information.

    Args:
        df: DataFrame to validate.
        target: Target column name.
        raise_on_violation: If True, raises on violation.

    Returns:
        List of warning messages.

    Raises:
        LizyMLError: With ``LEAKAGE_SUSPECTED`` when a perfect correlation is found.
    """
    if target not in df.columns:
        return []

    y = df[target]
    warnings: list[str] = []
    for col in df.columns:
        if col == target:
            continue
        try:
            if df[col].equals(y) or (
                pd.api.types.is_numeric_dtype(df[col])
                and pd.api.types.is_numeric_dtype(y)
                and np.allclose(df[col].dropna(), y.dropna(), equal_nan=True)
                and df[col].isna().equals(y.isna())
            ):
                msg = (
                    f"Column '{col}' is perfectly correlated with target '{target}'. "
                    "This is a strong signal of target leakage."
                )
                if raise_on_violation:
                    raise LizyMLError(
                        ErrorCode.LEAKAGE_SUSPECTED,
                        user_message=msg,
                        context={"leaking_column": col, "target": target},
                    )
                warnings.append(msg)
        except (TypeError, ValueError):
            # Non-comparable types; skip
            pass
    return warnings


def validate_group_split(
    groups: pd.Series,
    train_idx: npt.NDArray[np.intp],
    valid_idx: npt.NDArray[np.intp],
    *,
    raise_on_violation: bool = True,
) -> list[str]:
    """Validate that no group appears in both train and validation folds.

    Args:
        groups: Group labels for each sample.
        train_idx: Indices of the training set.
        valid_idx: Indices of the validation set.
        raise_on_violation: If True, raises on violation.

    Returns:
        List of warning messages.

    Raises:
        LizyMLError: With ``LEAKAGE_CONFIRMED`` when groups overlap.
    """
    train_groups = set(groups.iloc[train_idx].unique())
    valid_groups = set(groups.iloc[valid_idx].unique())
    overlap = train_groups & valid_groups
    if overlap:
        msg = (
            f"Groups {sorted(overlap)} appear in both train and validation folds. "
            "This violates the group split constraint."
        )
        if raise_on_violation:
            raise LizyMLError(
                ErrorCode.LEAKAGE_CONFIRMED,
                user_message=msg,
                context={"overlapping_groups": sorted(str(g) for g in overlap)},
            )
        return [msg]
    return []
