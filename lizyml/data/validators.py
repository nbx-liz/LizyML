"""Data validators: detect time series violations, group leakage, and target leakage."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError


def _require_column(
    df: pd.DataFrame, name: str, *, role: str, label: str, check: str
) -> None:
    """Raise ``DATA_SCHEMA_INVALID`` when the column a check is named for is absent.

    Returning ``[]`` here would answer like a check that ran and found nothing
    (#311, H-0112), so the absence is reported whatever ``raise_on_violation``
    is. ``context`` follows ``dataframe_builder``'s missing-column shape.

    Membership is tested against complete labels: on ``MultiIndex`` columns
    ``"a" in df.columns`` is true for a partial key, which would let a check
    run on a sub-frame (or answer ``[]``) for a column that does not exist.
    """
    # pandas-stubs types ``df.columns`` as ``Index[str]`` without
    # ``to_flat_index``; at runtime every Index has it (identity unless Multi).
    if name in df.columns.to_flat_index():  # type: ignore[attr-defined]
        return
    raise LizyMLError(
        ErrorCode.DATA_SCHEMA_INVALID,
        user_message=(
            f"{label} column '{name}' is not in the DataFrame, so the "
            f"{check} checked nothing."
        ),
        context={
            role: name,
            "missing_columns": [name],
            "available_columns": list(df.columns),
        },
    )


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
        LizyMLError: With ``DATA_SCHEMA_INVALID`` (whatever
            ``raise_on_violation`` is) when ``time_col`` is not a column of
            ``df``, and with ``LEAKAGE_SUSPECTED`` when
            ``raise_on_violation=True`` and the time column is not sorted.
    """
    _require_column(
        df, time_col, role="time_col", label="Time", check="time-order check"
    )
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
        LizyMLError: With ``DATA_SCHEMA_INVALID`` (whatever
            ``raise_on_violation`` is) when ``target`` is not a column of
            ``df``, before any column is compared. Then, with
            ``LEAKAGE_SUSPECTED`` when a perfect correlation is found and
            ``raise_on_violation`` is true, and with ``DATA_SCHEMA_INVALID``
            (whatever ``raise_on_violation`` is) when a column cannot be
            compared with the target. Columns are checked in order, so the
            first of these conditions met is the one raised.
    """
    _require_column(df, target, role="target", label="Target", check="leakage check")

    y = df[target]
    warnings: list[str] = []
    for col in df.columns:
        if col == target:
            continue
        try:
            correlated = _series_perfectly_correlated(df[col], y)
        except Exception as exc:
            # A column that cannot be compared was not checked. Skipping it
            # made it indistinguishable from a checked, clean column; adding
            # it to the warnings would mix "not checked" into the list of
            # suspected leaks, which callers would then have to tell apart by
            # wording. So it raises (#267, H-0107).
            raise LizyMLError(
                ErrorCode.DATA_SCHEMA_INVALID,
                user_message=(
                    f"Column '{col}' could not be compared with target "
                    f"'{target}' for the leakage check: {type(exc).__name__}: {exc}"
                ),
                context={"column": col, "target": target},
                cause=exc,
            ) from exc
        if correlated:
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
    return warnings


def _series_perfectly_correlated(col: pd.Series, y: pd.Series) -> bool:
    """Return True when *col* is a perfect duplicate of the target *y*.

    The NaN-position guard (``isna().equals``) MUST be evaluated before
    ``np.allclose`` so that columns with a differing number of NaNs never
    reach ``dropna()`` with mismatched lengths (which would raise
    ``ValueError``). The short-circuiting ``and`` chain encodes that ordering;
    keeping it in one pure helper makes the ordering unit-testable without
    patching ``np.allclose``.
    """
    if col.equals(y):
        return True
    return bool(
        pd.api.types.is_numeric_dtype(col)
        and pd.api.types.is_numeric_dtype(y)
        and col.isna().equals(y.isna())
        and np.allclose(col.dropna(), y.dropna(), equal_nan=True)
    )


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
