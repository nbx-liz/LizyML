"""DataFrameBuilder: separate target/time/group columns and apply feature config.

Also drives target encoding for non-numeric classification labels (H-0070).
The encoder is built from ``ProblemSpec.task`` and applied here so that
downstream layers (training/estimators/calibration) always see int y.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.specs.feature_spec import FeatureSpec
from lizyml.core.specs.problem_spec import ProblemSpec
from lizyml.core.types.target_encoder import TargetEncoder


@dataclass
class DataFrameComponents:
    """Result of splitting a raw DataFrame into features and label columns."""

    X: pd.DataFrame
    y: pd.Series
    time_col: pd.Series | None
    group_col: pd.Series | None
    target_encoder: TargetEncoder


def build(
    df: pd.DataFrame,
    problem_spec: ProblemSpec,
    feature_spec: FeatureSpec,
) -> DataFrameComponents:
    """Separate target, time, group columns and apply feature configuration.

    Args:
        df: Raw input DataFrame (not modified).
        problem_spec: Defines target, time_col, group_col, and task.
        feature_spec: Defines exclude, auto_categorical, categorical.

    Returns:
        ``DataFrameComponents`` with X, y, optional time/group columns, and
        a :class:`TargetEncoder`. y is encoded to int when the original
        target dtype is non-numeric and the task is classification.

    Raises:
        LizyMLError: With ``DATA_SCHEMA_INVALID`` when required columns are
            missing, or with ``TARGET_NOT_NUMERIC`` when ``task='regression'``
            and the target is non-numeric.
    """
    _validate_required_columns(df, problem_spec)

    raw_y = df[problem_spec.target].copy()
    target_encoder = TargetEncoder.fit(raw_y, problem_spec.task)
    y = target_encoder.transform(raw_y)

    time_col = df[problem_spec.time_col].copy() if problem_spec.time_col else None
    group_col = df[problem_spec.group_col].copy() if problem_spec.group_col else None

    # Build the set of columns to drop from X
    drop_cols = {problem_spec.target}
    if problem_spec.time_col:
        drop_cols.add(problem_spec.time_col)
    if problem_spec.group_col:
        drop_cols.add(problem_spec.group_col)
    drop_cols.update(feature_spec.exclude)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    X = _apply_categorical(X, feature_spec)

    return DataFrameComponents(
        X=X,
        y=y,
        time_col=time_col,
        group_col=group_col,
        target_encoder=target_encoder,
    )


def _validate_required_columns(df: pd.DataFrame, spec: ProblemSpec) -> None:
    missing: list[str] = []
    for col in [spec.target, spec.time_col, spec.group_col]:
        if col is not None and col not in df.columns:
            missing.append(col)
    if missing:
        raise LizyMLError(
            ErrorCode.DATA_SCHEMA_INVALID,
            user_message=f"Required columns not found in DataFrame: {missing}",
            context={"missing_columns": missing, "available_columns": list(df.columns)},
        )


def _apply_categorical(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Apply categorical dtype to explicitly specified and auto-detected columns."""
    df = df.copy()
    cat_set = set(spec.categorical)

    if spec.auto_categorical:
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                cat_set.add(col)

    for col in cat_set:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
