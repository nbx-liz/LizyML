"""DataFrameBuilder: separate target/time/group columns and apply feature config.

Also drives target encoding for non-numeric classification labels (H-0070).
The encoder is built from ``ProblemSpec.task`` and applied here so that
downstream layers (training/estimators/calibration) always see int y.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
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


def sort_components(
    sort_order: npt.NDArray[np.intp] | pd.Series,
    components: DataFrameComponents,
) -> DataFrameComponents:
    """Apply *sort_order* to every Series in *components* and return a new
    :class:`DataFrameComponents` (#115, moved from the Facade in #209).

    ``groups`` (an ``np.ndarray`` outside ``components``) is *not* touched here —
    each split branch handles it differently. ``target_encoder`` is propagated
    unchanged.
    """
    X = components.X.iloc[sort_order].reset_index(drop=True)
    y = components.y.iloc[sort_order].reset_index(drop=True)
    sorted_time = (
        components.time_col.iloc[sort_order].reset_index(drop=True)
        if components.time_col is not None
        else None
    )
    sorted_group = (
        components.group_col.iloc[sort_order].reset_index(drop=True)
        if components.group_col is not None
        else None
    )
    return DataFrameComponents(
        X=X,
        y=y,
        time_col=sorted_time,
        group_col=sorted_group,
        target_encoder=components.target_encoder,
    )


def prepare_for_split(
    df: pd.DataFrame,
    components: DataFrameComponents,
    groups: npt.NDArray[Any] | None,
    *,
    time_series: bool,
    method_name: str,
    blocked: tuple[str, str] | None,
) -> tuple[DataFrameComponents, npt.NDArray[Any] | None, npt.NDArray[Any] | None]:
    """Apply split-driven ordering/extraction to already-built components (#209).

    This is the data-domain transformation the Facade previously inlined in
    ``_prepare_training_data`` / ``_sort_and_rebuild_components``. The Facade
    supplies primitives (``time_series`` / ``method_name`` / ``blocked`` column
    names) so this stays free of any ``config`` dependency.

    Args:
        df: The raw DataFrame (block/group columns are read from here before the
            feature pipeline drops them).
        components: Components from :func:`build`.
        groups: Group array from ``components.group_col`` (or ``None``).
        time_series: True when the outer split is time-ordered and rows must be
            sorted by ``components.time_col``.
        method_name: Split method name, for error context.
        blocked: ``(blocks_col, groups_col)`` for a blocked-group split, else
            ``None``. Both columns are validated against *df*.

    Returns:
        ``(components, groups, block_values)`` — ``block_values`` is ``None``
        unless *blocked* is set.

    Raises:
        LizyMLError with ``CONFIG_INVALID`` when a time-series split lacks a time
        column, or a blocked column is missing from *df*.
    """
    block_values: npt.NDArray[Any] | None = None

    if time_series:
        if components.time_col is None:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    f"split.method='{method_name}' requires data.time_col to be set."
                ),
                context={"split_method": method_name},
            )
        sort_order = components.time_col.argsort()
        components = sort_components(sort_order, components)
        if groups is not None:
            groups = groups[sort_order]

    if blocked is not None:
        blocks_col_name, groups_col_name = blocked
        for col_name, label in [
            (blocks_col_name, "blocks.col"),
            (groups_col_name, "groups.col"),
        ]:
            if col_name not in df.columns:
                raise LizyMLError(
                    code=ErrorCode.CONFIG_INVALID,
                    user_message=(
                        f"split.{label}='{col_name}' not found in DataFrame columns."
                    ),
                    context={"column": col_name, "available": list(df.columns)},
                )

        block_series = df[blocks_col_name]
        group_series = df[groups_col_name]
        sort_order = block_series.argsort()
        components = sort_components(sort_order, components)
        groups = group_series.iloc[sort_order].to_numpy()
        block_values = block_series.iloc[sort_order].to_numpy()

    return components, groups, block_values
