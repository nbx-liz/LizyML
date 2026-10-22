"""CategoricalEncoder — learns category dictionaries and handles unseen values."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd

from lizyml.core.category_values import (
    CATEGORY_IDENTITY_VERSION,
    check_keyless_temporals,
    identity_eligible,
    identity_fields,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.features.encoders.category_identity import (
    keyed_categories,
    keyed_codes,
)

UnseenPolicy = Literal["mode", "nan", "error"]

#: Float types pandas cannot hold as category values (H-0127, #309): their
#: values are compared as float64 instead.
_UNINDEXABLE_FLOATS = frozenset({np.dtype(np.float16), np.dtype(np.longdouble)}) - {
    np.dtype(np.float64)
}


def _is_unindexable_float(value: Any) -> bool:
    return isinstance(value, np.floating) and value.dtype in _UNINDEXABLE_FLOATS


def _filled(series: pd.Series, value: Any) -> pd.Series:
    """*value* in every row, kept whole even when it is a tuple."""
    return pd.Series([value] * len(series), index=series.index, dtype=object)


def comparable_values(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """*series* with float16 / longdouble values as float64, and the rows whose
    value float64 changes (H-0127, #309).

    A changed value equals no category: the caller treats it as unseen. A
    missing value is never compared and stays missing. A column of any other
    dtype, or an ``object`` column holding no such scalar, is returned as is.
    """
    unchanged = pd.Series(False, index=series.index)
    dtype = series.dtype
    if isinstance(dtype, np.dtype) and dtype in _UNINDEXABLE_FLOATS:
        values = series.to_numpy()
        as64 = values.astype(np.float64)
        changed = ~np.isnan(values) & (as64.astype(dtype) != values)
        return (
            pd.Series(as64, index=series.index, name=series.name),
            pd.Series(changed, index=series.index),
        )
    if not pd.api.types.is_object_dtype(dtype) or pd.api.types.infer_dtype(
        series, skipna=True
    ) in (
        "string",
        "empty",
    ):
        return series, unchanged
    # By position, not label: an index may repeat a label.
    values = series.to_numpy(dtype=object)
    positions = [i for i, v in enumerate(values) if _is_unindexable_float(v)]
    if not positions:
        return series, unchanged
    converted = values.copy()
    changed = np.zeros(len(values), dtype=bool)
    for pos in positions:
        value: Any = values[pos]
        scalar64 = np.float64(value)
        converted[pos] = float(scalar64)
        if not np.isnan(value):
            changed[pos] = bool(scalar64.astype(value.dtype) != value)
    return (
        pd.Series(converted, index=series.index, name=series.name, dtype=object),
        pd.Series(changed, index=series.index),
    )


class CategoricalEncoder:
    """Per-column categorical encoder that records known categories from training.

    Args:
        unseen_policy: How to handle categories not seen during fit:
            - ``"mode"``: Replace with the most frequent training category.
            - ``"nan"``: Replace with NaN (pandas NA for category dtype).
            - ``"error"``: Raise ``LizyMLError(DATA_SCHEMA_INVALID)``.
        needs_stand_in: Whether a category value cannot reach the estimator's
            saved model as itself (H-0133). Fit records every column holding
            such a value; transform hands those columns over with their
            categories renamed to the positions ``0..k-1`` (the codes, and so
            the predictions, are unchanged). ``None`` replaces no column.
    """

    def __init__(
        self,
        unseen_policy: UnseenPolicy = "mode",
        needs_stand_in: Callable[[Any], bool] | None = None,
    ) -> None:
        self.unseen_policy: UnseenPolicy = unseen_policy
        self._needs_stand_in = needs_stand_in
        self._categories: dict[str, list[Any]] = {}
        self._modes: dict[str, Any] = {}
        self._stand_in_columns: list[str] = []
        # H-0138: the identity the state records (None: a state saved before
        # H-0138, looked up the old way) and the columns looked up by key.
        self._identity_version: int | None = None
        self._identity_columns: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, categorical_cols: list[str]) -> CategoricalEncoder:
        """Record categories from training data.

        A column whose values are all in V is looked up by the identity key
        (H-0138; temporal values H-0139): its categories are today's where
        they are faithful under the key, else built by key (and it becomes a
        stand-in column).

        Args:
            X: Training DataFrame.
            categorical_cols: Columns to encode.

        Returns:
            ``self`` for chaining.

        Raises:
            LizyMLError: ``DATA_SCHEMA_INVALID`` for a column holding a
                ``np.timedelta64`` in ``Y`` / ``M`` units (H-0139).
        """
        for col in categorical_cols:
            if col in X.columns:
                check_keyless_temporals(X[col], col)
        self._categories = {}
        self._modes = {}
        self._stand_in_columns = []
        self._identity_version = CATEGORY_IDENTITY_VERSION
        self._identity_columns = []
        for col in categorical_cols:
            if col not in X.columns:
                continue
            series = X[col]
            rebuilt = False
            if identity_eligible(series):
                cats, mode, rebuilt = keyed_categories(series)
                self._identity_columns.append(col)
            else:
                cats, mode = _today_categories(series)
            self._categories[col] = cats
            self._modes[col] = mode
            # Decided here, not in transform: CV saves the state right after
            # fit, before any transform (H-0133). The values are the ones the
            # estimator receives after set_categories(cats). A column built by
            # key holds values pandas cannot hold: always positions (H-0138).
            if rebuilt or (
                self._needs_stand_in is not None
                and any(
                    self._needs_stand_in(v)
                    for v in pd.CategoricalDtype(cats).categories
                )
            ):
                self._stand_in_columns.append(col)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns using the fitted category dictionaries.

        Substitutions are applied silently here; use
        :meth:`transform_with_warnings` to learn which were made.

        Args:
            X: DataFrame to transform (may include unseen categories).

        Returns:
            Copy of ``X`` with categorical columns cast to the fitted dtype.

        Raises:
            LizyMLError: With ``DATA_SCHEMA_INVALID`` when ``unseen_policy="error"``
                and an unseen category is encountered.
        """
        result, _ = self.transform_with_warnings(X)
        return result

    def transform_with_warnings(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Encode, and report every unseen-category substitution (H-0104).

        Under ``"mode"`` and ``"nan"`` an unseen category is a correction the
        caller did not write, so each affected column yields one message naming
        the column, the unseen values and what replaced them.

        Args:
            X: DataFrame to transform (may include unseen categories).

        Returns:
            Tuple of ``(encoded_df, warnings)``.

        Raises:
            LizyMLError: With ``DATA_SCHEMA_INVALID`` when ``unseen_policy="error"``
                and an unseen category is encountered.
        """
        if not self._fitted:
            raise RuntimeError("CategoricalEncoder must be fitted before transform.")

        warnings: list[str] = []
        X = X.copy()
        for col, known_cats in self._categories.items():
            if col not in X.columns:
                continue
            arrived = X[col]
            # H-0127 (#309): float16 / longdouble values are compared as
            # float64; one float64 changes matches no category (unseen).
            series, changed = comparable_values(arrived)
            if col in self._identity_columns:
                # H-0138: codes by key, never through pandas' object Index.
                codes, notes = keyed_codes(
                    series, arrived, changed, known_cats, self._modes.get(col),
                    self.unseen_policy, col,
                )  # fmt: skip
                warnings.extend(notes)
                stand_in = col in self._stand_in_columns
                categories: Any = range(len(known_cats)) if stand_in else known_cats
                X[col] = pd.Series(
                    # Older pandas stubs (the 3.10 lane) type codes as
                    # Sequence[int]; an int64 array is what pandas reads.
                    pd.Categorical.from_codes(
                        codes,  # type: ignore[arg-type, unused-ignore]
                        categories=categories,
                    ),
                    index=X.index,
                    name=arrived.name,
                )
                continue
            # Determine unseen values
            if hasattr(series, "cat"):
                current_cats = set(series.cat.categories.tolist())
            else:
                current_cats = set(series[~changed].dropna().unique().tolist())
            unseen = current_cats - set(known_cats)

            if unseen or changed.any():
                unseen_shown = sorted(
                    {str(v) for v in unseen} | {str(v) for v in arrived[changed]}
                )
                if self.unseen_policy == "error":
                    raise LizyMLError(
                        ErrorCode.DATA_SCHEMA_INVALID,
                        user_message=(
                            f"Column '{col}' contains unseen categories: {unseen_shown}"
                        ),
                        context={
                            "column": col,
                            "unseen_categories": unseen_shown,
                        },
                    )
                # Convert to object dtype first so replacement values are accepted
                series = series.astype(object)
                hit = series.isin(list(unseen)) | changed
                n_rows = int(hit.sum())
                # A category dtype can declare values no row holds; replacing
                # nothing is not a correction, so only values present count.
                present = sorted({str(v) for v in arrived[hit]})
                if self.unseen_policy == "mode":
                    replacement = self._modes.get(col)
                    series = series.where(~hit, _filled(series, replacement))
                    replaced_by = f"the training mode {replacement!r}"
                else:  # "nan"
                    series = series.where(~hit, _filled(series, None))
                    replaced_by = "a missing value"
                if n_rows:
                    warnings.append(
                        f"Column '{col}': {n_rows} row(s) with unseen categories "
                        f"{present} replaced by {replaced_by} "
                        f"(unseen_policy='{self.unseen_policy}')"
                    )

            # Set categories to the known list
            series = series.astype("category")
            series = series.cat.set_categories(known_cats)
            if col in self._stand_in_columns:
                # H-0133: the same codes, with positions as the categories.
                series = series.cat.rename_categories(range(len(series.cat.categories)))
            X[col] = series
        return X, warnings

    def get_state(self) -> dict[str, Any]:
        """Return serializable state."""
        state: dict[str, Any] = {
            "unseen_policy": self.unseen_policy,
            "categories": self._categories,
            "modes": self._modes,
            "stand_in_columns": list(self._stand_in_columns),
        }
        if self._identity_version is not None:
            state["category_identity_version"] = self._identity_version
            state["identity_columns"] = list(self._identity_columns)
        return state

    def load_state(self, state: dict[str, Any]) -> CategoricalEncoder:
        """Restore from a previously saved state.

        Raises:
            LizyMLError: ``DESERIALIZATION_FAILED`` when the state names an
                identity version other than ``1`` or a malformed column list.
        """
        version, identity_columns = identity_fields(state)
        self.unseen_policy = state["unseen_policy"]
        self._categories = state["categories"]
        self._modes = state["modes"]
        # A state saved before H-0133 (format_version 1 / 2) has no field:
        # every column keeps its raw categories, as when it was saved.
        self._stand_in_columns = list(state.get("stand_in_columns", []))
        # A state saved before H-0138 has no identity: today's lookup, so it
        # predicts as when it was saved (H-0138 decision 2).
        self._identity_version = version
        self._identity_columns = identity_columns
        self._fitted = True
        return self


def _today_categories(series: pd.Series) -> tuple[list[Any], Any]:
    """The categories and mode a column outside H-0138's key gets (as before)."""
    if hasattr(series, "cat"):
        cats = list(series.cat.categories)
    else:
        cats = sorted(series.dropna().unique().tolist(), key=str)
    if not cats:
        return cats, None
    mode_val = series.mode()
    return cats, mode_val.iloc[0] if len(mode_val) > 0 else cats[0]
