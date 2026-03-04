"""CategoricalEncoder — learns category dictionaries and handles unseen values."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError

UnseenPolicy = Literal["mode", "nan", "error"]


class CategoricalEncoder:
    """Per-column categorical encoder that records known categories from training.

    Args:
        unseen_policy: How to handle categories not seen during fit:
            - ``"mode"``: Replace with the most frequent training category.
            - ``"nan"``: Replace with NaN (pandas NA for category dtype).
            - ``"error"``: Raise ``LizyMLError(DATA_SCHEMA_INVALID)``.
    """

    def __init__(self, unseen_policy: UnseenPolicy = "mode") -> None:
        self.unseen_policy: UnseenPolicy = unseen_policy
        self._categories: dict[str, list[Any]] = {}
        self._modes: dict[str, Any] = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame, categorical_cols: list[str]) -> CategoricalEncoder:
        """Record categories from training data.

        Args:
            X: Training DataFrame.
            categorical_cols: Columns to encode.

        Returns:
            ``self`` for chaining.
        """
        self._categories = {}
        self._modes = {}
        for col in categorical_cols:
            if col not in X.columns:
                continue
            series = X[col]
            if hasattr(series, "cat"):
                cats = list(series.cat.categories)
            else:
                cats = sorted(series.dropna().unique().tolist(), key=str)
            self._categories[col] = cats
            if cats:
                mode_val = series.mode()
                self._modes[col] = mode_val.iloc[0] if len(mode_val) > 0 else cats[0]
            else:
                self._modes[col] = None
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns using the fitted category dictionaries.

        Args:
            X: DataFrame to transform (may include unseen categories).

        Returns:
            Copy of ``X`` with categorical columns cast to the fitted dtype.

        Raises:
            LizyMLError: With ``DATA_SCHEMA_INVALID`` when ``unseen_policy="error"``
                and an unseen category is encountered.
        """
        if not self._fitted:
            raise RuntimeError("CategoricalEncoder must be fitted before transform.")

        X = X.copy()
        for col, known_cats in self._categories.items():
            if col not in X.columns:
                continue
            series = X[col]
            # Determine unseen values
            if hasattr(series, "cat"):
                current_cats = set(series.cat.categories.tolist())
            else:
                current_cats = set(series.dropna().unique().tolist())
            unseen = current_cats - set(known_cats)

            if unseen:
                if self.unseen_policy == "error":
                    raise LizyMLError(
                        ErrorCode.DATA_SCHEMA_INVALID,
                        user_message=(
                            f"Column '{col}' contains unseen categories: "
                            f"{sorted(str(v) for v in unseen)}"
                        ),
                        context={
                            "column": col,
                            "unseen_categories": sorted(str(v) for v in unseen),
                        },
                    )
                # Convert to object dtype first so replacement values are accepted
                series = series.astype(object)
                if self.unseen_policy == "mode":
                    replacement = self._modes.get(col)
                    series = series.replace(list(unseen), replacement)
                else:  # "nan"
                    series = series.replace(list(unseen), None)

            # Set categories to the known list
            series = series.astype("category")
            series = series.cat.set_categories(known_cats)
            X[col] = series
        return X

    def get_state(self) -> dict[str, Any]:
        """Return serializable state."""
        return {
            "unseen_policy": self.unseen_policy,
            "categories": self._categories,
            "modes": self._modes,
        }

    def load_state(self, state: dict[str, Any]) -> CategoricalEncoder:
        """Restore from a previously saved state."""
        self.unseen_policy = state["unseen_policy"]
        self._categories = state["categories"]
        self._modes = state["modes"]
        self._fitted = True
        return self
