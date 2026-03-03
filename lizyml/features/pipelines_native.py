"""NativeFeaturePipeline — LightGBM-native feature pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.features.encoders.categorical_encoder import (
    CategoricalEncoder,
    UnseenPolicy,
)
from lizyml.features.pipeline_base import BaseFeaturePipeline
from lizyml.features.transformers.feature_transformer import FeatureTransformer


class NativeFeaturePipeline(BaseFeaturePipeline):
    """Feature pipeline optimised for LightGBM native categorical support.

    Responsibilities:
    - Identifies categorical columns from training data.
    - Encodes them with a ``CategoricalEncoder`` that records training categories.
    - Applies a passthrough ``FeatureTransformer`` (extensible for future use).

    Column-drift policy at ``transform`` time:
    - Extra columns (present in new data but not seen at fit): warning added;
      column dropped from the output.
    - Missing columns (expected from fit but absent): raises
      ``LizyMLError(DATA_SCHEMA_INVALID)``.
    - Unseen categories: delegated to ``CategoricalEncoder.unseen_policy``.

    Args:
        unseen_policy: How to handle categories unseen during ``fit``.
    """

    def __init__(self, unseen_policy: UnseenPolicy = "mode") -> None:
        self._encoder = CategoricalEncoder(unseen_policy=unseen_policy)
        self._transformer = FeatureTransformer()
        self._feature_names: list[str] = []
        self._categorical_cols: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NativeFeaturePipeline:
        """Fit the pipeline on training data.

        Args:
            X: Training features.
            y: Training target (unused here but required by contract).

        Returns:
            ``self`` for chaining.
        """
        self._feature_names = list(X.columns)
        self._categorical_cols = [
            c
            for c in X.columns
            if hasattr(X[c], "cat") or pd.api.types.is_string_dtype(X[c])
        ]
        self._encoder.fit(X, self._categorical_cols)
        self._transformer.fit(X)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted pipeline (discards column-drift warnings).

        For access to warnings use :meth:`transform_with_warnings`.

        Args:
            X: Feature DataFrame to transform.

        Returns:
            Transformed DataFrame.

        Raises:
            LizyMLError: With ``DATA_SCHEMA_INVALID`` when required columns are
                missing from ``X``.
        """
        result, _ = self.transform_with_warnings(X)
        return result

    def transform_with_warnings(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Apply the fitted pipeline and return column-drift warnings.

        Args:
            X: Feature DataFrame to transform.

        Returns:
            Tuple of ``(transformed_df, warnings)``.

        Raises:
            LizyMLError: With ``DATA_SCHEMA_INVALID`` when required columns are
                missing from ``X``.
        """
        if not self._fitted:
            raise RuntimeError("NativeFeaturePipeline must be fitted before transform.")

        col_warnings: list[str] = []
        expected = set(self._feature_names)
        present = set(X.columns)

        missing = expected - present
        if missing:
            raise LizyMLError(
                ErrorCode.DATA_SCHEMA_INVALID,
                user_message=f"Required feature columns missing: {sorted(missing)}",
                context={"missing_columns": sorted(missing)},
            )

        extra = present - expected
        if extra:
            col_warnings.append(
                f"Extra columns ignored during transform: {sorted(extra)}"
            )

        # Keep only expected columns in training order
        X = X[self._feature_names].copy()

        # Encode categoricals
        X = self._encoder.transform(X)

        # Feature-level transforms
        X = self._transformer.transform(X)

        return X, col_warnings

    def get_state(self) -> dict[str, Any]:
        """Return serializable pipeline state."""
        return {
            "feature_names": self._feature_names,
            "categorical_cols": self._categorical_cols,
            "encoder": self._encoder.get_state(),
            "transformer": self._transformer.get_state(),
        }

    def load_state(self, state: dict[str, Any]) -> NativeFeaturePipeline:
        """Restore pipeline from saved state."""
        self._feature_names = state["feature_names"]
        self._categorical_cols = state["categorical_cols"]
        self._encoder.load_state(state["encoder"])
        self._transformer.load_state(state["transformer"])
        self._fitted = True
        return self
