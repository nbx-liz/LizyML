"""FeatureTransformer — stateless passthrough (extensible for future transforms)."""

from __future__ import annotations

from typing import Any

import pandas as pd


class FeatureTransformer:
    """Placeholder for feature-level transformations (normalization, etc.).

    Currently a no-op passthrough. Future implementations may add
    standard scaling, power transforms, or other feature engineering.
    """

    def fit(self, X: pd.DataFrame) -> FeatureTransformer:
        """Fit on training data (no-op for base implementation)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation (no-op for base implementation)."""
        return X.copy()

    def get_state(self) -> dict[str, Any]:
        """Return serializable state."""
        return {}

    def load_state(self, state: dict[str, Any]) -> FeatureTransformer:
        """Restore from saved state."""
        return self
