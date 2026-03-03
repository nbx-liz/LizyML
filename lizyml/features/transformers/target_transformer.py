"""TargetTransformer — stateful target-variable transformations (e.g. log1p)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

TargetTransform = Literal["none", "log1p"]


class TargetTransformer:
    """Stateful wrapper for target variable transformations.

    Args:
        transform: Transformation to apply. Supported: ``"none"``, ``"log1p"``.
    """

    def __init__(self, transform: TargetTransform = "none") -> None:
        self.transform: TargetTransform = transform

    def fit(self, y: pd.Series) -> TargetTransformer:
        """Fit on the training target (no-op for current transforms)."""
        return self

    def transform_y(self, y: pd.Series) -> pd.Series:
        """Apply the forward transformation."""
        if self.transform == "log1p":
            return pd.Series(np.log1p(y.to_numpy()), index=y.index, name=y.name)
        return y.copy()

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Apply the inverse transformation to predictions."""
        if self.transform == "log1p":
            result: np.ndarray = np.expm1(y)
            return result
        return y

    def get_state(self) -> dict[str, Any]:
        """Return serializable state."""
        return {"transform": self.transform}

    def load_state(self, state: dict[str, Any]) -> TargetTransformer:
        """Restore from saved state."""
        self.transform = state["transform"]
        return self
