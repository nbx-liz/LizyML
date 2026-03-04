"""BaseFeaturePipeline — abstract contract for all feature pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseFeaturePipeline(ABC):
    """Abstract base for feature pipelines.

    All pipelines must support stateful fit/transform and serializable state
    so they can be saved inside a ``FitResult`` and restored at inference time.

    Leakage contract:
    - ``fit`` must be called only on training data for a given fold.
    - ``transform`` applies the state learned during ``fit`` without
      re-fitting on new data.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseFeaturePipeline:
        """Fit the pipeline on training data.

        Args:
            X: Feature DataFrame (training fold only).
            y: Target series (training fold only).

        Returns:
            ``self`` for chaining.
        """

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted pipeline to a DataFrame.

        Args:
            X: Feature DataFrame (any split — validation, test, etc.).

        Returns:
            Transformed DataFrame.
        """

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit then transform in one step."""
        return self.fit(X, y).transform(X)

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return a serializable snapshot of the pipeline state.

        Returns:
            A dict that can be passed to ``load_state`` to restore the pipeline.
        """

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> BaseFeaturePipeline:
        """Restore the pipeline from a previously saved state.

        Args:
            state: Dict returned by ``get_state``.

        Returns:
            ``self`` for chaining.
        """
