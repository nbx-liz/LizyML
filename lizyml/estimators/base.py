"""BaseEstimatorAdapter — abstract interface for ML estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import pandas as pd

ImportanceKind = Literal["split", "gain"]


class BaseEstimatorAdapter(ABC):
    """Uniform wrapper around a native ML estimator.

    All adapters expose the same fit/predict/importance surface so that the
    training core can remain estimator-agnostic.

    The adapter owns the native model object and is responsible for:
    - translating kwargs (early stopping, categorical features, etc.)
    - bridging importance kinds
    - storing ``best_iteration`` after early stopping
    """

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        **kwargs: Any,
    ) -> BaseEstimatorAdapter:
        """Fit the estimator.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_valid: Optional validation features (for early stopping).
            y_valid: Optional validation target (for early stopping).
            **kwargs: Adapter-specific arguments.

        Returns:
            ``self``
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw predictions (regression values or class labels)."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities, shape ``(n, k)`` for multiclass."""

    @abstractmethod
    def importance(self, kind: ImportanceKind = "split") -> dict[str, float]:
        """Return feature importance scores keyed by feature name.

        Args:
            kind: ``"split"`` (count of splits) or ``"gain"`` (total gain).
        """

    @abstractmethod
    def get_native_model(self) -> Any:
        """Return the underlying native model object."""

    @property
    def best_iteration(self) -> int | None:
        """Best iteration from early stopping (``None`` if not applicable)."""
        return None
