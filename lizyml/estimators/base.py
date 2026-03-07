"""BaseEstimatorAdapter — abstract interface for ML estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
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
    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return raw predictions (regression values or class labels)."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return class probabilities, shape ``(n, k)`` for multiclass."""

    @abstractmethod
    def predict_raw(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return raw scores (logits) before sigmoid/softmax.

        For regression, identical to ``predict()``.
        For binary, returns 1-D logit scores.
        For multiclass, returns ``(n, k)`` raw scores.
        """

    @abstractmethod
    def importance(self, kind: ImportanceKind = "split") -> dict[str, float]:
        """Return feature importance scores keyed by feature name.

        Args:
            kind: ``"split"`` (count of splits) or ``"gain"`` (total gain).
        """

    @abstractmethod
    def get_native_model(self) -> Any:
        """Return the underlying native model object."""

    def update_params(self, params: dict[str, Any]) -> None:
        """Update estimator parameters before ``fit()`` is called.

        Used by CVTrainer for per-fold ratio parameter resolution (H-0036).
        Subclasses should override if they store params internally.
        """
        return  # noqa: B027 — intentional no-op default

    @property
    def best_iteration(self) -> int | None:
        """Best iteration from early stopping (``None`` if not applicable)."""
        return None

    @property
    def eval_results(self) -> dict[str, Any]:
        """Evaluation results collected during training.

        Returns an empty dict by default. Subclasses should override to
        return the actual evaluation history.
        """
        return {}
