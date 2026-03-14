"""BaseMetric — abstract interface for all LizyML metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt


class BaseMetric(ABC):
    """Abstract base class for all metrics.

    Subclasses must declare:
    - ``name``: unique string identifier
    - ``needs_proba``: whether predicted probabilities are required
    - ``greater_is_better``: whether higher values indicate better performance
    - ``__call__``: compute metric from arrays

    The call signature always accepts positional ``y_true`` and ``y_pred``
    and returns a single ``float``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier (e.g. ``"rmse"``)."""

    @property
    @abstractmethod
    def needs_proba(self) -> bool:
        """True when *y_pred* must be class probabilities, not hard labels."""

    @property
    @abstractmethod
    def greater_is_better(self) -> bool:
        """True when a higher score means a better model."""

    @property
    def needs_simplex(self) -> bool:
        """True when multiclass *y_pred* must form a valid probability
        distribution (row sums = 1).

        Override to ``True`` for metrics like AUC and LogLoss that
        require simplex-normalised predictions.  Per-class OvR metrics
        (e.g. AUCPR, Brier) should keep the default ``False``.
        """
        return False

    @abstractmethod
    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        """Compute metric value.

        Args:
            y_true: Ground-truth targets, shape ``(n,)``.
            y_pred: Predictions or probabilities, shape ``(n,)`` or ``(n, k)``.

        Returns:
            Scalar metric value.
        """
