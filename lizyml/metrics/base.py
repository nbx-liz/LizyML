"""BaseMetric — abstract interface for all LizyML metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt

from lizyml.core.exceptions import ErrorCode, LizyMLError


class BaseMetric(ABC):
    """Abstract base class for all metrics.

    Subclasses must declare:
    - ``name``: unique string identifier
    - ``needs_proba``: whether predicted probabilities are required
    - ``greater_is_better``: whether higher values indicate better performance
    - ``__call__``: compute metric from arrays

    The call signature always accepts positional ``y_true`` and ``y_pred``
    and returns a single ``float``. A metric that also needs the training-side
    targets declares :attr:`needs_train_target` and is bound with
    :meth:`bind_train` first (H-0132).
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

        Note: a multiclass-compatible metric with ``needs_simplex=False``
        implies its ``__call__`` can accept a 2-D ``(n, k)`` array.
        """
        return False

    @property
    def needs_train_target(self) -> bool:
        """True when the metric needs the training-side targets (H-0132).

        Such a metric (MASE) must be bound with :meth:`bind_train` before it
        is called; callers bind only when this is ``True``. The training side
        is the rows the evaluated model was fitted on, never the scored rows.
        """
        return False

    def bind_train(self, y_train: npt.NDArray[Any]) -> BaseMetric:
        """Return a copy of this metric bound to the training-side targets.

        Only metrics with :attr:`needs_train_target` override this; the
        default refuses, so binding a metric that takes no training targets
        is an error rather than a silent no-op.

        Raises:
            LizyMLError: ``UNSUPPORTED_METRIC``.
        """

        raise LizyMLError(
            code=ErrorCode.UNSUPPORTED_METRIC,
            user_message=f"Metric '{self.name}' takes no training targets.",
            context={"metric": self.name},
        )

    @abstractmethod
    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        """Compute metric value.

        Args:
            y_true: Ground-truth targets, shape ``(n,)``.
            y_pred: Predictions or probabilities, shape ``(n,)`` or ``(n, k)``.

        Returns:
            Scalar metric value.
        """
