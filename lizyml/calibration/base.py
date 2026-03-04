"""BaseCalibratorAdapter — abstract interface for probability calibrators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class BaseCalibratorAdapter(ABC):
    """Abstract base for all probability calibrators.

    Contract:
    - ``fit`` receives only ``(oof_scores, y)`` — no X allowed.
    - ``predict`` maps raw scores to calibrated probabilities in [0, 1].
    - The same (oof_scores, y) must produce identical results across calls
      with the same random state.
    """

    @abstractmethod
    def fit(
        self,
        oof_scores: npt.NDArray[np.float64],
        y: npt.NDArray[Any],
    ) -> BaseCalibratorAdapter:
        """Fit the calibrator on OOF scores and ground-truth labels.

        Args:
            oof_scores: 1-D array of raw model scores (probabilities or logits).
            y: 1-D array of binary ground-truth labels (0/1).

        Returns:
            ``self`` for chaining.
        """

    @abstractmethod
    def predict(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map raw scores to calibrated probabilities.

        Args:
            scores: 1-D array of raw model scores.

        Returns:
            1-D array of calibrated probabilities in [0, 1].
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this calibration method."""
