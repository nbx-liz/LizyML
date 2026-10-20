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
    - ``predict`` maps raw scores (logits) to calibrated probabilities in [0, 1].
    - Input scores are raw logits (before sigmoid/softmax), not probabilities.
    - The same (oof_scores, y) must produce identical results across calls
      with the same random state.
    """

    @abstractmethod
    def fit(
        self,
        oof_scores: npt.NDArray[np.float64],
        y: npt.NDArray[Any],
    ) -> BaseCalibratorAdapter:
        """Fit the calibrator on OOF raw scores and ground-truth labels.

        Args:
            oof_scores: 1-D array of raw model scores (logits).
            y: 1-D array of binary ground-truth labels (0/1).

        Returns:
            ``self`` for chaining.
        """

    @abstractmethod
    def predict(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map raw scores (logits) to calibrated probabilities.

        Args:
            scores: 1-D array of raw model scores (logits).

        Returns:
            1-D array of calibrated probabilities in [0, 1].
        """

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        """Refuse ``calibration.params`` this calibrator cannot honour (H-0100).

        The Facade calls this before any training, so a setting the calibrator
        would not read is refused instead of being accepted and ignored. Every
        registered calibrator declares its own; this default accepts nothing,
        so a calibrator that forgets to declare cannot silently discard params.

        Raises:
            LizyMLError: ``CONFIG_INVALID`` naming ``calibration.params``.
        """
        from lizyml.core.exceptions import ErrorCode, LizyMLError

        if params:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    f"calibration.params is not accepted by this calibrator; "
                    f"got {sorted(params)}."
                ),
                context={"surface": "calibration.params", "parameters": sorted(params)},
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this calibration method."""

    @abstractmethod
    def export_params(self) -> dict[str, Any]:
        """Export calibrator parameters as a JSON-serializable dict.

        The dict must include a ``"method"`` key matching :attr:`name`.
        Raises ``RuntimeError`` if the calibrator has not been fitted.
        """


class BaseCalibrationResult:
    """Marker base of a cross-fit calibration outcome (H-0130).

    Layer 3 (``plots/``) checks a fitted model's calibrator against this marker
    instead of importing ``calibration/cross_fit.py`` (BLUEPRINT §2.1). Within
    lizyml only ``CalibrationResult`` inherits it. A class that inherits it
    promises a ``calibrated_oof`` array, the only field the plots read.

    The marker holds no state and declares no dataclass field, so a pickled
    ``CalibrationResult`` keeps its recorded class name, and one saved before
    the marker existed is an instance of it once loaded.
    """

    calibrated_oof: npt.NDArray[np.float64]
