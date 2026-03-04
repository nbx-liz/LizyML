"""IsotonicCalibrator — isotonic regression calibration."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.core.registries import CalibratorRegistry


@CalibratorRegistry.register("isotonic")
class IsotonicCalibrator(BaseCalibratorAdapter):
    """Isotonic regression calibration.

    Accepts only 1-D OOF scores (no X).
    """

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None

    @property
    def name(self) -> str:
        return "isotonic"

    def fit(self, oof_scores: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(oof_scores, y.astype(float))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsotonicCalibrator has not been fitted.")
        raw: np.ndarray = self._model.predict(scores)
        clipped: np.ndarray = np.clip(raw, 0.0, 1.0)
        return clipped
