"""IsotonicCalibrator — isotonic regression calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
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

    def fit(
        self, oof_scores: npt.NDArray[np.float64], y: npt.NDArray[Any]
    ) -> IsotonicCalibrator:
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(oof_scores, y.astype(float))
        return self

    def predict(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("IsotonicCalibrator has not been fitted.")
        raw: npt.NDArray[np.float64] = self._model.predict(scores)
        clipped: npt.NDArray[np.float64] = np.clip(raw, 0.0, 1.0)
        return clipped
