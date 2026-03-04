"""PlattCalibrator — logistic regression calibration (Platt scaling)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.core.registries import CalibratorRegistry


@CalibratorRegistry.register("platt")
class PlattCalibrator(BaseCalibratorAdapter):
    """Platt scaling: fits a logistic regression on OOF scores.

    Accepts only 1-D OOF scores (no X).
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None

    @property
    def name(self) -> str:
        return "platt"

    def fit(self, oof_scores: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        self._model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        self._model.fit(oof_scores.reshape(-1, 1), y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PlattCalibrator has not been fitted.")
        result: np.ndarray = self._model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return result
