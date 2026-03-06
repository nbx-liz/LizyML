"""IsotonicCalibrator — isotonic regression via LGBM monotone constraints.

Uses a single-feature LightGBM regressor with ``monotone_constraints=[1]``
to learn a monotone mapping from raw scores to calibrated probabilities
(BLUEPRINT §12.2).
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgbm
import numpy as np
import numpy.typing as npt

from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.core.registries import CalibratorRegistry

_ISOTONIC_DEFAULTS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
}


@CalibratorRegistry.register("isotonic")
class IsotonicCalibrator(BaseCalibratorAdapter):
    """Isotonic calibration using LGBM with monotone constraints.

    Learns a monotone non-decreasing mapping from raw scores (logits)
    to calibrated probabilities using a single-feature LightGBM regressor.

    Args:
        params: Optional LGBM parameter overrides.  ``monotone_constraints``
            is always forced to ``[1]`` and cannot be overridden.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = {**_ISOTONIC_DEFAULTS, **(params or {})}
        merged["monotone_constraints"] = [1]
        merged["verbose"] = -1
        self._lgbm_params = merged
        self._model: lgbm.LGBMRegressor | None = None

    @property
    def name(self) -> str:
        return "isotonic"

    def fit(
        self, oof_scores: npt.NDArray[np.float64], y: npt.NDArray[Any]
    ) -> IsotonicCalibrator:
        X_cal = oof_scores.reshape(-1, 1)
        self._model = lgbm.LGBMRegressor(**self._lgbm_params)
        self._model.fit(X_cal, y.astype(float))
        return self

    def predict(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("IsotonicCalibrator has not been fitted.")
        X = scores.reshape(-1, 1)
        raw: npt.NDArray[np.float64] = self._model.predict(X)
        clipped: npt.NDArray[np.float64] = np.clip(raw, 0.0, 1.0)
        return clipped
