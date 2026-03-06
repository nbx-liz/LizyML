"""BetaCalibrator — Beta calibration for binary classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.registries import CalibratorRegistry

_scipy: Any = None
try:
    import scipy  # noqa: F401

    _scipy = scipy
except ImportError:
    pass


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    result: npt.NDArray[np.float64] = np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
    return result


@CalibratorRegistry.register("beta")
class BetaCalibrator(BaseCalibratorAdapter):
    """Beta calibration: 3-parameter model ``sigmoid(a*log(s) + b*log(1-s) + c)``.

    Accepts raw scores (logits) as input. Internally converts to probabilities
    via sigmoid, then applies the beta calibration model.

    Requires ``scipy`` (optional dependency).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params: tuple[float, float, float] | None = None

    @property
    def name(self) -> str:
        return "beta"

    def fit(
        self, oof_scores: npt.NDArray[np.float64], y: npt.NDArray[Any]
    ) -> BetaCalibrator:
        if _scipy is None:
            raise LizyMLError(
                code=ErrorCode.OPTIONAL_DEP_MISSING,
                user_message=(
                    "scipy is required for Beta calibration. "
                    "Install with: pip install 'lizyml[calibration]'"
                ),
                context={"package": "scipy"},
            )
        from scipy.optimize import minimize

        s = _sigmoid(oof_scores)
        s = np.clip(s, 1e-10, 1 - 1e-10)
        y_f = y.astype(np.float64)

        log_s = np.log(s)
        log_1ms = np.log(1 - s)

        def neg_log_likelihood(params: npt.NDArray[np.float64]) -> float:
            a, b, c = params[0], params[1], params[2]
            logit = a * log_s + b * log_1ms + c
            p = _sigmoid(logit)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            nll: float = float(-np.sum(y_f * np.log(p) + (1 - y_f) * np.log(1 - p)))
            return nll

        result = minimize(
            neg_log_likelihood,
            x0=np.array([1.0, 1.0, 0.0]),
            method="L-BFGS-B",
        )
        self._params = (float(result.x[0]), float(result.x[1]), float(result.x[2]))
        return self

    def predict(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._params is None:
            raise RuntimeError("BetaCalibrator has not been fitted.")

        a, b, c = self._params
        s = _sigmoid(scores)
        s = np.clip(s, 1e-10, 1 - 1e-10)
        logit: npt.NDArray[np.float64] = a * np.log(s) + b * np.log(1 - s) + c
        result = _sigmoid(logit)
        clipped: npt.NDArray[np.float64] = np.clip(result, 0.0, 1.0)
        return clipped
