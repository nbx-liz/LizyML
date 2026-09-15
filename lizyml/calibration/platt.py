"""PlattCalibrator — Platt scaling, fitted as Platt defined it (H-0100).

Platt (1999) maps a raw score ``f`` to ``P(y=1|f) = 1 / (1 + exp(A f + B))`` and
fits the slope ``A`` and the intercept ``B`` jointly by maximum likelihood, with
smoothed targets ``t+ = (N+ + 1)/(N+ + 2)`` and ``t- = 1/(N- + 2)`` and no penalty.
The intercept is part of the model: it corrects a score whose zero is not
probability one half, and it is never removed.

Until H-0100 this used ``LogisticRegression(C=1.0)`` -- an L2 penalty and 0/1
targets, neither of which is Platt's method -- and ignored ``calibration.params``.

The coefficients are held in the exported form ``sigmoid(a * s + b)``, so
``a = -A`` and ``b = -B``. ``x0`` and ``bounds`` in ``calibration.params`` are
written in those same coordinates, the ones ``calibrator.json`` shows.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from lizyml.calibration._optimizer import (
    METHODS,
    resolve_minimize_kwargs,
    validate_optimizer_params,
)
from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.registries import CalibratorRegistry

#: Scores larger than this in absolute value are rescaled before optimising, as
#: scikit-learn's ``_sigmoid_calibration`` does. Numerical only: the rescaled
#: problem is the same problem once ``x0`` and ``bounds`` are transformed.
_RESCALE_THRESHOLD = 30.0

#: Per-method default ``options``: scikit-learn's Platt uses these for L-BFGS-B.
_DEFAULT_OPTIONS: dict[str, dict[str, float]] = {
    "L-BFGS-B": {"gtol": 1e-6, "ftol": 64 * float(np.finfo(float).eps)},
}

_N_COEF = 2


def _check_target_smoothing(value: Any) -> str | None:
    if not isinstance(value, bool):
        return f"must be true or false; got {value!r}."
    return None


@CalibratorRegistry.register("platt")
class PlattCalibrator(BaseCalibratorAdapter):
    """Platt scaling on 1-D raw OOF scores (no X).

    Args:
        params: ``calibration.params``. Accepted: ``x0`` (``[a, b]``), ``method``,
            ``bounds`` (``[[a_low, a_high], [b_low, b_high]]``), ``tol``,
            ``options``, ``target_smoothing`` (default ``True``).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params: dict[str, Any] = dict(params) if params else {}
        self.validate_params(self._params)
        self._coef: tuple[float, float] | None = None

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        """Refuse any ``calibration.params`` entry Platt scaling cannot honour."""
        validate_optimizer_params(
            params,
            calibrator="platt",
            n_coef=_N_COEF,
            extra={"target_smoothing": _check_target_smoothing},
        )

    @property
    def name(self) -> str:
        return "platt"

    def fit(
        self, oof_scores: npt.NDArray[np.float64], y: npt.NDArray[Any]
    ) -> PlattCalibrator:
        from scipy.optimize import minimize

        scores = np.asarray(oof_scores, dtype=np.float64).ravel()
        positive = np.asarray(y, dtype=np.float64).ravel() > 0
        n_pos = float(positive.sum())
        n_neg = float(positive.size - n_pos)

        if self._params.get("target_smoothing", True):
            targets = np.where(
                positive, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0)
            )
        else:
            targets = positive.astype(np.float64)

        largest = float(np.max(np.abs(scores))) if scores.size else 0.0
        scale = largest if largest >= _RESCALE_THRESHOLD else 1.0
        rescaled = scores / scale

        kwargs = resolve_minimize_kwargs(
            self._params,
            n_coef=_N_COEF,
            # Platt's starting point, as scikit-learn uses it: A = 0 and
            # B = log((N- + 1) / (N+ + 1)), so a = 0 and b = -B.
            default_x0=[0.0, -math.log((n_neg + 1.0) / (n_pos + 1.0))],
            default_options=_DEFAULT_OPTIONS,
        )
        # The slope of the rescaled problem is scale * a; move the written start
        # and bounds into those coordinates so the constrained problem is the same.
        kwargs["x0"] = [kwargs["x0"][0] * scale, kwargs["x0"][1]]
        if "bounds" in kwargs:
            (a_low, a_high), b_bounds = kwargs["bounds"]
            kwargs["bounds"] = [
                (
                    None if a_low is None else a_low * scale,
                    None if a_high is None else a_high * scale,
                ),
                b_bounds,
            ]

        gradient = METHODS[kwargs["method"]].gradient

        def objective(coef: npt.NDArray[np.float64]) -> Any:
            logits = coef[0] * rescaled + coef[1]
            loss = float(np.sum(np.logaddexp(0.0, logits) - targets * logits))
            if not gradient:
                return loss
            residual = _expit(logits) - targets
            return loss, np.array([residual @ rescaled, residual.sum()])

        result = minimize(objective, jac=True if gradient else None, **kwargs)
        self._coef = (float(result.x[0]) / scale, float(result.x[1]))
        return self

    def predict(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a, b = self._fitted()
        return _expit(a * np.asarray(scores, dtype=np.float64) + b)

    def export_params(self) -> dict[str, Any]:
        """Export Platt parameters: sigmoid(a * score + b)."""
        a, b = self._fitted()
        return {"method": "platt", "a": a, "b": b}

    def _fitted(self) -> tuple[float, float]:
        if self._coef is None:
            raise LizyMLError(
                code=ErrorCode.CALIBRATION_NOT_FITTED,
                user_message="PlattCalibrator has not been fitted.",
                context={"calibrator": "platt"},
            )
        return self._coef

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled calibrator, including one saved before H-0100.

        Artifacts pickle the calibrator object. Before H-0100 its state was
        ``{"_model": LogisticRegression}``; the fitted slope and intercept are the
        exported ``a`` and ``b``, so the old model keeps predicting exactly the same
        ``sigmoid(a * s + b)``. ``FORMAT_VERSION`` does not change: the artifact
        contract and the old model's inference are both preserved.
        """
        if "_model" in state and "_coef" not in state:
            model = state["_model"]
            coef = (
                None
                if model is None
                else (float(model.coef_[0, 0]), float(model.intercept_[0]))
            )
            state = {"_params": {}, "_coef": coef}
        self.__dict__.update(state)


def _expit(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    from scipy.special import expit

    result: npt.NDArray[np.float64] = expit(x)
    return result
