"""Calibrators raise a unified ``LizyMLError`` when used before fit (#214).

Before H-0086/#214 these guards raised a bare ``RuntimeError`` with no error
code or context, so ``except LizyMLError`` (the library's convention) missed
them. They are user-reachable via ``fit_result.calibrator``. Now they raise
``LizyMLError(CALIBRATION_NOT_FITTED)`` with a ``calibrator`` context tag.
"""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.calibration.beta import BetaCalibrator
from lizyml.calibration.isotonic import IsotonicCalibrator
from lizyml.calibration.platt import PlattCalibrator
from lizyml.core.exceptions import ErrorCode, LizyMLError

_SCORES = np.array([0.1, 0.5, 0.9], dtype=np.float64)


@pytest.mark.parametrize(
    ("factory", "tag"),
    [
        (PlattCalibrator, "platt"),
        (BetaCalibrator, "beta"),
        (IsotonicCalibrator, "isotonic"),
    ],
)
def test_predict_before_fit_raises_lizyml_error(factory, tag) -> None:
    calibrator = factory()
    with pytest.raises(LizyMLError) as exc:
        calibrator.predict(_SCORES)
    assert exc.value.code is ErrorCode.CALIBRATION_NOT_FITTED
    assert exc.value.context["calibrator"] == tag


@pytest.mark.parametrize(
    ("factory", "tag"),
    [
        (PlattCalibrator, "platt"),
        (BetaCalibrator, "beta"),
        (IsotonicCalibrator, "isotonic"),
    ],
)
def test_export_params_before_fit_raises_lizyml_error(factory, tag) -> None:
    calibrator = factory()
    with pytest.raises(LizyMLError) as exc:
        calibrator.export_params()
    assert exc.value.code is ErrorCode.CALIBRATION_NOT_FITTED
    assert exc.value.context["calibrator"] == tag
