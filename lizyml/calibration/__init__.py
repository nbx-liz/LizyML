"""Calibration — probability calibration for binary classification."""

from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.calibration.cross_fit import CalibrationResult, cross_fit_calibrate
from lizyml.calibration.isotonic import IsotonicCalibrator
from lizyml.calibration.platt import PlattCalibrator
from lizyml.calibration.registry import get_calibrator

__all__ = [
    "BaseCalibratorAdapter",
    "CalibrationResult",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "cross_fit_calibrate",
    "get_calibrator",
]
