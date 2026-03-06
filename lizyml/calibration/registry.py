"""Calibrator registry helpers."""

from __future__ import annotations

from typing import Any

# Side-effect imports to register all calibrators
import lizyml.calibration.beta  # noqa: F401
import lizyml.calibration.isotonic  # noqa: F401
import lizyml.calibration.platt  # noqa: F401
from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.registries import CalibratorRegistry

# Methods declared in the schema but not yet implemented.
_NOT_IMPLEMENTED: dict[str, str] = {}


def get_calibrator(
    name: str,
    params: dict[str, Any] | None = None,
) -> BaseCalibratorAdapter:
    """Return a fresh calibrator instance by name.

    Args:
        name: Registered calibrator name (e.g. ``"platt"``, ``"isotonic"``).
        params: Optional method-specific parameters (e.g. LGBM params for
            isotonic calibration).

    Returns:
        A new (unfitted) calibrator instance.

    Raises:
        LizyMLError: With ``CALIBRATION_NOT_SUPPORTED`` when *name* is not
            registered or is a known-but-unimplemented method.
    """
    if name in _NOT_IMPLEMENTED:
        raise LizyMLError(
            ErrorCode.CALIBRATION_NOT_SUPPORTED,
            user_message=_NOT_IMPLEMENTED[name],
            context={"method": name},
        )
    try:
        cls = CalibratorRegistry.get(name)
    except KeyError as err:
        raise LizyMLError(
            ErrorCode.CALIBRATION_NOT_SUPPORTED,
            user_message=f"Unknown calibration method: '{name}'.",
            context={"method": name},
        ) from err
    instance: BaseCalibratorAdapter = cls(params=params)
    return instance
