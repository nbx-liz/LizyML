"""Calibrator registry helpers."""

from __future__ import annotations

# Side-effect imports to register all calibrators
import lizyml.calibration.isotonic  # noqa: F401
import lizyml.calibration.platt  # noqa: F401
from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.core.registries import CalibratorRegistry


def get_calibrator(name: str) -> BaseCalibratorAdapter:
    """Return a fresh calibrator instance by name.

    Args:
        name: Registered calibrator name (e.g. ``"platt"``, ``"isotonic"``).

    Returns:
        A new (unfitted) calibrator instance.

    Raises:
        KeyError: When *name* is not registered.
    """
    cls = CalibratorRegistry.get(name)
    instance: BaseCalibratorAdapter = cls()
    return instance
