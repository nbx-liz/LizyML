"""CalibrationSpec: calibration configuration derived from LizyMLConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CalibrationSpec:
    """Normalized calibration configuration passed downstream."""

    method: Literal["platt", "isotonic", "beta"]
    n_splits: int
