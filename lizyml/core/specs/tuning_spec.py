"""TuningSpec: hyperparameter tuning configuration derived from LizyMLConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class TuningSpec:
    """Normalized tuning configuration passed downstream."""

    backend: Literal["optuna"]
    n_trials: int
    direction: Literal["minimize", "maximize"]
    timeout: float | None
    space: dict[str, Any]
