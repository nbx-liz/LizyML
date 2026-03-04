"""TrainingSpec: training configuration derived from LizyMLConfig."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InnerValidSpec:
    """Configuration for the inner validation strategy within each CV fold."""

    method: str  # "holdout"
    ratio: float
    random_state: int


@dataclass(frozen=True)
class EarlyStoppingSpec:
    """Early stopping configuration."""

    enabled: bool
    rounds: int
    inner_valid: InnerValidSpec | None


@dataclass(frozen=True)
class TrainingSpec:
    """Normalized training configuration passed downstream."""

    seed: int
    early_stopping: EarlyStoppingSpec
