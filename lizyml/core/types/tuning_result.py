"""TuningResult — result types for hyperparameter tuning (H-0023)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrialResult:
    """Result of a single tuning trial."""

    number: int
    params: dict[str, Any]
    score: float
    state: str  # "complete" | "pruned" | "fail"


@dataclass(frozen=True)
class TuningResult:
    """Result of a full hyperparameter search."""

    best_params: dict[str, Any]
    best_score: float
    trials: list[TrialResult]
    metric_name: str
    direction: str  # "minimize" | "maximize"
