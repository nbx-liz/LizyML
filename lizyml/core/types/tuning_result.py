"""TuningResult — result types for hyperparameter tuning (H-0023, H-0048)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrialResult:
    """Result of a single tuning trial."""

    number: int
    params: dict[str, Any]
    score: float
    state: str  # "complete" | "pruned" | "fail"

    def __post_init__(self) -> None:
        # Deep-copy mutable fields to prevent external mutation
        object.__setattr__(self, "params", dict(self.params))


@dataclass(frozen=True)
class TuningResult:
    """Result of a full hyperparameter search."""

    best_params: dict[str, Any]
    best_score: float
    trials: list[TrialResult]
    metric_name: str
    direction: str  # "minimize" | "maximize"

    def __post_init__(self) -> None:
        # Deep-copy mutable fields to prevent external mutation
        object.__setattr__(self, "best_params", dict(self.best_params))
        object.__setattr__(self, "trials", list(self.trials))


@dataclass(frozen=True)
class TuneProgressInfo:
    """Progress information emitted after each tuning trial (H-0048).

    Attributes:
        current_trial: Current trial number (1-indexed).
        total_trials: Total number of trials.
        elapsed_seconds: Seconds elapsed since tune() started.
        best_score: Best score so far (None if no complete trial yet).
        latest_score: Score of the latest trial (None if fail/pruned).
        latest_state: State of the latest trial ("complete"|"pruned"|"fail").
    """

    current_trial: int
    total_trials: int
    elapsed_seconds: float
    best_score: float | None
    latest_score: float | None
    latest_state: str  # "complete" | "pruned" | "fail"


TuneProgressCallback = Callable[[TuneProgressInfo], None]
"""Callback type for receiving tuning progress updates."""
