"""TuningResult — result types for hyperparameter tuning (H-0023, H-0048, H-0068)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from lizyml.core.types.search_dim import SearchDim

# ---------------------------------------------------------------------------
# Boundary detection types (H-0068)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryDimStatus:
    """Boundary analysis for a single search dimension (H-0068).

    Attributes:
        name: Dimension name.
        best_value: Best parameter value found.
        low: Lower bound of the search range (None for categorical).
        high: Upper bound of the search range (None for categorical).
        position_pct: Relative position of best in [0.0, 1.0] (None for categorical).
        edge: Which edge is near — "lower", "upper", or "none".
        expanded: Whether this dim was expanded in the current round.
        new_low: New lower bound after expansion (None if not expanded).
        new_high: New upper bound after expansion (None if not expanded).
    """

    name: str
    best_value: float | int | str | None
    low: float | int | None
    high: float | int | None
    position_pct: float | None
    edge: str  # "lower" | "upper" | "none"
    expanded: bool
    new_low: float | int | None
    new_high: float | int | None


@dataclass(frozen=True)
class BoundaryReport:
    """Boundary detection results for all dimensions (H-0068).

    Attributes:
        dims: Per-dimension boundary analysis.
        expanded_names: Names of dimensions that were expanded.
    """

    dims: tuple[BoundaryDimStatus, ...]
    expanded_names: tuple[str, ...]


# ---------------------------------------------------------------------------
# Round summary (H-0068)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundSummary:
    """Summary of a single tuning round (H-0068).

    Attributes:
        round: Round number (1-indexed).
        n_trials: Number of trials in this round.
        best_score_before: Best score at start of round (None for round 1).
        best_score_after: Best score at end of round.
        expanded_dims: Names of dimensions expanded before this round.
        space_snapshot: Search space used in this round.
    """

    round: int
    n_trials: int
    best_score_before: float | None
    best_score_after: float
    expanded_dims: tuple[str, ...]
    space_snapshot: tuple[SearchDim, ...]


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialResult:
    """Result of a single tuning trial."""

    number: int
    params: dict[str, Any]
    score: float
    state: str  # "complete" | "pruned" | "fail"
    round: int = 1  # H-0068: which round this trial belongs to

    def __post_init__(self) -> None:
        # Deep-copy mutable fields to prevent external mutation
        object.__setattr__(self, "params", dict(self.params))


# ---------------------------------------------------------------------------
# Tuning result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuningResult:
    """Result of a full hyperparameter search (H-0050, H-0068).

    H-0068 adds ``rounds`` and ``boundary_report`` for re-tune support.
    """

    best_model_params: dict[str, Any]
    best_smart_params: dict[str, Any]
    best_training_params: dict[str, Any]
    best_score: float
    trials: list[TrialResult]
    metric_name: str
    direction: str  # "minimize" | "maximize"
    # H-0068: re-tune tracking
    rounds: tuple[RoundSummary, ...] = ()
    boundary_report: BoundaryReport | None = None

    def __post_init__(self) -> None:
        # Deep-copy mutable fields to prevent external mutation
        object.__setattr__(self, "best_model_params", dict(self.best_model_params))
        object.__setattr__(self, "best_smart_params", dict(self.best_smart_params))
        object.__setattr__(
            self, "best_training_params", dict(self.best_training_params)
        )
        object.__setattr__(self, "trials", list(self.trials))

    @property
    def best_params(self) -> dict[str, Any]:
        """Flat view of all best parameters (convenience / backward compat)."""
        return {
            **self.best_model_params,
            **self.best_smart_params,
            **self.best_training_params,
        }


# ---------------------------------------------------------------------------
# Progress callback (H-0048, H-0068)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuneProgressInfo:
    """Progress information emitted after each tuning trial (H-0048, H-0068).

    H-0068 adds round, cumulative_trials, and expanded_dims fields for
    re-tune progress monitoring.
    """

    current_trial: int
    total_trials: int
    elapsed_seconds: float
    best_score: float | None
    latest_score: float | None
    latest_state: str  # "complete" | "pruned" | "fail"
    # H-0068: re-tune progress fields
    round: int = 1
    cumulative_trials: int = field(default=0)
    expanded_dims: tuple[str, ...] = ()


TuneProgressCallback = Callable[[TuneProgressInfo], None]
"""Callback type for receiving tuning progress updates."""
