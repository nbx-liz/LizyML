"""ProblemSpec: normalized problem definition derived from LizyMLConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProblemSpec:
    """Normalized problem definition passed downstream to all components."""

    task: Literal["regression", "binary", "multiclass"]
    target: str
    time_col: str | None
    group_col: str | None
    data_path: str | None
