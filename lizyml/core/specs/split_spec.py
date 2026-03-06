"""SplitSpec: split configuration derived from LizyMLConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SplitSpec:
    """Normalized split configuration passed downstream to Splitters."""

    method: Literal[
        "kfold",
        "stratified_kfold",
        "group_kfold",
        "time_series",
        "purged_time_series",
        "group_time_series",
    ]
    n_splits: int
    random_state: int | None
    shuffle: bool
    gap: int
    purge_gap: int = 0
    embargo: int = 0
    train_size_max: int | None = None
    test_size_max: int | None = None
