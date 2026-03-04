"""Splitters package — re-exports and factory helper."""

from __future__ import annotations

from lizyml.core.specs.split_spec import SplitSpec
from lizyml.splitters.base import BaseSplitter
from lizyml.splitters.group_kfold import (
    GroupKFoldSplitter,
    StratifiedGroupKFoldSplitter,
)
from lizyml.splitters.group_time_series import GroupTimeSeriesSplitter
from lizyml.splitters.kfold import KFoldSplitter, StratifiedKFoldSplitter
from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter
from lizyml.splitters.time_series import TimeSeriesSplitter

__all__ = [
    "BaseSplitter",
    "GroupKFoldSplitter",
    "GroupTimeSeriesSplitter",
    "KFoldSplitter",
    "PurgedTimeSeriesSplitter",
    "StratifiedGroupKFoldSplitter",
    "StratifiedKFoldSplitter",
    "TimeSeriesSplitter",
]


def _build_splitter(spec: SplitSpec) -> BaseSplitter:
    """Instantiate the correct splitter from a ``SplitSpec``."""
    method = spec.method

    if method == "kfold":
        return KFoldSplitter(
            n_splits=spec.n_splits,
            shuffle=spec.shuffle,
            random_state=spec.random_state,
        )
    if method == "stratified_kfold":
        return StratifiedKFoldSplitter(
            n_splits=spec.n_splits,
            shuffle=spec.shuffle,
            random_state=spec.random_state,
        )
    if method == "group_kfold":
        return GroupKFoldSplitter(n_splits=spec.n_splits)
    if method == "time_series":
        return TimeSeriesSplitter(
            n_splits=spec.n_splits,
            gap=spec.gap,
        )

    raise ValueError(f"Unknown split method: {method!r}")
