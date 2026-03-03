"""KFoldSplitter and StratifiedKFoldSplitter."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from lizyml.core.registries import SplitterRegistry

from .base import BaseSplitter


@SplitterRegistry.register("kfold")
class KFoldSplitter(BaseSplitter):
    """Standard K-fold cross-validation splitter.

    Args:
        n_splits: Number of folds.
        shuffle: Whether to shuffle before splitting.
        random_state: Seed for reproducibility (used only when ``shuffle=True``).
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = 42,
    ) -> None:
        self._kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    def split(
        self,
        n_samples: int,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(n_samples)
        yield from self._kf.split(indices)


@SplitterRegistry.register("stratified_kfold")
class StratifiedKFoldSplitter(BaseSplitter):
    """Stratified K-fold splitter that preserves class distribution.

    Args:
        n_splits: Number of folds.
        shuffle: Whether to shuffle before splitting.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = 42,
    ) -> None:
        self._skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    def split(
        self,
        n_samples: int,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if y is None:
            raise ValueError("StratifiedKFoldSplitter requires y to be provided.")
        indices = np.arange(n_samples)
        yield from self._skf.split(indices, y)
