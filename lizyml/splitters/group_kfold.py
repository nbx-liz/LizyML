"""GroupKFoldSplitter and StratifiedGroupKFoldSplitter."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .base import BaseSplitter


class GroupKFoldSplitter(BaseSplitter):
    """K-fold splitter that keeps all samples from each group in the same fold.

    Args:
        n_splits: Number of folds.
    """

    def __init__(self, n_splits: int = 5) -> None:
        self._gkf = GroupKFold(n_splits=n_splits)

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        if groups is None:
            raise ValueError("GroupKFoldSplitter requires groups to be provided.")
        indices = np.arange(n_samples)
        yield from self._gkf.split(indices, y, groups)


class StratifiedGroupKFoldSplitter(BaseSplitter):
    """Stratified K-fold splitter with group constraints.

    Combines stratification by label with group-level isolation.

    Args:
        n_splits: Number of folds.
        shuffle: Whether to shuffle groups before splitting.
        random_state: Seed for reproducibility (used only when ``shuffle=True``).
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = 42,
    ) -> None:
        self._sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        if y is None:
            raise ValueError("StratifiedGroupKFoldSplitter requires y to be provided.")
        if groups is None:
            raise ValueError(
                "StratifiedGroupKFoldSplitter requires groups to be provided."
            )
        indices = np.arange(n_samples)
        yield from self._sgkf.split(indices, y, groups)
