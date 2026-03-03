"""PurgedTimeSeriesSplitter — time series split with a purge window."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from lizyml.core.registries import SplitterRegistry

from .base import BaseSplitter


@SplitterRegistry.register("purged_time_series")
class PurgedTimeSeriesSplitter(BaseSplitter):
    """Expanding-window time series splitter with a purge window.

    After each train/valid split, ``purge_window`` samples at the *end* of the
    training set are removed from training.  This prevents label leakage when
    the target is constructed from a look-forward window (e.g. future returns).

    Args:
        n_splits: Number of folds.
        purge_window: Number of samples purged from the tail of each training
            set.
        gap: Additional samples dropped between purged train and validation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 0,
        gap: int = 0,
    ) -> None:
        if purge_window < 0:
            raise ValueError("purge_window must be >= 0.")
        if gap < 0:
            raise ValueError("gap must be >= 0.")
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.gap = gap

    def split(
        self,
        n_samples: int,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield purged ``(train_idx, valid_idx)`` pairs.

        The dataset is divided into ``n_splits + 1`` equally sized chunks.
        Fold ``k`` uses chunks ``0..k`` for training (minus the purge tail)
        and chunk ``k+1`` for validation (offset by ``gap``).
        """
        indices = np.arange(n_samples)
        fold_size = n_samples // (self.n_splits + 1)
        if fold_size == 0:
            raise ValueError(
                f"n_samples={n_samples} is too small for n_splits={self.n_splits}."
            )

        for k in range(self.n_splits):
            valid_start = (k + 1) * fold_size + self.gap
            valid_end = (k + 2) * fold_size
            if valid_end > n_samples:
                valid_end = n_samples
            if valid_start >= valid_end:
                continue

            train_end = (k + 1) * fold_size - self.purge_window
            if train_end <= 0:
                continue

            train_idx = indices[:train_end]
            valid_idx = indices[valid_start:valid_end]
            yield train_idx, valid_idx
