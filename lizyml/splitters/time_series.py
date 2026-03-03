"""TimeSeriesSplitter — forward-chaining split for time-ordered data."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from lizyml.core.registries import SplitterRegistry

from .base import BaseSplitter


@SplitterRegistry.register("time_series")
class TimeSeriesSplitter(BaseSplitter):
    """Forward-chaining (expanding window) time series splitter.

    Each successive fold adds more training data, with a fixed validation
    window. An optional ``gap`` drops samples between the end of the training
    set and the start of the validation set to avoid temporal leakage.

    Args:
        n_splits: Number of folds.
        gap: Number of samples to drop between train and validation.
        max_train_size: Maximum training set size (None = expanding window).
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        max_train_size: int | None = None,
    ) -> None:
        self._tss = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size,
        )

    def split(
        self,
        n_samples: int,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(n_samples)
        yield from self._tss.split(indices)
