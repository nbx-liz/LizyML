"""TimeSeriesSplitter — forward-chaining split for time-ordered data."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import TimeSeriesSplit

from .base import BaseSplitter


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
        max_test_size: int | None = None,
    ) -> None:
        self._tss = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size,
            test_size=max_test_size,
        )

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        indices = np.arange(n_samples)
        yield from self._tss.split(indices)
