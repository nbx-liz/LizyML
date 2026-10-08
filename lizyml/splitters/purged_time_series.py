"""PurgedTimeSeriesSplitter — forward-chaining time series split with a purge gap."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from .base import BaseSplitter


class PurgedTimeSeriesSplitter(BaseSplitter):
    """Expanding-window time series splitter with a purge gap.

    Each fold trains on the rows before its validation block, minus the last
    ``purge_gap`` rows. This prevents label leakage when the target is built
    from a look-forward window (e.g. future returns). Training is always
    entirely before validation, so there is no position for an embargo after
    the validation block (BLUEPRINT §10.2, H-0115).

    Args:
        n_splits: Number of folds.
        purge_gap: Number of samples purged from the tail of each training
            set (gap between train and valid).
        embargo: Deprecated (H-0115). It subtracted at the same position as
            ``purge_gap``; when given (even ``0``) it warns and is added to
            ``purge_gap``. Will be removed in v1.0.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo: int | None = None,
        max_train_size: int | None = None,
        max_test_size: int | None = None,
    ) -> None:
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0.")
        if embargo is not None:
            if embargo < 0:
                raise ValueError("embargo must be >= 0.")
            warnings.warn(
                "PurgedTimeSeriesSplitter(embargo=...) is deprecated: it widened "
                "the same gap between train and valid as 'purge_gap', and its "
                "value is now added to 'purge_gap'. Pass the total as "
                "'purge_gap'. Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            purge_gap += embargo
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.max_train_size = max_train_size
        self.max_test_size = max_test_size

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        """Yield purged ``(train_idx, valid_idx)`` pairs.

        The dataset is divided into ``n_splits + 1`` equally sized chunks.
        Fold ``k`` uses chunks ``0..k`` for training (minus the purge tail)
        and chunk ``k+1`` for validation.
        """
        indices = np.arange(n_samples)
        fold_size = n_samples // (self.n_splits + 1)
        if fold_size == 0:
            raise ValueError(
                f"n_samples={n_samples} is too small for n_splits={self.n_splits}."
            )

        for k in range(self.n_splits):
            valid_start = (k + 1) * fold_size
            valid_end = (k + 2) * fold_size
            if valid_end > n_samples:
                valid_end = n_samples
            if valid_start >= valid_end:
                continue

            train_end = (k + 1) * fold_size - self.purge_gap
            if train_end <= 0:
                continue

            train_idx = indices[:train_end]
            valid_idx = indices[valid_start:valid_end]
            if self.max_train_size is not None:
                train_idx = train_idx[-self.max_train_size :]
            if self.max_test_size is not None:
                valid_idx = valid_idx[: self.max_test_size]
            yield train_idx, valid_idx
