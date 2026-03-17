"""GroupTimeSeriesSplitter — time-ordered split with group-level isolation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from .base import BaseSplitter


class GroupTimeSeriesSplitter(BaseSplitter):
    """Time series splitter that respects group boundaries.

    Groups are sorted by their first occurrence in the sequence and then
    assigned to folds in chronological order.  No group will ever appear in
    both the training set and the validation set of the same fold.

    Args:
        n_splits: Number of folds.
        gap: Number of groups to skip between the last training group and the
            first validation group (temporal buffer).
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        max_train_size: int | None = None,
        max_test_size: int | None = None,
    ) -> None:
        if gap < 0:
            raise ValueError("gap must be >= 0.")
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
        self.max_test_size = max_test_size

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        if groups is None:
            raise ValueError("GroupTimeSeriesSplitter requires groups to be provided.")

        groups_arr = np.asarray(groups)
        # Determine the unique groups in order of first appearance
        _, first_idx = np.unique(groups_arr, return_index=True)
        ordered_unique = groups_arr[np.sort(first_idx)]
        n_groups = len(ordered_unique)

        # Need at least (n_splits + 1) groups to form n_splits folds
        min_groups = self.n_splits + 1 + self.gap
        if n_groups < min_groups:
            raise ValueError(
                f"Not enough groups ({n_groups}) for n_splits={self.n_splits} "
                f"with gap={self.gap}. Need at least {min_groups} groups."
            )

        group_fold_size = n_groups // (self.n_splits + 1)

        for k in range(self.n_splits):
            train_group_end = (k + 1) * group_fold_size
            valid_group_start = train_group_end + self.gap
            # Last fold extends to include all trailing groups
            if k == self.n_splits - 1:
                valid_group_end = n_groups
            else:
                valid_group_end = valid_group_start + group_fold_size
            if valid_group_end > n_groups:
                valid_group_end = n_groups
            if valid_group_start >= valid_group_end:
                continue

            train_group_list = ordered_unique[:train_group_end]
            valid_group_list = ordered_unique[valid_group_start:valid_group_end]
            if self.max_train_size is not None:
                train_group_list = train_group_list[-self.max_train_size :]
            if self.max_test_size is not None:
                valid_group_list = valid_group_list[: self.max_test_size]
            train_groups = set(train_group_list)
            valid_groups = set(valid_group_list)

            train_idx = np.where(np.isin(groups_arr, list(train_groups)))[0]
            valid_idx = np.where(np.isin(groups_arr, list(valid_groups)))[0]
            yield train_idx, valid_idx
