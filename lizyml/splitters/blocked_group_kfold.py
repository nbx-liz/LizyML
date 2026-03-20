"""BlockedGroupKFoldSplitter — 2-axis CV: period blocks × group KFold (H-0060).

Splits data along two axes:
1. **Period axis**: cutoffs define time/ordinal boundaries for train/valid periods
2. **Group axis**: entities are KFold-split so no group appears in both train and valid

Each fold is a cartesian product: (time fold t) × (group fold u).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from lizyml.splitters.base import BaseSplitter


class BlockedGroupKFoldSplitter(BaseSplitter):
    """2-axis cross-validation: period blocks × group KFold.

    Args:
        block_values: Sorted column values defining period membership.
        cutoffs: Boundary values. Each cutoff starts a new valid period.
        mode: ``"expanding"`` (cumulative train) or ``"sliding"`` (fixed window).
        train_window: Number of periods in train for sliding mode.
        n_splits: Number of group folds (K).
        stratify: Stratify group folds by majority-class label.
        shuffle: Shuffle groups before splitting.
        random_state: Seed for reproducibility.
        min_train_rows: Skip fold if train has fewer rows.
        min_valid_rows: Skip fold if valid has fewer rows.
    """

    def __init__(
        self,
        block_values: npt.NDArray[Any],
        cutoffs: list[Any],
        mode: Literal["expanding", "sliding"] = "expanding",
        train_window: int | None = None,
        n_splits: int = 3,
        stratify: bool = False,
        shuffle: bool = True,
        random_state: int | None = None,
        min_train_rows: int = 1,
        min_valid_rows: int = 1,
    ) -> None:
        self._block_values = block_values
        self._cutoffs = sorted(cutoffs)
        self._mode = mode
        self._train_window = train_window
        self._n_splits = n_splits
        self._stratify = stratify
        self._shuffle = shuffle
        self._random_state = random_state
        self._min_train_rows = min_train_rows
        self._min_valid_rows = min_valid_rows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        if groups is None:
            raise ValueError("blocked_group_kfold requires groups (groups.col values)")

        period_masks = self._build_period_masks()

        for t in range(len(self._cutoffs)):
            train_mask, valid_mask = self._assign_periods(period_masks, t)
            train_period_idx = np.where(train_mask)[0]
            valid_period_idx = np.where(valid_mask)[0]

            if len(train_period_idx) == 0 or len(valid_period_idx) == 0:
                continue

            # Collect all users from both periods
            all_users_in_fold = np.unique(
                np.concatenate([groups[train_period_idx], groups[valid_period_idx]])
            )

            # Split users into K groups
            user_folds = self._split_users(all_users_in_fold, y, groups)

            for train_users, valid_users in user_folds:
                train_user_set = set(train_users)
                valid_user_set = set(valid_users)

                train_idx = train_period_idx[
                    np.isin(groups[train_period_idx], list(train_user_set))
                ]
                valid_idx = valid_period_idx[
                    np.isin(groups[valid_period_idx], list(valid_user_set))
                ]

                if (
                    len(train_idx) < self._min_train_rows
                    or len(valid_idx) < self._min_valid_rows
                ):
                    warnings.warn(
                        f"Skipping fold: train={len(train_idx)} rows, "
                        f"valid={len(valid_idx)} rows "
                        f"(min_train={self._min_train_rows}, "
                        f"min_valid={self._min_valid_rows})",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                yield (
                    train_idx.astype(np.intp),
                    valid_idx.astype(np.intp),
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_period_masks(self) -> list[npt.NDArray[np.bool_]]:
        """Build boolean masks for each period P0..Pn."""
        bv = self._block_values
        masks: list[npt.NDArray[np.bool_]] = []

        # P0: block_values < cutoffs[0]
        masks.append(bv < self._cutoffs[0])

        # P1..Pn-1: cutoffs[i-1] <= block_values < cutoffs[i]
        for i in range(1, len(self._cutoffs)):
            masks.append((bv >= self._cutoffs[i - 1]) & (bv < self._cutoffs[i]))

        # Pn: block_values >= cutoffs[-1]
        masks.append(bv >= self._cutoffs[-1])

        return masks

    def _assign_periods(
        self,
        period_masks: list[npt.NDArray[np.bool_]],
        time_fold: int,
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Assign periods to train/valid for a given time fold."""
        n = len(self._block_values)
        train_mask = np.zeros(n, dtype=bool)
        valid_mask = period_masks[time_fold + 1]  # valid = next period after cutoff

        if self._mode == "expanding":
            # Train = P0 .. P(time_fold)
            for i in range(time_fold + 1):
                train_mask |= period_masks[i]
        else:
            # Sliding: train = last train_window periods before valid
            if self._train_window is None:
                raise ValueError("train_window must be set when mode='sliding'")
            start = max(0, time_fold + 1 - self._train_window)
            for i in range(start, time_fold + 1):
                train_mask |= period_masks[i]

        return train_mask, valid_mask

    def _split_users(
        self,
        all_users: npt.NDArray[Any],
        y: npt.NDArray[Any] | None,
        groups: npt.NDArray[Any],
    ) -> list[tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
        """Split users into n_splits (train_users, valid_users) pairs."""
        rng = np.random.RandomState(self._random_state)
        n_users = len(all_users)

        if self._stratify and y is not None:
            # Compute majority-class label per user
            user_labels = self._majority_labels(all_users, y, groups)
            user_order = self._stratified_partition(all_users, user_labels, rng)
        elif self._shuffle:
            user_order = rng.permutation(n_users)
        else:
            user_order = np.arange(n_users)

        # KFold-style split over user indices
        folds: list[tuple[npt.NDArray[Any], npt.NDArray[Any]]] = []
        fold_sizes = np.full(self._n_splits, n_users // self._n_splits, dtype=int)
        fold_sizes[: n_users % self._n_splits] += 1

        current = 0
        for k in range(self._n_splits):
            valid_user_idx = user_order[current : current + fold_sizes[k]]
            train_user_idx = np.concatenate(
                [user_order[:current], user_order[current + fold_sizes[k] :]]
            )
            folds.append((all_users[train_user_idx], all_users[valid_user_idx]))
            current += fold_sizes[k]

        return folds

    def _majority_labels(
        self,
        users: npt.NDArray[Any],
        y: npt.NDArray[Any],
        groups: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Compute majority-class label for each user."""
        labels = np.empty(len(users), dtype=y.dtype)
        for i, user in enumerate(users):
            mask = groups == user
            if not np.any(mask):  # pragma: no cover — should not be reachable
                labels[i] = y[0]
                continue
            user_y = y[mask]
            values, counts = np.unique(user_y, return_counts=True)
            labels[i] = values[counts.argmax()]
        return labels

    def _stratified_partition(
        self,
        users: npt.NDArray[Any],
        user_labels: npt.NDArray[Any],
        rng: np.random.RandomState,
    ) -> npt.NDArray[np.intp]:
        """Return a permutation of user indices that interleaves classes."""
        classes = np.unique(user_labels)
        per_class: list[npt.NDArray[np.intp]] = []
        for cls in classes:
            idx = np.where(user_labels == cls)[0]
            if self._shuffle:
                rng.shuffle(idx)
            per_class.append(idx)

        # Round-robin interleave to distribute classes evenly across folds
        result: list[int] = []
        max_len = max(len(c) for c in per_class)
        for i in range(max_len):
            for cls_idx in per_class:
                if i < len(cls_idx):
                    result.append(int(cls_idx[i]))

        return np.array(result, dtype=np.intp)
