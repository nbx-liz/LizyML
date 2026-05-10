"""Inner validation strategies for early stopping within outer CV folds."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.task import TaskType


class BaseInnerValidStrategy(ABC):
    """Abstract strategy for producing a single inner train/valid split.

    The split is applied to the outer fold's training set to obtain
    a validation subset used for early stopping.

    Implementations must return relative indices (0-based within the
    outer fold's training data), or ``None`` to skip early stopping.
    """

    @abstractmethod
    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]] | None:
        """Produce one (inner_train_idx, inner_valid_idx) split.

        Args:
            n_samples: Number of samples in the outer fold's training set.
            y: Target values (optional, used for stratified splits).
            groups: Group labels (optional, used for group-aware splits).

        Returns:
            ``(inner_train_idx, inner_valid_idx)`` with 0-based positions
            within the outer fold's training set, or ``None`` to disable
            inner validation (no early stopping).
        """


class NoInnerValid(BaseInnerValidStrategy):
    """Disables inner validation — no early stopping is applied."""

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> None:
        return None


class HoldoutInnerValid(BaseInnerValidStrategy):
    """Random holdout split for inner validation.

    Args:
        ratio: Fraction of the outer fold's training set reserved for
            inner validation (early stopping).
        random_state: Random seed for reproducibility.
        stratify: If True, use stratified sampling to preserve class
            distribution (requires y to be provided).
    """

    def __init__(
        self,
        ratio: float = 0.1,
        random_state: int = 42,
        stratify: bool = False,
    ) -> None:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        self.ratio = ratio
        self.random_state = random_state
        self.stratify = stratify

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        if self.stratify and y is not None:
            from sklearn.model_selection import StratifiedShuffleSplit

            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.ratio,
                random_state=self.random_state,
            )
            indices = np.arange(n_samples)
            train_rel, valid_rel = next(sss.split(indices, y))
            return (
                np.sort(train_rel.astype(np.intp)),
                np.sort(valid_rel.astype(np.intp)),
            )
        rng = np.random.default_rng(self.random_state)
        n_valid = max(1, int(np.ceil(n_samples * self.ratio)))
        if n_valid >= n_samples:
            raise ValueError(
                f"Inner validation would consume all {n_samples} sample(s) "
                f"(n_valid={n_valid}, ratio={self.ratio}). "
                "Increase training data or decrease validation_ratio."
            )
        perm = rng.permutation(n_samples)
        valid_idx = np.sort(perm[:n_valid])
        train_idx = np.sort(perm[n_valid:])
        return train_idx, valid_idx


class GroupHoldoutInnerValid(BaseInnerValidStrategy):
    """Group-aware holdout: tail groups (by input order) go to validation.

    Groups in the validation set have NO overlap with training groups.
    The last ``ratio`` fraction of unique groups (in input order) are
    assigned to validation — no shuffle is applied.

    Args:
        ratio: Fraction of unique groups to assign to validation.
        random_state: Kept for signature compatibility (unused).
    """

    def __init__(self, ratio: float = 0.1, random_state: int = 42) -> None:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        self.ratio = ratio
        self.random_state = random_state

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        if groups is None:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    "GroupHoldoutInnerValid requires groups to be provided. "
                    "Set data.group_col in the config."
                ),
                context={
                    "n_samples": n_samples,
                    "strategy": "GroupHoldoutInnerValid",
                },
            )
        # Preserve input order of groups (np.unique sorts, so use dict.fromkeys)
        seen: dict[Any, None] = dict.fromkeys(groups.tolist())
        ordered_groups = list(seen.keys())
        n_valid_groups = max(1, int(len(ordered_groups) * self.ratio))
        valid_groups = ordered_groups[-n_valid_groups:]
        all_idx = np.arange(n_samples, dtype=np.intp)
        # Vectorised membership test (#116) — C implementation, ~5x+ faster
        # than the previous Python comprehension on large datasets.
        valid_mask = np.isin(groups, valid_groups)
        valid_idx = all_idx[valid_mask]
        train_idx = all_idx[~valid_mask]
        return train_idx, valid_idx


class TimeHoldoutInnerValid(BaseInnerValidStrategy):
    """Time-aware holdout: last ratio of rows go to validation.

    No shuffle is applied — assumes rows are in chronological order.

    Args:
        ratio: Fraction of rows to assign to validation (from the end).
    """

    def __init__(self, ratio: float = 0.1) -> None:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        self.ratio = ratio

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        n_valid = max(1, int(n_samples * self.ratio))
        if n_valid >= n_samples:
            raise ValueError(
                f"Inner validation would consume all {n_samples} sample(s) "
                f"(n_valid={n_valid}, ratio={self.ratio}). "
                "Increase training data or decrease validation_ratio."
            )
        all_idx = np.arange(n_samples, dtype=np.intp)
        train_idx = all_idx[:-n_valid]
        valid_idx = all_idx[-n_valid:]
        return train_idx, valid_idx


class StratifiedTimeHoldoutInnerValid(BaseInnerValidStrategy):
    """Per-class tail selection for inner validation (H-0060).

    Within each class, selects the last ``ratio`` fraction of rows for
    validation. Ensures every class has at least 1 row in inner valid
    while preserving time ordering within each class.

    Falls back to simple tail holdout when ``y`` is ``None``.

    Args:
        ratio: Fraction of each class to assign to validation.
    """

    def __init__(self, ratio: float = 0.1) -> None:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        self.ratio = ratio

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        if y is None:
            # Inline tail holdout (#124): avoids per-call construction of
            # TimeHoldoutInnerValid, which is hot in tuning workloads.
            n_valid = max(1, int(n_samples * self.ratio))
            if n_valid >= n_samples:
                raise ValueError(
                    f"Inner validation would consume all {n_samples} sample(s) "
                    f"(n_valid={n_valid}, ratio={self.ratio}). "
                    "Increase training data or decrease validation_ratio."
                )
            all_idx = np.arange(n_samples, dtype=np.intp)
            return all_idx[:-n_valid], all_idx[-n_valid:]

        valid_indices: list[int] = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            n_valid = max(1, int(len(cls_idx) * self.ratio))
            valid_indices.extend(cls_idx[-n_valid:].tolist())

        # Vectorised mask construction (#116) — replaces an O(n) Python
        # ``i not in valid_set`` filter with a single boolean array.
        valid_mask = np.zeros(n_samples, dtype=bool)
        valid_mask[np.asarray(valid_indices, dtype=np.intp)] = True
        all_idx = np.arange(n_samples, dtype=np.intp)
        valid_idx = all_idx[valid_mask]
        train_idx = all_idx[~valid_mask]
        return train_idx, valid_idx


class BlockedGroupInnerValid(BaseInnerValidStrategy):
    """Group-isolated, time-ordered, stratified inner valid (H-0060).

    For ``blocked_group_kfold``: selects tail groups (by last appearance)
    for inner validation, with per-class stratification for classification.

    Falls back to :class:`StratifiedTimeHoldoutInnerValid` when fewer than
    4 unique groups are available.

    .. note::

        Assumes ``groups`` is passed in temporal order (ascending by block
        value), as guaranteed by the ``_BLOCK_METHODS`` data preparation
        path in ``Model._prepare_training_data``.

    Args:
        ratio: Fraction of groups to assign to validation.
        task: ``"binary"`` / ``"multiclass"`` / ``"regression"``.
    """

    _MIN_GROUPS_FOR_ISOLATION = 4

    def __init__(self, ratio: float = 0.1, task: TaskType = "regression") -> None:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        self.ratio = ratio
        self.task = task

    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        if groups is None:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    "BlockedGroupInnerValid requires groups. "
                    "Set groups.col in the split config."
                ),
                context={
                    "n_samples": n_samples,
                    "strategy": "BlockedGroupInnerValid",
                    "task": self.task,
                },
            )

        # Preserve input order (= time order)
        seen: dict[Any, None] = dict.fromkeys(groups.tolist())
        ordered_groups = list(seen.keys())
        n_unique = len(ordered_groups)

        if n_unique < self._MIN_GROUPS_FOR_ISOLATION:
            warnings.warn(
                f"Too few groups ({n_unique}) for group-isolated inner "
                f"valid (need >= {self._MIN_GROUPS_FOR_ISOLATION}). "
                f"Falling back to StratifiedTimeHoldout.",
                UserWarning,
                stacklevel=2,
            )
            return StratifiedTimeHoldoutInnerValid(self.ratio).split(
                n_samples, y=y, groups=groups
            )

        # Compute last occurrence index per group (for time ordering)
        last_occurrence: dict[Any, int] = {}
        for i, g in enumerate(groups.tolist()):
            last_occurrence[g] = i

        # Sort groups by last occurrence (ascending = earliest first)
        sorted_groups = sorted(ordered_groups, key=lambda g: last_occurrence[g])

        if self.task in ("binary", "multiclass") and y is not None:
            valid_groups = self._stratified_tail_groups(sorted_groups, y, groups)
        else:
            n_valid = max(1, int(len(sorted_groups) * self.ratio))
            valid_groups = set(sorted_groups[-n_valid:])

        # Build index arrays via vectorised membership (#116).
        all_idx = np.arange(n_samples, dtype=np.intp)
        valid_mask = np.isin(groups, list(valid_groups))
        return all_idx[~valid_mask], all_idx[valid_mask]

    def _stratified_tail_groups(
        self,
        sorted_groups: list[Any],
        y: npt.NDArray[Any],
        groups: npt.NDArray[Any],
    ) -> set[Any]:
        """Select tail groups ensuring each class has >= 1 group."""
        # Majority label per group
        group_labels: dict[Any, Any] = {}
        for g in sorted_groups:
            mask = groups == g
            values, counts = np.unique(y[mask], return_counts=True)
            group_labels[g] = values[counts.argmax()]

        # Group by class
        classes = np.unique(y)
        per_class: dict[Any, list[Any]] = {c: [] for c in classes}
        for g in sorted_groups:
            per_class[group_labels[g]].append(g)

        # Take tail ratio groups per class (min 1)
        valid_groups: set[Any] = set()
        for cls in classes:
            cls_groups = per_class[cls]
            if not cls_groups:
                continue
            n_valid = max(1, int(len(cls_groups) * self.ratio))
            valid_groups.update(cls_groups[-n_valid:])

        return valid_groups
