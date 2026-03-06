"""Inner validation strategies for early stopping within outer CV folds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from lizyml.core.exceptions import ErrorCode, LizyMLError


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
        n_valid = max(1, int(n_samples * self.ratio))
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
                context={},
            )
        # Preserve input order of groups (np.unique sorts, so use dict.fromkeys)
        seen: dict[Any, None] = dict.fromkeys(groups.tolist())
        ordered_groups = list(seen.keys())
        n_valid_groups = max(1, int(len(ordered_groups) * self.ratio))
        valid_groups = set(ordered_groups[-n_valid_groups:])
        all_idx = np.arange(n_samples, dtype=np.intp)
        valid_mask = np.array([g in valid_groups for g in groups])
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
        all_idx = np.arange(n_samples, dtype=np.intp)
        train_idx = all_idx[:-n_valid]
        valid_idx = all_idx[-n_valid:]
        return train_idx, valid_idx
