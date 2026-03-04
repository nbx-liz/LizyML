"""Inner validation strategies for early stopping within outer CV folds."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


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
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
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
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> None:
        return None


class HoldoutInnerValid(BaseInnerValidStrategy):
    """Random holdout split for inner validation.

    Args:
        ratio: Fraction of the outer fold's training set reserved for
            inner validation (early stopping).
        random_state: Random seed for reproducibility.
    """

    def __init__(self, ratio: float = 0.1, random_state: int = 42) -> None:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        self.ratio = ratio
        self.random_state = random_state

    def split(
        self,
        n_samples: int,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        n_valid = max(1, int(n_samples * self.ratio))
        perm = rng.permutation(n_samples)
        valid_idx = np.sort(perm[:n_valid])
        train_idx = np.sort(perm[n_valid:])
        return train_idx, valid_idx
