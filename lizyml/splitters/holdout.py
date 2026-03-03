"""HoldoutSplitter — single random train/valid split."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .base import BaseSplitter


class HoldoutSplitter(BaseSplitter):
    """Single random holdout split (no cross-validation loop).

    Yields exactly one ``(train_idx, valid_idx)`` pair.

    Args:
        ratio: Fraction of samples assigned to the validation set.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        ratio: float = 0.1,
        random_state: int | None = 42,
    ) -> None:
        if not (0.0 < ratio < 1.0):
            raise ValueError(f"ratio must be in (0, 1), got {ratio}.")
        self.ratio = ratio
        self.random_state = random_state

    def split(
        self,
        n_samples: int,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        n_valid = max(1, int(np.ceil(n_samples * self.ratio)))
        valid_idx = np.sort(indices[:n_valid])
        train_idx = np.sort(indices[n_valid:])
        yield train_idx, valid_idx
