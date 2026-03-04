"""BaseSplitter — abstract interface for all CV splitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt


class BaseSplitter(ABC):
    """Abstract base for all splitters.

    Contract:
    - ``split`` receives only integer counts / arrays of labels or group labels.
    - Splitters MUST NOT accept or modify DataFrames.
    - Splitters MUST NOT shuffle in place or produce side effects.
    - The same seed must produce the same index sequences across calls.
    """

    @abstractmethod
    def split(
        self,
        n_samples: int,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        """Yield ``(train_indices, valid_indices)`` pairs.

        Args:
            n_samples: Total number of samples.
            y: Target array used by stratified splitters (ignored otherwise).
            groups: Group label array used by group-aware splitters.

        Yields:
            Tuple of 1-D integer arrays ``(train_idx, valid_idx)``.
        """
