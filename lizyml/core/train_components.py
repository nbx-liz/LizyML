"""TrainComponents — shared state for CVTrainer and RefitTrainer (H-0050).

A single ``_build_train_components()`` call in ``Model`` produces this
dataclass, which is then passed to **both** CVTrainer and RefitTrainer.
This guarantees that the two trainers always receive identical parameter
resolution, factory closures, and inner-validation strategies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.training.inner_valid import BaseInnerValidStrategy


@dataclass(frozen=True)
class TrainComponents:
    """Shared training components for CVTrainer and RefitTrainer.

    Attributes:
        estimator_factory: Callable returning a fresh estimator instance.
        sample_weight: Optional per-sample weights (e.g. balanced multiclass).
        ratio_resolver: Optional callable ``(n_rows) -> dict`` for per-fold
            ratio-dependent params like ``min_data_in_leaf``.
        inner_valid: Inner validation strategy for early stopping.
    """

    estimator_factory: Callable[[], BaseEstimatorAdapter]
    sample_weight: npt.NDArray[np.float64] | None
    ratio_resolver: Callable[[int], dict[str, Any]] | None
    inner_valid: BaseInnerValidStrategy
