"""The shape of a refit's result, as a Layer 2 interface (H-0130).

``RefitResult`` stays in ``training/refit_trainer.py`` so pickled artifacts keep
naming ``lizyml.training.refit_trainer.RefitResult``. Layer 3 (``persistence/``)
annotates with this Protocol instead of importing the trainer module
(BLUEPRINT §2.1). Its fields mirror ``RefitResult``'s field for field, name and
type; ``tests/test_architecture/test_h0130_contracts.py`` compares the two.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from lizyml.estimators.base import BaseEstimatorAdapter

__all__ = ["RefitResultView"]


class RefitResultView(Protocol):
    """What a refit returns: see ``training.refit_trainer.RefitResult``."""

    model: BaseEstimatorAdapter
    pipeline_state: Any
    feature_names: list[str]
    categorical_features: list[str]
    best_iteration: int | None
    train_pred: npt.NDArray[np.float64]
    history: dict[str, Any]
