"""OOF (out-of-fold) prediction assembly helpers.

Assembly rules (leakage prevention):
- Each sample's OOF prediction is produced by a model that did NOT train
  on that sample's fold.
- Only the valid_idx portion of each fold's predictions is written into
  the OOF array; train_idx positions are never written by that fold's model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from lizyml.estimators.base import BaseEstimatorAdapter

TaskType = Literal["regression", "binary", "multiclass"]


def init_oof(
    n_samples: int,
    task: TaskType,
    n_classes: int | None = None,
) -> npt.NDArray[np.float64]:
    """Allocate an OOF prediction array filled with NaN.

    Args:
        n_samples: Total number of samples in the dataset.
        task: ML task type.
        n_classes: Number of classes for multiclass (required when
            ``task="multiclass"``).

    Returns:
        - Shape ``(n_samples,)`` for regression and binary.
        - Shape ``(n_samples, n_classes)`` for multiclass.
    """
    if task == "multiclass":
        if n_classes is None:
            raise ValueError("n_classes is required for multiclass OOF initialisation.")
        return np.full((n_samples, n_classes), np.nan)
    return np.full(n_samples, np.nan)


def fill_oof(
    oof: npt.NDArray[np.float64],
    valid_idx: npt.NDArray[np.intp],
    fold_pred: npt.NDArray[np.float64],
) -> None:
    """Write fold predictions into the OOF array at ``valid_idx`` positions.

    Args:
        oof: Pre-allocated OOF array (modified in-place).
        valid_idx: Row positions in the original dataset for this fold's
            validation set.
        fold_pred: Predictions produced by the fold model on the validation
            set. Must have ``len(fold_pred) == len(valid_idx)``.

    Raises:
        ValueError: When shapes are inconsistent.
    """
    if len(fold_pred) != len(valid_idx):
        raise ValueError(
            f"fold_pred length {len(fold_pred)} != valid_idx length {len(valid_idx)}."
        )
    oof[valid_idx] = fold_pred


def get_fold_pred(
    estimator: BaseEstimatorAdapter,
    X_valid: pd.DataFrame,
    task: TaskType,
) -> npt.NDArray[np.float64]:
    """Extract fold predictions from an estimator for the given task.

    Args:
        estimator: A fitted :class:`~lizyml.estimators.base.BaseEstimatorAdapter`.
        X_valid: Transformed validation features.
        task: ML task type.

    Returns:
        - ``(n_valid,)`` for regression (raw predictions).
        - ``(n_valid,)`` for binary (positive-class probability).
        - ``(n_valid, n_classes)`` for multiclass (class probabilities).
    """
    if task == "regression":
        pred: npt.NDArray[np.float64] = estimator.predict(X_valid)
        return pred
    # classification
    proba: npt.NDArray[np.float64] = estimator.predict_proba(X_valid)
    if task == "binary":
        return proba[:, 1]  # positive-class probability
    # multiclass: return full probability matrix
    return proba


def get_fold_raw(
    estimator: BaseEstimatorAdapter,
    X_valid: pd.DataFrame,
    task: TaskType,
) -> npt.NDArray[np.float64]:
    """Extract raw scores (logits) from an estimator for calibration.

    Args:
        estimator: A fitted :class:`~lizyml.estimators.base.BaseEstimatorAdapter`.
        X_valid: Transformed validation features.
        task: ML task type.

    Returns:
        - ``(n_valid,)`` for regression (raw predictions) and binary (logits).
        - ``(n_valid, n_classes)`` for multiclass (raw scores).
    """
    raw: npt.NDArray[np.float64] = estimator.predict_raw(X_valid)
    return raw
