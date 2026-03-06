"""Confusion matrix table computation (IS/OOS)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import confusion_matrix

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult


def confusion_matrix_table(
    fit_result: FitResult,
    y_true: npt.NDArray[Any],
    *,
    threshold: float = 0.5,
    task: str,
) -> dict[str, pd.DataFrame]:
    """Compute IS and OOS confusion matrices.

    Args:
        fit_result: Completed CV training output.
        y_true: Ground-truth target array.
        threshold: Decision boundary for binary classification.
        task: ``"binary"`` or ``"multiclass"``.

    Returns:
        ``{"is": DataFrame, "oos": DataFrame}`` with confusion matrix values.

    Raises:
        LizyMLError with UNSUPPORTED_TASK for regression.
    """
    if task == "regression":
        raise LizyMLError(
            code=ErrorCode.UNSUPPORTED_TASK,
            user_message="confusion_matrix() requires a binary or multiclass task.",
            context={"task": task},
        )

    y_arr = np.asarray(y_true)
    oof_pred = fit_result.oof_pred

    # OOS labels
    if task == "binary":
        oof_labels: npt.NDArray[Any] = (oof_pred >= threshold).astype(int)
    else:
        oof_labels = oof_pred.argmax(axis=1)

    cm_oos = confusion_matrix(y_arr, oof_labels)
    df_oos = pd.DataFrame(cm_oos)

    # IS: assemble all fold predictions and labels
    is_preds: list[npt.NDArray[Any]] = []
    is_y: list[npt.NDArray[Any]] = []
    for (train_idx, _), if_pred in zip(
        fit_result.splits.outer, fit_result.if_pred_per_fold, strict=True
    ):
        is_preds.append(if_pred)
        is_y.append(y_arr[train_idx])

    is_pred_all = np.concatenate(is_preds)
    is_y_all = np.concatenate(is_y)

    if task == "binary":
        is_labels: npt.NDArray[Any] = (is_pred_all >= threshold).astype(int)
    else:
        is_labels = is_pred_all.argmax(axis=1)

    cm_is = confusion_matrix(is_y_all, is_labels)
    df_is = pd.DataFrame(cm_is)

    return {"is": df_is, "oos": df_oos}
