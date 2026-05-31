"""Classification metrics.

LogLoss, AUC-ROC, AUC-PR, F1, Accuracy, Brier, ECE, Precision@K.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.registries import MetricRegistry
from lizyml.metrics.base import BaseMetric


def _require_1d_same_len(
    y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any], name: str
) -> None:
    if y_true.ndim != 1:
        raise LizyMLError(
            code=ErrorCode.UNSUPPORTED_METRIC,
            user_message=f"Metric '{name}' requires 1-D y_true.",
            context={"metric": name, "y_true_shape": y_true.shape},
        )
    if len(y_true) != len(y_pred):
        raise LizyMLError(
            code=ErrorCode.UNSUPPORTED_METRIC,
            user_message=(
                f"Metric '{name}' requires y_true and y_pred to have the same length. "
                f"Got {len(y_true)} vs {len(y_pred)}."
            ),
            context={"metric": name},
        )


def _multiclass_ovr_macro(
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
    name: str,
    per_class_fn: Callable[[npt.NDArray[Any], npt.NDArray[Any]], float],
    *,
    min_classes: int,
) -> float:
    """Macro-average a One-vs-Rest metric over classes present in ``y_true``.

    Per-fold CV evaluation can legitimately produce a validation fold whose
    ``y_true`` is missing one or more classes. Rather than letting the
    underlying sklearn call raise an opaque ``ValueError`` (AUC) or silently
    fold a degenerate zero-filled column into the mean (AUC-PR / Brier), the
    macro average is restricted to the classes that actually occur in
    ``y_true`` and at least ``min_classes`` of them are required. When every
    class is present (the non-degenerate case) the result is identical to
    averaging over all ``y_pred`` columns, so golden / calibrated numbers are
    unchanged.

    Raises:
        LizyMLError(UNSUPPORTED_METRIC): on sample-count mismatch, a label
            outside the ``y_pred`` column range, or fewer than ``min_classes``
            classes present in this fold.
    """
    if y_pred.shape[0] != len(y_true):
        raise LizyMLError(
            code=ErrorCode.UNSUPPORTED_METRIC,
            user_message=(
                f"Metric '{name}' requires y_true and y_pred to have the same "
                f"number of samples. Got {len(y_true)} vs {y_pred.shape[0]}."
            ),
            context={"metric": name},
        )
    present = np.unique(y_true).astype(np.intp)
    n_columns = int(y_pred.shape[1])
    if present.size and (int(present.min()) < 0 or int(present.max()) >= n_columns):
        raise LizyMLError(
            code=ErrorCode.UNSUPPORTED_METRIC,
            user_message=(
                f"Metric '{name}' received a y_true label outside the range of "
                f"y_pred columns (0..{n_columns - 1})."
            ),
            context={"metric": name, "n_columns": n_columns},
        )
    if len(present) < min_classes:
        raise LizyMLError(
            code=ErrorCode.UNSUPPORTED_METRIC,
            user_message=(
                f"Metric '{name}' on multiclass needs at least {min_classes} "
                f"class(es) present in this fold's y_true; found {len(present)}. "
                f"Evaluate on the full OOF, or aggregate folds that each cover "
                f"all classes."
            ),
            context={
                "metric": name,
                "n_present_classes": int(len(present)),
                "n_columns": n_columns,
            },
        )
    per_class = [per_class_fn((y_true == k).astype(int), y_pred[:, k]) for k in present]
    return float(np.mean(per_class))


@MetricRegistry.register("logloss")
class LogLoss(BaseMetric):
    """Binary cross-entropy (log loss)."""

    @property
    def name(self) -> str:
        return "logloss"

    @property
    def needs_proba(self) -> bool:
        return True

    @property
    def needs_simplex(self) -> bool:
        return True

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _require_1d_same_len(y_true, y_pred, self.name)
        return float(log_loss(y_true, y_pred))


@MetricRegistry.register("auc")
class AUC(BaseMetric):
    """Area Under the ROC Curve."""

    @property
    def name(self) -> str:
        return "auc"

    @property
    def needs_proba(self) -> bool:
        return True

    @property
    def needs_simplex(self) -> bool:
        return True

    @property
    def greater_is_better(self) -> bool:
        return True

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        if y_pred.ndim == 2:
            # Multiclass OvR: macro-average per-class ROC AUC over classes
            # present in y_true (needs >= 2 so each OvR split has pos + neg).
            return _multiclass_ovr_macro(
                y_true,
                y_pred,
                self.name,
                lambda yb, s: float(roc_auc_score(yb, s)),
                min_classes=2,
            )
        _require_1d_same_len(y_true, y_pred, self.name)
        return float(roc_auc_score(y_true, y_pred))


@MetricRegistry.register("auc_pr")
class AUCPR(BaseMetric):
    """Area Under the Precision-Recall Curve."""

    @property
    def name(self) -> str:
        return "auc_pr"

    @property
    def needs_proba(self) -> bool:
        return True

    @property
    def greater_is_better(self) -> bool:
        return True

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        if y_pred.ndim == 2:
            # Multiclass OvR: macro-average per-class average precision over
            # classes present in y_true (each present class has >= 1 positive).
            return _multiclass_ovr_macro(
                y_true,
                y_pred,
                self.name,
                lambda yb, s: float(average_precision_score(yb, s)),
                min_classes=1,
            )
        _require_1d_same_len(y_true, y_pred, self.name)
        return float(average_precision_score(y_true, y_pred))


@MetricRegistry.register("f1")
class F1(BaseMetric):
    """Binary F1 score (threshold = 0.5 for probabilities)."""

    @property
    def name(self) -> str:
        return "f1"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return True

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _require_1d_same_len(y_true, y_pred, self.name)
        # Binarise if probabilities are provided
        pred = (y_pred >= 0.5).astype(int) if y_pred.dtype.kind == "f" else y_pred
        average = "binary" if len(np.unique(y_true)) == 2 else "macro"
        return float(f1_score(y_true, pred, zero_division=0, average=average))


@MetricRegistry.register("accuracy")
class Accuracy(BaseMetric):
    """Classification accuracy."""

    @property
    def name(self) -> str:
        return "accuracy"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return True

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _require_1d_same_len(y_true, y_pred, self.name)
        pred = (y_pred >= 0.5).astype(int) if y_pred.dtype.kind == "f" else y_pred
        return float(accuracy_score(y_true, pred))


@MetricRegistry.register("brier")
class Brier(BaseMetric):
    """Brier Score (mean squared error for probabilities)."""

    @property
    def name(self) -> str:
        return "brier"

    @property
    def needs_proba(self) -> bool:
        return True

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        if y_pred.ndim == 2:
            # Multiclass OvR: macro-average per-class Brier over classes
            # present in y_true.
            return _multiclass_ovr_macro(
                y_true,
                y_pred,
                self.name,
                lambda yb, s: float(brier_score_loss(yb, s)),
                min_classes=1,
            )
        _require_1d_same_len(y_true, y_pred, self.name)
        return float(brier_score_loss(y_true, y_pred))


@MetricRegistry.register("ece")
class ECE(BaseMetric):
    """Expected Calibration Error (equal-width bins, M=10)."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "ece"

    @property
    def needs_proba(self) -> bool:
        return True

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _require_1d_same_len(y_true, y_pred, self.name)
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        n = len(y_true)
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
            # Last bin is right-inclusive to capture y_pred == 1.0
            mask = (y_pred >= lo) & (y_pred <= hi if hi == 1.0 else y_pred < hi)
            if not mask.any():
                continue
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_pred[mask]))
            ece += (mask.sum() / n) * abs(acc - conf)
        return ece


@MetricRegistry.register("precision_at_k")
class PrecisionAtK(BaseMetric):
    """Precision at top-K percent of predicted probabilities.

    Args:
        k: Top-K percentage cutoff (default 10 = top 10%).
    """

    def __init__(self, k: int = 10) -> None:
        if not 1 <= k <= 100:
            raise ValueError(f"k must be in [1, 100], got {k}")
        self.k = k

    @property
    def name(self) -> str:
        return "precision_at_k"

    @property
    def needs_proba(self) -> bool:
        return True

    @property
    def greater_is_better(self) -> bool:
        return True

    def __call__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
        _require_1d_same_len(y_true, y_pred, self.name)
        n = len(y_true)
        n_top = max(1, int(n * self.k / 100))
        top_idx: npt.NDArray[np.intp] = np.argsort(y_pred)[::-1][:n_top].astype(np.intp)
        return float(np.mean(y_true[top_idx]))
