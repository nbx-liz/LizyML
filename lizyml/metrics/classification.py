"""Classification metrics.

LogLoss, AUC-ROC, AUC-PR, F1, Accuracy, Brier, ECE, Precision@K.
"""

from __future__ import annotations

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
            # Multiclass OvR: y_pred is (n_samples, n_classes)
            if y_pred.shape[0] != len(y_true):
                raise LizyMLError(
                    code=ErrorCode.UNSUPPORTED_METRIC,
                    user_message=(
                        f"Metric '{self.name}' requires y_true and y_pred to have "
                        f"the same number of samples. "
                        f"Got {len(y_true)} vs {y_pred.shape[0]}."
                    ),
                    context={"metric": self.name},
                )
            return float(
                roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")
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
            # Multiclass OvR: per-class average_precision_score, macro average
            if y_pred.shape[0] != len(y_true):
                raise LizyMLError(
                    code=ErrorCode.UNSUPPORTED_METRIC,
                    user_message=(
                        f"Metric '{self.name}' requires y_true and y_pred to have "
                        f"the same number of samples. "
                        f"Got {len(y_true)} vs {y_pred.shape[0]}."
                    ),
                    context={"metric": self.name},
                )
            from sklearn.preprocessing import label_binarize

            classes = np.arange(y_pred.shape[1])
            y_bin = label_binarize(y_true, classes=classes)
            per_class = [
                float(average_precision_score(y_bin[:, k], y_pred[:, k]))
                for k in range(y_pred.shape[1])
            ]
            return float(np.mean(per_class))
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
            # Multiclass OvR: per-class brier_score_loss, macro average
            if y_pred.shape[0] != len(y_true):
                raise LizyMLError(
                    code=ErrorCode.UNSUPPORTED_METRIC,
                    user_message=(
                        f"Metric '{self.name}' requires y_true and y_pred to have "
                        f"the same number of samples. "
                        f"Got {len(y_true)} vs {y_pred.shape[0]}."
                    ),
                    context={"metric": self.name},
                )
            from sklearn.preprocessing import label_binarize

            classes = np.arange(y_pred.shape[1])
            y_bin = label_binarize(y_true, classes=classes)
            per_class = [
                float(brier_score_loss(y_bin[:, k], y_pred[:, k]))
                for k in range(y_pred.shape[1])
            ]
            return float(np.mean(per_class))
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
            acc = float(np.mean((y_pred[mask] >= 0.5).astype(int) == y_true[mask]))
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
