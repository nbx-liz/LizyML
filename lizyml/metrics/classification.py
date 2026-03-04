"""Classification metrics: LogLoss, AUC-ROC, AUC-PR, F1, Accuracy, Brier, ECE."""

from __future__ import annotations

import numpy as np
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


def _require_1d_same_len(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
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
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
    def greater_is_better(self) -> bool:
        return True

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        _require_1d_same_len(y_true, y_pred, self.name)
        # Binarise if probabilities are provided
        pred = (y_pred >= 0.5).astype(int) if y_pred.dtype.kind == "f" else y_pred
        return float(f1_score(y_true, pred, zero_division=0))


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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        _require_1d_same_len(y_true, y_pred, self.name)
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        n = len(y_true)
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
            mask = (y_pred >= lo) & (y_pred < hi)
            if not mask.any():
                continue
            acc = float(np.mean((y_pred[mask] >= 0.5).astype(int) == y_true[mask]))
            conf = float(np.mean(y_pred[mask]))
            ece += (mask.sum() / n) * abs(acc - conf)
        return ece
