"""Binary classification threshold optimisation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def optimise_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    greater_is_better: bool = True,
    n_thresholds: int = 100,
) -> tuple[float, float]:
    """Find the decision threshold that optimises *metric_fn*.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        metric_fn: Function ``(y_true, y_pred_hard) -> float`` to optimise.
        greater_is_better: When ``True`` (default) the threshold that
            *maximises* ``metric_fn`` is selected; otherwise it is minimised.
        n_thresholds: Number of candidate thresholds to evaluate in ``[0, 1]``.

    Returns:
        ``(best_threshold, best_score)`` tuple.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_thresh = 0.5
    best_score = float("-inf") if greater_is_better else float("inf")

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = metric_fn(y_true, y_pred)
        if greater_is_better:
            if score > best_score:
                best_score = score
                best_thresh = float(t)
        elif score < best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh, best_score
