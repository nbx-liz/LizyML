"""OOF prediction distribution plot.

Visualises the distribution of out-of-fold predictions from ``FitResult``.
Works for all task types without requiring ``y_true``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult

_mpl: Any = None
try:
    import matplotlib.pyplot as plt

    _mpl = plt
except ImportError:  # pragma: no cover
    pass


def plot_oof_distribution(fit_result: FitResult) -> Any:
    """Plot the distribution of out-of-fold predictions.

    For regression and binary tasks, plots a histogram of ``oof_pred``.
    For multiclass, plots a stacked histogram of class probabilities.

    Args:
        fit_result: A fitted :class:`~lizyml.core.types.fit_result.FitResult`.

    Returns:
        A ``matplotlib.figure.Figure`` object.

    Raises:
        LizyMLError with ``OPTIONAL_DEP_MISSING`` when matplotlib is not installed.
    """
    if _mpl is None:
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                "matplotlib is required for plots. "
                "Install with: pip install 'lizyml[plots]'"
            ),
            context={"package": "matplotlib"},
        )

    oof = fit_result.oof_pred

    if oof.ndim == 2:
        # Multiclass: (n_samples, n_classes)
        n_classes = oof.shape[1]
        fig, ax = _mpl.subplots(figsize=(8, 4))
        for cls_idx in range(n_classes):
            ax.hist(
                oof[:, cls_idx],
                bins=30,
                alpha=0.5,
                label=f"Class {cls_idx}",
            )
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.set_title("OOF Prediction Distribution (Multiclass)")
        ax.legend()
    else:
        # Regression or binary: (n_samples,)
        fig, ax = _mpl.subplots(figsize=(8, 4))
        ax.hist(oof, bins=30)
        has_proba = bool(np.all((oof >= 0) & (oof <= 1)))
        xlabel = "Predicted probability" if has_proba else "Predicted value"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title("OOF Prediction Distribution")

    fig.tight_layout()
    return fig
