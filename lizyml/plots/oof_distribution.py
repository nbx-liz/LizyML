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

_plotly: Any = None
try:
    import plotly.graph_objects as go

    _plotly = go
except ImportError:  # pragma: no cover
    pass


def plot_oof_distribution(fit_result: FitResult) -> Any:
    """Plot the distribution of out-of-fold predictions.

    For regression and binary tasks, plots a histogram of ``oof_pred``.
    For multiclass, plots overlaid histograms of class probabilities.

    Args:
        fit_result: A fitted :class:`~lizyml.core.types.fit_result.FitResult`.

    Returns:
        A ``plotly.graph_objects.Figure`` object.

    Raises:
        LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
    """
    if _plotly is None:
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                "plotly is required for plots. "
                "Install with: pip install 'lizyml[plots]'"
            ),
            context={"package": "plotly"},
        )

    oof = fit_result.oof_pred
    fig = _plotly.Figure()

    if oof.ndim == 2:
        # Multiclass: (n_samples, n_classes)
        n_classes = oof.shape[1]
        for cls_idx in range(n_classes):
            fig.add_trace(
                _plotly.Histogram(
                    x=oof[:, cls_idx],
                    name=f"Class {cls_idx}",
                    opacity=0.5,
                    nbinsx=30,
                )
            )
        fig.update_layout(
            title="OOF Prediction Distribution (Multiclass)",
            xaxis_title="Predicted probability",
            yaxis_title="Count",
            barmode="overlay",
        )
    else:
        # Regression or binary: (n_samples,)
        has_proba = bool(np.all((oof >= 0) & (oof <= 1)))
        xlabel = "Predicted probability" if has_proba else "Predicted value"
        fig.add_trace(_plotly.Histogram(x=oof, nbinsx=30))
        fig.update_layout(
            title="OOF Prediction Distribution",
            xaxis_title=xlabel,
            yaxis_title="Count",
        )

    return fig
