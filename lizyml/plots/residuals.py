"""Residual distribution plot (regression only).

Two-panel layout: histogram of residuals + QQ plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from lizyml.core.exceptions import ErrorCode, LizyMLError

_plotly: Any = None
_make_subplots: Any = None
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _plotly = go
    _make_subplots = make_subplots
except ImportError:  # pragma: no cover
    pass


def plot_residuals(residuals: npt.NDArray[np.float64]) -> Any:
    """Plot residual distribution as histogram + QQ plot.

    Args:
        residuals: 1-D array of residuals (y_true - y_pred).

    Returns:
        A ``plotly.graph_objects.Figure`` with two subplots.

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

    from scipy import stats  # type: ignore[import-untyped]  # noqa: I001

    fig = _make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Residual Distribution", "QQ Plot"],
    )

    # --- Left panel: histogram ---
    mean_val = float(np.mean(residuals))
    std_val = float(np.std(residuals))

    fig.add_trace(
        _plotly.Histogram(
            x=residuals,
            nbinsx=30,
            name="Residuals",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Residual", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    # Add mean/std annotation
    fig.add_annotation(
        text=f"mean={mean_val:.4f}<br>std={std_val:.4f}",
        xref="x",
        yref="y domain",
        x=mean_val,
        y=0.95,
        showarrow=False,
        font=dict(size=11),
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
    )

    # --- Right panel: QQ plot ---
    (osm, osr), _ = stats.probplot(residuals, dist="norm")

    fig.add_trace(
        _plotly.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker=dict(size=4),
            name="QQ",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Reference line (45-degree)
    line_min = float(min(osm))
    line_max = float(max(osm))
    fig.add_trace(
        _plotly.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Reference",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    fig.update_layout(title="Residual Analysis", height=400, width=900)
    return fig
