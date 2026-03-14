"""Residual analysis plots (regression only).

Supports four display modes via the ``kind`` argument:

- ``"scatter"``: Actual vs Predicted, IS and OOS overlaid.
- ``"histogram"``: residual distribution, IS and OOS overlaid.
- ``"qq"``: QQ plot of OOS residuals.
- ``"all"`` (default): all three in a single figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult

_plotly: Any = None
_make_subplots: Any = None
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _plotly = go
    _make_subplots = make_subplots
except ImportError:  # pragma: no cover
    pass

_scipy_stats: Any = None
try:
    from scipy import stats as _scipy_stats_mod

    _scipy_stats = _scipy_stats_mod
except ImportError:  # pragma: no cover
    pass

_VALID_KINDS = ("scatter", "histogram", "qq", "all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_is_data(
    fit_result: FitResult,
    y_true: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(is_pred_all, is_actual_all)`` from per-fold IF data."""
    is_preds: list[npt.NDArray[np.float64]] = []
    is_actuals: list[npt.NDArray[np.float64]] = []
    for (train_idx, _), if_pred in zip(
        fit_result.splits.outer, fit_result.if_pred_per_fold, strict=True
    ):
        y_train = y_true[train_idx]
        is_preds.append(if_pred.astype(np.float64))
        is_actuals.append(y_train.astype(np.float64))
    if is_preds:
        return np.concatenate(is_preds), np.concatenate(is_actuals)
    empty: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
    return empty, empty


def _downsample_is(
    arr1: npt.NDArray[np.float64],
    arr2: npt.NDArray[np.float64],
    n_oos: int,
    *,
    seed: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Downsample IS arrays to ``n_oos`` when IS > OOS (reproducible, seed=0)."""
    if len(arr1) <= n_oos:
        return arr1, arr2
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr1), size=n_oos, replace=False)
    return arr1[idx], arr2[idx]


def _add_scatter_traces(
    fig: Any,
    oof_pred: npt.NDArray[np.float64],
    oos_actual: npt.NDArray[np.float64],
    is_pred: npt.NDArray[np.float64],
    is_actual: npt.NDArray[np.float64],
    *,
    row: int | None = None,
    col: int | None = None,
    show_legend: bool = True,
) -> None:
    kw: dict[str, Any] = {}
    if row is not None:
        kw["row"] = row
        kw["col"] = col
    ax_kw: dict[str, Any] = {"row": row, "col": col} if row is not None else {}

    fig.add_trace(
        _plotly.Scatter(
            x=oof_pred,
            y=oos_actual,
            mode="markers",
            marker=dict(size=3, opacity=0.3, color="steelblue"),
            name="OOS",
            showlegend=show_legend,
        ),
        **kw,
    )
    if len(is_pred) > 0:
        fig.add_trace(
            _plotly.Scatter(
                x=is_pred,
                y=is_actual,
                mode="markers",
                marker=dict(size=3, opacity=0.15, color="darkorange"),
                name="IS",
                showlegend=show_legend,
            ),
            **kw,
        )
    # y=x perfect-prediction reference line
    all_x = np.concatenate([oof_pred, is_pred]) if len(is_pred) > 0 else oof_pred
    all_y = (
        np.concatenate([oos_actual, is_actual]) if len(is_actual) > 0 else oos_actual
    )
    ref_min = float(min(all_x.min(), all_y.min()))
    ref_max = float(max(all_x.max(), all_y.max()))
    fig.add_trace(
        _plotly.Scatter(
            x=[ref_min, ref_max],
            y=[ref_min, ref_max],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect",
            showlegend=False,
        ),
        **kw,
    )
    fig.update_xaxes(title_text="Predicted", **ax_kw)
    fig.update_yaxes(title_text="Actual", **ax_kw)


def _add_histogram_traces(
    fig: Any,
    oos_resid: npt.NDArray[np.float64],
    is_resid: npt.NDArray[np.float64],
    *,
    row: int | None = None,
    col: int | None = None,
    show_legend: bool = True,
    show_annotation: bool = True,
) -> None:
    kw: dict[str, Any] = {}
    if row is not None:
        kw["row"] = row
        kw["col"] = col
    ax_kw: dict[str, Any] = {"row": row, "col": col} if row is not None else {}

    fig.add_trace(
        _plotly.Histogram(
            x=oos_resid,
            nbinsx=30,
            opacity=0.6,
            name="OOS",
            marker_color="steelblue",
            showlegend=show_legend,
        ),
        **kw,
    )
    if len(is_resid) > 0:
        fig.add_trace(
            _plotly.Histogram(
                x=is_resid,
                nbinsx=30,
                opacity=0.4,
                name="IS",
                marker_color="darkorange",
                showlegend=show_legend,
            ),
            **kw,
        )
    fig.update_xaxes(title_text="Residual", **ax_kw)
    fig.update_yaxes(title_text="Count", **ax_kw)

    if show_annotation:
        mean_val = float(np.mean(oos_resid))
        std_val = float(np.std(oos_resid))
        fig.add_annotation(
            text=f"OOS mean={mean_val:.4f}<br>OOS std={std_val:.4f}",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.95,
            xanchor="right",
            showarrow=False,
            font=dict(size=10),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
        )


def _add_qq_traces(
    fig: Any,
    oos_resid: npt.NDArray[np.float64],
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    if _scipy_stats is None:
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                "scipy is required for QQ plots. Install with: pip install scipy"
            ),
            context={"package": "scipy"},
        )

    kw: dict[str, Any] = {}
    if row is not None:
        kw["row"] = row
        kw["col"] = col
    ax_kw: dict[str, Any] = {"row": row, "col": col} if row is not None else {}

    (osm, osr), _ = _scipy_stats.probplot(oos_resid, dist="norm")
    fig.add_trace(
        _plotly.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker=dict(size=4),
            name="QQ",
            showlegend=False,
        ),
        **kw,
    )
    line_min, line_max = float(min(osm)), float(max(osm))
    fig.add_trace(
        _plotly.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Reference",
            showlegend=False,
        ),
        **kw,
    )
    fig.update_xaxes(title_text="Theoretical Quantiles", **ax_kw)
    fig.update_yaxes(title_text="Sample Quantiles", **ax_kw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_residuals(
    fit_result: FitResult,
    y_true: npt.NDArray[np.float64],
    *,
    kind: str = "all",
) -> Any:
    """Plot residual analysis for regression.

    Args:
        fit_result: A fitted :class:`~lizyml.core.types.fit_result.FitResult`.
        y_true: Target values used during fitting.
        kind: Which plot to render.
            ``"scatter"`` — Actual vs Predicted (IS + OOS overlay, y=x reference).
            ``"histogram"`` — residual distribution (IS + OOS overlay).
            ``"qq"`` — QQ plot of OOS residuals.
            ``"all"`` — all three panels in one figure (default).

    Returns:
        A ``plotly.graph_objects.Figure`` object.

    Raises:
        LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
        LizyMLError with ``CONFIG_INVALID`` for an unknown ``kind`` value.
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

    if kind not in _VALID_KINDS:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"Unknown kind={kind!r} for residuals_plot(). "
                f"Valid options: {_VALID_KINDS}."
            ),
            context={"kind": kind},
        )

    y_arr = np.asarray(y_true, dtype=np.float64)
    oof_pred = fit_result.oof_pred.astype(np.float64)
    oos_resid: npt.NDArray[np.float64] = y_arr - oof_pred

    is_pred, is_actual = _build_is_data(fit_result, y_arr)
    is_resid: npt.NDArray[np.float64] = (
        is_actual - is_pred if len(is_pred) > 0 else is_actual
    )

    # Downsample IS to OOS count for scatter and histogram (H-0012)
    is_pred_ds, is_actual_ds = _downsample_is(is_pred, is_actual, len(oof_pred))
    _, is_resid_ds = _downsample_is(is_pred, is_resid, len(oos_resid))

    if kind == "scatter":
        fig = _plotly.Figure()
        _add_scatter_traces(fig, oof_pred, y_arr, is_pred_ds, is_actual_ds)
        fig.update_layout(
            title="Actual vs Predicted (IS vs OOS)", height=450, width=700
        )
        return fig

    if kind == "histogram":
        fig = _plotly.Figure()
        _add_histogram_traces(fig, oos_resid, is_resid_ds, show_annotation=True)
        fig.update_layout(
            title="Residual Distribution (IS vs OOS)",
            barmode="overlay",
            height=400,
            width=600,
        )
        return fig

    if kind == "qq":
        fig = _plotly.Figure()
        _add_qq_traces(fig, oos_resid)
        fig.update_layout(title="QQ Plot (OOS)", height=450, width=500)
        return fig

    # kind == "all"
    fig = _make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Actual vs Predicted",
            "Residual Distribution",
            "QQ Plot",
        ],
    )
    _add_scatter_traces(
        fig,
        oof_pred,
        y_arr,
        is_pred_ds,
        is_actual_ds,
        row=1,
        col=1,
        show_legend=True,
    )
    _add_histogram_traces(
        fig,
        oos_resid,
        is_resid_ds,
        row=1,
        col=2,
        show_legend=False,
        show_annotation=False,
    )
    _add_qq_traces(fig, oos_resid, row=1, col=3)
    fig.update_layout(
        title="Residual Analysis",
        barmode="overlay",
        height=400,
        width=1200,
    )
    return fig
