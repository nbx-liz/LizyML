"""Learning curve plot.

Visualises per-fold training history from ``FitResult.history``.
Requires early stopping to be enabled (otherwise eval history is empty).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult

_plotly: Any = None
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots as _make_subplots

    _plotly = go
except ImportError:  # pragma: no cover
    _make_subplots = None


def plot_learning_curve(fit_result: FitResult) -> Any:
    """Plot per-fold training/validation loss vs iteration.

    Args:
        fit_result: A fitted :class:`~lizyml.core.types.fit_result.FitResult`.

    Returns:
        A ``plotly.graph_objects.Figure`` object.

    Raises:
        LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
        LizyMLError with ``MODEL_NOT_FIT`` when no history is available or the
            history contains no evaluation metrics (e.g. early stopping disabled).
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

    if not fit_result.history:
        raise LizyMLError(
            code=ErrorCode.MODEL_NOT_FIT,
            user_message="No training history found in FitResult.",
            context={},
        )

    # Gather eval_history entries; skip folds with empty history
    fold_histories: list[dict[str, list[float]]] = []
    for fold_hist in fit_result.history:
        eval_hist = fold_hist.get("eval_history", {})
        if not eval_hist:
            continue
        # eval_hist is {dataset_name: {metric_name: [values]}}
        flat: dict[str, list[float]] = {}
        for ds_name, metrics in eval_hist.items():
            for metric_name, values in metrics.items():
                key = f"{ds_name}/{metric_name}"
                flat[key] = list(values)
        if flat:
            fold_histories.append(flat)

    if not fold_histories:
        raise LizyMLError(
            code=ErrorCode.MODEL_NOT_FIT,
            user_message=(
                "No evaluation history found. "
                "Enable early stopping to record validation metrics per fold."
            ),
            context={},
        )

    # Use first fold's keys as reference
    metric_keys = list(fold_histories[0].keys())
    n_metrics = len(metric_keys)

    fig = _make_subplots(
        rows=1,
        cols=n_metrics,
        subplot_titles=metric_keys,
    )

    for col, key in enumerate(metric_keys):
        for fold_idx, fh in enumerate(fold_histories):
            if key in fh:
                fig.add_trace(
                    _plotly.Scatter(
                        y=fh[key],
                        mode="lines",
                        name=f"fold {fold_idx}",
                        opacity=0.7,
                        showlegend=(col == 0),
                    ),
                    row=1,
                    col=col + 1,
                )
        fig.update_xaxes(title_text="Iteration", row=1, col=col + 1)
        fig.update_yaxes(title_text="Loss", row=1, col=col + 1)

    fig.update_layout(
        title="Learning Curve",
        height=400,
        width=500 * n_metrics,
    )
    return fig
