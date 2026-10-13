"""Learning curve plot.

Visualises per-fold training history from ``FitResult.history``.
Requires early stopping to be enabled (otherwise eval history is empty).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.plots._deps import require_plotly
from lizyml.plots._theme import apply_default_layout

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult

#: The learning-curve metrics whose LightGBM definition differs from the
#: LizyML metric of the same name (H-0125 policy 7, #354), with LightGBM's
#: definition. ``FitResult.history`` records LightGBM's own evaluation, so the
#: values of these names differ from ``FitResult.metrics`` under the same name,
#: and their subplots are titled with this definition. ``huber``'s threshold is
#: LightGBM's ``alpha`` (LizyML: ``delta``); its value is not in ``FitResult``,
#: so it is not written. ``mape`` is a ratio over ``max(1, |y|)`` (LizyML: a
#: percentage over ``|y|``). The keys are pinned to the measured differences by
#: ``tests/test_plots/test_learning_curve_definitions.py`` and to the documents
#: by ``tests/test_docs/test_h0125_docs.py``.
LIGHTGBM_CURVE_DEFINITIONS: Final[dict[str, str]] = {
    "huber": "threshold = alpha",
    "mape": "|y-p| / max(1,|y|)",
}


def _subplot_title(key: str) -> str:
    """``dataset/metric``, with LightGBM's definition where it differs (H-0125)."""
    definition = LIGHTGBM_CURVE_DEFINITIONS.get(key.split("/", 1)[-1])
    if definition is None:
        return key
    return f"{key} (LightGBM: {definition})"


_plotly: Any = None
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots as _make_subplots

    _plotly = go
except ImportError:  # pragma: no cover
    _make_subplots = None


def plot_learning_curve(
    fit_result: FitResult,
    *,
    metrics: list[str] | None = None,
) -> Any:
    """Plot per-fold training/validation loss vs iteration.

    The history is LightGBM's evaluation, under LightGBM's metric names and
    definitions. A subplot whose metric is a key of
    :data:`LIGHTGBM_CURVE_DEFINITIONS` (``huber``, ``mape``) is titled with
    LightGBM's definition, e.g. ``valid_0/huber (LightGBM: threshold = alpha)``,
    because its values differ from ``FitResult.metrics`` under the same name
    (H-0125). The ``metrics`` filter matches the bare name; the legend and the
    axes are unchanged.

    Args:
        fit_result: A fitted :class:`~lizyml.core.types.fit_result.FitResult`.
        metrics: Optional list of metric names to display. Matches against
            the metric part of eval history keys (after the ``/`` in
            ``dataset/metric``). ``None`` plots all metrics (H-0062).

    Returns:
        A ``plotly.graph_objects.Figure`` object.

    Raises:
        LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
        LizyMLError with ``MODEL_NOT_FIT`` when no history is available or the
            history contains no evaluation metrics (e.g. early stopping disabled).
        LizyMLError with ``CONFIG_INVALID`` when *metrics* filter matches no
            keys in the eval history.
    """
    require_plotly(_plotly, feature="learning curve plots")

    if not fit_result.history:
        raise LizyMLError(
            code=ErrorCode.MODEL_NOT_FIT,
            user_message="No training history found in FitResult.",
            context={"plot": "learning_curve", "n_history": 0},
        )

    # Gather eval_history entries; skip folds with empty history
    fold_histories: list[dict[str, list[float]]] = []
    for fold_hist in fit_result.history:
        eval_hist = fold_hist.get("eval_history", {})
        if not eval_hist:
            continue
        # eval_hist is {dataset_name: {metric_name: [values]}}
        flat: dict[str, list[float]] = {}
        for ds_name, ds_metrics in eval_hist.items():
            for metric_name, values in ds_metrics.items():
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
            context={
                "plot": "learning_curve",
                "n_folds": len(fit_result.history),
            },
        )

    # Use first fold's keys as reference
    all_metric_keys = list(fold_histories[0].keys())

    if metrics is not None:
        metric_keys = [k for k in all_metric_keys if k.split("/", 1)[-1] in metrics]
        if not metric_keys:
            available = sorted({k.split("/", 1)[-1] for k in all_metric_keys})
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    f"No matching metrics found for {metrics}. "
                    f"Available metrics: {available}"
                ),
                context={
                    "requested_metrics": metrics,
                    "available_metrics": available,
                },
            )
    else:
        metric_keys = all_metric_keys

    n_metrics = len(metric_keys)

    fig = _make_subplots(
        rows=1,
        cols=n_metrics,
        subplot_titles=[_subplot_title(key) for key in metric_keys],
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

    apply_default_layout(
        fig,
        title="Learning Curve",
        height=400,
        width=500 * n_metrics,
    )
    return fig
