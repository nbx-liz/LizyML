"""Feature importance plot.

Uses the fold-averaged importance scores stored in ``FitResult.models``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult

_plotly: Any = None
try:
    import plotly.graph_objects as go

    _plotly = go
except ImportError:  # pragma: no cover
    pass


def _check_plotly() -> None:
    if _plotly is None:
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                "plotly is required for plots. "
                "Install with: pip install 'lizyml[plots]'"
            ),
            context={"package": "plotly"},
        )


def _render_bar_chart(
    sorted_items: list[tuple[str, float]],
    title: str,
) -> Any:
    """Render a horizontal bar chart from (feature, value) pairs."""
    _check_plotly()
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig = _plotly.Figure(
        _plotly.Bar(
            x=values[::-1],
            y=features[::-1],
            orientation="h",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, len(features) * 25),
        margin=dict(l=150),
    )
    return fig


def plot_importance(
    fit_result: FitResult,
    *,
    kind: str = "split",
    top_n: int | None = 20,
) -> Any:
    """Plot fold-averaged feature importances as a horizontal bar chart.

    Args:
        fit_result: A fitted :class:`~lizyml.core.types.fit_result.FitResult`.
        kind: ``"split"`` or ``"gain"``.
        top_n: Maximum number of features to show.  ``None`` shows all.

    Returns:
        A ``plotly.graph_objects.Figure`` object.

    Raises:
        LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
        LizyMLError with ``MODEL_NOT_FIT`` when no fold models are available.
    """
    _check_plotly()

    if not fit_result.models:
        raise LizyMLError(
            code=ErrorCode.MODEL_NOT_FIT,
            user_message="No trained models found in FitResult.",
            context={},
        )

    # Average importance across folds
    agg: dict[str, float] = {}
    n_models = len(fit_result.models)
    for model in fit_result.models:
        for feat, val in model.importance(kind=kind).items():
            agg[feat] = agg.get(feat, 0.0) + val / n_models

    # Sort descending by importance
    sorted_items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    return _render_bar_chart(sorted_items, f"Feature Importance ({kind})")


def plot_importance_from_dict(
    importance: dict[str, float],
    *,
    title: str = "Feature Importance (SHAP)",
    top_n: int | None = 20,
) -> Any:
    """Plot feature importances from a pre-computed dict.

    Args:
        importance: Mapping of feature name to importance score.
        title: Chart title.
        top_n: Maximum number of features to show.  ``None`` shows all.

    Returns:
        A ``plotly.graph_objects.Figure`` object.

    Raises:
        LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
    """
    sorted_items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    return _render_bar_chart(sorted_items, title)
