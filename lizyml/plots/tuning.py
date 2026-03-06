"""Tuning history plot: trial scores and best-score progression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.tuning_result import TuningResult

_plotly: Any = None
try:
    import plotly.graph_objects as go

    _plotly = go
except ImportError:  # pragma: no cover
    pass


def _require_plotly() -> None:
    if _plotly is None:
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                "plotly is required for tuning plots. "
                "Install with: pip install 'lizyml[plots]'"
            ),
            context={"package": "plotly"},
        )


_STATE_COLORS: dict[str, str] = {
    "complete": "steelblue",
    "pruned": "orange",
    "fail": "red",
}


def plot_tuning_history(tuning_result: TuningResult) -> Any:
    """Plot trial-by-trial score with best-score cumulative line.

    Args:
        tuning_result: Result from ``Model.tune()``.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    _require_plotly()
    go = _plotly

    fig = go.Figure()

    # Group trials by state for legend
    by_state: dict[str, list[tuple[int, float]]] = {}
    for t in tuning_result.trials:
        by_state.setdefault(t.state, []).append((t.number, t.score))

    for state, points in by_state.items():
        xs, ys = zip(*points, strict=True)
        fig.add_trace(
            go.Scatter(
                x=list(xs),
                y=list(ys),
                mode="markers",
                name=state.capitalize(),
                marker={"color": _STATE_COLORS.get(state, "gray"), "size": 8},
            )
        )

    # Best score cumulative line
    completed = [t for t in tuning_result.trials if t.state == "complete"]
    if completed:
        completed_sorted = sorted(completed, key=lambda t: t.number)
        best_xs: list[int] = []
        best_ys: list[float] = []
        is_minimize = tuning_result.direction == "minimize"
        running_best = float("inf") if is_minimize else float("-inf")
        for t in completed_sorted:
            if is_minimize:
                running_best = min(running_best, t.score)
            else:
                running_best = max(running_best, t.score)
            best_xs.append(t.number)
            best_ys.append(running_best)

        fig.add_trace(
            go.Scatter(
                x=best_xs,
                y=best_ys,
                mode="lines",
                name="Best Score",
                line={"color": "crimson", "width": 2, "dash": "dash"},
            )
        )

    fig.update_layout(
        title="Tuning History",
        xaxis_title="Trial",
        yaxis_title=tuning_result.metric_name,
        height=500,
        width=700,
    )
    return fig
