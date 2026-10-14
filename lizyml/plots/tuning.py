"""Tuning history plot: trial scores, best-score progression, and round separators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lizyml.plots._deps import require_plotly
from lizyml.plots._theme import apply_default_layout

if TYPE_CHECKING:
    from lizyml.core.types.tuning_result import TuningResult

_plotly: Any = None
try:
    import plotly.graph_objects as go

    _plotly = go
except ImportError:  # pragma: no cover
    pass


_STATE_COLORS: dict[str, str] = {
    "complete": "steelblue",
    "pruned": "orange",
    "fail": "red",
}


def _round_spans(tuning_result: TuningResult) -> list[tuple[int, int, int]]:
    """``(round, first trial number, last trial number)`` per run of trials.

    Derived from each trial's ``round`` (decided by its number from the
    study's round records, H-0122 §6.5), not from ``RoundSummary.n_trials``:
    a round closed ``failed`` has no summary, yet its trials are plotted.
    """
    spans: list[tuple[int, int, int]] = []
    for trial in sorted(tuning_result.trials, key=lambda t: t.number):
        if spans and spans[-1][0] == trial.round:
            k, first, _ = spans[-1]
            spans[-1] = (k, first, trial.number)
        else:
            spans.append((trial.round, trial.number, trial.number))
    return spans


def plot_tuning_history(tuning_result: TuningResult) -> Any:
    """Plot trial-by-trial score with best-score cumulative line.

    When the trials span more than one round (H-0068), dotted lines are drawn
    at the round boundaries, with annotations showing each round's expanded
    dims. The boundaries come from the trials' rounds (H-0122 §6.5), so a
    round closed ``failed`` or cut short by its timeout is placed correctly.

    Args:
        tuning_result: Result from ``Model.tune()``.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    require_plotly(_plotly, feature="tuning plots")
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

    # H-0068, H-0122 §6.5: round separators, from the trials' own rounds.
    spans = _round_spans(tuning_result)
    if len(spans) > 1:
        for _, first, _ in spans[1:]:
            fig.add_vline(
                x=first - 0.5,
                line_dash="dot",
                line_color="gray",
                line_width=1,
                opacity=0.7,
            )
        by_round = {rs.round: rs for rs in tuning_result.rounds}
        for k, first, last in spans:
            label = f"Round {k}"
            summary = by_round.get(k)
            # A round without a summary (closed `failed`) gets a plain label.
            if summary is not None and summary.expanded_dims:
                label += f" ({', '.join(summary.expanded_dims)})"
            fig.add_annotation(
                x=(first + last) / 2,
                y=1.0,
                yref="paper",
                text=label,
                showarrow=False,
                font={"size": 10, "color": "gray"},
            )

    apply_default_layout(
        fig,
        title="Tuning History",
        xaxis_title="Trial",
        yaxis_title=tuning_result.metric_name,
        height=500,
        width=700,
    )
    return fig
