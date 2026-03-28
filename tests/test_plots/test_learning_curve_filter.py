"""Tests for H-0062 — metrics filter parameter in plot_learning_curve().

Covers:
- metrics=None plots all metrics (backward compatible)
- metrics=["auc"] filters to matching subplots
- Non-existent metric raises LizyMLError with available metric names
- Multiple metric filter
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.fit_result import FitResult
from lizyml.plots.learning_curve import plot_learning_curve
from tests._helpers import make_config, make_regression_df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fit_result_with_history() -> FitResult:
    """Create a FitResult with synthetic multi-metric eval history."""
    df = make_regression_df(n=100)
    cfg = make_config("regression")
    m = Model(cfg)
    fit_result = m.fit(data=df)

    # Inject synthetic eval history with multiple metrics
    fit_result.history = [
        {
            "best_iteration": 10,
            "eval_history": {
                "valid_0": {
                    "huber": [0.5, 0.4, 0.3, 0.2],
                    "mae": [0.6, 0.5, 0.4, 0.3],
                    "mape": [0.8, 0.7, 0.6, 0.5],
                },
            },
        }
        for _ in fit_result.history
    ]
    return fit_result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetricsFilterNone:
    """metrics=None should plot all metrics (backward compatible)."""

    def test_none_plots_all(self) -> None:
        fit_result = _make_fit_result_with_history()
        fig = plot_learning_curve(fit_result, metrics=None)
        assert isinstance(fig, go.Figure)
        # Should have 3 metrics * n_folds traces
        # Subplot titles should contain all 3 metrics
        titles = [ann.text for ann in fig.layout.annotations if hasattr(ann, "text")]
        assert any("huber" in t for t in titles)
        assert any("mae" in t for t in titles)
        assert any("mape" in t for t in titles)

    def test_default_arg_backward_compatible(self) -> None:
        """Calling without metrics arg should work as before."""
        fit_result = _make_fit_result_with_history()
        fig = plot_learning_curve(fit_result)
        assert isinstance(fig, go.Figure)


class TestMetricsFilterSingle:
    """metrics=["mae"] should only show mae subplot."""

    def test_single_metric_filter(self) -> None:
        fit_result = _make_fit_result_with_history()
        fig = plot_learning_curve(fit_result, metrics=["mae"])
        assert isinstance(fig, go.Figure)

        titles = [ann.text for ann in fig.layout.annotations if hasattr(ann, "text")]
        # Should contain mae but NOT huber or mape
        assert any("mae" in t for t in titles)
        assert not any("huber" in t for t in titles)
        assert not any("mape" in t for t in titles)


class TestMetricsFilterMultiple:
    """metrics=["mae", "huber"] should show both subplots."""

    def test_multiple_metric_filter(self) -> None:
        fit_result = _make_fit_result_with_history()
        fig = plot_learning_curve(fit_result, metrics=["mae", "huber"])
        assert isinstance(fig, go.Figure)

        titles = [ann.text for ann in fig.layout.annotations if hasattr(ann, "text")]
        assert any("mae" in t for t in titles)
        assert any("huber" in t for t in titles)
        assert not any("mape" in t for t in titles)


class TestMetricsFilterInvalid:
    """Non-existent metric should raise LizyMLError with available metrics."""

    def test_nonexistent_metric_raises(self) -> None:
        fit_result = _make_fit_result_with_history()
        with pytest.raises(LizyMLError) as exc_info:
            plot_learning_curve(fit_result, metrics=["nonexistent_metric"])

        err = exc_info.value
        assert err.code == ErrorCode.CONFIG_INVALID
        # Should include available metrics in context
        assert "available_metrics" in err.context
        available = err.context["available_metrics"]
        assert "huber" in available
        assert "mae" in available
        assert "mape" in available
