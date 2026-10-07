"""Tests for the centralised plot theme helper (#123).

The helper exists so future theme changes (template, brand colors, fonts)
can be made in one place. These tests pin down the contract:

- ``apply_default_layout`` calls ``fig.update_layout`` with the requested
  title and forwards arbitrary kwargs.
- Default ``height`` / ``width`` are applied when not provided.
- A central ``DEFAULT_TEMPLATE`` is propagated to every plot.
"""

from __future__ import annotations

from typing import Any

import pytest

from lizyml.plots._theme import (
    DEFAULT_HEIGHT,
    DEFAULT_TEMPLATE,
    DEFAULT_WIDTH,
    apply_default_layout,
)


class _FigureSpy:
    """Minimal stub that records ``update_layout`` calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def update_layout(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class TestApplyDefaultLayout:
    def test_title_is_required_and_forwarded(self) -> None:
        fig = _FigureSpy()
        apply_default_layout(fig, title="My Plot")
        assert len(fig.calls) == 1
        assert fig.calls[0]["title"] == "My Plot"

    def test_default_template_is_applied(self) -> None:
        fig = _FigureSpy()
        apply_default_layout(fig, title="x")
        assert fig.calls[0]["template"] == DEFAULT_TEMPLATE

    def test_default_height_and_width_when_unset(self) -> None:
        fig = _FigureSpy()
        apply_default_layout(fig, title="x")
        assert fig.calls[0]["height"] == DEFAULT_HEIGHT
        assert fig.calls[0]["width"] == DEFAULT_WIDTH

    def test_explicit_height_and_width_override_defaults(self) -> None:
        fig = _FigureSpy()
        apply_default_layout(fig, title="x", height=900, width=1200)
        assert fig.calls[0]["height"] == 900
        assert fig.calls[0]["width"] == 1200

    def test_extra_layout_forwarded(self) -> None:
        fig = _FigureSpy()
        apply_default_layout(
            fig,
            title="x",
            xaxis_title="t",
            yaxis_title="y",
            barmode="overlay",
            margin={"l": 200},
        )
        call = fig.calls[0]
        assert call["xaxis_title"] == "t"
        assert call["yaxis_title"] == "y"
        assert call["barmode"] == "overlay"
        assert call["margin"] == {"l": 200}

    def test_none_values_dropped_to_use_plotly_default(self) -> None:
        """Passing ``None`` for an optional dim signals 'let plotly decide';
        the helper must not forward it (so plotly's auto-sizing kicks in)."""
        fig = _FigureSpy()
        apply_default_layout(fig, title="x", height=None, width=None)
        call = fig.calls[0]
        assert "height" not in call
        assert "width" not in call


_PLOT_MODULES = [
    "lizyml.plots.calibration",
    "lizyml.plots.classification",
    "lizyml.plots.importance",
    "lizyml.plots.learning_curve",
    "lizyml.plots.oof_distribution",
    "lizyml.plots.residuals",
    "lizyml.plots.tuning",
]


class TestThemeAppliedToAllPlots:
    """Every public plot method returns a figure the theme helper styled."""

    def test_every_plot_module_imports_apply_default_layout(self) -> None:
        import importlib

        for name in _PLOT_MODULES:
            mod = importlib.import_module(name)
            assert hasattr(mod, "apply_default_layout"), (
                f"{name} must import apply_default_layout for theme consistency"
            )

    def test_every_public_plot_figure_passes_through_the_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importing the helper is not using it (#270).

        The helper is wrapped in every plot module with a spy that delegates, and
        each public ``Model`` plot method is called on a fitted model. Every
        returned figure must be one the helper was applied to.
        """
        import importlib

        from lizyml import Model
        from tests._helpers import make_binary_df, make_config, make_regression_df

        styled: list[int] = []
        for name in _PLOT_MODULES:
            mod = importlib.import_module(name)
            real = mod.apply_default_layout

            def spy(fig: Any, *args: Any, _real: Any = real, **kwargs: Any) -> None:
                styled.append(id(fig))
                _real(fig, *args, **kwargs)

            monkeypatch.setattr(mod, "apply_default_layout", spy)

        reg = Model(
            make_config("regression", n_estimators=5, tuning_n_trials=2, num_threads=1)
        )
        reg.tune(data=make_regression_df(n=100))
        reg.fit(data=make_regression_df(n=100))
        binary = Model(
            make_config("binary", n_estimators=5, calibration="platt", num_threads=1)
        )
        binary.fit(data=make_binary_df(n=200))

        figures = {
            "residuals_plot": reg.residuals_plot(),
            "importance_plot": reg.importance_plot(),
            "plot_learning_curve": reg.plot_learning_curve(),
            "plot_oof_distribution": reg.plot_oof_distribution(),
            "tuning_plot": reg.tuning_plot(),
            "roc_curve_plot": binary.roc_curve_plot(),
            "calibration_plot": binary.calibration_plot(),
            "probability_histogram_plot": binary.probability_histogram_plot(),
        }
        unstyled = [name for name, fig in figures.items() if id(fig) not in styled]
        assert not unstyled, f"returned without the default layout: {unstyled}"
        for name, fig in figures.items():
            assert fig.layout.title.text, f"{name} has no title"
