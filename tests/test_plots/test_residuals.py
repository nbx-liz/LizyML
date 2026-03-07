"""Tests for residuals() and residuals_plot() (H-0006, H-0009)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_binary_df, make_config, make_regression_df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fitted_reg_model() -> Model:
    m = Model(make_config("regression"))
    m.fit(data=make_regression_df(n=100))
    return m


# ---------------------------------------------------------------------------
# residuals()
# ---------------------------------------------------------------------------


class TestResiduals:
    def test_shape_and_values(self) -> None:
        df = make_regression_df(n=100)
        m = Model(make_config("regression"))
        m.fit(data=df)

        resid = m.residuals()
        assert resid.shape == (len(df),)
        # Residuals = y - oof_pred
        expected = np.asarray(m._y) - m._fit_result.oof_pred  # type: ignore[union-attr]
        np.testing.assert_array_almost_equal(resid, expected)

    def test_regression_only(self) -> None:
        df = make_binary_df(n=100)
        m = Model(make_config("binary"))
        m.fit(data=df)

        with pytest.raises(LizyMLError) as exc_info:
            m.residuals()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_before_fit_raises(self) -> None:
        m = Model(make_config("regression"))
        with pytest.raises(LizyMLError) as exc_info:
            m.residuals()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_after_load_succeeds(self, tmp_path: object) -> None:
        import tempfile
        from pathlib import Path

        df = make_regression_df(n=100)
        m = Model(make_config("regression"))
        m.fit(data=df)

        with tempfile.TemporaryDirectory() as td:
            export_dir = Path(td) / "model"
            m.export(str(export_dir))
            loaded = Model.load(str(export_dir))

        result = loaded.residuals()
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# residuals_plot() — kind tests
# ---------------------------------------------------------------------------


class TestResidualsPlot:
    def test_default_returns_figure(self) -> None:
        m = _fitted_reg_model()
        fig = m.residuals_plot()
        assert isinstance(fig, go.Figure)

    def test_kind_all(self) -> None:
        m = _fitted_reg_model()
        fig = m.residuals_plot(kind="all")
        assert isinstance(fig, go.Figure)
        # "all" uses make_subplots → multiple traces
        assert len(fig.data) >= 3

    def test_kind_scatter(self) -> None:
        m = _fitted_reg_model()
        fig = m.residuals_plot(kind="scatter")
        assert isinstance(fig, go.Figure)
        # Expects OOS trace, IS trace, and y=x reference line
        assert len(fig.data) >= 2

    def test_kind_histogram(self) -> None:
        m = _fitted_reg_model()
        fig = m.residuals_plot(kind="histogram")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_kind_qq(self) -> None:
        m = _fitted_reg_model()
        fig = m.residuals_plot(kind="qq")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_kind_invalid(self) -> None:
        m = _fitted_reg_model()
        with pytest.raises(LizyMLError) as exc_info:
            m.residuals_plot(kind="invalid_kind")
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_scatter_has_oos_and_is_traces(self) -> None:
        m = _fitted_reg_model()
        fig = m.residuals_plot(kind="scatter")
        trace_names = [t.name for t in fig.data]
        assert "OOS" in trace_names
        assert "IS" in trace_names

    def test_histogram_has_oos_and_is_traces(self) -> None:
        m = _fitted_reg_model()
        fig = m.residuals_plot(kind="histogram")
        trace_names = [t.name for t in fig.data]
        assert "OOS" in trace_names
        assert "IS" in trace_names

    def test_no_plotly_raises(self) -> None:
        m = _fitted_reg_model()
        with (
            patch("lizyml.plots.residuals._plotly", None),
            pytest.raises(LizyMLError) as exc_info,
        ):
            m.residuals_plot()
        assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING

    def test_binary_raises(self) -> None:
        df = make_binary_df(n=100)
        m = Model(make_config("binary"))
        m.fit(data=df)

        with pytest.raises(LizyMLError) as exc_info:
            m.residuals_plot()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_is_downsampled_in_scatter(self) -> None:
        """IS must be downsampled to OOS count (3-fold, 100 rows → IS≈200 > OOS=100)."""
        m = _fitted_reg_model()
        fig = m.residuals_plot(kind="scatter")
        oos_trace = next(t for t in fig.data if t.name == "OOS")
        is_trace = next(t for t in fig.data if t.name == "IS")
        assert len(is_trace.x) <= len(oos_trace.x)

    def test_is_downsampled_in_histogram(self) -> None:
        """IS histogram must have no more data points than OOS."""
        m = _fitted_reg_model()
        fig = m.residuals_plot(kind="histogram")
        oos_trace = next(t for t in fig.data if t.name == "OOS")
        is_trace = next(t for t in fig.data if t.name == "IS")
        assert len(is_trace.x) <= len(oos_trace.x)
