"""Tests for Calibration Curve + Probability Histogram (H-0017)."""

from __future__ import annotations

from unittest.mock import patch

import plotly.graph_objects as go
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_binary_df, make_config, make_regression_df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calibrated_binary_model() -> Model:
    cfg = make_config("binary")
    cfg["calibration"] = {"method": "platt", "n_splits": 3}
    m = Model(cfg)
    m.fit(data=make_binary_df())
    return m


# ---------------------------------------------------------------------------
# Calibration Curve
# ---------------------------------------------------------------------------


class TestCalibrationCurve:
    def test_returns_figure(self) -> None:
        m = _calibrated_binary_model()
        fig = m.calibration_plot()
        assert isinstance(fig, go.Figure)

    def test_has_three_traces(self) -> None:
        m = _calibrated_binary_model()
        fig = m.calibration_plot()
        # Reference line + Raw + Calibrated
        assert len(fig.data) == 3

    def test_trace_names(self) -> None:
        m = _calibrated_binary_model()
        fig = m.calibration_plot()
        names = [t.name for t in fig.data]
        assert "Perfect" in names
        assert "Raw OOF" in names
        assert "Calibrated OOF" in names


# ---------------------------------------------------------------------------
# Probability Histogram
# ---------------------------------------------------------------------------


class TestProbabilityHistogram:
    def test_returns_figure(self) -> None:
        m = _calibrated_binary_model()
        fig = m.probability_histogram_plot()
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self) -> None:
        m = _calibrated_binary_model()
        fig = m.probability_histogram_plot()
        assert len(fig.data) == 2

    def test_trace_names(self) -> None:
        m = _calibrated_binary_model()
        fig = m.probability_histogram_plot()
        names = [t.name for t in fig.data]
        assert "Raw OOF" in names
        assert "Calibrated OOF" in names


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestCalibrationPlotErrors:
    def test_no_calibration_raises(self) -> None:
        m = Model(make_config("binary"))
        m.fit(data=make_binary_df())
        with pytest.raises(LizyMLError) as exc_info:
            m.calibration_plot()
        assert exc_info.value.code == ErrorCode.CALIBRATION_NOT_SUPPORTED

    def test_regression_raises(self) -> None:
        m = Model(make_config("regression"))
        m.fit(data=make_regression_df(n=100))
        with pytest.raises(LizyMLError) as exc_info:
            m.calibration_plot()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_before_fit_raises(self) -> None:
        m = Model(make_config("binary"))
        with pytest.raises(LizyMLError) as exc_info:
            m.calibration_plot()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_optional_dep_missing(self) -> None:
        m = _calibrated_binary_model()
        with patch("lizyml.plots.calibration._plotly", None):
            with pytest.raises(LizyMLError) as exc_info:
                m.calibration_plot()
            assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING


class TestProbabilityHistogramErrors:
    def test_no_calibration_raises(self) -> None:
        m = Model(make_config("binary"))
        m.fit(data=make_binary_df())
        with pytest.raises(LizyMLError) as exc_info:
            m.probability_histogram_plot()
        assert exc_info.value.code == ErrorCode.CALIBRATION_NOT_SUPPORTED

    def test_regression_raises(self) -> None:
        m = Model(make_config("regression"))
        m.fit(data=make_regression_df(n=100))
        with pytest.raises(LizyMLError) as exc_info:
            m.probability_histogram_plot()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_optional_dep_missing(self) -> None:
        m = _calibrated_binary_model()
        with patch("lizyml.plots.calibration._plotly", None):
            with pytest.raises(LizyMLError) as exc_info:
                m.probability_histogram_plot()
            assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING
