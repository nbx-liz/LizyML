"""Tests for Calibration Curve + Probability Histogram (H-0017)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bin_df(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    return df


def _reg_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = df["feat_a"] * 2.0 + rng.normal(0, 0.1, n)
    return df


def _base_cfg(task: str) -> dict:
    return {
        "config_version": 1,
        "task": task,
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


def _calibrated_binary_model() -> Model:
    cfg = _base_cfg("binary")
    cfg["calibration"] = {"method": "platt", "n_splits": 3}
    m = Model(cfg)
    m.fit(data=_bin_df())
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
        m = Model(_base_cfg("binary"))
        m.fit(data=_bin_df())
        with pytest.raises(LizyMLError) as exc_info:
            m.calibration_plot()
        assert exc_info.value.code == ErrorCode.CALIBRATION_NOT_SUPPORTED

    def test_regression_raises(self) -> None:
        m = Model(_base_cfg("regression"))
        m.fit(data=_reg_df())
        with pytest.raises(LizyMLError) as exc_info:
            m.calibration_plot()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_before_fit_raises(self) -> None:
        m = Model(_base_cfg("binary"))
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
        m = Model(_base_cfg("binary"))
        m.fit(data=_bin_df())
        with pytest.raises(LizyMLError) as exc_info:
            m.probability_histogram_plot()
        assert exc_info.value.code == ErrorCode.CALIBRATION_NOT_SUPPORTED

    def test_regression_raises(self) -> None:
        m = Model(_base_cfg("regression"))
        m.fit(data=_reg_df())
        with pytest.raises(LizyMLError) as exc_info:
            m.probability_histogram_plot()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_optional_dep_missing(self) -> None:
        m = _calibrated_binary_model()
        with patch("lizyml.plots.calibration._plotly", None):
            with pytest.raises(LizyMLError) as exc_info:
                m.probability_histogram_plot()
            assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING
