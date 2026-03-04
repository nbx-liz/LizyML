"""Tests for residuals() and residuals_plot() (H-0006, H-0009)."""

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


def _reg_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"] + rng.normal(0, 0.1, n)
    return df


def _bin_df(n: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
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


def _fitted_reg_model() -> Model:
    m = Model(_base_cfg("regression"))
    m.fit(data=_reg_df())
    return m


# ---------------------------------------------------------------------------
# residuals()
# ---------------------------------------------------------------------------


class TestResiduals:
    def test_shape_and_values(self) -> None:
        df = _reg_df()
        m = Model(_base_cfg("regression"))
        m.fit(data=df)

        resid = m.residuals()
        assert resid.shape == (len(df),)
        # Residuals = y - oof_pred
        expected = np.asarray(m._y) - m._fit_result.oof_pred  # type: ignore[union-attr]
        np.testing.assert_array_almost_equal(resid, expected)

    def test_regression_only(self) -> None:
        df = _bin_df()
        m = Model(_base_cfg("binary"))
        m.fit(data=df)

        with pytest.raises(LizyMLError) as exc_info:
            m.residuals()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_before_fit_raises(self) -> None:
        m = Model(_base_cfg("regression"))
        with pytest.raises(LizyMLError) as exc_info:
            m.residuals()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_after_load_raises(self, tmp_path: object) -> None:
        import tempfile
        from pathlib import Path

        df = _reg_df()
        m = Model(_base_cfg("regression"))
        m.fit(data=df)

        with tempfile.TemporaryDirectory() as td:
            export_dir = Path(td) / "model"
            m.export(str(export_dir))
            loaded = Model.load(str(export_dir))

        with pytest.raises(LizyMLError) as exc_info:
            loaded.residuals()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT
        assert "not available" in str(exc_info.value)


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
        # Expects OOS trace, IS trace, and y=0 line
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
        df = _bin_df()
        m = Model(_base_cfg("binary"))
        m.fit(data=df)

        with pytest.raises(LizyMLError) as exc_info:
            m.residuals_plot()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK
