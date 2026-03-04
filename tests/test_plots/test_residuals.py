"""Tests for residuals() and residuals_plot() (H-0006)."""

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
# residuals_plot()
# ---------------------------------------------------------------------------


class TestResidualsPlot:
    def test_returns_figure(self) -> None:
        df = _reg_df()
        m = Model(_base_cfg("regression"))
        m.fit(data=df)

        fig = m.residuals_plot()
        assert isinstance(fig, go.Figure)

    def test_no_plotly_raises(self) -> None:
        df = _reg_df()
        m = Model(_base_cfg("regression"))
        m.fit(data=df)

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
