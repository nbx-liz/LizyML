"""Tests for ROC Curve plots: binary + multiclass OvR (H-0015, H-0019)."""

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


def _multi_df(n: int = 200, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = pd.cut(df["feat_a"], bins=3, labels=[0, 1, 2]).astype(int)
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


# ---------------------------------------------------------------------------
# ROC Curve Binary
# ---------------------------------------------------------------------------


class TestROCCurveBinary:
    def test_returns_figure(self) -> None:
        m = Model(_base_cfg("binary"))
        m.fit(data=_bin_df())
        fig = m.roc_curve_plot()
        assert isinstance(fig, go.Figure)

    def test_has_is_oos_traces(self) -> None:
        m = Model(_base_cfg("binary"))
        m.fit(data=_bin_df())
        fig = m.roc_curve_plot()
        trace_names = [t.name for t in fig.data if t.name]
        has_oos = any("OOS" in n for n in trace_names)
        has_is = any("IS" in n for n in trace_names)
        assert has_oos
        assert has_is

    def test_has_reference_line(self) -> None:
        m = Model(_base_cfg("binary"))
        m.fit(data=_bin_df())
        fig = m.roc_curve_plot()
        # At least 3 traces: OOS, IS, reference
        assert len(fig.data) >= 3


# ---------------------------------------------------------------------------
# ROC Curve Multiclass
# ---------------------------------------------------------------------------


class TestROCCurveMulticlass:
    def test_returns_figure(self) -> None:
        m = Model(_base_cfg("multiclass"))
        m.fit(data=_multi_df())
        fig = m.roc_curve_plot()
        assert isinstance(fig, go.Figure)

    def test_has_class_traces(self) -> None:
        m = Model(_base_cfg("multiclass"))
        m.fit(data=_multi_df())
        fig = m.roc_curve_plot()
        trace_names = [t.name for t in fig.data if t.name]
        # Should have traces for each class (IS + OOS)
        has_class_0 = any("Class 0" in n for n in trace_names)
        has_class_1 = any("Class 1" in n for n in trace_names)
        assert has_class_0
        assert has_class_1

    def test_title_contains_macro_auc(self) -> None:
        m = Model(_base_cfg("multiclass"))
        m.fit(data=_multi_df())
        fig = m.roc_curve_plot()
        assert "Macro AUC" in fig.layout.title.text


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestROCCurveErrors:
    def test_regression_raises(self) -> None:
        m = Model(_base_cfg("regression"))
        m.fit(data=_reg_df())
        with pytest.raises(LizyMLError) as exc_info:
            m.roc_curve_plot()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_before_fit_raises(self) -> None:
        m = Model(_base_cfg("binary"))
        with pytest.raises(LizyMLError) as exc_info:
            m.roc_curve_plot()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_optional_dep_missing(self) -> None:
        m = Model(_base_cfg("binary"))
        m.fit(data=_bin_df())
        with patch("lizyml.plots.classification._plotly", None):
            with pytest.raises(LizyMLError) as exc_info:
                m.roc_curve_plot()
            assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING
