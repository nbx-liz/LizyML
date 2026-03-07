"""Tests for ROC Curve plots: binary + multiclass OvR (H-0015, H-0019)."""

from __future__ import annotations

from unittest.mock import patch

import plotly.graph_objects as go
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

# ---------------------------------------------------------------------------
# ROC Curve Binary
# ---------------------------------------------------------------------------


class TestROCCurveBinary:
    def test_returns_figure(self) -> None:
        m = Model(make_config("binary"))
        m.fit(data=make_binary_df())
        fig = m.roc_curve_plot()
        assert isinstance(fig, go.Figure)

    def test_has_is_oos_traces(self) -> None:
        m = Model(make_config("binary"))
        m.fit(data=make_binary_df())
        fig = m.roc_curve_plot()
        trace_names = [t.name for t in fig.data if t.name]
        has_oos = any("OOS" in n for n in trace_names)
        has_is = any("IS" in n for n in trace_names)
        assert has_oos
        assert has_is

    def test_has_reference_line(self) -> None:
        m = Model(make_config("binary"))
        m.fit(data=make_binary_df())
        fig = m.roc_curve_plot()
        # At least 3 traces: OOS, IS, reference
        assert len(fig.data) >= 3


# ---------------------------------------------------------------------------
# ROC Curve Multiclass
# ---------------------------------------------------------------------------


class TestROCCurveMulticlass:
    def test_returns_figure(self) -> None:
        m = Model(make_config("multiclass"))
        m.fit(data=make_multiclass_df(n=200))
        fig = m.roc_curve_plot()
        assert isinstance(fig, go.Figure)

    def test_has_class_traces(self) -> None:
        m = Model(make_config("multiclass"))
        m.fit(data=make_multiclass_df(n=200))
        fig = m.roc_curve_plot()
        trace_names = [t.name for t in fig.data if t.name]
        # Should have traces for each class (IS + OOS)
        has_class_0 = any("Class 0" in n for n in trace_names)
        has_class_1 = any("Class 1" in n for n in trace_names)
        assert has_class_0
        assert has_class_1

    def test_title_contains_macro_auc(self) -> None:
        m = Model(make_config("multiclass"))
        m.fit(data=make_multiclass_df(n=200))
        fig = m.roc_curve_plot()
        assert "Macro AUC" in fig.layout.title.text


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestROCCurveErrors:
    def test_regression_raises(self) -> None:
        m = Model(make_config("regression"))
        m.fit(data=make_regression_df(n=100))
        with pytest.raises(LizyMLError) as exc_info:
            m.roc_curve_plot()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_before_fit_raises(self) -> None:
        m = Model(make_config("binary"))
        with pytest.raises(LizyMLError) as exc_info:
            m.roc_curve_plot()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_optional_dep_missing(self) -> None:
        m = Model(make_config("binary"))
        m.fit(data=make_binary_df())
        with patch("lizyml.plots.classification._plotly", None):
            with pytest.raises(LizyMLError) as exc_info:
                m.roc_curve_plot()
            assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING
