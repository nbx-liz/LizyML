"""Tests for Plots (Plotly-based).

Covers:
- plot_importance: runs without exception from FitResult, returns Plotly Figure
- plot_learning_curve: runs when eval_history present; raises on missing history
- plot_oof_distribution: runs for regression / binary / multiclass
- plot_importance_from_dict: renders from pre-computed dict
- OPTIONAL_DEP_MISSING when plotly is None
- MODEL_NOT_FIT on missing data
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.fit_result import FitResult
from lizyml.plots.importance import plot_importance, plot_importance_from_dict
from lizyml.plots.learning_curve import plot_learning_curve
from lizyml.plots.oof_distribution import plot_oof_distribution

# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------


def _reg_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"] + rng.normal(0, 0.1, n)
    return df


def _bin_df(n: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    return df


def _multi_df(n: int = 150, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = pd.cut(df["feat_a"], bins=3, labels=[0, 1, 2]).astype(int)
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


def _fit(task: str) -> FitResult:
    if task == "regression":
        df = _reg_df()
    elif task == "binary":
        df = _bin_df()
    else:
        df = _multi_df()
    m = Model(_base_cfg(task))
    return m.fit(data=df)


# ---------------------------------------------------------------------------
# importance plot
# ---------------------------------------------------------------------------


class TestImportancePlot:
    def test_returns_figure(self) -> None:
        fit_result = _fit("regression")
        fig = plot_importance(fit_result)
        assert isinstance(fig, go.Figure)

    def test_binary(self) -> None:
        fit_result = _fit("binary")
        fig = plot_importance(fit_result, kind="gain")
        assert isinstance(fig, go.Figure)

    def test_top_n(self) -> None:
        fit_result = _fit("regression")
        fig = plot_importance(fit_result, top_n=1)
        assert fig is not None

    def test_via_model(self) -> None:
        df = _reg_df()
        m = Model(_base_cfg("regression"))
        m.fit(data=df)
        fig = m.importance_plot()
        assert isinstance(fig, go.Figure)

    def test_optional_dep_missing(self) -> None:
        fit_result = _fit("regression")
        with (
            patch("lizyml.plots.importance._plotly", None),
            pytest.raises(LizyMLError) as exc_info,
        ):
            plot_importance(fit_result)
        assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING

    def test_no_models_raises(self) -> None:
        fit_result = _fit("regression")
        fit_result.models = []
        with pytest.raises(LizyMLError) as exc_info:
            plot_importance(fit_result)
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT


# ---------------------------------------------------------------------------
# importance from dict
# ---------------------------------------------------------------------------


class TestImportanceFromDict:
    def test_returns_figure(self) -> None:
        imp = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2}
        fig = plot_importance_from_dict(imp)
        assert isinstance(fig, go.Figure)

    def test_top_n(self) -> None:
        imp = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2}
        fig = plot_importance_from_dict(imp, top_n=1)
        assert isinstance(fig, go.Figure)

    def test_optional_dep_missing(self) -> None:
        imp = {"feat_a": 0.5}
        with (
            patch("lizyml.plots.importance._plotly", None),
            pytest.raises(LizyMLError) as exc_info,
        ):
            plot_importance_from_dict(imp)
        assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING


# ---------------------------------------------------------------------------
# learning curve plot
# ---------------------------------------------------------------------------


class TestLearningCurvePlot:
    def test_no_eval_history_raises(self) -> None:
        """Without early stopping, eval_history is empty → raise MODEL_NOT_FIT."""
        fit_result = _fit("regression")
        fit_result.history = [
            {"best_iteration": None, "eval_history": {}} for _ in fit_result.history
        ]
        with pytest.raises(LizyMLError) as exc_info:
            plot_learning_curve(fit_result)
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_no_history_raises(self) -> None:
        fit_result = _fit("regression")
        fit_result.history = []
        with pytest.raises(LizyMLError) as exc_info:
            plot_learning_curve(fit_result)
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_optional_dep_missing(self) -> None:
        fit_result = _fit("regression")
        with (
            patch("lizyml.plots.learning_curve._plotly", None),
            pytest.raises(LizyMLError) as exc_info,
        ):
            plot_learning_curve(fit_result)
        assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING

    def test_with_eval_history(self) -> None:
        """Inject synthetic eval history and verify figure is produced."""
        fit_result = _fit("regression")
        fit_result.history = [
            {
                "best_iteration": 10,
                "eval_history": {"valid_0": {"rmse": [0.5, 0.4, 0.3, 0.2]}},
            }
            for _ in fit_result.history
        ]
        fig = plot_learning_curve(fit_result)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# OOF distribution plot
# ---------------------------------------------------------------------------


class TestOofDistributionPlot:
    def test_regression(self) -> None:
        fit_result = _fit("regression")
        fig = plot_oof_distribution(fit_result)
        assert isinstance(fig, go.Figure)

    def test_binary(self) -> None:
        fit_result = _fit("binary")
        fig = plot_oof_distribution(fit_result)
        assert isinstance(fig, go.Figure)

    def test_multiclass(self) -> None:
        fit_result = _fit("multiclass")
        fig = plot_oof_distribution(fit_result)
        assert isinstance(fig, go.Figure)

    def test_via_model(self) -> None:
        df = _reg_df()
        m = Model(_base_cfg("regression"))
        m.fit(data=df)
        fig = m.plot_oof_distribution()
        assert isinstance(fig, go.Figure)

    def test_optional_dep_missing(self) -> None:
        fit_result = _fit("regression")
        with (
            patch("lizyml.plots.oof_distribution._plotly", None),
            pytest.raises(LizyMLError) as exc_info,
        ):
            plot_oof_distribution(fit_result)
        assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING


# ---------------------------------------------------------------------------
# Model.plot_*() before fit
# ---------------------------------------------------------------------------


class TestModelPlotMethods:
    def test_plot_learning_curve_before_fit_raises(self) -> None:
        m = Model(_base_cfg("regression"))
        with pytest.raises(LizyMLError) as exc_info:
            m.plot_learning_curve()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_importance_plot_before_fit_raises(self) -> None:
        m = Model(_base_cfg("regression"))
        with pytest.raises(LizyMLError) as exc_info:
            m.importance_plot()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_plot_oof_distribution_before_fit_raises(self) -> None:
        m = Model(_base_cfg("regression"))
        with pytest.raises(LizyMLError) as exc_info:
            m.plot_oof_distribution()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT
