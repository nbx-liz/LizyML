"""Tests for 22-I — Tuning Plot.

Covers:
- tuning_plot() returns Plotly Figure after tune()
- tuning_plot() raises MODEL_NOT_FIT before tune()
- tuning_plot() raises OPTIONAL_DEP_MISSING without plotly
- Figure contains expected traces (state markers + best score line)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TrialResult, TuningResult
from lizyml.plots.tuning import plot_tuning_history


def _reg_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"]
    return df


def _reg_config_with_tuning() -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
        "tuning": {
            "optuna": {
                "params": {"n_trials": 3, "direction": "minimize"},
                "space": {
                    "num_leaves": {"type": "int", "low": 8, "high": 32},
                },
            }
        },
    }


class TestTuningPlotE2E:
    def test_returns_figure_after_tune(self) -> None:
        m = Model(_reg_config_with_tuning())
        m.tune(data=_reg_df())
        fig = m.tuning_plot()
        assert isinstance(fig, go.Figure)

    def test_before_tune_raises(self) -> None:
        m = Model(_reg_config_with_tuning())
        with pytest.raises(LizyMLError) as exc_info:
            m.tuning_plot()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT


class TestTuningPlotUnit:
    @pytest.fixture()
    def tuning_result(self) -> TuningResult:
        return TuningResult(
            best_params={"num_leaves": 16},
            best_score=0.5,
            trials=[
                TrialResult(
                    number=0,
                    params={"num_leaves": 8},
                    score=1.0,
                    state="complete",
                ),
                TrialResult(
                    number=1,
                    params={"num_leaves": 16},
                    score=0.5,
                    state="complete",
                ),
                TrialResult(
                    number=2,
                    params={"num_leaves": 32},
                    score=0.8,
                    state="pruned",
                ),
            ],
            metric_name="rmse",
            direction="minimize",
        )

    def test_figure_has_traces(self, tuning_result: TuningResult) -> None:
        fig = plot_tuning_history(tuning_result)
        assert isinstance(fig, go.Figure)
        # At least: complete markers, pruned markers, best score line
        assert len(fig.data) >= 3

    def test_best_score_line_present(self, tuning_result: TuningResult) -> None:
        fig = plot_tuning_history(tuning_result)
        line_traces = [t for t in fig.data if t.mode == "lines"]
        assert len(line_traces) == 1
        assert line_traces[0].name == "Best Score"

    def test_optional_dep_missing(self, tuning_result: TuningResult) -> None:
        with patch("lizyml.plots.tuning._plotly", None):
            with pytest.raises(LizyMLError) as exc_info:
                plot_tuning_history(tuning_result)
            assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING
