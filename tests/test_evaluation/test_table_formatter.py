"""Tests for evaluation table formatter (H-0005)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.evaluation.table_formatter import format_metrics_table

# ---------------------------------------------------------------------------
# Unit tests for format_metrics_table
# ---------------------------------------------------------------------------


class TestFormatMetricsTable:
    def test_basic_structure(self) -> None:
        metrics = {
            "raw": {
                "oof": {"rmse": 0.5, "mae": 0.3},
                "if_mean": {"rmse": 0.4, "mae": 0.25},
                "if_per_fold": [
                    {"rmse": 0.38, "mae": 0.23},
                    {"rmse": 0.42, "mae": 0.27},
                ],
            }
        }
        df = format_metrics_table(metrics)
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ["rmse", "mae"]
        assert df.index.name == "metric"
        assert "oof" in df.columns
        assert "if_mean" in df.columns
        assert "fold_0" in df.columns
        assert "fold_1" in df.columns
        # Column order: if_mean first, then oof (H-0011)
        assert list(df.columns[:2]) == ["if_mean", "oof"]
        assert df.loc["rmse", "oof"] == pytest.approx(0.5)
        assert df.loc["mae", "if_mean"] == pytest.approx(0.25)
        assert df.loc["rmse", "fold_0"] == pytest.approx(0.38)

    def test_with_calibrated(self) -> None:
        metrics = {
            "raw": {
                "oof": {"logloss": 0.5},
                "if_mean": {"logloss": 0.45},
                "if_per_fold": [{"logloss": 0.45}],
            },
            "calibrated": {
                "oof": {"logloss": 0.48},
            },
        }
        df = format_metrics_table(metrics)
        assert "cal_oof" in df.columns
        assert df.loc["logloss", "cal_oof"] == pytest.approx(0.48)

    def test_no_calibrated(self) -> None:
        metrics = {
            "raw": {
                "oof": {"rmse": 0.5},
                "if_mean": {"rmse": 0.4},
                "if_per_fold": [],
            }
        }
        df = format_metrics_table(metrics)
        assert "cal_oof" not in df.columns

    def test_per_fold_columns(self) -> None:
        metrics = {
            "raw": {
                "oof": {"rmse": 0.5},
                "if_mean": {"rmse": 0.4},
                "if_per_fold": [
                    {"rmse": 0.38},
                    {"rmse": 0.40},
                    {"rmse": 0.42},
                    {"rmse": 0.44},
                    {"rmse": 0.46},
                ],
            }
        }
        df = format_metrics_table(metrics)
        for i in range(5):
            assert f"fold_{i}" in df.columns

    def test_empty_metrics(self) -> None:
        df = format_metrics_table({})
        assert df.empty

    def test_empty_raw(self) -> None:
        raw = {"oof": {}, "if_mean": {}, "if_per_fold": []}
        df = format_metrics_table({"raw": raw})
        assert df.empty


# ---------------------------------------------------------------------------
# E2E via Model
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


class TestModelEvaluateTable:
    def test_via_model(self) -> None:
        cfg = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "target"},
            "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
            "model": {"name": "lgbm", "params": {"n_estimators": 10}},
            "training": {"seed": 0},
        }
        m = Model(cfg)
        m.fit(data=_reg_df())
        df = m.evaluate_table()
        assert isinstance(df, pd.DataFrame)
        assert "oof" in df.columns
        assert "if_mean" in df.columns
        assert len(df) > 0

    def test_before_fit_raises(self) -> None:
        cfg = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "target"},
            "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
            "model": {"name": "lgbm", "params": {"n_estimators": 10}},
            "training": {"seed": 0},
        }
        m = Model(cfg)
        with pytest.raises(LizyMLError) as exc_info:
            m.evaluate_table()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT
