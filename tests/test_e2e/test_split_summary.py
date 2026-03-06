"""Tests for Phase 23-D — split_summary (H-0033).

Covers:
- split_summary for kfold (size info only)
- split_summary for time_series (includes time range)
- split_summary before fit raises MODEL_NOT_FIT
- time_range stored in FitResult.splits
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reg_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = df["feat_a"] * 0.5 + rng.normal(0, 0.5, n)
    return df


def _ts_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = df["feat_a"] * 0.5 + rng.normal(0, 0.5, n)
    return df


def _kfold_config(n_splits: int = 3) -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": n_splits, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


def _ts_config(n_splits: int = 3) -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target", "time_col": "date"},
        "split": {"method": "time_series", "n_splits": n_splits},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSplitSummary:
    def test_kfold_returns_size_columns(self) -> None:
        df = _reg_df()
        model = Model(_kfold_config(), data=df)
        model.fit()

        summary = model.split_summary()
        assert isinstance(summary, pd.DataFrame)
        assert list(summary.columns) == ["fold", "train_size", "valid_size"]
        assert len(summary) == 3
        # All samples should be covered
        assert summary["valid_size"].sum() == len(df)

    def test_time_series_includes_time_columns(self) -> None:
        df = _ts_df()
        model = Model(_ts_config(), data=df)
        model.fit()

        summary = model.split_summary()
        assert "train_start" in summary.columns
        assert "train_end" in summary.columns
        assert "valid_start" in summary.columns
        assert "valid_end" in summary.columns
        assert len(summary) == 3
        # Each fold should have valid_start > train_end (time ordering)
        for _, row in summary.iterrows():
            assert row["valid_start"] > row["train_end"]

    def test_not_fit_raises(self) -> None:
        model = Model(_kfold_config())
        with pytest.raises(LizyMLError) as exc_info:
            model.split_summary()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_time_range_in_fit_result(self) -> None:
        df = _ts_df()
        model = Model(_ts_config(), data=df)
        fr = model.fit()

        assert fr.splits.time_range is not None
        assert len(fr.splits.time_range) == 3
        for entry in fr.splits.time_range:
            assert "fold" in entry
            assert "train_start" in entry
            assert "valid_end" in entry

    def test_no_time_range_for_kfold(self) -> None:
        df = _reg_df()
        model = Model(_kfold_config(), data=df)
        fr = model.fit()

        assert fr.splits.time_range is None
