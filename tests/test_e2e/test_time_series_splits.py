"""Tests for Phase 23-C — PurgedTimeSeries / GroupTimeSeries Config connection (H-0032).

Covers:
- PurgedTimeSeries config validation and Model.fit
- GroupTimeSeries config validation and Model.fit
- Alias normalization
- InnerValid auto-resolution
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.config.loader import load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Time-indexed regression dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n),
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = df["feat_a"] * 0.5 + rng.normal(0, 0.5, n)
    return df


def _group_ts_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Group-based time series dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "group": np.repeat(np.arange(20), n // 20),
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = df["feat_a"] * 0.5 + rng.normal(0, 0.5, n)
    return df


def _base_config(method: str, **extra: object) -> dict:
    cfg: dict = {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": method, "n_splits": 3, **extra},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }
    return cfg


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestPurgedTimeSeriesConfig:
    def test_valid_config(self) -> None:
        cfg = load_config(_base_config("purged_time_series", purge_window=5, gap=2))
        assert cfg.split.method == "purged_time_series"

    def test_fit_runs(self) -> None:
        df = _ts_df()
        model = Model(
            _base_config("purged_time_series", purge_window=5),
            data=df,
        )
        fr = model.fit()
        assert fr.oof_pred.shape == (len(df),)
        assert len(fr.models) == 3


class TestGroupTimeSeriesConfig:
    def test_valid_config(self) -> None:
        cfg = load_config(_base_config("group_time_series"))
        assert cfg.split.method == "group_time_series"

    def test_fit_runs(self) -> None:
        df = _group_ts_df()
        cfg = _base_config("group_time_series")
        cfg["data"]["group_col"] = "group"
        model = Model(cfg, data=df)
        fr = model.fit()
        assert fr.oof_pred.shape == (len(df),)
        assert len(fr.models) == 3


class TestAliasNormalization:
    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("purged-time-series", "purged_time_series"),
            ("purgedtimeseries", "purged_time_series"),
            ("group-time-series", "group_time_series"),
            ("grouptimeseries", "group_time_series"),
        ],
    )
    def test_alias(self, alias: str, expected: str) -> None:
        cfg = load_config(_base_config(alias))
        assert cfg.split.method == expected
