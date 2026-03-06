"""Tests for Phase 23-C — PurgedTimeSeries / GroupTimeSeries Config connection (H-0032).

Covers:
- PurgedTimeSeries config validation and Model.fit
- GroupTimeSeries config validation and Model.fit
- Alias normalization
- Legacy key deprecation warnings
"""

from __future__ import annotations

import warnings

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
            "time_idx": np.arange(n),
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
        "data": {"target": "target", "time_col": "time_idx"},
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
        cfg = load_config(_base_config("purged_time_series", purge_gap=5, embargo=2))
        assert cfg.split.method == "purged_time_series"
        assert cfg.split.purge_gap == 5  # type: ignore[union-attr]
        assert cfg.split.embargo == 2  # type: ignore[union-attr]

    def test_fit_runs(self) -> None:
        df = _ts_df()
        model = Model(
            _base_config("purged_time_series", purge_gap=5),
            data=df,
        )
        fr = model.fit()
        assert fr.oof_pred.shape == (len(df),)
        assert len(fr.models) == 3

    def test_legacy_keys_warning(self) -> None:
        """Legacy keys purge_window/gap should emit deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = load_config(_base_config("purged_time_series", purge_window=5, gap=2))
        assert cfg.split.method == "purged_time_series"
        assert cfg.split.purge_gap == 5  # type: ignore[union-attr]
        assert cfg.split.embargo == 2  # type: ignore[union-attr]
        deprecation_msgs = [
            str(x.message) for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert any("purge_window" in m for m in deprecation_msgs)
        assert any("gap" in m for m in deprecation_msgs)

    def test_legacy_keys_fit_works(self) -> None:
        """Legacy keys should still produce a working model."""
        df = _ts_df()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = Model(
                _base_config("purged_time_series", purge_window=5),
                data=df,
            )
        fr = model.fit()
        assert fr.oof_pred.shape == (len(df),)

    def test_legacy_embargo_pct_warning(self) -> None:
        """Legacy key embargo_pct should emit deprecation warning and convert to int."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = load_config(
                _base_config("purged_time_series", purge_gap=5, embargo_pct=3.0)
            )
        assert cfg.split.embargo == 3  # type: ignore[union-attr]
        deprecation_msgs = [
            str(x.message) for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert any("embargo_pct" in m for m in deprecation_msgs)


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


class TestTimeColRequired:
    """time_col must be set for all time series split methods."""

    @pytest.mark.parametrize(
        "method",
        ["time_series", "purged_time_series", "group_time_series"],
    )
    def test_time_col_required(self, method: str) -> None:
        from lizyml.core.exceptions import ErrorCode, LizyMLError

        cfg_dict = _base_config(method)
        cfg_dict["data"] = {"target": "target"}  # no time_col
        if method == "group_time_series":
            cfg_dict["data"]["group_col"] = "group"
        df = _group_ts_df() if method == "group_time_series" else _ts_df()
        model = Model(cfg_dict, data=df)
        with pytest.raises(LizyMLError) as exc_info:
            model.fit()
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID
        assert "time_col" in str(exc_info.value.user_message)

    def test_unsorted_data_sorted_by_time_col(self) -> None:
        """Shuffled data should be sorted by time_col before splitting."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "time_idx": np.arange(n),
                "feat_a": rng.uniform(0, 10, n),
                "target": rng.normal(0, 1, n),
            }
        )
        # Shuffle the data
        df = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
        assert not df["time_idx"].is_monotonic_increasing  # confirm shuffled

        model = Model(_base_config("time_series"), data=df)
        fr = model.fit()
        # Verify OOF predictions were generated for all rows
        assert fr.oof_pred.shape == (n,)
        # Verify train indices are temporally ordered in each fold
        for train_idx, valid_idx in fr.splits.outer:
            assert train_idx.max() < valid_idx.min()
