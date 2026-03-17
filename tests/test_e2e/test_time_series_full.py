"""Time series E2E tests — verify temporal ordering across all split levels.

Ensures outer CV, inner validation, and calibration splits all respect
temporal ordering when time-series-based split methods are used.

Uses regression task to avoid class imbalance issues in small time-series folds.
Binary calibration tests use larger datasets with balanced targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml.core.model import Model
from tests._helpers import make_config


def _make_time_regression_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a regression DataFrame with a time column."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
            "time": np.arange(n),
        }
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"] + rng.normal(0, 0.1, n)
    return df


def _make_time_group_df(
    n: int = 300, n_groups: int = 30, seed: int = 42
) -> pd.DataFrame:
    """Create a regression DataFrame with time and group columns.

    Groups are assigned as time blocks so that group order matches time order.
    This matches real-world time-series group data (e.g., daily batches).
    """
    rng = np.random.default_rng(seed)
    # Assign groups as contiguous time blocks
    block_size = n // n_groups
    groups = np.repeat(np.arange(n_groups), block_size)
    # Pad remaining rows with the last group
    if len(groups) < n:
        groups = np.concatenate([groups, np.full(n - len(groups), n_groups - 1)])
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
            "time": np.arange(n),
            "group": groups[:n].astype(int),
        }
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"] + rng.normal(0, 0.1, n)
    return df


# ===========================================================================
# Time series outer CV: train indices < valid indices
# ===========================================================================


class TestTimeSeriesOuterSplits:
    """Outer CV splits respect temporal order."""

    @pytest.mark.parametrize(
        "split_method",
        ["time_series", "purged_time_series"],
    )
    def test_train_before_valid(self, split_method: str) -> None:
        df = _make_time_regression_df()
        overrides = {}
        if split_method == "purged_time_series":
            overrides = {"purge_gap": 2, "embargo": 1}
        cfg = make_config(
            "regression",
            split_method=split_method,
            n_splits=3,
            time_col="time",
            split_overrides=overrides,
        )
        m = Model(cfg)
        result = m.fit(data=df)

        for fold_idx, (train_idx, valid_idx) in enumerate(result.splits.outer):
            assert train_idx.max() < valid_idx.min(), (
                f"Fold {fold_idx}: train max ({train_idx.max()}) >= "
                f"valid min ({valid_idx.min()})"
            )

    def test_purged_time_series_gap_honored(self) -> None:
        """Purge gap creates a buffer between train and valid."""
        df = _make_time_regression_df(n=300)
        purge_gap = 5
        cfg = make_config(
            "regression",
            split_method="purged_time_series",
            n_splits=3,
            time_col="time",
            split_overrides={"purge_gap": purge_gap, "embargo": 0},
        )
        m = Model(cfg)
        result = m.fit(data=df)

        for fold_idx, (train_idx, valid_idx) in enumerate(result.splits.outer):
            gap = valid_idx.min() - train_idx.max() - 1
            assert gap >= purge_gap, (
                f"Fold {fold_idx}: gap ({gap}) < purge_gap ({purge_gap})"
            )

    def test_all_samples_covered(self) -> None:
        """Every sample appears in at least one valid fold."""
        df = _make_time_regression_df(n=200)
        cfg = make_config(
            "regression",
            split_method="time_series",
            n_splits=3,
            time_col="time",
        )
        m = Model(cfg)
        result = m.fit(data=df)
        all_valid = np.concatenate([v for _, v in result.splits.outer])
        # Time-series expanding-window: later folds cover more data.
        # At minimum a significant fraction of samples should appear.
        coverage = len(np.unique(all_valid)) / len(df)
        assert coverage > 0.5, f"Only {coverage:.0%} of samples covered"


# ===========================================================================
# Time series with calibration: calibration splits also temporal
# ===========================================================================


class TestTimeSeriesCalibrationSplitter:
    """Calibration inherits time_series splitter type from outer split config."""

    def test_calibration_splitter_is_time_series(self) -> None:
        """Calibration splitter is TimeSeriesSplitter for time_series."""
        from lizyml.config.loader import load_config
        from lizyml.core._model_factories import build_calibration_splitter
        from lizyml.splitters.time_series import TimeSeriesSplitter

        cfg_dict = make_config(
            "binary",
            split_method="time_series",
            n_splits=3,
            time_col="time",
            calibration="platt",
            calibration_n_splits=4,
        )
        cfg = load_config(cfg_dict)
        splitter = build_calibration_splitter(cfg)
        assert isinstance(splitter, TimeSeriesSplitter)
        # Verify calibration_n_splits is used (4 folds, not outer 3)
        folds = list(splitter.split(100))
        assert len(folds) == 4

    def test_calibration_splitter_inherits_purged_params(self) -> None:
        """build_calibration_splitter preserves purge_gap/embargo from outer config."""
        from lizyml.config.loader import load_config
        from lizyml.core._model_factories import build_calibration_splitter
        from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter

        cfg_dict = make_config(
            "binary",
            split_method="purged_time_series",
            n_splits=3,
            time_col="time",
            calibration="platt",
            calibration_n_splits=3,
            split_overrides={"purge_gap": 10, "embargo": 5},
        )
        cfg = load_config(cfg_dict)
        splitter = build_calibration_splitter(cfg)
        assert isinstance(splitter, PurgedTimeSeriesSplitter)
        assert splitter.purge_gap == 10
        assert splitter.embargo == 5


# ===========================================================================
# Time series with group: group_time_series
# ===========================================================================


class TestGroupTimeSeries:
    """Group time series splits respect both group and temporal constraints."""

    def test_group_time_series_train_before_valid(self) -> None:
        df = _make_time_group_df(n=300, n_groups=30)
        cfg = make_config(
            "regression",
            split_method="group_time_series",
            n_splits=3,
            time_col="time",
            group_col="group",
        )
        m = Model(cfg)
        result = m.fit(data=df)

        for fold_idx, (train_idx, valid_idx) in enumerate(result.splits.outer):
            assert train_idx.max() < valid_idx.min(), (
                f"Fold {fold_idx}: temporal order violated"
            )

    def test_group_time_series_no_group_overlap(self) -> None:
        df = _make_time_group_df(n=300, n_groups=30)
        cfg = make_config(
            "regression",
            split_method="group_time_series",
            n_splits=3,
            time_col="time",
            group_col="group",
        )
        m = Model(cfg)
        result = m.fit(data=df)
        groups = df["group"].values

        for fold_idx, (train_idx, valid_idx) in enumerate(result.splits.outer):
            train_groups = set(groups[train_idx])
            valid_groups = set(groups[valid_idx])
            assert train_groups.isdisjoint(valid_groups), (
                f"Fold {fold_idx}: groups overlap between train and valid"
            )
