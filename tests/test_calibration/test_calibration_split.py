"""Tests for calibration split behavior (H-0044, H-0058).

After H-0058, calibration cross-fit reuses outer CV splits.
Tests for independent calibration.n_splits have been replaced with
tests verifying calibration splits == outer splits.

Tests that call build_calibration_splitter() directly are kept for
backward-compat coverage but wrapped with DeprecationWarning handling.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from lizyml import Model
from lizyml.calibration.cross_fit import CalibrationResult
from lizyml.config.loader import load_config
from lizyml.core._model_factories import build_calibration_splitter
from tests._helpers import make_binary_df, make_config


class TestCalibrationSplitKFold:
    """kfold: calibration splits match outer splits (H-0058)."""

    def test_calibration_splits_match_outer(self) -> None:
        cfg = make_config(
            "binary",
            n_estimators=10,
            n_splits=3,
            calibration="platt",
        )
        m = Model(cfg)
        result = m.fit(data=make_binary_df(n=200))
        assert result.splits.calibration is not None
        assert len(result.splits.calibration) == len(result.splits.outer)

    def test_explicit_calibration_n_splits_ignored(self) -> None:
        """calibration_n_splits=4 is ignored; calibration uses outer n_splits=3."""
        cfg = make_config(
            "binary",
            n_estimators=10,
            n_splits=3,
            calibration="platt",
            calibration_n_splits=4,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            m = Model(cfg)
            result = m.fit(data=make_binary_df(n=200))
        assert len(result.splits.outer) == 3
        assert result.splits.calibration is not None
        assert len(result.splits.calibration) == 3


class TestCalibrationSplitStratified:
    """stratified_kfold: calibration splits preserve class distribution."""

    def test_both_classes_in_each_fold(self) -> None:
        cfg = make_config(
            "binary",
            n_estimators=10,
            n_splits=3,
            split_method="stratified_kfold",
            calibration="platt",
        )
        df = make_binary_df(n=200)
        m = Model(cfg)
        result = m.fit(data=df)
        y = df["target"].to_numpy()
        assert result.splits.calibration is not None
        for train_idx, valid_idx in result.splits.calibration:
            assert len(np.unique(y[train_idx])) == 2
            assert len(np.unique(y[valid_idx])) == 2


class TestCalibrationSplitGroupKFold:
    """group_kfold: calibration splits have no group overlap."""

    def test_no_group_overlap(self) -> None:
        cfg = make_config(
            "binary",
            n_estimators=10,
            n_splits=3,
            split_method="group_kfold",
            group_col="grp",
            calibration="platt",
        )
        df = make_binary_df(n=200, group_col="grp", n_groups=15)
        m = Model(cfg)
        result = m.fit(data=df)
        groups = df["grp"].to_numpy()
        assert result.splits.calibration is not None
        for train_idx, valid_idx in result.splits.calibration:
            train_groups = set(groups[train_idx])
            valid_groups = set(groups[valid_idx])
            assert train_groups.isdisjoint(valid_groups), (
                f"Group overlap in calibration split: {train_groups & valid_groups}"
            )


class TestCalibrationSplitTimeSeries:
    """time_series: calibration splits respect temporal ordering (via outer)."""

    def test_train_before_valid(self) -> None:
        """Verify temporal ordering via Model.fit (outer splits reused)."""
        cfg = make_config(
            "binary",
            n_splits=3,
            split_method="time_series",
            time_col="ts",
            calibration="platt",
        )
        df = make_binary_df(n=200, time_col="ts")
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.splits.calibration is not None
        for train_idx, valid_idx in result.splits.calibration:
            assert train_idx.max() < valid_idx.min(), (
                f"Temporal violation: max(train)={train_idx.max()} >= "
                f"min(valid)={valid_idx.min()}"
            )


class TestCalibrationLeakageRegression:
    """Existing leakage contract: cross-fit OOF != c_final."""

    def test_cross_fit_differs_from_c_final(self) -> None:
        cfg = make_config(
            "binary",
            n_estimators=20,
            calibration="platt",
        )
        df = make_binary_df(n=200)
        m = Model(cfg)
        result = m.fit(data=df)
        assert isinstance(result.calibrator, CalibrationResult)
        c_final_preds = result.calibrator.c_final.predict(
            result.oof_raw_scores
            if result.oof_raw_scores is not None
            else result.oof_pred
        )
        assert not np.allclose(
            result.calibrator.calibrated_oof, c_final_preds, atol=1e-6
        )


class TestCalibrationSplitPurgedTimeSeries:
    """purged_time_series: calibration respects purge_gap + embargo."""

    def test_train_before_valid_with_gap(self) -> None:
        purge_gap, embargo = 5, 3
        cfg = make_config(
            "binary",
            n_splits=3,
            split_method="purged_time_series",
            time_col="ts",
            split_overrides={"purge_gap": purge_gap, "embargo": embargo},
            calibration="platt",
        )
        df = make_binary_df(n=200, time_col="ts")
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.splits.calibration is not None
        for train_idx, valid_idx in result.splits.calibration:
            assert valid_idx.min() - train_idx.max() > purge_gap, (
                f"Gap too small: valid_min={valid_idx.min()}, "
                f"train_max={train_idx.max()}, purge_gap={purge_gap}"
            )


class TestCalibrationSplitGroupTimeSeries:
    """group_time_series: calibration respects group boundaries."""

    def test_no_group_overlap(self) -> None:
        cfg = make_config(
            "binary",
            n_splits=3,
            split_method="group_time_series",
            group_col="grp",
            time_col="ts",
            calibration="platt",
        )
        df = make_binary_df(n=300, group_col="grp", n_groups=20, time_col="ts")
        m = Model(cfg)
        result = m.fit(data=df)
        groups = df["grp"].to_numpy()
        assert result.splits.calibration is not None
        for train_idx, valid_idx in result.splits.calibration:
            train_groups = set(groups[train_idx])
            valid_groups = set(groups[valid_idx])
            assert train_groups.isdisjoint(valid_groups), (
                f"Group overlap: {train_groups & valid_groups}"
            )

    def test_calibration_splits_match_outer(self) -> None:
        """Calibration splits are identical to outer splits (H-0058)."""
        cfg = make_config(
            "binary",
            n_splits=3,
            split_method="group_time_series",
            group_col="grp",
            time_col="ts",
            calibration="platt",
        )
        df = make_binary_df(n=300, group_col="grp", n_groups=20, time_col="ts")
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.splits.calibration is not None
        assert len(result.splits.calibration) == len(result.splits.outer)
        for (cal_t, cal_v), (out_t, out_v) in zip(
            result.splits.calibration, result.splits.outer, strict=True
        ):
            np.testing.assert_array_equal(cal_t, out_t)
            np.testing.assert_array_equal(cal_v, out_v)


class TestBuildCalibrationSplitterDeprecated:
    """build_calibration_splitter() emits DeprecationWarning (H-0058)."""

    def test_still_functional(self) -> None:
        """The deprecated function still returns a valid splitter."""
        raw = make_config(
            "binary",
            n_splits=3,
            split_method="time_series",
            time_col="ts",
            calibration="platt",
            calibration_n_splits=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cfg = load_config(raw)

        with pytest.warns(DeprecationWarning, match="build_calibration_splitter"):
            splitter = build_calibration_splitter(cfg)
        splits = list(splitter.split(200))
        assert len(splits) == 2
        for train_idx, valid_idx in splits:
            assert train_idx.max() < valid_idx.min()
