"""Tests for calibration split.method inheritance (H-0044, Phase 26).

Verifies that calibration cross-fit splits inherit the outer split.method
and maintain group/time/purge boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from lizyml import Model
from lizyml.calibration.cross_fit import CalibrationResult
from lizyml.config.loader import load_config
from lizyml.core._model_factories import build_calibration_splitter
from lizyml.core.exceptions import LizyMLError
from tests._helpers import make_binary_df, make_config


class TestCalibrationSplitKFold:
    """kfold: calibration fold count matches calibration.n_splits."""

    def test_calibration_splits_count(self) -> None:
        cal_n_splits = 4
        cfg = make_config(
            "binary",
            n_estimators=10,
            n_splits=3,
            calibration="platt",
            calibration_n_splits=cal_n_splits,
        )
        m = Model(cfg)
        result = m.fit(data=make_binary_df(n=200))
        assert result.splits.calibration is not None
        assert len(result.splits.calibration) == cal_n_splits

    def test_outer_and_calibration_n_splits_independent(self) -> None:
        """outer n_splits=3, calibration n_splits=4 — they differ."""
        cfg = make_config(
            "binary",
            n_estimators=10,
            n_splits=3,
            calibration="platt",
            calibration_n_splits=4,
        )
        m = Model(cfg)
        result = m.fit(data=make_binary_df(n=200))
        assert len(result.splits.outer) == 3
        assert result.splits.calibration is not None
        assert len(result.splits.calibration) == 4


class TestCalibrationSplitStratified:
    """stratified_kfold: calibration splits preserve class distribution."""

    def test_both_classes_in_each_fold(self) -> None:
        cfg = make_config(
            "binary",
            n_estimators=10,
            n_splits=3,
            split_method="stratified_kfold",
            calibration="platt",
            calibration_n_splits=3,
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
            calibration_n_splits=3,
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
    """time_series: calibration splits respect temporal ordering."""

    def test_train_before_valid(self) -> None:
        """Verify temporal ordering directly via splitter (no full fit)."""
        raw = make_config(
            "binary",
            n_splits=3,
            split_method="time_series",
            time_col="ts",
            calibration="platt",
            calibration_n_splits=2,
        )
        cfg = load_config(raw)
        splitter = build_calibration_splitter(cfg)
        n = 200
        splits = list(splitter.split(n))
        assert len(splits) == 2
        for train_idx, valid_idx in splits:
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


class TestCalibrationSplitError:
    """Impossible calibration splits raise explicit errors."""

    def test_too_many_splits_raises(self) -> None:
        cfg = make_config(
            "binary",
            n_estimators=10,
            n_splits=3,
            calibration="platt",
            calibration_n_splits=50,  # way too many
        )
        m = Model(cfg)
        with pytest.raises((ValueError, LizyMLError)):
            m.fit(data=make_binary_df(n=20))
