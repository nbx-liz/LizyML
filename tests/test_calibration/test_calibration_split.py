"""Tests for calibration split.method inheritance (H-0044, Phase 26 + 28).

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
from lizyml.core.exceptions import ErrorCode, LizyMLError
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


class TestCalibrationSplitPurgedTimeSeries:
    """purged_time_series: calibration splits respect purge_gap + embargo."""

    def test_train_before_valid_with_gap(self) -> None:
        purge_gap, embargo = 5, 3
        raw = make_config(
            "binary",
            n_splits=3,
            split_method="purged_time_series",
            time_col="ts",
            split_overrides={"purge_gap": purge_gap, "embargo": embargo},
            calibration="platt",
            calibration_n_splits=2,
        )
        cfg = load_config(raw)
        splitter = build_calibration_splitter(cfg)
        splits = list(splitter.split(200))
        assert len(splits) >= 1
        for train_idx, valid_idx in splits:
            assert valid_idx.min() - train_idx.max() > purge_gap, (
                f"Gap too small: valid_min={valid_idx.min()}, "
                f"train_max={train_idx.max()}, purge_gap={purge_gap}"
            )

    def test_embargo_respected(self) -> None:
        purge_gap, embargo = 5, 3
        raw = make_config(
            "binary",
            n_splits=3,
            split_method="purged_time_series",
            time_col="ts",
            split_overrides={"purge_gap": purge_gap, "embargo": embargo},
            calibration="platt",
            calibration_n_splits=2,
        )
        cfg = load_config(raw)
        splitter = build_calibration_splitter(cfg)
        for train_idx, valid_idx in splitter.split(200):
            dead_zone = valid_idx.min() - train_idx.max() - 1
            assert dead_zone >= purge_gap + embargo, (
                f"Dead zone {dead_zone} < purge_gap({purge_gap})+embargo({embargo})"
            )

    def test_calibration_n_splits_independent(self) -> None:
        raw = make_config(
            "binary",
            n_splits=3,
            split_method="purged_time_series",
            time_col="ts",
            split_overrides={"purge_gap": 2, "embargo": 1},
            calibration="platt",
            calibration_n_splits=2,
        )
        cfg = load_config(raw)
        splitter = build_calibration_splitter(cfg)
        splits = list(splitter.split(200))
        assert len(splits) == 2


class TestCalibrationSplitGroupTimeSeries:
    """group_time_series: calibration splits respect group + temporal boundaries."""

    def _make_splits(
        self, n: int = 300, n_groups: int = 20, cal_n_splits: int = 2
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        raw = make_config(
            "binary",
            n_splits=3,
            split_method="group_time_series",
            group_col="grp",
            time_col="ts",
            calibration="platt",
            calibration_n_splits=cal_n_splits,
        )
        cfg = load_config(raw)
        splitter = build_calibration_splitter(cfg)
        groups = np.repeat(np.arange(n_groups), n // n_groups)
        return list(splitter.split(len(groups), groups=groups)), groups

    def test_no_group_overlap(self) -> None:
        splits, groups = self._make_splits()
        for train_idx, valid_idx in splits:
            train_groups = set(groups[train_idx])
            valid_groups = set(groups[valid_idx])
            assert train_groups.isdisjoint(valid_groups), (
                f"Group overlap: {train_groups & valid_groups}"
            )

    def test_temporal_ordering(self) -> None:
        splits, _groups = self._make_splits()
        for train_idx, valid_idx in splits:
            assert train_idx.max() < valid_idx.min(), (
                f"Temporal violation: max(train)={train_idx.max()} >= "
                f"min(valid)={valid_idx.min()}"
            )

    def test_calibration_n_splits_independent(self) -> None:
        splits, _ = self._make_splits(cal_n_splits=2)
        assert len(splits) == 2


class TestCalibrationSplitError:
    """Impossible calibration splits raise LizyMLError(CONFIG_INVALID)."""

    def test_too_many_splits_raises(self) -> None:
        cfg = make_config(
            "binary",
            n_estimators=10,
            n_splits=3,
            calibration="platt",
            calibration_n_splits=50,  # way too many
        )
        m = Model(cfg)
        with pytest.raises(LizyMLError) as exc_info:
            m.fit(data=make_binary_df(n=20))
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID
        assert "split_method" in exc_info.value.context
        assert "calibration_n_splits" in exc_info.value.context
        assert "n_samples" in exc_info.value.context

    def test_too_few_groups_raises(self) -> None:
        """group_time_series with too few groups for calibration n_splits."""
        cfg = make_config(
            "binary",
            n_estimators=10,
            n_splits=2,
            split_method="group_time_series",
            group_col="grp",
            time_col="ts",
            calibration="platt",
            calibration_n_splits=20,  # need 21 groups, only 10 available
        )
        m = Model(cfg)
        df = make_binary_df(n=200, group_col="grp", n_groups=10, time_col="ts")
        with pytest.raises(LizyMLError) as exc_info:
            m.fit(data=df)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID
        assert "n_groups" in exc_info.value.context
