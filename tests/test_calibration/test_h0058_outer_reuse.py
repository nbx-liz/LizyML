"""H-0058: Verify calibration cross-fit reuses outer CV splits.

Acceptance criteria from HISTORY.md H-0058 Proposal:
- calibration.n_splits specified → UserWarning emitted.
- calibration cross-fit uses fit_result.splits.outer.
- SplitIndices.calibration == SplitIndices.outer.
- _model_metrics.py splits workaround removed.
- TimeSeriesCV: calibrated OOF coverage == raw OOF coverage.
- Existing Model.load() loads old artifacts without error.
- Leakage detection tests still pass.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from lizyml.config.schema import CalibrationConfig
from lizyml.core.model import Model
from tests._helpers import (
    make_binary_df,
    make_config,
)

# ---------------------------------------------------------------------------
# Config deprecation warning
# ---------------------------------------------------------------------------


class TestNSplitsDeprecation:
    """CalibrationConfig.n_splits is deprecated (H-0058)."""

    def test_explicit_n_splits_warns(self) -> None:
        """Passing n_splits explicitly emits a UserWarning."""
        with pytest.warns(UserWarning, match="n_splits.*deprecated"):
            CalibrationConfig(**{"method": "platt", "n_splits": 3})

    def test_default_n_splits_no_warning(self) -> None:
        """Omitting n_splits does NOT emit a warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CalibrationConfig(method="platt")

    def test_n_splits_value_ignored(self) -> None:
        """Even with explicit n_splits=3 and outer n_splits=5,
        calibration splits match the outer (5 folds)."""
        df = make_binary_df(n=200, seed=0)
        cfg = make_config(
            "binary",
            n_splits=5,
            calibration="platt",
            calibration_n_splits=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            m = Model(cfg)
            result = m.fit(data=df)

        cal_splits = result.splits.calibration
        outer_splits = result.splits.outer
        assert cal_splits is not None
        assert len(cal_splits) == len(outer_splits)


# ---------------------------------------------------------------------------
# Calibration splits == outer splits
# ---------------------------------------------------------------------------


class TestCalibrationSplitsEqualOuter:
    """Calibration splits must be identical to outer splits (H-0058)."""

    def test_kfold_calibration_splits_identical(self) -> None:
        df = make_binary_df(n=200, seed=0)
        cfg = make_config("binary", n_splits=3, calibration="platt")
        m = Model(cfg)
        result = m.fit(data=df)

        assert result.splits.calibration is not None
        assert len(result.splits.calibration) == len(result.splits.outer)
        for (cal_t, cal_v), (out_t, out_v) in zip(
            result.splits.calibration, result.splits.outer, strict=True
        ):
            np.testing.assert_array_equal(cal_t, out_t)
            np.testing.assert_array_equal(cal_v, out_v)

    def test_time_series_calibration_splits_identical(self) -> None:
        df = make_binary_df(n=200, seed=0)
        df["time"] = range(len(df))
        cfg = make_config(
            "binary",
            n_splits=3,
            split_method="time_series",
            time_col="time",
            calibration="platt",
        )
        m = Model(cfg)
        result = m.fit(data=df)

        assert result.splits.calibration is not None
        assert len(result.splits.calibration) == len(result.splits.outer)
        for (cal_t, cal_v), (out_t, out_v) in zip(
            result.splits.calibration, result.splits.outer, strict=True
        ):
            np.testing.assert_array_equal(cal_t, out_t)
            np.testing.assert_array_equal(cal_v, out_v)


# ---------------------------------------------------------------------------
# Coverage identity
# ---------------------------------------------------------------------------


class TestCoverageIdentity:
    """Calibrated OOF coverage must match raw OOF coverage."""

    def test_kfold_full_coverage(self) -> None:
        """KFold: both raw and calibrated OOF have full coverage."""
        df = make_binary_df(n=200, seed=0)
        cfg = make_config("binary", n_splits=3, calibration="platt")
        m = Model(cfg)
        result = m.fit(data=df)

        # Raw OOF: no NaN
        assert not np.any(np.isnan(result.oof_pred))
        # Calibrated OOF: no NaN
        assert result.calibrator is not None
        assert not np.any(np.isnan(result.calibrator.calibrated_oof))

    def test_time_series_calibrated_coverage_equals_raw(self) -> None:
        """TimeSeriesCV: calibrated OOF NaN positions match raw OOF."""
        df = make_binary_df(n=200, seed=0)
        df["time"] = range(len(df))
        cfg = make_config(
            "binary",
            n_splits=3,
            split_method="time_series",
            time_col="time",
            calibration="platt",
        )
        m = Model(cfg)
        result = m.fit(data=df)

        assert result.calibrator is not None
        raw_nan_mask = np.isnan(result.oof_pred)
        cal_nan_mask = np.isnan(result.calibrator.calibrated_oof)
        np.testing.assert_array_equal(raw_nan_mask, cal_nan_mask)


# ---------------------------------------------------------------------------
# build_calibration_splitter deprecation
# ---------------------------------------------------------------------------


class TestBuildCalibrationSplitterDeprecated:
    """build_calibration_splitter() must emit DeprecationWarning."""

    def test_build_calibration_splitter_warns(self) -> None:
        from lizyml.core._model_factories import build_calibration_splitter

        cfg_dict = make_config("binary", calibration="platt")
        from lizyml.config.schema import LizyMLConfig

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cfg = LizyMLConfig(**cfg_dict)

        with pytest.warns(DeprecationWarning, match="build_calibration_splitter"):
            build_calibration_splitter(cfg)
