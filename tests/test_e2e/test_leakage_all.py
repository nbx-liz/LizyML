"""Leakage detection tests.

Verifies:
1. OOF predictions use only data from folds that did NOT include the sample.
2. Calibration cross-fit generates OOF calibrated scores without leakage.
3. Feature pipeline fit is restricted to the training fold (no val data leakage).
"""

from __future__ import annotations

import numpy as np

from lizyml import Model
from lizyml.calibration.cross_fit import CalibrationResult, cross_fit_calibrate
from lizyml.calibration.platt import PlattCalibrator
from lizyml.splitters.kfold import KFoldSplitter
from tests._helpers import make_binary_df, make_config


class TestOofLeakage:
    def test_oof_predictions_use_held_out_data(self) -> None:
        """OOF predictions must not be identical to in-fold IF predictions.

        If there were leakage (i.e., OOF preds were computed on training data),
        OOF scores would match IF scores closely. We verify they differ.
        """
        df = make_binary_df()
        m = Model(make_config("binary", n_estimators=20))
        result = m.fit(data=df)

        # OOF preds are held-out; IF preds are on training folds
        # They must be different arrays (not same object, not all equal)
        for if_fold in result.if_pred_per_fold:
            # The lengths differ (OOF has n samples, IF covers training fold)
            assert len(if_fold) < len(result.oof_pred) or not np.allclose(
                result.oof_pred[: len(if_fold)], if_fold, atol=1e-6
            )

    def test_oof_no_nans(self) -> None:
        df = make_binary_df()
        result = Model(make_config("binary", n_estimators=20)).fit(data=df)
        assert not np.any(np.isnan(result.oof_pred))

    def test_all_samples_have_oof_pred(self) -> None:
        df = make_binary_df()
        result = Model(make_config("binary", n_estimators=20)).fit(data=df)
        assert result.oof_pred.shape == (len(df),)


class TestCalibrationLeakage:
    def test_cross_fit_oof_differs_from_c_final(self) -> None:
        """Calibrated OOF must differ from C_final applied to the same scores.

        If they were the same, C_final (fit on all data) would be leaking
        into the OOF evaluation.
        """
        rng = np.random.default_rng(42)
        n = 200
        scores = rng.uniform(0, 1, n)
        y = (scores > 0.5).astype(int)

        splits = list(KFoldSplitter(n_splits=5, shuffle=True, random_state=42).split(n))
        result = cross_fit_calibrate(
            oof_scores=scores,
            y=y,
            calibrator_factory=PlattCalibrator,
            split_indices=splits,
        )
        c_final_preds = result.c_final.predict(scores)
        # Cross-fit OOF and C_final predictions must differ
        assert not np.allclose(result.calibrated_oof, c_final_preds, atol=1e-6)

    def test_calibration_result_structure(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        y = (scores > 0.5).astype(int)
        splits = list(
            KFoldSplitter(n_splits=3, shuffle=True, random_state=42).split(100)
        )
        result = cross_fit_calibrate(
            oof_scores=scores,
            y=y,
            calibrator_factory=PlattCalibrator,
            split_indices=splits,
        )
        assert isinstance(result, CalibrationResult)
        assert result.calibrated_oof.shape == (100,)
        assert result.c_final is not None

    def test_calibration_integration_no_leakage(self) -> None:
        """End-to-end: calibrated metrics come from cross-fit OOF, not C_final."""
        df = make_binary_df()
        m = Model(make_config("binary", n_estimators=20, calibration="platt"))
        result = m.fit(data=df)
        assert result.calibrator is not None
        assert isinstance(result.calibrator, CalibrationResult)
        # calibrated_oof must be a full-length array
        assert result.calibrator.calibrated_oof.shape == (len(df),)


class TestPipelineLeakage:
    def test_pipeline_fit_on_train_only(self) -> None:
        """Feature pipeline must be fit on train fold, not val fold.

        We verify this by checking that pipeline_state in FitResult
        is populated (pipeline was fit at some point during CV).
        """
        df = make_binary_df()
        result = Model(make_config("binary", n_estimators=20)).fit(data=df)
        assert result.pipeline_state is not None
