"""Leakage detection tests — fail-closed traps (H-0085 / #207).

Verifies, each with a test that fails closed on a regression:
1. OOF fold models never train on their own validation rows (CVTrainer
   training-data disjointness) — replaces the former tautological assertion.
2. Cross-fit calibrators are fit on the complementary train slice, not the
   validation rows they score.
3. ``FitResult.splits.inner`` relative indices stay within each outer train fold.
4. A NaN numeric/regression target is rejected (contract, #207 item 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.calibration.cross_fit import CalibrationResult, cross_fit_calibrate
from lizyml.calibration.platt import PlattCalibrator
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.artifacts import RunMeta
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.features.pipelines_native import NativeFeaturePipeline
from lizyml.splitters.kfold import KFoldSplitter
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.inner_valid import NoInnerValid
from tests._helpers import make_binary_df, make_config, make_regression_df


def _reg_data_with_rid(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    """Regression data carrying a row-identity column (``rid == row index``)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"rid": np.arange(n, dtype=float), "f": rng.normal(size=n)})
    y = pd.Series(rng.normal(size=n), name="target")
    return X, y


def _run_cv_recording_train_rids(
    X: pd.DataFrame, y: pd.Series, sink: list[set[int]], n_splits: int = 3
):
    """Run CVTrainer with an estimator that records the ``rid`` set it trains on."""

    def factory() -> LGBMAdapter:
        adapter = LGBMAdapter(
            task="regression", params={"n_estimators": 10}, random_state=0
        )
        orig_fit = adapter.fit

        def spy_fit(X_train, y_train, X_valid=None, y_valid=None, **kw):  # type: ignore[no-untyped-def]
            sink.append({int(r) for r in X_train["rid"].tolist()})
            return orig_fit(X_train, y_train, X_valid, y_valid, **kw)

        adapter.fit = spy_fit  # type: ignore[method-assign]
        return adapter

    trainer = CVTrainer(
        outer_splitter=KFoldSplitter(n_splits=n_splits, shuffle=True, random_state=0),
        inner_valid=NoInnerValid(),
        pipeline_factory=NativeFeaturePipeline,
        estimator_factory=factory,
        task="regression",
    )
    rm = RunMeta(
        lizyml_version="0.0.0",
        python_version="3.11",
        deps_versions={},
        config_normalized={},
        config_version=1,
        run_id="oof-trap",
        timestamp="2026-01-01T00:00:00",
    )
    return trainer.fit(
        X, y, data_fingerprint=fp_compute(X, file_path=None), run_meta=rm
    )


class TestOofLeakage:
    def test_oof_fold_models_never_train_on_their_valid_rows(self) -> None:
        """Fail-closed: each fold estimator's training rows must equal that
        fold's outer-train rows and be disjoint from its validation rows.

        A CVTrainer regression that fed validation rows into a fold's training
        set would make the recorded ``rid`` set intersect ``valid_idx``.
        """
        X, y = _reg_data_with_rid(60)
        sink: list[set[int]] = []
        result = _run_cv_recording_train_rids(X, y, sink)

        assert len(sink) == len(result.splits.outer)
        for train_rids, (train_idx, valid_idx) in zip(
            sink, result.splits.outer, strict=True
        ):
            assert train_rids == set(train_idx.tolist())
            assert train_rids.isdisjoint(set(valid_idx.tolist()))

    def test_oof_no_nans(self) -> None:
        df = make_binary_df()
        result = Model(make_config("binary", n_estimators=20)).fit(data=df)
        assert not np.any(np.isnan(result.oof_pred))

    def test_all_samples_have_oof_pred(self) -> None:
        df = make_binary_df()
        result = Model(make_config("binary", n_estimators=20)).fit(data=df)
        assert result.oof_pred.shape == (len(df),)


class TestInnerIndexContainment:
    def test_inner_indices_are_relative_to_outer_train(self) -> None:
        """Fail-closed: inner_train / inner_valid indices are 0-based relative
        to each outer train fold and must stay within ``[0, len(train))``.

        A CVTrainer regression storing absolute indices would exceed the fold's
        train length.
        """
        df = make_regression_df(n=180)
        cfg = make_config("regression", n_estimators=30, n_splits=3)
        cfg["training"]["early_stopping"] = {"enabled": True, "rounds": 5}
        result = Model(cfg).fit(data=df)

        assert result.splits.inner is not None
        assert len(result.splits.inner) == len(result.splits.outer)
        for (train_idx, _valid_idx), (inner_train, inner_valid) in zip(
            result.splits.outer, result.splits.inner, strict=True
        ):
            n_train = len(train_idx)
            for arr in (inner_train, inner_valid):
                assert arr.min() >= 0
                assert arr.max() < n_train


class TestCalibrationLeakage:
    def test_calibrator_fit_on_train_slice_not_valid(self) -> None:
        """Fail-closed: per-fold calibrated OOF must match a calibrator refit on
        the complementary train slice, and must differ from one fit on the
        validation rows themselves.
        """
        rng = np.random.default_rng(42)
        n = 200
        scores = rng.uniform(0, 1, n)
        y = (scores + rng.normal(0, 0.3, n) > 0.5).astype(int)

        splits = list(KFoldSplitter(n_splits=5, shuffle=True, random_state=42).split(n))
        result = cross_fit_calibrate(
            oof_scores=scores,
            y=y,
            calibrator_factory=PlattCalibrator,
            split_indices=splits,
        )

        train_idx, val_idx = splits[0]
        # Refit on the train slice → must equal the cross-fit OOF on val rows.
        cal_train = PlattCalibrator()
        cal_train.fit(scores[train_idx], y[train_idx])
        expected = cal_train.predict(scores[val_idx])
        np.testing.assert_allclose(result.calibrated_oof[val_idx], expected, atol=1e-9)
        # Sensitivity: a calibrator fit on the val rows themselves must differ,
        # proving the cross-fit did NOT leak val rows into calibrator training.
        cal_valid = PlattCalibrator()
        cal_valid.fit(scores[val_idx], y[val_idx])
        leaked = cal_valid.predict(scores[val_idx])
        assert not np.allclose(result.calibrated_oof[val_idx], leaked, atol=1e-6)

    def test_cross_fit_oof_differs_from_c_final(self) -> None:
        """Calibrated OOF must differ from C_final applied to the same scores."""
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
        assert result.calibrator.calibrated_oof.shape == (len(df),)


class TestPipelineLeakage:
    def test_pipeline_fit_on_train_only(self) -> None:
        """Pipeline state is populated after CV.

        The genuine train-only fail-closed trap lives in
        ``tests/test_training/test_pipeline_fit_boundary.py`` (valid-only
        category with ``unseen_policy='error'``); this end-to-end check pins
        that the fitted state is exported with the expected feature schema.
        """
        df = make_binary_df()
        result = Model(make_config("binary", n_estimators=20)).fit(data=df)
        assert result.pipeline_state is not None
        assert "feature_names" in result.pipeline_state


class TestTargetContract:
    def test_numeric_target_nan_rejected(self) -> None:
        """Fail-closed: a NaN in a numeric/regression target is rejected with
        ``DATA_SCHEMA_INVALID`` and a ``nan_count`` context (#207 item 4)."""
        df = make_regression_df(n=50)
        df.loc[0, "target"] = np.nan
        m = Model(make_config("regression", n_estimators=10))
        with pytest.raises(LizyMLError) as exc:
            m.fit(data=df)
        assert exc.value.code == ErrorCode.DATA_SCHEMA_INVALID
        assert exc.value.context.get("nan_count") == 1
