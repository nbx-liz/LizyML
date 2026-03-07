"""Full E2E pipeline test for binary classification.

Pipeline: Config → fit → evaluate → predict → export → load → predict
Also covers: calibration integration, proba output, logloss/auc metrics.
"""

from __future__ import annotations

import numpy as np

from lizyml import Model
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.predict_result import PredictionResult
from tests._helpers import make_binary_df, make_config


class TestBinaryFullPipeline:
    def test_fit_returns_fit_result(self) -> None:
        m = Model(make_config("binary", n_estimators=20))
        result = m.fit(data=make_binary_df())
        assert isinstance(result, FitResult)

    def test_oof_in_probability_range(self) -> None:
        df = make_binary_df()
        m = Model(make_config("binary", n_estimators=20))
        result = m.fit(data=df)
        assert result.oof_pred.shape == (len(df),)
        assert np.all(result.oof_pred >= 0) and np.all(result.oof_pred <= 1)

    def test_evaluate_binary_metrics(self) -> None:
        m = Model(make_config("binary", n_estimators=20))
        m.fit(data=make_binary_df())
        oof = m.evaluate()["raw"]["oof"]
        assert "logloss" in oof
        assert "auc" in oof

    def test_predict_proba_shape(self) -> None:
        df = make_binary_df()
        m = Model(make_config("binary", n_estimators=20))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        result = m.predict(X_new)
        assert isinstance(result, PredictionResult)
        assert result.pred.shape == (5,)
        assert result.proba is not None
        assert result.proba.shape == (5,)
        assert np.all(result.proba >= 0) and np.all(result.proba <= 1)

    def test_export_then_load_predict(self, tmp_path: object) -> None:
        from pathlib import Path

        tmp = Path(str(tmp_path))
        df = make_binary_df()
        m = Model(make_config("binary", n_estimators=20))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        original_pred = m.predict(X_new).pred
        m.export(tmp / "bin_model")

        loaded = Model.load(tmp / "bin_model")
        loaded_pred = loaded.predict(X_new).pred
        np.testing.assert_array_equal(original_pred, loaded_pred)


class TestBinaryWithCalibration:
    def test_fit_with_calibration(self) -> None:
        m = Model(make_config("binary", n_estimators=20, calibration="platt"))
        result = m.fit(data=make_binary_df())
        assert result.calibrator is not None

    def test_calibrated_metrics_present(self) -> None:
        m = Model(make_config("binary", n_estimators=20, calibration="platt"))
        result = m.fit(data=make_binary_df())
        assert "calibrated" in result.metrics

    def test_calibrated_proba_range(self) -> None:
        df = make_binary_df()
        m = Model(make_config("binary", n_estimators=20, calibration="platt"))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        result = m.predict(X_new)
        assert result.proba is not None
        assert np.all(result.proba >= 0) and np.all(result.proba <= 1)
