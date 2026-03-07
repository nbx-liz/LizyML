"""Full E2E pipeline test for multiclass classification.

Pipeline: Config → fit → evaluate → predict → export → load → predict
Calibration is not supported for multiclass.
"""

from __future__ import annotations

import numpy as np

from lizyml import Model
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.predict_result import PredictionResult
from tests._helpers import make_config, make_multiclass_df


class TestMulticlassFullPipeline:
    def test_fit_returns_fit_result(self) -> None:
        m = Model(make_config("multiclass", n_estimators=20))
        result = m.fit(data=make_multiclass_df())
        assert isinstance(result, FitResult)

    def test_oof_shape_is_2d(self) -> None:
        df = make_multiclass_df()
        m = Model(make_config("multiclass", n_estimators=20))
        result = m.fit(data=df)
        assert result.oof_pred.shape == (len(df), 3)

    def test_oof_proba_sums_to_one(self) -> None:
        df = make_multiclass_df()
        m = Model(make_config("multiclass", n_estimators=20))
        result = m.fit(data=df)
        row_sums = result.oof_pred.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_evaluate_multiclass_metrics(self) -> None:
        m = Model(make_config("multiclass", n_estimators=20))
        m.fit(data=make_multiclass_df())
        oof = m.evaluate()["raw"]["oof"]
        assert "logloss" in oof
        assert "f1" in oof
        assert "accuracy" in oof

    def test_predict_shape(self) -> None:
        df = make_multiclass_df()
        m = Model(make_config("multiclass", n_estimators=20))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        result = m.predict(X_new)
        assert isinstance(result, PredictionResult)
        assert result.pred.shape == (5,)
        assert result.proba is not None
        assert result.proba.shape == (5, 3)

    def test_predict_argmax_consistent(self) -> None:
        df = make_multiclass_df()
        m = Model(make_config("multiclass", n_estimators=20))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:10].reset_index(drop=True)
        result = m.predict(X_new)
        assert result.proba is not None
        expected_pred = result.proba.argmax(axis=1)
        np.testing.assert_array_equal(result.pred, expected_pred)

    def test_export_then_load_predict(self, tmp_path: object) -> None:
        from pathlib import Path

        tmp = Path(str(tmp_path))
        df = make_multiclass_df()
        m = Model(make_config("multiclass", n_estimators=20))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        original_pred = m.predict(X_new).pred
        m.export(tmp / "multi_model")

        loaded = Model.load(tmp / "multi_model")
        loaded_pred = loaded.predict(X_new).pred
        np.testing.assert_array_equal(original_pred, loaded_pred)
