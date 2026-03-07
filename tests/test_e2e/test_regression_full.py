"""Full E2E pipeline test for regression.

Pipeline: Config → fit → evaluate → predict → export → load → predict

Tuning is skipped here as it is slow; tune E2E is covered separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from lizyml import Model
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.predict_result import PredictionResult
from tests._helpers import make_config, make_regression_df


class TestRegressionFullPipeline:
    def test_fit_returns_fit_result(self) -> None:
        m = Model(make_config("regression", n_estimators=20))
        result = m.fit(data=make_regression_df())
        assert isinstance(result, FitResult)

    def test_evaluate_returns_metrics(self) -> None:
        m = Model(make_config("regression", n_estimators=20))
        m.fit(data=make_regression_df())
        metrics = m.evaluate()
        assert "raw" in metrics
        assert "oof" in metrics["raw"]
        assert "rmse" in metrics["raw"]["oof"]
        assert "mae" in metrics["raw"]["oof"]

    def test_predict_returns_prediction_result(self) -> None:
        df = make_regression_df()
        m = Model(make_config("regression", n_estimators=20))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:10].reset_index(drop=True)
        result = m.predict(X_new)
        assert isinstance(result, PredictionResult)
        assert result.pred.shape == (10,)
        assert result.proba is None
        assert result.shap_values is None

    def test_export_then_load_predict(self, tmp_path: object) -> None:
        assert isinstance(tmp_path, type(tmp_path))  # ensure it's a Path-like
        from pathlib import Path

        tmp = Path(str(tmp_path))
        df = make_regression_df()
        m = Model(make_config("regression", n_estimators=20))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)

        original_pred = m.predict(X_new).pred
        m.export(tmp / "reg_model")

        loaded = Model.load(tmp / "reg_model")
        loaded_pred = loaded.predict(X_new).pred
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_evaluate_after_load(self, tmp_path: object) -> None:
        from pathlib import Path

        tmp = Path(str(tmp_path))
        m = Model(make_config("regression", n_estimators=20))
        m.fit(data=make_regression_df())
        original_metrics = m.evaluate()
        m.export(tmp / "reg_model")

        loaded = Model.load(tmp / "reg_model")
        loaded_metrics = loaded.evaluate()
        assert loaded_metrics["raw"]["oof"]["rmse"] == pytest.approx(
            original_metrics["raw"]["oof"]["rmse"]
        )
