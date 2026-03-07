"""Golden contract tests — FitResult / PredictionResult / RunMeta field locking.

These tests fix the public shape and field structure.  Any breaking change
to the contracts must update both the implementation and this file.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from lizyml import Model
from lizyml.core.types.artifacts import RunMeta, SplitIndices
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.predict_result import PredictionResult
from tests._helpers import make_config, make_regression_df


class TestFitResultContract:
    def test_required_fields_present(self) -> None:
        Model(make_config("regression")).fit(data=make_regression_df(n=100))
        required = {
            "oof_pred",
            "if_pred_per_fold",
            "metrics",
            "models",
            "history",
            "feature_names",
            "dtypes",
            "categorical_features",
            "splits",
            "data_fingerprint",
            "pipeline_state",
            "calibrator",
            "run_meta",
            "oof_raw_scores",
        }
        actual = {f.name for f in dataclasses.fields(FitResult)}
        assert required == actual

    def test_oof_pred_shape_regression(self) -> None:
        df = make_regression_df(n=100)
        result = Model(make_config("regression")).fit(data=df)
        assert result.oof_pred.shape == (len(df),)

    def test_oof_pred_shape_multiclass(self) -> None:
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {"feat_a": rng.uniform(0, 10, 150), "feat_b": rng.uniform(-1, 1, 150)}
        )
        df["target"] = pd.cut(df["feat_a"], bins=3, labels=[0, 1, 2]).astype(int)
        result = Model(make_config("multiclass")).fit(data=df)
        assert result.oof_pred.shape == (len(df), 3)

    def test_if_pred_per_fold_length(self) -> None:
        result = Model(make_config("regression")).fit(data=make_regression_df(n=100))
        assert len(result.if_pred_per_fold) == 3

    def test_models_length_equals_n_splits(self) -> None:
        result = Model(make_config("regression")).fit(data=make_regression_df(n=100))
        assert len(result.models) == 3

    def test_history_length_equals_n_splits(self) -> None:
        result = Model(make_config("regression")).fit(data=make_regression_df(n=100))
        assert len(result.history) == 3

    def test_history_has_eval_history_key(self) -> None:
        result = Model(make_config("regression")).fit(data=make_regression_df(n=100))
        for fold_hist in result.history:
            assert "eval_history" in fold_hist
            assert "best_iteration" in fold_hist

    def test_feature_names_type(self) -> None:
        result = Model(make_config("regression")).fit(data=make_regression_df(n=100))
        assert isinstance(result.feature_names, list)
        assert all(isinstance(n, str) for n in result.feature_names)

    def test_run_meta_type(self) -> None:
        result = Model(make_config("regression")).fit(data=make_regression_df(n=100))
        assert isinstance(result.run_meta, RunMeta)
        assert isinstance(result.run_meta.run_id, str)
        assert len(result.run_meta.run_id) > 0

    def test_splits_type(self) -> None:
        result = Model(make_config("regression")).fit(data=make_regression_df(n=100))
        assert isinstance(result.splits, SplitIndices)

    def test_metrics_structure(self) -> None:
        result = Model(make_config("regression")).fit(data=make_regression_df(n=100))
        assert "raw" in result.metrics
        assert set(result.metrics["raw"].keys()) == {"oof", "if_mean", "if_per_fold"}

    def test_calibrator_none_when_no_calibration(self) -> None:
        result = Model(make_config("regression")).fit(data=make_regression_df(n=100))
        assert result.calibrator is None


class TestPredictionResultContract:
    def test_required_fields_present(self) -> None:
        required = {"pred", "proba", "shap_values", "used_features", "warnings"}
        actual = {f.name for f in dataclasses.fields(PredictionResult)}
        assert required == actual

    def test_regression_proba_is_none(self) -> None:
        df = make_regression_df(n=100)
        m = Model(make_config("regression"))
        m.fit(data=df)
        result = m.predict(df.drop(columns=["target"]).iloc[:5])
        assert result.proba is None

    def test_shap_values_none_by_default(self) -> None:
        df = make_regression_df(n=100)
        m = Model(make_config("regression"))
        m.fit(data=df)
        result = m.predict(df.drop(columns=["target"]).iloc[:5])
        assert result.shap_values is None

    def test_used_features_is_list_of_str(self) -> None:
        df = make_regression_df(n=100)
        m = Model(make_config("regression"))
        m.fit(data=df)
        result = m.predict(df.drop(columns=["target"]).iloc[:5])
        assert isinstance(result.used_features, list)
        assert all(isinstance(f, str) for f in result.used_features)

    def test_warnings_is_list(self) -> None:
        df = make_regression_df(n=100)
        m = Model(make_config("regression"))
        m.fit(data=df)
        result = m.predict(df.drop(columns=["target"]).iloc[:5])
        assert isinstance(result.warnings, list)


class TestRunMetaContract:
    def test_required_fields_present(self) -> None:
        required = {
            "lizyml_version",
            "python_version",
            "deps_versions",
            "config_normalized",
            "config_version",
            "run_id",
            "timestamp",
        }
        actual = {f.name for f in dataclasses.fields(RunMeta)}
        assert required == actual

    def test_run_id_unique_across_fits(self) -> None:
        df = make_regression_df(n=100)
        r1 = Model(make_config("regression")).fit(data=df)
        r2 = Model(make_config("regression")).fit(data=df)
        assert r1.run_meta.run_id != r2.run_meta.run_id
