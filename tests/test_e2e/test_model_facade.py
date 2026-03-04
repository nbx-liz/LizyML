"""Tests for Phase 11 — Model Facade.

Covers:
- Config → fit → evaluate → predict pipeline for regression and binary
- FitResult returned from fit() has correct structure
- evaluate() returns structured metrics dict
- predict() returns PredictionResult with correct shape
- importance() returns dict keyed by feature names
- MODEL_NOT_FIT error before fit
- predict raises DATA_SCHEMA_INVALID on column mismatch
- Reproducibility: same config+seed → same OOF predictions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.predict_result import PredictionResult

# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------


def _reg_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
            "target": rng.uniform(0, 10, n),
        }
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"] + rng.normal(0, 0.1, n)
    return df


def _bin_df(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    return df


def _multi_df(n: int = 300, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = pd.cut(df["feat_a"], bins=3, labels=[0, 1, 2]).astype(int)
    return df


def _reg_config(n_splits: int = 3) -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": n_splits, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 20}},
        "training": {"seed": 0},
    }


def _bin_config(n_splits: int = 3) -> dict:
    return {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": n_splits, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 20}},
        "training": {"seed": 0},
    }


def _multi_config(n_splits: int = 3) -> dict:
    return {
        "config_version": 1,
        "task": "multiclass",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": n_splits, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 20}},
        "training": {"seed": 0},
    }


# ---------------------------------------------------------------------------
# Regression E2E
# ---------------------------------------------------------------------------


class TestModelRegression:
    def test_fit_returns_fit_result(self) -> None:
        m = Model(_reg_config())
        result = m.fit(data=_reg_df())
        assert isinstance(result, FitResult)

    def test_oof_shape(self) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        result = m.fit(data=df)
        assert result.oof_pred.shape == (len(df),)

    def test_oof_no_nan(self) -> None:
        m = Model(_reg_config())
        result = m.fit(data=_reg_df())
        assert not np.any(np.isnan(result.oof_pred))

    def test_evaluate_structure(self) -> None:
        m = Model(_reg_config())
        m.fit(data=_reg_df())
        metrics = m.evaluate()
        assert "raw" in metrics
        assert set(metrics["raw"].keys()) == {"oof", "if_mean", "if_per_fold"}

    def test_evaluate_metrics_keys(self) -> None:
        m = Model(_reg_config())
        m.fit(data=_reg_df())
        oof_metrics = m.evaluate()["raw"]["oof"]
        # Default metrics: rmse, mae
        assert "rmse" in oof_metrics
        assert "mae" in oof_metrics

    def test_predict_shape(self) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:10].reset_index(drop=True)
        result = m.predict(X_new)
        assert isinstance(result, PredictionResult)
        assert result.pred.shape == (10,)
        assert result.proba is None

    def test_importance_keys(self) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        imp = m.importance()
        assert set(imp.keys()) == {"feat_a", "feat_b"}

    def test_reproducibility(self) -> None:
        df = _reg_df()
        m1 = Model(_reg_config())
        m2 = Model(_reg_config())
        r1 = m1.fit(data=df)
        r2 = m2.fit(data=df)
        np.testing.assert_array_almost_equal(r1.oof_pred, r2.oof_pred)

    def test_metrics_stored_in_fit_result(self) -> None:
        m = Model(_reg_config())
        result = m.fit(data=_reg_df())
        # Evaluator populated metrics during fit()
        assert "raw" in result.metrics


# ---------------------------------------------------------------------------
# Binary classification E2E
# ---------------------------------------------------------------------------


class TestModelBinary:
    def test_fit_returns_fit_result(self) -> None:
        m = Model(_bin_config())
        result = m.fit(data=_bin_df())
        assert isinstance(result, FitResult)

    def test_oof_proba_range(self) -> None:
        df = _bin_df()
        m = Model(_bin_config())
        result = m.fit(data=df)
        assert result.oof_pred.shape == (len(df),)
        assert np.all(result.oof_pred >= 0) and np.all(result.oof_pred <= 1)

    def test_predict_proba_shape(self) -> None:
        df = _bin_df()
        m = Model(_bin_config())
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        pred_result = m.predict(X_new)
        assert pred_result.proba is not None
        assert pred_result.proba.shape == (5,)
        assert pred_result.pred.shape == (5,)

    def test_evaluate_binary_metrics(self) -> None:
        m = Model(_bin_config())
        m.fit(data=_bin_df())
        oof_metrics = m.evaluate()["raw"]["oof"]
        # Default binary metrics: logloss, auc
        assert "logloss" in oof_metrics
        assert "auc" in oof_metrics


# ---------------------------------------------------------------------------
# Multiclass E2E
# ---------------------------------------------------------------------------


class TestModelMulticlass:
    def test_oof_shape_multiclass(self) -> None:
        df = _multi_df()
        m = Model(_multi_config())
        result = m.fit(data=df)
        # 3 classes → (n_samples, 3)
        assert result.oof_pred.shape == (len(df), 3)

    def test_predict_multiclass(self) -> None:
        df = _multi_df()
        m = Model(_multi_config())
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        pred_result = m.predict(X_new)
        assert pred_result.proba is not None
        assert pred_result.proba.shape == (5, 3)
        assert pred_result.pred.shape == (5,)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestModelErrors:
    def test_predict_before_fit_raises(self) -> None:
        m = Model(_reg_config())
        X = pd.DataFrame({"feat_a": [1.0], "feat_b": [0.5]})
        with pytest.raises(LizyMLError) as exc_info:
            m.predict(X)
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_evaluate_before_fit_raises(self) -> None:
        m = Model(_reg_config())
        with pytest.raises(LizyMLError) as exc_info:
            m.evaluate()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_importance_before_fit_raises(self) -> None:
        m = Model(_reg_config())
        with pytest.raises(LizyMLError) as exc_info:
            m.importance()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_predict_missing_column_raises(self) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        # Missing feat_b
        X_bad = pd.DataFrame({"feat_a": [1.0, 2.0]})
        with pytest.raises(LizyMLError) as exc_info:
            m.predict(X_bad)
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID

    def test_fit_without_data_raises(self) -> None:
        m = Model(_reg_config())
        with pytest.raises(LizyMLError) as exc_info:
            m.fit()  # no data, no path in config
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID

    def test_tune_without_tuning_config_raises(self) -> None:
        # tune() is now implemented; requires a tuning config section
        m = Model(_reg_config())
        data = pd.DataFrame({"feat_a": [1.0], "feat_b": [0.5], "target": [1.0]})
        with pytest.raises(LizyMLError) as exc_info:
            m.tune(data=data)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_export_not_implemented(self) -> None:
        m = Model(_reg_config())
        with pytest.raises(NotImplementedError):
            m.export("/tmp/model")

    def test_load_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            Model.load("/tmp/model")


# ---------------------------------------------------------------------------
# Import surface test
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_model_importable_from_lizyml(self) -> None:
        from lizyml import Model as M

        assert M is Model

    def test_version_importable(self) -> None:
        from lizyml import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0
