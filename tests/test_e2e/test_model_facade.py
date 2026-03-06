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

from typing import Any

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

    def test_export_before_fit_raises(self, tmp_path: Any) -> None:
        # export() is now implemented; requires fit() first
        m = Model(_reg_config())
        with pytest.raises(LizyMLError) as exc_info:
            m.export(tmp_path / "out")
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_load_nonexistent_path_raises(self, tmp_path: Any) -> None:
        # load() is now implemented; raises DESERIALIZATION_FAILED for bad path
        with pytest.raises(LizyMLError) as exc_info:
            Model.load(tmp_path / "does_not_exist")
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED


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


# ---------------------------------------------------------------------------
# fit_result property (22-J)
# ---------------------------------------------------------------------------


class TestFitResultProperty:
    def test_returns_fit_result_after_fit(self) -> None:
        m = Model(_reg_config())
        m.fit(data=_reg_df())
        assert isinstance(m.fit_result, FitResult)

    def test_before_fit_raises(self) -> None:
        m = Model(_reg_config())
        with pytest.raises(LizyMLError) as exc_info:
            _ = m.fit_result
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT


# ---------------------------------------------------------------------------
# params_table (22-M / H-0035)
# ---------------------------------------------------------------------------


class TestParamsTable:
    def test_returns_dataframe(self) -> None:
        m = Model(_reg_config())
        m.fit(data=_reg_df())
        table = m.params_table()
        assert isinstance(table, pd.DataFrame)
        assert table.index.name == "parameter"
        assert "value" in table.columns

    def test_contains_expected_params(self) -> None:
        m = Model(_reg_config())
        m.fit(data=_reg_df())
        table = m.params_table()
        params = set(table.index)
        assert "objective" in params
        assert "num_leaves" in params
        assert "auto_num_leaves" in params
        assert "best_iteration_0" in params

    def test_before_fit_raises(self) -> None:
        m = Model(_reg_config())
        with pytest.raises(LizyMLError) as exc_info:
            m.params_table()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT


# ---------------------------------------------------------------------------
# H-0036: ratio params resolved with inner_train size
# ---------------------------------------------------------------------------


class TestRatioParamsInnerTrainSize:
    """Verify min_data_in_leaf uses inner_train size, not full dataset."""

    def test_min_data_in_leaf_uses_inner_train_size(self) -> None:
        """With n=200, 3-fold, validation_ratio=0.1:
        outer_train = 134 rows (200 * 2/3)
        inner_train = ceil(134 * 0.9) = 121 rows
        min_data_in_leaf = ceil(121 * 0.01) = 2
        NOT ceil(200 * 0.01) = 2 (happens to be the same for small n)
        Use ratio=0.1 to see a clearer difference.
        """
        config = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "target"},
            "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
            "model": {
                "name": "lgbm",
                "params": {"n_estimators": 20},
                "min_data_in_leaf_ratio": 0.1,
                "auto_num_leaves": False,
            },
            "training": {
                "seed": 0,
                "early_stopping": {
                    "enabled": True,
                    "rounds": 10,
                    "validation_ratio": 0.2,
                },
            },
        }
        n = 300
        df = _reg_df(n=n)
        m = Model(config)
        m.fit(data=df)

        # Expected: n_outer_train ≈ 200 (300 * 2/3), inner_train ≈ 160 (200 * 0.8)
        # min_data_in_leaf = ceil(160 * 0.1) = 16
        # NOT ceil(300 * 0.1) = 30
        booster = m.fit_result.models[0].get_native_model().booster_
        mdil = int(booster.params["min_data_in_leaf"])

        # Full dataset would give ceil(300 * 0.1) = 30
        # Inner train should give something around ceil(160 * 0.1) = 16
        assert mdil < 30, (
            f"min_data_in_leaf={mdil} should be less than "
            f"ceil(n_full * ratio)=30, indicating inner_train size was used"
        )
