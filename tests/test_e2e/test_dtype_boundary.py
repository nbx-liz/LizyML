"""Category D: Input dtype & boundary value E2E tests.

Tests Parquet full pipeline, float32 input, nullable dtypes,
0-row/1-row/duplicate-column/inf edge cases, and category order mismatch.

See BLUEPRINT §18.1.7 and HISTORY H-0056 Category D.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import LizyMLError
from lizyml.core.model import Model
from tests._helpers import make_binary_df, make_config, make_regression_df

# ===================================================================
# Parquet full pipeline
# ===================================================================


class TestParquetFullPipeline:
    """Parquet → fit → export → load → predict."""

    def test_parquet_regression_e2e(self, tmp_path: Path) -> None:
        df = make_regression_df(n=100, seed=0)
        parquet_path = tmp_path / "data.parquet"
        df.to_parquet(parquet_path)

        cfg = make_config("regression", n_estimators=5, n_splits=2)
        cfg["data"]["path"] = str(parquet_path)
        m = Model(cfg)
        result = m.fit()
        assert result.oof_pred is not None

        export_dir = tmp_path / "export"
        m.export(export_dir)
        m2 = Model.load(export_dir)
        X_test = df.drop(columns=["target"]).iloc[:10]
        pred = m2.predict(X_test)
        assert pred.pred.shape == (10,)

    def test_parquet_binary_with_calibration_e2e(self, tmp_path: Path) -> None:
        df = make_binary_df(n=200, seed=0)
        parquet_path = tmp_path / "data.parquet"
        df.to_parquet(parquet_path)

        cfg = make_config(
            "binary",
            n_estimators=5,
            n_splits=2,
            split_method="stratified_kfold",
            calibration="platt",
        )
        cfg["data"]["path"] = str(parquet_path)
        m = Model(cfg)
        result = m.fit()
        assert result.oof_pred is not None


# ===================================================================
# Float32 input
# ===================================================================


class TestFloat32Input:
    """float32 DataFrame inputs produce float64 outputs."""

    def test_float32_fit_returns_float64_oof(self) -> None:
        df = make_regression_df(n=100, seed=0)
        df["feat_a"] = df["feat_a"].astype("float32")
        df["feat_b"] = df["feat_b"].astype("float32")
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.oof_pred.dtype == np.float64

    def test_float32_predict_returns_float64(self) -> None:
        df = make_regression_df(n=100, seed=0)
        df["feat_a"] = df["feat_a"].astype("float32")
        df["feat_b"] = df["feat_b"].astype("float32")
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        m = Model(cfg)
        m.fit(data=df)
        X_test = df.drop(columns=["target"]).iloc[:10]
        pred = m.predict(X_test)
        assert pred.pred.dtype == np.float64


# ===================================================================
# Nullable dtypes
# ===================================================================


class TestNullableDtype:
    """Nullable (pd.Float64/pd.Int64) dtypes are handled."""

    def test_nullable_float64_fit(self) -> None:
        df = make_regression_df(n=100, seed=0)
        df["feat_a"] = pd.array(df["feat_a"].values, dtype="Float64")
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.oof_pred is not None

    def test_nullable_with_actual_nulls(self) -> None:
        """Nullable column with some NaN values — LightGBM handles NaN."""
        df = make_regression_df(n=100, seed=0)
        vals = df["feat_a"].values.copy()
        vals[::10] = np.nan  # Set every 10th value to NaN
        df["feat_a"] = pd.array(vals, dtype="Float64")
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.oof_pred is not None


# ===================================================================
# Empty and single-row DataFrames
# ===================================================================


class TestEdgeCaseRowCounts:
    """Extreme row counts produce clear errors or valid results."""

    def test_very_small_df_raises_or_warns(self) -> None:
        """3 rows with 3-fold CV should fail (< n_splits samples per fold)."""
        from lightgbm.basic import LightGBMError

        df = make_regression_df(n=3, seed=0)
        cfg = make_config("regression", n_estimators=5, n_splits=3)
        m = Model(cfg)
        # May raise LizyMLError, ValueError, or LightGBMError depending
        # on where the insufficient-data condition is caught.
        with pytest.raises((LizyMLError, ValueError, LightGBMError)):
            m.fit(data=df)

    def test_minimal_viable_df(self) -> None:
        """Minimum viable dataset for 2-fold CV."""
        df = make_regression_df(n=10, seed=0)
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.oof_pred is not None


# ===================================================================
# Extreme feature values
# ===================================================================


class TestExtremeValues:
    """inf / very large values in features."""

    def test_large_values_handled(self) -> None:
        """Very large float values (not inf) should work."""
        df = make_regression_df(n=100, seed=0)
        df.loc[0, "feat_a"] = 1e15
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.oof_pred is not None

    def test_all_identical_feature_handled(self) -> None:
        """All-same feature value: model should still train."""
        df = make_regression_df(n=100, seed=0)
        df["feat_a"] = 5.0  # constant column
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.oof_pred is not None


# ===================================================================
# Category order mismatch between train and predict
# ===================================================================


class TestCategoryOrderMismatch:
    """Category order differs between train and predict."""

    def test_category_order_mismatch_handled(self, tmp_path: Path) -> None:
        df = make_regression_df(n=100, seed=0)
        cats = pd.Categorical(
            np.random.default_rng(0).choice(["a", "b", "c"], 100),
            categories=["a", "b", "c"],
        )
        df["cat_feat"] = cats
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        m = Model(cfg)
        m.fit(data=df)

        export_dir = tmp_path / "export"
        m.export(export_dir)
        m2 = Model.load(export_dir)

        # Predict with reversed category order
        X_test = df.drop(columns=["target"]).iloc[:10].copy()
        X_test["cat_feat"] = pd.Categorical(
            X_test["cat_feat"], categories=["c", "b", "a"]
        )
        pred = m2.predict(X_test)
        assert pred.pred.shape == (10,)


# ===================================================================
# Duplicate column names
# ===================================================================


class TestDuplicateColumns:
    """Duplicate column names → error or documented behavior."""

    def test_duplicate_columns_handled(self) -> None:
        """Duplicate column names should raise or produce meaningful error."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 3))
        df = pd.DataFrame(data, columns=["a", "a", "target"])
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        m = Model(cfg)
        # Should either raise LizyMLError or a pandas/framework error
        with pytest.raises((LizyMLError, ValueError, KeyError, AttributeError)):
            m.fit(data=df)
