"""Column drift tests.

Covers:
- Missing column → DATA_SCHEMA_INVALID (hard error, not warning)
- Extra columns → handled gracefully (warning added to PredictionResult.warnings)
- Unseen category handled without exception
- T-6 contract:
  - Extra column warning message contains the column name
  - used_features preserves training column order even if predict columns are reordered
  - used_features matches fit_result.feature_names
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError


def _reg_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"]
    return df


def _cat_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "feat_num": rng.uniform(0, 10, n),
            "feat_cat": rng.choice(["a", "b", "c"], n),
        }
    )
    df["target"] = (df["feat_num"] > 5).astype(int)
    return df


def _cfg(task: str = "regression") -> dict:
    return {
        "config_version": 1,
        "task": task,
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


class TestMissingColumn:
    def test_missing_feature_raises(self) -> None:
        df = _reg_df()
        m = Model(_cfg())
        m.fit(data=df)
        # Drop feat_b at predict time
        X_bad = df.drop(columns=["target", "feat_b"])
        with pytest.raises(LizyMLError) as exc_info:
            m.predict(X_bad)
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID

    def test_all_features_missing_raises(self) -> None:
        df = _reg_df()
        m = Model(_cfg())
        m.fit(data=df)
        X_empty = pd.DataFrame({"completely_wrong": [1.0, 2.0]})
        with pytest.raises(LizyMLError) as exc_info:
            m.predict(X_empty)
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID


class TestExtraColumns:
    def test_extra_column_does_not_raise(self) -> None:
        """Extra columns at predict time should be ignored (not raise)."""
        df = _reg_df()
        m = Model(_cfg())
        m.fit(data=df)
        X_extra = df.drop(columns=["target"]).copy()
        X_extra["extra_col"] = 999.0
        # Should not raise — extra columns are dropped
        result = m.predict(X_extra)
        assert result.pred.shape == (len(X_extra),)


class TestUnseenCategory:
    def test_unseen_category_does_not_raise(self) -> None:
        """Unseen categories at predict time should be handled gracefully."""
        df = _cat_df()
        cfg = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target"},
            "features": {"categorical": ["feat_cat"]},
            "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
            "model": {"name": "lgbm", "params": {"n_estimators": 10}},
            "training": {"seed": 0},
        }
        m = Model(cfg)
        m.fit(data=df)

        # Introduce unseen category at predict time
        X_unseen = df.drop(columns=["target"]).iloc[:5].copy()
        X_unseen["feat_cat"] = "UNSEEN_VALUE"
        # Should not raise; unseen categories handled by CategoricalEncoder
        result = m.predict(X_unseen)
        assert result.pred.shape == (5,)


# ---------------------------------------------------------------------------
# T-6: Column drift / PredictionResult meaning contract
# ---------------------------------------------------------------------------


class TestColumnDriftContract:
    def test_extra_column_warning_contains_column_name(self) -> None:
        """PredictionResult.warnings must name the extra column."""
        df = _reg_df()
        m = Model(_cfg())
        m.fit(data=df)

        X_extra = df.drop(columns=["target"]).copy()
        X_extra["surprise_col"] = 42.0
        result = m.predict(X_extra)

        assert len(result.warnings) > 0, "Expected at least one warning for extra column"
        combined = " ".join(result.warnings)
        assert "surprise_col" in combined, (
            f"Warning did not mention 'surprise_col': {result.warnings}"
        )

    def test_missing_column_is_hard_error_not_warning(self) -> None:
        """A missing required column must raise LizyMLError, NOT add a warning."""
        df = _reg_df()
        m = Model(_cfg())
        m.fit(data=df)

        X_missing = df.drop(columns=["target", "feat_b"])
        with pytest.raises(LizyMLError) as exc_info:
            m.predict(X_missing)
        # Must be an error, not silently degrade to warnings
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID

    def test_used_features_matches_training_feature_names(self) -> None:
        """PredictionResult.used_features must equal FitResult.feature_names."""
        df = _reg_df()
        m = Model(_cfg())
        result = m.fit(data=df)

        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        pred = m.predict(X_new)

        assert pred.used_features == result.feature_names, (
            f"used_features {pred.used_features} != feature_names {result.feature_names}"
        )

    def test_used_features_preserves_training_column_order(self) -> None:
        """used_features must reflect training column order, not predict-time order."""
        df = _reg_df()
        m = Model(_cfg())
        result = m.fit(data=df)

        # Reverse column order at predict time
        X_reversed = df.drop(columns=["target"]).iloc[:5][["feat_b", "feat_a"]]
        pred = m.predict(X_reversed)

        # used_features must match training order: ["feat_a", "feat_b"]
        assert pred.used_features == result.feature_names, (
            f"Column reorder changed used_features: {pred.used_features} "
            f"(expected {result.feature_names})"
        )
