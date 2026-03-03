"""Tests for Phase 6: Feature Pipeline — NativeFeaturePipeline, CategoricalEncoder."""

from __future__ import annotations

import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from lizyml.features.pipelines_native import NativeFeaturePipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": pd.Categorical(["a", "b", "a", "c", "b"]),
        }
    )


@pytest.fixture()
def y_train(train_df: pd.DataFrame) -> pd.Series:
    return pd.Series([0, 1, 0, 1, 0], name="target")


# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------


class TestCategoricalEncoder:
    def test_fit_records_categories(self, train_df: pd.DataFrame) -> None:
        enc = CategoricalEncoder()
        enc.fit(train_df, ["cat"])
        assert set(enc._categories["cat"]) == {"a", "b", "c"}

    def test_transform_preserves_dtype(self, train_df: pd.DataFrame) -> None:
        enc = CategoricalEncoder()
        enc.fit(train_df, ["cat"])
        result = enc.transform(train_df)
        assert result["cat"].dtype.name == "category"

    def test_transform_restricts_to_training_categories(
        self, train_df: pd.DataFrame
    ) -> None:
        enc = CategoricalEncoder()
        enc.fit(train_df, ["cat"])
        test_df = pd.DataFrame({"num": [1.0], "cat": pd.Categorical(["a"])})
        result = enc.transform(test_df)
        assert set(result["cat"].cat.categories) == {"a", "b", "c"}

    def test_unseen_mode_policy(self, train_df: pd.DataFrame) -> None:
        enc = CategoricalEncoder(unseen_policy="mode")
        enc.fit(train_df, ["cat"])
        test_df = pd.DataFrame({"num": [1.0], "cat": pd.Categorical(["z"])})
        result = enc.transform(test_df)
        # unseen "z" replaced with mode; no NaN
        assert result["cat"].isna().sum() == 0

    def test_unseen_nan_policy(self, train_df: pd.DataFrame) -> None:
        enc = CategoricalEncoder(unseen_policy="nan")
        enc.fit(train_df, ["cat"])
        test_df = pd.DataFrame({"num": [1.0], "cat": pd.Categorical(["z"])})
        result = enc.transform(test_df)
        assert result["cat"].isna().sum() == 1

    def test_unseen_error_policy(self, train_df: pd.DataFrame) -> None:
        enc = CategoricalEncoder(unseen_policy="error")
        enc.fit(train_df, ["cat"])
        test_df = pd.DataFrame({"num": [1.0], "cat": pd.Categorical(["z"])})
        with pytest.raises(LizyMLError) as exc_info:
            enc.transform(test_df)
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID

    def test_state_roundtrip(self, train_df: pd.DataFrame) -> None:
        enc = CategoricalEncoder()
        enc.fit(train_df, ["cat"])
        state = enc.get_state()
        enc2 = CategoricalEncoder()
        enc2.load_state(state)
        result1 = enc.transform(train_df)
        result2 = enc2.transform(train_df)
        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# NativeFeaturePipeline — basic fit/transform
# ---------------------------------------------------------------------------


class TestNativeFeaturePipelineBasic:
    def test_fit_records_feature_names(
        self, train_df: pd.DataFrame, y_train: pd.Series
    ) -> None:
        pipe = NativeFeaturePipeline()
        pipe.fit(train_df, y_train)
        assert pipe._feature_names == list(train_df.columns)

    def test_transform_returns_dataframe(
        self, train_df: pd.DataFrame, y_train: pd.Series
    ) -> None:
        pipe = NativeFeaturePipeline()
        pipe.fit(train_df, y_train)
        result = pipe.transform(train_df)
        assert isinstance(result, pd.DataFrame)

    def test_categorical_col_becomes_category_dtype(
        self, train_df: pd.DataFrame, y_train: pd.Series
    ) -> None:
        pipe = NativeFeaturePipeline()
        pipe.fit(train_df, y_train)
        result = pipe.transform(train_df)
        assert result["cat"].dtype.name == "category"

    def test_fit_transform_consistent(
        self, train_df: pd.DataFrame, y_train: pd.Series
    ) -> None:
        pipe1 = NativeFeaturePipeline()
        pipe2 = NativeFeaturePipeline()
        ft = pipe1.fit_transform(train_df, y_train)
        pipe2.fit(train_df, y_train)
        t = pipe2.transform(train_df)
        pd.testing.assert_frame_equal(ft, t)


# ---------------------------------------------------------------------------
# NativeFeaturePipeline — column drift
# ---------------------------------------------------------------------------


class TestNativeFeaturePipelineColumnDrift:
    def test_extra_col_is_dropped_with_warning(
        self, train_df: pd.DataFrame, y_train: pd.Series
    ) -> None:
        pipe = NativeFeaturePipeline()
        pipe.fit(train_df, y_train)
        extra_df = train_df.copy()
        extra_df["bonus"] = 99
        result, warnings = pipe.transform_with_warnings(extra_df)
        assert "bonus" not in result.columns
        assert len(warnings) == 1

    def test_missing_col_raises(
        self, train_df: pd.DataFrame, y_train: pd.Series
    ) -> None:
        pipe = NativeFeaturePipeline()
        pipe.fit(train_df, y_train)
        missing_df = train_df.drop(columns=["cat"])
        with pytest.raises(LizyMLError) as exc_info:
            pipe.transform(missing_df)
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID

    def test_no_drift_no_warnings(
        self, train_df: pd.DataFrame, y_train: pd.Series
    ) -> None:
        pipe = NativeFeaturePipeline()
        pipe.fit(train_df, y_train)
        _, warnings = pipe.transform_with_warnings(train_df)
        assert warnings == []


# ---------------------------------------------------------------------------
# NativeFeaturePipeline — state save/load
# ---------------------------------------------------------------------------


class TestNativeFeaturePipelineState:
    def test_state_roundtrip(self, train_df: pd.DataFrame, y_train: pd.Series) -> None:
        pipe = NativeFeaturePipeline()
        pipe.fit(train_df, y_train)
        state = pipe.get_state()
        pipe2 = NativeFeaturePipeline()
        pipe2.load_state(state)
        result1 = pipe.transform(train_df)
        result2 = pipe2.transform(train_df)
        pd.testing.assert_frame_equal(result1, result2)

    def test_state_contains_feature_names(
        self, train_df: pd.DataFrame, y_train: pd.Series
    ) -> None:
        pipe = NativeFeaturePipeline()
        pipe.fit(train_df, y_train)
        state = pipe.get_state()
        assert "feature_names" in state
        assert state["feature_names"] == list(train_df.columns)

    def test_transform_before_fit_raises(self) -> None:
        pipe = NativeFeaturePipeline()
        with pytest.raises(RuntimeError):
            pipe.transform(pd.DataFrame({"a": [1]}))


# ---------------------------------------------------------------------------
# Leakage test — validation data must NOT influence fit state
# ---------------------------------------------------------------------------


class TestNativeFeaturePipelineLeakage:
    def test_fit_on_train_only_no_leak_into_valid(self) -> None:
        """Ensure that validation categories do not appear in encoder state.

        This is the key leakage test: the pipeline must be fitted on train_idx
        only, and 'z' (which exists only in valid) must remain unseen.
        """
        train_df = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a", "b"])})
        valid_df = pd.DataFrame({"cat": pd.Categorical(["a", "z"])})
        y_train = pd.Series([0, 1, 0, 1])

        pipe = NativeFeaturePipeline(unseen_policy="error")
        pipe.fit(train_df, y_train)

        # "z" was never seen during fit — it must trigger the unseen policy
        with pytest.raises(LizyMLError) as exc_info:
            pipe.transform(valid_df)
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID
        assert "z" in str(exc_info.value.context)
