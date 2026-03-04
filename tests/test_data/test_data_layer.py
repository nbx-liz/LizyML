"""Phase 3 data layer tests: datasource, dataframe_builder, validators, fingerprint."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.specs.feature_spec import FeatureSpec
from lizyml.core.specs.problem_spec import ProblemSpec
from lizyml.data import dataframe_builder, datasource, fingerprint, validators

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "feature_a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat_col": ["a", "b", "a", "c", "b"],
            "y": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture()
def problem_spec() -> ProblemSpec:
    return ProblemSpec(
        task="binary",
        target="y",
        time_col=None,
        group_col=None,
        data_path=None,
    )


@pytest.fixture()
def feature_spec() -> FeatureSpec:
    return FeatureSpec(exclude=("id",), auto_categorical=True, categorical=())


# ---------------------------------------------------------------------------
# DataSource
# ---------------------------------------------------------------------------


class TestDataSource:
    def test_read_dataframe_returns_copy(self, simple_df: pd.DataFrame) -> None:
        result = datasource.read(simple_df)
        assert result is not simple_df
        pd.testing.assert_frame_equal(result, simple_df)

    def test_read_csv(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "data.csv"
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df.to_csv(p, index=False)
        result = datasource.read(p)
        pd.testing.assert_frame_equal(result, df)

    def test_read_parquet(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "data.parquet"
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df.to_parquet(p, index=False)
        result = datasource.read(p)
        pd.testing.assert_frame_equal(result, df)

    def test_unsupported_format_raises(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "data.xlsx"
        p.write_text("dummy")
        with pytest.raises(LizyMLError) as exc_info:
            datasource.read(p)
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID

    def test_missing_file_raises(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "nonexistent.csv"
        with pytest.raises(LizyMLError) as exc_info:
            datasource.read(p)
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID


# ---------------------------------------------------------------------------
# DataFrameBuilder
# ---------------------------------------------------------------------------


class TestDataFrameBuilder:
    def test_basic_build(
        self,
        simple_df: pd.DataFrame,
        problem_spec: ProblemSpec,
        feature_spec: FeatureSpec,
    ) -> None:
        result = dataframe_builder.build(simple_df, problem_spec, feature_spec)
        assert "y" not in result.X.columns
        assert "id" not in result.X.columns
        assert list(result.y) == [0, 1, 0, 1, 0]

    def test_target_separated(
        self,
        simple_df: pd.DataFrame,
        problem_spec: ProblemSpec,
        feature_spec: FeatureSpec,
    ) -> None:
        result = dataframe_builder.build(simple_df, problem_spec, feature_spec)
        assert result.y.name == "y"
        assert len(result.y) == len(simple_df)

    def test_time_col_separated(self, simple_df: pd.DataFrame) -> None:
        df = simple_df.copy()
        df["date"] = pd.date_range("2024-01-01", periods=5)
        spec = ProblemSpec(
            task="regression",
            target="y",
            time_col="date",
            group_col=None,
            data_path=None,
        )
        feat_spec = FeatureSpec(exclude=(), auto_categorical=True, categorical=())
        result = dataframe_builder.build(df, spec, feat_spec)
        assert "date" not in result.X.columns
        assert result.time_col is not None

    def test_group_col_separated(self, simple_df: pd.DataFrame) -> None:
        df = simple_df.copy()
        df["group"] = ["g1", "g1", "g2", "g2", "g3"]
        spec = ProblemSpec(
            task="binary",
            target="y",
            time_col=None,
            group_col="group",
            data_path=None,
        )
        feat_spec = FeatureSpec(exclude=(), auto_categorical=True, categorical=())
        result = dataframe_builder.build(df, spec, feat_spec)
        assert "group" not in result.X.columns
        assert result.group_col is not None

    def test_auto_categorical_applied(
        self,
        simple_df: pd.DataFrame,
        problem_spec: ProblemSpec,
    ) -> None:
        feat_spec = FeatureSpec(exclude=("id",), auto_categorical=True, categorical=())
        result = dataframe_builder.build(simple_df, problem_spec, feat_spec)
        assert result.X["cat_col"].dtype.name == "category"

    def test_explicit_categorical_applied(
        self,
        simple_df: pd.DataFrame,
        problem_spec: ProblemSpec,
    ) -> None:
        feat_spec = FeatureSpec(
            exclude=("id",), auto_categorical=False, categorical=("cat_col",)
        )
        result = dataframe_builder.build(simple_df, problem_spec, feat_spec)
        assert result.X["cat_col"].dtype.name == "category"

    def test_missing_target_column_raises(self, simple_df: pd.DataFrame) -> None:
        spec = ProblemSpec(
            task="binary",
            target="nonexistent",
            time_col=None,
            group_col=None,
            data_path=None,
        )
        with pytest.raises(LizyMLError) as exc_info:
            dataframe_builder.build(simple_df, spec, FeatureSpec())
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID

    def test_missing_time_col_raises(self, simple_df: pd.DataFrame) -> None:
        spec = ProblemSpec(
            task="regression",
            target="y",
            time_col="no_such_col",
            group_col=None,
            data_path=None,
        )
        with pytest.raises(LizyMLError) as exc_info:
            dataframe_builder.build(simple_df, spec, FeatureSpec())
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID


# ---------------------------------------------------------------------------
# Validators — 「落ちるべき例」
# ---------------------------------------------------------------------------


class TestValidators:
    # Time series order

    def test_sorted_time_col_passes(self) -> None:
        df = pd.DataFrame({"date": [1, 2, 3, 4, 5], "x": range(5)})
        warnings = validators.validate_time_series_order(df, "date")
        assert warnings == []

    def test_unsorted_time_col_raises(self) -> None:
        df = pd.DataFrame({"date": [1, 3, 2, 4, 5], "x": range(5)})
        with pytest.raises(LizyMLError) as exc_info:
            validators.validate_time_series_order(df, "date", raise_on_violation=True)
        assert exc_info.value.code == ErrorCode.LEAKAGE_SUSPECTED

    def test_unsorted_time_col_returns_warning_when_not_raising(self) -> None:
        df = pd.DataFrame({"date": [1, 3, 2, 4, 5], "x": range(5)})
        warnings = validators.validate_time_series_order(
            df, "date", raise_on_violation=False
        )
        assert len(warnings) == 1

    # Target leakage

    def test_no_leakage_passes(self, simple_df: pd.DataFrame) -> None:
        warnings = validators.validate_no_target_leakage(simple_df, "y")
        assert warnings == []

    def test_target_duplicate_col_raises(self) -> None:
        df = pd.DataFrame({"y": [0, 1, 0], "y_copy": [0, 1, 0]})
        with pytest.raises(LizyMLError) as exc_info:
            validators.validate_no_target_leakage(df, "y", raise_on_violation=True)
        assert exc_info.value.code == ErrorCode.LEAKAGE_SUSPECTED

    def test_target_duplicate_col_returns_warning(self) -> None:
        df = pd.DataFrame({"y": [0, 1, 0], "y_copy": [0, 1, 0]})
        warnings = validators.validate_no_target_leakage(
            df, "y", raise_on_violation=False
        )
        assert len(warnings) == 1

    # Group split validation

    def test_clean_group_split_passes(self) -> None:
        groups = pd.Series(["g1", "g1", "g2", "g2", "g3"])
        train_idx = np.array([0, 1])
        valid_idx = np.array([2, 3, 4])
        warnings = validators.validate_group_split(groups, train_idx, valid_idx)
        assert warnings == []

    def test_overlapping_groups_raises(self) -> None:
        groups = pd.Series(["g1", "g1", "g1", "g2", "g2"])
        train_idx = np.array([0, 1])  # both g1
        valid_idx = np.array([2, 3, 4])  # also g1
        with pytest.raises(LizyMLError) as exc_info:
            validators.validate_group_split(
                groups, train_idx, valid_idx, raise_on_violation=True
            )
        assert exc_info.value.code == ErrorCode.LEAKAGE_CONFIRMED

    def test_overlapping_groups_returns_warning(self) -> None:
        groups = pd.Series(["g1", "g1", "g1", "g2", "g2"])
        train_idx = np.array([0, 1])
        valid_idx = np.array([2, 3, 4])
        warnings = validators.validate_group_split(
            groups, train_idx, valid_idx, raise_on_violation=False
        )
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# DataFingerprint
# ---------------------------------------------------------------------------


class TestDataFingerprint:
    def test_same_data_same_fingerprint(self, simple_df: pd.DataFrame) -> None:
        fp1 = fingerprint.compute(simple_df)
        fp2 = fingerprint.compute(simple_df)
        assert fp1 == fp2

    def test_different_row_count_different_fingerprint(
        self, simple_df: pd.DataFrame
    ) -> None:
        fp1 = fingerprint.compute(simple_df)
        fp2 = fingerprint.compute(simple_df.head(3))
        assert fp1 != fp2
        assert not fp1.matches(fp2)

    def test_added_column_different_fingerprint(self, simple_df: pd.DataFrame) -> None:
        fp1 = fingerprint.compute(simple_df)
        df2 = simple_df.copy()
        df2["extra"] = 0
        fp2 = fingerprint.compute(df2)
        assert fp1.column_hash != fp2.column_hash
        assert not fp1.matches(fp2)

    def test_file_hash_included_when_path_given(
        self, simple_df: pd.DataFrame, tmp_path: pathlib.Path
    ) -> None:
        p = tmp_path / "data.csv"
        simple_df.to_csv(p, index=False)
        fp = fingerprint.compute(simple_df, p)
        assert fp.file_hash is not None

    def test_different_files_different_file_hash(self, tmp_path: pathlib.Path) -> None:
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        p1 = tmp_path / "a.csv"
        p2 = tmp_path / "b.csv"
        df1.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)
        fp1 = fingerprint.compute(df1, p1)
        fp2 = fingerprint.compute(df2, p2)
        assert fp1.file_hash != fp2.file_hash

    def test_matches_ignores_file_hash_when_one_is_none(
        self, simple_df: pd.DataFrame
    ) -> None:
        fp1 = fingerprint.compute(simple_df)
        fp2 = fingerprint.compute(simple_df)
        assert fp1.file_hash is None
        assert fp2.file_hash is None
        assert fp1.matches(fp2)
