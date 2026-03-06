"""Tests for Confusion Matrix table (H-0016)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bin_df(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    return df


def _multi_df(n: int = 200, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = pd.cut(df["feat_a"], bins=3, labels=[0, 1, 2]).astype(int)
    return df


def _reg_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = df["feat_a"] * 2.0 + rng.normal(0, 0.1, n)
    return df


def _base_cfg(task: str) -> dict:
    return {
        "config_version": 1,
        "task": task,
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfusionMatrixBinary:
    def test_returns_dict_with_is_oos(self) -> None:
        m = Model(_base_cfg("binary"))
        m.fit(data=_bin_df())
        result = m.confusion_matrix()
        assert "is" in result
        assert "oos" in result

    def test_oos_shape_binary(self) -> None:
        m = Model(_base_cfg("binary"))
        m.fit(data=_bin_df())
        result = m.confusion_matrix()
        assert result["oos"].shape == (2, 2)

    def test_is_shape_binary(self) -> None:
        m = Model(_base_cfg("binary"))
        m.fit(data=_bin_df())
        result = m.confusion_matrix()
        assert result["is"].shape == (2, 2)

    def test_is_dataframe(self) -> None:
        m = Model(_base_cfg("binary"))
        m.fit(data=_bin_df())
        result = m.confusion_matrix()
        assert isinstance(result["oos"], pd.DataFrame)
        assert isinstance(result["is"], pd.DataFrame)

    def test_threshold_changes_result(self) -> None:
        m = Model(_base_cfg("binary"))
        m.fit(data=_bin_df())
        result_low = m.confusion_matrix(threshold=0.1)
        result_high = m.confusion_matrix(threshold=0.9)
        # Different thresholds should produce different confusion matrices
        assert not result_low["oos"].equals(result_high["oos"])


class TestConfusionMatrixMulticlass:
    def test_shape_multiclass(self) -> None:
        m = Model(_base_cfg("multiclass"))
        m.fit(data=_multi_df())
        result = m.confusion_matrix()
        assert result["oos"].shape == (3, 3)
        assert result["is"].shape == (3, 3)


class TestConfusionMatrixErrors:
    def test_regression_raises(self) -> None:
        m = Model(_base_cfg("regression"))
        m.fit(data=_reg_df())
        with pytest.raises(LizyMLError) as exc_info:
            m.confusion_matrix()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_before_fit_raises(self) -> None:
        m = Model(_base_cfg("binary"))
        with pytest.raises(LizyMLError) as exc_info:
            m.confusion_matrix()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT
