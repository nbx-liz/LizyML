"""Tests for H-0021 / H-0050 — resolve_smart_params logic."""

from __future__ import annotations

import pandas as pd
import pytest

from lizyml.config.schema import LGBMConfig
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.lgbm import (
    extract_smart_params,
    resolve_ratio_params,
    resolve_smart_params,
)


def _base_config(**kwargs: object) -> LGBMConfig:
    defaults = {"name": "lgbm", "auto_num_leaves": False}
    defaults.update(kwargs)
    return LGBMConfig(**defaults)  # type: ignore[arg-type]


def _smart(**kwargs: object) -> dict:
    """Build a smart params dict with sensible defaults."""
    defaults: dict = {
        "auto_num_leaves": False,
        "num_leaves_ratio": 1.0,
        "min_data_in_leaf_ratio": None,
        "min_data_in_bin_ratio": None,
        "feature_weights": None,
        "balanced": None,
    }
    defaults.update(kwargs)
    return defaults


class TestExtractSmartParams:
    def test_round_trip(self) -> None:
        cfg = LGBMConfig(
            name="lgbm",
            auto_num_leaves=True,
            num_leaves_ratio=0.8,
            min_data_in_leaf_ratio=0.02,
            min_data_in_bin_ratio=0.03,
            feature_weights={"a": 2.0},
            balanced=True,
        )
        smart = extract_smart_params(cfg)
        assert smart["auto_num_leaves"] is True
        assert smart["num_leaves_ratio"] == 0.8
        assert smart["min_data_in_leaf_ratio"] == 0.02
        assert smart["min_data_in_bin_ratio"] == 0.03
        assert smart["feature_weights"] == {"a": 2.0}
        assert smart["balanced"] is True

    def test_defaults(self) -> None:
        cfg = LGBMConfig(name="lgbm")
        smart = extract_smart_params(cfg)
        assert smart["auto_num_leaves"] is True
        assert smart["num_leaves_ratio"] == 1.0
        assert smart["balanced"] is None


class TestAutoNumLeaves:
    def test_with_max_depth_5(self) -> None:
        smart = _smart(auto_num_leaves=True, num_leaves_ratio=1.0)
        resolved, _ = resolve_smart_params(
            smart, {"max_depth": 5}, 1000, ["a"], pd.Series([0]), "regression"
        )
        assert resolved["num_leaves"] == 32  # 2^5 * 1.0

    def test_with_ratio(self) -> None:
        smart = _smart(auto_num_leaves=True, num_leaves_ratio=0.5)
        resolved, _ = resolve_smart_params(
            smart, {"max_depth": 5}, 1000, ["a"], pd.Series([0]), "regression"
        )
        assert resolved["num_leaves"] == 16  # ceil(32 * 0.5)

    def test_no_max_depth(self) -> None:
        smart = _smart(auto_num_leaves=True)
        resolved, _ = resolve_smart_params(
            smart, {}, 1000, ["a"], pd.Series([0]), "regression"
        )
        assert resolved["num_leaves"] == 131072

    def test_negative_max_depth(self) -> None:
        smart = _smart(auto_num_leaves=True)
        resolved, _ = resolve_smart_params(
            smart, {"max_depth": -1}, 1000, ["a"], pd.Series([0]), "regression"
        )
        assert resolved["num_leaves"] == 131072

    def test_clamp_min(self) -> None:
        smart = _smart(auto_num_leaves=True, num_leaves_ratio=0.01)
        resolved, _ = resolve_smart_params(
            smart, {"max_depth": 2}, 1000, ["a"], pd.Series([0]), "regression"
        )
        assert resolved["num_leaves"] == 8  # clamp to min 8


class TestRatioParams:
    """Test resolve_ratio_params (H-0036: per-fold resolution)."""

    def test_min_data_in_leaf_ratio(self) -> None:
        resolved = resolve_ratio_params(0.01, None, 10000)
        assert resolved["min_data_in_leaf"] == 100

    def test_min_data_in_bin_ratio(self) -> None:
        resolved = resolve_ratio_params(None, 0.005, 10000)
        assert resolved["min_data_in_bin"] == 50

    def test_min_at_least_one(self) -> None:
        resolved = resolve_ratio_params(0.001, None, 10)
        assert resolved["min_data_in_leaf"] == 1

    def test_both_ratios(self) -> None:
        resolved = resolve_ratio_params(0.01, 0.005, 10000)
        assert resolved["min_data_in_leaf"] == 100
        assert resolved["min_data_in_bin"] == 50

    def test_none_returns_empty(self) -> None:
        resolved = resolve_ratio_params(None, None, 10000)
        assert resolved == {}

    def test_resolve_smart_params_excludes_ratios(self) -> None:
        """resolve_smart_params no longer resolves ratio params (H-0036)."""
        smart = _smart(min_data_in_leaf_ratio=0.01, min_data_in_bin_ratio=0.01)
        resolved, _ = resolve_smart_params(
            smart, {}, 10000, ["a"], pd.Series([0]), "regression"
        )
        assert "min_data_in_leaf" not in resolved
        assert "min_data_in_bin" not in resolved


class TestFeatureWeights:
    def test_dict_to_list(self) -> None:
        smart = _smart(feature_weights={"a": 2.0})
        resolved, _ = resolve_smart_params(
            smart, {}, 100, ["a", "b", "c"], pd.Series([0]), "regression"
        )
        assert resolved["feature_weights"] == [2.0, 1.0, 1.0]
        assert resolved["feature_pre_filter"] is False

    def test_unknown_feature_raises(self) -> None:
        smart = _smart(feature_weights={"unknown": 1.0})
        with pytest.raises(LizyMLError) as exc_info:
            resolve_smart_params(
                smart, {}, 100, ["a", "b"], pd.Series([0]), "regression"
            )
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID


class TestBalanced:
    def test_binary_scale_pos_weight(self) -> None:
        smart = _smart(balanced=True)
        y = pd.Series([0, 0, 0, 1])
        resolved, sw = resolve_smart_params(smart, {}, 4, ["a"], y, "binary")
        assert resolved["scale_pos_weight"] == 3.0
        assert sw is None

    def test_multiclass_sample_weight(self) -> None:
        smart = _smart(balanced=True)
        y = pd.Series([0, 0, 0, 1, 2])
        resolved, sw = resolve_smart_params(smart, {}, 5, ["a"], y, "multiclass")
        assert sw is not None
        assert len(sw) == 5

    def test_regression_raises(self) -> None:
        smart = _smart(balanced=True)
        with pytest.raises(LizyMLError) as exc_info:
            resolve_smart_params(smart, {}, 100, ["a"], pd.Series([0]), "regression")
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK


class TestResolveSmartFromDict:
    """Test that resolve_smart_params works with plain dicts (H-0050).

    These tests replace the old resolve_smart_params_from_dict tests,
    verifying that the unified function handles dict inputs correctly.
    """

    def test_num_leaves_ratio(self) -> None:
        smart = {"auto_num_leaves": True, "num_leaves_ratio": 0.5}
        resolved, _ = resolve_smart_params(
            smart, {"max_depth": 5}, 1000, ["a"], pd.Series([0]), "regression"
        )
        assert resolved["num_leaves"] == 16  # ceil(32 * 0.5)

    def test_excludes_ratios(self) -> None:
        """resolve_smart_params does not resolve ratio params (H-0036)."""
        smart = {"min_data_in_leaf_ratio": 0.05}
        resolved, _ = resolve_smart_params(
            smart, {}, 1000, ["a"], pd.Series([0]), "regression"
        )
        assert "min_data_in_leaf" not in resolved

    def test_empty_returns_empty(self) -> None:
        resolved, _ = resolve_smart_params(
            {}, {}, 1000, ["a"], pd.Series([0]), "regression"
        )
        assert resolved == {}
