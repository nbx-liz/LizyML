"""Tests for H-0021 — LGBMConfig smart parameter validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lizyml.config.schema import LGBMConfig


class TestAutoNumLeavesConflict:
    def test_auto_num_leaves_with_params_num_leaves_raises(self) -> None:
        with pytest.raises(ValidationError, match="auto_num_leaves"):
            LGBMConfig(
                name="lgbm",
                auto_num_leaves=True,
                params={"num_leaves": 64},
            )

    def test_auto_num_leaves_false_with_params_num_leaves_ok(self) -> None:
        cfg = LGBMConfig(
            name="lgbm",
            auto_num_leaves=False,
            params={"num_leaves": 64},
        )
        assert cfg.params["num_leaves"] == 64


class TestRatioConflict:
    def test_min_data_in_leaf_ratio_with_absolute_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_data_in_leaf"):
            LGBMConfig(
                name="lgbm",
                min_data_in_leaf_ratio=0.01,
                params={"min_data_in_leaf": 100},
            )

    def test_min_data_in_bin_ratio_with_absolute_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_data_in_bin"):
            LGBMConfig(
                name="lgbm",
                min_data_in_bin_ratio=0.01,
                params={"min_data_in_bin": 10},
            )


class TestNumLeavesRatioRange:
    def test_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="num_leaves_ratio"):
            LGBMConfig(name="lgbm", num_leaves_ratio=0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="num_leaves_ratio"):
            LGBMConfig(name="lgbm", num_leaves_ratio=-0.5)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="num_leaves_ratio"):
            LGBMConfig(name="lgbm", num_leaves_ratio=1.5)

    def test_one_is_valid(self) -> None:
        cfg = LGBMConfig(name="lgbm", num_leaves_ratio=1.0)
        assert cfg.num_leaves_ratio == 1.0


class TestMinDataInLeafRatioRange:
    def test_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_data_in_leaf_ratio"):
            LGBMConfig(name="lgbm", min_data_in_leaf_ratio=0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_data_in_leaf_ratio"):
            LGBMConfig(name="lgbm", min_data_in_leaf_ratio=-0.1)

    def test_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_data_in_leaf_ratio"):
            LGBMConfig(name="lgbm", min_data_in_leaf_ratio=1.0)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_data_in_leaf_ratio"):
            LGBMConfig(name="lgbm", min_data_in_leaf_ratio=1.5)

    def test_valid_values(self) -> None:
        for val in [0.01, 0.2, 0.99]:
            cfg = LGBMConfig(name="lgbm", min_data_in_leaf_ratio=val)
            assert cfg.min_data_in_leaf_ratio == val


class TestMinDataInBinRatioRange:
    def test_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_data_in_bin_ratio"):
            LGBMConfig(name="lgbm", min_data_in_bin_ratio=0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_data_in_bin_ratio"):
            LGBMConfig(name="lgbm", min_data_in_bin_ratio=-0.1)

    def test_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_data_in_bin_ratio"):
            LGBMConfig(name="lgbm", min_data_in_bin_ratio=1.0)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_data_in_bin_ratio"):
            LGBMConfig(name="lgbm", min_data_in_bin_ratio=1.5)

    def test_valid_values(self) -> None:
        for val in [0.01, 0.2, 0.99]:
            cfg = LGBMConfig(name="lgbm", min_data_in_bin_ratio=val)
            assert cfg.min_data_in_bin_ratio == val


class TestFeatureWeightsValidation:
    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValidationError, match="feature_weights"):
            LGBMConfig(name="lgbm", feature_weights={"a": -1.0})

    def test_zero_weight_raises(self) -> None:
        with pytest.raises(ValidationError, match="feature_weights"):
            LGBMConfig(name="lgbm", feature_weights={"a": 0.0})

    def test_valid_weights(self) -> None:
        cfg = LGBMConfig(name="lgbm", feature_weights={"a": 2.0, "b": 0.5})
        assert cfg.feature_weights == {"a": 2.0, "b": 0.5}
