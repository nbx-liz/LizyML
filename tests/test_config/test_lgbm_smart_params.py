"""Tests for H-0021 — LGBMConfig smart parameter validation.

H-0121 moved the refusal of a ``params`` name an active smart parameter writes
from config construction to ``fit`` / ``tune`` (``Model._merge_params``), where
it is alias-aware and covers every layer. The three conflict tests below pin
that contract: the config constructs, and the fit refuses before training.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from lizyml import Model
from lizyml.config.schema import LGBMConfig
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_binary_df, make_config
from tests._train_spy import record_lightgbm_calls


def _assert_fit_refuses(smart: dict[str, Any], params: dict[str, Any]) -> None:
    cfg = make_config("binary", n_estimators=3, n_splits=2, **params)
    cfg["model"].update(smart)
    model = Model(cfg, data=make_binary_df(n=160))
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit()
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    for name in (*smart, *params, "model.params"):
        assert name in exc.value.user_message, exc.value.user_message
    assert not seen["train_params"]


class TestAutoNumLeavesConflict:
    def test_auto_num_leaves_with_params_num_leaves_is_refused_at_fit(self) -> None:
        cfg = LGBMConfig(
            name="lgbm",
            auto_num_leaves=True,
            params={"num_leaves": 64},
        )
        assert cfg.params["num_leaves"] == 64
        _assert_fit_refuses({"auto_num_leaves": True}, {"num_leaves": 64})

    def test_auto_num_leaves_false_with_params_num_leaves_ok(self) -> None:
        cfg = LGBMConfig(
            name="lgbm",
            auto_num_leaves=False,
            params={"num_leaves": 64},
        )
        assert cfg.params["num_leaves"] == 64


class TestRatioConflict:
    def test_min_data_in_leaf_ratio_with_absolute_is_refused_at_fit(self) -> None:
        cfg = LGBMConfig(
            name="lgbm",
            min_data_in_leaf_ratio=0.01,
            params={"min_data_in_leaf": 100},
        )
        assert cfg.params["min_data_in_leaf"] == 100
        _assert_fit_refuses({"min_data_in_leaf_ratio": 0.01}, {"min_data_in_leaf": 100})

    def test_min_data_in_bin_ratio_with_absolute_is_refused_at_fit(self) -> None:
        cfg = LGBMConfig(
            name="lgbm",
            min_data_in_bin_ratio=0.01,
            params={"min_data_in_bin": 10},
        )
        assert cfg.params["min_data_in_bin"] == 10
        _assert_fit_refuses({"min_data_in_bin_ratio": 0.01}, {"min_data_in_bin": 10})


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
