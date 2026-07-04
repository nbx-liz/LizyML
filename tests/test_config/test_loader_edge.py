"""Edge-case coverage for config/loader.py (model normalization + env overrides)."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from lizyml.config.loader import (
    _apply_env_overrides,
    _coerce_env_value,
    _normalize_model_config,
    _read_raw,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError


class TestConfigLoader:
    def test_normalize_model_not_dict(self) -> None:
        result = _normalize_model_config({"model": "not_a_dict", "task": "regression"})
        assert result["model"] == "not_a_dict"

    def test_normalize_model_env_merge(self) -> None:
        raw = {
            "model": {
                "name": "lgbm",
                "params": {"learning_rate": 0.01},
                "lgbm": {"params": {"n_estimators": 100}, "auto_num_leaves": True},
            }
        }
        model = _normalize_model_config(raw)["model"]
        assert model["name"] == "lgbm"
        assert model["params"]["n_estimators"] == 100
        assert model["params"]["learning_rate"] == 0.01
        assert model["auto_num_leaves"] is True
        assert "lgbm" not in model

    def test_normalize_model_no_match(self) -> None:
        raw = {"model": {"unknown_model": {"params": {}}}}
        result = _normalize_model_config(raw)
        assert result["model"] == {"unknown_model": {"params": {}}}

    def test_env_override_empty_path(self) -> None:
        with patch.dict(os.environ, {"LIZYML__": "value"}):
            result = _apply_env_overrides({"task": "regression"})
        assert result == {"task": "regression"}

    def test_env_coerce_float(self) -> None:
        assert _coerce_env_value("1.5") == 1.5
        assert _coerce_env_value("abc") == "abc"

    def test_env_override_non_dict_node(self) -> None:
        with patch.dict(os.environ, {"LIZYML__TASK__NESTED": "value"}):
            result = _apply_env_overrides({"task": "regression"})
        assert result["task"] == "regression"

    def test_load_yaml_not_dict(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("- item1\n- item2\n")
            f.flush()
            with pytest.raises(LizyMLError) as exc_info:
                _read_raw(f.name)
            assert exc_info.value.code == ErrorCode.CONFIG_INVALID
            os.unlink(f.name)
