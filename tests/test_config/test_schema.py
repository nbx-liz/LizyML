"""Tests for Config schema validation and loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lizyml import Model
from lizyml.config.loader import (
    SUPPORTED_CONFIG_VERSIONS,
    load_config,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG: dict = {
    "config_version": 1,
    "task": "regression",
    "data": {"target": "y"},
    "split": {"method": "kfold"},
    "model": {"lgbm": {}},
}

_FULL_CONFIG: dict = {
    "config_version": 1,
    "task": "binary",
    "data": {
        "path": "data.csv",
        "target": "label",
        "time_col": "date",
        "group_col": "group",
    },
    "features": {"exclude": ["id"], "auto_categorical": True, "categorical": ["cat"]},
    "split": {"method": "kfold", "n_splits": 5, "random_state": 42},
    "model": {"lgbm": {"params": {"n_estimators": 100}}},
    "training": {
        "seed": 123,
        "early_stopping": {
            "enabled": True,
            "rounds": 50,
            "inner_valid": {"method": "holdout", "ratio": 0.1, "random_state": 42},
        },
    },
    "tuning": {
        "optuna": {
            "params": {"n_trials": 10, "direction": "minimize"},
            "space": {"learning_rate": [0.01, 0.05]},
        }
    },
    "evaluation": {"metrics": ["logloss", "auc"]},
    "calibration": {"method": "platt", "n_splits": 3},
}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_minimal_config_validates(self) -> None:
        cfg = load_config(_MINIMAL_CONFIG)
        assert cfg.config_version == 1
        assert cfg.task == "regression"

    def test_full_config_validates(self) -> None:
        cfg = load_config(_FULL_CONFIG)
        assert cfg.task == "binary"
        assert cfg.calibration is not None
        assert cfg.tuning is not None

    def test_unknown_top_level_key_raises_config_invalid(self) -> None:
        raw = {**_MINIMAL_CONFIG, "typo_key": "value"}
        with pytest.raises(LizyMLError) as exc_info:
            load_config(raw)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_unknown_nested_key_raises_config_invalid(self) -> None:
        raw = {**_MINIMAL_CONFIG, "data": {"target": "y", "unknown_field": True}}
        with pytest.raises(LizyMLError) as exc_info:
            load_config(raw)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_missing_config_version_raises(self) -> None:
        raw = {k: v for k, v in _MINIMAL_CONFIG.items() if k != "config_version"}
        with pytest.raises(LizyMLError) as exc_info:
            load_config(raw)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_missing_target_raises(self) -> None:
        raw = {**_MINIMAL_CONFIG, "data": {}}
        with pytest.raises(LizyMLError) as exc_info:
            load_config(raw)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_invalid_task_raises(self) -> None:
        raw = {**_MINIMAL_CONFIG, "task": "unsupported_task"}
        with pytest.raises(LizyMLError) as exc_info:
            load_config(raw)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_all_tasks_are_valid(self) -> None:
        for task in ("regression", "binary", "multiclass"):
            cfg = load_config({**_MINIMAL_CONFIG, "task": task})
            assert cfg.task == task

    def test_defaults_are_applied(self) -> None:
        cfg = load_config(_MINIMAL_CONFIG)
        assert cfg.features.auto_categorical is True
        assert cfg.features.exclude == []
        assert cfg.training.seed == 42
        assert cfg.tuning is None
        assert cfg.calibration is None


# ---------------------------------------------------------------------------
# Alias normalization
# ---------------------------------------------------------------------------


class TestAliasNormalization:
    @pytest.mark.parametrize(
        "alias",
        ["kfold", "k-fold"],
    )
    def test_kfold_aliases(self, alias: str) -> None:
        raw = {**_MINIMAL_CONFIG, "split": {"method": alias}}
        cfg = load_config(raw)
        assert cfg.split.method == "kfold"

    @pytest.mark.parametrize(
        "alias",
        ["stratified_kfold", "stratified-kfold", "stratifiedkfold"],
    )
    def test_stratified_kfold_aliases(self, alias: str) -> None:
        raw = {**_MINIMAL_CONFIG, "split": {"method": alias}}
        cfg = load_config(raw)
        assert cfg.split.method == "stratified_kfold"

    @pytest.mark.parametrize(
        "alias",
        ["time_series", "time-series", "timeseries"],
    )
    def test_time_series_aliases(self, alias: str) -> None:
        raw = {**_MINIMAL_CONFIG, "split": {"method": alias}}
        cfg = load_config(raw)
        assert cfg.split.method == "time_series"


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


class TestFileLoading:
    def test_load_from_json(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text(json.dumps(_MINIMAL_CONFIG))
        cfg = load_config(p)
        assert cfg.config_version == 1

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        import yaml

        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(_MINIMAL_CONFIG))
        cfg = load_config(p)
        assert cfg.config_version == 1

    def test_load_from_yml(self, tmp_path: Path) -> None:
        import yaml

        p = tmp_path / "config.yml"
        p.write_text(yaml.dump(_MINIMAL_CONFIG))
        cfg = load_config(p)
        assert cfg.task == "regression"

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "config.toml"
        p.write_text("[project]\nname = 'x'")
        with pytest.raises(LizyMLError) as exc_info:
            load_config(p)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.yaml"
        with pytest.raises(LizyMLError) as exc_info:
            load_config(p)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        with pytest.raises(LizyMLError) as exc_info:
            load_config(p)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID


# ---------------------------------------------------------------------------
# Environment variable override
# ---------------------------------------------------------------------------


class TestEnvOverride:
    def test_override_lgbm_param(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIZYML__model__lgbm__params__learning_rate", "0.01")
        cfg = load_config(_MINIMAL_CONFIG)
        assert cfg.model.params.get("learning_rate") == 0.01  # type: ignore[union-attr]

    def test_override_training_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIZYML__training__seed", "999")
        cfg = load_config(_MINIMAL_CONFIG)
        assert cfg.training.seed == 999

    def test_env_bool_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIZYML__features__auto_categorical", "false")
        cfg = load_config(_MINIMAL_CONFIG)
        assert cfg.features.auto_categorical is False


# ---------------------------------------------------------------------------
# T-2: Config version compatibility gate
# ---------------------------------------------------------------------------


class TestConfigVersionGate:
    def test_supported_versions_constant_is_nonempty(self) -> None:
        """SUPPORTED_CONFIG_VERSIONS must be a non-empty list so the gate is active."""
        assert isinstance(SUPPORTED_CONFIG_VERSIONS, list)
        assert len(SUPPORTED_CONFIG_VERSIONS) > 0

    def test_version_1_is_supported(self) -> None:
        cfg = load_config(_MINIMAL_CONFIG)
        assert cfg.config_version == 1

    def test_missing_config_version_raises_config_invalid(self) -> None:
        """Missing config_version → CONFIG_INVALID (Pydantic required field)."""
        raw = {k: v for k, v in _MINIMAL_CONFIG.items() if k != "config_version"}
        with pytest.raises(LizyMLError) as exc_info:
            load_config(raw)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_unsupported_config_version_raises_version_unsupported(self) -> None:
        """config_version=999 must raise CONFIG_VERSION_UNSUPPORTED."""
        raw = {**_MINIMAL_CONFIG, "config_version": 999}
        with pytest.raises(LizyMLError) as exc_info:
            load_config(raw)
        assert exc_info.value.code == ErrorCode.CONFIG_VERSION_UNSUPPORTED

    def test_unsupported_version_context_includes_version(self) -> None:
        """Error context must expose the offending version for debugging."""
        raw = {**_MINIMAL_CONFIG, "config_version": 42}
        with pytest.raises(LizyMLError) as exc_info:
            load_config(raw)
        assert exc_info.value.context.get("config_version") == 42

    def test_model_init_with_unsupported_version_raises(self) -> None:
        """Facade must propagate CONFIG_VERSION_UNSUPPORTED to callers."""
        raw = {**_MINIMAL_CONFIG, "config_version": 999}
        with pytest.raises(LizyMLError) as exc_info:
            Model(raw)
        assert exc_info.value.code == ErrorCode.CONFIG_VERSION_UNSUPPORTED


# ---------------------------------------------------------------------------
# H-0010: validation_ratio shorthand
# ---------------------------------------------------------------------------


class TestValidationRatio:
    def test_validation_ratio_sets_inner_valid(self) -> None:
        """validation_ratio=0.2 → inner_valid.ratio == 0.2 (H-0010)."""
        raw = {
            **_MINIMAL_CONFIG,
            "training": {
                "early_stopping": {
                    "enabled": True,
                    "rounds": 50,
                    "validation_ratio": 0.2,
                }
            },
        }
        cfg = load_config(raw)
        es = cfg.training.early_stopping
        assert es.inner_valid is not None
        assert es.inner_valid.ratio == pytest.approx(0.2)
        assert es.inner_valid.method == "holdout"

    def test_validation_ratio_and_inner_valid_conflict_raises(self) -> None:
        """Both validation_ratio and inner_valid specified → CONFIG_INVALID."""
        raw = {
            **_MINIMAL_CONFIG,
            "training": {
                "early_stopping": {
                    "enabled": True,
                    "validation_ratio": 0.1,
                    "inner_valid": {"method": "holdout", "ratio": 0.2},
                }
            },
        }
        with pytest.raises(LizyMLError) as exc_info:
            load_config(raw)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_inner_valid_backward_compat(self) -> None:
        """Existing inner_valid form continues to work unchanged."""
        raw = {
            **_MINIMAL_CONFIG,
            "training": {
                "early_stopping": {
                    "enabled": True,
                    "rounds": 30,
                    "inner_valid": {"method": "holdout", "ratio": 0.15},
                }
            },
        }
        cfg = load_config(raw)
        es = cfg.training.early_stopping
        assert es.inner_valid is not None
        assert es.inner_valid.ratio == pytest.approx(0.15)
        assert es.validation_ratio == 0.1  # default, but inner_valid takes precedence
