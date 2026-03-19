"""Tests for codegen config_writer — generates config.json for exported code."""

from __future__ import annotations

import json
from typing import Any

from lizyml.codegen.config_writer import build_config

# ── Helpers ──────────────────────────────────────────────────


def _make_run_meta(
    *,
    task: str = "binary",
    target_col: str = "y",
    lizyml_version: str = "0.2.0",
    run_id: str = "test-run-id",
    timestamp: str = "2026-03-19T12:00:00",
) -> dict[str, Any]:
    """Minimal RunMeta-like dict for config_writer tests."""
    return {
        "lizyml_version": lizyml_version,
        "run_id": run_id,
        "timestamp": timestamp,
        "config_normalized": {
            "task": task,
            "data": {"target_col": target_col},
        },
    }


def _make_lgbm_params() -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbosity": -1,
    }


# ── Tests ────────────────────────────────────────────────────


class TestBuildConfig:
    def test_returns_dict(self) -> None:
        cfg = build_config(
            run_meta=_make_run_meta(),
            feature_names=["a", "b"],
            categorical_features=["b"],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=1000,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
        )
        assert isinstance(cfg, dict)

    def test_meta_fields_prefixed(self) -> None:
        cfg = build_config(
            run_meta=_make_run_meta(
                lizyml_version="0.2.0",
                run_id="abc-123",
                timestamp="2026-01-01T00:00:00",
            ),
            feature_names=["x"],
            categorical_features=[],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=500,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
        )
        assert cfg["_generated_by"] == "lizyml 0.2.0"
        assert cfg["_run_id"] == "abc-123"
        assert cfg["_timestamp"] == "2026-01-01T00:00:00"

    def test_task_and_target(self) -> None:
        cfg = build_config(
            run_meta=_make_run_meta(task="regression", target_col="price"),
            feature_names=["x"],
            categorical_features=[],
            lgbm_params={"objective": "huber"},
            num_boost_round=100,
            early_stopping_rounds=None,
            validation_ratio=0.0,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
        )
        assert cfg["_task"] == "regression"
        assert cfg["_target_col"] == "price"

    def test_feature_names_and_categoricals(self) -> None:
        cfg = build_config(
            run_meta=_make_run_meta(),
            feature_names=["age", "income", "cat_a"],
            categorical_features=["cat_a"],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=1000,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
        )
        assert cfg["feature_names"] == ["age", "income", "cat_a"]
        assert cfg["categorical_features"] == ["cat_a"]

    def test_lgbm_params_stored(self) -> None:
        params = {"objective": "binary", "num_leaves": 63, "learning_rate": 0.1}
        cfg = build_config(
            run_meta=_make_run_meta(),
            feature_names=["x"],
            categorical_features=[],
            lgbm_params=params,
            num_boost_round=800,
            early_stopping_rounds=100,
            validation_ratio=0.15,
            seed=99,
            calibration_method=None,
            calibration_n_splits=5,
        )
        assert cfg["lgbm_params"] == params
        assert cfg["num_boost_round"] == 800
        assert cfg["early_stopping_rounds"] == 100
        assert cfg["validation_ratio"] == 0.15
        assert cfg["seed"] == 99

    def test_calibration_method_binary(self) -> None:
        cfg = build_config(
            run_meta=_make_run_meta(task="binary"),
            feature_names=["x"],
            categorical_features=[],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=500,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method="platt",
            calibration_n_splits=5,
        )
        assert cfg["calibration_method"] == "platt"
        assert cfg["calibration_n_splits"] == 5

    def test_calibration_none_when_no_calibrator(self) -> None:
        cfg = build_config(
            run_meta=_make_run_meta(task="binary"),
            feature_names=["x"],
            categorical_features=[],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=500,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
        )
        assert cfg["calibration_method"] is None

    def test_json_serializable(self) -> None:
        cfg = build_config(
            run_meta=_make_run_meta(),
            feature_names=["x", "y_feat"],
            categorical_features=["y_feat"],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=1000,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method="isotonic",
            calibration_n_splits=3,
        )
        # Must not raise
        text = json.dumps(cfg)
        roundtripped = json.loads(text)
        assert roundtripped == cfg

    def test_key_ordering(self) -> None:
        """Meta keys first, then feature, then lgbm, then calibration."""
        cfg = build_config(
            run_meta=_make_run_meta(),
            feature_names=["x"],
            categorical_features=[],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=1000,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method="platt",
            calibration_n_splits=5,
        )
        keys = list(cfg.keys())
        # Meta keys should come before feature keys
        assert keys.index("_generated_by") < keys.index("feature_names")
        # Feature keys before lgbm
        assert keys.index("feature_names") < keys.index("lgbm_params")
        # lgbm before calibration
        assert keys.index("lgbm_params") < keys.index("calibration_method")

    def test_multiclass_no_calibration_fields(self) -> None:
        """Multiclass should still include calibration keys (as None)."""
        cfg = build_config(
            run_meta=_make_run_meta(task="multiclass"),
            feature_names=["x"],
            categorical_features=[],
            lgbm_params={"objective": "multiclass"},
            num_boost_round=500,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
        )
        assert cfg["calibration_method"] is None
        assert cfg["calibration_n_splits"] == 5
