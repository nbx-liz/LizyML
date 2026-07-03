"""Tests for codegen generator — orchestrates full codegen export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.codegen.generator import generate_code

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def run_meta() -> dict[str, Any]:
    return {
        "lizyml_version": "0.2.0",
        "run_id": "gen-test-id",
        "timestamp": "2026-03-19T12:00:00",
        "config_normalized": {
            "task": "binary",
            "data": {"target_col": "y"},
        },
    }


@pytest.fixture()
def fitted_adapter() -> Any:
    from lizyml.estimators.lgbm.adapter import LGBMAdapter

    adapter = LGBMAdapter(task="binary", params={})
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    y = pd.Series(rng.integers(0, 2, size=100).astype(float))
    adapter.fit(X, y)
    return adapter


@pytest.fixture()
def pipeline_state() -> dict[str, Any]:
    return {
        "feature_names": ["a", "b"],
        "categorical_cols": [],
        "encoder": {},
        "transformer": {},
    }


# ── Tests ────────────────────────────────────────────────────


class TestGenerateCode:
    def test_creates_all_files(
        self,
        tmp_path: Path,
        run_meta: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        generate_code(
            output_dir=tmp_path,
            run_meta=run_meta,
            feature_names=["a", "b"],
            categorical_features=[],
            lgbm_params={"objective": "binary"},
            num_boost_round=100,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=None,
        )

        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "train.py").exists()
        assert (tmp_path / "predict.py").exists()
        assert (tmp_path / "test_equivalence.py").exists()
        assert (tmp_path / "requirements.txt").exists()
        assert (tmp_path / "artifacts" / "model.txt").exists()
        assert (tmp_path / "artifacts" / "pipeline_state.json").exists()

    def test_train_py_is_valid_python(
        self,
        tmp_path: Path,
        run_meta: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        import ast

        generate_code(
            output_dir=tmp_path,
            run_meta=run_meta,
            feature_names=["a", "b"],
            categorical_features=[],
            lgbm_params={"objective": "binary"},
            num_boost_round=100,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=None,
        )

        train_src = (tmp_path / "train.py").read_text()
        predict_src = (tmp_path / "predict.py").read_text()
        ast.parse(train_src)
        ast.parse(predict_src)

    def test_config_json_content(
        self,
        tmp_path: Path,
        run_meta: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        generate_code(
            output_dir=tmp_path,
            run_meta=run_meta,
            feature_names=["a", "b"],
            categorical_features=[],
            lgbm_params={"objective": "binary", "num_leaves": 31},
            num_boost_round=500,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method="platt",
            calibration_n_splits=5,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=None,
        )

        with open(tmp_path / "config.json") as f:
            cfg = json.load(f)

        assert cfg["_task"] == "binary"
        assert cfg["lgbm_params"]["num_leaves"] == 31
        assert cfg["calibration_method"] == "platt"

    def test_with_platt_calibrator(
        self,
        tmp_path: Path,
        run_meta: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        from lizyml.calibration.platt import PlattCalibrator

        rng = np.random.default_rng(42)
        cal = PlattCalibrator()
        cal.fit(rng.normal(size=100), rng.integers(0, 2, size=100).astype(float))

        generate_code(
            output_dir=tmp_path,
            run_meta=run_meta,
            feature_names=["a", "b"],
            categorical_features=[],
            lgbm_params={"objective": "binary"},
            num_boost_round=100,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method="platt",
            calibration_n_splits=5,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=cal,
        )

        cal_path = tmp_path / "artifacts" / "calibrator.json"
        assert cal_path.exists()
        with open(cal_path) as f:
            params = json.load(f)
        assert params["method"] == "platt"

    def test_returns_output_path(
        self,
        tmp_path: Path,
        run_meta: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        result = generate_code(
            output_dir=tmp_path,
            run_meta=run_meta,
            feature_names=["a"],
            categorical_features=[],
            lgbm_params={"objective": "binary"},
            num_boost_round=100,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=None,
        )
        assert result == tmp_path

    def test_requirements_txt_content(
        self,
        tmp_path: Path,
        run_meta: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        generate_code(
            output_dir=tmp_path,
            run_meta=run_meta,
            feature_names=["a"],
            categorical_features=[],
            lgbm_params={"objective": "binary"},
            num_boost_round=100,
            early_stopping_rounds=50,
            validation_ratio=0.2,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=None,
        )

        reqs = (tmp_path / "requirements.txt").read_text()
        assert "lightgbm" in reqs
        assert "numpy" in reqs
        assert "pandas" in reqs
        # No beta calibration here -> scipy must not be pinned (#218).
        assert "scipy" not in reqs
