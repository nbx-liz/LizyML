"""Tests for codegen artifact_writer — writes config.json and artifacts/ to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.codegen.artifact_writer import write_artifacts

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def config_dict() -> dict[str, Any]:
    return {
        "_generated_by": "lizyml 0.2.0",
        "_run_id": "test-id",
        "_task": "binary",
        "_target_col": "y",
        "_timestamp": "2026-01-01T00:00:00",
        "feature_names": ["a", "b"],
        "categorical_features": ["b"],
        "lgbm_params": {"objective": "binary"},
        "num_boost_round": 100,
        "early_stopping_rounds": 50,
        "validation_ratio": 0.2,
        "seed": 42,
        "calibration_method": "platt",
        "calibration_n_splits": 5,
    }


@pytest.fixture()
def fitted_adapter() -> Any:
    """Minimal fitted LGBMAdapter mock."""
    from lizyml.estimators.lgbm.adapter import LGBMAdapter

    adapter = LGBMAdapter(task="regression", params={})
    X = pd.DataFrame({"a": np.random.default_rng(0).normal(size=50)})
    y = pd.Series(np.random.default_rng(0).normal(size=50))
    adapter.fit(X, y)
    return adapter


@pytest.fixture()
def pipeline_state() -> dict[str, Any]:
    return {
        "feature_names": ["a", "b"],
        "categorical_cols": ["b"],
        "encoder": {"categories": {"b": ["x", "y"]}},
        "transformer": {},
    }


# ── Tests ────────────────────────────────────────────────────


class TestWriteArtifacts:
    def test_creates_config_json(
        self,
        tmp_path: Path,
        config_dict: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        write_artifacts(
            output_dir=tmp_path,
            config=config_dict,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=None,
        )
        config_path = tmp_path / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            loaded = json.load(f)
        assert loaded["_generated_by"] == "lizyml 0.2.0"

    def test_creates_model_txt(
        self,
        tmp_path: Path,
        config_dict: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        write_artifacts(
            output_dir=tmp_path,
            config=config_dict,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=None,
        )
        model_path = tmp_path / "artifacts" / "model.txt"
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_creates_pipeline_state_json(
        self,
        tmp_path: Path,
        config_dict: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        write_artifacts(
            output_dir=tmp_path,
            config=config_dict,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=None,
        )
        state_path = tmp_path / "artifacts" / "pipeline_state.json"
        assert state_path.exists()
        with open(state_path) as f:
            loaded = json.load(f)
        assert loaded["feature_names"] == ["a", "b"]

    def test_no_calibrator_files_when_none(
        self,
        tmp_path: Path,
        config_dict: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        write_artifacts(
            output_dir=tmp_path,
            config=config_dict,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=None,
        )
        assert not (tmp_path / "artifacts" / "calibrator.json").exists()
        assert not (tmp_path / "artifacts" / "calibrator_model.txt").exists()

    def test_platt_calibrator_creates_json(
        self,
        tmp_path: Path,
        config_dict: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        from lizyml.calibration.platt import PlattCalibrator

        rng = np.random.default_rng(42)
        scores = rng.normal(size=100)
        y = rng.integers(0, 2, size=100).astype(float)
        cal = PlattCalibrator()
        cal.fit(scores, y)

        write_artifacts(
            output_dir=tmp_path,
            config=config_dict,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=cal,
        )

        cal_path = tmp_path / "artifacts" / "calibrator.json"
        assert cal_path.exists()
        with open(cal_path) as f:
            params = json.load(f)
        assert params["method"] == "platt"
        assert "a" in params and "b" in params

    def test_isotonic_calibrator_creates_model_file(
        self,
        tmp_path: Path,
        config_dict: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        from lizyml.calibration.isotonic import IsotonicCalibrator

        rng = np.random.default_rng(42)
        scores = rng.normal(size=200)
        y = rng.integers(0, 2, size=200).astype(float)
        cal = IsotonicCalibrator()
        cal.fit(scores, y)

        write_artifacts(
            output_dir=tmp_path,
            config=config_dict,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=cal,
        )

        cal_json = tmp_path / "artifacts" / "calibrator.json"
        cal_model = tmp_path / "artifacts" / "calibrator_model.txt"
        assert cal_json.exists()
        assert cal_model.exists()
        with open(cal_json) as f:
            params = json.load(f)
        assert params["method"] == "isotonic"
        assert params["model_file"] == "calibrator_model.txt"

    def test_output_dir_created_if_missing(
        self,
        tmp_path: Path,
        config_dict: dict,
        fitted_adapter: Any,
        pipeline_state: dict,
    ) -> None:
        out = tmp_path / "nonexistent" / "deep" / "path"
        write_artifacts(
            output_dir=out,
            config=config_dict,
            model_adapter=fitted_adapter,
            pipeline_state=pipeline_state,
            calibrator=None,
        )
        assert (out / "config.json").exists()
        assert (out / "artifacts" / "model.txt").exists()


class TestUnseenPolicyExport:
    """#205: pipeline_state.json must carry unseen_policy + per-column mode code
    so predict.py can reproduce the runtime unseen_policy='mode' behavior."""

    def test_convert_exports_unseen_policy_and_codes(self) -> None:
        from lizyml.codegen.artifact_writer import _convert_pipeline_state

        state = {
            "feature_names": ["a", "b"],
            "encoder": {
                "unseen_policy": "mode",
                "categories": {"b": ["x", "y", "z"]},
                "modes": {"b": "y"},
            },
        }
        out = _convert_pipeline_state(state, {"categorical_features": ["b"]})

        assert out["unseen_policy"] == "mode"
        assert out["category_mappings"]["b"] == {"x": 0, "y": 1, "z": 2}
        # mode "y" -> code 1 is the unseen replacement.
        assert out["unseen_codes"] == {"b": 1}
