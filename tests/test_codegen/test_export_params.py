"""Tests for export_params() on calibrators, save_model_text, export_state_json."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from lizyml.calibration.isotonic import IsotonicCalibrator
from lizyml.calibration.platt import PlattCalibrator

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple binary classification OOF scores and labels."""
    rng = np.random.default_rng(42)
    n = 200
    y = rng.integers(0, 2, size=n).astype(np.float64)
    scores = y * 2 - 1 + rng.normal(0, 0.5, size=n)  # logits
    return scores, y


# ── PlattCalibrator.export_params() ───────────────────────────


class TestPlattExportParams:
    def test_export_params_returns_correct_keys(self, binary_data: tuple) -> None:
        scores, y = binary_data
        cal = PlattCalibrator()
        cal.fit(scores, y)
        params = cal.export_params()

        assert params["method"] == "platt"
        assert "a" in params
        assert "b" in params
        assert isinstance(params["a"], float)
        assert isinstance(params["b"], float)

    def test_export_params_roundtrip(self, binary_data: tuple) -> None:
        """Exported a, b reproduce the same calibration as predict()."""
        scores, y = binary_data
        cal = PlattCalibrator()
        cal.fit(scores, y)
        params = cal.export_params()

        # Reproduce with exported params
        test_scores = np.linspace(-3, 3, 50)
        expected = cal.predict(test_scores)
        reconstructed = 1.0 / (1.0 + np.exp(-(params["a"] * test_scores + params["b"])))

        np.testing.assert_allclose(reconstructed, expected, rtol=1e-7)

    def test_export_params_unfitted_raises(self) -> None:
        cal = PlattCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.export_params()

    def test_export_params_json_serializable(self, binary_data: tuple) -> None:
        scores, y = binary_data
        cal = PlattCalibrator()
        cal.fit(scores, y)
        params = cal.export_params()
        # Should not raise
        json.dumps(params)


# ── BetaCalibrator.export_params() ────────────────────────────


class TestBetaExportParams:
    def test_export_params_returns_correct_keys(self, binary_data: tuple) -> None:
        from lizyml.calibration.beta import BetaCalibrator

        scores, y = binary_data
        cal = BetaCalibrator()
        cal.fit(scores, y)
        params = cal.export_params()

        assert params["method"] == "beta"
        assert "a" in params
        assert "b" in params
        assert "c" in params
        assert all(isinstance(params[k], float) for k in ("a", "b", "c"))

    def test_export_params_roundtrip(self, binary_data: tuple) -> None:
        from lizyml.calibration.beta import BetaCalibrator, _sigmoid

        scores, y = binary_data
        cal = BetaCalibrator()
        cal.fit(scores, y)
        params = cal.export_params()

        test_scores = np.linspace(-3, 3, 50)
        expected = cal.predict(test_scores)

        s = np.clip(_sigmoid(test_scores), 1e-10, 1 - 1e-10)
        logit = params["a"] * np.log(s) + params["b"] * np.log(1 - s) + params["c"]
        reconstructed = np.clip(_sigmoid(logit), 0.0, 1.0)

        np.testing.assert_allclose(reconstructed, expected, rtol=1e-7)

    def test_export_params_unfitted_raises(self) -> None:
        from lizyml.calibration.beta import BetaCalibrator

        cal = BetaCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.export_params()


# ── IsotonicCalibrator.export_params() + save_model_text() ────


class TestIsotonicExportParams:
    def test_export_params_returns_correct_keys(self, binary_data: tuple) -> None:
        scores, y = binary_data
        cal = IsotonicCalibrator()
        cal.fit(scores, y)
        params = cal.export_params()

        assert params["method"] == "isotonic"

    def test_save_model_text_creates_file(
        self, binary_data: tuple, tmp_path: Path
    ) -> None:
        scores, y = binary_data
        cal = IsotonicCalibrator()
        cal.fit(scores, y)

        model_path = tmp_path / "cal_model.txt"
        cal.save_model_text(model_path)

        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_save_model_text_roundtrip(
        self, binary_data: tuple, tmp_path: Path
    ) -> None:
        """Booster text export/load produces same predictions."""
        scores, y = binary_data
        cal = IsotonicCalibrator()
        cal.fit(scores, y)

        model_path = tmp_path / "cal_model.txt"
        cal.save_model_text(model_path)

        # Load and predict
        loaded = lgb.Booster(model_file=str(model_path))
        test_scores = np.linspace(-3, 3, 50)
        expected = cal.predict(test_scores)
        reconstructed = np.clip(loaded.predict(test_scores.reshape(-1, 1)), 0.0, 1.0)

        np.testing.assert_allclose(reconstructed, expected, rtol=1e-7)

    def test_save_model_text_unfitted_raises(self, tmp_path: Path) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.save_model_text(tmp_path / "model.txt")

    def test_export_params_unfitted_raises(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.export_params()


# ── LGBMAdapter.save_model_text() ────────────────────────────


class TestLGBMAdapterSaveModelText:
    @pytest.fixture()
    def fitted_adapter(self) -> object:
        from lizyml.estimators.lgbm.adapter import LGBMAdapter

        adapter = LGBMAdapter(task="regression", params={})
        X = pd.DataFrame({"a": np.random.default_rng(0).normal(size=100)})
        y = pd.Series(np.random.default_rng(0).normal(size=100))
        adapter.fit(X, y)
        return adapter

    def test_creates_file(self, fitted_adapter: object, tmp_path: Path) -> None:
        path = tmp_path / "model.txt"
        fitted_adapter.save_model_text(path)  # type: ignore[attr-defined]
        assert path.exists()
        assert path.stat().st_size > 0

    def test_roundtrip_predictions(
        self, fitted_adapter: object, tmp_path: Path
    ) -> None:
        path = tmp_path / "model.txt"
        fitted_adapter.save_model_text(path)  # type: ignore[attr-defined]

        loaded = lgb.Booster(model_file=str(path))
        X = pd.DataFrame({"a": np.random.default_rng(99).normal(size=20)})
        expected = fitted_adapter.predict(X)  # type: ignore[attr-defined]
        actual = loaded.predict(X)

        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_unfitted_raises(self, tmp_path: Path) -> None:
        from lizyml.estimators.lgbm.adapter import LGBMAdapter

        adapter = LGBMAdapter(task="regression", params={})
        from lizyml.core.exceptions import LizyMLError

        with pytest.raises(LizyMLError):
            adapter.save_model_text(tmp_path / "model.txt")


# ── NativeFeaturePipeline.export_state_json() ────────────────


class TestPipelineExportStateJson:
    @pytest.fixture()
    def fitted_pipeline(self) -> object:
        from lizyml.features.pipelines_native import NativeFeaturePipeline

        pipe = NativeFeaturePipeline()
        X = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
        y = pd.Series([0, 1, 0])
        pipe.fit(X, y)
        return pipe

    def test_creates_json_file(self, fitted_pipeline: object, tmp_path: Path) -> None:
        path = tmp_path / "pipeline.json"
        fitted_pipeline.export_state_json(path)  # type: ignore[attr-defined]
        assert path.exists()

        with open(path) as f:
            state = json.load(f)
        assert "feature_names" in state

    def test_state_matches_get_state(
        self, fitted_pipeline: object, tmp_path: Path
    ) -> None:
        path = tmp_path / "pipeline.json"
        fitted_pipeline.export_state_json(path)  # type: ignore[attr-defined]

        with open(path) as f:
            exported = json.load(f)

        internal = fitted_pipeline.get_state()  # type: ignore[attr-defined]
        assert exported["feature_names"] == internal["feature_names"]
        assert exported["categorical_cols"] == internal["categorical_cols"]

    def test_unfitted_raises(self, tmp_path: Path) -> None:
        from lizyml.features.pipelines_native import NativeFeaturePipeline

        pipe = NativeFeaturePipeline()
        with pytest.raises(RuntimeError, match="not.*fitted"):
            pipe.export_state_json(tmp_path / "state.json")
