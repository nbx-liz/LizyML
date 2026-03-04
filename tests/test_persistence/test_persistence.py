"""Tests for Phase 14 — Persistence & Export.

Covers:
- export() creates metadata.json, fit_result.pkl, refit_model.pkl
- metadata.json contains required fields and correct format_version
- load() → predict() returns same results as original model
- load() with unknown format_version raises DESERIALIZATION_FAILED
- load() with missing metadata fields raises DESERIALIZATION_FAILED
- load() on non-existent directory raises DESERIALIZATION_FAILED
- Model.export() before fit raises MODEL_NOT_FIT
- Model.load() class method works end-to-end
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.persistence.exporter import FORMAT_VERSION

# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------


def _reg_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"]
    return df


def _bin_df(n: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    return df


def _reg_config() -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


def _bin_config() -> dict:
    return {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_creates_directory(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "model_out"
        m.export(out)
        assert out.is_dir()

    def test_export_creates_required_files(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "model_out"
        m.export(out)
        assert (out / "metadata.json").exists()
        assert (out / "fit_result.pkl").exists()
        assert (out / "refit_model.pkl").exists()

    def test_metadata_has_required_fields(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "model_out"
        m.export(out)
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["format_version"] == FORMAT_VERSION
        assert meta["task"] == "regression"
        assert "feature_names" in meta
        assert "config" in meta
        assert "run_id" in meta
        assert "metrics" in meta

    def test_export_before_fit_raises(self, tmp_path: Path) -> None:
        m = Model(_reg_config())
        with pytest.raises(LizyMLError) as exc_info:
            m.export(tmp_path / "out")
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT


# ---------------------------------------------------------------------------
# Load tests
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_regression_e2e(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "model_reg"
        m.export(out)

        m2 = Model.load(out)
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        pred1 = m.predict(X_new)
        pred2 = m2.predict(X_new)
        np.testing.assert_array_almost_equal(pred1.pred, pred2.pred)

    def test_load_binary_e2e(self, tmp_path: Path) -> None:
        df = _bin_df()
        m = Model(_bin_config())
        m.fit(data=df)
        out = tmp_path / "model_bin"
        m.export(out)

        m2 = Model.load(out)
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        pred1 = m.predict(X_new)
        pred2 = m2.predict(X_new)
        np.testing.assert_array_almost_equal(pred1.pred, pred2.pred)
        assert pred2.proba is not None

    def test_load_evaluate_works(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "model_eval"
        m.export(out)

        m2 = Model.load(out)
        metrics = m2.evaluate()
        assert "raw" in metrics

    def test_load_importance_works(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "model_imp"
        m.export(out)

        m2 = Model.load(out)
        imp = m2.importance()
        assert set(imp.keys()) == {"feat_a", "feat_b"}

    def test_load_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            Model.load(tmp_path / "does_not_exist")
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_load_unknown_format_version_raises(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "model_fv"
        m.export(out)

        # Tamper with format_version
        meta_path = out / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["format_version"] = 999
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(LizyMLError) as exc_info:
            Model.load(out)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_load_missing_metadata_field_raises(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "model_missing"
        m.export(out)

        # Remove required field
        meta_path = out / "metadata.json"
        meta = json.loads(meta_path.read_text())
        del meta["task"]
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(LizyMLError) as exc_info:
            Model.load(out)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_load_missing_pkl_raises(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "model_nopkl"
        m.export(out)

        # Remove a pkl file
        (out / "fit_result.pkl").unlink()

        with pytest.raises(LizyMLError) as exc_info:
            Model.load(out)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_load_missing_metadata_json_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "model_nometa"
        out.mkdir()
        # Directory exists but no metadata.json
        with pytest.raises(LizyMLError) as exc_info:
            Model.load(out)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED
