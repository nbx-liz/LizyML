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


def _bin_config(with_calibration: bool = False) -> dict:
    cfg: dict = {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }
    if with_calibration:
        cfg["calibration"] = {"method": "platt", "n_splits": 3}
    return cfg


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

    def test_export_load_preserves_calibrator(self, tmp_path: Path) -> None:
        """Calibrator must survive export → load round-trip."""
        df = _bin_df()
        m = Model(_bin_config(with_calibration=True))
        m.fit(data=df)
        assert m._fit_result is not None
        # sanity: calibrator set before export
        assert m._fit_result.calibrator is not None

        out = tmp_path / "model_cal"
        m.export(out)

        m2 = Model.load(out)
        assert m2._fit_result is not None
        assert m2._fit_result.calibrator is not None  # must survive round-trip

        X_new = df.drop(columns=["target"]).iloc[:10].reset_index(drop=True)
        pred = m2.predict(X_new)
        assert pred.proba is not None
        assert np.all(pred.proba >= 0.0) and np.all(pred.proba <= 1.0)


# ---------------------------------------------------------------------------
# T-5: Persistence contract — all saved fields are preserved
# ---------------------------------------------------------------------------


class TestPersistenceContract:
    def test_metadata_json_has_all_required_fields(self, tmp_path: Path) -> None:
        """metadata.json must contain every contractual field."""
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "meta_contract"
        m.export(out)
        meta = json.loads((out / "metadata.json").read_text())

        required_fields = {
            "format_version",
            "lizyml_version",
            "python_version",
            "timestamp",
            "run_id",
            "task",
            "feature_names",
            "config",
            "metrics",
        }
        missing = required_fields - set(meta.keys())
        assert missing == set(), f"metadata.json missing fields: {missing}"

    def test_metadata_format_version_is_integer(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "fv_type"
        m.export(out)
        meta = json.loads((out / "metadata.json").read_text())
        assert isinstance(meta["format_version"], int)

    def test_metadata_config_is_dict(self, tmp_path: Path) -> None:
        """'config' field must be a non-empty dict (normalized config)."""
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "cfg_dict"
        m.export(out)
        meta = json.loads((out / "metadata.json").read_text())
        assert isinstance(meta["config"], dict)
        assert len(meta["config"]) > 0

    def test_fit_result_splits_preserved_after_load(self, tmp_path: Path) -> None:
        """FitResult.splits.outer must survive export → load with correct fold count."""
        n_splits = 3
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "splits_contract"
        m.export(out)

        m2 = Model.load(out)
        assert m2._fit_result is not None
        assert len(m2._fit_result.splits.outer) == n_splits

    def test_fit_result_data_fingerprint_preserved(self, tmp_path: Path) -> None:
        """data_fingerprint must survive export → load (row_count preserved)."""
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        original_fp = m._fit_result.data_fingerprint  # type: ignore[union-attr]
        out = tmp_path / "fp_contract"
        m.export(out)

        m2 = Model.load(out)
        assert m2._fit_result is not None
        loaded_fp = m2._fit_result.data_fingerprint
        assert loaded_fp.row_count == original_fp.row_count
        assert loaded_fp.column_hash == original_fp.column_hash

    def test_fit_result_run_meta_preserved(self, tmp_path: Path) -> None:
        """run_meta.lizyml_version must survive export → load."""
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "run_meta_contract"
        m.export(out)

        m2 = Model.load(out)
        assert m2._fit_result is not None
        assert m2._fit_result.run_meta.lizyml_version is not None
        assert len(m2._fit_result.run_meta.lizyml_version) > 0

    def test_predict_after_load_produces_same_result(self, tmp_path: Path) -> None:
        """Pipeline state must be preserved: predict() output identical before and after
        load."""
        df = _reg_df()
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        m = Model(_reg_config())
        m.fit(data=df)
        pred_before = m.predict(X_new)
        out = tmp_path / "pipeline_contract"
        m.export(out)

        m2 = Model.load(out)
        pred_after = m2.predict(X_new)
        np.testing.assert_array_almost_equal(pred_before.pred, pred_after.pred)


# ---------------------------------------------------------------------------
# 22-A: Diagnostic APIs after Model.load() (H-0026)
# ---------------------------------------------------------------------------


class TestDiagnosticAPIsAfterLoad:
    """Verify diagnostic APIs work after export → load cycle."""

    def test_export_creates_analysis_context(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "ctx_export"
        m.export(out)
        assert (out / "analysis_context.pkl").exists()

    def test_residuals_after_load(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        res_before = m.residuals()
        out = tmp_path / "res_load"
        m.export(out)

        m2 = Model.load(out)
        res_after = m2.residuals()
        np.testing.assert_array_almost_equal(res_before, res_after)

    def test_confusion_matrix_after_load(self, tmp_path: Path) -> None:
        df = _bin_df()
        m = Model(_bin_config())
        m.fit(data=df)
        cm_before = m.confusion_matrix()
        out = tmp_path / "cm_load"
        m.export(out)

        m2 = Model.load(out)
        cm_after = m2.confusion_matrix()
        pd.testing.assert_frame_equal(cm_before["oos"], cm_after["oos"])

    def test_roc_curve_plot_after_load(self, tmp_path: Path) -> None:
        pytest.importorskip("plotly")
        df = _bin_df()
        m = Model(_bin_config())
        m.fit(data=df)
        out = tmp_path / "roc_load"
        m.export(out)

        m2 = Model.load(out)
        fig = m2.roc_curve_plot()
        assert fig is not None

    def test_importance_shap_after_load(self, tmp_path: Path) -> None:
        pytest.importorskip("shap")
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "shap_load"
        m.export(out)

        m2 = Model.load(out)
        imp = m2.importance(kind="shap")
        assert set(imp.keys()) == {"feat_a", "feat_b"}

    def test_calibration_plot_after_load(self, tmp_path: Path) -> None:
        pytest.importorskip("plotly")
        df = _bin_df()
        m = Model(_bin_config(with_calibration=True))
        m.fit(data=df)
        out = tmp_path / "cal_load"
        m.export(out)

        m2 = Model.load(out)
        fig = m2.calibration_plot()
        assert fig is not None

    def test_probability_histogram_after_load(self, tmp_path: Path) -> None:
        pytest.importorskip("plotly")
        df = _bin_df()
        m = Model(_bin_config(with_calibration=True))
        m.fit(data=df)
        out = tmp_path / "proba_hist_load"
        m.export(out)

        m2 = Model.load(out)
        fig = m2.probability_histogram_plot()
        assert fig is not None


class TestLegacyArtifactCompat:
    """Legacy artifacts (without analysis_context.pkl) load but diagnostic APIs fail."""

    @staticmethod
    def _export_as_legacy(
        model: Model,
        out: Path,
    ) -> Model:
        model.export(out)
        (out / "analysis_context.pkl").unlink()
        return Model.load(out)

    def test_load_without_context_succeeds(self, tmp_path: Path) -> None:
        df = _reg_df()
        m = Model(_reg_config())
        m.fit(data=df)
        out = tmp_path / "legacy"
        m2 = self._export_as_legacy(m, out)
        # predict still works
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        pred = m2.predict(X_new)
        assert pred.pred is not None

    def test_predict_and_evaluate_still_work_without_context(
        self, tmp_path: Path
    ) -> None:
        reg_df = _reg_df()
        reg_model = Model(_reg_config())
        reg_model.fit(data=reg_df)
        legacy_reg = self._export_as_legacy(reg_model, tmp_path / "legacy_reg")

        reg_X_new = reg_df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        reg_pred = legacy_reg.predict(reg_X_new)
        reg_eval = legacy_reg.evaluate()
        assert reg_pred.pred is not None
        assert "raw" in reg_eval

        bin_df = _bin_df()
        bin_model = Model(_bin_config(with_calibration=True))
        bin_model.fit(data=bin_df)
        legacy_bin = self._export_as_legacy(bin_model, tmp_path / "legacy_bin")

        bin_X_new = bin_df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        bin_pred = legacy_bin.predict(bin_X_new)
        bin_eval = legacy_bin.evaluate()
        assert bin_pred.pred is not None
        assert bin_pred.proba is not None
        assert "raw" in bin_eval

    def test_all_diagnostic_apis_raise_model_not_fit_for_legacy_artifact(
        self, tmp_path: Path
    ) -> None:
        reg_df = _reg_df()
        reg_model = Model(_reg_config())
        reg_model.fit(data=reg_df)
        legacy_reg = self._export_as_legacy(reg_model, tmp_path / "legacy_reg_diag")

        bin_df = _bin_df()
        bin_model = Model(_bin_config(with_calibration=True))
        bin_model.fit(data=bin_df)
        legacy_bin = self._export_as_legacy(bin_model, tmp_path / "legacy_bin_diag")

        checks = [
            ("residuals", lambda: legacy_reg.residuals()),
            ("residuals_plot", lambda: legacy_reg.residuals_plot()),
            ("importance_shap", lambda: legacy_reg.importance(kind="shap")),
            ("roc_curve_plot", lambda: legacy_bin.roc_curve_plot()),
            ("confusion_matrix", lambda: legacy_bin.confusion_matrix()),
            ("calibration_plot", lambda: legacy_bin.calibration_plot()),
            (
                "probability_histogram_plot",
                lambda: legacy_bin.probability_histogram_plot(),
            ),
        ]

        for name, call in checks:
            with pytest.raises(LizyMLError) as exc_info:
                call()
            assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT, name
            assert "Re-export" in str(exc_info.value.user_message), name
