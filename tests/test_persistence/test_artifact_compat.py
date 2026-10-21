"""Category A: Artifact compatibility tests.

Tests backward compatibility, format_version rejection, legacy calibration
path (model.py lines 321-326), Booster string roundtrip, and metadata
field-level validation.

See BLUEPRINT §18.1.4 and HISTORY H-0056 Category A.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.persistence.exporter import FORMAT_VERSION
from tests._helpers import make_binary_df, make_config, make_regression_df
from tests.test_persistence._artifact_versions import resign


def _fit_and_export(cfg: dict[str, Any], df: Any, tmp_path: Path) -> Path:
    """Fit a model and export to tmp_path/export."""
    m = Model(cfg)
    m.fit(data=df)
    export_dir = tmp_path / "export"
    m.export(export_dir)
    return export_dir


# ===================================================================
# Frozen artifact roundtrip (export → load → predict)
# ===================================================================


class TestArtifactRoundtrip:
    """Export → load → predict produces identical predictions."""

    def test_regression_roundtrip(self, tmp_path: Path) -> None:
        df = make_regression_df(n=100, seed=0)
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        m = Model(cfg)
        m.fit(data=df)

        X_test = df.drop(columns=["target"]).iloc[:10]
        pred_before = m.predict(X_test)

        export_dir = tmp_path / "export"
        m.export(export_dir)
        m2 = Model.load(export_dir)
        pred_after = m2.predict(X_test)

        np.testing.assert_array_almost_equal(
            pred_before.pred, pred_after.pred, decimal=6
        )

    def test_binary_calibrated_roundtrip(self, tmp_path: Path) -> None:
        df = make_binary_df(n=200, seed=0)
        cfg = make_config("binary", n_estimators=5, n_splits=2, calibration="platt")
        m = Model(cfg)
        m.fit(data=df)

        X_test = df.drop(columns=["target"]).iloc[:10]
        pred_before = m.predict(X_test)

        export_dir = tmp_path / "export"
        m.export(export_dir)
        m2 = Model.load(export_dir)
        pred_after = m2.predict(X_test)

        np.testing.assert_array_almost_equal(
            pred_before.pred, pred_after.pred, decimal=6
        )
        np.testing.assert_array_almost_equal(
            pred_before.proba, pred_after.proba, decimal=6
        )


# ===================================================================
# Format version rejection
# ===================================================================


class TestFormatVersionRejection:
    """Unknown format_version values are rejected on load.

    H-0070: format_version 1 and 2 are both accepted; H-0133 adds 3 and
    H-0134 adds 4 and H-0138 adds 5. Only versions outside the supported set
    are rejected.
    """

    @pytest.mark.parametrize("bad_version", [0, 6, 99, -1, None])
    def test_invalid_format_version_raises(
        self, tmp_path: Path, bad_version: Any
    ) -> None:
        df = make_regression_df(n=100, seed=0)
        export_dir = _fit_and_export(
            make_config("regression", n_estimators=5, n_splits=2), df, tmp_path
        )

        # Tamper with format_version
        meta_path = export_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["format_version"] = bad_version
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(LizyMLError) as exc_info:
            Model.load(export_dir)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED
        assert "format_version" in str(exc_info.value)


class TestFormatVersionV2AndV3:
    """H-0133 (criterion 6): the v2 shape still loads; v3 is what export writes,
    and a loader that knows only {1, 2} refuses it."""

    def test_a_v2_shaped_artifact_loads_and_predicts_as_before(
        self, tmp_path: Path
    ) -> None:
        from tests.test_persistence._artifact_versions import has_v3_fields, to_v2_shape

        df = make_binary_df(n=120, seed=1)
        df["c"] = np.where(np.arange(len(df)) % 3 == 0, "a", "b")
        model = Model(make_config("binary", n_estimators=5, n_splits=2))
        model.fit(data=df)
        export_dir = tmp_path / "export"
        model.export(export_dir)
        expected = model.predict(df.drop(columns=["target"])).pred
        to_v2_shape(export_dir)
        assert not has_v3_fields(export_dir)
        assert (
            json.loads((export_dir / "metadata.json").read_text())["format_version"]
            == 2
        )
        loaded = Model.load(export_dir)
        np.testing.assert_array_equal(
            loaded.predict(df.drop(columns=["target"])).pred, expected
        )
        # The per-fold states drive SHAP importance (criterion 2 on v2).
        pytest.importorskip("shap")
        assert loaded.importance(kind="shap") == pytest.approx(
            model.importance(kind="shap"), rel=1e-12
        )
        # The public low-level load reads the same artifact.
        from lizyml.persistence import load

        _, refit_result, metadata, _ = load(export_dir)
        assert metadata["format_version"] == 2
        assert "stand_in_columns" not in refit_result.pipeline_state["encoder"]

    def test_a_v3_artifact_is_refused_by_a_loader_without_3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lizyml.persistence import loader
        from tests.test_persistence._artifact_versions import to_v3_shape

        df = make_regression_df(n=100, seed=0)
        export_dir = _fit_and_export(
            make_config("regression", n_estimators=5, n_splits=2), df, tmp_path
        )
        to_v3_shape(export_dir)
        assert Model.load(export_dir) is not None  # the v3 shape is readable
        monkeypatch.setattr(loader, "_SUPPORTED_FORMAT_VERSIONS", frozenset({1, 2}))
        with pytest.raises(LizyMLError) as exc_info:
            Model.load(export_dir)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_a_v4_artifact_is_refused_by_a_loader_without_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """H-0134 P8: a loader that knows only 1-3 refuses v4 instead of
        ignoring the tuning history file and showing an empty study."""
        from lizyml.persistence import loader
        from tests.test_persistence._artifact_versions import to_v4_shape

        df = make_regression_df(n=100, seed=0)
        export_dir = _fit_and_export(
            make_config("regression", n_estimators=5, n_splits=2), df, tmp_path
        )
        to_v4_shape(export_dir)
        assert (
            json.loads((export_dir / "metadata.json").read_text())["format_version"]
            == 4
        )
        assert Model.load(export_dir) is not None  # the v4 shape is readable
        monkeypatch.setattr(loader, "_SUPPORTED_FORMAT_VERSIONS", frozenset({1, 2, 3}))
        with pytest.raises(LizyMLError) as exc_info:
            Model.load(export_dir)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_a_v5_artifact_is_refused_by_a_loader_without_5(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """H-0138 acceptance 10: a loader that knows only 1-4 refuses v5
        instead of looking the categories up through pandas."""
        from lizyml.persistence import loader

        df = make_regression_df(n=100, seed=0)
        export_dir = _fit_and_export(
            make_config("regression", n_estimators=5, n_splits=2), df, tmp_path
        )
        meta = json.loads((export_dir / "metadata.json").read_text())
        assert meta["format_version"] == 5
        monkeypatch.setattr(
            loader, "_SUPPORTED_FORMAT_VERSIONS", frozenset({1, 2, 3, 4})
        )
        with pytest.raises(LizyMLError) as exc_info:
            Model.load(export_dir)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED
        assert exc_info.value.context["format_version"] == 5

    @pytest.mark.parametrize("values", [["a", "b"], [1, "1", 2.5]], ids=repr)
    def test_a_v4_artifact_keeps_the_pandas_lookup(
        self, values: list[Any], tmp_path: Path
    ) -> None:
        """H-0138 decision 2: an unmarked state predicts as when it was saved,
        and its ``export_code`` writes the pandas lookup."""
        from tests.test_features.test_h0138_identity import _model_on
        from tests.test_persistence._artifact_versions import to_v4_shape

        model, df = _model_on(values)
        X = df.drop(columns=["target"])
        expected = model.predict(X).pred
        export_dir = tmp_path / "artifact"
        model.export(export_dir)
        to_v4_shape(export_dir)
        loaded = Model.load(export_dir)
        encoder = loaded._refit_result.pipeline_state["encoder"]  # noqa: SLF001
        assert "category_identity_version" not in encoder
        np.testing.assert_array_equal(loaded.predict(X).pred, expected)
        np.testing.assert_array_equal(loaded.predict(X.iloc[::-1]).pred, expected[::-1])
        loaded.export_code(tmp_path / "gen")
        config = json.loads((tmp_path / "gen" / "config.json").read_text())
        assert config["category_identity_version"] is None


def _rewrite_encoders(export_dir: Path, change: Any) -> None:
    """Apply *change* to every saved encoder state and re-sign the pickles."""
    from tests.test_persistence._artifact_versions import _encoder_states

    for name in ("fit_result.pkl", "refit_model.pkl"):
        pkl = export_dir / name
        result = joblib.load(pkl)
        for encoder in _encoder_states(result):
            change(encoder)
        joblib.dump(result, pkl, compress=3)
    resign(export_dir)


class TestIdentityMarkerAtLoad:
    """H-0138 acceptance 10: ``Model.load`` itself refuses an unknown marker
    (implementation review round 1, major 2), not the first ``predict``."""

    @pytest.fixture(scope="class")
    def exported(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        from tests.test_features.test_h0138_identity import _model_on

        model, _ = _model_on(["a", "b"])
        path = tmp_path_factory.mktemp("marked") / "artifact"
        model.export(path)
        return path

    def _copy(self, exported: Path, tmp_path: Path) -> Path:
        import shutil

        target = tmp_path / "artifact"
        shutil.copytree(exported, target)
        return target

    @pytest.mark.parametrize("marker", [2, "1", None, -1, True, 1.0])
    def test_an_unknown_marker_is_refused_by_load(
        self, marker: Any, exported: Path, tmp_path: Path
    ) -> None:
        artifact = self._copy(exported, tmp_path)
        _rewrite_encoders(
            artifact, lambda e: e.__setitem__("category_identity_version", marker)
        )
        with pytest.raises(LizyMLError) as exc_info:
            Model.load(artifact)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED
        assert "category identity version" in exc_info.value.user_message

    @pytest.mark.parametrize(
        "columns", [None, "c", ["missing"], [[]], [{}], [set()]], ids=repr
    )
    def test_malformed_identity_columns_are_refused_by_load(
        self, columns: Any, exported: Path, tmp_path: Path
    ) -> None:
        artifact = self._copy(exported, tmp_path)
        _rewrite_encoders(
            artifact, lambda e: e.__setitem__("identity_columns", columns)
        )
        with pytest.raises(LizyMLError) as exc_info:
            Model.load(artifact)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_a_v4_artifact_carrying_a_marker_is_refused(
        self, exported: Path, tmp_path: Path
    ) -> None:
        artifact = self._copy(exported, tmp_path)
        meta = json.loads((artifact / "metadata.json").read_text())
        meta["format_version"] = 4
        (artifact / "metadata.json").write_text(json.dumps(meta))
        with pytest.raises(LizyMLError) as exc_info:
            Model.load(artifact)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED
        assert exc_info.value.context["format_version"] == 4

    def test_the_marked_artifact_itself_loads(self, exported: Path) -> None:
        assert Model.load(exported) is not None


class TestFormatVersionV1Migration:
    """H-0070 (INV-5): v1 artifacts load with no-op encoder injected."""

    def test_v1_metadata_still_loads(self, tmp_path: Path) -> None:
        df = make_regression_df(n=100, seed=0)
        export_dir = _fit_and_export(
            make_config("regression", n_estimators=5, n_splits=2), df, tmp_path
        )

        # Tamper metadata to claim v1 to simulate an older artifact. Real v1
        # artifacts predate the H-0083 integrity field, so drop ``checksums``
        # too (otherwise the re-pickled payload below would mismatch). They
        # predate the H-0133 stand-ins and the H-0138 marker as well.
        from tests.test_persistence._artifact_versions import to_v2_shape

        to_v2_shape(export_dir)
        meta_path = export_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["format_version"] = 1
        meta.pop("checksums", None)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        # Re-pickle the FitResult without target_encoder to mirror v1 layout.
        fit_pkl = export_dir / "fit_result.pkl"
        fit_result = joblib.load(fit_pkl)
        # Strip target_encoder via in-place dict mutation (frozen=False on
        # FitResult, so attribute assignment is OK).
        fit_result.target_encoder = None  # type: ignore[assignment]
        joblib.dump(fit_result, fit_pkl, compress=3)

        m2 = Model.load(export_dir)
        # After migration, the no-op encoder is in place
        assert m2.fit_result.target_encoder.needs_encoding is False
        # Predict still works
        X_test = df.drop(columns=["target"]).iloc[:5]
        out = m2.predict(X_test)
        assert out.pred.shape == (5,)


# ===================================================================
# Metadata field-level validation
# ===================================================================


class TestMetadataFieldValidation:
    """Each required metadata field individually validated on load."""

    @pytest.mark.parametrize(
        "field", ["format_version", "task", "feature_names", "config", "run_id"]
    )
    def test_missing_field_raises(self, tmp_path: Path, field: str) -> None:
        df = make_regression_df(n=100, seed=0)
        export_dir = _fit_and_export(
            make_config("regression", n_estimators=5, n_splits=2), df, tmp_path
        )

        # Remove one required field
        meta_path = export_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta[field]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(LizyMLError) as exc_info:
            Model.load(export_dir)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        df = make_regression_df(n=100, seed=0)
        export_dir = _fit_and_export(
            make_config("regression", n_estimators=5, n_splits=2), df, tmp_path
        )
        (export_dir / "metadata.json").write_text("{invalid json", encoding="utf-8")

        with pytest.raises(LizyMLError) as exc_info:
            Model.load(export_dir)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED


# ===================================================================
# Legacy calibration path (model.py lines 321-326)
# ===================================================================


class TestLegacyCalibrationPath:
    """oof_raw_scores=None → probability-based calibration path."""

    def test_legacy_calibrator_uses_probability_input(self, tmp_path: Path) -> None:
        """When oof_raw_scores is None, C_final calibrates on probabilities."""
        df = make_binary_df(n=200, seed=0)
        cfg = make_config("binary", n_estimators=5, n_splits=2, calibration="platt")
        m = Model(cfg)
        m.fit(data=df)

        export_dir = tmp_path / "export"
        m.export(export_dir)

        # Tamper fit_result.pkl: set oof_raw_scores = None (legacy format)
        fit_pkl = export_dir / "fit_result.pkl"
        fit_result = joblib.load(fit_pkl)
        # Use object.__setattr__ since FitResult might be frozen
        if hasattr(fit_result, "__dataclass_fields__"):
            object.__setattr__(fit_result, "oof_raw_scores", None)
        else:
            fit_result.oof_raw_scores = None
        joblib.dump(fit_result, fit_pkl, compress=3)

        # Re-sign the re-pickled payload: a current-format artifact without
        # checksums is refused (H-0135 P2), and the test is about the
        # calibration path, not about the integrity field.
        resign(export_dir)

        m2 = Model.load(export_dir)
        X_test = df.drop(columns=["target"]).iloc[:20]
        pred = m2.predict(X_test)

        # Legacy path should still produce valid predictions
        assert pred.pred is not None
        assert len(pred.pred) == 20
        assert pred.proba is not None
        assert np.all((pred.proba >= 0) & (pred.proba <= 1))
        assert not np.any(np.isnan(pred.proba))


# ===================================================================
# Booster string roundtrip (LightGBM #7186 guard)
# ===================================================================


class TestBoosterStringRoundtrip:
    """model_to_string → Booster(model_str=...) preserves predictions."""

    @pytest.mark.parametrize("task", ["regression", "binary"])
    def test_booster_string_roundtrip(self, task: str) -> None:
        import lightgbm as lgb

        from lizyml.estimators.lgbm.adapter import LGBMAdapter

        if task == "regression":
            df = make_regression_df(n=100, seed=0)
        else:
            df = make_binary_df(n=100, seed=0)

        X = df.drop(columns=["target"])
        y = df["target"]
        adapter = LGBMAdapter(task=task, params={"n_estimators": 5}, random_state=42)
        adapter.fit(X, y)

        booster = adapter.get_native_model()
        model_str = booster.model_to_string()
        booster2 = lgb.Booster(model_str=model_str)

        preds1 = booster.predict(X)
        preds2 = booster2.predict(X)
        np.testing.assert_array_equal(preds1, preds2)


# ===================================================================
# Missing pkl files
# ===================================================================


class TestMissingArtifactFiles:
    """Required pkl files missing → clear error."""

    @pytest.mark.parametrize("pkl_name", ["fit_result.pkl", "refit_model.pkl"])
    def test_missing_pkl_raises(self, tmp_path: Path, pkl_name: str) -> None:
        df = make_regression_df(n=100, seed=0)
        export_dir = _fit_and_export(
            make_config("regression", n_estimators=5, n_splits=2), df, tmp_path
        )
        (export_dir / pkl_name).unlink()

        with pytest.raises(LizyMLError) as exc_info:
            Model.load(export_dir)
        assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED


# ===================================================================
# Metadata format_version matches current
# ===================================================================


class TestCurrentFormatVersion:
    """Exported metadata always has current FORMAT_VERSION."""

    def test_export_metadata_format_version(self, tmp_path: Path) -> None:
        df = make_regression_df(n=100, seed=0)
        export_dir = _fit_and_export(
            make_config("regression", n_estimators=5, n_splits=2), df, tmp_path
        )
        meta = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["format_version"] == FORMAT_VERSION
