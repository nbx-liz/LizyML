"""Error-path coverage for persistence exporter/loader (serialization failures)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError


class TestExportError:
    def test_serialization_error(self) -> None:
        from lizyml.persistence.exporter import export

        mock_fr = MagicMock()
        mock_fr.run_meta.lizyml_version = "0.1.0"
        mock_fr.run_meta.python_version = "3.11"
        mock_fr.run_meta.timestamp = "2026-01-01"
        mock_fr.run_meta.run_id = "test"
        mock_fr.metrics = {}
        mock_fr.feature_names = ["a"]

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("lizyml.persistence.exporter.joblib") as mock_jl,
        ):
            mock_jl.dump.side_effect = OSError("disk full")
            with pytest.raises(LizyMLError) as exc_info:
                export(
                    path=tmpdir,
                    fit_result=mock_fr,
                    refit_result=MagicMock(),
                    config={"task": "regression"},
                    task="regression",
                )
            assert exc_info.value.code == ErrorCode.SERIALIZATION_FAILED


class TestLoadErrors:
    def test_corrupt_metadata_json(self) -> None:
        from lizyml.persistence.loader import load

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "metadata.json").write_text(
                "NOT VALID JSON{{{", encoding="utf-8"
            )
            with pytest.raises(LizyMLError) as exc_info:
                load(tmpdir)
            assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_corrupt_fit_result_pkl(self) -> None:
        from lizyml.persistence.exporter import FORMAT_VERSION
        from lizyml.persistence.loader import load

        with tempfile.TemporaryDirectory() as tmpdir:
            meta = {
                "format_version": FORMAT_VERSION,
                "task": "regression",
                "feature_names": ["a"],
                "config": {},
                "run_id": "test",
            }
            (Path(tmpdir) / "metadata.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            (Path(tmpdir) / "fit_result.pkl").write_bytes(b"corrupt data")
            (Path(tmpdir) / "refit_model.pkl").write_bytes(b"corrupt data")
            with pytest.raises(LizyMLError) as exc_info:
                load(tmpdir)
            assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_corrupt_analysis_context(self) -> None:
        import joblib

        from lizyml.persistence.exporter import FORMAT_VERSION
        from lizyml.persistence.loader import load

        with tempfile.TemporaryDirectory() as tmpdir:
            meta = {
                "format_version": FORMAT_VERSION,
                "task": "regression",
                "feature_names": ["a"],
                "config": {},
                "run_id": "test",
            }
            (Path(tmpdir) / "metadata.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            joblib.dump({"dummy": True}, Path(tmpdir) / "fit_result.pkl")
            joblib.dump({"dummy": True}, Path(tmpdir) / "refit_model.pkl")
            (Path(tmpdir) / "analysis_context.pkl").write_bytes(b"corrupt")
            with pytest.raises(LizyMLError) as exc_info:
                load(tmpdir)
            assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED
