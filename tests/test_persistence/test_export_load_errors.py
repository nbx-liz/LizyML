"""Error-path coverage for persistence exporter/loader (serialization failures)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError


def _write_signed_metadata(out: Path) -> None:
    """A current-format metadata.json whose checksums list every file in *out*.

    Each digest is of the bytes on disk, corrupt ones included, so the
    integrity check passes and the failure under test is joblib's: a format
    3-4 artifact without checksums is refused before that (H-0135 P2).
    """
    import hashlib

    from lizyml.persistence.exporter import FORMAT_VERSION

    files = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(out.iterdir())
    }
    meta = {
        "format_version": FORMAT_VERSION,
        "task": "regression",
        "feature_names": ["a"],
        "config": {},
        "run_id": "test",
        "checksums": {"algorithm": "sha256", "files": files},
    }
    (out / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


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
        import joblib

        from lizyml.persistence.loader import load

        with tempfile.TemporaryDirectory() as tmpdir:
            # Only fit_result.pkl is corrupt (#270): with both corrupt, a
            # failure on refit_model.pkl would satisfy the same assertion.
            (Path(tmpdir) / "fit_result.pkl").write_bytes(b"corrupt data")
            joblib.dump({"dummy": True}, Path(tmpdir) / "refit_model.pkl")
            _write_signed_metadata(Path(tmpdir))
            with pytest.raises(LizyMLError) as exc_info:
                load(tmpdir)
            assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED
            assert exc_info.value.__cause__ is not None

    def test_corrupt_analysis_context(self) -> None:
        import joblib

        from lizyml.persistence.loader import load

        with tempfile.TemporaryDirectory() as tmpdir:
            joblib.dump({"dummy": True}, Path(tmpdir) / "fit_result.pkl")
            joblib.dump({"dummy": True}, Path(tmpdir) / "refit_model.pkl")
            (Path(tmpdir) / "analysis_context.pkl").write_bytes(b"corrupt")
            _write_signed_metadata(Path(tmpdir))
            with pytest.raises(LizyMLError) as exc_info:
                load(tmpdir)
            assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED
            assert exc_info.value.__cause__ is not None
