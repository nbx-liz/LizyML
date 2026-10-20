"""Artifact integrity tests — SHA-256 binding between metadata and .pkl (H-0083).

``Model.load`` deserializes ``.pkl`` payloads with joblib (arbitrary-code
execution). H-0083 records the SHA-256 of each ``.pkl`` in ``metadata.json`` at
export and verifies it before ``joblib.load`` so tampering/corruption is detected
(``DESERIALIZATION_FAILED``). Verification is skipped for legacy artifacts that
predate the ``checksums`` field (format 1-2, back-compatible read); a format 3-4
artifact without it is refused (H-0135 P2).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_config, make_regression_df
from tests.test_persistence._artifact_versions import to_v2_shape


def _export_model(tmp_path: Path) -> Path:
    m = Model(make_config("regression"))
    m.fit(data=make_regression_df(n=120))
    out = tmp_path / "artifact"
    m.export(out)
    return out


def _read_metadata(out: Path) -> dict:
    return json.loads((out / "metadata.json").read_text(encoding="utf-8"))


def test_export_records_sha256_for_each_pkl(tmp_path: Path) -> None:
    import hashlib

    out = _export_model(tmp_path)
    checksums = _read_metadata(out)["checksums"]

    assert checksums["algorithm"] == "sha256"
    files = checksums["files"]
    # Required artifacts plus the optional analysis_context written after fit.
    for name in ("fit_result.pkl", "refit_model.pkl", "analysis_context.pkl"):
        expected = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert files[name] == expected


def test_roundtrip_with_checksums_succeeds(tmp_path: Path) -> None:
    out = _export_model(tmp_path)
    # Should load without error when digests match.
    Model.load(out)


@pytest.mark.parametrize(
    "pkl_name", ["fit_result.pkl", "refit_model.pkl", "analysis_context.pkl"]
)
def test_tampered_pkl_is_rejected(tmp_path: Path, pkl_name: str) -> None:
    out = _export_model(tmp_path)
    # Overwrite with a *valid but different* joblib payload: joblib.load would
    # succeed, so only the checksum mismatch can catch the swap.
    joblib.dump({"tampered": True}, out / pkl_name, compress=3)

    with pytest.raises(LizyMLError) as excinfo:
        Model.load(out)
    assert excinfo.value.code == ErrorCode.DESERIALIZATION_FAILED
    assert excinfo.value.context.get("file") == pkl_name


def test_legacy_artifact_without_checksums_still_loads(tmp_path: Path) -> None:
    out = _export_model(tmp_path)
    # Simulate an artifact exported before H-0083: those were format 2 (the
    # v0.16.0 release of H-0083 still wrote 2) and had no integrity field.
    to_v2_shape(out)
    metadata = _read_metadata(out)
    metadata.pop("checksums", None)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Back-compatible read: verification is skipped, load still works.
    Model.load(out)


def test_a_current_format_artifact_without_checksums_is_refused(
    tmp_path: Path,
) -> None:
    """Format 3 and 4 writers always record checksums (H-0135 P2)."""
    out = _export_model(tmp_path)
    metadata = _read_metadata(out)
    metadata.pop("checksums")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with pytest.raises(LizyMLError) as excinfo:
        Model.load(out)
    assert excinfo.value.code == ErrorCode.DESERIALIZATION_FAILED
    assert excinfo.value.context["reason"] == "checksums_missing"
