"""Artifacts in the shape of an earlier ``format_version`` (H-0133 P8, H-0134).

No artifact saved by an earlier release is stored in the repository, so a
test makes one from a current export.

- :func:`to_v4_shape` removes what format_version 5 added: the encoder's
  identity marker (H-0138). It writes ``format_version`` 4.
- :func:`to_v3_shape` does that, then removes what format_version 4 added:
  the tuning history file, its checksum and the ``tuning.history``
  declaration. It writes ``format_version`` 3.
- :func:`to_v2_shape` does that, then removes what format_version 3 added: the
  encoder's ``stand_in_columns``. It writes ``format_version`` 2 and re-signs
  the pickles.

Only a model without stand-in columns has a v2 shape. A v2 artifact could not
hold the values that need them, because its export failed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib

from lizyml.core.tuning_codec import HISTORY_FILE

_PICKLES = ("fit_result.pkl", "refit_model.pkl")


def _encoder_states(result: Any) -> Iterator[dict[str, Any]]:
    states = [getattr(result, "pipeline_state", None)]
    states += list(getattr(result, "pipeline_state_per_fold", None) or [])
    for state in states:
        if isinstance(state, dict) and isinstance(state.get("encoder"), dict):
            yield state["encoder"]


def to_v4_shape(export_dir: Path) -> None:
    """Rewrite *export_dir* into the format_version 4 shape, in place.

    Removes the encoder's identity marker (``category_identity_version`` and
    ``identity_columns``, H-0138), writes ``format_version`` 4 and re-signs
    the pickles. Only a model whose columns pandas holds faithfully has a v4
    shape: a v4 fit could not build a column by key.
    """
    meta_path = export_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for name in _PICKLES:
        pkl = export_dir / name
        result = joblib.load(pkl)
        for encoder in _encoder_states(result):
            encoder.pop("category_identity_version")
            encoder.pop("identity_columns")
        joblib.dump(result, pkl, compress=3)
        if "checksums" in meta:
            digest = hashlib.sha256(pkl.read_bytes()).hexdigest()
            meta["checksums"]["files"][name] = digest
    meta["format_version"] = 4
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def to_v3_shape(export_dir: Path) -> None:
    """Rewrite *export_dir* into the format_version 3 shape, in place."""
    to_v4_shape(export_dir)
    meta_path = export_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if isinstance(meta.get("tuning"), dict):
        meta["tuning"].pop("history", None)
    if "checksums" in meta:
        meta["checksums"]["files"].pop(HISTORY_FILE, None)
    (export_dir / HISTORY_FILE).unlink(missing_ok=True)
    meta["format_version"] = 3
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def to_v2_shape(export_dir: Path) -> None:
    """Rewrite *export_dir* into the format_version 2 shape, in place."""
    to_v3_shape(export_dir)
    meta_path = export_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for name in _PICKLES:
        pkl = export_dir / name
        result = joblib.load(pkl)
        for encoder in _encoder_states(result):
            stood = encoder.pop("stand_in_columns")
            assert stood == [], f"{name}: a v2 artifact cannot hold stand-ins {stood}"
        joblib.dump(result, pkl, compress=3)
        if "checksums" in meta:  # a caller may have dropped them already
            digest = hashlib.sha256(pkl.read_bytes()).hexdigest()
            meta["checksums"]["files"][name] = digest
    meta["format_version"] = 2
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def to_v1_shape(export_dir: Path) -> None:
    """Rewrite *export_dir* as a format_version 1 artifact, in place.

    The v2 shape with ``format_version`` 1. This is enough for the tuning
    surfaces, which only ask whether the artifact carries a history.
    """
    to_v2_shape(export_dir)
    meta_path = export_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format_version"] = 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def resign(export_dir: Path) -> None:
    """Record the digest of every file ``checksums`` lists, as it is now on disk.

    For a test that rewrites a payload of a current-format artifact: the
    loader refuses a format 3-4 artifact without checksums (H-0135 P2), so the
    test keeps the manifest and re-signs instead of dropping it.
    """
    meta_path = export_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files = meta["checksums"]["files"]
    for name in files:
        files[name] = hashlib.sha256((export_dir / name).read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def has_v3_fields(export_dir: Path) -> bool:
    """Whether any saved encoder state carries ``stand_in_columns``."""
    return any(
        "stand_in_columns" in encoder
        for name in _PICKLES
        for encoder in _encoder_states(joblib.load(export_dir / name))
    )
