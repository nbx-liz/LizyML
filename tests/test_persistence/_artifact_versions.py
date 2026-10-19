"""Artifacts in the shape of an earlier ``format_version`` (H-0133 P8).

No artifact saved by an earlier release is stored in the repository, so a
test makes one from a current export: :func:`to_v2_shape` removes what
format_version 3 added (the encoder's ``stand_in_columns``), writes
``format_version`` 2 and re-signs the pickles. Only a model without stand-in
columns has a v2 shape -- a v2 artifact could not hold the values that need
them (its export failed).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib

_PICKLES = ("fit_result.pkl", "refit_model.pkl")


def _encoder_states(result: Any) -> Iterator[dict[str, Any]]:
    states = [getattr(result, "pipeline_state", None)]
    states += list(getattr(result, "pipeline_state_per_fold", None) or [])
    for state in states:
        if isinstance(state, dict) and isinstance(state.get("encoder"), dict):
            yield state["encoder"]


def to_v2_shape(export_dir: Path) -> None:
    """Rewrite *export_dir* into the format_version 2 shape, in place."""
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


def has_v3_fields(export_dir: Path) -> bool:
    """Whether any saved encoder state carries ``stand_in_columns``."""
    return any(
        "stand_in_columns" in encoder
        for name in _PICKLES
        for encoder in _encoder_states(joblib.load(export_dir / name))
    )
