"""Golden test pinning the top-level public export surface (H-0086, #213).

Contract types (`FitResult` / `PredictionResult` / `TuningResult`), the unified
exception (`LizyMLError` / `ErrorCode`), `load_config`, and `TaskType` must be
importable from the top-level ``lizyml`` package so users never reach into the
private-looking ``lizyml.core.*`` paths for type annotations or
``except LizyMLError`` handling. Pinning ``__all__`` makes any accidental
removal a failing test rather than a silent breaking change.
"""

from __future__ import annotations

import lizyml

_EXPECTED_TOP_LEVEL_ALL = {
    "BoundaryDimStatus",
    "BoundaryReport",
    "ErrorCode",
    "FitResult",
    "LizyMLError",
    "Model",
    "PredictionResult",
    "RoundSummary",
    "TaskType",
    "TuneProgressCallback",
    "TuneProgressInfo",
    "TuningResult",
    "__version__",
    "__version_tuple__",
    "load_config",
}


def test_top_level_all_is_pinned() -> None:
    assert set(lizyml.__all__) == _EXPECTED_TOP_LEVEL_ALL


def test_contract_types_importable_from_top_level() -> None:
    from lizyml import (  # noqa: F401
        ErrorCode,
        FitResult,
        LizyMLError,
        Model,
        PredictionResult,
        TaskType,
        TuningResult,
        load_config,
    )

    # LizyMLError is a real exception usable in ``except`` clauses.
    assert issubclass(LizyMLError, Exception)


def test_every_all_entry_is_a_real_attribute() -> None:
    for name in lizyml.__all__:
        assert hasattr(lizyml, name), f"{name} listed in __all__ but not importable"


def test_data_fingerprint_exported_from_core_types() -> None:
    from lizyml.core.types import DataFingerprint  # noqa: F401

    assert (
        "DataFingerprint"
        in __import__("lizyml.core.types", fromlist=["__all__"]).__all__
    )
