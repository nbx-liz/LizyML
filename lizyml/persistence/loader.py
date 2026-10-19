"""Loader — restore Model artifacts from a directory.

Security note: joblib.load() executes arbitrary Python code.
Only load artifacts from trusted sources.

Raises DESERIALIZATION_FAILED when:
- metadata.json is missing or malformed
- format_version is unknown
- Required metadata fields are absent
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import joblib

from lizyml.core.category_values import decode as decode_category_value
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.tuning_codec import (
    HISTORY_FILE,
    HISTORY_SCHEMA,
    LineageCodecError,
    TuningHistory,
    parse_history,
)
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.target_encoder import TargetEncoder
from lizyml.persistence.exporter import CHECKSUM_ALGORITHM, FORMAT_VERSION

_REQUIRED_METADATA_KEYS = frozenset(
    {"format_version", "task", "feature_names", "config", "run_id"}
)

# Format versions this loader can read. Older versions are upgraded in-memory
# via _migrate_fit_result so behavior remains equivalent to a fresh fit on
# the current library (numeric targets).
_SUPPORTED_FORMAT_VERSIONS = frozenset({1, 2, 3, 4})

#: The one ``tuning.history`` declaration a format-4 artifact may carry besides
#: ``null`` (H-0134 P4): any other shape is refused, never read as "no history".
_HISTORY_DECLARATION = {"file": HISTORY_FILE, "schema": HISTORY_SCHEMA}


def _refuse_history(path: str | Path, message: str, **context: Any) -> LizyMLError:
    return LizyMLError(
        code=ErrorCode.DESERIALIZATION_FAILED,
        user_message=f"Stored tuning history: {message}",
        context={"path": str(path), "file": HISTORY_FILE, **context},
    )


def _declared_tuning_history(
    src: Path, metadata: dict[str, Any], path: str | Path
) -> TuningHistory | None:
    """The tuning history the artifact declares, verified and decoded (H-0134).

    Format 1-3 never carry one (``None``). In format 4 a tuned model's
    ``tuning.history`` is required and is either ``null`` (the model had no
    history to write) or exactly :data:`_HISTORY_DECLARATION`; only then is
    ``tuning_history.json`` read -- once, checked against the SHA-256 in
    ``checksums`` before it is parsed -- so an undeclared file in the directory
    (left by an earlier export) is never read.

    Raises:
        LizyMLError: ``DESERIALIZATION_FAILED`` for a missing or malformed
            declaration, a declared file that is absent, unlisted or does not
            match its checksum, or a history the strict codec refuses.
    """
    tuning = metadata["tuning"]
    if metadata["format_version"] < 4:
        return None
    if "history" not in tuning:
        raise _refuse_history(path, "a format-4 tuning block must declare 'history'.")
    declaration = tuning["history"]
    if declaration is None:
        return None
    if type(declaration) is not dict or declaration != _HISTORY_DECLARATION:
        raise _refuse_history(
            path,
            f"the declaration must be null or {_HISTORY_DECLARATION}, "
            f"got {declaration!r}.",
        )
    checksums = metadata.get("checksums")
    files = checksums.get("files") if isinstance(checksums, dict) else None
    expected = files.get(HISTORY_FILE) if isinstance(files, dict) else None
    if (
        not isinstance(checksums, dict)
        or checksums.get("algorithm") != CHECKSUM_ALGORITHM
    ):
        raise _refuse_history(path, f"checksums must use {CHECKSUM_ALGORITHM!r}.")
    if type(expected) is not str:
        raise _refuse_history(path, f"'{HISTORY_FILE}' is not listed in checksums.")
    history_path = src / HISTORY_FILE
    if not history_path.is_file():
        raise _refuse_history(path, f"the declared '{HISTORY_FILE}' is missing.")
    raw = history_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise _refuse_history(
            path,
            "the file does not match the SHA-256 recorded in metadata.json; the "
            "artifact may be corrupted or tampered with.",
            expected=expected,
            actual=actual,
        )
    try:
        return parse_history(raw)
    except LineageCodecError as exc:
        raise _refuse_history(path, f"malformed: {exc}") from exc


def _decoded_declared_categories(record: Any, path: str | Path) -> Any:
    """The ``declared_categories`` record with each value decoded (H-0133 P6).

    Decoded here, once, so ``lizyml.persistence.load`` and ``Model.load``
    return the same values. A v1 / v2 record holds plain JSON scalars, which
    decode to themselves.

    Raises:
        LizyMLError: ``DESERIALIZATION_FAILED`` when the record is not an
            object of lists, or naming the column, for a value that is not a
            shape the exporter writes.
    """
    if not isinstance(record, dict):
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message="Stored declared_categories must be an object.",
            context={"path": str(path), "type": type(record).__name__},
        )
    for col, cats in record.items():
        if not isinstance(cats, list):
            raise LizyMLError(
                code=ErrorCode.DESERIALIZATION_FAILED,
                user_message=(
                    f"Stored declared_categories holds a {type(cats).__name__} "
                    f"for {col!r}, not a list."
                ),
                context={"path": str(path), "key": col, "type": type(cats).__name__},
            )
    decoded: dict[str, list[Any]] = {}
    for col, cats in record.items():
        try:
            decoded[col] = [decode_category_value(v) for v in cats]
        except LizyMLError as exc:
            raise LizyMLError(
                code=ErrorCode.DESERIALIZATION_FAILED,
                user_message=(
                    f"Stored declared_categories of column {col!r} hold a "
                    f"malformed value: {exc.context.get('reason')}"
                ),
                context={"path": str(path), "key": col, **exc.context},
                cause=exc,
            ) from exc
    return decoded


def _migrate_fit_result(fit_result: Any, source_version: int) -> Any:
    """Upgrade a deserialized FitResult to the current FORMAT_VERSION.

    H-0070 (v1 → v2): inject a no-op :class:`TargetEncoder` so consumers
    can call ``inverse_transform`` unconditionally. v1 artifacts only ever
    held numeric targets, so a no-op encoder yields identical behavior.
    """
    if source_version >= FORMAT_VERSION:
        return fit_result

    if not isinstance(fit_result, FitResult):
        return fit_result

    if not hasattr(fit_result, "target_encoder") or fit_result.target_encoder is None:
        fit_result = dataclasses.replace(
            fit_result, target_encoder=TargetEncoder.no_op()
        )
    return fit_result


def _verify_and_read_pkl(pkl_path: Path, checksums: dict[str, Any] | None) -> bytes:
    """Read *pkl_path*, verify its digest, and return the bytes (H-0083).

    Reading the bytes once and loading from them (the caller passes the return
    value to ``joblib.load(io.BytesIO(...))``) closes the TOCTOU window between
    hashing and deserialization — the file is never re-opened after the check.

    No verification is performed when *checksums* is absent (legacy artifacts
    exported before H-0083 — back-compatible read) or when the file is not
    listed. Raises ``DESERIALIZATION_FAILED`` on an unknown algorithm or a
    digest mismatch, detecting tampering/corruption *before* pickle executes.
    """
    raw = pkl_path.read_bytes()
    if not checksums:
        return raw
    expected = checksums.get("files", {}).get(pkl_path.name)
    if expected is None:
        return raw

    algorithm = checksums.get("algorithm")
    if algorithm != CHECKSUM_ALGORITHM:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=(
                f"Unsupported checksum algorithm {algorithm!r} for "
                f"'{pkl_path.name}'. Expected {CHECKSUM_ALGORITHM!r}."
            ),
            context={"file": pkl_path.name, "algorithm": algorithm},
        )

    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=(
                f"Integrity check failed for '{pkl_path.name}': the file does "
                f"not match the SHA-256 recorded in metadata.json. The artifact "
                f"may be corrupted or tampered with."
            ),
            context={"file": pkl_path.name, "expected": expected, "actual": actual},
        )
    return raw


def load(path: str | Path) -> tuple[Any, Any, dict[str, Any], Any]:
    """Load Model artifacts from *path*.

    Args:
        path: Directory previously created by
            :func:`~lizyml.persistence.exporter.export`.

    Returns:
        ``(fit_result, refit_result, metadata, analysis_context)`` tuple.
        ``analysis_context`` is ``None`` for legacy artifacts that lack it.
        For a tuned model, ``metadata["tuning"]["history"]`` is the decoded
        :class:`~lizyml.core.tuning_codec.TuningHistory`, or ``None`` when the
        artifact carries no history (format 1-3, or written without one).

    Raises:
        LizyMLError with DESERIALIZATION_FAILED on any validation or I/O error.

    Warning:
        Only load from trusted sources — joblib uses pickle internally.
    """
    src = Path(path)
    metadata_path = src / "metadata.json"
    fit_pkl = src / "fit_result.pkl"
    refit_pkl = src / "refit_model.pkl"

    # --- Validate directory and metadata -------------------------------------
    if not src.is_dir():
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"Export directory not found: '{path}'",
            context={"path": str(path)},
        )

    if not metadata_path.exists():
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"metadata.json not found in '{path}'.",
            context={"path": str(path)},
        )

    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"Failed to parse metadata.json: {exc}",
            context={"path": str(path)},
            cause=exc,
        ) from exc

    # Check required fields
    missing = _REQUIRED_METADATA_KEYS - set(metadata.keys())
    if missing:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"metadata.json is missing required fields: {sorted(missing)}",
            context={"missing": sorted(missing)},
        )

    # Check format_version (H-0070: accept any version in the supported set)
    fv = metadata.get("format_version")
    if fv not in _SUPPORTED_FORMAT_VERSIONS:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=(
                f"Unsupported format_version={fv!r}. "
                f"This version of LizyML supports "
                f"format_version in {sorted(_SUPPORTED_FORMAT_VERSIONS)}."
            ),
            context={
                "format_version": fv,
                "supported": sorted(_SUPPORTED_FORMAT_VERSIONS),
            },
        )

    if "declared_categories" in metadata:
        metadata["declared_categories"] = _decoded_declared_categories(
            metadata["declared_categories"], path
        )

    # H-0134: replace the declaration with the decoded history (None when the
    # artifact carries none), once, so persistence.load and Model.load agree.
    if isinstance(metadata.get("tuning"), dict):
        metadata["tuning"]["history"] = _declared_tuning_history(src, metadata, path)

    # --- Load pickled artifacts ----------------------------------------------
    for pkl_path in (fit_pkl, refit_pkl):
        if not pkl_path.exists():
            raise LizyMLError(
                code=ErrorCode.DESERIALIZATION_FAILED,
                user_message=f"Required artifact not found: '{pkl_path.name}'",
                context={"path": str(pkl_path)},
            )

    # H-0083: verify payload integrity, then load from the verified bytes (no
    # second file open) so pickle never executes unverified bytes.
    checksums = metadata.get("checksums")
    fit_raw = _verify_and_read_pkl(fit_pkl, checksums)
    refit_raw = _verify_and_read_pkl(refit_pkl, checksums)

    try:
        fit_result = joblib.load(io.BytesIO(fit_raw))
        refit_result = joblib.load(io.BytesIO(refit_raw))
    except Exception as exc:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"Failed to load model artifacts: {exc}",
            context={"path": str(path)},
            cause=exc,
        ) from exc

    # H-0070: upgrade older artifacts in-memory so callers always see the
    # current FitResult contract (e.g. target_encoder field present).
    fit_result = _migrate_fit_result(fit_result, int(fv))

    # Optional: analysis_context for diagnostic APIs after load
    analysis_context = None
    ctx_pkl = src / "analysis_context.pkl"
    if ctx_pkl.exists():
        ctx_raw = _verify_and_read_pkl(ctx_pkl, checksums)
        try:
            analysis_context = joblib.load(io.BytesIO(ctx_raw))
        except Exception as exc:
            raise LizyMLError(
                code=ErrorCode.DESERIALIZATION_FAILED,
                user_message=f"Failed to load analysis_context: {exc}",
                context={"path": str(path)},
                cause=exc,
            ) from exc

    return fit_result, refit_result, metadata, analysis_context
