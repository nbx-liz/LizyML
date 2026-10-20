"""Loader — restore Model artifacts from a directory.

Security note: joblib.load() executes arbitrary Python code.
Only load artifacts from trusted sources.

Raises DESERIALIZATION_FAILED when:
- metadata.json is missing or malformed
- format_version is unknown
- Required metadata fields are absent
- ``checksums`` is malformed, or absent in format 3-4, or does not list a
  required pickle, or lists a file that is missing (H-0135)
- a file does not match the digest ``checksums`` records (H-0083)

Only the files ``checksums`` declares are read; format 1-2 artifacts without
checksums are read as before H-0083.
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
from lizyml.core.logging import get_logger
from lizyml.core.tuning_codec import (
    HISTORY_FILE,
    HISTORY_SCHEMA,
    LineageCodecError,
    TuningHistory,
    parse_history,
)
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.target_encoder import TargetEncoder
from lizyml.persistence._manifest import (
    ANALYSIS_CONTEXT_FILE,
    FIT_RESULT_FILE,
    REFIT_MODEL_FILE,
    declared_files,
)
from lizyml.persistence.exporter import FORMAT_VERSION

_log = get_logger("persistence.loader")

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
    src: Path,
    metadata: dict[str, Any],
    files: dict[str, str] | None,
    path: str | Path,
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
    # declared_files has checked the manifest's shape and algorithm, and a
    # format-4 artifact cannot be without one (H-0135 P1, P2).
    expected = None if files is None else files.get(HISTORY_FILE)
    if expected is None:
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


def _expected(files: dict[str, str] | None, name: str) -> str | None:
    """The digest the manifest declares for *name*; ``None`` without a manifest."""
    return None if files is None else files[name]


def _is_payload(file: Path, path: str | Path) -> bool:
    """Whether *file* is there as a regular file (a symlink to one included).

    Something else under a payload's name (a directory, a socket) is refused
    here, before any pickle is read, rather than failing later with an
    ``OSError`` (H-0135 P7).
    """
    if not file.exists():
        return False
    if not file.is_file():
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"'{file.name}' in '{path}' is not a regular file.",
            context={"path": str(path), "reason": "not_a_file", "file": file.name},
        )
    return True


def _context_is_read(
    ctx_pkl: Path, files: dict[str, str] | None, path: str | Path
) -> bool:
    """Whether ``analysis_context.pkl`` is part of the artifact (H-0135 P4).

    With a manifest, only a declared file is read; a declared file that is
    missing is refused, and a present but undeclared one (left in a reused
    directory by an earlier export, #373) is ignored and logged. Without one
    (format 1-2, pre-H-0083) the file is read when present, as before.
    """
    if files is None:
        return _is_payload(ctx_pkl, path)
    if ANALYSIS_CONTEXT_FILE in files:
        if not _is_payload(ctx_pkl, path):
            raise LizyMLError(
                code=ErrorCode.DESERIALIZATION_FAILED,
                user_message=(
                    f"Stored checksums list '{ANALYSIS_CONTEXT_FILE}', but the "
                    "file is missing."
                ),
                context={
                    "path": str(path),
                    "reason": "declared_file_missing",
                    "file": ANALYSIS_CONTEXT_FILE,
                },
            )
        return True
    if ctx_pkl.exists():
        _log.info(
            "Ignoring '%s' in '%s': checksums do not list it, so it is not part "
            "of this artifact (left by an earlier export).",
            ANALYSIS_CONTEXT_FILE,
            path,
        )
    return False


def _verify_and_read_pkl(pkl_path: Path, expected: str | None) -> bytes:
    """Read *pkl_path*, verify its digest, and return the bytes (H-0083).

    Reading the bytes once and loading from them (the caller passes the return
    value to ``joblib.load(io.BytesIO(...))``) closes the TOCTOU window between
    hashing and deserialization — the file is never re-opened after the check.

    *expected* is the SHA-256 the manifest declares for the file
    (:func:`~lizyml.persistence._manifest.declared_files`). It is ``None`` only
    for a format 1-2 artifact without checksums, which is read unverified as
    before H-0083 (H-0135 P2). Raises ``DESERIALIZATION_FAILED`` on a digest
    mismatch, detecting tampering/corruption *before* pickle executes.
    """
    try:
        raw = pkl_path.read_bytes()
    except OSError as exc:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"Failed to read '{pkl_path.name}': {exc}",
            context={"file": pkl_path.name},
            cause=exc,
        ) from exc
    if expected is None:
        return raw

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

    if type(metadata) is not dict:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message="metadata.json must hold a JSON object.",
            context={"path": str(path), "type": type(metadata).__name__},
        )

    # Check required fields
    missing = _REQUIRED_METADATA_KEYS - set(metadata.keys())
    if missing:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"metadata.json is missing required fields: {sorted(missing)}",
            context={"missing": sorted(missing)},
        )

    # Check format_version (H-0070: accept any version in the supported set).
    # Exactly an int: JSON true and 2.0 compare equal to 1 and 2 and would
    # pass as a format that may lack checksums (H-0135).
    fv = metadata.get("format_version")
    if type(fv) is not int or fv not in _SUPPORTED_FORMAT_VERSIONS:
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

    # H-0135 P1-P3: the manifest decides what is read. None only for a format
    # 1-2 artifact without checksums (pre-H-0083).
    files = declared_files(metadata, fv, path)

    if "declared_categories" in metadata:
        metadata["declared_categories"] = _decoded_declared_categories(
            metadata["declared_categories"], path
        )

    # H-0134: a format-4 writer records, for a tuned model, a tuning object
    # with a history declaration. Check it before any pickle is read, whatever
    # the declaration says (null included).
    if fv >= 4 and "tuning" in metadata and type(metadata["tuning"]) is not dict:
        raise _refuse_history(
            path,
            f"a format-4 'tuning' block must be an object, got "
            f"{type(metadata['tuning']).__name__}.",
        )
    # Replace the declaration with the decoded history (None when the artifact
    # carries none), once, so persistence.load and Model.load agree.
    if isinstance(metadata.get("tuning"), dict):
        metadata["tuning"]["history"] = _declared_tuning_history(
            src, metadata, files, path
        )

    # --- Load pickled artifacts ----------------------------------------------
    for pkl_path in (fit_pkl, refit_pkl):
        if not _is_payload(pkl_path, path):
            raise LizyMLError(
                code=ErrorCode.DESERIALIZATION_FAILED,
                user_message=f"Required artifact not found: '{pkl_path.name}'",
                context={"path": str(pkl_path)},
            )
    ctx_pkl = src / ANALYSIS_CONTEXT_FILE
    read_context = _context_is_read(ctx_pkl, files, path)

    # H-0083: verify payload integrity, then load from the verified bytes (no
    # second file open) so pickle never executes unverified bytes.
    fit_raw = _verify_and_read_pkl(fit_pkl, _expected(files, FIT_RESULT_FILE))
    refit_raw = _verify_and_read_pkl(refit_pkl, _expected(files, REFIT_MODEL_FILE))

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
    if read_context:
        ctx_raw = _verify_and_read_pkl(ctx_pkl, _expected(files, ANALYSIS_CONTEXT_FILE))
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
