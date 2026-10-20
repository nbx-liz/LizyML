"""The ``checksums`` manifest of an artifact: its one admitted shape (H-0135).

``checksums`` is absent, or exactly ``{"algorithm": "sha256", "files": {<str>:
<64 lowercase hex>}}`` (P1). Absent is admitted only in format 1-2, which were
written before H-0083 (P2). When present it must list ``fit_result.pkl`` and
``refit_model.pkl`` (P3). Every refusal here happens before any pickle is read.

Names are compared exactly, case included, with the fixed artifact names; a
manifest key is never joined to the artifact path, so an unknown key, a
path-like one included, is ignored (P5).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.persistence.exporter import CHECKSUM_ALGORITHM

FIT_RESULT_FILE = "fit_result.pkl"
REFIT_MODEL_FILE = "refit_model.pkl"
ANALYSIS_CONTEXT_FILE = "analysis_context.pkl"

#: Pickles every manifest must list (P3).
REQUIRED_FILES = (FIT_RESULT_FILE, REFIT_MODEL_FILE)

#: The last format whose writers predate H-0083 and so carry no checksums (P2).
LAST_FORMAT_WITHOUT_CHECKSUMS = 2

_DIGEST = re.compile(r"[0-9a-f]{64}")


def _refuse(path: str | Path, reason: str, message: str, **context: Any) -> LizyMLError:
    return LizyMLError(
        code=ErrorCode.DESERIALIZATION_FAILED,
        user_message=f"Stored checksums: {message}",
        context={"path": str(path), "reason": reason, **context},
    )


def _is_well_formed(checksums: Any) -> bool:
    if type(checksums) is not dict or set(checksums) != {"algorithm", "files"}:
        return False
    files = checksums["files"]
    if type(checksums["algorithm"]) is not str or type(files) is not dict:
        return False
    return all(
        type(name) is str and type(digest) is str and _DIGEST.fullmatch(digest)
        for name, digest in files.items()
    )


def declared_files(
    metadata: dict[str, Any], format_version: int, path: str | Path
) -> dict[str, str] | None:
    """The file -> digest map the manifest declares, or ``None`` for a legacy one.

    ``None`` means the artifact is format 1-2 without ``checksums``: its files
    are read unverified, as they were before H-0083.

    Raises:
        LizyMLError: ``DESERIALIZATION_FAILED`` (context ``reason``) for a
            malformed manifest (``checksums_malformed``), an unsupported
            algorithm (``checksums_algorithm``), a format 3-4 artifact without
            one (``checksums_missing``), or a required pickle it does not list
            (``required_file_unlisted``).
    """
    if "checksums" not in metadata:
        if format_version <= LAST_FORMAT_WITHOUT_CHECKSUMS:
            return None
        raise _refuse(
            path,
            "checksums_missing",
            f"a format-{format_version} metadata.json must record checksums; only "
            f"format 1-{LAST_FORMAT_WITHOUT_CHECKSUMS} artifacts predate them.",
            format_version=format_version,
        )
    checksums = metadata["checksums"]
    if not _is_well_formed(checksums):
        raise _refuse(
            path,
            "checksums_malformed",
            f"expected {{'algorithm': {CHECKSUM_ALGORITHM!r}, 'files': "
            "{<name>: <64 lowercase hex digits>}}, got a malformed "
            f"{type(checksums).__name__}.",
            type=type(checksums).__name__,
        )
    if checksums["algorithm"] != CHECKSUM_ALGORITHM:
        raise _refuse(
            path,
            "checksums_algorithm",
            f"unsupported algorithm {checksums['algorithm']!r}; expected "
            f"{CHECKSUM_ALGORITHM!r}.",
            algorithm=checksums["algorithm"],
        )
    files: dict[str, str] = checksums["files"]
    for name in REQUIRED_FILES:
        if name not in files:
            raise _refuse(
                path,
                "required_file_unlisted",
                f"'{name}' is not listed.",
                file=name,
            )
    return files
