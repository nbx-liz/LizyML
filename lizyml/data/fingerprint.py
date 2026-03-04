"""DataFingerprint: compute a lightweight hash to identify a dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataFingerprint:
    """Lightweight fingerprint for verifying dataset identity.

    Attributes:
        row_count: Number of rows.
        column_hash: SHA-256 of sorted ``column_name:dtype`` pairs.
        file_hash: SHA-256 of the raw file bytes (None if not file-based).
    """

    row_count: int
    column_hash: str
    file_hash: str | None = None

    def matches(self, other: DataFingerprint) -> bool:
        """Return True if this fingerprint is compatible with *other*.

        File hash is only compared when both fingerprints have it.
        """
        if self.row_count != other.row_count:
            return False
        if self.column_hash != other.column_hash:
            return False
        if self.file_hash is not None and other.file_hash is not None:
            return self.file_hash == other.file_hash
        return True


def compute(
    df: pd.DataFrame,
    file_path: str | Path | None = None,
) -> DataFingerprint:
    """Compute a DataFingerprint for the given DataFrame.

    Args:
        df: The DataFrame to fingerprint.
        file_path: Optional path to the source file for ``file_hash``.

    Returns:
        ``DataFingerprint`` instance.
    """
    column_hash = _hash_columns(df)
    file_hash = _hash_file(file_path) if file_path is not None else None
    return DataFingerprint(
        row_count=len(df),
        column_hash=column_hash,
        file_hash=file_hash,
    )


def _hash_columns(df: pd.DataFrame) -> str:
    """Hash column names and dtypes in declaration order."""
    parts = [f"{col}:{df[col].dtype}" for col in df.columns]
    payload = "|".join(parts).encode()
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: str | Path) -> str | None:
    """Hash file bytes; return None if the file cannot be read."""
    try:
        data = Path(path).read_bytes()
        return hashlib.sha256(data).hexdigest()
    except OSError:
        return None
