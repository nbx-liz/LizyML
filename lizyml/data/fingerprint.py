"""Compute a lightweight hash to identify a dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from lizyml.core.types.artifacts import DataFingerprint

# Re-export for backward compatibility
__all__ = ["DataFingerprint", "compute"]


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
