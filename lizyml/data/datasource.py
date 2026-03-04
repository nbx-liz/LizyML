"""DataSource: read-only data ingestion from CSV, Parquet, or DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError


def read(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read a DataFrame from a file path or return a copy of an existing DataFrame.

    This function is intentionally read-only: it does not modify the source.

    Args:
        source: A CSV file path, a Parquet file path, or an existing DataFrame.

    Returns:
        A ``pd.DataFrame``.

    Raises:
        LizyMLError: With ``DATA_SCHEMA_INVALID`` when the file cannot be read
            or the format is unsupported.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()

    path = Path(source)
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        raise LizyMLError(
            ErrorCode.DATA_SCHEMA_INVALID,
            user_message=f"Unsupported file format '{suffix}': {path}",
            context={"path": str(path)},
        )
    except LizyMLError:
        raise
    except Exception as exc:
        raise LizyMLError(
            ErrorCode.DATA_SCHEMA_INVALID,
            user_message=f"Failed to read data from: {path}",
            debug_message=str(exc),
            cause=exc,
            context={"path": str(path)},
        ) from exc
