"""SplitIndices, RunMeta, and DataFingerprint — auxiliary artifacts for FitResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


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


@dataclass
class SplitIndices:
    """Index arrays for every split tier.

    Attributes:
        outer: Per-fold ``(train_idx, valid_idx)`` tuples for outer CV.
        inner: Per-fold inner validation indices (None when not used).
        calibration: Per-fold calibration CV indices (None when not used).
        time_range: Per-fold time period info (None for non-time-series splits).
            Each entry is a dict with keys: ``fold``, ``train_start``,
            ``train_end``, ``valid_start``, ``valid_end``.
    """

    outer: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]
    inner: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] | None
    calibration: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] | None
    time_range: list[dict[str, Any]] | None = field(default=None)


@dataclass
class RunMeta:
    """Runtime metadata captured at fit time.

    Attributes:
        lizyml_version: The installed version of LizyML.
        python_version: Python runtime version string.
        deps_versions: Mapping of key dependency names to their versions.
        config_normalized: The normalized config dict used for this run.
        config_version: Schema version taken from the config.
        run_id: UUID string uniquely identifying this run.
        timestamp: ISO 8601 timestamp of when the run was started.
    """

    lizyml_version: str
    python_version: str
    deps_versions: dict[str, str]
    config_normalized: dict[str, object]
    config_version: int
    run_id: str
    timestamp: str
