"""SplitIndices and RunMeta — auxiliary artifacts for FitResult."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class SplitIndices:
    """Index arrays for every split tier.

    Attributes:
        outer: Per-fold ``(train_idx, valid_idx)`` tuples for outer CV.
        inner: Per-fold inner validation indices (None when not used).
        calibration: Per-fold calibration CV indices (None when not used).
    """

    outer: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]
    inner: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] | None
    calibration: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] | None


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
