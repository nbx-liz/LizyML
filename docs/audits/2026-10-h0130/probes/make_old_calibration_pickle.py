"""Write a CalibrationResult pickled by the code before H-0130 (criterion 6(a2)).

Run with an extracted ``e8b4c4e`` tree first on the path, so the pickle comes
from the class as it was before ``BaseCalibrationResult`` existed::

    git archive -o "$OLD/lizyml.tar" e8b4c4e lizyml
    tar -xf "$OLD/lizyml.tar" -C "$OLD"
    # _version.py is generated, not tracked: write __version__ = "0.19.0"
    # and __version_tuple__ = (0, 19, 0) into "$OLD/lizyml/_version.py".
    PYTHONPATH="$OLD" uv run python make_old_calibration_pickle.py OUT.pkl

The fixture ``tests/test_architecture/data/calibration_result_e8b4c4e.pkl``
was written this way on 2026-10-14; the script printed the old tree's path.

The script refuses to write when the imported lizyml already has the marker,
so it cannot silently produce a pickle from the new code.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

import lizyml.calibration.base as base
from lizyml.calibration.cross_fit import CalibrationResult
from lizyml.calibration.platt import PlattCalibrator


def main() -> int:
    if hasattr(base, "BaseCalibrationResult"):
        print(f"refused: {base.__file__} already defines BaseCalibrationResult")
        return 1
    result = CalibrationResult(
        c_final=PlattCalibrator(),
        calibrated_oof=np.array([0.25, 0.75], dtype=np.float64),
        method="platt",
        split_indices=[(np.array([0], dtype=np.intp), np.array([1], dtype=np.intp))],
    )
    out = Path(sys.argv[1])
    out.write_bytes(pickle.dumps(result, protocol=4))
    print(f"wrote {out} from {base.__file__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
