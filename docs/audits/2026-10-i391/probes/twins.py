"""#391: the per-NumPy collapsing value and its key-equal V-safe twin.

For each candidate prints, on the running NumPy: whether today's rule and the
public round trip admit it, and whether the twin's identity key equals it.

    PYTHONPATH=<tree> python -B docs/audits/2026-10-i391/probes/twins.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from lizyml.core import category_values as cv

PAIRS = {
    "pair": (
        frozenset({np.uint64(4), np.timedelta64(4, "s")}),
        frozenset({4, dt.timedelta(seconds=4)}),
    ),
    "big": (
        frozenset({np.uint64(9265762820973641783), np.timedelta64(4, "ms")}),
        frozenset({9265762820973641783, dt.timedelta(milliseconds=4)}),
    ),
}


def reads_back(value: object) -> bool:
    try:
        cv.decode(cv.encode(value))
        return True
    except Exception:  # noqa: BLE001 -- any failure means it does not read back
        return False


print("numpy", np.__version__)
for name, (v, twin) in PAIRS.items():
    same = cv.category_identity_key(v) == cv.category_identity_key(twin) if cv.refusal(v) is None else "n/a"
    print(
        f"{name}: v in V today={cv.refusal(v) is None}, v reads back={reads_back(v)}; "
        f"twin in V={cv.refusal(twin) is None}, twin reads back={reads_back(twin)}, same key={same}"
    )
