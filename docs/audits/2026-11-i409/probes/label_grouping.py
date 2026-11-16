"""Probe: does check_duplicate_columns group labels as pandas selection does? (#409)

For each pair of labels, compare (a) pandas' own duplicate verdict
(``Index.duplicated``), (b) whether ``df[first]`` selects more than one
column, and (c) the grouping check_duplicate_columns uses (factorize).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

PAIRS: list[tuple[Any, Any]] = [
    (0, False),
    (1, True),
    (1, 1.0),
    (None, np.nan),
    (np.nan, np.nan),
    (("a", 1), ("a", 1)),
    ("0", 0),
    (np.float64(1.5), 1.5),
    (np.int64(2), 2),
]


def main() -> int:
    for a, b in PAIRS:
        index = pd.Index(["x", a, b], dtype=object)
        dup = bool(index.duplicated(keep=False)[1])
        codes, _ = index.factorize(use_na_sentinel=False)
        grouped = bool(codes[1] == codes[2])
        df = pd.DataFrame(np.zeros((2, 3)))
        df.columns = index
        try:
            selected = df[a]
            many = isinstance(selected, pd.DataFrame)
        except Exception as exc:  # noqa: BLE001 - the probe reports every failure
            many = f"{type(exc).__name__}"
        print(f"{a!r} / {b!r}\tduplicated={dup}\tfactorize={grouped}\tdf[a] many={many}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
