"""H-0133: which admitted types survive pandas' cast to ``category``.

For each admitted type, a column of its samples (``SAMPLES`` of
``tests/test_core/test_category_values.py``) is cast as the data builder casts
it (``astype("category")``). Prints the learned categories' types, and the
admitted types that never appear among them (pandas turns them into another
type first). A cast pandas refuses is printed as such.

Usage (repository root): uv run python docs/audits/2026-10-h0133/probes/learned_types_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lizyml.core.category_values import ADMITTED_TYPES  # noqa: E402
from tests.test_core.test_category_values import SAMPLES  # noqa: E402

learned_all: set[type] = set()
print(f"pandas {pd.__version__}")
for kind, samples in SAMPLES.items():
    try:
        cats = list(pd.Series(samples * 3, dtype=object).astype("category").cat.categories)
    except Exception as exc:  # noqa: BLE001 -- reported
        print(f"{kind.__name__:14} pandas refuses: {type(exc).__name__}: {exc}")
        continue
    learned = {type(v) for v in cats}
    learned_all |= learned
    print(f"{kind.__name__:14} -> {sorted(t.__name__ for t in learned)}")
missing = sorted(t.__name__ for t in set(ADMITTED_TYPES) - learned_all)
print(f"admitted types never learned: {missing}")
