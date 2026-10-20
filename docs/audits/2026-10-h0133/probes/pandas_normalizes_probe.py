"""H-0133: values outside V that pandas turns into V values before the encoder.

A fit casts an object column to ``category``; the encoder learns the result.
For each value below, prints the learned categories' types and whether the
encoder sees a value outside V. Such a value never reaches the export check.

Usage: uv run python pandas_normalizes_probe.py
"""

from __future__ import annotations

import datetime as dt
import zoneinfo

import pandas as pd

from lizyml.core.category_values import refusal


class _Str(str):
    pass


CASES = {
    "complex NaN": [complex(float("nan"), 1), complex(1, 1)],
    "str subclass": [_Str("a"), "b"],
    "datetime with a no_cache ZoneInfo": [
        dt.datetime(2024, 1, 1, tzinfo=zoneinfo.ZoneInfo.no_cache("Asia/Tokyo")),
        dt.datetime(2024, 1, 2, tzinfo=zoneinfo.ZoneInfo.no_cache("Asia/Tokyo")),
    ],
}
print(f"pandas {pd.__version__}")
for name, values in CASES.items():
    cats = list(pd.Series(values * 3, dtype=object).astype("category").cat.categories)
    kinds = sorted({type(v).__name__ for v in cats})
    outside = [refusal(v) for v in cats if refusal(v) is not None]
    print(f"{name}: {len(cats)} categories of {kinds}; outside V: {outside}")
