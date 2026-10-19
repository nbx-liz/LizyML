"""Check v_reference on fixed values, including a caller tzinfo with a ``zone`` property.

The property counts its calls; P1 requires that judging the value runs none of
the caller's code (Proposal review round 2). Exit status 1 on any mismatch.

Usage: uv run python v_reference_check.py
"""

from __future__ import annotations

import datetime as dt
import decimal
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_reference import reason  # noqa: E402

CALLS = {"zone": 0}


class CallerTz(dt.tzinfo):
    @property
    def zone(self) -> str:
        CALLS["zone"] += 1
        return "Asia/Tokyo"

    def utcoffset(self, d):  # type: ignore[no-untyped-def]
        return dt.timedelta(hours=9)


nest = 1
for _ in range(8):
    nest = (nest,)

EXPECT_INSIDE = {
    "str": "a",
    "np.int64": np.int64(1),
    "bytes": b"a",
    "Decimal": decimal.Decimal("1.5"),
    "Timestamp zoneinfo": pd.Timestamp("2024-01-01", tz="Asia/Tokyo"),
    "datetime fixed offset": dt.datetime(2024, 1, 1, tzinfo=dt.timezone(dt.timedelta(microseconds=1))),
    "date": dt.date(2024, 1, 1),
    "datetime64 2s": np.datetime64(4, "2s"),
    "tuple": (1, "a"),
    "frozenset": frozenset({1, 2}),
    "8 containers": nest,
}
EXPECT_OUTSIDE = {
    "nested NaN": (1, float("nan")),
    "Decimal NaN": decimal.Decimal("NaN"),
    "complex NaN": complex(float("nan"), 0),
    "NaT": np.datetime64("NaT"),
    "9 containers": (nest,),
    "str subclass": type("S", (str,), {})("a"),
    "longdouble": np.longdouble(1),
    "caller tzinfo": dt.datetime(2024, 1, 1, tzinfo=CallerTz()),
}

bad = 0
for name, value in EXPECT_INSIDE.items():
    r = reason(value)
    print(f"inside  {name:22} -> {r}")
    bad += r is not None
for name, value in EXPECT_OUTSIDE.items():
    r = reason(value)
    print(f"outside {name:22} -> {r}")
    bad += r is None
print(f"caller zone property calls: {CALLS['zone']}")
bad += CALLS["zone"] != 0
print(f"mismatches: {bad}")
raise SystemExit(1 if bad else 0)
