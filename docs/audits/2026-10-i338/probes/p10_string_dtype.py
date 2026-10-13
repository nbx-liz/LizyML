"""#338: which object columns the native pipeline treats as categorical when
``auto_categorical`` is off (``pd.api.types.is_string_dtype``)."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pandas as pd

for label, values in (
    ("str", ["a", "b"]),
    ("bytes", [b"a", b"b"]),
    ("str + bytes", ["a", b"b"]),
    ("Decimal", [Decimal("1.5"), Decimal("2.5")]),
    ("Timestamp", [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]),
    ("date", [dt.date(2020, 1, 1), dt.date(2020, 1, 2)]),
):
    print(f"{label:<12} is_string_dtype={pd.api.types.is_string_dtype(pd.Series(values, dtype=object))}")
