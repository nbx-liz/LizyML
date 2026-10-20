"""H-0136 probe (#372): which serialization makes the pair collide.

Checks the claim of the design critique: the collision comes from joblib (the
pickled pipeline state), not from the H-0133 codec. For the zoneinfo fold=1 /
pytz pair, print hash equality and whether ``set_categories`` accepts the list
after (a) no round trip, (b) joblib, (c) the H-0133 codec. Also check the
critique's mixed Decimal / np.int8 claim.

Usage: PYTHONHASHSEED=<n> python roundtrip_cause.py
"""

from __future__ import annotations

import datetime as dt
import io
import zoneinfo
from decimal import Decimal

import joblib
import numpy as np
import pandas as pd
import pytz

from lizyml.core.category_values import decode, encode

ny = zoneinfo.ZoneInfo("America/New_York")
a = pd.Timestamp(dt.datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=ny))
b = pd.Timestamp(
    pytz.timezone("America/New_York").localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=False)
)


def joblib_trip(v: object) -> object:
    buf = io.BytesIO()
    joblib.dump(v, buf)
    buf.seek(0)
    return joblib.load(buf)


def accepts(cats: list[object]) -> str:
    try:
        pd.Series(cats, dtype=object).astype("category").cat.set_categories(cats)
        return "ok"
    except Exception as exc:  # noqa: BLE001 -- the outcome is the measurement
        return type(exc).__name__


for name, cats in {
    "none": [a, b],
    "joblib": joblib_trip([a, b]),
    "codec": [decode(encode(a)), decode(encode(b))],
}.items():
    x, y = cats  # type: ignore[misc]
    print(
        f"{name:7s} equal={x == y} hash_equal={hash(x) == hash(y)} "
        f"b.tz={type(y.tzinfo).__name__} b.fold={y.fold} set_categories={accepts(list(cats))}"
    )

print("Decimal('1') == np.int8(1):", end=" ")
try:
    print(Decimal("1") == np.int8(1))
except TypeError as exc:
    print("TypeError", exc)
print("np.int8(1) == Decimal('1'):", np.int8(1) == Decimal("1"))
