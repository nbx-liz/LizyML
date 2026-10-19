"""H-0133 round 3: equality and hash across tzinfo instances in a DST fold.

For 2024-11-03 01:30 in America/New_York (both folds), compares a value with
its reconstruction in another tzinfo instance of the same zone: a cached
``ZoneInfo``, a ``ZoneInfo.no_cache`` instance and, when installed, pytz. Run
for ``datetime.datetime`` (Python's PEP 495 interzone rule) and ``pd.Timestamp``.
Also checks that a wall-clock-plus-fold reconstruction with the cached
``ZoneInfo`` is equal, and that ``datetime.min``/``max`` with a 23-hour offset
round-trip through wall-clock fields.

Usage: uv run python tz_fold_probe.py
"""

from __future__ import annotations

import datetime as dt
import zoneinfo

import pandas as pd

NAME = "America/New_York"
cached = zoneinfo.ZoneInfo(NAME)
fresh = zoneinfo.ZoneInfo.no_cache(NAME)
backends = {"ZoneInfo cached": cached, "ZoneInfo no_cache": fresh}
try:
    import pytz

    backends["pytz"] = pytz.timezone(NAME)
except ImportError:
    pass


def show(label: str, a: object, b: object) -> None:
    print(f"  {label:44} eq={a == b!s:5} hash_eq={hash(a) == hash(b)!s:5}")


print(f"pandas {pd.__version__}")
for fold in (0, 1):
    print(f"fold {fold}:")
    for name, tz in backends.items():
        if name == "pytz":
            orig = tz.localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=(fold == 0))
        else:
            orig = dt.datetime(2024, 11, 3, 1, 30, fold=fold, tzinfo=tz)
        utc = orig.astimezone(dt.timezone.utc)
        via_utc = utc.astimezone(cached)
        show(f"datetime {name} vs via UTC (cached ZoneInfo)", orig, via_utc)
        if name != "pytz":
            wall = orig.replace(tzinfo=None).replace(tzinfo=cached, fold=orig.fold)
            show(f"datetime {name} vs wall+fold (cached ZoneInfo)", orig, wall)
        ts = pd.Timestamp(orig)
        ts_back = pd.Timestamp(ts.value, tz="UTC").tz_convert(cached)
        show(f"Timestamp {name} vs via UTC (cached ZoneInfo)", ts, ts_back)

print("boundaries through wall-clock fields:")
for value in (
    dt.datetime.min.replace(tzinfo=dt.timezone(dt.timedelta(hours=23))),
    dt.datetime.max.replace(tzinfo=dt.timezone(dt.timedelta(hours=-23))),
):
    us = (value.replace(tzinfo=None) - dt.datetime(1970, 1, 1)) // dt.timedelta(microseconds=1)
    back = (dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=us)).replace(
        tzinfo=dt.timezone(value.utcoffset()), fold=value.fold
    )
    print(f"  {value!r}: equal={value == back}, hash_eq={hash(value) == hash(back)}")
