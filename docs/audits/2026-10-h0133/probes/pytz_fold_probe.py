"""H-0133 confirmation review: a pytz-backed Timestamp at a DST fold.

Compares the original with two reconstructions -- through ``ZoneInfo(name)``
(revision 3's rule) and through ``datetime.timezone`` at the value's own UTC
offset (the proposed fix) -- for equality, hash and dict lookup, both folds.
Needs pytz (pandas 2.x depends on it; pandas 3.0 does not).

Usage (the recorded run used a temporary venv with pandas 3.0.1 and pytz):
    uv venv <dir> && uv pip install --python <dir>/bin/python pandas==3.0.1 pytz
    <dir>/bin/python pytz_fold_probe.py
"""

from __future__ import annotations

import datetime as dt
import zoneinfo

import pandas as pd
import pytz

NAME = "America/New_York"
print(f"pandas {pd.__version__}, pytz {pytz.__version__}")
for fold in (0, 1):
    naive = dt.datetime(2024, 11, 3, 1, 30)
    orig = pd.Timestamp(pytz.timezone(NAME).localize(naive, is_dst=(fold == 0)))
    via_zone = pd.Timestamp(orig.value, tz="UTC").tz_convert(zoneinfo.ZoneInfo(NAME))
    via_offset = pd.Timestamp(orig.value, tz="UTC").tz_convert(dt.timezone(orig.utcoffset()))
    for label, back in (("ZoneInfo(name)", via_zone), ("timezone(offset)", via_offset)):
        print(
            f"fold {fold} {label:16} eq={orig == back!s:5} hash_eq={hash(orig) == hash(back)!s:5} "
            f"dict_hit={({orig: 1}.get(back) == 1)!s:5}"
        )
