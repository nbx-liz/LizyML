"""H-0136: the prototype rule on the measured cases and the stable controls.

Positive cases must break; the controls -- equal-but-hash-apart pairs that
survive both serializers, per the Codex analysis -- must not.

Usage: python rule_controls.py   (run from this directory, repository on the path)
"""

from __future__ import annotations

import datetime as dt
import zoneinfo

import numpy as np
import pandas as pd
from h0136_rule import declared_breaks, state_breaks

from lizyml.features.encoders.categorical_encoder import CategoricalEncoder

NY = zoneinfo.ZoneInfo("America/New_York")
FOLD1 = pd.Timestamp(dt.datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=NY))
FIXED = pd.Timestamp("2024-11-03 01:30", tz=dt.timezone(dt.timedelta(hours=-5)))
CASES: dict[str, tuple[bool, list]] = {
    "uint64 + timedelta64(M)": (True, [np.uint64(4), np.timedelta64(4, "M")]),
    "zoneinfo fold=1 + fixed offset (control)": (False, [FOLD1, FIXED]),
    "1 + timedelta64(1, s) (control)": (False, [1, np.timedelta64(1, "s")]),
    "strings (control)": (False, ["a", "b"]),
    "summer zoneinfo (control)": (False, [pd.Timestamp("2024-06-01 01:30", tz=NY)]),
}
try:
    import pytz

    _ny = pytz.timezone("America/New_York")
    PYTZ = pd.Timestamp(_ny.localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=False))
    CASES["pytz DST standard"] = (True, [PYTZ])
    CASES["pytz DST standard in a frozenset"] = (True, [frozenset({PYTZ}), "x"])
    CASES["zoneinfo fold=1 + pytz (#372)"] = (True, [FOLD1, PYTZ])
    CASES["pytz summer (control)"] = (
        False,
        [pd.Timestamp(_ny.localize(dt.datetime(2024, 6, 1, 1, 30)))],
    )
except ImportError:
    pass

print(f"pandas={pd.__version__}")
wrong = 0
for name, (expect, values) in CASES.items():
    df = pd.DataFrame({"c": pd.Series([*values, "x"], dtype=object)})
    state = {"encoder": CategoricalEncoder().fit(df, ["c"]).get_state()}
    breaks, fails = state_breaks(state)
    declared = declared_breaks({"c": [*values, "x"]})
    got = bool(breaks)
    wrong += got != expect
    print(f"{'ok ' if got == expect else 'BAD'} {name}: breaks={breaks} fails={fails} declared={declared}")
print(f"wrong={wrong}")
