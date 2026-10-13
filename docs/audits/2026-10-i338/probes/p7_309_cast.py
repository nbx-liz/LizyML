"""#309: what an upcast of float16 / longdouble arrivals does, value by value.

Checks (1) float16 -> float64 is exact for every float16, (2) which longdouble
values change under longdouble -> float64, (3) what the encoder's unseen check
sees for each arrival dtype (the set it compares with the integer categories).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (1) float16 -> float64 round-trips every finite float16 bit pattern.
bits = np.arange(0, 2**16, dtype=np.uint16)
f16 = bits.view(np.float16)
finite = f16[np.isfinite(f16)]
back = finite.astype(np.float64).astype(np.float16)
print("float16 finite values:", finite.size, "round-trip exact:", bool(np.array_equal(back, finite)))

# (2) longdouble values float64 cannot hold.
print("longdouble:", np.dtype(np.longdouble), "eps", np.finfo(np.longdouble).eps)
ld = np.array([3, 2**53 + 1, 2**63 - 1], dtype=np.longdouble)
ld = np.append(ld, np.longdouble(3) + np.finfo(np.longdouble).eps * 3)
f64 = ld.astype(np.float64)
for x, y in zip(ld, f64):
    print(f"  {x!r:>40} -> {y!r:<24} exact={np.longdouble(y) == x}")

# (3) the encoder's unseen set per arrival dtype, against int categories 0..4.
known = [0, 1, 2, 3, 4]
for dt in ("float16", "float32", "float64", np.longdouble):
    s = pd.Series([0, 1, 2, 3, 4, 2.5], dtype=dt)
    current = set(s.dropna().unique().tolist())
    types = sorted({type(v).__name__ for v in s.dropna().unique().tolist()})
    print(f"  {np.dtype(dt)!s:<10} unique types={types} unseen={sorted(current - set(known))}")
    for label, step in (("astype(category)", lambda t: t.astype("category")),):
        try:
            step(s)
            print(f"    {label}: ok")
        except Exception as exc:  # noqa: BLE001 - the probe reports what escapes
            print(f"    {label}: {type(exc).__name__}: {exc}")
    try:
        s.astype(object).astype("category").cat.set_categories(known)
        print("    object -> category -> set_categories: ok")
    except Exception as exc:  # noqa: BLE001
        print(f"    object -> category -> set_categories: {type(exc).__name__}: {exc}")
