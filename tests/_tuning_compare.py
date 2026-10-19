"""Type-exact recursive equality for tuning values (H-0134 P7).

``==`` on the tuning dataclasses is not enough. A NaN score never equals
itself, so a correct round trip would fail. ``int`` / ``float`` / ``bool``
compare equal across types, so a wrong one would pass. And dict order, which
decides ``tuning_table``'s column order, is not compared at all.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any


def same_value(a: Any, b: Any) -> bool:
    """Equal types, NaN equal to NaN, dict key order and sequence order kept."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict):
        return list(a) == list(b) and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(
            same_value(x, y) for x, y in zip(a, b, strict=True)
        )
    if dataclasses.is_dataclass(a):
        return all(
            same_value(getattr(a, f.name), getattr(b, f.name))
            for f in dataclasses.fields(a)
        )
    return bool(a == b)
