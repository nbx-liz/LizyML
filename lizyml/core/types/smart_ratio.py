"""The ranges of the ratio smart parameters (H-0121 amendment 2, H-0130).

``config/schema.py`` validates the ratio fields against these ranges, and the
LightGBM provider builds its smart value rules from them, so a config field and
a search dimension cannot disagree about a ratio (DC3). The declaration lives in
Layer 0 so that both Layer 1 categories read it without importing each other
(BLUEPRINT §2.1); ``config/schema.py`` re-exports both names.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["SMART_RATIO_RANGES", "RatioRange"]


class RatioRange(NamedTuple):
    """An interval open at ``low``, closed at ``high`` when ``upper_inclusive``."""

    low: float
    high: float
    upper_inclusive: bool

    def contains(self, value: float) -> bool:
        """Whether *value* lies in the interval (NaN never does)."""
        upper_ok = value <= self.high if self.upper_inclusive else value < self.high
        return bool(value > self.low and upper_ok)

    def describe(self) -> str:
        """The interval in the usual notation, e.g. ``(0, 1]``."""
        close = "]" if self.upper_inclusive else ")"
        return f"({self.low:g}, {self.high:g}{close}"


#: The ranges of the ratio smart parameters: the one declaration both the
#: config's validation and the estimator provider's smart value rule read
#: (H-0121 amendment 2, policy 1), so a config field and a search dimension
#: cannot disagree about a ratio (DC3).
SMART_RATIO_RANGES: dict[str, RatioRange] = {
    "num_leaves_ratio": RatioRange(low=0.0, high=1.0, upper_inclusive=True),
    "min_data_in_leaf_ratio": RatioRange(low=0.0, high=1.0, upper_inclusive=False),
    "min_data_in_bin_ratio": RatioRange(low=0.0, high=1.0, upper_inclusive=False),
}
