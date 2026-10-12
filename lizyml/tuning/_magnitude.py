"""The magnitudes a search dimension may have (H-0124 amendment 1, policy 9).

Optuna samples every number as a ``float`` (``docs/audits/2026-10-tuning-resume/
probes/optuna_magnitude_facts.txt``):

* a ``float`` range whose width ``high - low`` is not a finite ``float`` fails
  to sample (``OverflowError: Range exceeds valid bounds``), even when each
  bound is finite;
* an ``int`` range is widened by half a step on each side and sampled as a
  ``float``. Half-integers are exact only below ``2**52``; from ``2**53``
  neighbouring integers share a ``float`` and the search silently gets coarser,
  and far enough out it fails.

So a ``float`` dimension's exact width must be at most ``sys.float_info.max``
and an ``int`` dimension's ``low``, ``high`` and ``step`` at most ``2**52`` in
absolute value. These predicates are pure and import nothing from LizyML, so
the format readers, the expansion and the lineage codec share them.
"""

from __future__ import annotations

import math
import sys
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np

__all__ = [
    "FLOAT_WIDTH_LIMIT",
    "INT_LIMIT",
    "exact",
    "finite_float",
    "int_too_large",
    "width_too_large",
]

INT_LIMIT: int = 2**52
"""The largest absolute value of an ``int`` dim's ``low``, ``high`` or ``step``."""

FLOAT_WIDTH_LIMIT: Fraction = Fraction(sys.float_info.max)
"""The largest exact width ``high - low`` of a ``float`` dimension."""


def exact(value: Any) -> Fraction:
    """*value* as an exact rational: an integer as itself, else its float's text.

    A ``float`` (or a numpy floating scalar) is read through the shortest
    ``str`` of its ``float`` value, the reading Optuna's own step arithmetic
    uses; an ``int`` or numpy integer is exact already. The value must be
    finite.
    """
    if type(value) is int or isinstance(value, np.integer):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    return Fraction(Decimal(str(float(value))))


def int_too_large(value: int) -> bool:
    """Whether an ``int`` dimension's *value* is past ``INT_LIMIT``."""
    return abs(value) > INT_LIMIT


def _binary(value: Any) -> Fraction:
    """The exact value a finite bound has as Optuna receives it."""
    if type(value) is int or isinstance(value, np.integer):
        return Fraction(int(value))
    return Fraction(float(value))


def width_too_large(low: Any, high: Any) -> bool:
    """Whether the exact width ``high - low`` of finite bounds is past the limit.

    The width of the values themselves (a ``float`` read as its binary value,
    not as its shortest text the grid uses): when it is at most the largest
    float, the ``float`` subtraction Optuna does is finite too. Through the
    shortest text, ``[-max / 2, max / 2]`` would read wider than ``max`` though
    its ``float`` width is exactly ``max``.
    """
    return _binary(high) - _binary(low) > FLOAT_WIDTH_LIMIT


def finite_float(value: Any) -> bool:
    """Whether *value* is a number whose ``float`` is finite.

    An ``int`` too large for a ``float`` is not (``float()`` raises).
    """
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False
