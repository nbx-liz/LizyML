"""Comparing two parameter values without assuming what type they are.

A parameter value is whatever the user wrote. Most are numbers or strings, but
``feature_contri`` is a sequence, and a caller may pass a numpy array for it.
``a != b`` on an array returns an **array**, and ``bool()`` of that raises
``ValueError: The truth value of an array with more than one element is
ambiguous`` -- so a check written as ``if a != b`` crashes on a value it was
supposed to be comparing (H-0094, review round 6). It crashed even when the
value appeared once, because the comparison was made against itself.

This module has no imports on purpose: it is Layer 0, and both the facade's
duplicate-spelling refusal and the estimator adapter's identity pop need the
same notion of "the same value". Two notions would mean the same call is
accepted or refused depending on which of them saw it, which is the defect
that made round 5 necessary.
"""

from __future__ import annotations

from typing import Any


def values_differ(first: Any, second: Any) -> bool:
    """Return whether the two are *not* the same value.

    Args:
        first: A parameter value.
        second: Another parameter value.

    Returns:
        ``True`` when they differ. Equality that is not a plain ``bool`` -- an
        elementwise comparison, as numpy and pandas return -- is reduced by
        requiring every element to be equal; anything that cannot be reduced
        that way falls back to comparing the printed forms, which is a weaker
        answer but never an exception.

    Note:
        ``float("nan")`` is not equal to itself, so a parameter written twice
        as ``nan`` is reported as differing. That is the honest answer here:
        nothing can establish that those two are the same value, and LightGBM
        refuses a NaN parameter anyway.
    """
    equal = first == second
    if isinstance(equal, bool):
        return not equal
    try:
        return not all(bool(element) for element in equal)
    except TypeError:
        # A 0-dimensional array, or an object whose equality is neither a bool
        # nor iterable. The printed form is the last thing both values have.
        return repr(first) != repr(second)
