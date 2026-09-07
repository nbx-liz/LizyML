"""Comparing two parameter values without assuming what type they are.

A parameter value is whatever the user wrote. Most are numbers or strings, but
``feature_contri`` and ``monotone_constraints`` are sequences, and a caller may
pass a numpy array or a numpy scalar for any of them. That makes ``a == b`` an
unreliable expression on its own:

- ``np.array([1.0]) != np.array([1.0])`` yields an **array**, and ``bool()`` of
  a multi-element one raises;
- ``np.float64(0.5) == 0.5`` yields ``np.bool_``, which is not a ``bool``;
- ``np.array([1.0, 1.0]) == np.array([1.0])`` **broadcasts** to all-true, so two
  sequences of different lengths compare equal;
- an object's ``__eq__`` may raise outright.

Each of those was a live defect in an earlier form of this function (H-0094,
review rounds 5 and 6): the first crashed on a value written once, the second
refused a call that meant one thing twice, and the third would have accepted two
different values as the same.

This module depends on nothing but the standard library, so it sits in Layer 0
and both the facade's duplicate-spelling refusal and the estimator adapter's
identity pop use the same notion of "the same value". Two notions would mean the
same call is accepted or refused depending on which of them saw it.
"""

from __future__ import annotations

from typing import Any


def _length_or_none(value: Any) -> int | None:
    """Return ``len(value)``, or ``None`` when the value has no length."""
    try:
        return len(value)
    except TypeError:
        return None


def values_differ(first: Any, second: Any) -> bool:
    """Return whether the two are *not* the same value.

    Args:
        first: A parameter value.
        second: Another parameter value.

    Returns:
        ``True`` when they differ.

    The order of the checks is the point:

    1. **Length, when both have one.** Elementwise comparison broadcasts, so
       ``[1.0, 1.0]`` and ``[1.0]`` would otherwise compare equal, and an empty
       sequence would agree with anything by a vacuous ``all()``.
    2. **The comparison as a truth value**, which covers ordinary values and the
       library scalars whose result is not a ``bool`` but converts to one.
    3. **Elementwise**, requiring every element to be equal, for the array-like
       results that cannot convert to a single truth value.
    4. **The printed forms**, for anything that raised on the way -- an
       ``__eq__`` that fails, or a shape whose elements are themselves arrays.
       A weaker answer than equality, and the only one both values always have.

    Note:
        ``float("nan")`` is not equal to itself, so a parameter written twice as
        ``nan`` is reported as differing. That is the honest answer: nothing can
        establish those are the same value, and LightGBM refuses a NaN parameter
        anyway.
    """
    first_length = _length_or_none(first)
    second_length = _length_or_none(second)
    if (
        first_length is not None
        and second_length is not None
        and first_length != second_length
    ):
        return True

    try:
        equal = first == second
    except Exception:  # noqa: BLE001 - a user value may define a failing __eq__
        return repr(first) != repr(second)

    try:
        return not bool(equal)
    except (ValueError, TypeError):
        pass  # An array-like result: reduce it below.

    try:
        return not all(bool(element) for element in equal)
    except Exception:  # noqa: BLE001 - nested arrays, exotic containers
        return repr(first) != repr(second)
