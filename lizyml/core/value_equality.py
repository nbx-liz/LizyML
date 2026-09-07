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
    """Return ``len(value)``, or ``None`` when the value has no usable length.

    Every exception is a "no length", not only ``TypeError``: a user object may
    define a ``__len__`` that fails, and this function exists to make a decision
    safely rather than to diagnose the value.
    """
    try:
        return len(value)
    except Exception:  # noqa: BLE001 - a user value may define a failing __len__
        return None


def _printed_forms_differ(first: Any, second: Any) -> bool:
    """Compare printed forms, and answer "the same" when even that fails.

    ``repr`` is the last thing two values have in common, and it is not
    guaranteed either: an object may define a ``__repr__`` that raises. This is
    the function's floor, so it cannot propagate.
    """
    try:
        return repr(first) != repr(second)
    except Exception:  # noqa: BLE001 - a user value may define a failing __repr__
        return False


def values_differ(first: Any, second: Any) -> bool:
    """Return whether the two are *not* the same value.

    Args:
        first: A parameter value.
        second: Another parameter value.

    Returns:
        ``True`` when they differ.

    The order of the checks is the point:

    0. **Identity.** One object is the same value as itself, and no comparison
       can improve on that. This is also the case the callers make most often:
       a parameter written under one spelling is compared with itself, and
       before this it went the long way round and could raise on the way.
    1. **Length, when both have one.** Elementwise comparison broadcasts, so
       ``[1.0, 1.0]`` and ``[1.0]`` would otherwise compare equal, and an empty
       sequence would agree with anything by a vacuous ``all()``.
    2. **The comparison as a truth value**, which covers ordinary values and the
       library scalars whose result is not a ``bool`` but converts to one.
    3. **Elementwise**, requiring every element to be equal, for the array-like
       results that cannot convert to a single truth value -- and only when
       every element can state a truth of its own, since iterating a comparison
       does not always yield the comparison.
    4. **The printed forms**, for anything that raised on the way -- an
       ``__eq__`` that fails, or a shape whose elements are themselves arrays.
       A weaker answer than equality, and the only one both values always have.
    5. **"The same"**, when even the printed forms raise.

    **This function does not raise.** Step 5 is what makes that true by
    construction rather than by having thought of enough value types: three
    review rounds each found one more shape that got past the previous
    enumeration, so the last step is a decision rather than another case. It
    answers "the same" and not "different" on purpose -- the refusal it feeds
    exists to catch a parameter the caller wrote twice, and refusing a value
    nothing can analyse would block a legitimate call to prevent an ambiguity
    that may not be there.

    Note:
        ``float("nan")`` is not equal to itself, so a parameter written twice as
        ``nan`` is reported as differing. That is the honest answer: nothing can
        establish those are the same value, and LightGBM refuses a NaN parameter
        anyway. A single ``nan``, being one object, is caught by step 0.
    """
    if first is second:
        return False

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
        return _printed_forms_differ(first, second)

    try:
        return not bool(equal)
    except Exception:  # noqa: BLE001 - see below
        # Every exception, not the two an array raises. A comparison result is
        # an object the caller supplied too, and its `__bool__` may fail for a
        # reason of its own; catching only `ValueError` and `TypeError` let
        # that escape a function declared not to raise (H-0094, review round 7).
        pass  # Reduce it below, or fall through to the printed forms.

    try:
        elements = list(equal)
    except Exception:  # noqa: BLE001 - nested arrays, exotic containers
        return _printed_forms_differ(first, second)

    # Iterating a comparison result does not always yield the comparison, and
    # what it yields instead is usually truthy, so the reduction below would
    # read "equal" for two different values. Measured twice, by two different
    # routes: a DataFrame comparison iterates over its **column labels**, and a
    # comparison object that cannot be a boolean but is iterable yields
    # whatever it likes.
    #
    # The property that separates a comparison outcome from junk is that the
    # element can state a truth **of its own**. `bool`, `numpy.bool_` and the
    # library scalars define `__bool__`; a string, a list and a bare object do
    # not -- their truthiness comes from length or from the default, neither of
    # which is an answer to "are these equal". An element without `__bool__`
    # therefore means the iteration is not the comparison, and the printed
    # forms are the honest fallback.
    if not all(hasattr(element, "__bool__") for element in elements):
        return _printed_forms_differ(first, second)

    try:
        return not all(bool(element) for element in elements)
    except Exception:  # noqa: BLE001 - elements that are themselves array-like
        return _printed_forms_differ(first, second)
