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
    3. **The printed forms**, for everything else -- an ``__eq__`` that raises,
       and every comparison result that is not a truth value. A weaker answer
       than equality, and the only one both values always have.
    4. **"The same"**, when even the printed forms raise.

    **There is no step that inspects the comparison result's contents**, and
    that absence is deliberate. Three review rounds each found one: reducing an
    array-like result elementwise requires knowing that iterating it yields the
    comparison, and nothing about an arbitrary object establishes that. A
    ``DataFrame`` comparison yields its **column labels**, which read as "equal"
    when the labels are truthy -- found with string labels in round 6 and again
    with integer labels in round 8, through two different guards written to
    exclude exactly that. Each guard was a hypothesis about object structure,
    and the next round refuted it. Every step that remains rests on a single
    protocol call on the values themselves, so there is nothing left of that
    kind to refute.

    **What that costs, stated rather than hidden.** Two array-likes whose
    comparison cannot be a truth value are now decided by ``repr``. Two arrays
    holding equal numbers under different dtypes print differently and are
    therefore reported as differing, so a caller who writes one parameter twice,
    once as ``[1, 2]`` and once as ``np.array([1.0, 2.0])``, is refused. The
    refusal names both spellings and both values, so it is legible and the
    caller can settle it. Conversely two arrays that print alike -- past
    ``numpy``'s summarisation threshold, or below its display precision -- are
    reported as the same, and the callers then keep the first spelling written.
    That is the direction the floor already chose: this function feeds refusals,
    and answering "the same" declines to block a call rather than blocking one
    on an ambiguity nothing here can resolve.

    **This function does not raise an ``Exception``.** Step 4 is what makes that
    true by construction rather than by having thought of enough value types.
    Every expression that touches a caller's value is inside a ``try``. A
    ``BaseException`` a caller's value raises -- ``KeyboardInterrupt`` and
    ``SystemExit`` are the ones that matter -- is **not** caught and propagates
    on purpose: swallowing those would make a hung comparison uninterruptible,
    which is a worse failure than the one this bound prevents.

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
        # reason of its own; catching only `ValueError` and `TypeError` let that
        # escape a function declared not to raise (H-0094, review round 7).
        return _printed_forms_differ(first, second)
