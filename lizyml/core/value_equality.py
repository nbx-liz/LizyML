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


def _as_plain_python(value: Any) -> Any:
    """Convert an array-like to ordinary Python objects, or return it as is.

    ``tolist`` is the documented conversion on ``numpy`` arrays and scalars and
    on ``pandas`` Series: it yields nested lists and Python numbers, whose
    ``==`` answers with a real ``bool``. That is what makes the step below a
    *normalisation* rather than a rule about what iterating an object yields --
    the removed elementwise step guessed at the latter and three rounds refuted
    it, while this converts and then asks the ordinary question again.

    A value with no such conversion, or one whose conversion fails, is returned
    unchanged and decided by the printed forms as before.
    """
    conversion = getattr(value, "tolist", None)
    if not callable(conversion):
        return value
    try:
        return conversion()
    except Exception:  # noqa: BLE001 - a user object may define a failing tolist
        return value


def _printed_forms_differ(first: Any, second: Any) -> bool:
    """Compare printed forms, and answer "the same" when even that fails.

    ``repr`` is the last thing two values have in common, and it is not
    guaranteed either: an object may define a ``__repr__`` that raises. This is
    the function's floor, so it cannot propagate.

    The texts are compared through ``str.__eq__`` rather than with ``!=``.
    ``repr`` may return a **subclass** of ``str``, and a subclass may override
    the comparison, so ``!=`` handed the decision back to the caller's object at
    the very step that exists to escape it -- the fallback returned a list, and
    two values printing identically were refused (H-0094, review round 9).
    Reading the characters is what this step was always meant to do.

    Anything other than a definite "not equal" answers "the same", which is the
    same direction the floor takes: this feeds refusals, and blocking a call on
    an ambiguity nothing here can resolve is the worse error.
    """
    try:
        return str.__eq__(repr(first), repr(second)) is False
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
    3. **The same comparison over plain Python**, when the values convert --
       ``tolist`` on an array or a Series yields nested lists and Python
       numbers, whose ``==`` is an ordinary truth value. A conversion followed
       by the question already asked, not a new rule.
    4. **The printed forms**, for everything else -- an ``__eq__`` that raises,
       and every comparison result that is neither a truth value nor
       convertible. A weaker answer than equality, and the only one both values
       always have.
    5. **"The same"**, when even the printed forms raise.

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

    **What that costs, stated rather than hidden.** Step 3 covers the values
    this library actually passes -- arrays, Series and library scalars -- so
    equal numbers under different dtypes are the same value and a very long
    pair that ``repr`` summarises identically still differs. What is left for
    ``repr`` is the values with no faithful conversion to plain Python: a
    ``DataFrame``, and a caller's own object. Two of those that print alike are
    reported as the same, and the callers then keep the first spelling written.
    That is the direction the floor already chose: this function feeds refusals,
    and answering "the same" declines to block a call rather than blocking one
    on an ambiguity nothing here can resolve. The remaining cost is in the case
    table, executed rather than asserted here, because a limit nothing reaches
    is a limit nobody has checked.

    An earlier form decided every array-like by ``repr``, and review round 11
    reproduced what that cost on ordinary input: ``fit`` refused a call naming
    one parameter twice as ``np.array([1, 2])`` and ``np.array([1.0, 2.0])``,
    values LightGBM accepts individually and which are the same value.

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
        pass

    # The comparison could not be a truth value, so convert each side to plain
    # Python and ask again. Two arrays holding equal numbers under different
    # dtypes print differently, and deciding them by `repr` refused a call that
    # named one value twice -- with ordinary arrays, not adversarial objects
    # (H-0094, review round 11).
    plain_first = _as_plain_python(first)
    plain_second = _as_plain_python(second)
    if plain_first is not first or plain_second is not second:
        try:
            return not bool(plain_first == plain_second)
        except Exception:  # noqa: BLE001 - the conversion is not guaranteed either
            pass

    return _printed_forms_differ(first, second)
