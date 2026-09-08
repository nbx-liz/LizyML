"""Deciding whether two spellings of one parameter carry the same value.

Both callers -- the facade's duplicate-spelling refusal and the estimator
adapter's identity pop -- ask the same question, and they share this function so
that the same call cannot be accepted by one and refused by the other.

Its domain is closed
--------------------
Every value reaching here has passed :mod:`lizyml.core.param_domain`, which
normalises each parameter value at the surface it was written on and refuses
anything outside a small set of plain types. So the values compared here are
``None``, ``bool``, ``int``, ``float``, ``str``, ``pathlib.Path``, a ``list``
of those, or a ``dict`` with ``str`` keys holding those -- and nothing else. The
mapping is a LizyML-level form the adapter consumes (a metric entry) rather than
something LightGBM is sent.

That closure is why this module is short. An earlier form of it was asked the
same question about *any* Python object, and five consecutive review rounds
(H-0094 rounds 16-20) each found one more object whose ``__format__``,
``__class__``, ``tolist`` or ``__eq__`` answered in a way it had not
anticipated. Every fix was right about the object the round named and silent
about the next one, because the domain had no boundary and no finite set of
guards could close it. H-0095 gave it one; what is left here is the part that
was never about hostile objects.

What is left, and why
---------------------
Two things that ``==`` alone still gets wrong on plain values:

* **A sequence and its comma-separated text form are one value.** LightGBM
  writes every sequence parameter as ``",".join(...)`` and passes a ``str``
  through unchanged, so ``feature_contri: [1, 2]`` and
  ``feature_penalty: "1,2"`` are the same bytes on the wire. Refusing that pair
  was a false refusal on ordinary input (H-0094 review round 13).
* **The comparison is elementwise, not textual.** The wire form is not a
  canonical form: ``[1.0, 2.0]`` joins to ``"1.0,2.0"`` and ``[1, 2]`` to
  ``"1,2"``, and the two are the same value. Comparing the joined strings would
  reinstate exactly the false refusal this exists to remove -- which is why
  option E was executed and rejected when H-0095 was decided.
"""

from __future__ import annotations

from typing import Any


def _wire_elements(value: Any) -> list[Any] | None:
    """The elements LightGBM would join for this value, or ``None`` for neither.

    ``_param_dict_to_str`` writes a joined list for a sequence and ``str(val)``
    for a scalar, so a value's wire form is a sequence of elements either way --
    a scalar being a sequence of one. Returning those elements makes the
    comparison against a text form the same question for every type.

    ``None`` is excluded, because the serialiser skips a ``None`` entirely: it
    means "not sent", which is not the string ``"None"``. A ``str`` is excluded
    because it is the *other* operand of that comparison, not a sequence of
    parameter elements.
    """
    if value is None or isinstance(value, str):
        return None
    if isinstance(value, list):
        return value
    return [value]


def _comma_form_matches(text: Any, sequence: Any) -> bool | None:
    """Compare a comma-separated text against a sequence, elementwise.

    Returns ``True`` when the two are one value, ``False`` when they are two,
    and ``None`` when this comparison has no opinion and the caller should fall
    through to ordinary equality.

    Elements are compared as numbers first and as text second, because the wire
    form is not canonical: ``0.5`` writes ``0.5`` and ``"0.50"`` writes
    ``0.50``, and they are one value. A member that is itself a ``list`` --
    ``interaction_constraints`` -- is declined rather than guessed at: the
    serialiser writes it with its own bracketed form, which this does not parse.
    """
    if not isinstance(text, str):
        return None
    elements = _wire_elements(sequence)
    if elements is None:
        return None
    parts = text.split(",")
    if len(parts) != len(elements):
        return False
    for part, element in zip(parts, elements, strict=True):
        if isinstance(element, list):
            return None
        try:
            if float(part) == float(element):
                continue
        except (TypeError, ValueError, OverflowError):
            # `OverflowError` too: an `int` has no width in Python, so `10**400`
            # is an ordinary accepted value whose `float()` cannot exist. Review
            # round 21 measured it escaping a function declared total over the
            # accepted set -- the declaration was right and the code was one
            # exception short of it.
            pass
        if part.strip() != str(element).strip():
            return False
    return True


def values_differ(first: Any, second: Any) -> bool:
    """Return whether the two are *not* the same value.

    Args:
        first: A normalised parameter value.
        second: Another normalised parameter value.

    Returns:
        ``True`` when they differ.

    The order is: identity, then the comma-form question in both directions,
    then ordinary equality. Identity comes first because it is the case the
    callers make most often -- a parameter written under one spelling compared
    with itself -- and because it is the only answer that needs nothing else.

    **This is total over the values ``param_domain`` admits, and it raises on
    none of them** -- including an ``int`` too large to convert to ``float``,
    which Python allows and review round 21 measured escaping. That is a
    finite, enumerable claim, checked by executing it
    over the accepted set rather than argued from the code. The bound it
    replaces was quantified over every Python object, which is not satisfiable
    and was itself the DC7 that kept the review loop running (H-0095).

    Note:
        ``float("nan")`` is not equal to itself, so a parameter written twice as
        two separate ``nan`` values is reported as differing. That is the honest
        answer: nothing can establish those are the same value, and LightGBM
        refuses a NaN parameter anyway. A single ``nan``, being one object, is
        caught by the identity step.
    """
    if first is second:
        return False

    for text, sequence in ((first, second), (second, first)):
        matched = _comma_form_matches(text, sequence)
        if matched is not None:
            return not matched

    # `bool(...)` rather than the expression itself: the accepted set holds no
    # value whose `__eq__` answers with anything but a bool, and stating that
    # here is what keeps the return type honest to the annotation.
    return bool(first != second)
