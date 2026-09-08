"""The closed set of parameter values that may reach the estimator.

Every parameter value a caller writes -- in ``model.params``, in
``fit(params=)``, in a restored ``tuning best_model_params``, in
``calibration.params`` -- is normalised here, once, at the surface it enters
through, and anything outside the accepted set is refused before training
starts (H-0095).

Why a closed set rather than a tolerant one
-------------------------------------------
:func:`lizyml.core.value_equality.values_differ` has to answer whether two
writings of one parameter mean the same value, and it used to be asked that
question about *any* Python object. Five consecutive review rounds (H-0094,
rounds 16-20) each found one more object whose ``__format__``, ``__class__``,
``tolist`` or ``__eq__`` answered in a way the comparison had not anticipated,
and every fix was correct for the object the round named and silent about the
next one. The defect was not the missing guard; it was that the domain had no
boundary, so no finite set of guards could close it.

What "accepted" is derived from
-------------------------------
LightGBM serialises a parameter dict in ``lightgbm.basic._param_dict_to_str``,
and that function is the authority on what a value *means*, because the text it
writes is the whole of what the trainer sees. It uses two different formatters:

* a **scalar** is written with ``__format__`` (``f"{key}={val}"``);
* a **sequence element** is written with ``str`` (``_to_string``), and a
  ``list`` element of a sequence is written as ``[`` + its own elements joined
  with commas + ``]``.

Those two disagree for a value that overrides one and not the other, which is
why this module distinguishes the two positions instead of normalising by type
alone.

The accepted set is therefore the subset of the serialiser's own accepted set
on which a plain Python stand-in writes the **identical bytes**. Where no plain
value writes the same bytes -- ``str(numpy.float32(1e8))`` is ``1e+08`` and no
Python float prints that -- this module refuses, rather than let training run
on bytes the caller did not write. That refusal is the deliberate half of the
trade: the defect being hunted is training silently on a different value, so
failing loudly at the surface is the acceptable side to fall on.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from lizyml.core.exceptions import ErrorCode, LizyMLError

#: Scalar types that pass through untouched.
#:
#: Matched by **exact type**, never by ``isinstance``. A subclass may override
#: ``__format__`` -- the serialiser's scalar formatter -- and print bytes its
#: base class would not, so a subclass instance can compare equal to a plain
#: value here and then train as something else (H-0094 review round 20). The
#: measured cost of refusing subclasses over this repository's own suite is
#: zero: no parameter value it constructs is one.
PLAIN_SCALAR_TYPES: tuple[type, ...] = (
    type(None),
    bool,
    int,
    float,
    str,
    # `Path()` instantiates a concrete flavour, so the exact types are listed
    # rather than the abstract base.
    pathlib.PurePosixPath,
    pathlib.PureWindowsPath,
    pathlib.PosixPath,
    pathlib.WindowsPath,
)

#: Sequence types the serialiser joins with commas, alongside a 1-D ndarray.
PLAIN_SEQUENCE_TYPES: tuple[type, ...] = (list, tuple, set)

#: What a caller is told they may write.
ACCEPTED_DESCRIPTION = (
    "None, bool, int, float, str, pathlib.Path, a numpy scalar, or a list, "
    "tuple, set or 1-D numpy array of those (whose elements may themselves be "
    "lists of those)"
)


class _Unaccepted(Exception):
    """A value outside the accepted set, carried back to the surface."""

    def __init__(self, value: Any, reason: str) -> None:
        super().__init__(reason)
        self.value = value
        self.reason = reason


def _describe(value: Any) -> str:
    """Name a rejected value by type, without asking the value to print itself.

    A rejected value is by definition one whose ``__format__``/``__str__`` are
    not trusted, so the message is built from its type and never from it.
    """
    return type(value).__name__


def _plain_scalar(value: Any) -> Any:
    """The plain stand-in for a value in the serialiser's *scalar* position.

    The scalar formatter is ``__format__``, and ``.item()`` on a numpy scalar
    preserves it for every dtype measured (see the wire-preservation test).
    """
    if type(value) in PLAIN_SCALAR_TYPES:
        return value
    if isinstance(value, np.generic):
        plain = value.item()
        if type(plain) in PLAIN_SCALAR_TYPES:
            return plain
        raise _Unaccepted(value, f"numpy scalar of dtype {value.dtype} is not plain")
    raise _Unaccepted(value, f"{_describe(value)} is not an accepted scalar")


def _plain_element(value: Any) -> Any:
    """The plain stand-in for a value in the serialiser's *element* position.

    The element formatter is ``str``, so the stand-in is the plain value whose
    ``str`` is the same text. For most numpy scalars ``.item()`` is that value;
    for the reduced-precision floats it is not, because numpy chooses an
    exponent form Python would not -- ``str(numpy.float32(1e8))`` is ``1e+08``
    and ``str(1e8)`` is ``100000000.0`` -- and there the text itself is parsed
    back. Where no plain value prints that text, this refuses.
    """
    if type(value) in PLAIN_SCALAR_TYPES:
        return value
    if isinstance(value, np.generic):
        text = str(value)
        for candidate in _element_candidates(value, text):
            if type(candidate) in PLAIN_SCALAR_TYPES and str(candidate) == text:
                return candidate
        raise _Unaccepted(
            value,
            f"no plain value prints as {text!r}, which is what the estimator "
            "would have been sent",
        )
    raise _Unaccepted(value, f"{_describe(value)} is not an accepted element")


def _element_candidates(value: np.generic, text: str) -> list[Any]:
    """Plain values that might print exactly as ``text``."""
    candidates: list[Any] = [value.item()]
    for build in (int, float):
        try:
            candidates.append(build(text))
        except (TypeError, ValueError):
            continue
    return candidates


def _plain_member(member: Any) -> Any:
    """The stand-in for one member of a top-level sequence.

    A ``list`` member is the one nested form the serialiser gives meaning to:
    ``_to_string`` writes it as its own elements joined with commas, which is
    how ``interaction_constraints`` is spelled. A ``tuple``, ``set`` or array
    in that position is written with Python's or numpy's repr instead --
    ``(1, 2)``, ``[1 2]`` -- which LightGBM cannot read, and rewriting it to a
    list would change the bytes. Both are refused rather than guessed at.
    """
    if type(member) is list:
        return [_plain_element(inner) for inner in member]
    return _plain_element(member)


def _plain_sequence(value: Any) -> list[Any]:
    """The plain stand-in for a sequence the serialiser joins with commas."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise _Unaccepted(
                value, f"a {value.ndim}-D numpy array is not a parameter value"
            )
        members: list[Any] = list(value)
    else:
        members = list(value)
    return [_plain_member(member) for member in members]


def normalise_value(value: Any) -> Any:
    """Return the plain stand-in for one parameter value.

    Raises:
        _Unaccepted: when the value is outside the accepted set.
    """
    if isinstance(value, np.ndarray) or type(value) in PLAIN_SEQUENCE_TYPES:
        return _plain_sequence(value)
    return _plain_scalar(value)


def is_plain(value: Any) -> bool:
    """Whether ``value`` is already inside the accepted set, unchanged."""
    if type(value) in PLAIN_SCALAR_TYPES:
        return True
    if type(value) is not list:
        return False
    return all(
        type(member) in PLAIN_SCALAR_TYPES
        or (
            type(member) is list
            and all(type(inner) in PLAIN_SCALAR_TYPES for inner in member)
        )
        for member in value
    )


def normalise_params(params: dict[str, Any], *, surface: str) -> dict[str, Any]:
    """Return ``params`` with every value replaced by its plain stand-in.

    Args:
        params: the parameters exactly as the caller wrote them.
        surface: the input they arrived through, named in any refusal.

    Returns:
        A new dict. Keys are untouched; every value is a plain Python value.

    Raises:
        LizyMLError: with ``CONFIG_INVALID``, naming every rejected parameter.
    """
    normalised: dict[str, Any] = {}
    rejected: list[tuple[str, str, str]] = []
    for name, value in params.items():
        try:
            normalised[name] = normalise_value(value)
        except _Unaccepted as unaccepted:
            rejected.append((name, _describe(unaccepted.value), unaccepted.reason))
    if not rejected:
        return normalised
    lines = [
        f"  {surface}: '{name}' is a {kind} -- {reason}."
        for name, kind, reason in sorted(rejected)
    ]
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            "Parameter value(s) the estimator cannot be given as written:\n"
            + "\n".join(lines)
            + f"\nWrite {ACCEPTED_DESCRIPTION}."
        ),
        context={
            "rejected": [
                {"surface": surface, "parameter": name, "type": kind, "reason": reason}
                for name, kind, reason in sorted(rejected)
            ]
        },
    )


def assert_plain_params(params: dict[str, Any], *, where: str) -> None:
    """Refuse to hand the estimator a value that never passed a surface.

    Normalising at the four surfaces is a claim about wiring, and a claim about
    wiring is exactly what fails silently when a fifth route is added later
    (DC4). This turns it into a property: the two places that call
    ``lgb.train`` check it, so a value that reached training without being
    normalised stops the run and names itself instead of training on bytes
    nobody chose.

    Raises:
        LizyMLError: with ``CONFIG_INVALID``, naming every offending parameter.
    """
    offending = sorted(
        (name, _describe(value))
        for name, value in params.items()
        if not is_plain(value)
    )
    if not offending:
        return
    lines = [f"  '{name}' is a {kind}." for name, kind in offending]
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            f"Parameter value(s) reached {where} without being normalised at a "
            "surface:\n" + "\n".join(lines) + f"\nAccepted: {ACCEPTED_DESCRIPTION}."
        ),
        context={
            "where": where,
            "unnormalised": [
                {"parameter": name, "type": kind} for name, kind in offending
            ],
        },
    )
