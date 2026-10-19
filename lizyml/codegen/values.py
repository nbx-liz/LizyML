"""values — the closed set of label values codegen writes (H-0120), and the
writer for feature-category values (H-0133, :func:`category_value`).

The generated project reads category values and target labels back from JSON
and compares them with the training data by value, so a value is written only
when JSON returns the same value with a type the comparison treats the same:
Python ``str``, ``int``, ``float`` and ``bool``, and numpy scalars whose
``.item()`` is one of those and equal to the original (written as that plain
value). Anything else -- a ``tuple`` (JSON returns a list), ``bytes``, a
timestamp, a ``Decimal`` -- is refused rather than written as ``str``, which
would silently change the codes a retrain assigns.
"""

from __future__ import annotations

from typing import Any

from lizyml.core.category_values import (
    NUMPY_PLAIN_TYPES,
    encode,
    refusal,
    shown,
    type_name,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError

_ACCEPTED = (str, int, float, bool)


def plain_value(value: Any, *, where: str) -> Any:
    """Return *value* as the plain JSON value codegen writes, or refuse it.

    Args:
        value: A category, mode or target label from the fit.
        where: Where the value comes from, for the error message (for example
            ``"categories of column 'c'"``).

    Raises:
        LizyMLError: With ``SERIALIZATION_FAILED`` for a value outside the
            accepted set.
    """
    if type(value) in _ACCEPTED:
        return value
    # Exact numpy types only: an np.timedelta64 in Y / M units has an int
    # .item() that numpy calls equal to it, so "item() is plain and equal"
    # admitted it and wrote a label as a bare number (H-0133 review).
    if any(type(value) is kind for kind in NUMPY_PLAIN_TYPES):
        plain = value.item()
        if type(plain) in _ACCEPTED and plain == value:
            return plain
    raise LizyMLError(
        code=ErrorCode.SERIALIZATION_FAILED,
        user_message=(
            f"export_code cannot write the {where}: the value {value!r} of type "
            f"{type(value).__name__} does not survive JSON with its type. "
            "Accepted: str, int, float, bool, and numpy scalars of those."
        ),
        context={"where": where, "type": type(value).__name__},
    )


def plain_values(values: list[Any], *, where: str) -> list[Any]:
    """:func:`plain_value` over a list, keeping the order."""
    return [plain_value(v, where=where) for v in values]


def category_value(value: Any, *, where: str) -> Any:
    """Return a feature-category value as the JSON codegen writes (H-0133).

    Feature categories, modes and declared categories accept the closed set
    of :mod:`lizyml.core.category_values` -- the set ``Model.export``
    accepts -- written with its tagged codec, which the generated code
    decodes with the same table. Target labels keep :func:`plain_value`.

    Raises:
        LizyMLError: With ``SERIALIZATION_FAILED`` for a value outside the set.
    """
    reason = refusal(value)
    if reason is not None:
        raise LizyMLError(
            code=ErrorCode.SERIALIZATION_FAILED,
            user_message=(
                f"export_code cannot write the {where}: the value {shown(value)} "
                f"of type {type_name(value)} cannot be saved ({reason})."
            ),
            context={"where": where, "type": type_name(value), "reason": reason},
        )
    return encode(value)


def category_values(values: list[Any], *, where: str) -> list[Any]:
    """:func:`category_value` over a list, keeping the order."""
    return [category_value(v, where=where) for v in values]
