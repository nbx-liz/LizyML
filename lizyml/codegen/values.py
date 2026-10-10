"""values — the closed set of category and label values codegen writes (H-0120).

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

import numpy as np

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
    if isinstance(value, np.generic):
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
