"""The documented format of a search-space dimension (H-0123).

``tuning.optuna.space`` and ``tune(space=...)`` take one format, stated in
docs/config-reference.md and BLUEPRINT §11.1: per dimension a ``type``
(``float`` / ``int`` / ``categorical``), numeric ``low`` / ``high``, an optional
``bool`` ``log``, a ``list`` of ``choices``, and an optional ``category``. The
format is written in the types YAML and JSON produce.

Before H-0123 ``parse_space`` read a dimension through ``float()``, ``int()``
and ``bool()``, so a spec outside the format became another search without a
word: ``log: "false"`` searched on a log scale, a ``step`` key was dropped, an
``int`` bound ``1.7`` became ``1``, ``choices: "abc"`` became three choices.
These functions accept the format and refuse everything else with
``CONFIG_INVALID``, naming the dimension, the key, a reason and the surface.

The checks close the grammar instead of tolerating one more form at a time
(the value-preserving alternative did not close in review): every accepted
form is listed here, and anything not listed is refused.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import numpy as np

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.tuning.lineage_codec import DIM_CATEGORIES

__all__ = [
    "CATEGORICAL_KEYS",
    "DIM_KEYS",
    "NUMBER_LITERAL",
    "NUMERIC_KEYS",
    "REQUIRED_KEYS",
    "check_dim_keys",
    "check_space_shape",
    "format_error",
    "read_choices",
    "read_float_bound",
    "read_int_bound",
    "read_log",
]

NUMERIC_KEYS: frozenset[str] = frozenset({"type", "low", "high", "log", "category"})
"""The keys a ``float`` or ``int`` dimension may have."""

CATEGORICAL_KEYS: frozenset[str] = frozenset({"type", "choices", "category"})
"""The keys a ``categorical`` dimension may have."""

DIM_KEYS: dict[str, frozenset[str]] = {
    "float": NUMERIC_KEYS,
    "int": NUMERIC_KEYS,
    "categorical": CATEGORICAL_KEYS,
}
"""The allowed keys of each dimension ``type``; its keys are the types."""

REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "float": ("low", "high"),
    "int": ("low", "high"),
    "categorical": ("choices",),
}
"""The keys each dimension ``type`` must have."""

NUMBER_LITERAL = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")
"""A number written as text, matched in full. ``[0-9]`` is ASCII only.

PyYAML reads YAML 1.1, where ``1e-4`` (an exponent without a decimal point) is
a string, so a bound written that way in a config file arrives as text. Only a
plain decimal or exponent literal is a number here: ``"nan"``, ``"inf"``,
``"0x10"``, ``"1,000"`` and a trailing newline are not.
"""

_FORMAT_HINT = (
    "A dimension takes the documented form {type: float | int, low, high, "
    "log?, category?} or {type: categorical, choices: [...], category?} "
    "(docs/config-reference.md)."
)


def format_error(
    surface: str,
    message: str,
    *,
    reason: str,
    dim: str | None = None,
    key: str | None = None,
    **context: Any,
) -> LizyMLError:
    """A ``CONFIG_INVALID`` refusal of a space outside the format.

    Args:
        surface: Where the space was declared (``tuning.optuna.space`` or
            ``tune(space=)``); quoted first in the message.
        message: What is wrong, naming the dimension.
        reason: A stable token for ``context["reason"]``.
        dim: The dimension's name.
        key: The key of the dimension spec the refusal is about.
        **context: Further ``context`` entries.
    """
    ctx: dict[str, Any] = {"surface": surface, "reason": reason}
    if dim is not None:
        ctx["dim"] = dim
    if key is not None:
        ctx["key"] = key
    ctx.update(context)
    return LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=f"{surface}: {message} {_FORMAT_HINT}",
        context=ctx,
    )


def check_space_shape(space: Any, *, surface: str) -> Mapping[Any, Any]:
    """The space is a mapping of ``str`` names to mapping specs (H-0122 order).

    Every dimension's name, mapping and ``category`` are checked before any
    dimension's other keys, the order ``tune(space=...)`` has refused in since
    H-0122.

    Returns:
        *space*, known to be a mapping.
    """
    if not isinstance(space, Mapping):
        raise format_error(
            surface,
            f"expected a mapping of dimensions, got {type(space).__name__}.",
            reason="not_a_mapping",
        )
    for name, spec in space.items():
        if type(name) is not str:
            raise format_error(
                surface,
                f"dimension name {name!r} is not a str.",
                reason="name_not_str",
                dim=repr(name),
            )
        if not isinstance(spec, Mapping):
            raise format_error(
                surface,
                f"dimension '{name}' must be a mapping, got {type(spec).__name__}.",
                reason="dim_not_a_mapping",
                dim=name,
            )
        category = spec.get("category", "model")
        if type(category) is not str or category not in DIM_CATEGORIES:
            raise format_error(
                surface,
                f"dimension '{name}' has category {category!r}; expected one of "
                f"{sorted(DIM_CATEGORIES)}.",
                reason="unknown_category",
                dim=name,
                key="category",
            )
    return space


def check_dim_keys(
    name: str, spec: Mapping[Any, Any], kind: str, *, surface: str
) -> None:
    """Refuse a key the dimension's ``type`` does not have, or a missing one.

    ``step`` gets its own message: a UI that sends one (LizyStudio's range
    inputs) is told the search had no step, rather than that a key is unknown.
    """
    if "step" in spec:
        raise format_error(
            surface,
            f"dimension '{name}' has a 'step', but LizyML dimensions have no step "
            "yet: the search would sample the whole range, not multiples of it. "
            "Remove 'step'.",
            reason="step_not_supported",
            dim=name,
            key="step",
        )
    allowed = DIM_KEYS[kind]
    unknown = [key for key in spec if key not in allowed]
    if unknown:
        first = unknown[0]
        raise format_error(
            surface,
            f"dimension '{name}' ({kind}) has unknown key(s) "
            f"{', '.join(repr(k) for k in unknown)}; allowed: {sorted(allowed)}.",
            reason="unknown_key",
            dim=name,
            key=first if type(first) is str else repr(first),
        )
    for key in REQUIRED_KEYS[kind]:
        if key not in spec:
            raise format_error(
                surface,
                f"dimension '{name}' ({kind}) has no '{key}'.",
                reason="malformed_dim",
                dim=name,
                key=key,
            )


def _bound_error(
    name: str, key: str, value: Any, *, reason: str, surface: str, why: str
) -> LizyMLError:
    return format_error(
        surface,
        f"dimension '{name}' has {key}={value!r} ({type(value).__name__}): {why}",
        reason=reason,
        dim=name,
        key=key,
    )


_NOT_A_NUMBER = (
    "a bound is an int or a float (not a bool), a numpy integer or floating "
    "scalar, or a number written as text such as '1e-4'."
)


def _is_plain_number(value: Any) -> bool:
    """An ``int`` or ``float`` by identity: not ``bool``, nor a subclass.

    Identity, not ``type(value) in (int, float)``: tuple membership compares
    with ``==``, which a metaclass can answer (see
    ``search_space._validate_categorical_choices``).
    """
    return type(value) is int or type(value) is float


def _literal(name: str, key: str, value: str, *, surface: str) -> float:
    """The float value of a number literal; refuse text that is not one."""
    if not NUMBER_LITERAL.fullmatch(value):
        raise _bound_error(
            name,
            key,
            value,
            reason="bound_not_a_number",
            surface=surface,
            why=_NOT_A_NUMBER,
        )
    return float(value)


def _finite(name: str, key: str, value: Any, number: float, *, surface: str) -> float:
    if not math.isfinite(number):
        raise _bound_error(
            name,
            key,
            value,
            reason="bound_not_finite",
            surface=surface,
            why="a bound must be finite.",
        )
    return number


def read_float_bound(name: str, key: str, value: Any, *, surface: str) -> float:
    """A ``float`` dimension's bound, as a plain finite ``float``.

    Accepted: ``int`` / ``float`` (by exact type, so not ``bool``), a numpy
    integer or floating scalar (H-0095; ``numpy.bool_`` is neither), or a
    :data:`NUMBER_LITERAL` string. An ``int`` too large for a ``float`` is
    refused as not finite.
    """
    if type(value) is str:
        return _finite(
            name,
            key,
            value,
            _literal(name, key, value, surface=surface),
            surface=surface,
        )
    if _is_plain_number(value) or isinstance(value, (np.integer, np.floating)):
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        return _finite(name, key, value, number, surface=surface)
    raise _bound_error(
        name,
        key,
        value,
        reason="bound_not_a_number",
        surface=surface,
        why=_NOT_A_NUMBER,
    )


def read_int_bound(name: str, key: str, value: Any, *, surface: str) -> int:
    """An ``int`` dimension's bound, as a plain ``int`` with the same value.

    The accepted types are :func:`read_float_bound`'s. A value must be
    integral: ``16.0``, ``"16.0"`` and ``"1e2"`` are ``16``, ``16`` and
    ``100``; ``1.7`` and ``"1.7"`` are refused rather than truncated. Text is
    read exactly (``Decimal``), after its float value is checked finite, so an
    exponent such as ``"1e999"`` is refused instead of being expanded.
    """
    if type(value) is int:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if type(value) is str:
        _finite(
            name,
            key,
            value,
            _literal(name, key, value, surface=surface),
            surface=surface,
        )
        exact = Decimal(value)
        if exact != exact.to_integral_value():
            raise _not_integral(name, key, value, surface=surface)
        return int(exact)
    if type(value) is float or isinstance(value, np.floating):
        number = _finite(name, key, value, float(value), surface=surface)
        if not number.is_integer():
            raise _not_integral(name, key, value, surface=surface)
        return int(number)
    raise _bound_error(
        name,
        key,
        value,
        reason="bound_not_a_number",
        surface=surface,
        why=_NOT_A_NUMBER,
    )


def _not_integral(name: str, key: str, value: Any, *, surface: str) -> LizyMLError:
    return _bound_error(
        name,
        key,
        value,
        reason="bound_not_integral",
        surface=surface,
        why="an int dimension's bound must be a whole number; it is not rounded.",
    )


def read_log(name: str, spec: Mapping[Any, Any], *, surface: str) -> bool:
    """``log``, which is a ``bool`` when given (YAML ``true`` / ``no`` are)."""
    if "log" not in spec:
        return False
    value = spec["log"]
    if type(value) is not bool:
        raise format_error(
            surface,
            f"dimension '{name}' has log={value!r} ({type(value).__name__}); "
            "log is true or false.",
            reason="log_not_bool",
            dim=name,
            key="log",
        )
    return value


def read_choices(
    name: str, spec: Mapping[Any, Any], *, surface: str
) -> list[Any] | tuple[Any, ...]:
    """``choices``: a non-empty ``list`` or ``tuple``, by exact type.

    A string, ``set``, dict, ``range``, numpy array or generator is refused:
    each was read as a different set of choices, or as one whose order could
    change between runs. The values are checked by
    ``search_space._validate_categorical_choices``.
    """
    choices = spec["choices"]
    if not (type(choices) is list or type(choices) is tuple):
        raise format_error(
            surface,
            f"dimension '{name}' has choices of type {type(choices).__name__}; "
            "choices is a list.",
            reason="choices_not_a_list",
            dim=name,
            key="choices",
        )
    if not choices:
        raise format_error(
            surface,
            f"Categorical dim '{name}' requires non-empty 'choices'.",
            reason="empty_choices",
            dim=name,
            key="choices",
            param=name,
        )
    return choices
