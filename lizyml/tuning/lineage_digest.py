"""The canonical value digest behind the objective identity (H-0122 §4).

One function, :func:`value_digest`, turns a config value into the sha256 hex
digest an identity element holds. It lives here, in Layer 2 and free of any
config or provider import, so that both sides use the very same function:

* ``lizyml.core._tuning_identity.config_parts`` digests every ``config.*``
  element with it, ``config.fixed`` included;
* ``lineage_records.decode_study`` re-digests a stored record's ``fixed`` with
  it and refuses the record unless the result is the record's own
  ``config.fixed`` element (:data:`FIXED_ELEMENT`). Identity proves that
  element equal to this call's; this proves the stored value -- the one the
  trials train on (§5.3) -- is the value the element describes.

Pure: no I/O, no Optuna.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["FIXED_ELEMENT", "canonical", "type_name", "value_digest"]

FIXED_ELEMENT = "config.fixed"
"""The identity element that digests the study's effective ``fixed``."""


def type_name(value: Any) -> str:
    """``module.qualname`` of *value*'s type."""
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def canonical(value: Any) -> Any:
    """A JSON-safe, type-tagged, order-stable form of a config value.

    ``True`` / ``1`` / ``1.0`` stay distinct; lists and tuples are one kind
    (a stored ``fixed`` comes back from JSON as lists); dict keys of mixed
    types sort without error; a numpy scalar equals its plain value; anything
    else is tagged with its type and ``repr``.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        return ["float", repr(value)]
    if isinstance(value, Mapping):
        items = [[canonical(k), canonical(v)] for k, v in value.items()]
        return ["dict", sorted(items, key=lambda kv: json.dumps(kv[0]))]
    if isinstance(value, list | tuple):
        return ["seq", [canonical(v) for v in value]]
    if isinstance(value, set | frozenset):
        return ["set", sorted((canonical(v) for v in value), key=json.dumps)]
    return ["repr", type_name(value), repr(value)]


def value_digest(value: Any) -> str:
    """sha256 hex digest of :func:`canonical` (*value*)."""
    text = json.dumps(canonical(value), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()
