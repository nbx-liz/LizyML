"""Codec for the values LizyML writes into Optuna study attributes (H-0122).

Two facts about Optuna storages decide the shape of everything here:

* The in-memory storage keeps an attribute value as written, while an RDB
  storage hands back ``json.loads(json.dumps(value))``. :func:`to_attr_value`
  applies that round trip on the writer's side, strictly, so both storages hold
  identical list-based values and a value that could not be stored fails before
  anything is written (INV-R4).
* JSON cannot tell ``True`` from ``1`` from ``1.0`` once a reader is lenient,
  and has no NaN; the value codecs (tagged scalars, search dims and boundary
  reports) therefore tag every value. They live in
  :mod:`lizyml.core.tuning_codec` (Layer 0, H-0134) and are re-exported here.

The three record types built from them live in
:mod:`lizyml.tuning.lineage_records`.

This module is pure (no Optuna import) and sits in Layer 2. Bad input raises
:class:`LineageCodecError`; the reader that classifies a study
(``lineage.classify_study``) turns that into the ``unreadable`` or
``inconsistent`` state rather than skipping the record.
"""

from __future__ import annotations

import json
from typing import Any

# Every private name this module defined before H-0134 moved it, re-exported
# under the same name (``as`` marks the re-export for mypy), so an existing
# import keeps working.
from lizyml.core.tuning_codec import _CATEGORICAL_DIM_KEYS as _CATEGORICAL_DIM_KEYS
from lizyml.core.tuning_codec import _EDGES as _EDGES
from lizyml.core.tuning_codec import _NUMERIC_DIM_KEYS as _NUMERIC_DIM_KEYS
from lizyml.core.tuning_codec import _SPECIAL_FLOATS as _SPECIAL_FLOATS
from lizyml.core.tuning_codec import (
    DIM_CATEGORIES,
    LineageCodecError,
    decode_dim,
    decode_dims,
    decode_report,
    decode_scalar,
    dim_signature,
    encode_dim,
    encode_dims,
    encode_report,
    encode_scalar,
)
from lizyml.core.tuning_codec import _category as _category
from lizyml.core.tuning_codec import _check_magnitude as _check_magnitude
from lizyml.core.tuning_codec import _check_number as _check_number
from lizyml.core.tuning_codec import _decode_number as _decode_number
from lizyml.core.tuning_codec import (
    _decode_optional_number as _decode_optional_number,
)
from lizyml.core.tuning_codec import _decode_status as _decode_status
from lizyml.core.tuning_codec import _encode_status as _encode_status
from lizyml.core.tuning_codec import _expect_bool as _expect_bool
from lizyml.core.tuning_codec import _expect_dict as _expect_dict
from lizyml.core.tuning_codec import _expect_list as _expect_list
from lizyml.core.tuning_codec import _expect_str as _expect_str
from lizyml.core.tuning_codec import _field as _field

__all__ = [
    "DIM_CATEGORIES",
    "SCHEMA_VERSION",
    "STUDY_KEY",
    "LineageCodecError",
    "decode_dim",
    "decode_dims",
    "decode_report",
    "decode_scalar",
    "dim_signature",
    "encode_dim",
    "encode_dims",
    "encode_report",
    "encode_scalar",
    "to_attr_value",
]

SCHEMA_VERSION = 1
"""The newest ``lizyml:study`` schema this LizyML reads and writes."""

STUDY_KEY = "lizyml:study"


# ---------------------------------------------------------------------------
# Strict JSON round trip
# ---------------------------------------------------------------------------


def _check_keys(value: Any, path: str) -> None:
    """Refuse non-``str`` dict keys, which ``json.dumps`` would silently coerce."""
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise LineageCodecError(
                    f"{path}: dict key {key!r} is not a str and would not survive "
                    "a JSON round trip unchanged"
                )
            _check_keys(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for i, item in enumerate(value):
            _check_keys(item, f"{path}[{i}]")


def to_attr_value(payload: Any) -> Any:
    """Return *payload* as an RDB storage would hand it back.

    ``json.loads(json.dumps(payload, allow_nan=False))``: tuples become lists,
    and NaN, infinities, non-``str`` keys and non-JSON objects are refused.
    Idempotent on its own output.
    """
    _check_keys(payload, "$")
    try:
        text = json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LineageCodecError(f"value is not strict JSON: {exc}") from exc
    return json.loads(text)
