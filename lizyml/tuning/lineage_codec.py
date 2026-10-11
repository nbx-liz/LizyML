"""Codec for the values LizyML writes into Optuna study attributes (H-0122).

Two facts about Optuna storages decide the shape of everything here:

* The in-memory storage keeps an attribute value as written, while an RDB
  storage hands back ``json.loads(json.dumps(value))``. :func:`to_attr_value`
  applies that round trip on the writer's side, strictly, so both storages hold
  identical list-based values and a value that could not be stored fails before
  anything is written (INV-R4).
* JSON cannot tell ``True`` from ``1`` from ``1.0`` once a reader is lenient,
  and has no NaN. Every value slot that is not a plain flag or name is therefore
  a *tagged scalar* ``[tag, value]``, tagged by ``type(v) is ...`` exactly as a
  categorical choice is admitted (``search_space._ALLOWED_CHOICE_TYPES``), and
  the decoder checks the tag against the JSON type with the same identity test.

This module holds the value codecs: tagged scalars, search dims (and
:func:`dim_signature`, the one comparison form of a dim) and boundary reports.
The three record types built from them live in
:mod:`lizyml.tuning.lineage_records`.

This module is pure (no Optuna import) and sits in Layer 2. Bad input raises
:class:`LineageCodecError`; the reader that classifies a study
(``lineage.classify_study``) turns that into the ``unreadable`` or
``inconsistent`` state rather than skipping the record.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any, cast

from lizyml.core.types.search_dim import (
    CategoricalDim,
    DimCategory,
    FloatDim,
    IntDim,
    SearchDim,
)
from lizyml.core.types.tuning_result import BoundaryDimStatus, BoundaryReport

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

DIM_CATEGORIES: frozenset[str] = frozenset({"model", "smart", "training"})
"""The values ``SearchDim.category`` may take (``DimCategory``)."""

_SPECIAL_FLOATS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
_EDGES = frozenset({"lower", "upper", "none"})


class LineageCodecError(ValueError):
    """A value cannot be encoded, or a stored payload cannot be decoded."""


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


# ---------------------------------------------------------------------------
# Tagged scalars
# ---------------------------------------------------------------------------


def encode_scalar(value: Any) -> list[Any]:
    """Encode a plain scalar as ``[tag, value]``.

    The tag comes from ``type(value) is ...``, so a numpy scalar or any
    subclass of a plain type is refused instead of being widened.
    """
    kind = type(value)
    if kind is bool:
        return ["bool", value]
    if kind is int:
        return ["int", value]
    if kind is float:
        if math.isnan(value):
            return ["float", "nan"]
        if math.isinf(value):
            return ["float", "inf" if value > 0 else "-inf"]
        return ["float", value]
    if kind is str:
        return ["str", value]
    if value is None:
        return ["none", None]
    raise LineageCodecError(
        f"cannot encode {value!r} (type={kind.__qualname__}); only plain bool, "
        "int, float, str and None are stored"
    )


def decode_scalar(payload: Any) -> Any:
    """Decode a ``[tag, value]`` pair, checking the tag against the JSON type."""
    if type(payload) is not list or len(payload) != 2:
        raise LineageCodecError(f"tagged scalar must be a 2-item list: {payload!r}")
    tag, value = payload
    if tag == "bool" and type(value) is bool:
        return value
    if tag == "int" and type(value) is int:
        return value
    if tag == "float":
        if type(value) is float:
            return value
        if type(value) is str and value in _SPECIAL_FLOATS:
            return _SPECIAL_FLOATS[value]
    if tag == "str" and type(value) is str:
        return value
    if tag == "none" and value is None:
        return None
    raise LineageCodecError(f"malformed tagged scalar: {payload!r}")


def _decode_number(payload: Any, *, ints_only: bool = False) -> int | float:
    value = decode_scalar(payload)
    allowed: tuple[type, ...] = (int,) if ints_only else (int, float)
    if not any(type(value) is t for t in allowed):
        raise LineageCodecError(f"expected a number, got {payload!r}")
    return cast("int | float", value)


def _decode_optional_number(payload: Any, *, ints_only: bool = False) -> Any:
    if decode_scalar(payload) is None:
        return None
    return _decode_number(payload, ints_only=ints_only)


def _check_number(value: Any, what: str, *, ints_only: bool = False) -> None:
    allowed: tuple[type, ...] = (int,) if ints_only else (int, float)
    if not any(type(value) is t for t in allowed):
        raise LineageCodecError(f"{what}: {value!r} is not a plain number")


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------


def _expect_dict(value: Any, what: str) -> dict[str, Any]:
    """Return *value* if it is a ``dict``; raise otherwise."""
    if type(value) is not dict:
        raise LineageCodecError(f"{what} must be an object, got {value!r}")
    return value


def _expect_list(value: Any, what: str) -> list[Any]:
    """Return *value* if it is a ``list``; raise otherwise."""
    if type(value) is not list:
        raise LineageCodecError(f"{what} must be a list, got {value!r}")
    return value


def _expect_str(value: Any, what: str) -> str:
    """Return *value* if it is a ``str``; raise otherwise."""
    if type(value) is not str:
        raise LineageCodecError(f"{what} must be a string, got {value!r}")
    return value


def _expect_bool(value: Any, what: str) -> bool:
    """Return *value* if it is a ``bool``; raise otherwise."""
    if type(value) is not bool:
        raise LineageCodecError(f"{what} must be a boolean, got {value!r}")
    return value


def _field(payload: Mapping[str, Any], key: str, what: str) -> Any:
    if key not in payload:
        raise LineageCodecError(f"{what} is missing '{key}'")
    return payload[key]


def _category(value: Any, what: str) -> DimCategory:
    if type(value) is not str or value not in DIM_CATEGORIES:
        raise LineageCodecError(f"{what}: unknown category {value!r}")
    return cast("DimCategory", value)


# ---------------------------------------------------------------------------
# SearchDim
# ---------------------------------------------------------------------------


def encode_dim(dim: SearchDim) -> dict[str, Any]:
    """Encode a ``SearchDim`` so that ``decode_dim`` returns an equal dim."""
    what = f"dim '{dim.name}'"
    _expect_str(dim.name, f"{what} name")
    _category(dim.category, what)
    if isinstance(dim, CategoricalDim):
        if not dim.choices:
            raise LineageCodecError(f"{what} has no choices")
        return {
            "kind": "categorical",
            "name": dim.name,
            "category": dim.category,
            "choices": [encode_scalar(c) for c in dim.choices],
        }
    ints_only = isinstance(dim, IntDim)
    for label in ("low", "high"):
        _check_number(getattr(dim, label), f"{what} {label}", ints_only=ints_only)
    for label in ("min_allowed", "max_allowed"):
        bound = getattr(dim, label)
        if bound is not None:
            _check_number(bound, f"{what} {label}", ints_only=ints_only)
    _expect_bool(dim.log, f"{what} log")
    return {
        "kind": "int" if ints_only else "float",
        "name": dim.name,
        "category": dim.category,
        "low": encode_scalar(dim.low),
        "high": encode_scalar(dim.high),
        "log": dim.log,
        "min_allowed": encode_scalar(dim.min_allowed),
        "max_allowed": encode_scalar(dim.max_allowed),
    }


def decode_dim(payload: Any) -> SearchDim:
    """Decode one encoded dim; refuse anything ``encode_dim`` cannot produce."""
    obj = _expect_dict(payload, "dim")
    name = _expect_str(_field(obj, "name", "dim"), "dim name")
    what = f"dim '{name}'"
    category = _category(_field(obj, "category", what), what)
    kind = _field(obj, "kind", what)
    if kind == "categorical":
        raw = _expect_list(_field(obj, "choices", what), f"{what} choices")
        if not raw:
            raise LineageCodecError(f"{what} has no choices")
        choices = tuple(decode_scalar(c) for c in raw)
        return CategoricalDim(name=name, choices=choices, category=category)
    if kind not in ("float", "int"):
        raise LineageCodecError(f"{what}: unknown kind {kind!r}")
    ints_only = kind == "int"
    low = _decode_number(_field(obj, "low", what), ints_only=ints_only)
    high = _decode_number(_field(obj, "high", what), ints_only=ints_only)
    log = _expect_bool(_field(obj, "log", what), f"{what} log")
    lo_bound = _decode_optional_number(
        _field(obj, "min_allowed", what), ints_only=ints_only
    )
    hi_bound = _decode_optional_number(
        _field(obj, "max_allowed", what), ints_only=ints_only
    )
    if ints_only:
        return IntDim(
            name=name,
            low=int(low),
            high=int(high),
            log=log,
            category=category,
            min_allowed=lo_bound,
            max_allowed=hi_bound,
        )
    return FloatDim(
        name=name,
        low=low,
        high=high,
        log=log,
        category=category,
        min_allowed=lo_bound,
        max_allowed=hi_bound,
    )


def encode_dims(dims: Iterable[SearchDim]) -> list[dict[str, Any]]:
    """Encode dims in order."""
    return [encode_dim(d) for d in dims]


def decode_dims(payload: Any) -> tuple[SearchDim, ...]:
    """Decode a list of encoded dims, keeping order."""
    return tuple(decode_dim(d) for d in _expect_list(payload, "dims"))


def dim_signature(dim: SearchDim) -> str:
    """A dim's comparison form: the JSON of :func:`encode_dim`, keys sorted.

    Two dims are the same declaration iff their signatures are equal. The
    tagged encoding keeps ``True`` / ``1`` / ``1.0`` apart and makes NaN equal
    to NaN, which dataclass equality would not; the name, the category and
    any bounds take part.

    Raises:
        LineageCodecError: The dim holds a value the codec cannot store.
    """
    return json.dumps(encode_dim(dim), sort_keys=True)


# ---------------------------------------------------------------------------
# BoundaryReport
# ---------------------------------------------------------------------------


def _encode_status(status: BoundaryDimStatus) -> dict[str, Any]:
    what = f"boundary status '{status.name}'"
    _expect_str(status.name, what)
    if status.edge not in _EDGES:
        raise LineageCodecError(f"{what}: unknown edge {status.edge!r}")
    for label in ("low", "high", "new_low", "new_high", "position_pct"):
        value = getattr(status, label)
        if value is not None:
            _check_number(value, f"{what} {label}")
    return {
        "name": status.name,
        "best_value": encode_scalar(status.best_value),
        "low": encode_scalar(status.low),
        "high": encode_scalar(status.high),
        "position_pct": encode_scalar(status.position_pct),
        "edge": status.edge,
        "expanded": _expect_bool(status.expanded, f"{what} expanded"),
        "new_low": encode_scalar(status.new_low),
        "new_high": encode_scalar(status.new_high),
        "clamped_to_bound": _expect_bool(
            status.clamped_to_bound, f"{what} clamped_to_bound"
        ),
    }


def _decode_status(payload: Any) -> BoundaryDimStatus:
    obj = _expect_dict(payload, "boundary status")
    name = _expect_str(_field(obj, "name", "boundary status"), "boundary status name")
    what = f"boundary status '{name}'"
    edge = _field(obj, "edge", what)
    if type(edge) is not str or edge not in _EDGES:
        raise LineageCodecError(f"{what}: unknown edge {edge!r}")
    return BoundaryDimStatus(
        name=name,
        best_value=decode_scalar(_field(obj, "best_value", what)),
        low=_decode_optional_number(_field(obj, "low", what)),
        high=_decode_optional_number(_field(obj, "high", what)),
        position_pct=_decode_optional_number(_field(obj, "position_pct", what)),
        edge=edge,
        expanded=_expect_bool(_field(obj, "expanded", what), f"{what} expanded"),
        new_low=_decode_optional_number(_field(obj, "new_low", what)),
        new_high=_decode_optional_number(_field(obj, "new_high", what)),
        clamped_to_bound=_expect_bool(
            _field(obj, "clamped_to_bound", what), f"{what} clamped_to_bound"
        ),
    )


def encode_report(report: BoundaryReport) -> dict[str, Any]:
    """Encode a ``BoundaryReport`` in full (every dim's status, not just names)."""
    return {
        "dims": [_encode_status(s) for s in report.dims],
        "expanded_names": [
            _expect_str(n, "expanded name") for n in report.expanded_names
        ],
    }


def decode_report(payload: Any) -> BoundaryReport:
    """Decode an encoded ``BoundaryReport``."""
    obj = _expect_dict(payload, "boundary report")
    dims = _expect_list(_field(obj, "dims", "boundary report"), "report dims")
    names = _expect_list(
        _field(obj, "expanded_names", "boundary report"), "expanded_names"
    )
    return BoundaryReport(
        dims=tuple(_decode_status(d) for d in dims),
        expanded_names=tuple(_expect_str(n, "expanded name") for n in names),
    )
