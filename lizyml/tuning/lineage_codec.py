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

It also holds the three record types and their codecs: ``lizyml:study``
(:class:`StoredStudy`), ``lizyml:round:<k>`` (:class:`StoredRound`) and
``lizyml:round:<k>:end`` (:class:`StoredRoundEnd`). Each encoder returns the
stored (JSON) form and first proves that its decoder accepts it, so a record
the reader would refuse is refused while planning, before any write.

This module is pure (no Optuna import) and sits in Layer 2. Bad input raises
:class:`LineageCodecError`; the reader that classifies a study
(``lineage.classify_study``) turns that into the ``unreadable`` or
``inconsistent`` state rather than skipping the record.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

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
    "Provenance",
    "Route",
    "RoundSource",
    "StoredRound",
    "StoredRoundEnd",
    "StoredStudy",
    "decode_dim",
    "decode_dims",
    "decode_report",
    "decode_round",
    "decode_round_end",
    "decode_scalar",
    "decode_study",
    "encode_dim",
    "encode_dims",
    "encode_report",
    "encode_round",
    "encode_round_end",
    "encode_scalar",
    "encode_study",
    "objective_digest",
    "to_attr_value",
]

SCHEMA_VERSION = 1
"""The newest ``lizyml:study`` schema this LizyML reads and writes."""

STUDY_KEY = "lizyml:study"

DIM_CATEGORIES: frozenset[str] = frozenset({"model", "smart", "training"})
"""The values ``SearchDim.category`` may take (``DimCategory``)."""

RoundSource = Literal["config", "previous"]
_ROUND_SOURCES: frozenset[str] = frozenset({"config", "previous"})
_ORIGINS: frozenset[str] = frozenset({"user", "default"})

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


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredStudy:
    """The decoded ``lizyml:study`` record."""

    schema: int
    objective_digest: str
    objective_elements: Mapping[str, str]
    direction: Literal["minimize", "maximize"]
    versions: Mapping[str, str]
    space_mode: Literal["merge", "replace"]
    fixed: Mapping[str, Any]
    origin_space: tuple[SearchDim, ...]
    created_at: str


@dataclass(frozen=True)
class Route:
    """Where a trial parameter goes: ``name`` is the Optuna param key."""

    name: str
    canonical: str
    category: DimCategory


@dataclass(frozen=True)
class Provenance:
    """Where a declared dimension came from: the user or the provider default."""

    category: DimCategory
    name: str
    origin: Literal["user", "default"]


@dataclass(frozen=True)
class StoredRound:
    """The decoded ``lizyml:round:<k>`` record."""

    round: int
    source: RoundSource
    space_arg: Mapping[str, Any] | None
    provenance: tuple[Provenance, ...]
    declared: tuple[SearchDim, ...]
    effective: tuple[SearchDim, ...]
    expand_boundary: bool
    boundary_threshold: float
    boundary_report: BoundaryReport | None
    n_trials: int
    timeout: float | None
    start_trial: int
    best_before: float | None
    routes_added: tuple[Route, ...]


@dataclass(frozen=True)
class StoredRoundEnd:
    """The decoded ``lizyml:round:<k>:end`` record."""

    status: Literal["completed", "failed"]
    best_after: float | None
    ended_at: str


# ---------------------------------------------------------------------------
# Record codecs
# ---------------------------------------------------------------------------


def objective_digest(elements: Mapping[str, str]) -> str:
    """sha256 over the sorted ``(element, digest)`` pairs."""
    h = hashlib.sha256()
    for name in sorted(elements):
        h.update(f"{name}\x00{elements[name]}\x00".encode())
    return h.hexdigest()


def _choice(value: Any, allowed: Iterable[str], what: str) -> str:
    if type(value) is not str or value not in frozenset(allowed):
        raise LineageCodecError(f"{what}: unexpected value {value!r}")
    return value


def _str_map(value: Any, what: str) -> dict[str, str]:
    obj = _expect_dict(value, what)
    return {k: _expect_str(v, f"{what}.{k}") for k, v in obj.items()}


def _count(value: Any, what: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise LineageCodecError(f"{what} must be an int >= {minimum}, got {value!r}")
    return value


def _number_or_none(payload: Any, what: str) -> float | None:
    value = decode_scalar(payload)
    if value is None:
        return None
    if type(value) is not float and type(value) is not int:
        raise LineageCodecError(f"{what} must be a number, got {payload!r}")
    return cast("float", value)


def _checked(
    payload: dict[str, Any], decoder: Callable[[Any], object]
) -> dict[str, Any]:
    """Return *payload* as stored, after proving the reader accepts it.

    A record the writer can produce but the reader refuses would make the
    study unreadable on the next call; refusing it here keeps that failure
    before any write (INV-R4).
    """
    stored = to_attr_value(payload)
    decoder(stored)
    return cast("dict[str, Any]", stored)


def encode_study(study: StoredStudy) -> dict[str, Any]:
    """Encode a ``lizyml:study`` record in its stored (JSON) form.

    Raises:
        LineageCodecError: The record would not decode back.
    """
    return _checked(
        {
            "schema": study.schema,
            "objective": {
                "digest": study.objective_digest,
                "elements": dict(study.objective_elements),
            },
            "direction": study.direction,
            "versions": dict(study.versions),
            "space_mode": study.space_mode,
            "fixed": dict(study.fixed),
            "origin_space": encode_dims(study.origin_space),
            "created_at": study.created_at,
        },
        decode_study,
    )


def decode_study(payload: Any) -> StoredStudy:
    """Decode a ``lizyml:study`` record whose schema this LizyML reads."""
    obj = _expect_dict(payload, STUDY_KEY)
    schema = _field(obj, "schema", STUDY_KEY)
    if type(schema) is not int or not 1 <= schema <= SCHEMA_VERSION:
        raise LineageCodecError(
            f"{STUDY_KEY} schema {schema!r} is not readable by this LizyML "
            f"(reads 1..{SCHEMA_VERSION})"
        )
    objective = _expect_dict(_field(obj, "objective", STUDY_KEY), "objective")
    elements = _str_map(_field(objective, "elements", "objective"), "elements")
    digest = _expect_str(_field(objective, "digest", "objective"), "objective digest")
    if digest != objective_digest(elements):
        raise LineageCodecError("objective digest does not match its elements")
    return StoredStudy(
        schema=schema,
        objective_digest=digest,
        objective_elements=elements,
        direction=cast(
            "Literal['minimize', 'maximize']",
            _choice(
                _field(obj, "direction", STUDY_KEY),
                ("minimize", "maximize"),
                "direction",
            ),
        ),
        versions=_str_map(_field(obj, "versions", STUDY_KEY), "versions"),
        space_mode=cast(
            "Literal['merge', 'replace']",
            _choice(
                _field(obj, "space_mode", STUDY_KEY), ("merge", "replace"), "space_mode"
            ),
        ),
        fixed=_expect_dict(_field(obj, "fixed", STUDY_KEY), "fixed"),
        origin_space=decode_dims(_field(obj, "origin_space", STUDY_KEY)),
        created_at=_expect_str(_field(obj, "created_at", STUDY_KEY), "created_at"),
    )


def _encode_route(route: Route) -> dict[str, Any]:
    return {
        "name": route.name,
        "canonical": route.canonical,
        "category": route.category,
    }


def _decode_route(payload: Any) -> Route:
    obj = _expect_dict(payload, "route")
    return Route(
        name=_expect_str(_field(obj, "name", "route"), "route name"),
        canonical=_expect_str(_field(obj, "canonical", "route"), "route canonical"),
        category=cast(
            "DimCategory",
            _choice(_field(obj, "category", "route"), DIM_CATEGORIES, "route"),
        ),
    )


def _decode_provenance(payload: Any) -> Provenance:
    items = _expect_list(payload, "provenance entry")
    if len(items) != 3:
        raise LineageCodecError(f"provenance entry must have 3 items: {payload!r}")
    category, name, origin = items
    return Provenance(
        category=cast("DimCategory", _choice(category, DIM_CATEGORIES, "provenance")),
        name=_expect_str(name, "provenance name"),
        origin=cast(
            "Literal['user', 'default']", _choice(origin, _ORIGINS, "provenance")
        ),
    )


def encode_round(record: StoredRound) -> dict[str, Any]:
    """Encode a ``lizyml:round:<k>`` record in its stored (JSON) form.

    Raises:
        LineageCodecError: The record would not decode back.
    """
    report = record.boundary_report
    return _checked(
        {
            "round": record.round,
            "declaration": {
                "source": record.source,
                "space_arg": None
                if record.space_arg is None
                else dict(record.space_arg),
                "provenance": [
                    [p.category, p.name, p.origin] for p in record.provenance
                ],
            },
            "declared": encode_dims(record.declared),
            "effective": encode_dims(record.effective),
            "expand_boundary": record.expand_boundary,
            "boundary_threshold": encode_scalar(record.boundary_threshold),
            "boundary_report": None if report is None else encode_report(report),
            "budget": {
                "n_trials": record.n_trials,
                "timeout": encode_scalar(record.timeout),
            },
            "start_trial": record.start_trial,
            "best_before": encode_scalar(record.best_before),
            "routes_added": [_encode_route(r) for r in record.routes_added],
        },
        decode_round,
    )


def decode_round(payload: Any) -> StoredRound:
    """Decode a ``lizyml:round:<k>`` record."""
    what = "round record"
    obj = _expect_dict(payload, what)
    declaration = _expect_dict(_field(obj, "declaration", what), "declaration")
    space_arg = _field(declaration, "space_arg", "declaration")
    if space_arg is not None:
        space_arg = _expect_dict(space_arg, "space_arg")
    budget = _expect_dict(_field(obj, "budget", what), "budget")
    threshold = _number_or_none(_field(obj, "boundary_threshold", what), "threshold")
    if threshold is None:
        raise LineageCodecError("boundary_threshold must be a number")
    report = _field(obj, "boundary_report", what)
    return StoredRound(
        round=_count(_field(obj, "round", what), "round", minimum=1),
        source=cast(
            "RoundSource",
            _choice(
                _field(declaration, "source", "declaration"), _ROUND_SOURCES, "source"
            ),
        ),
        space_arg=space_arg,
        provenance=tuple(
            _decode_provenance(p)
            for p in _expect_list(
                _field(declaration, "provenance", "declaration"), "provenance"
            )
        ),
        declared=decode_dims(_field(obj, "declared", what)),
        effective=decode_dims(_field(obj, "effective", what)),
        expand_boundary=_expect_bool(
            _field(obj, "expand_boundary", what), "expand_boundary"
        ),
        boundary_threshold=threshold,
        boundary_report=None if report is None else decode_report(report),
        n_trials=_count(_field(budget, "n_trials", "budget"), "n_trials", minimum=0),
        timeout=_number_or_none(_field(budget, "timeout", "budget"), "timeout"),
        start_trial=_count(_field(obj, "start_trial", what), "start_trial", minimum=0),
        best_before=_number_or_none(_field(obj, "best_before", what), "best_before"),
        routes_added=tuple(
            _decode_route(r)
            for r in _expect_list(_field(obj, "routes_added", what), "routes")
        ),
    )


def encode_round_end(end: StoredRoundEnd) -> dict[str, Any]:
    """Encode a ``lizyml:round:<k>:end`` record in its stored (JSON) form.

    Raises:
        LineageCodecError: The record would not decode back.
    """
    return _checked(
        {
            "status": end.status,
            "best_after": encode_scalar(end.best_after),
            "ended_at": end.ended_at,
        },
        decode_round_end,
    )


def decode_round_end(payload: Any) -> StoredRoundEnd:
    """Decode a ``lizyml:round:<k>:end`` record.

    ``completed`` must carry a numeric ``best_after`` and ``failed`` must not.
    """
    what = "round end record"
    obj = _expect_dict(payload, what)
    status = _choice(_field(obj, "status", what), ("completed", "failed"), "status")
    best = _number_or_none(_field(obj, "best_after", what), "best_after")
    if (status == "completed") != (best is not None):
        raise LineageCodecError(f"{status} round end has best_after={best!r}")
    return StoredRoundEnd(
        status=cast("Literal['completed', 'failed']", status),
        best_after=best,
        ended_at=_expect_str(_field(obj, "ended_at", what), "ended_at"),
    )
