"""The three study records LizyML writes, and their codecs (H-0122 §7.1).

``lizyml:study`` (:class:`StoredStudy`), ``lizyml:round:<k>``
(:class:`StoredRound`) and ``lizyml:round:<k>:end`` (:class:`StoredRoundEnd`).
Each encoder returns the stored (JSON) form and first proves that its decoder
accepts it, so a record the reader would refuse is refused while planning,
before any write (INV-R4).

Values inside the records -- scalars, search dims, boundary reports -- go
through :mod:`lizyml.tuning.lineage_codec`, which this module builds on. Pure
(no Optuna import), Layer 2. Bad input raises
:class:`~lizyml.tuning.lineage_codec.LineageCodecError`; the reader that
classifies a study (``lineage.classify_study``) turns that into the
``unreadable`` or ``inconsistent`` state rather than skipping the record.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from lizyml.core.tuning_codec import (
    DIM_CATEGORIES,
    LineageCodecError,
    _expect_bool,
    _expect_dict,
    _expect_list,
    _expect_str,
    _field,
    decode_dims,
    decode_report,
    decode_scalar,
    encode_dims,
    encode_report,
    encode_scalar,
)
from lizyml.core.types.search_dim import DimCategory, SearchDim
from lizyml.core.types.tuning_result import BoundaryReport
from lizyml.tuning.lineage_codec import SCHEMA_VERSION, STUDY_KEY, to_attr_value
from lizyml.tuning.lineage_digest import FIXED_ELEMENT, value_digest

__all__ = [
    "Provenance",
    "Route",
    "RoundSource",
    "StoredRound",
    "StoredRoundEnd",
    "StoredStudy",
    "decode_round",
    "decode_round_end",
    "decode_study",
    "encode_round",
    "encode_round_end",
    "encode_study",
    "objective_digest",
]

RoundSource = Literal["config", "previous", "space_arg"]
"""Where a round's declaration came from: the config (round 1), the previous
round (a resume without ``space=``), or ``tune(space=...)``."""
_ROUND_SOURCES: frozenset[str] = frozenset({"config", "previous", "space_arg"})
_ORIGINS: frozenset[str] = frozenset({"user", "default"})


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
    """The decoded ``lizyml:round:<k>`` record.

    ``space_arg`` is the round's ``tune(space=...)`` as parsed -- raw names
    and order as written, omitted keys filled in -- and is set exactly when
    ``source == "space_arg"``. The parsed dims, not the raw dict, are stored:
    the tagged codec carries a NaN choice, which a JSON dict cannot.
    """

    round: int
    source: RoundSource
    space_arg: tuple[SearchDim, ...] | None
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
    fixed = _expect_dict(_field(obj, "fixed", STUDY_KEY), "fixed")
    # The trials train on the stored `fixed`; identity checks only the element.
    # So the value must be the one the element digests (INV-R1, DC1).
    if FIXED_ELEMENT not in elements:
        raise LineageCodecError(f"objective elements lack {FIXED_ELEMENT}")
    if value_digest(fixed) != elements[FIXED_ELEMENT]:
        raise LineageCodecError(
            f"fixed does not match its objective element {FIXED_ELEMENT} "
            "(the record was edited or is corrupt)"
        )
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
        fixed=fixed,
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
                else encode_dims(record.space_arg),
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
    """Decode a ``lizyml:round:<k>`` record.

    ``space_arg`` must be present exactly when the source is ``space_arg``.
    """
    what = "round record"
    obj = _expect_dict(payload, what)
    declaration = _expect_dict(_field(obj, "declaration", what), "declaration")
    source = cast(
        "RoundSource",
        _choice(_field(declaration, "source", "declaration"), _ROUND_SOURCES, "source"),
    )
    raw_space_arg = _field(declaration, "space_arg", "declaration")
    space_arg = None if raw_space_arg is None else decode_dims(raw_space_arg)
    if (source == "space_arg") != (space_arg is not None):
        raise LineageCodecError(
            f"declaration source {source!r} with space_arg "
            f"{'present' if space_arg is not None else 'absent'}"
        )
    budget = _expect_dict(_field(obj, "budget", what), "budget")
    threshold = _number_or_none(_field(obj, "boundary_threshold", what), "threshold")
    if threshold is None:
        raise LineageCodecError("boundary_threshold must be a number")
    report = _field(obj, "boundary_report", what)
    return StoredRound(
        round=_count(_field(obj, "round", what), "round", minimum=1),
        source=source,
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
