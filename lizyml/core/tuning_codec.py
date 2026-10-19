"""Codecs for the tuning values LizyML stores as JSON (H-0122, H-0134).

Moved from :mod:`lizyml.tuning.lineage_codec` to Layer 0 by H-0134 (judgement 4),
so that both the Optuna study lineage (Layer 2) and the artifact writer and
reader (Layer 3) encode a tuning value the same way. The move changes no
behaviour; :mod:`lizyml.tuning.lineage_codec` re-exports every name.

JSON cannot tell ``True`` from ``1`` from ``1.0`` once a reader is lenient, and
has no NaN. Every value slot that is not a plain flag or name is therefore a
*tagged scalar* ``[tag, value]``, tagged by ``type(v) is ...`` exactly as a
categorical choice is admitted (``search_space._ALLOWED_CHOICE_TYPES``), and the
decoder checks the tag against the JSON type with the same identity test.

This module holds tagged scalars, search dims (and :func:`dim_signature`, the
one comparison form of a dim), boundary reports, and the tuning history an
artifact stores (:func:`history_bytes` / :func:`parse_history`, H-0134 P2). It
is pure (no Optuna import). Bad input raises :class:`LineageCodecError`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple, cast

from lizyml.core.magnitude import finite_float, int_too_large, width_too_large
from lizyml.core.types.search_dim import (
    CategoricalDim,
    DimCategory,
    FloatDim,
    IntDim,
    SearchDim,
)
from lizyml.core.types.tuning_result import (
    BoundaryDimStatus,
    BoundaryReport,
    RoundSummary,
    TrialResult,
)

__all__ = [
    "DIM_CATEGORIES",
    "HISTORY_FILE",
    "HISTORY_SCHEMA",
    "TRIAL_STATES",
    "LineageCodecError",
    "TuningHistory",
    "history_bytes",
    "parse_history",
    "decode_dim",
    "decode_dims",
    "decode_report",
    "decode_scalar",
    "dim_signature",
    "encode_dim",
    "encode_dims",
    "encode_report",
    "encode_scalar",
]

DIM_CATEGORIES: frozenset[str] = frozenset({"model", "smart", "training"})
"""The values ``SearchDim.category`` may take (``DimCategory``)."""

_SPECIAL_FLOATS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
_EDGES = frozenset({"lower", "upper", "none"})


class LineageCodecError(ValueError):
    """A value cannot be encoded, or a stored payload cannot be decoded."""


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


_NUMERIC_DIM_KEYS = frozenset(
    {
        "kind",
        "name",
        "category",
        "low",
        "high",
        "log",
        "min_allowed",
        "max_allowed",
        "step",
    }
)
"""The keys of an encoded ``float`` / ``int`` dim; ``step`` may be absent."""

_CATEGORICAL_DIM_KEYS = frozenset({"kind", "name", "category", "choices"})
"""The keys of an encoded ``categorical`` dim."""


def encode_dim(dim: SearchDim) -> dict[str, Any]:
    """Encode a ``SearchDim`` so that ``decode_dim`` returns an equal dim.

    A numeric dim's ``step`` (H-0124) is written, as ``null`` when it is
    ``None``, so :func:`dim_signature` compares it.
    """
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
    for label in ("min_allowed", "max_allowed", "step"):
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
        "step": encode_scalar(dim.step),
    }


def decode_dim(payload: Any) -> SearchDim:
    """Decode one encoded dim; refuse anything ``encode_dim`` cannot produce.

    A key the dim's kind does not have is refused, not dropped. A numeric dim
    without ``step`` -- a record written before H-0124 -- has ``step=None``.
    """
    obj = _expect_dict(payload, "dim")
    name = _expect_str(_field(obj, "name", "dim"), "dim name")
    what = f"dim '{name}'"
    category = _category(_field(obj, "category", what), what)
    kind = _field(obj, "kind", what)
    if kind not in ("categorical", "float", "int"):
        raise LineageCodecError(f"{what}: unknown kind {kind!r}")
    allowed = _CATEGORICAL_DIM_KEYS if kind == "categorical" else _NUMERIC_DIM_KEYS
    unknown = sorted(key for key in obj if key not in allowed)
    if unknown:
        raise LineageCodecError(f"{what} has unknown key(s) {unknown}")
    if kind == "categorical":
        raw = _expect_list(_field(obj, "choices", what), f"{what} choices")
        if not raw:
            raise LineageCodecError(f"{what} has no choices")
        choices = tuple(decode_scalar(c) for c in raw)
        return CategoricalDim(name=name, choices=choices, category=category)
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
    step = (
        _decode_optional_number(obj["step"], ints_only=ints_only)
        if "step" in obj
        else None
    )
    _check_magnitude(what, ints_only, low, high, step, lo_bound, hi_bound)
    if ints_only:
        return IntDim(
            name=name,
            low=int(low),
            high=int(high),
            log=log,
            category=category,
            min_allowed=lo_bound,
            max_allowed=hi_bound,
            step=step,
        )
    return FloatDim(
        name=name,
        low=low,
        high=high,
        log=log,
        category=category,
        min_allowed=lo_bound,
        max_allowed=hi_bound,
        step=step,
    )


def _check_magnitude(
    what: str,
    ints_only: bool,
    low: int | float,
    high: int | float,
    step: Any,
    lo_bound: Any,
    hi_bound: Any,
) -> None:
    """Refuse a stored dim outside H-0124 amendment 1 policy 9.

    Every dim LizyML records passed policy 9 (and the provider bounds are
    constants), so a dim outside it was written before the policy existed or
    by hand: a ``float`` dim's ``low`` / ``high`` / ``step`` are finite and
    its exact width is at most the largest float; an ``int`` dim's ``low`` /
    ``high`` / ``step`` are at most ``2**52`` in absolute value. A recorded
    ``min_allowed`` / ``max_allowed`` is a finite ``float`` (the expansion and
    the edge detection read it as one). Refused here, such a record makes the
    study ``unreadable`` / ``inconsistent`` before anything plans with it.
    """
    for label, bound in (("min_allowed", lo_bound), ("max_allowed", hi_bound)):
        if bound is not None and not finite_float(bound):
            raise LineageCodecError(f"{what} {label} {bound!r} is not a finite float")
    values = {"low": low, "high": high, "step": step}
    if ints_only:
        for label, value in values.items():
            if value is not None and int_too_large(value):
                raise LineageCodecError(
                    f"{what} {label} {value!r} is past 2**52 in absolute value"
                )
        return
    for label, value in values.items():
        if value is not None and not finite_float(value):
            raise LineageCodecError(f"{what} {label} {value!r} is not finite")
    if width_too_large(low, high):
        raise LineageCodecError(
            f"{what} range [{low!r}, {high!r}] is wider than the largest float"
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


# ---------------------------------------------------------------------------
# Tuning history (H-0134)
# ---------------------------------------------------------------------------

HISTORY_FILE = "tuning_history.json"
"""The artifact file that holds a tuned model's history (H-0134 P3)."""

HISTORY_SCHEMA = "lizyml-tuning-history/v1"
"""The schema the history file and its ``metadata.json`` declaration name."""

TRIAL_STATES: frozenset[str] = frozenset(
    {"complete", "pruned", "fail", "running", "waiting"}
)
"""``TrialResult.state``: Optuna's five ``TrialState`` names, lower-cased."""

_HISTORY_KEYS = frozenset({"schema", "trials", "rounds", "boundary_report"})
_TRIAL_KEYS = frozenset({"number", "params", "score", "state", "round"})
_ROUND_KEYS = frozenset(
    {
        "round",
        "n_trials",
        "best_score_before",
        "best_score_after",
        "expanded_dims",
        "space_snapshot",
    }
)
_REPORT_KEYS = frozenset({"dims", "expanded_names"})
_STATUS_KEYS = frozenset(
    {
        "name",
        "best_value",
        "low",
        "high",
        "position_pct",
        "edge",
        "expanded",
        "new_low",
        "new_high",
        "clamped_to_bound",
    }
)


class TuningHistory(NamedTuple):
    """The part of a ``TuningResult`` that ``metadata.json`` does not hold."""

    trials: list[TrialResult]
    rounds: tuple[RoundSummary, ...]
    boundary_report: BoundaryReport | None


def _exact_keys(obj: dict[str, Any], keys: frozenset[str], what: str) -> None:
    if set(obj) != keys:
        raise LineageCodecError(
            f"{what} must have exactly the keys {sorted(keys)}, got {sorted(obj)}"
        )


def _int_at_least(value: Any, minimum: int, what: str) -> int:
    if type(value) is not int or value < minimum:
        raise LineageCodecError(f"{what} must be an int >= {minimum}, got {value!r}")
    return value


def _encode_score(value: Any, what: str) -> list[Any]:
    _check_number(value, what)
    return encode_scalar(value)


def _decode_score(payload: Any, what: str) -> int | float:
    try:
        return _decode_number(payload)
    except LineageCodecError as exc:
        raise LineageCodecError(f"{what}: {exc}") from exc


def _encode_trial(trial: TrialResult) -> dict[str, Any]:
    what = f"trial {trial.number!r}"
    _int_at_least(trial.number, 0, f"{what} number")
    _int_at_least(trial.round, 1, f"{what} round")
    if trial.state not in TRIAL_STATES:
        raise LineageCodecError(f"{what}: unknown state {trial.state!r}")
    return {
        "number": trial.number,
        "params": [
            [_expect_str(name, f"{what} param name"), encode_scalar(value)]
            for name, value in trial.params.items()
        ],
        "score": _encode_score(trial.score, f"{what} score"),
        "state": trial.state,
        "round": trial.round,
    }


def _decode_trial(payload: Any) -> TrialResult:
    obj = _expect_dict(payload, "trial")
    _exact_keys(obj, _TRIAL_KEYS, "trial")
    number = _int_at_least(obj["number"], 0, "trial number")
    what = f"trial {number}"
    state = obj["state"]
    if type(state) is not str or state not in TRIAL_STATES:
        raise LineageCodecError(f"{what}: unknown state {state!r}")
    params: dict[str, Any] = {}
    for pair in _expect_list(obj["params"], f"{what} params"):
        if type(pair) is not list or len(pair) != 2:
            raise LineageCodecError(f"{what}: a param must be a [name, value] pair")
        name = _expect_str(pair[0], f"{what} param name")
        if name in params:
            raise LineageCodecError(f"{what}: param {name!r} appears twice")
        params[name] = decode_scalar(pair[1])
    return TrialResult(
        number=number,
        params=params,
        score=_decode_score(obj["score"], f"{what} score"),
        state=state,
        round=_int_at_least(obj["round"], 1, f"{what} round"),
    )


def _encode_round(summary: RoundSummary) -> dict[str, Any]:
    what = f"round {summary.round!r}"
    _int_at_least(summary.round, 1, what)
    _int_at_least(summary.n_trials, 1, f"{what} n_trials")
    before = summary.best_score_before
    return {
        "round": summary.round,
        "n_trials": summary.n_trials,
        "best_score_before": (
            encode_scalar(None)
            if before is None
            else _encode_score(before, f"{what} best_score_before")
        ),
        "best_score_after": _encode_score(
            summary.best_score_after, f"{what} best_score_after"
        ),
        "expanded_dims": [
            _expect_str(n, f"{what} expanded dim") for n in summary.expanded_dims
        ],
        "space_snapshot": encode_dims(summary.space_snapshot),
    }


def _decode_round(payload: Any) -> RoundSummary:
    obj = _expect_dict(payload, "round")
    _exact_keys(obj, _ROUND_KEYS, "round")
    number = _int_at_least(obj["round"], 1, "round")
    what = f"round {number}"
    before_raw = obj["best_score_before"]
    return RoundSummary(
        round=number,
        n_trials=_int_at_least(obj["n_trials"], 1, f"{what} n_trials"),
        best_score_before=(
            None
            if decode_scalar(before_raw) is None
            else _decode_score(before_raw, f"{what} best_score_before")
        ),
        best_score_after=_decode_score(
            obj["best_score_after"], f"{what} best_score_after"
        ),
        expanded_dims=tuple(
            _expect_str(n, f"{what} expanded dim")
            for n in _expect_list(obj["expanded_dims"], f"{what} expanded_dims")
        ),
        space_snapshot=decode_dims(obj["space_snapshot"]),
    )


def _strict_report(payload: Any) -> BoundaryReport:
    """``decode_report`` that also refuses unknown keys.

    The study-lineage reader keeps ``decode_report`` as it was (H-0134 P1);
    only the history reader is strict.
    """
    obj = _expect_dict(payload, "boundary report")
    _exact_keys(obj, _REPORT_KEYS, "boundary report")
    for status in _expect_list(obj["dims"], "report dims"):
        status_obj = _expect_dict(status, "boundary status")
        _exact_keys(status_obj, _STATUS_KEYS, "boundary status")
    return decode_report(obj)


def _encode_parts(history: TuningHistory) -> dict[str, Any]:
    numbers = [t.number for t in history.trials]
    if len(set(numbers)) != len(numbers):
        raise LineageCodecError("trial numbers must be unique")
    report = history.boundary_report
    return {
        "schema": HISTORY_SCHEMA,
        "trials": [_encode_trial(t) for t in history.trials],
        "rounds": [_encode_round(r) for r in history.rounds],
        "boundary_report": None if report is None else encode_report(report),
    }


def history_bytes(history: TuningHistory) -> bytes:
    """Encode *history* as the bytes of ``tuning_history.json``.

    Strict JSON (no bare NaN: NaN and the infinities are tagged), keeping each
    trial's parameter order. Raises ``LineageCodecError`` for a value outside
    the tagged scalar set, an unknown trial state, a duplicate trial number or
    a round number below 1.
    """
    payload = _encode_parts(history)
    text = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    return text.encode("utf-8")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise LineageCodecError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def _no_constant(name: str) -> Any:
    raise LineageCodecError(f"bare {name} is not JSON")


def parse_history(raw: bytes) -> TuningHistory:
    """Decode the bytes of ``tuning_history.json``, in its canonical form only.

    Refuses bytes that are not UTF-8 strict JSON, duplicate keys, an unknown
    schema, an object whose keys differ from the written set, a value of the
    wrong type, an unknown tag or trial state, a duplicate parameter name or
    trial number, and any payload that does not re-encode to itself.
    """
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_no_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LineageCodecError(f"tuning history is not JSON: {exc}") from exc
    obj = _expect_dict(payload, "tuning history")
    _exact_keys(obj, _HISTORY_KEYS, "tuning history")
    if obj["schema"] != HISTORY_SCHEMA:
        raise LineageCodecError(f"unknown tuning history schema {obj['schema']!r}")
    report_raw = obj["boundary_report"]
    history = TuningHistory(
        trials=[_decode_trial(t) for t in _expect_list(obj["trials"], "trials")],
        rounds=tuple(_decode_round(r) for r in _expect_list(obj["rounds"], "rounds")),
        boundary_report=None if report_raw is None else _strict_report(report_raw),
    )
    if _encode_parts(history) != obj:
        raise LineageCodecError("tuning history is not in the form the writer writes")
    return history
