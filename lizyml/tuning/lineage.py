"""Study lineage: the records LizyML keeps in study attributes (H-0122 §7).

A study carries three kinds of ``lizyml:`` attribute, each written once:

* ``lizyml:study`` -- written when the study is created: the objective
  identity, library versions, ``space_mode``, ``fixed`` and the config space.
* ``lizyml:round:<k>`` -- written after round *k* is fully planned and before
  its first trial: the declared and effective spaces, the boundary report,
  the budget, the first trial number and the dimension routes it added.
* ``lizyml:round:<k>:end`` -- written when round *k* closes.

This module holds the pure reader :func:`classify_study`, which judges a
study's state by the closed §7.4 table, and the arithmetic over a lineage
(round membership, routes, remaining budget). The record types and their
codecs live in :mod:`lizyml.tuning.lineage_codec` and are re-exported here.
It imports no Optuna: trials are read through :class:`TrialLike`. It sits in
Layer 2 and must not import ``config`` or a concrete provider.

Round keys follow an exact grammar, ``lizyml:round:<k>`` and
``lizyml:round:<k>:end`` with *k* a positive integer written without leading
zeros. A key that starts with ``lizyml:round:`` and does not match it makes the
study ``inconsistent``; it is never skipped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

from lizyml.tuning.lineage_codec import (
    SCHEMA_VERSION,
    STUDY_KEY,
    LineageCodecError,
    Provenance,
    Route,
    StoredRound,
    StoredRoundEnd,
    StoredStudy,
    decode_round,
    decode_round_end,
    decode_study,
    encode_round,
    encode_round_end,
    encode_study,
    objective_digest,
)

__all__ = [
    "END_SUFFIX",
    "ROUND_KEY_PREFIX",
    "SCHEMA_VERSION",
    "STUDY_KEY",
    "Lineage",
    "LineageError",
    "LineageState",
    "Provenance",
    "Route",
    "RouteConflict",
    "StoredRound",
    "StoredRoundEnd",
    "StoredStudy",
    "TrialLike",
    "UnroutedParam",
    "classify_study",
    "decode_round",
    "decode_round_end",
    "decode_study",
    "encode_round",
    "encode_round_end",
    "encode_study",
    "end_key",
    "merge_routes",
    "objective_digest",
    "parse_round_key",
    "remaining_timeout",
    "remaining_trials",
    "round_key",
    "round_of",
    "split_by_routes",
    "trials_of_round",
]

ROUND_KEY_PREFIX = "lizyml:round:"
END_SUFFIX = ":end"

_ROUND_KEY_RE = re.compile(r"lizyml:round:([1-9][0-9]*)(:end)?")
"""Matched with ``fullmatch`` only; ``[0-9]`` because ``\\d`` admits any
Unicode digit and ``$`` admits a trailing newline."""

LineageState = Literal[
    "legacy",
    "uninitialized",
    "unreadable",
    "no_round",
    "inconsistent",
    "open",
    "completed",
    "failed",
]
_FINISHED_STATES = frozenset({"COMPLETE", "FAIL", "PRUNED", "RUNNING"})
_TIMED_STATES = frozenset({"COMPLETE", "PRUNED"})


class LineageError(ValueError):
    """Base class for lineage errors that are not codec errors."""


class RouteConflict(LineageError):
    """A dimension route would change a name's category or canonical name."""


class UnroutedParam(LineageError):
    """A trial parameter has no route in the lineage."""


class _StateLike(Protocol):
    @property
    def name(self) -> str: ...


class TrialLike(Protocol):
    """The part of ``optuna.trial.FrozenTrial`` the lineage reader uses."""

    @property
    def number(self) -> int: ...

    @property
    def state(self) -> _StateLike: ...

    @property
    def datetime_start(self) -> datetime | None: ...

    @property
    def datetime_complete(self) -> datetime | None: ...

    @property
    def params(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Lineage:
    """A study's state by the §7.4 table, with its decoded records.

    ``rounds`` / ``ends`` / ``routes`` are filled for ``open``, ``completed``
    and ``failed`` only; ``ends`` is aligned with ``rounds`` (``None`` for an
    open last round). ``reason`` explains ``unreadable`` and ``inconsistent``.
    """

    state: LineageState
    study: StoredStudy | None = None
    rounds: tuple[StoredRound, ...] = ()
    ends: tuple[StoredRoundEnd | None, ...] = ()
    routes: Mapping[str, Route] = field(default_factory=dict)
    open_round: int | None = None
    reason: str | None = None

    @property
    def last_round(self) -> StoredRound | None:
        """The highest-numbered round record, if any."""
        return self.rounds[-1] if self.rounds else None


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def _check_round_number(k: int) -> None:
    if type(k) is not int or k < 1:
        raise ValueError(f"round number must be a positive int, got {k!r}")


def round_key(k: int) -> str:
    """Return ``lizyml:round:<k>``."""
    _check_round_number(k)
    return f"{ROUND_KEY_PREFIX}{k}"


def end_key(k: int) -> str:
    """Return ``lizyml:round:<k>:end``."""
    return round_key(k) + END_SUFFIX


def parse_round_key(key: str) -> tuple[int, bool] | None:
    """Return ``(k, is_end)`` for a well-formed round key, else ``None``."""
    match = _ROUND_KEY_RE.fullmatch(key)
    if match is None:
        return None
    return int(match.group(1)), match.group(2) is not None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def merge_routes(
    existing: Mapping[str, Route], added: Iterable[Route]
) -> dict[str, Route]:
    """Return *existing* plus *added*; routes are append-only (§4.1b).

    Raises:
        RouteConflict: A name is re-routed differently, or a canonical name
            would be reached from two categories.
    """
    merged = dict(existing)
    by_canonical = {r.canonical: r for r in merged.values()}
    for route in added:
        prior = merged.get(route.name)
        if prior is not None and prior != route:
            raise RouteConflict(
                f"dimension '{route.name}' is routed as {prior.category}/"
                f"{prior.canonical}, not {route.category}/{route.canonical}"
            )
        same = by_canonical.get(route.canonical)
        if same is not None and same.category != route.category:
            raise RouteConflict(
                f"'{route.canonical}' is routed to {same.category} (as "
                f"'{same.name}'), not {route.category} (as '{route.name}')"
            )
        merged[route.name] = route
        by_canonical.setdefault(route.canonical, route)
    return merged


def split_by_routes(
    params: Mapping[str, Any], routes: Mapping[str, Route]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split trial params into ``(model, smart, training)`` by their routes.

    Keys keep the raw param name. Raises :class:`UnroutedParam` for a name
    with no route instead of guessing a category.
    """
    parts: dict[str, dict[str, Any]] = {"model": {}, "smart": {}, "training": {}}
    for name, value in params.items():
        route = routes.get(name)
        if route is None:
            raise UnroutedParam(f"trial parameter '{name}' has no route in the lineage")
        parts[route.category][name] = value
    return parts["model"], parts["smart"], parts["training"]


# ---------------------------------------------------------------------------
# classify_study (§7.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RoundKeys:
    rounds: dict[int, Any]
    ends: dict[int, Any]
    malformed: tuple[str, ...]

    @property
    def empty(self) -> bool:
        return not (self.rounds or self.ends or self.malformed)


def _collect_round_keys(user_attrs: Mapping[str, Any]) -> _RoundKeys:
    rounds: dict[int, Any] = {}
    ends: dict[int, Any] = {}
    malformed: list[str] = []
    for key, value in user_attrs.items():
        if not key.startswith(ROUND_KEY_PREFIX):
            continue
        parsed = parse_round_key(key)
        if parsed is None:
            malformed.append(key)
        elif parsed[1]:
            ends[parsed[0]] = value
        else:
            rounds[parsed[0]] = value
    return _RoundKeys(rounds, ends, tuple(sorted(malformed)))


def _structure_problem(keys: _RoundKeys, n: int) -> str | None:
    """The §7.4 X conditions readable from key names alone (with R2)."""
    if keys.malformed:
        return f"malformed round keys {list(keys.malformed)}"
    if not keys.rounds:
        return f"{n} trials but no round record" if n else "a round end with no round"
    numbers = sorted(keys.rounds)
    if numbers != list(range(1, len(numbers) + 1)):
        return f"round numbers {numbers} are not 1..{len(numbers)}"
    orphans = sorted(set(keys.ends) - set(keys.rounds))
    if orphans:
        return f"round end without its round: {orphans}"
    unclosed = [k for k in numbers[:-1] if k not in keys.ends]
    if unclosed:
        return f"rounds {unclosed} have no end but are not the last round"
    return None


def _decode_records(
    keys: _RoundKeys,
) -> tuple[tuple[StoredRound, ...], tuple[StoredRoundEnd | None, ...]]:
    rounds: list[StoredRound] = []
    ends: list[StoredRoundEnd | None] = []
    for k in sorted(keys.rounds):
        try:
            record = decode_round(keys.rounds[k])
        except LineageCodecError as exc:
            raise LineageCodecError(f"{round_key(k)}: {exc}") from exc
        if record.round != k:
            raise LineageCodecError(f"{round_key(k)} records round {record.round}")
        rounds.append(record)
        try:
            ends.append(decode_round_end(keys.ends[k]) if k in keys.ends else None)
        except LineageCodecError as exc:
            raise LineageCodecError(f"{end_key(k)}: {exc}") from exc
    return tuple(rounds), tuple(ends)


def _trial_problem(
    rounds: tuple[StoredRound, ...],
    routes: Mapping[str, Route],
    trials: Sequence[TrialLike],
) -> str | None:
    starts = [r.start_trial for r in rounds]
    if any(b < a for a, b in zip(starts, starts[1:], strict=False)):
        return f"round start trials {starts} decrease"
    if starts[-1] > len(trials):
        return (
            f"last round starts at trial {starts[-1]} but the study has {len(trials)}"
        )
    if trials and starts[0] > 0:
        return f"trials precede round 1, which starts at trial {starts[0]}"
    unrouted = sorted({p for t in trials for p in t.params} - set(routes))
    if unrouted:
        return f"trial parameters with no route: {unrouted}"
    return None


def classify_study(
    user_attrs: Mapping[str, Any], trials: Sequence[TrialLike]
) -> Lineage:
    """Judge a study's lineage state by the closed §7.4 table.

    The rows are tried in order and the first match wins: legacy, uninitialized,
    unreadable, no_round, inconsistent, open, then completed / failed from the
    last round's end. Nothing malformed is skipped: an undecodable study record
    is ``unreadable`` and an undecodable or contradictory round record is
    ``inconsistent``. Reads only; neither argument is modified.
    """
    n = len(trials)
    if STUDY_KEY not in user_attrs:
        return Lineage(state="legacy" if n else "uninitialized")
    try:
        study = decode_study(user_attrs[STUDY_KEY])
    except LineageCodecError as exc:
        return Lineage(state="unreadable", reason=str(exc))
    keys = _collect_round_keys(user_attrs)
    if keys.empty and n == 0:
        return Lineage(state="no_round", study=study)

    def inconsistent(reason: str) -> Lineage:
        return Lineage(state="inconsistent", study=study, reason=reason)

    problem = _structure_problem(keys, n)
    if problem is not None:
        return inconsistent(problem)
    try:
        rounds, ends = _decode_records(keys)
        routes: dict[str, Route] = {}
        for record in rounds:
            routes = merge_routes(routes, record.routes_added)
    except (LineageCodecError, RouteConflict) as exc:
        return inconsistent(str(exc))
    problem = _trial_problem(rounds, routes, trials)
    if problem is not None:
        return inconsistent(problem)
    last_end = ends[-1]
    if last_end is None:
        state: LineageState = "open"
    else:
        state = last_end.status
    return Lineage(
        state=state,
        study=study,
        rounds=rounds,
        ends=ends,
        routes=routes,
        open_round=rounds[-1].round if last_end is None else None,
    )


# ---------------------------------------------------------------------------
# Round membership and budgets (§7.1, §7.2)
# ---------------------------------------------------------------------------


def round_of(rounds: Sequence[StoredRound], number: int) -> int:
    """The round trial *number* belongs to: the largest k with start <= number."""
    owner: int | None = None
    for record in rounds:
        if record.start_trial <= number:
            owner = record.round
    if owner is None:
        raise ValueError(f"trial {number} precedes every round")
    return owner


def trials_of_round(
    lineage: Lineage, k: int, trials: Sequence[TrialLike]
) -> list[TrialLike]:
    """The trials whose number falls in round *k*'s range."""
    by_number = {r.round: r for r in lineage.rounds}
    if k not in by_number:
        raise ValueError(f"the lineage has no round {k}")
    start = by_number[k].start_trial
    nxt = by_number.get(k + 1)
    stop = nxt.start_trial if nxt is not None else None
    return [
        t for t in trials if t.number >= start and (stop is None or t.number < stop)
    ]


def remaining_trials(record: StoredRound, round_trials: Sequence[TrialLike]) -> int:
    """Recorded ``n_trials`` minus the round's COMPLETE/FAIL/PRUNED/RUNNING trials.

    RUNNING counts because the caller FAILs it before running the round (§7.2).
    """
    used = sum(1 for t in round_trials if t.state.name in _FINISHED_STATES)
    return max(0, record.n_trials - used)


def remaining_timeout(
    record: StoredRound, round_trials: Sequence[TrialLike]
) -> float | None:
    """Recorded timeout minus the COMPLETE and PRUNED trials' run time (§7.2).

    FAIL trials are excluded: one FAILed on reattach has a ``datetime_complete``
    that spans the downtime. ``None`` when the round recorded no timeout. May be
    zero or negative, meaning the budget is spent.

    Raises:
        ValueError: A COMPLETE or PRUNED trial lacks a timestamp; summing only
            the trials that have one would overstate the time left.
    """
    if record.timeout is None:
        return None
    spent = 0.0
    for t in round_trials:
        if t.state.name not in _TIMED_STATES:
            continue
        if t.datetime_start is None or t.datetime_complete is None:
            raise ValueError(f"trial {t.number} ({t.state.name}) has no start/end time")
        spent += (t.datetime_complete - t.datetime_start).total_seconds()
    return float(record.timeout) - spent
