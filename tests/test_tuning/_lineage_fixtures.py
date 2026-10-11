"""Synthetic lineage records and trial stand-ins for the lineage unit tests.

Attribute values pass through the encoders, which return the stored (JSON)
form, so every helper produces what a storage would hand back.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from lizyml.core.types.search_dim import FloatDim
from lizyml.tuning.lineage import (
    SCHEMA_VERSION,
    STUDY_KEY,
    Provenance,
    Route,
    StoredRound,
    StoredRoundEnd,
    StoredStudy,
    encode_round,
    encode_round_end,
    encode_study,
    end_key,
    objective_digest,
    round_key,
)
from lizyml.tuning.lineage_codec import to_attr_value

# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------


class St(Enum):
    COMPLETE = 1
    FAIL = 2
    PRUNED = 3
    RUNNING = 4
    WAITING = 5


T0 = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)


@dataclass
class FakeTrial:
    number: int
    state: St = St.COMPLETE
    params: dict[str, Any] = field(default_factory=lambda: {"learning_rate": 0.1})
    value: float | None = 1.0
    seconds: float | None = 10.0
    datetime_start: datetime | None = None
    datetime_complete: datetime | None = None

    def __post_init__(self) -> None:
        if self.datetime_start is None:
            self.datetime_start = T0 + timedelta(hours=self.number)
        if self.datetime_complete is None and self.seconds is not None:
            self.datetime_complete = self.datetime_start + timedelta(
                seconds=self.seconds
            )


LR = FloatDim("learning_rate", 0.01, 0.3, log=True)
LR_BOUNDED = replace(LR, min_allowed=1e-8, max_allowed=1.0)


def stored_study(**overrides: Any) -> StoredStudy:
    elements = {"data.values": "a" * 64, "config.model": "b" * 64}
    base: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "objective_digest": objective_digest(elements),
        "objective_elements": elements,
        "direction": "minimize",
        "versions": {"lizyml": "0.19.0", "optuna": "4.7.0"},
        "space_mode": "replace",
        "fixed": {},
        "origin_space": (LR,),
        "created_at": T0.isoformat(),
    }
    base.update(overrides)
    return StoredStudy(**base)


def stored_round(k: int, start: int, **overrides: Any) -> StoredRound:
    base: dict[str, Any] = {
        "round": k,
        "source": "config" if k == 1 else "previous",
        "space_arg": None,
        "provenance": (Provenance("model", "learning_rate", "user"),),
        "declared": (LR,),
        "effective": (LR_BOUNDED,),
        "expand_boundary": False,
        "boundary_threshold": 0.05,
        "boundary_report": None,
        "n_trials": 3,
        "timeout": None,
        "start_trial": start,
        "best_before": None if k == 1 else 1.0,
        "routes_added": (Route("learning_rate", "learning_rate", "model"),)
        if k == 1
        else (),
    }
    base.update(overrides)
    return StoredRound(**base)


def stored_end(status: str = "completed", best: float | None = 1.0) -> StoredRoundEnd:
    return StoredRoundEnd(
        status=status,  # type: ignore[arg-type]
        best_after=best if status == "completed" else None,
        ended_at=T0.isoformat(),
    )


def attrs_of(
    study: StoredStudy | None = None,
    rounds: tuple[StoredRound, ...] = (),
    ends: dict[int, StoredRoundEnd] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if study is not None:
        attrs[STUDY_KEY] = to_attr_value(encode_study(study))
    for r in rounds:
        attrs[round_key(r.round)] = to_attr_value(encode_round(r))
    for k, e in (ends or {}).items():
        attrs[end_key(k)] = to_attr_value(encode_round_end(e))
    attrs.update(extra or {})
    return attrs


def make_trials(n: int, **kw: Any) -> list[FakeTrial]:
    return [FakeTrial(number=i, **kw) for i in range(n)]


def two_round_attrs(end2: bool = True) -> dict[str, Any]:
    return attrs_of(
        stored_study(),
        (stored_round(1, 0), stored_round(2, 3)),
        {1: stored_end(), 2: stored_end()} if end2 else {1: stored_end()},
    )
