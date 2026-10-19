"""The tuning-history codec (H-0134 P2): exact round trip, strict refusal."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from lizyml.core.tuning_codec import (
    HISTORY_SCHEMA,
    TRIAL_STATES,
    LineageCodecError,
    TuningHistory,
    history_bytes,
    parse_history,
)
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.core.types.tuning_result import (
    BoundaryDimStatus,
    BoundaryReport,
    RoundSummary,
    TrialResult,
)
from tests._tuning_compare import same_value

NAN = float("nan")


def _space() -> tuple[Any, ...]:
    return (
        FloatDim("learning_rate", 0.01, 0.3, log=True, category="model"),
        IntDim("max_depth", 3, 9, category="model", step=2),
        CategoricalDim("boosting", (None, True, 1, 1.5, "gbdt", NAN), category="model"),
    )


def _history() -> TuningHistory:
    """Every field shape the writers can produce (P2, criterion 3)."""
    trials = [
        TrialResult(
            0,
            {"max_depth": 5, "learning_rate": 0.1, "boosting": "gbdt"},
            0.5,
            "complete",
            1,
        ),
        TrialResult(1, {"learning_rate": NAN, "boosting": None}, NAN, "fail", 1),
        TrialResult(2, {"boosting": True, "max_depth": 3}, 0.7, "pruned", 2),
        TrialResult(3, {"learning_rate": math.inf, "boosting": 1}, NAN, "running", 2),
        TrialResult(
            4, {"learning_rate": -math.inf, "boosting": 1.5}, NAN, "waiting", 3
        ),
    ]
    rounds = (
        RoundSummary(1, 2, None, 0.5, (), _space()),
        RoundSummary(2, 2, 0.5, 0.7, ("max_depth",), _space()),
    )
    report = BoundaryReport(
        dims=(
            BoundaryDimStatus("max_depth", 9, 3, 9, 1.0, "upper", True, 3, 13, True),
            BoundaryDimStatus(
                "boosting", "gbdt", None, None, None, "none", False, None, None
            ),
        ),
        expanded_names=("max_depth",),
    )
    return TuningHistory(trials=trials, rounds=rounds, boundary_report=report)


def _payload() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(history_bytes(_history()))
    return payload


def _parse(payload: Any) -> TuningHistory:
    return parse_history(json.dumps(payload).encode("utf-8"))


def test_round_trip_keeps_every_value_type_order_and_nan() -> None:
    history = _history()
    back = parse_history(history_bytes(history))
    assert same_value(back, history)
    assert type(back.trials) is list and type(back.rounds) is tuple
    assert list(back.trials[0].params) == ["max_depth", "learning_rate", "boosting"]


def test_written_bytes_are_strict_json_and_name_the_schema() -> None:
    text = history_bytes(_history()).decode("utf-8")
    json.loads(text, parse_constant=lambda c: pytest.fail(f"bare {c} in the file"))
    assert json.loads(text)["schema"] == HISTORY_SCHEMA


def test_empty_history_round_trips() -> None:
    empty = TuningHistory(trials=[], rounds=(), boundary_report=None)
    assert same_value(parse_history(history_bytes(empty)), empty)


def test_trial_states_are_optunas() -> None:
    optuna = pytest.importorskip("optuna")
    assert {s.name.lower() for s in optuna.trial.TrialState} == TRIAL_STATES


def test_a_value_outside_the_scalar_set_is_refused_on_write() -> None:
    bad = TuningHistory(
        trials=[TrialResult(0, {"x": (1, 2)}, 0.1, "complete", 1)],
        rounds=(),
        boundary_report=None,
    )
    with pytest.raises(LineageCodecError):
        history_bytes(bad)


def _mutations() -> dict[str, Any]:
    """One malformed payload per refusal rule of P2 (criterion 4)."""

    def edit(fn: Any) -> Any:
        payload = _payload()
        fn(payload)
        return payload

    trial = lambda p: p["trials"][0]  # noqa: E731
    rnd = lambda p: p["rounds"][0]  # noqa: E731
    status = lambda p: p["boundary_report"]["dims"][0]  # noqa: E731
    return {
        "unknown schema": edit(lambda p: p.update(schema="lizyml-tuning-history/v2")),
        "missing top key": edit(lambda p: p.pop("rounds")),
        "extra top key": edit(lambda p: p.update(extra=1)),
        "trials not a list": edit(lambda p: p.update(trials={})),
        "extra trial key": edit(lambda p: trial(p).update(note="x")),
        "missing trial key": edit(lambda p: trial(p).pop("state")),
        "bool trial number": edit(lambda p: trial(p).update(number=True)),
        "negative trial number": edit(lambda p: trial(p).update(number=-1)),
        "duplicate trial number": edit(lambda p: p["trials"][1].update(number=0)),
        "round zero": edit(lambda p: trial(p).update(round=0)),
        "unknown state": edit(lambda p: trial(p).update(state="cancelled")),
        "unknown tag": edit(lambda p: trial(p).update(score=["decimal", "1"])),
        "string score": edit(lambda p: trial(p).update(score=["str", "x"])),
        "params not pairs": edit(lambda p: trial(p).update(params={"a": ["int", 1]})),
        "duplicate param name": edit(
            lambda p: trial(p).update(params=[["a", ["int", 1]], ["a", ["int", 2]]])
        ),
        "extra round key": edit(lambda p: rnd(p).update(note="x")),
        "round n_trials zero": edit(lambda p: rnd(p).update(n_trials=0)),
        "expanded dim not str": edit(lambda p: rnd(p).update(expanded_dims=[1])),
        "bad snapshot dim": edit(
            lambda p: rnd(p)["space_snapshot"][0].update(kind="x")
        ),
        "extra report key": edit(lambda p: p["boundary_report"].update(note="x")),
        "extra status key": edit(lambda p: status(p).update(note="x")),
        # decode_dim reads a numeric dim without "step" (a pre-H-0124 study
        # record) as step=None, which encode_dim writes back as null: not the
        # form the history writer produces, so the canonical check refuses it.
        "non-canonical dim without step": edit(
            lambda p: rnd(p)["space_snapshot"][0].pop("step")
        ),
    }


@pytest.mark.parametrize("case", sorted(_mutations()))
def test_malformed_history_is_refused(case: str) -> None:
    with pytest.raises(LineageCodecError):
        _parse(_mutations()[case])


def test_duplicate_json_keys_are_refused() -> None:
    raw = history_bytes(_history()).decode("utf-8")
    doubled = raw.replace('"schema":', '"schema":"x","schema":', 1)
    with pytest.raises(LineageCodecError):
        parse_history(doubled.encode("utf-8"))


@pytest.mark.parametrize("raw", [b"", b"[]", b"null", b"\xff", b'{"schema": NaN}'])
def test_unparseable_bytes_are_refused(raw: bytes) -> None:
    with pytest.raises(LineageCodecError):
        parse_history(raw)
