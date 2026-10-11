"""Unit tests for the lineage record codecs (H-0122 §7.1, PR 1 design §2).

``lizyml:study``, ``lizyml:round:<k>`` and ``lizyml:round:<k>:end`` round-trip
exactly, malformed payloads are refused, and an encoder refuses any record its
own decoder would refuse (writer/reader symmetry).
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import optuna
import pytest

from lizyml.core.types.search_dim import CategoricalDim, IntDim
from lizyml.core.types.tuning_result import BoundaryDimStatus, BoundaryReport
from lizyml.tuning.lineage import (
    SCHEMA_VERSION,
    STUDY_KEY,
    Provenance,
    StoredRoundEnd,
    classify_study,
    decode_round,
    decode_round_end,
    decode_study,
    encode_round,
    encode_round_end,
    encode_study,
    end_key,
    objective_digest,
    round_key,
)
from lizyml.tuning.lineage_codec import LineageCodecError, to_attr_value
from tests.test_tuning._lineage_fixtures import (
    LR,
    LR_BOUNDED,
    stored_end,
    stored_round,
    stored_study,
)

# ---------------------------------------------------------------------------
# Record codecs
# ---------------------------------------------------------------------------


class TestRecordCodecs:
    def test_study_round_trip(self) -> None:
        study = stored_study(
            space_mode="merge",
            fixed={"first_metric_only": True, "metric": ["huber", "mae"]},
            origin_space=(LR, CategoricalDim("b", (True, 1, 1.0))),
        )
        assert decode_study(to_attr_value(encode_study(study))) == study

    def test_round_round_trip_with_report_and_timeout(self) -> None:
        report = BoundaryReport(
            dims=(
                BoundaryDimStatus(
                    name="learning_rate",
                    best_value=0.29,
                    low=0.01,
                    high=0.3,
                    position_pct=0.97,
                    edge="upper",
                    expanded=True,
                    new_low=None,
                    new_high=0.9,
                ),
            ),
            expanded_names=("learning_rate",),
        )
        record = stored_round(
            2,
            5,
            boundary_report=report,
            timeout=3600.0,
            expand_boundary=True,
            provenance=(
                Provenance("model", "learning_rate", "default"),
                Provenance("training", "early_stopping_rounds", "user"),
            ),
            effective=(replace(LR_BOUNDED, high=0.9), IntDim("x", 1, 2)),
        )
        assert decode_round(to_attr_value(encode_round(record))) == record

    def test_end_round_trip(self) -> None:
        for end in (stored_end("completed", 1.5), stored_end("failed")):
            assert decode_round_end(to_attr_value(encode_round_end(end))) == end

    def test_unknown_payload_keys_are_ignored(self) -> None:
        payload = to_attr_value(encode_round(stored_round(1, 0)))
        payload["future_field"] = 1
        assert decode_round(payload) == stored_round(1, 0)

    def test_objective_digest_is_order_independent(self) -> None:
        a = {"x": "1", "y": "2"}
        assert objective_digest(a) == objective_digest({"y": "2", "x": "1"})
        assert objective_digest(a) != objective_digest({"x": "1", "y": "3"})
        assert objective_digest(a) != objective_digest({"x": "1"})

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: p.pop("objective"),
            lambda p: p.update(direction="up"),
            lambda p: p.update(space_mode="mix"),
            lambda p: p.update(versions={"a": 1}),
            lambda p: p.update(fixed=[1]),
            lambda p: p.update(origin_space=[{"kind": "x"}]),
            lambda p: p["objective"].update(digest="0" * 64),
            lambda p: p["objective"]["elements"].update({"data.values": 1}),
            lambda p: p.update(created_at=1),
        ],
    )
    def test_malformed_study_payload_is_refused(self, mutate: Any) -> None:
        payload = to_attr_value(encode_study(stored_study()))
        mutate(payload)
        with pytest.raises(LineageCodecError):
            decode_study(payload)

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: p.update(round=True),
            lambda p: p.update(round=0),
            lambda p: p.update(start_trial=-1),
            lambda p: p.update(start_trial=1.0),
            lambda p: p["budget"].update(n_trials=True),
            lambda p: p["budget"].update(n_trials=-1),
            lambda p: p["budget"].update(timeout=["str", "1"]),
            lambda p: p.update(expand_boundary=None),
            lambda p: p.update(best_before=["str", "x"]),
            lambda p: p.update(routes_added=[{"name": "a", "canonical": "a"}]),
            lambda p: p.update(
                routes_added=[{"name": "a", "canonical": "a", "category": "x"}]
            ),
            lambda p: p["declaration"].update(source="elsewhere"),
            lambda p: p["declaration"].update(provenance=[["model", "a", "maybe"]]),
            lambda p: p["declaration"].update(provenance=[["model", "a"]]),
            lambda p: p.update(boundary_report={"dims": 1}),
            lambda p: p.update(effective="x"),
            lambda p: p.pop("declared"),
        ],
    )
    def test_malformed_round_payload_is_refused(self, mutate: Any) -> None:
        payload = to_attr_value(encode_round(stored_round(1, 0)))
        mutate(payload)
        with pytest.raises(LineageCodecError):
            decode_round(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            {"status": "done", "best_after": ["none", None], "ended_at": "t"},
            {"status": "completed", "best_after": ["none", None], "ended_at": "t"},
            {"status": "failed", "best_after": ["float", 1.0], "ended_at": "t"},
            {"status": "completed", "best_after": ["str", "1"], "ended_at": "t"},
            {"status": "completed", "best_after": ["float", 1.0]},
            ["completed"],
        ],
    )
    def test_malformed_end_payload_is_refused(self, payload: Any) -> None:
        with pytest.raises(LineageCodecError):
            decode_round_end(payload)

    @pytest.mark.parametrize(
        "build",
        [
            lambda: encode_study(stored_study(direction="up")),
            lambda: encode_study(stored_study(schema=SCHEMA_VERSION + 1)),
            lambda: encode_study(stored_study(objective_digest="0" * 64)),
            lambda: encode_study(stored_study(fixed={"x": float("nan")})),
            lambda: encode_round(stored_round(1, 0, n_trials=-1)),
            lambda: encode_round(stored_round(1, 0, source="space_arg")),
            lambda: encode_round(stored_round(1, 0, best_before="x")),
            lambda: encode_round_end(StoredRoundEnd("completed", None, "t")),
            lambda: encode_round_end(StoredRoundEnd("failed", 1.0, "t")),
        ],
    )
    def test_encoder_refuses_what_the_reader_would_refuse(self, build: Any) -> None:
        # Writer/reader symmetry: a record that would read back as S or X is
        # refused while planning, before anything is written (INV-R4).
        with pytest.raises(LineageCodecError):
            build()

    def test_encoders_return_the_stored_form(self) -> None:
        stored = encode_round(stored_round(1, 0))
        assert to_attr_value(stored) == stored
        assert isinstance(stored["declared"], list)

    def test_records_are_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            stored_round(1, 0).start_trial = 4  # type: ignore[misc]


# ---------------------------------------------------------------------------
# A real RDB storage (DC7: the JSON round trip is executed, not emulated)
# ---------------------------------------------------------------------------


def test_records_survive_a_sqlite_storage(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 's.db'}"
    study_rec = stored_study(
        origin_space=(CategoricalDim("c", (True, 1, 1.0, float("nan"), None)),)
    )
    round_rec = stored_round(
        1,
        0,
        effective=(IntDim("big", 2**53 + 1, 2**60), LR_BOUNDED),
        timeout=3600.0,
    )
    written = {
        STUDY_KEY: encode_study(study_rec),
        round_key(1): encode_round(round_rec),
        end_key(1): encode_round_end(stored_end()),
    }
    study = optuna.create_study(storage=url, study_name="s")
    for key, value in written.items():
        study.set_user_attr(key, value)
    study.optimize(lambda t: t.suggest_float("learning_rate", 0.01, 0.3), n_trials=1)

    loaded = optuna.load_study(storage=url, study_name="s")
    for key, value in written.items():
        assert loaded.user_attrs[key] == value, key
    assert decode_round(loaded.user_attrs[round_key(1)]) == round_rec
    choices = decode_study(loaded.user_attrs[STUDY_KEY]).origin_space[0].choices
    assert [type(c) for c in choices] == [bool, int, float, float, type(None)]
    assert math.isnan(choices[3])
    lin = classify_study(loaded.user_attrs, loaded.trials)
    assert lin.state == "completed"
    assert lin.rounds[0].effective[0] == IntDim("big", 2**53 + 1, 2**60)
