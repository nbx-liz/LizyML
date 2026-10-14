"""Unit tests for the pure lineage reader (H-0122 §7.4, PR 1 design §3.1).

The study states are built from synthetic ``user_attrs`` and trial stand-ins
(``_lineage_fixtures``), so every row of the closed §7.4 table and every
``inconsistent`` sub-case is pinned without running a tune. Attribute values
are always in the stored (JSON) form a storage would return.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from typing import Any

import optuna
import pytest

from lizyml.core.types.search_dim import FloatDim, IntDim
from lizyml.tuning.lineage import (
    STUDY_KEY,
    Lineage,
    Route,
    RouteConflict,
    StoredRoundEnd,
    UnroutedParam,
    classify_study,
    encode_round,
    encode_study,
    end_key,
    merge_routes,
    parse_round_key,
    remaining_timeout,
    remaining_trials,
    round_key,
    round_of,
    split_by_routes,
    trials_of_round,
)
from lizyml.tuning.lineage_codec import SCHEMA_VERSION, to_attr_value
from tests.test_tuning._lineage_fixtures import (
    LR_BOUNDED,
    FakeTrial,
    St,
    attrs_of,
    make_trials,
    stored_end,
    stored_round,
    stored_study,
    two_round_attrs,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Key grammar (DC2: anchored, no prefix matching)
# ---------------------------------------------------------------------------


class TestKeyGrammar:
    def test_round_and_end_keys(self) -> None:
        assert round_key(3) == "lizyml:round:3"
        assert end_key(3) == "lizyml:round:3:end"
        assert parse_round_key("lizyml:round:3") == (3, False)
        assert parse_round_key("lizyml:round:12:end") == (12, True)

    @pytest.mark.parametrize(
        "key",
        [
            "lizyml:round:0",
            "lizyml:round:01",
            "lizyml:round:-1",
            "lizyml:round:",
            "lizyml:round:1:end:extra",
            "lizyml:round:1:END",
            "lizyml:round:1:",
            "lizyml:round:1 ",
            "lizyml:round:1\n",
            "lizyml:round:١",  # an Arabic-Indic digit one; \d would admit it
            "lizyml:round:x",
        ],
    )
    def test_malformed_round_keys_parse_to_none(self, key: str) -> None:
        assert parse_round_key(key) is None

    def test_round_key_refuses_non_positive(self) -> None:
        with pytest.raises(ValueError):
            round_key(0)
        with pytest.raises(ValueError):
            end_key(-1)


class TestClassifyStates:
    def test_legacy(self) -> None:
        lin = classify_study({}, make_trials(2))
        assert lin.state == "legacy"
        assert lin.study is None

    def test_legacy_ignores_other_attrs(self) -> None:
        assert classify_study({"foo": 1}, make_trials(1)).state == "legacy"

    def test_uninitialized(self) -> None:
        lin = classify_study({}, [])
        assert lin.state == "uninitialized"
        assert lin.rounds == () and lin.open_round is None

    def test_round_records_without_study_follow_the_table(self) -> None:
        # First match wins: no lizyml:study decides L / E before any round key
        # is read. LizyML's own writes cannot produce this combination.
        orphan = {round_key(1): to_attr_value(encode_round(stored_round(1, 0)))}
        assert classify_study(orphan, []).state == "uninitialized"
        assert classify_study(orphan, make_trials(1)).state == "legacy"

    @pytest.mark.parametrize(
        "payload",
        [
            {"schema": 999},
            {"schema": SCHEMA_VERSION + 1},
            {"schema": 0},
            {"schema": True},
            {"schema": 1.0},
            {"schema": "1"},
            {},
            "not a dict",
            None,
            [1],
        ],
    )
    def test_unreadable(self, payload: Any) -> None:
        lin = classify_study({STUDY_KEY: payload}, make_trials(1))
        assert lin.state == "unreadable"
        assert lin.reason

    def test_readable_schema_with_bad_body_is_unreadable(self) -> None:
        payload = to_attr_value(encode_study(stored_study()))
        payload["direction"] = "sideways"
        assert classify_study({STUDY_KEY: payload}, []).state == "unreadable"

    def test_unreadable_and_empty_is_unreadable_not_no_round(self) -> None:
        # Criterion 33: S is checked before N.
        lin = classify_study({STUDY_KEY: {"schema": 999}}, [])
        assert lin.state == "unreadable"

    def test_no_round(self) -> None:
        lin = classify_study(attrs_of(stored_study()), [])
        assert lin.state == "no_round"
        assert lin.study == stored_study()
        assert lin.rounds == ()

    def test_open_round_one(self) -> None:
        lin = classify_study(
            attrs_of(stored_study(), (stored_round(1, 0),)), make_trials(2)
        )
        assert lin.state == "open"
        assert lin.open_round == 1
        assert lin.rounds == (stored_round(1, 0),)
        assert lin.ends == (None,)

    def test_open_with_no_trial_yet(self) -> None:
        # Record written, crash before the first trial: start_trial == n.
        lin = classify_study(
            attrs_of(
                stored_study(),
                (stored_round(1, 0), stored_round(2, 3)),
                {1: stored_end()},
            ),
            make_trials(3),
        )
        assert lin.state == "open"
        assert lin.open_round == 2

    def test_open_round_one_with_zero_trials(self) -> None:
        lin = classify_study(attrs_of(stored_study(), (stored_round(1, 0),)), [])
        assert lin.state == "open"
        assert lin.open_round == 1

    def test_completed(self) -> None:
        lin = classify_study(two_round_attrs(), make_trials(5))
        assert lin.state == "completed"
        assert lin.open_round is None
        assert [r.round for r in lin.rounds] == [1, 2]
        assert lin.ends == (stored_end(), stored_end())
        assert lin.last_round == stored_round(2, 3)

    def test_failed(self) -> None:
        attrs = attrs_of(
            stored_study(), (stored_round(1, 0),), {1: stored_end("failed")}
        )
        lin = classify_study(attrs, make_trials(1, state=St.FAIL, value=None))
        assert lin.state == "failed"

    def test_routes_are_merged_from_rounds(self) -> None:
        r2 = stored_round(
            2, 3, routes_added=(Route("lambda_l1", "lambda_l1", "model"),)
        )
        attrs = attrs_of(
            stored_study(), (stored_round(1, 0), r2), {1: stored_end(), 2: stored_end()}
        )
        lin = classify_study(attrs, make_trials(5))
        assert set(lin.routes) == {"learning_rate", "lambda_l1"}

    def test_other_lizyml_keys_are_ignored(self) -> None:
        attrs = two_round_attrs()
        attrs["lizyml:note"] = "x"
        attrs["lizyml:roundabout"] = 1
        attrs["unrelated"] = 1
        assert classify_study(attrs, make_trials(5)).state == "completed"

    def test_lineage_is_frozen(self) -> None:
        lin = classify_study(two_round_attrs(), make_trials(5))
        assert isinstance(lin, Lineage)
        assert isinstance(lin.rounds, tuple)
        with pytest.raises(FrozenInstanceError):
            lin.state = "open"  # type: ignore[misc]

    def test_does_not_mutate_inputs(self) -> None:
        attrs = two_round_attrs()
        snapshot = to_attr_value(attrs)
        trials = make_trials(5)
        classify_study(attrs, trials)
        assert attrs == snapshot
        assert len(trials) == 5


class TestClassifyInconsistent:
    """Every X sub-case of §7.4 plus the design's R2 extensions."""

    def _x(self, attrs: dict[str, Any], n: int) -> Lineage:
        lin = classify_study(attrs, make_trials(n))
        assert lin.state == "inconsistent", lin
        assert lin.reason
        assert lin.study is not None
        return lin

    def test_no_rounds_with_trials(self) -> None:
        self._x(attrs_of(stored_study()), 1)

    def test_round_numbers_skip(self) -> None:
        self._x(
            attrs_of(
                stored_study(),
                (stored_round(1, 0), stored_round(3, 3)),
                {1: stored_end()},
            ),
            5,
        )

    def test_round_numbers_do_not_start_at_one(self) -> None:
        self._x(attrs_of(stored_study(), (stored_round(2, 0),)), 1)

    def test_non_last_round_without_end(self) -> None:
        self._x(
            attrs_of(
                stored_study(),
                (stored_round(1, 0), stored_round(2, 3)),
                {2: stored_end()},
            ),
            5,
        )

    def test_trial_before_round_one_start(self) -> None:
        self._x(attrs_of(stored_study(), (stored_round(1, 2),)), 4)

    def test_round_one_start_beyond_trials_on_empty_study(self) -> None:
        self._x(attrs_of(stored_study(), (stored_round(1, 1),)), 0)

    # --- R2 extensions -----------------------------------------------------

    def test_orphan_end(self) -> None:
        self._x(
            attrs_of(
                stored_study(),
                (stored_round(1, 0),),
                {1: stored_end(), 2: stored_end()},
            ),
            3,
        )

    def test_orphan_end_on_empty_study(self) -> None:
        # Not N: an :end key is a round key.
        self._x(attrs_of(stored_study(), (), {1: stored_end()}), 0)

    @pytest.mark.parametrize(
        "key", ["lizyml:round:01", "lizyml:round:0", "lizyml:round:1:end:x"]
    )
    def test_malformed_round_key(self, key: str) -> None:
        attrs = two_round_attrs()
        attrs[key] = attrs[round_key(1)]
        self._x(attrs, 5)

    def test_malformed_round_key_on_empty_study_is_not_no_round(self) -> None:
        self._x(attrs_of(stored_study(), extra={"lizyml:round:abc": 1}), 0)

    def test_undecodable_round(self) -> None:
        attrs = two_round_attrs()
        attrs[round_key(2)] = {"round": 2}
        self._x(attrs, 5)

    def test_undecodable_end(self) -> None:
        attrs = two_round_attrs()
        attrs[end_key(1)] = {"status": "completed"}
        self._x(attrs, 5)

    def test_round_field_disagrees_with_key(self) -> None:
        attrs = two_round_attrs()
        attrs[round_key(2)] = to_attr_value(encode_round(stored_round(3, 3)))
        self._x(attrs, 5)

    def test_start_trial_decreasing(self) -> None:
        self._x(
            attrs_of(
                stored_study(),
                (stored_round(1, 0), stored_round(2, 3), stored_round(3, 2)),
                {1: stored_end(), 2: stored_end()},
            ),
            5,
        )

    def test_last_start_beyond_trial_count(self) -> None:
        self._x(
            attrs_of(
                stored_study(),
                (stored_round(1, 0), stored_round(2, 6)),
                {1: stored_end()},
            ),
            5,
        )

    def test_equal_start_trials_are_allowed(self) -> None:
        # A round closed with no trial (exhausted budget) leaves the next round
        # starting at the same number.
        attrs = attrs_of(
            stored_study(),
            (stored_round(1, 0), stored_round(2, 3), stored_round(3, 3)),
            {1: stored_end(), 2: stored_end(), 3: stored_end()},
        )
        assert classify_study(attrs, make_trials(5)).state == "completed"

    def test_unrouted_trial_param(self) -> None:
        lin = classify_study(
            two_round_attrs(), make_trials(5, params={"learning_rate": 0.1, "zzz": 1})
        )
        assert lin.state == "inconsistent"
        assert "zzz" in (lin.reason or "")

    def test_route_category_conflict(self) -> None:
        r2 = stored_round(
            2, 3, routes_added=(Route("learning_rate", "learning_rate", "smart"),)
        )
        self._x(
            attrs_of(
                stored_study(),
                (stored_round(1, 0), r2),
                {1: stored_end(), 2: stored_end()},
            ),
            5,
        )

    def test_x_is_checked_before_open(self) -> None:
        # Gap + last round open: X wins over O.
        self._x(
            attrs_of(
                stored_study(),
                (stored_round(1, 0), stored_round(3, 3)),
                {1: stored_end()},
            ),
            4,
        )


# ---------------------------------------------------------------------------
# Real Optuna trials satisfy the TrialLike protocol (DC7: run the real thing)
# ---------------------------------------------------------------------------


def test_classify_reads_a_real_optuna_study() -> None:
    study = optuna.create_study(direction="minimize")
    study.set_user_attr(STUDY_KEY, to_attr_value(encode_study(stored_study())))
    study.set_user_attr(round_key(1), to_attr_value(encode_round(stored_round(1, 0))))
    study.optimize(lambda t: t.suggest_float("learning_rate", 0.01, 0.3), n_trials=3)
    trial = study.ask()
    trial.suggest_float("learning_rate", 0.01, 0.3)  # stays RUNNING
    lin = classify_study(study.user_attrs, study.trials)
    assert lin.state == "open" and lin.open_round == 1
    rec = lin.rounds[0]
    assert remaining_trials(rec, trials_of_round(lin, 1, study.trials)) == 0
    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    timeout_rec = replace(rec, timeout=3600.0)
    left = remaining_timeout(timeout_rec, trials_of_round(lin, 1, study.trials))
    assert left is not None and 3500.0 < left <= 3600.0


# ---------------------------------------------------------------------------
# round_of / trials_of_round
# ---------------------------------------------------------------------------


class TestRoundOf:
    def test_round_of_by_start_trial(self) -> None:
        rounds = (
            stored_round(1, 0),
            stored_round(2, 3),
            stored_round(3, 3),
            stored_round(4, 7),
        )
        assert [round_of(rounds, n) for n in range(9)] == [1, 1, 1, 3, 3, 3, 3, 4, 4]

    def test_round_of_before_first_start_raises(self) -> None:
        with pytest.raises(ValueError):
            round_of((stored_round(1, 2),), 1)
        with pytest.raises(ValueError):
            round_of((), 0)

    def test_trials_of_round(self) -> None:
        lin = classify_study(two_round_attrs(), make_trials(5))
        trials = make_trials(5)
        assert [t.number for t in trials_of_round(lin, 1, trials)] == [0, 1, 2]
        assert [t.number for t in trials_of_round(lin, 2, trials)] == [3, 4]
        with pytest.raises(ValueError):
            trials_of_round(lin, 3, trials)


# ---------------------------------------------------------------------------
# Budget arithmetic (§7.2)
# ---------------------------------------------------------------------------


class TestRemaining:
    def test_remaining_trials_counts_finished_and_running(self) -> None:
        rec = stored_round(2, 3, n_trials=6)
        trials = [
            FakeTrial(3, St.COMPLETE),
            FakeTrial(4, St.FAIL),
            FakeTrial(5, St.PRUNED),
            FakeTrial(6, St.RUNNING, seconds=None),
            FakeTrial(7, St.WAITING, seconds=None),
        ]
        assert remaining_trials(rec, trials) == 2

    def test_remaining_trials_never_negative(self) -> None:
        assert remaining_trials(stored_round(1, 0, n_trials=1), make_trials(3)) == 0

    def test_remaining_timeout_none_without_recorded_timeout(self) -> None:
        assert remaining_timeout(stored_round(1, 0), make_trials(2)) is None

    def test_remaining_timeout_subtracts_complete_and_pruned_only(self) -> None:
        rec = stored_round(1, 0, timeout=100.0)
        trials = [
            FakeTrial(0, St.COMPLETE, seconds=10.0),
            FakeTrial(1, St.PRUNED, seconds=5.5),
            FakeTrial(2, St.FAIL, seconds=1000.0),
            FakeTrial(3, St.RUNNING, seconds=None),
        ]
        assert remaining_timeout(rec, trials) == pytest.approx(84.5)

    def test_second_crash_does_not_subtract_failed_downtime(self) -> None:
        # Criterion 16: the first reattach FAILed a RUNNING trial whose
        # datetime_complete spans the downtime (here 10 hours). A second crash
        # and reattach must not subtract it.
        rec = stored_round(2, 3, timeout=3600.0)
        failed_on_reattach = FakeTrial(4, St.FAIL, seconds=36000.0)
        trials = [FakeTrial(3, St.COMPLETE, seconds=60.0), failed_on_reattach]
        assert remaining_timeout(rec, trials) == pytest.approx(3540.0)

    def test_exhausted_timeout_is_not_positive(self) -> None:
        rec = stored_round(1, 0, timeout=15.0)
        assert remaining_timeout(rec, make_trials(2, seconds=10.0)) == pytest.approx(
            -5.0
        )

    def test_finished_trial_without_timestamps_raises(self) -> None:
        # DC1: summing only what parses would overstate the remaining time.
        rec = stored_round(1, 0, timeout=100.0)
        broken = FakeTrial(0, St.COMPLETE, seconds=None)
        with pytest.raises(ValueError):
            remaining_timeout(rec, [broken])


# ---------------------------------------------------------------------------
# Routes (§4.1b, §6.4)
# ---------------------------------------------------------------------------


class TestRoutes:
    def test_merge_routes_appends(self) -> None:
        merged = merge_routes(
            {"a": Route("a", "a", "model")}, (Route("b", "b", "smart"),)
        )
        assert merged == {"a": Route("a", "a", "model"), "b": Route("b", "b", "smart")}

    def test_merge_routes_does_not_mutate(self) -> None:
        existing = {"a": Route("a", "a", "model")}
        merge_routes(existing, (Route("b", "b", "smart"),))
        assert existing == {"a": Route("a", "a", "model")}

    def test_repeat_of_the_same_route_is_allowed(self) -> None:
        r = Route("a", "a", "model")
        assert merge_routes({"a": r}, (r,)) == {"a": r}

    def test_alias_and_canonical_name_share_a_route_category(self) -> None:
        merged = merge_routes(
            {"eta": Route("eta", "learning_rate", "model")},
            (Route("learning_rate", "learning_rate", "model"),),
        )
        assert set(merged) == {"eta", "learning_rate"}

    def test_canonical_category_conflict_raises(self) -> None:
        with pytest.raises(RouteConflict):
            merge_routes(
                {"eta": Route("eta", "learning_rate", "model")},
                (Route("learning_rate", "learning_rate", "smart"),),
            )

    def test_same_name_different_category_raises(self) -> None:
        with pytest.raises(RouteConflict):
            merge_routes(
                {"a": Route("a", "a", "model")}, (Route("a", "a", "training"),)
            )

    def test_same_name_different_canonical_raises(self) -> None:
        with pytest.raises(RouteConflict):
            merge_routes(
                {"eta": Route("eta", "learning_rate", "model")},
                (Route("eta", "eta", "model"),),
            )

    def test_split_by_routes(self) -> None:
        routes = {
            "eta": Route("eta", "learning_rate", "model"),
            "num_leaves_ratio": Route("num_leaves_ratio", "num_leaves_ratio", "smart"),
            "early_stopping_rounds": Route(
                "early_stopping_rounds", "early_stopping_rounds", "training"
            ),
        }
        model, smart, training = split_by_routes(
            {"eta": 0.1, "num_leaves_ratio": 0.5, "early_stopping_rounds": 20}, routes
        )
        assert model == {"eta": 0.1}
        assert smart == {"num_leaves_ratio": 0.5}
        assert training == {"early_stopping_rounds": 20}

    def test_removed_smart_dim_still_routes_to_smart(self) -> None:
        # Criterion 6 (PR 1 half): the round that declared the smart dim
        # keeps its route after a later round no longer declares it.
        # Round 1's trials hold the dim and round 2's do not (H-0124: each
        # trial's names are within its own round's space).
        ratio = FloatDim("num_leaves_ratio", 0.5, 1.0, category="smart")
        r1 = stored_round(
            1,
            0,
            effective=(LR_BOUNDED, ratio),
            routes_added=(
                Route("learning_rate", "learning_rate", "model"),
                Route("num_leaves_ratio", "num_leaves_ratio", "smart"),
            ),
        )
        attrs = attrs_of(
            stored_study(), (r1, stored_round(2, 3)), {1: stored_end(), 2: stored_end()}
        )
        best = {"learning_rate": 0.1, "num_leaves_ratio": 0.7}
        trials = make_trials(3, params=best) + [
            FakeTrial(n, params={"learning_rate": 0.2}) for n in (3, 4)
        ]
        lin = classify_study(attrs, trials)
        assert lin.state == "completed"
        model, smart, _ = split_by_routes(best, lin.routes)
        assert smart == {"num_leaves_ratio": 0.7}
        assert "num_leaves_ratio" not in model

    def test_removed_training_dim_still_routes_to_training(self) -> None:
        # Criterion 6b (PR 1 half).
        rounds_dim = IntDim("early_stopping_rounds", 10, 50, category="training")
        r1 = stored_round(
            1,
            0,
            effective=(LR_BOUNDED, rounds_dim),
            routes_added=(
                Route("learning_rate", "learning_rate", "model"),
                Route("early_stopping_rounds", "early_stopping_rounds", "training"),
            ),
        )
        attrs = attrs_of(
            stored_study(), (r1, stored_round(2, 3)), {1: stored_end(), 2: stored_end()}
        )
        best = {"learning_rate": 0.1, "early_stopping_rounds": 30}
        trials = make_trials(3, params=best) + [
            FakeTrial(n, params={"learning_rate": 0.2}) for n in (3, 4)
        ]
        lin = classify_study(attrs, trials)
        assert lin.state == "completed"
        model, _, training = split_by_routes(best, lin.routes)
        assert training == {"early_stopping_rounds": 30}
        assert "early_stopping_rounds" not in model

    def test_unrouted_param_raises(self) -> None:
        with pytest.raises(UnroutedParam):
            split_by_routes({"x": 1}, {})

    def test_errors_are_value_errors(self) -> None:
        assert issubclass(RouteConflict, ValueError)
        assert issubclass(UnroutedParam, ValueError)


def test_nan_best_values_do_not_break_classification() -> None:
    attrs = attrs_of(
        stored_study(),
        (stored_round(1, 0),),
        {1: StoredRoundEnd("completed", math.inf, "t")},
    )
    assert classify_study(attrs, make_trials(3)).state == "completed"
