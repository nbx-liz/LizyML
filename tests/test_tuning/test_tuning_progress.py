"""Tests for TuneProgressInfo / TuneProgressCallback (H-0048).

Covers:
- TuneProgressInfo is frozen and has correct fields
- progress_callback is invoked for each trial
- current_trial increments from 1 to n_trials
- elapsed_seconds is non-negative
- best_score is populated after first complete trial
- callback=None does not break tuning
- callback exceptions do not abort tuning
"""

from __future__ import annotations

import pytest

from lizyml import Model
from lizyml.core.types.tuning_result import TuneProgressCallback, TuneProgressInfo
from tests._helpers import make_config, make_regression_df

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestTuneProgressInfo:
    def test_frozen(self) -> None:
        info = TuneProgressInfo(
            current_trial=1,
            total_trials=10,
            elapsed_seconds=0.5,
            best_score=None,
            latest_score=0.8,
            latest_state="complete",
        )
        with pytest.raises(AttributeError):
            info.current_trial = 2  # type: ignore[misc]

    def test_fields(self) -> None:
        info = TuneProgressInfo(
            current_trial=3,
            total_trials=10,
            elapsed_seconds=1.5,
            best_score=0.7,
            latest_score=0.8,
            latest_state="complete",
        )
        assert info.current_trial == 3
        assert info.total_trials == 10
        assert info.elapsed_seconds == 1.5
        assert info.best_score == 0.7
        assert info.latest_score == 0.8
        assert info.latest_state == "complete"

    def test_none_scores(self) -> None:
        info = TuneProgressInfo(
            current_trial=1,
            total_trials=5,
            elapsed_seconds=0.1,
            best_score=None,
            latest_score=None,
            latest_state="fail",
        )
        assert info.best_score is None
        assert info.latest_score is None
        assert info.latest_state == "fail"


class TestTuneProgressCallback:
    def test_callback_type_alias(self) -> None:
        """TuneProgressCallback should be a callable type alias."""
        assert TuneProgressCallback is not None


class TestPublicImport:
    def test_import_from_lizyml(self) -> None:
        """TuneProgressInfo and TuneProgressCallback should be importable."""
        from lizyml import TuneProgressCallback as TPC
        from lizyml import TuneProgressInfo as TPI

        assert TPI is TuneProgressInfo
        assert TPC is TuneProgressCallback


# ---------------------------------------------------------------------------
# Integration tests (29-F DoD)
# ---------------------------------------------------------------------------


def _reg_config_with_tuning(n_trials: int = 3) -> dict:
    cfg = make_config("regression")
    cfg["tuning"] = {
        "optuna": {
            "params": {"n_trials": n_trials, "direction": "minimize"},
            "space": {
                "num_leaves": {"type": "int", "low": 8, "high": 32},
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log": True,
                },
            },
        }
    }
    return cfg


class TestProgressCallbackIntegration:
    """Integration tests for progress_callback via Model.tune()."""

    def test_callback_called_n_times(self) -> None:
        """Callback should be invoked once per trial."""
        n_trials = 3
        received: list[TuneProgressInfo] = []
        df = make_regression_df(n=200)
        cfg = _reg_config_with_tuning(n_trials=n_trials)
        model = Model(cfg, data=df)
        model.tune(progress_callback=received.append)
        assert len(received) == n_trials

    def test_current_trial_increments(self) -> None:
        """current_trial should be 1-indexed and increment sequentially."""
        n_trials = 3
        received: list[TuneProgressInfo] = []
        df = make_regression_df(n=200)
        cfg = _reg_config_with_tuning(n_trials=n_trials)
        model = Model(cfg, data=df)
        model.tune(progress_callback=received.append)
        trial_nums = [info.current_trial for info in received]
        assert trial_nums == [1, 2, 3]

    def test_total_trials_consistent(self) -> None:
        """total_trials should equal n_trials for all invocations."""
        n_trials = 3
        received: list[TuneProgressInfo] = []
        df = make_regression_df(n=200)
        cfg = _reg_config_with_tuning(n_trials=n_trials)
        model = Model(cfg, data=df)
        model.tune(progress_callback=received.append)
        for info in received:
            assert info.total_trials == n_trials

    def test_elapsed_seconds_non_negative(self) -> None:
        """elapsed_seconds should be >= 0 for all invocations."""
        received: list[TuneProgressInfo] = []
        df = make_regression_df(n=200)
        cfg = _reg_config_with_tuning(n_trials=3)
        model = Model(cfg, data=df)
        model.tune(progress_callback=received.append)
        for info in received:
            assert info.elapsed_seconds >= 0.0

    def test_best_score_populated_after_complete(self) -> None:
        """best_score should be non-None after at least one complete trial."""
        received: list[TuneProgressInfo] = []
        df = make_regression_df(n=200)
        cfg = _reg_config_with_tuning(n_trials=3)
        model = Model(cfg, data=df)
        model.tune(progress_callback=received.append)
        # Find first complete trial
        complete_infos = [info for info in received if info.latest_state == "complete"]
        assert len(complete_infos) > 0, "Expected at least one complete trial"
        # All infos after the first complete should have best_score set
        first_complete_idx = received.index(complete_infos[0])
        for info in received[first_complete_idx:]:
            if info.latest_state == "complete":
                assert info.best_score is not None

    def test_latest_state_valid_values(self) -> None:
        """latest_state should be one of 'complete', 'pruned', 'fail'."""
        received: list[TuneProgressInfo] = []
        df = make_regression_df(n=200)
        cfg = _reg_config_with_tuning(n_trials=3)
        model = Model(cfg, data=df)
        model.tune(progress_callback=received.append)
        valid_states = {"complete", "pruned", "fail"}
        for info in received:
            assert info.latest_state in valid_states

    def test_none_callback_does_not_break(self) -> None:
        """tune() should work normally when progress_callback is None."""
        df = make_regression_df(n=200)
        cfg = _reg_config_with_tuning(n_trials=2)
        model = Model(cfg, data=df)
        result = model.tune(progress_callback=None)
        assert result.best_params is not None
        assert len(result.trials) == 2

    def test_callback_exception_does_not_abort(self) -> None:
        """Exceptions in callback should be caught, tuning continues."""

        call_count = 0

        def bad_callback(info: TuneProgressInfo) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("intentional error in callback")

        df = make_regression_df(n=200)
        cfg = _reg_config_with_tuning(n_trials=3)
        model = Model(cfg, data=df)

        with pytest.warns(RuntimeWarning, match="progress_callback raised"):
            result = model.tune(progress_callback=bad_callback)

        assert call_count == 3
        assert result.best_params is not None
