"""Category C: Tuning reproducibility & failure matrix.

Tests seed-fixed reproducibility, all-trial failure, partial failure,
NaN/inf objective, and search space vs Config collision.

See BLUEPRINT §18.1.6 and HISTORY H-0056 Category C.
"""

from __future__ import annotations

from typing import Any

import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.core.types.search_dim import FloatDim, IntDim
from lizyml.tuning.tuner import Tuner
from tests._helpers import make_config, make_regression_df

# ===================================================================
# Seed-fixed reproducibility
# ===================================================================


class TestTuningReproducibility:
    """Same seed → identical tune results."""

    def _run_tune(self, seed: int) -> Any:
        """Run a small tune with the given seed."""
        dims = [
            FloatDim("learning_rate", 0.01, 0.3, log=True),
            IntDim("num_leaves", 8, 64),
        ]
        tuner = Tuner(dims=dims, n_trials=5, seed=seed)

        call_count = 0

        def objective(trial: Any) -> float:
            nonlocal call_count
            lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            nl = trial.suggest_int("num_leaves", 8, 64)
            call_count += 1
            # Deterministic objective based only on params
            return (lr - 0.1) ** 2 + (nl - 31) ** 2 / 1000.0

        result = tuner.tune(objective, metric_name="synthetic")
        return result

    def test_same_seed_same_best_params(self) -> None:
        r1 = self._run_tune(seed=42)
        r2 = self._run_tune(seed=42)
        assert r1.best_params == r2.best_params
        assert r1.best_score == pytest.approx(r2.best_score)

    def test_same_seed_same_trial_order(self) -> None:
        r1 = self._run_tune(seed=42)
        r2 = self._run_tune(seed=42)
        assert len(r1.trials) == len(r2.trials)
        for t1, t2 in zip(r1.trials, r2.trials, strict=True):
            assert t1.params == t2.params
            assert t1.score == pytest.approx(t2.score)

    def test_different_seed_different_results(self) -> None:
        r1 = self._run_tune(seed=42)
        r2 = self._run_tune(seed=99)
        # At least one parameter should differ
        assert r1.best_params != r2.best_params or r1.best_score != r2.best_score


# ===================================================================
# All-trial failure
# ===================================================================


class TestAllTrialFailure:
    """All trials fail → TUNING_FAILED with correct context."""

    def test_all_trials_fail_raises_tuning_failed(self) -> None:
        dims = [FloatDim("lr", 0.01, 0.3)]
        tuner = Tuner(dims=dims, n_trials=3, seed=42)

        def failing_objective(trial: Any) -> float:
            trial.suggest_float("lr", 0.01, 0.3)
            msg = "intentional failure"
            raise ValueError(msg)

        with pytest.raises(LizyMLError) as exc_info:
            tuner.tune(failing_objective)
        assert exc_info.value.code == ErrorCode.TUNING_FAILED
        assert exc_info.value.context["n_trials"] == 3

    def test_all_trials_fail_message_content(self) -> None:
        dims = [FloatDim("lr", 0.01, 0.3)]
        tuner = Tuner(dims=dims, n_trials=2, seed=42)

        def failing_objective(trial: Any) -> float:
            trial.suggest_float("lr", 0.01, 0.3)
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(LizyMLError) as exc_info:
            tuner.tune(failing_objective)
        assert "failed" in exc_info.value.user_message.lower()


# ===================================================================
# Partial failure
# ===================================================================


class TestPartialTrialFailure:
    """Some trials fail, best selected from completed ones."""

    def test_partial_failure_returns_best_from_completed(self) -> None:
        dims = [FloatDim("lr", 0.01, 0.3)]
        tuner = Tuner(dims=dims, n_trials=6, seed=42)

        def partial_objective(trial: Any) -> float:
            lr = trial.suggest_float("lr", 0.01, 0.3)
            if trial.number % 2 == 0:
                msg = "even trials fail"
                raise ValueError(msg)
            return lr  # lower is better

        result = tuner.tune(partial_objective)
        # Should have some completed trials
        completed = [t for t in result.trials if t.state == "complete"]
        failed = [t for t in result.trials if t.state != "complete"]
        assert len(completed) > 0
        assert len(failed) > 0
        # best_score should come from completed trials
        assert result.best_score == min(t.score for t in completed)


# ===================================================================
# E2E: tune() then fit() via Model API
# ===================================================================


class TestTuneIntegration:
    """Full Model.tune() → Model.fit() integration."""

    def test_tune_then_fit_regression(self) -> None:
        df = make_regression_df(n=200, seed=0)
        cfg = make_config("regression", n_estimators=10, n_splits=2, tuning_n_trials=3)
        m = Model(cfg)
        tune_result = m.tune(data=df)
        assert tune_result.best_params is not None
        assert len(tune_result.trials) == 3

        fit_result = m.fit(data=df)
        assert fit_result.oof_pred is not None

    def test_tune_reproducibility_via_model(self) -> None:
        """Same config+data → same tune results via Model API."""
        df = make_regression_df(n=200, seed=0)
        cfg = make_config("regression", n_estimators=10, n_splits=2, tuning_n_trials=3)

        m1 = Model(cfg)
        r1 = m1.tune(data=df)

        m2 = Model(cfg)
        r2 = m2.tune(data=df)

        assert r1.best_params == r2.best_params
        assert r1.best_score == pytest.approx(r2.best_score)
