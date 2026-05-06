"""Tests for H-0072: Optuna persistent storage for resumable tuning.

Covers:
- INV-1: storage=None preserves in-memory behavior (no disk IO)
- INV-2: storage=<url> persists trial state to disk
- INV-3: re-attach with same storage+study_name resumes (load_if_exists=True)
- INV-4: storage without study_name raises CONFIG_INVALID
- INV-5: separate study_names do not mix in same storage
- INV-6: progress_callback behaves identically with/without storage

Also covers:
- crash-and-resume: trial mid-failure followed by resume from a fresh Tuner
- Model.tune() pass-through: storage / study_name forwarded to Tuner
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TuneProgressInfo, TuningResult
from lizyml.tuning.search_space import parse_space
from lizyml.tuning.tuner import Tuner
from tests._helpers import make_config, make_regression_df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trivial_dims() -> list[Any]:
    """Single FloatDim search space with no model coupling."""
    return parse_space({"x": {"type": "float", "low": 0.0, "high": 1.0}})


def _quadratic_objective() -> Any:
    """Deterministic objective: y = (x - 0.3)^2; min at x=0.3."""

    def objective(trial: Any) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        return float((x - 0.3) ** 2)

    return objective


def _reg_config_with_tuning(n_trials: int = 3) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# INV-1: storage=None preserves in-memory behavior
# ---------------------------------------------------------------------------


class TestInMemoryDefault:
    def test_storage_none_uses_in_memory_storage(self) -> None:
        tuner = Tuner(dims=_trivial_dims(), n_trials=3, seed=0)
        _, study = tuner.tune(_quadratic_objective())
        assert isinstance(study._storage, optuna.storages.InMemoryStorage)

    def test_storage_none_makes_no_disk_files(self, tmp_path: Path) -> None:
        """No files should be created under tmp_path when storage=None."""
        tuner = Tuner(dims=_trivial_dims(), n_trials=2, seed=0)
        _, _ = tuner.tune(_quadratic_objective())
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# INV-2 + INV-3: persistence + resume via SQLite
# ---------------------------------------------------------------------------


class TestSQLitePersistence:
    def test_trials_persisted_to_sqlite(self, tmp_path: Path) -> None:
        url = f"sqlite:///{tmp_path / 'study.db'}"
        tuner = Tuner(
            dims=_trivial_dims(),
            n_trials=4,
            seed=0,
            storage=url,
            study_name="t1",
        )
        _, study = tuner.tune(_quadratic_objective())
        assert (tmp_path / "study.db").exists()
        assert len(study.trials) == 4

    def test_resume_via_same_storage_and_study_name(self, tmp_path: Path) -> None:
        """Resume: same (storage, study_name) accumulates trials across Tuners."""
        url = f"sqlite:///{tmp_path / 'study.db'}"
        tuner1 = Tuner(
            dims=_trivial_dims(),
            n_trials=3,
            seed=0,
            storage=url,
            study_name="resumable",
        )
        _, study1 = tuner1.tune(_quadratic_objective())
        assert len(study1.trials) == 3

        # Fresh Tuner instance, same backing store
        tuner2 = Tuner(
            dims=_trivial_dims(),
            n_trials=2,
            seed=1,
            storage=url,
            study_name="resumable",
        )
        _, study2 = tuner2.tune(_quadratic_objective())
        # 3 prior + 2 new = 5
        assert len(study2.trials) == 5

    def test_separate_study_names_do_not_mix(self, tmp_path: Path) -> None:
        """Two study_names in same storage maintain separate trial sets (INV-5)."""
        url = f"sqlite:///{tmp_path / 'study.db'}"
        Tuner(
            dims=_trivial_dims(),
            n_trials=3,
            seed=0,
            storage=url,
            study_name="alpha",
        ).tune(_quadratic_objective())
        Tuner(
            dims=_trivial_dims(),
            n_trials=2,
            seed=0,
            storage=url,
            study_name="beta",
        ).tune(_quadratic_objective())

        alpha = optuna.load_study(study_name="alpha", storage=url)
        beta = optuna.load_study(study_name="beta", storage=url)
        assert len(alpha.trials) == 3
        assert len(beta.trials) == 2


# ---------------------------------------------------------------------------
# Crash-and-resume scenario (Studio-style recovery)
# ---------------------------------------------------------------------------


class TestCrashAndResume:
    def test_resume_after_objective_raises_mid_study(self, tmp_path: Path) -> None:
        """First Tuner crashes inside Tuner.tune (catch=tuple does NOT include
        BaseException, but our objective raises ValueError which is caught).
        After recovery, a fresh Tuner with same storage+study_name continues."""
        url = f"sqlite:///{tmp_path / 'study.db'}"

        call_count = {"n": 0}

        def flaky_objective(trial: Any) -> float:
            call_count["n"] += 1
            x = trial.suggest_float("x", 0.0, 1.0)
            # Trial 1 (0-indexed) raises ValueError — caught by Tuner via
            # `catch=(LizyMLError, ValueError, RuntimeError)` and recorded as FAIL
            if call_count["n"] == 2:
                raise ValueError("simulated transient failure")
            return float((x - 0.3) ** 2)

        tuner1 = Tuner(
            dims=_trivial_dims(),
            n_trials=3,
            seed=0,
            storage=url,
            study_name="job-42",
        )
        _, study1 = tuner1.tune(flaky_objective)
        # 3 trials recorded (one FAIL + two COMPLETE), persisted to disk
        assert len(study1.trials) == 3
        states = {t.state.name for t in study1.trials}
        assert "FAIL" in states

        # Fresh Tuner picks up from storage and runs additional trials
        tuner2 = Tuner(
            dims=_trivial_dims(),
            n_trials=2,
            seed=1,
            storage=url,
            study_name="job-42",
        )
        _, study2 = tuner2.tune(_quadratic_objective())
        assert len(study2.trials) == 5
        # At least one COMPLETE trial must exist for best_value
        completed = [
            t for t in study2.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        assert len(completed) >= 1


# ---------------------------------------------------------------------------
# INV-4: fail fast when storage is given without study_name
# ---------------------------------------------------------------------------


class TestFailFastValidation:
    def test_tuner_storage_without_study_name_raises(self, tmp_path: Path) -> None:
        url = f"sqlite:///{tmp_path / 'x.db'}"
        with pytest.raises(LizyMLError) as exc_info:
            Tuner(dims=_trivial_dims(), n_trials=1, storage=url, study_name=None)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID
        assert "study_name" in exc_info.value.user_message

    def test_model_tune_storage_without_study_name_raises(self, tmp_path: Path) -> None:
        cfg = _reg_config_with_tuning(n_trials=2)
        model = Model(cfg)
        url = f"sqlite:///{tmp_path / 'm.db'}"
        with pytest.raises(LizyMLError) as exc_info:
            model.tune(make_regression_df(n=100), storage=url, study_name=None)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID


# ---------------------------------------------------------------------------
# Model.tune() pass-through
# ---------------------------------------------------------------------------


class TestModelTunePassThrough:
    def test_model_tune_persists_via_sqlite(self, tmp_path: Path) -> None:
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df(n=120)
        model = Model(cfg)
        url = f"sqlite:///{tmp_path / 'mtune.db'}"

        result = model.tune(df, storage=url, study_name="mt")
        assert isinstance(result, TuningResult)
        assert (tmp_path / "mtune.db").exists()

        # Reload from disk via raw optuna and confirm trial count matches
        study = optuna.load_study(study_name="mt", storage=url)
        assert len(study.trials) == 2

    def test_model_tune_load_if_exists_resumes_across_instances(
        self, tmp_path: Path
    ) -> None:
        """Two separate Model instances pointing at the same journal file resume."""
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df(n=120)
        url = f"sqlite:///{tmp_path / 'shared.db'}"

        m1 = Model(cfg)
        m1.tune(df, storage=url, study_name="shared-job")

        # New process simulation: brand-new Model instance, no _study held
        m2 = Model(cfg)
        m2.tune(df, storage=url, study_name="shared-job")

        # 2 prior trials in journal + 2 new = 4 total
        study = optuna.load_study(study_name="shared-job", storage=url)
        assert len(study.trials) == 4


# ---------------------------------------------------------------------------
# INV-6: progress_callback behaves identically with/without storage
# ---------------------------------------------------------------------------


class TestProgressCallbackParity:
    def test_progress_callback_fires_with_storage(self, tmp_path: Path) -> None:
        url = f"sqlite:///{tmp_path / 'p.db'}"
        events: list[TuneProgressInfo] = []

        def cb(info: TuneProgressInfo) -> None:
            events.append(info)

        tuner = Tuner(
            dims=_trivial_dims(),
            n_trials=3,
            seed=0,
            progress_callback=cb,
            storage=url,
            study_name="p1",
        )
        tuner.tune(_quadratic_objective())
        assert len(events) == 3
        # current_trial counts up 1..3 within the round
        assert [e.current_trial for e in events] == [1, 2, 3]
        # total_trials reflects n_trials passed to Tuner
        assert all(e.total_trials == 3 for e in events)
