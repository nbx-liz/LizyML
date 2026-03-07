"""Tests for TuningResult type and Model.tuning_table() (H-0023)."""

from __future__ import annotations

import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TrialResult, TuningResult
from tests._helpers import make_config, make_regression_df

# ---------------------------------------------------------------------------
# Unit tests — TuningResult / TrialResult dataclasses
# ---------------------------------------------------------------------------


class TestTuningResultDataclass:
    def test_trial_result_fields(self) -> None:
        tr = TrialResult(number=0, params={"lr": 0.1}, score=1.5, state="complete")
        assert tr.number == 0
        assert tr.params == {"lr": 0.1}
        assert tr.score == 1.5
        assert tr.state == "complete"

    def test_trial_result_frozen(self) -> None:
        tr = TrialResult(number=0, params={}, score=0.0, state="complete")
        with pytest.raises(AttributeError):
            tr.score = 999.0  # type: ignore[misc]

    def test_tuning_result_fields(self) -> None:
        trials = [
            TrialResult(number=0, params={"a": 1}, score=0.5, state="complete"),
            TrialResult(number=1, params={"a": 2}, score=0.3, state="complete"),
        ]
        result = TuningResult(
            best_params={"a": 2},
            best_score=0.3,
            trials=trials,
            metric_name="rmse",
            direction="minimize",
        )
        assert result.best_params == {"a": 2}
        assert result.best_score == 0.3
        assert len(result.trials) == 2
        assert result.metric_name == "rmse"
        assert result.direction == "minimize"

    def test_tuning_result_frozen(self) -> None:
        result = TuningResult(
            best_params={},
            best_score=0.0,
            trials=[],
            metric_name="rmse",
            direction="minimize",
        )
        with pytest.raises(AttributeError):
            result.best_score = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration tests — Model.tuning_table()
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


class TestTuningTable:
    def test_tuning_table_before_tune_raises(self) -> None:
        m = Model(_reg_config_with_tuning())
        with pytest.raises(LizyMLError) as exc_info:
            m.tuning_table()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_tuning_table_columns(self) -> None:
        m = Model(_reg_config_with_tuning(n_trials=3))
        m.tune(data=make_regression_df(n=100))
        table = m.tuning_table()
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 3
        assert "trial" in table.columns
        assert "num_leaves" in table.columns
        assert "learning_rate" in table.columns
        # metric column name should match the tuning metric
        assert m._tuning_result is not None
        assert m._tuning_result.metric_name in table.columns

    def test_tuning_table_trial_numbers(self) -> None:
        m = Model(_reg_config_with_tuning(n_trials=2))
        m.tune(data=make_regression_df(n=100))
        table = m.tuning_table()
        assert list(table["trial"]) == [0, 1]
