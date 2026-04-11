"""Tests for H-0068: Re-tune (Study Resume + Boundary Expansion).

Covers:
- Boundary detection: linear, log, categorical, edge cases
- Dimension expansion: asymmetric expansion, IntDim guards
- Model.tune(resume=True): study resume, trial accumulation, round tracking
- Model.tune(resume=True) without prior tune() → error
- TuningResult.rounds populated correctly
- TuneProgressInfo.round / cumulative_trials / expanded_dims
- TrialResult.round field
- tuning_table() with round/state columns
- boundary_table() output
- expand_boundary=None defaults (default space vs user space)
"""

from __future__ import annotations

from typing import Any

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.core.types.tuning_result import (
    BoundaryReport,
    RoundSummary,
    TrialResult,
    TuneProgressInfo,
    TuningResult,
)
from lizyml.tuning.search_space import detect_boundary, expand_dims
from tests._helpers import make_config, make_regression_df

# ---------------------------------------------------------------------------
# Boundary detection unit tests
# ---------------------------------------------------------------------------


class TestDetectBoundary:
    """Test detect_boundary() for various dimension types."""

    def test_float_dim_lower_edge(self) -> None:
        dims = [FloatDim("lr", low=0.001, high=0.1, log=False)]
        report = detect_boundary(dims, {"lr": 0.002}, threshold=0.05)
        assert len(report.dims) == 1
        s = report.dims[0]
        assert s.name == "lr"
        assert s.edge == "lower"
        assert s.expanded is True
        assert "lr" in report.expanded_names

    def test_float_dim_upper_edge(self) -> None:
        dims = [FloatDim("ff", low=0.5, high=1.0)]
        report = detect_boundary(dims, {"ff": 0.98}, threshold=0.05)
        s = report.dims[0]
        assert s.edge == "upper"
        assert s.expanded is True

    def test_float_dim_center_no_expand(self) -> None:
        dims = [FloatDim("lr", low=0.001, high=0.1)]
        report = detect_boundary(dims, {"lr": 0.05}, threshold=0.05)
        s = report.dims[0]
        assert s.edge == "none"
        assert s.expanded is False
        assert report.expanded_names == ()

    def test_log_dim_lower_edge(self) -> None:
        dims = [FloatDim("lr", low=0.0001, high=0.1, log=True)]
        # log(0.0001)=-9.21, log(0.1)=-2.30, span=6.91
        # log(0.00012)=-9.03, position=(9.21-9.03)/6.91=0.026 < 0.05
        report = detect_boundary(dims, {"lr": 0.00012}, threshold=0.05)
        s = report.dims[0]
        assert s.edge == "lower"
        assert s.expanded is True

    def test_log_dim_center_no_expand(self) -> None:
        dims = [FloatDim("lr", low=0.0001, high=0.1, log=True)]
        report = detect_boundary(dims, {"lr": 0.003}, threshold=0.05)
        s = report.dims[0]
        assert s.edge == "none"
        assert s.expanded is False

    def test_int_dim_upper_edge(self) -> None:
        dims = [IntDim("leaves", low=16, high=256)]
        report = detect_boundary(dims, {"leaves": 254}, threshold=0.05)
        s = report.dims[0]
        assert s.edge == "upper"
        assert s.expanded is True
        assert isinstance(s.new_high, int)
        assert s.new_high > 256

    def test_int_dim_lower_guard(self) -> None:
        """IntDim lower bound cannot go below 1."""
        dims = [IntDim("x", low=1, high=10)]
        report = detect_boundary(dims, {"x": 1}, threshold=0.05)
        s = report.dims[0]
        assert s.edge == "lower"
        assert s.expanded is True
        assert s.new_low is not None
        assert s.new_low >= 1  # Guard: min(1, ...)

    def test_categorical_no_expand(self) -> None:
        dims = [CategoricalDim("obj", choices=("huber", "fair"))]
        report = detect_boundary(dims, {"obj": "huber"}, threshold=0.05)
        s = report.dims[0]
        assert s.edge == "none"
        assert s.expanded is False
        assert s.position_pct is None

    def test_multiple_dims_partial_expand(self) -> None:
        dims = [
            FloatDim("lr", low=0.001, high=0.1),  # will be near lower
            IntDim("leaves", low=16, high=256),  # will be in center
            CategoricalDim("obj", choices=("a", "b")),  # categorical
        ]
        report = detect_boundary(
            dims, {"lr": 0.002, "leaves": 128, "obj": "a"}, threshold=0.05
        )
        assert report.expanded_names == ("lr",)
        assert len(report.dims) == 3

    def test_missing_param_uses_center(self) -> None:
        """Missing param key defaults to center → no expansion."""
        dims = [FloatDim("lr", low=0.001, high=0.1)]
        report = detect_boundary(dims, {}, threshold=0.05)
        s = report.dims[0]
        assert s.edge == "none"
        assert s.expanded is False


# ---------------------------------------------------------------------------
# Expand dims unit tests
# ---------------------------------------------------------------------------


class TestExpandDims:
    def test_expand_lower_linear(self) -> None:
        dims = [FloatDim("lr", low=0.001, high=0.1)]
        report = detect_boundary(dims, {"lr": 0.002}, threshold=0.05)
        new = expand_dims(dims, report)
        assert len(new) == 1
        assert isinstance(new[0], FloatDim)
        assert new[0].low < 0.001  # expanded downward
        assert new[0].high == 0.1  # upper unchanged

    def test_expand_upper_linear(self) -> None:
        dims = [IntDim("leaves", low=16, high=256)]
        report = detect_boundary(dims, {"leaves": 254}, threshold=0.05)
        new = expand_dims(dims, report)
        assert isinstance(new[0], IntDim)
        assert new[0].low == 16  # lower unchanged
        assert new[0].high > 256  # expanded upward

    def test_expand_log_lower(self) -> None:
        dims = [FloatDim("lr", low=0.0001, high=0.1, log=True)]
        report = detect_boundary(dims, {"lr": 0.00012}, threshold=0.05)
        new = expand_dims(dims, report)
        assert new[0].low < 0.0001
        assert new[0].log is True  # log flag preserved
        assert new[0].category == "model"  # category preserved

    def test_no_expand_center(self) -> None:
        dims = [FloatDim("lr", low=0.001, high=0.1)]
        report = detect_boundary(dims, {"lr": 0.05}, threshold=0.05)
        new = expand_dims(dims, report)
        assert new[0].low == 0.001
        assert new[0].high == 0.1

    def test_categorical_unchanged(self) -> None:
        dims = [CategoricalDim("obj", choices=("a", "b"))]
        report = detect_boundary(dims, {"obj": "a"}, threshold=0.05)
        new = expand_dims(dims, report)
        assert new[0] is dims[0]  # same object, untouched


# ---------------------------------------------------------------------------
# Type extension tests
# ---------------------------------------------------------------------------


class TestTrialResultRound:
    def test_default_round_is_1(self) -> None:
        t = TrialResult(number=0, params={"x": 1}, score=0.5, state="complete")
        assert t.round == 1

    def test_explicit_round(self) -> None:
        t = TrialResult(number=5, params={}, score=0.3, state="complete", round=2)
        assert t.round == 2


class TestTuningResultExtensions:
    def test_rounds_default_empty(self) -> None:
        r = TuningResult(
            best_model_params={},
            best_smart_params={},
            best_training_params={},
            best_score=0.5,
            trials=[],
            metric_name="rmse",
            direction="minimize",
        )
        assert r.rounds == ()
        assert r.boundary_report is None

    def test_rounds_populated(self) -> None:
        rs = RoundSummary(
            round=1,
            n_trials=10,
            best_score_before=None,
            best_score_after=0.5,
            expanded_dims=(),
            space_snapshot=(),
        )
        r = TuningResult(
            best_model_params={},
            best_smart_params={},
            best_training_params={},
            best_score=0.5,
            trials=[],
            metric_name="rmse",
            direction="minimize",
            rounds=(rs,),
        )
        assert len(r.rounds) == 1
        assert r.rounds[0].round == 1

    def test_backward_compat_best_params(self) -> None:
        r = TuningResult(
            best_model_params={"a": 1},
            best_smart_params={"b": 2},
            best_training_params={"c": 3},
            best_score=0.5,
            trials=[],
            metric_name="rmse",
            direction="minimize",
        )
        assert r.best_params == {"a": 1, "b": 2, "c": 3}


class TestTuneProgressInfoExtensions:
    def test_default_values(self) -> None:
        info = TuneProgressInfo(
            current_trial=1,
            total_trials=10,
            elapsed_seconds=0.5,
            best_score=None,
            latest_score=None,
            latest_state="complete",
        )
        assert info.round == 1
        assert info.cumulative_trials == 0
        assert info.expanded_dims == ()

    def test_h0068_fields(self) -> None:
        info = TuneProgressInfo(
            current_trial=5,
            total_trials=30,
            elapsed_seconds=10.0,
            best_score=0.3,
            latest_score=0.31,
            latest_state="complete",
            round=2,
            cumulative_trials=55,
            expanded_dims=("lr", "leaves"),
        )
        assert info.round == 2
        assert info.cumulative_trials == 55
        assert info.expanded_dims == ("lr", "leaves")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


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
# Model.tune(resume=True) integration tests
# ---------------------------------------------------------------------------


class TestModelRetuneResume:
    """Integration tests for tune(resume=True)."""

    def test_resume_without_prior_tune_raises(self) -> None:
        cfg = _reg_config_with_tuning(n_trials=2)
        model = Model(cfg)
        with pytest.raises(LizyMLError) as exc_info:
            model.tune(make_regression_df(), resume=True)
        assert exc_info.value.code == ErrorCode.TUNING_FAILED
        assert "resume" in str(exc_info.value.user_message).lower()

    def test_resume_accumulates_trials(self) -> None:
        cfg = _reg_config_with_tuning(n_trials=3)
        df = make_regression_df()
        model = Model(cfg)

        r1 = model.tune(df)
        assert len(r1.trials) == 3
        assert len(r1.rounds) == 1
        assert r1.rounds[0].round == 1

        r2 = model.tune(df, resume=True, n_trials=2)
        # 3 original + 1 enqueued + 2 new = 6 total,
        # but enqueued counts as a trial in study
        assert len(r2.trials) >= 5
        assert len(r2.rounds) == 2
        assert r2.rounds[1].round == 2

    def test_resume_best_score_monotonic(self) -> None:
        """Best score should not get worse across rounds."""
        cfg = _reg_config_with_tuning(n_trials=3)
        df = make_regression_df()
        model = Model(cfg)

        r1 = model.tune(df)
        r2 = model.tune(df, resume=True, n_trials=3)
        # minimize → lower or equal
        assert r2.best_score <= r1.best_score + 1e-10

    def test_resume_trial_round_numbers(self) -> None:
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df()
        model = Model(cfg)

        model.tune(df)
        r2 = model.tune(df, resume=True, n_trials=2)

        round_1_trials = [t for t in r2.trials if t.round == 1]
        round_2_trials = [t for t in r2.trials if t.round == 2]
        assert len(round_1_trials) >= 2
        assert len(round_2_trials) >= 2

    def test_resume_progress_callback(self) -> None:
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df()
        model = Model(cfg)

        model.tune(df)

        infos: list[TuneProgressInfo] = []
        model.tune(
            df,
            resume=True,
            n_trials=2,
            progress_callback=lambda i: infos.append(i),
        )
        assert len(infos) >= 2
        for info in infos:
            assert info.round == 2
            assert info.cumulative_trials > 0

    def test_n_trials_override(self) -> None:
        """n_trials parameter overrides config value."""
        cfg = _reg_config_with_tuning(n_trials=10)  # config says 10
        df = make_regression_df()
        model = Model(cfg)

        r1 = model.tune(df, n_trials=2)  # override to 2
        assert r1.rounds[0].n_trials == 2


# ---------------------------------------------------------------------------
# Boundary expansion integration tests
# ---------------------------------------------------------------------------


class TestModelRetuneBoundaryExpansion:
    """Integration tests for boundary detection + expansion during resume."""

    def test_expand_boundary_none_user_space_no_expand(self) -> None:
        """User-specified space + expand_boundary=None → no expansion."""
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df()
        model = Model(cfg)

        model.tune(df)
        r2 = model.tune(df, resume=True, n_trials=2)
        # expand_boundary defaults to False for user space
        assert r2.boundary_report is None

    def test_expand_boundary_explicit_true_user_space(self) -> None:
        """User-specified space + expand_boundary=True → expansion happens."""
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df()
        model = Model(cfg)

        model.tune(df)
        r2 = model.tune(df, resume=True, n_trials=2, expand_boundary=True)
        assert r2.boundary_report is not None
        assert isinstance(r2.boundary_report, BoundaryReport)

    def test_expand_boundary_false_no_report(self) -> None:
        """expand_boundary=False → no boundary detection."""
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df()
        model = Model(cfg)

        model.tune(df)
        r2 = model.tune(df, resume=True, n_trials=2, expand_boundary=False)
        assert r2.boundary_report is None


# ---------------------------------------------------------------------------
# Table tests
# ---------------------------------------------------------------------------


class TestTuningTableRetune:
    def test_tuning_table_has_round_and_state(self) -> None:
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df()
        model = Model(cfg)

        model.tune(df)
        model.tune(df, resume=True, n_trials=2)
        table = model.tuning_table()
        assert "round" in table.columns
        assert "state" in table.columns
        assert set(table["round"].unique()) == {1, 2}


class TestBoundaryTable:
    def test_boundary_table_no_report_raises(self) -> None:
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df()
        model = Model(cfg)
        model.tune(df)
        with pytest.raises(LizyMLError) as exc_info:
            model.boundary_table()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_boundary_table_with_report(self) -> None:
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df()
        model = Model(cfg)
        model.tune(df)
        model.tune(df, resume=True, n_trials=2, expand_boundary=True)
        table = model.boundary_table()
        assert "dim" in table.columns
        assert "edge" in table.columns
        assert "expanded" in table.columns
        assert len(table) == 2  # 2 dims in space
