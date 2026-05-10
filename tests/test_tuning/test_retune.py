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
    BoundaryDimStatus,
    BoundaryReport,
    RoundSummary,
    TrialResult,
    TuneProgressInfo,
    TuningResult,
)
from lizyml.tuning.search_space import attach_bounds, detect_boundary, expand_dims
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

    def test_float_dim_linear_lower_clamped_to_zero(self) -> None:
        """Regression for #110: linear FloatDim expansion must not produce
        negative lower bounds, otherwise downstream samplers (e.g. LightGBM
        learning_rate) crash on physically-impossible negative values.
        """
        dims = [FloatDim("learning_rate", low=0.001, high=0.1, log=False)]
        report = detect_boundary(dims, {"learning_rate": 0.001}, threshold=0.05)
        new = expand_dims(dims, report)
        assert isinstance(new[0], FloatDim)
        assert new[0].low >= 0.0, (
            f"FloatDim.low must not be negative after expansion, got {new[0].low}"
        )
        assert new[0].high == 0.1  # upper unchanged

    def test_float_dim_linear_lower_clamp_preserves_already_safe_values(
        self,
    ) -> None:
        """Linear expansion that produces a non-negative low must not be
        clamped (no over-correction)."""
        dims = [FloatDim("ratio", low=0.5, high=1.0, log=False)]
        report = detect_boundary(dims, {"ratio": 0.51}, threshold=0.05)
        new = expand_dims(dims, report)
        # 0.5 - (1.0 - 0.5) = 0.0, exactly the floor — should pass through
        assert new[0].low == 0.0
        assert new[0].high == 1.0


# ---------------------------------------------------------------------------
# H-0078 Phase 2: bounds-aware expansion
# ---------------------------------------------------------------------------


class TestExpandWithBounds:
    """Boundary expansion respects per-dim ``min_allowed`` / ``max_allowed``.

    Regression for #152: re-tuning grew ``learning_rate`` past 1.0 and
    ``feature_fraction`` past 1.0 because no parameter-aware ceiling
    existed. With ``max_allowed`` set, expansion clamps and signals via
    ``clamped_to_bound``.
    """

    def test_upper_expansion_clamps_to_max_allowed_log(self) -> None:
        """log dim near upper edge: 0.3 * 3 = 0.9 (under cap), but next round
        would exceed; with max_allowed=1.0 the expansion is clamped to 1.0."""
        dims = [
            FloatDim(
                "learning_rate",
                low=0.0001,
                high=0.5,
                log=True,
                max_allowed=1.0,
            )
        ]
        report = detect_boundary(dims, {"learning_rate": 0.49}, threshold=0.05)
        new = expand_dims(dims, report)
        assert isinstance(new[0], FloatDim)
        # 0.5 * 3.0 = 1.5 → clamped to 1.0
        assert new[0].high == 1.0
        # the dim status must signal clamp
        status = next(s for s in report.dims if s.name == "learning_rate")
        assert status.clamped_to_bound is True

    def test_upper_expansion_clamps_to_max_allowed_linear(self) -> None:
        dims = [FloatDim("ff", low=0.5, high=1.0, log=False, max_allowed=1.0)]
        report = detect_boundary(dims, {"ff": 0.99}, threshold=0.05)
        new = expand_dims(dims, report)
        assert new[0].high == 1.0
        status = next(s for s in report.dims if s.name == "ff")
        assert status.clamped_to_bound is True

    def test_lower_expansion_clamps_to_min_allowed(self) -> None:
        """validation_ratio with min_allowed=0.05 must not collapse to 0.0."""
        dims = [
            FloatDim(
                "validation_ratio",
                low=0.1,
                high=0.3,
                log=False,
                min_allowed=0.05,
            )
        ]
        report = detect_boundary(dims, {"validation_ratio": 0.11}, threshold=0.05)
        new = expand_dims(dims, report)
        # 0.1 - 0.2 = -0.1 → would clamp to 0.0 normally, but min_allowed
        # forces the floor up to 0.05.
        assert new[0].low == 0.05
        status = next(s for s in report.dims if s.name == "validation_ratio")
        assert status.clamped_to_bound is True

    def test_no_clamp_when_within_bounds(self) -> None:
        """Expansion that does not hit a bound must not signal clamped."""
        dims = [
            FloatDim(
                "lr",
                low=0.01,
                high=0.05,
                log=True,
                max_allowed=1.0,
            )
        ]
        report = detect_boundary(dims, {"lr": 0.049}, threshold=0.05)
        new = expand_dims(dims, report)
        # 0.05 * 3 = 0.15, well under 1.0 → no clamp
        assert new[0].high == pytest.approx(0.15, rel=1e-9)
        status = next(s for s in report.dims if s.name == "lr")
        assert status.clamped_to_bound is False

    def test_int_dim_upper_clamp_to_max_allowed(self) -> None:
        dims = [IntDim("max_depth", low=3, high=12, max_allowed=30)]
        report = detect_boundary(dims, {"max_depth": 12}, threshold=0.05)
        new = expand_dims(dims, report)
        # 12 + (12 - 3) = 21, under 30 → not clamped
        assert isinstance(new[0], IntDim)
        assert new[0].high == 21
        status = next(s for s in report.dims if s.name == "max_depth")
        assert status.clamped_to_bound is False

        # Second round of expansion would push past 30 → clamped
        dims2 = [IntDim("max_depth", low=3, high=21, max_allowed=30)]
        report2 = detect_boundary(dims2, {"max_depth": 21}, threshold=0.05)
        new2 = expand_dims(dims2, report2)
        # 21 + (21 - 3) = 39 → clamp to 30
        assert new2[0].high == 30
        status2 = next(s for s in report2.dims if s.name == "max_depth")
        assert status2.clamped_to_bound is True

    def test_no_max_allowed_means_unbounded_expansion(self) -> None:
        """Backward compat: no max_allowed → expansion grows unboundedly."""
        dims = [FloatDim("lr", low=0.0001, high=0.1, log=True)]
        report = detect_boundary(dims, {"lr": 0.099}, threshold=0.05)
        new = expand_dims(dims, report)
        assert new[0].high == pytest.approx(0.3, rel=1e-9)
        status = next(s for s in report.dims if s.name == "lr")
        assert status.clamped_to_bound is False

    def test_clamp_to_existing_high_is_a_noop(self) -> None:
        """If the expansion would land below the current high (degenerate
        max_allowed already at high), expansion is effectively a no-op:
        new high must not become smaller than the original high."""
        dims = [FloatDim("ff", low=0.5, high=1.0, log=False, max_allowed=1.0)]
        report = detect_boundary(dims, {"ff": 0.99}, threshold=0.05)
        new = expand_dims(dims, report)
        assert new[0].high == 1.0  # unchanged, not less
        # And the status reports clamp because expansion was bounded
        status = next(s for s in report.dims if s.name == "ff")
        assert status.clamped_to_bound is True

    def test_multi_round_log_expansion_never_exceeds_max_allowed(self) -> None:
        """Regression for #152: the original report shows ``learning_rate``
        drifting 0.1 → 0.3 → 0.9 → 2.7 over three re-tune rounds. With
        ``max_allowed=1.0`` propagated across rounds, ``high`` must never
        exceed 1.0 no matter how many rounds run."""
        dim: FloatDim = FloatDim(
            "learning_rate",
            low=0.0001,
            high=0.1,
            log=True,
            max_allowed=1.0,
        )
        dims: list = [dim]
        for _ in range(10):
            # always near upper edge → triggers expansion
            best = dims[0].high * 0.99
            report = detect_boundary(dims, {"learning_rate": best}, threshold=0.05)
            dims = expand_dims(dims, report)
            assert isinstance(dims[0], FloatDim)
            assert dims[0].high <= 1.0, (
                f"learning_rate.high must stay <= 1.0, got {dims[0].high}"
            )
            assert dims[0].max_allowed == 1.0  # bound is propagated
        # After 10 rounds it should be saturated at exactly 1.0
        assert dims[0].high == 1.0


class TestSearchDimMinMaxAllowed:
    """FloatDim / IntDim accept optional ``min_allowed`` / ``max_allowed``."""

    def test_floatdim_default_none(self) -> None:
        dim = FloatDim("x", low=0.0, high=1.0)
        assert dim.min_allowed is None
        assert dim.max_allowed is None

    def test_floatdim_with_bounds(self) -> None:
        dim = FloatDim("x", low=0.1, high=0.5, min_allowed=0.0, max_allowed=1.0)
        assert dim.min_allowed == 0.0
        assert dim.max_allowed == 1.0

    def test_intdim_default_none(self) -> None:
        dim = IntDim("n", low=1, high=10)
        assert dim.min_allowed is None
        assert dim.max_allowed is None

    def test_intdim_with_bounds(self) -> None:
        dim = IntDim("n", low=3, high=12, max_allowed=30)
        assert dim.max_allowed == 30
        assert dim.min_allowed is None


class TestBoundaryDimStatusClampedField:
    """``BoundaryDimStatus.clamped_to_bound`` defaults False."""

    def test_default_false(self) -> None:
        s = BoundaryDimStatus(
            name="x",
            best_value=0.5,
            low=0.0,
            high=1.0,
            position_pct=0.5,
            edge="none",
            expanded=False,
            new_low=None,
            new_high=None,
        )
        assert s.clamped_to_bound is False

    def test_set_true(self) -> None:
        s = BoundaryDimStatus(
            name="x",
            best_value=0.99,
            low=0.5,
            high=1.0,
            position_pct=0.99,
            edge="upper",
            expanded=True,
            new_low=0.5,
            new_high=1.0,
            clamped_to_bound=True,
        )
        assert s.clamped_to_bound is True


# ---------------------------------------------------------------------------
# H-0078 Phase 3: attach_bounds — provider->dim wiring
# ---------------------------------------------------------------------------


class TestAttachBounds:
    """``attach_bounds`` injects provider bounds onto matching dims."""

    def test_floatdim_matching_name_gets_bounds(self) -> None:
        dims = [FloatDim("learning_rate", low=0.001, high=0.1, log=True)]
        bounds = {"learning_rate": {"min": 1e-8, "max": 1.0}}
        new = attach_bounds(dims, bounds)
        assert isinstance(new[0], FloatDim)
        assert new[0].min_allowed == 1e-8
        assert new[0].max_allowed == 1.0
        # Other fields preserved
        assert new[0].low == 0.001
        assert new[0].high == 0.1
        assert new[0].log is True

    def test_intdim_matching_name_gets_bounds_as_int(self) -> None:
        dims = [IntDim("max_depth", low=3, high=12)]
        bounds = {"max_depth": {"min": -1, "max": 30}}
        new = attach_bounds(dims, bounds)
        assert isinstance(new[0], IntDim)
        assert new[0].min_allowed == -1
        assert new[0].max_allowed == 30
        assert isinstance(new[0].max_allowed, int)

    def test_dim_without_match_unchanged(self) -> None:
        dim = FloatDim("custom_param", low=0.1, high=0.5)
        bounds = {"learning_rate": {"min": 0.0, "max": 1.0}}
        new = attach_bounds([dim], bounds)
        assert new[0] is dim  # untouched, same object

    def test_categorical_dim_unchanged(self) -> None:
        dim = CategoricalDim("obj", choices=("a", "b"))
        bounds = {"obj": {"min": 0.0, "max": 1.0}}  # nonsensical but shouldn't crash
        new = attach_bounds([dim], bounds)
        assert new[0] is dim

    def test_user_set_min_allowed_preserved(self) -> None:
        """User-set min_allowed/max_allowed wins over provider bounds."""
        dims = [
            FloatDim(
                "learning_rate",
                low=0.001,
                high=0.1,
                log=True,
                min_allowed=1e-6,
                max_allowed=0.5,
            )
        ]
        bounds = {"learning_rate": {"min": 1e-8, "max": 1.0}}
        new = attach_bounds(dims, bounds)
        # user values preserved
        assert new[0].min_allowed == 1e-6
        assert new[0].max_allowed == 0.5

    def test_empty_bounds_returns_same_list(self) -> None:
        dims = [FloatDim("x", low=0.0, high=1.0)]
        new = attach_bounds(dims, {})
        assert new[0] is dims[0]

    def test_multiple_dims_some_matching(self) -> None:
        dims = [
            FloatDim("learning_rate", low=0.001, high=0.1),
            IntDim("max_depth", low=3, high=12),
            FloatDim("custom_unrelated", low=0.0, high=1.0),
        ]
        bounds = {
            "learning_rate": {"min": 1e-8, "max": 1.0},
            "max_depth": {"min": -1, "max": 30},
        }
        new = attach_bounds(dims, bounds)
        assert new[0].max_allowed == 1.0
        assert new[1].max_allowed == 30
        assert new[2].max_allowed is None


class TestModelTuneWiresParameterBounds:
    """Integration: ``Model.tune(re_tune=True)`` clamps boundary expansion
    using ``provider.parameter_bounds(task)`` (H-0078 Phase 3)."""

    def test_default_space_picks_up_lgbm_bounds(self) -> None:
        """Tuning with the default LightGBM space exposes max_allowed on
        ``learning_rate`` after the first ``tune()`` call."""
        cfg = make_config("regression")
        cfg["tuning"] = {"optuna": {"params": {"n_trials": 2}}}
        df = make_regression_df()
        model = Model(cfg)
        model.tune(df)
        space = model._space  # type: ignore[attr-defined]
        assert space is not None
        lr = next((d for d in space if d.name == "learning_rate"), None)
        assert lr is not None
        assert lr.max_allowed == 1.0

    def test_user_space_lr_picks_up_lgbm_bounds_by_name(self) -> None:
        """A user-supplied dim named ``learning_rate`` also picks up the
        provider bound by name match."""
        cfg = make_config("regression")
        cfg["tuning"] = {
            "optuna": {
                "params": {"n_trials": 2},
                "space": {
                    "learning_rate": {
                        "type": "float",
                        "low": 0.01,
                        "high": 0.3,
                        "log": True,
                    },
                },
            }
        }
        df = make_regression_df()
        model = Model(cfg)
        model.tune(df)
        space = model._space  # type: ignore[attr-defined]
        lr = next((d for d in space if d.name == "learning_rate"), None)
        assert lr is not None
        assert lr.max_allowed == 1.0
        assert lr.min_allowed == 1e-8

    def test_re_tune_clamps_learning_rate_at_one(self) -> None:
        """Regression for #152: re-tune cannot push ``learning_rate.high``
        past 1.0 because provider bounds are wired through ``Model.tune``."""
        cfg = make_config("regression")
        cfg["tuning"] = {
            "optuna": {
                "params": {"n_trials": 2},
                "space": {
                    "learning_rate": {
                        "type": "float",
                        "low": 0.5,
                        "high": 0.9,
                        "log": True,
                    },
                    "max_depth": {"type": "int", "low": 3, "high": 5},
                },
            }
        }
        df = make_regression_df()
        model = Model(cfg)
        model.tune(df)
        # Run several re-tune rounds with expand_boundary=True. The space
        # should never grow ``learning_rate.high`` past 1.0.
        for _ in range(5):
            model.tune(df, resume=True, n_trials=2, expand_boundary=True)
            space = model._space  # type: ignore[attr-defined]
            lr = next((d for d in space if d.name == "learning_rate"), None)
            assert lr is not None
            assert lr.high <= 1.0, f"learning_rate.high must stay <= 1.0, got {lr.high}"


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

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0, -0.1])
    def test_invalid_boundary_threshold_raises(self, threshold: float) -> None:
        cfg = _reg_config_with_tuning(n_trials=2)
        model = Model(cfg)
        with pytest.raises(LizyMLError) as exc_info:
            model.tune(make_regression_df(), boundary_threshold=threshold)
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_resume_accumulates_trials(self) -> None:
        cfg = _reg_config_with_tuning(n_trials=3)
        df = make_regression_df()
        model = Model(cfg)

        r1 = model.tune(df)
        assert len(r1.trials) == 3
        assert len(r1.rounds) == 1
        assert r1.rounds[0].round == 1

        r2 = model.tune(df, resume=True, n_trials=2)
        # Round 1: 3 trials. Round 2: optimize(n_trials=2) — enqueue counts
        # as 1 of the 2 trials. Total: 3 + 2 = 5 trials.
        assert len(r2.trials) == 5
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


# ---------------------------------------------------------------------------
# Edge case and branch coverage tests
# ---------------------------------------------------------------------------


class TestBoundaryEdgeCases:
    """Edge cases for boundary detection and expansion."""

    def test_zero_span_linear_returns_center(self) -> None:
        """Degenerate dim with low==high → position 0.5 → no expansion."""
        dims = [FloatDim("x", low=5.0, high=5.0)]
        report = detect_boundary(dims, {"x": 5.0}, threshold=0.05)
        s = report.dims[0]
        assert s.position_pct == 0.5
        assert s.edge == "none"
        assert s.expanded is False

    def test_zero_span_log_returns_center(self) -> None:
        """Degenerate log dim with low==high → position 0.5."""
        dims = [FloatDim("x", low=1.0, high=1.0, log=True)]
        report = detect_boundary(dims, {"x": 1.0}, threshold=0.05)
        s = report.dims[0]
        assert s.position_pct == 0.5
        assert s.edge == "none"

    def test_expand_range_edge_none_passthrough(self) -> None:
        """expand_dims with no expansion returns same dims."""
        dims = [
            FloatDim("a", low=0.0, high=1.0),
            IntDim("b", low=1, high=100),
        ]
        # Best in center → no expansion
        report = detect_boundary(dims, {"a": 0.5, "b": 50}, threshold=0.05)
        assert report.expanded_names == ()
        new = expand_dims(dims, report)
        assert new[0] is dims[0]
        assert new[1] is dims[1]

    def test_expand_dims_unrecognized_dim_passthrough(self) -> None:
        """Dim in expand_map but not matching FloatDim/IntDim falls through."""
        # This tests the else branch at L373 — a dim in expansion_map
        # that is somehow neither FloatDim nor IntDim (defensive code).
        # We can trigger it by creating a boundary report with CategoricalDim
        # manually marked as expanded (which shouldn't happen in practice).
        dims = [CategoricalDim("cat", choices=("a", "b"))]
        fake_status = BoundaryDimStatus(
            name="cat",
            best_value="a",
            low=None,
            high=None,
            position_pct=None,
            edge="none",
            expanded=True,  # Force expansion flag on categorical
            new_low=None,
            new_high=None,
        )
        fake_report = BoundaryReport(dims=(fake_status,), expanded_names=("cat",))
        new = expand_dims(dims, fake_report)
        # CategoricalDim should pass through unchanged
        assert new[0] is dims[0]

    def test_int_dim_log_upper_expansion(self) -> None:
        """IntDim with log scale expands upper bound correctly."""
        dims = [IntDim("x", low=1, high=100, log=True)]
        report = detect_boundary(dims, {"x": 99}, threshold=0.05)
        s = report.dims[0]
        assert s.edge == "upper"
        new = expand_dims(dims, report)
        assert isinstance(new[0], IntDim)
        assert new[0].high == 300  # 100 * 3.0, ceiled to int

    def test_float_linear_upper_expansion(self) -> None:
        """FloatDim linear upper expansion adds range."""
        dims = [FloatDim("x", low=0.0, high=1.0)]
        report = detect_boundary(dims, {"x": 0.98}, threshold=0.05)
        s = report.dims[0]
        assert s.edge == "upper"
        assert s.new_high is not None
        assert s.new_high == pytest.approx(2.0)  # 1.0 + (1.0 - 0.0)
        assert s.new_low == pytest.approx(0.0)  # lower bound unchanged

    def test_multiple_rounds_three_rounds(self) -> None:
        """Three sequential resume rounds build up correctly."""
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df()
        model = Model(cfg)

        r1 = model.tune(df)
        assert len(r1.rounds) == 1

        r2 = model.tune(df, resume=True, n_trials=2)
        assert len(r2.rounds) == 2
        assert r2.rounds[0].round == 1
        assert r2.rounds[1].round == 2

        r3 = model.tune(df, resume=True, n_trials=2)
        assert len(r3.rounds) == 3
        assert r3.rounds[2].round == 3
        # All trials should have round numbers 1, 2, or 3
        round_nums = {t.round for t in r3.trials}
        assert round_nums == {1, 2, 3}

    def test_round_summary_best_score_before(self) -> None:
        """Round 2 should record round 1's best as best_score_before."""
        cfg = _reg_config_with_tuning(n_trials=2)
        df = make_regression_df()
        model = Model(cfg)

        r1 = model.tune(df)
        r2 = model.tune(df, resume=True, n_trials=2)

        assert r2.rounds[0].best_score_before is None  # round 1 has no prior
        assert r2.rounds[1].best_score_before == r1.best_score


# ---------------------------------------------------------------------------
# Plot tests
# ---------------------------------------------------------------------------


class TestPlotTuningHistoryRounds:
    """Test plot_tuning_history with round separators."""

    def test_single_round_no_vlines(self) -> None:
        """Single round should not add vertical lines."""
        from lizyml.plots.tuning import plot_tuning_history

        result = TuningResult(
            best_model_params={},
            best_smart_params={},
            best_training_params={},
            best_score=0.5,
            trials=[
                TrialResult(number=0, params={"x": 1}, score=0.6, state="complete"),
                TrialResult(number=1, params={"x": 2}, score=0.5, state="complete"),
            ],
            metric_name="rmse",
            direction="minimize",
            rounds=(
                RoundSummary(
                    round=1,
                    n_trials=2,
                    best_score_before=None,
                    best_score_after=0.5,
                    expanded_dims=(),
                    space_snapshot=(),
                ),
            ),
        )
        fig = plot_tuning_history(result)
        # Should have traces but no vertical lines (shapes)
        assert len(fig.data) >= 1
        # No vline shapes for single round
        shapes = fig.layout.shapes or ()
        assert len(shapes) == 0

    def test_two_rounds_has_vline(self) -> None:
        """Two rounds should add a vertical dashed line at the boundary."""
        from lizyml.plots.tuning import plot_tuning_history

        result = TuningResult(
            best_model_params={},
            best_smart_params={},
            best_training_params={},
            best_score=0.4,
            trials=[
                TrialResult(
                    number=0, params={"x": 1}, score=0.6, state="complete", round=1
                ),
                TrialResult(
                    number=1, params={"x": 2}, score=0.5, state="complete", round=1
                ),
                TrialResult(
                    number=2, params={"x": 3}, score=0.4, state="complete", round=2
                ),
            ],
            metric_name="rmse",
            direction="minimize",
            rounds=(
                RoundSummary(
                    round=1,
                    n_trials=2,
                    best_score_before=None,
                    best_score_after=0.5,
                    expanded_dims=(),
                    space_snapshot=(),
                ),
                RoundSummary(
                    round=2,
                    n_trials=1,
                    best_score_before=0.5,
                    best_score_after=0.4,
                    expanded_dims=("lr",),
                    space_snapshot=(),
                ),
            ),
        )
        fig = plot_tuning_history(result)
        # Should have at least one vline shape
        shapes = fig.layout.shapes or ()
        assert len(shapes) >= 1
        # Should have round annotations
        annotations = fig.layout.annotations or ()
        assert len(annotations) >= 2  # one per round
        # Check annotation text
        texts = [a.text for a in annotations]
        assert any("Round 1" in t for t in texts)
        assert any("Round 2" in t and "lr" in t for t in texts)

    def test_plot_with_failed_trials(self) -> None:
        """Plot handles mix of complete and failed trials."""
        from lizyml.plots.tuning import plot_tuning_history

        result = TuningResult(
            best_model_params={},
            best_smart_params={},
            best_training_params={},
            best_score=0.5,
            trials=[
                TrialResult(number=0, params={"x": 1}, score=0.5, state="complete"),
                TrialResult(
                    number=1, params={"x": 2}, score=float("nan"), state="fail"
                ),
            ],
            metric_name="rmse",
            direction="minimize",
        )
        fig = plot_tuning_history(result)
        assert len(fig.data) >= 2  # complete + fail traces

    def test_plot_maximize_direction(self) -> None:
        """Best score line works for maximize direction."""
        from lizyml.plots.tuning import plot_tuning_history

        result = TuningResult(
            best_model_params={},
            best_smart_params={},
            best_training_params={},
            best_score=0.9,
            trials=[
                TrialResult(number=0, params={"x": 1}, score=0.7, state="complete"),
                TrialResult(number=1, params={"x": 2}, score=0.9, state="complete"),
            ],
            metric_name="auc",
            direction="maximize",
        )
        fig = plot_tuning_history(result)
        # Find the "Best Score" trace
        best_trace = [t for t in fig.data if t.name == "Best Score"]
        assert len(best_trace) == 1
        # Best line should be monotonically increasing
        ys = list(best_trace[0].y)
        assert ys[1] >= ys[0]
