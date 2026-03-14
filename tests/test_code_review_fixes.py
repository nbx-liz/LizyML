"""Tests for code review fixes — Phase 1 through Phase 3.

TDD RED: All tests written before implementation.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from lizyml.calibration.cross_fit import cross_fit_calibrate
from lizyml.calibration.platt import PlattCalibrator
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TrialResult, TuningResult
from lizyml.metrics.classification import ECE, LogLoss
from lizyml.metrics.regression import RMSLE
from lizyml.splitters import GroupTimeSeriesSplitter
from lizyml.splitters.kfold import KFoldSplitter


def _kfold_indices(
    n: int, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFoldSplitter(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(n))


# ---------------------------------------------------------------------------
# Phase 1: CRITICAL — LogLoss multiclass 2D support
# ---------------------------------------------------------------------------


class TestLogLossMulticlass:
    """LogLoss must handle 2D y_pred for multiclass (like AUC, Brier, AUCPR)."""

    def test_2d_returns_float(self) -> None:
        """LogLoss with 2D y_pred (n_samples, n_classes) should return a float."""
        rng = np.random.default_rng(42)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = rng.dirichlet([1, 1, 1], size=10)
        metric = LogLoss()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert result > 0.0

    def test_2d_good_predictions_lower_loss(self) -> None:
        """Better predictions should give lower log loss."""
        n = 100
        y_true = np.array([0] * 40 + [1] * 30 + [2] * 30)
        rng = np.random.default_rng(42)
        # Good predictions: high prob for correct class
        y_good = rng.dirichlet([0.3, 0.3, 0.3], size=n)
        for i in range(n):
            y_good[i, y_true[i]] += 2.0
        y_good = y_good / y_good.sum(axis=1, keepdims=True)
        # Bad predictions: random
        y_bad = rng.dirichlet([1, 1, 1], size=n)

        metric = LogLoss()
        loss_good = metric(y_true, y_good)
        loss_bad = metric(y_true, y_bad)
        assert loss_good < loss_bad

    def test_2d_shape_mismatch_raises(self) -> None:
        """2D y_pred with wrong n_samples should raise LizyMLError."""
        y_true = np.array([0, 1, 2])
        y_pred = np.random.default_rng(0).dirichlet([1, 1, 1], size=5)  # 5 != 3
        metric = LogLoss()
        with pytest.raises(LizyMLError):
            metric(y_true, y_pred)

    def test_1d_binary_unchanged(self) -> None:
        """1D binary case should still work exactly as before."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
        metric = LogLoss()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert result > 0.0


# ---------------------------------------------------------------------------
# Phase 1: cross_fit NaN initialization
# ---------------------------------------------------------------------------


class TestCrossFitNaNInit:
    """cross_fit_calibrate must use NaN-initialized array, not np.empty."""

    def test_unfilled_indices_are_nan(self) -> None:
        """If split_indices don't cover all rows, unfilled positions must be NaN."""
        n = 100
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, n).astype(float)
        scores = y + rng.normal(0, 0.3, n)
        scores = np.clip(scores, 0.01, 0.99)

        # Create split indices that deliberately skip index 0
        full_indices = _kfold_indices(n, n_splits=3)
        partial_indices = []
        for train_idx, val_idx in full_indices:
            # Remove index 0 from val_idx
            val_idx = val_idx[val_idx != 0]
            partial_indices.append((train_idx, val_idx))

        result = cross_fit_calibrate(
            oof_scores=scores,
            y=y,
            calibrator_factory=PlattCalibrator,
            split_indices=partial_indices,
        )
        # Index 0 was never filled — must be NaN, not garbage
        assert np.isnan(result.calibrated_oof[0]), (
            "Unfilled index should be NaN, not uninitialized memory"
        )

    def test_method_uses_c_final_name(self) -> None:
        """method field should use c_final.name, not create extra instance."""
        n = 100
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, n).astype(float)
        scores = np.clip(y + rng.normal(0, 0.3, n), 0.01, 0.99)
        result = cross_fit_calibrate(
            oof_scores=scores,
            y=y,
            calibrator_factory=PlattCalibrator,
            split_indices=_kfold_indices(n, n_splits=3),
        )
        assert result.method == "platt"


# ---------------------------------------------------------------------------
# Phase 1: GroupTimeSeriesSplitter trailing group fix
# ---------------------------------------------------------------------------


class TestGroupTimeSeriesTrailingGroups:
    """GroupTimeSeriesSplitter must not silently drop trailing groups."""

    def test_all_groups_covered_non_round(self) -> None:
        """With n_groups=11, n_splits=3, all 11 groups must appear in some fold."""
        n_groups = 11
        samples_per_group = 5
        groups = np.repeat(np.arange(n_groups), samples_per_group)
        n_samples = len(groups)

        splitter = GroupTimeSeriesSplitter(n_splits=3)
        folds = list(splitter.split(n_samples, groups=groups))

        # Collect all groups that appear in any validation fold
        all_valid_groups = set()
        all_train_groups = set()
        for train_idx, valid_idx in folds:
            all_valid_groups.update(groups[valid_idx].tolist())
            all_train_groups.update(groups[train_idx].tolist())

        all_covered = all_valid_groups | all_train_groups
        expected = set(range(n_groups))
        assert all_covered == expected, (
            f"Groups {expected - all_covered} were dropped from all folds"
        )

    def test_last_fold_extends_to_end(self) -> None:
        """The last fold's validation set should extend to include trailing groups."""
        n_groups = 7
        samples_per_group = 3
        groups = np.repeat(np.arange(n_groups), samples_per_group)
        n_samples = len(groups)

        splitter = GroupTimeSeriesSplitter(n_splits=3)
        folds = list(splitter.split(n_samples, groups=groups))

        # The last fold's valid groups should include the final group
        _, last_valid_idx = folds[-1]
        last_valid_groups = set(groups[last_valid_idx].tolist())
        assert n_groups - 1 in last_valid_groups, (
            f"Last group {n_groups - 1} not in last fold's validation set"
        )

    def test_no_leakage_between_folds(self) -> None:
        """Train and valid groups must be disjoint in each fold."""
        groups = np.repeat(np.arange(10), 4)
        splitter = GroupTimeSeriesSplitter(n_splits=3)
        for train_idx, valid_idx in splitter.split(len(groups), groups=groups):
            train_groups = set(groups[train_idx])
            valid_groups = set(groups[valid_idx])
            assert len(train_groups & valid_groups) == 0

    def test_gap_respected_with_trailing(self) -> None:
        """Even with trailing group fix, gap must be respected."""
        groups = np.repeat(np.arange(10), 3)
        splitter = GroupTimeSeriesSplitter(n_splits=2, gap=1)
        for train_idx, valid_idx in splitter.split(len(groups), groups=groups):
            train_groups = set(groups[train_idx])
            valid_groups = set(groups[valid_idx])
            # Gap means no overlap and a gap between max(train) and min(valid)
            if len(train_groups) > 0 and len(valid_groups) > 0:
                assert max(train_groups) + 1 < min(valid_groups), (
                    "Gap not respected between train and valid groups"
                )


# ---------------------------------------------------------------------------
# Phase 3: ECE boundary — y_pred == 1.0 inclusion
# ---------------------------------------------------------------------------


class TestECEBoundary:
    """ECE must include y_pred == 1.0 in the last bin."""

    def test_single_pred_one_wrong_class(self) -> None:
        """y_pred=1.0 with y_true=0 → ECE should be 1.0, not 0.0.

        Bug: if y_pred==1.0 falls outside all bins, ECE is 0.0 (vacuous).
        """
        y_true = np.array([0])
        y_pred = np.array([1.0])
        metric = ECE(n_bins=10)
        result = metric(y_true, y_pred)
        # Prediction is 1.0, label is 0 → max miscalibration → ECE = 1.0
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_all_ones_correct_class(self) -> None:
        """y_pred=1.0 with y_true=1 → perfectly calibrated → ECE ≈ 0."""
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1.0, 1.0, 1.0])
        metric = ECE(n_bins=10)
        result = metric(y_true, y_pred)
        assert result == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Phase 3: RMSLE negative prediction guard
# ---------------------------------------------------------------------------


class TestRMSLENegativeGuard:
    """RMSLE must raise LizyMLError for negative predictions/targets."""

    def test_negative_pred_raises(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, -0.5, 3.0])
        metric = RMSLE()
        with pytest.raises(LizyMLError) as exc_info:
            metric(y_true, y_pred)
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_negative_true_raises(self) -> None:
        y_true = np.array([1.0, -1.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metric = RMSLE()
        with pytest.raises(LizyMLError) as exc_info:
            metric(y_true, y_pred)
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_nonneg_values_pass(self) -> None:
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.1, 1.1, 1.9])
        metric = RMSLE()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# Phase 2: scipy optional dependency guard
# ---------------------------------------------------------------------------


class TestScipyOptionalDepGuard:
    """QQ plots must raise LizyMLError when scipy is not installed."""

    def test_scipy_missing_raises_lizyml_error(self) -> None:
        """scipy missing must raise OPTIONAL_DEP_MISSING for QQ plots."""
        pytest.importorskip("plotly")
        with patch("lizyml.plots.residuals._scipy_stats", None):
            from lizyml.plots.residuals import _add_qq_traces

            with pytest.raises(LizyMLError) as exc_info:
                _add_qq_traces(None, np.array([1.0, 2.0, 3.0]))
            assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING


# ---------------------------------------------------------------------------
# Phase 2: SHAP empty models guard
# ---------------------------------------------------------------------------


class TestShapEmptyModels:
    """compute_shap_importance must handle empty models list gracefully."""

    def test_empty_models_no_crash(self) -> None:
        pytest.importorskip("shap")
        from lizyml.explain.shap_explainer import compute_shap_importance

        result = compute_shap_importance(
            models=[],
            X=None,  # type: ignore[arg-type]
            splits_outer=[],
            task="regression",
            feature_names=["a", "b"],
            pipeline_state=None,
        )
        # Should return a dict with 0.0 values, not crash
        assert isinstance(result, dict)
        assert result == {"a": 0.0, "b": 0.0}


# ---------------------------------------------------------------------------
# Phase 2: FitResult immutability — model.py mutation
# ---------------------------------------------------------------------------


class TestFitResultImmutability:
    """FitResult should not be mutated in-place after construction."""

    def test_fit_returns_complete_result(self) -> None:
        """Smoke test that fit() returns a fully populated FitResult."""
        from lizyml import Model
        from tests._helpers import make_binary_df, make_config

        df = make_binary_df()
        m = Model(make_config("binary", calibration="platt", n_estimators=20))
        result = m.fit(data=df)
        # Verify calibrator and metrics are set
        assert result.calibrator is not None
        assert "raw" in result.metrics
        assert "calibrated" in result.metrics


# ---------------------------------------------------------------------------
# Phase 2: Objective overwrite protection
# ---------------------------------------------------------------------------


class TestObjectiveOverwriteProtection:
    """Task-locked objective must not be overridable by user params."""

    def test_objective_locked_after_user_params(self) -> None:
        """Even if user passes objective='regression', binary task keeps 'binary'."""
        from lizyml.estimators.lgbm import LGBMAdapter

        adapter = LGBMAdapter(
            task="binary",
            params={"objective": "regression"},
        )
        params, _ = adapter._build_params()
        assert params["objective"] == "binary"


# ---------------------------------------------------------------------------
# Phase 3: datetime UTC
# ---------------------------------------------------------------------------


class TestDatetimeUTC:
    """RunMeta timestamp must include timezone info."""

    def test_timestamp_has_utc(self) -> None:
        from lizyml import Model
        from tests._helpers import make_config, make_regression_df

        m = Model(make_config("regression", n_estimators=10))
        result = m.fit(data=make_regression_df(n=50))
        ts = result.run_meta.timestamp
        # UTC timestamps end with +00:00 or Z
        assert "+00:00" in ts or "Z" in ts, f"Timestamp lacks UTC: {ts}"


# ---------------------------------------------------------------------------
# Phase 3: TuningResult deep freeze
# ---------------------------------------------------------------------------


class TestTuningResultDeepFreeze:
    """Frozen dataclasses must not allow mutation of contained dicts/lists."""

    def test_best_params_immutable(self) -> None:
        source_params = {"lr": 0.1}
        tr = TuningResult(
            best_params=source_params,
            best_score=0.5,
            trials=[],
            metric_name="rmse",
            direction="minimize",
        )
        # Mutate the original dict
        source_params["lr"] = 999.0
        # TuningResult should have its own copy
        assert tr.best_params["lr"] == 0.1

    def test_trials_list_immutable(self) -> None:
        trials = [TrialResult(number=0, params={}, score=0.5, state="complete")]
        tr = TuningResult(
            best_params={},
            best_score=0.5,
            trials=trials,
            metric_name="rmse",
            direction="minimize",
        )
        trials.append(TrialResult(number=1, params={}, score=0.6, state="complete"))
        assert len(tr.trials) == 1


# ---------------------------------------------------------------------------
# Phase 3: bare assert → LizyMLError
# ---------------------------------------------------------------------------


class TestEvaluateAssertReplaced:
    """evaluate() must raise LizyMLError, not AssertionError, on unfitted model."""

    def test_evaluate_unfitted_raises_lizyml_error(self) -> None:
        from lizyml import Model
        from tests._helpers import make_config

        m = Model(make_config("regression"))
        with pytest.raises(LizyMLError) as exc_info:
            m.evaluate()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT


# ---------------------------------------------------------------------------
# Phase 3: _filter_metrics empty branches
# ---------------------------------------------------------------------------


class TestFilterMetricsNoBranches:
    """_filter_metrics must remove empty branches after filtering."""

    def test_no_empty_calibrated_branch(self) -> None:
        from lizyml.core.model import _filter_metrics

        metrics = {
            "raw": {"oof": {"rmse": 0.5, "mae": 0.3}, "if_mean": {"rmse": 0.4}},
            "calibrated": {"oof": {"logloss": 0.2}},
        }
        result = _filter_metrics(metrics, {"rmse"})
        # "calibrated" has no "rmse" metrics → should be removed
        if "calibrated" in result:
            cal = result["calibrated"]
            # All sub-dicts must be non-empty if branch exists
            for k, v in cal.items():
                if isinstance(v, dict):
                    assert len(v) > 0, f"Empty branch: calibrated.{k}"


# ---------------------------------------------------------------------------
# Phase 3: LGBMAdapter.update_params no mutation
# ---------------------------------------------------------------------------


class TestUpdateParamsNoMutation:
    """LGBMAdapter.update_params must not mutate the original params dict."""

    def test_original_params_unchanged(self) -> None:
        from lizyml.estimators.lgbm import LGBMAdapter

        original = {"learning_rate": 0.1}
        adapter = LGBMAdapter(task="regression", params=original)
        adapter.update_params({"max_depth": 5})
        # Original dict passed to constructor should not be modified
        assert "max_depth" not in original


# ---------------------------------------------------------------------------
# Phase 3: HoldoutInnerValid n_valid consistency
# ---------------------------------------------------------------------------


class TestHoldoutNValidConsistency:
    """HoldoutSplitter and HoldoutInnerValid must use same rounding for n_valid."""

    def test_same_n_valid_for_same_ratio(self) -> None:
        from lizyml.splitters.holdout import HoldoutSplitter
        from lizyml.training.inner_valid import HoldoutInnerValid

        n_samples = 15
        ratio = 0.1

        # HoldoutSplitter
        hs = HoldoutSplitter(ratio=ratio, random_state=0)
        hs_folds = list(hs.split(n_samples))
        hs_n_valid = len(hs_folds[0][1])

        # HoldoutInnerValid
        hiv = HoldoutInnerValid(ratio=ratio, random_state=0)
        hiv_result = hiv.split(n_samples)
        assert hiv_result is not None
        hiv_n_valid = len(hiv_result[1])

        assert hs_n_valid == hiv_n_valid, (
            f"Inconsistent n_valid: HoldoutSplitter={hs_n_valid}, "
            f"HoldoutInnerValid={hiv_n_valid}"
        )
