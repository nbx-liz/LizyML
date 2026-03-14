"""Tests for Phase 10 — Evaluation (Evaluator, thresholding).

Covers (golden tests):
- Output structure: "raw" key with "oof", "oof_per_fold", "if_mean", "if_per_fold"
- "if_per_fold" / "oof_per_fold" length == n_splits
- OOF metric values are finite floats
- IF-mean == mean of IF-per-fold
- oof_per_fold computed on valid_idx (H-0045)
- Task-incompatible metrics raise UNSUPPORTED_METRIC
- "calibrated" key absent when calibrator is None
- Thresholding: optimise_threshold returns valid (threshold, score) pair
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.artifacts import RunMeta
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.evaluation.evaluator import Evaluator, _pred_for_metric
from lizyml.evaluation.thresholding import optimise_threshold
from lizyml.features.pipelines_native import NativeFeaturePipeline
from lizyml.metrics.base import BaseMetric
from lizyml.splitters.kfold import KFoldSplitter
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.inner_valid import NoInnerValid
from tests._helpers import make_binary_df, make_config, make_regression_df

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _run_meta() -> RunMeta:
    return RunMeta(
        lizyml_version="0.1.0",
        python_version="3.11",
        deps_versions={},
        config_normalized={},
        config_version=1,
        run_id="test",
        timestamp="2026-01-01T00:00:00",
    )


def _cv_fit_regression(n: int = 150, n_splits: int = 3):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(-1, 1, n)})
    y = pd.Series(X["a"] * 2.0 + rng.normal(0, 0.1, n), name="target")
    trainer = CVTrainer(
        outer_splitter=KFoldSplitter(n_splits=n_splits, shuffle=True, random_state=0),
        inner_valid=NoInnerValid(),
        pipeline_factory=NativeFeaturePipeline,
        estimator_factory=lambda: LGBMAdapter(
            task="regression", params={"n_estimators": 20}, random_state=0
        ),
        task="regression",
    )
    fit_result = trainer.fit(
        X, y, data_fingerprint=fp_compute(X, None), run_meta=_run_meta()
    )
    return fit_result, y, n_splits


def _cv_fit_binary(n: int = 200, n_splits: int = 3):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(-1, 1, n)})
    y = pd.Series((X["a"] > 5).astype(int), name="target")
    trainer = CVTrainer(
        outer_splitter=KFoldSplitter(n_splits=n_splits, shuffle=True, random_state=0),
        inner_valid=NoInnerValid(),
        pipeline_factory=NativeFeaturePipeline,
        estimator_factory=lambda: LGBMAdapter(
            task="binary", params={"n_estimators": 20}, random_state=0
        ),
        task="binary",
    )
    fit_result = trainer.fit(
        X, y, data_fingerprint=fp_compute(X, None), run_meta=_run_meta()
    )
    return fit_result, y, n_splits


# ---------------------------------------------------------------------------
# Structure (golden tests)
# ---------------------------------------------------------------------------


class TestEvaluatorStructureRegression:
    def test_top_level_keys(self) -> None:
        fit_result, y, _ = _cv_fit_regression()
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse", "mae"])
        assert set(out.keys()) == {"raw"}

    def test_raw_keys(self) -> None:
        fit_result, y, _ = _cv_fit_regression()
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse"])
        assert set(out["raw"].keys()) == {
            "oof",
            "oof_per_fold",
            "if_mean",
            "if_per_fold",
        }

    def test_oof_keys_match_metrics(self) -> None:
        fit_result, y, _ = _cv_fit_regression()
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse", "mae"])
        assert set(out["raw"]["oof"].keys()) == {"rmse", "mae"}

    def test_if_mean_keys_match_metrics(self) -> None:
        fit_result, y, _ = _cv_fit_regression()
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse", "mae"])
        assert set(out["raw"]["if_mean"].keys()) == {"rmse", "mae"}

    def test_if_per_fold_length(self) -> None:
        n_splits = 4
        fit_result, y, _ = _cv_fit_regression(n_splits=n_splits)
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse"])
        assert len(out["raw"]["if_per_fold"]) == n_splits

    def test_if_per_fold_keys(self) -> None:
        fit_result, y, _ = _cv_fit_regression()
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse", "mae"])
        for fold_metrics in out["raw"]["if_per_fold"]:
            assert set(fold_metrics.keys()) == {"rmse", "mae"}

    def test_oof_values_are_finite(self) -> None:
        fit_result, y, _ = _cv_fit_regression()
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse", "mae"])
        for v in out["raw"]["oof"].values():
            assert np.isfinite(v)

    def test_if_mean_equals_mean_of_per_fold(self) -> None:
        fit_result, y, _ = _cv_fit_regression(n_splits=3)
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse"])
        per_fold_rmse = [fold["rmse"] for fold in out["raw"]["if_per_fold"]]
        assert out["raw"]["if_mean"]["rmse"] == pytest.approx(np.mean(per_fold_rmse))

    def test_oof_per_fold_length(self) -> None:
        n_splits = 4
        fit_result, y, _ = _cv_fit_regression(n_splits=n_splits)
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse"])
        assert len(out["raw"]["oof_per_fold"]) == n_splits

    def test_oof_per_fold_keys(self) -> None:
        fit_result, y, _ = _cv_fit_regression()
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse", "mae"])
        for fold_metrics in out["raw"]["oof_per_fold"]:
            assert set(fold_metrics.keys()) == {"rmse", "mae"}

    def test_oof_per_fold_values_finite(self) -> None:
        fit_result, y, _ = _cv_fit_regression()
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse", "mae"])
        for fold_dict in out["raw"]["oof_per_fold"]:
            for v in fold_dict.values():
                assert np.isfinite(v)

    def test_oof_per_fold_computed_on_valid_idx(self) -> None:
        """oof_per_fold values must match manual metric on valid_idx."""
        fit_result, y_series, n_splits = _cv_fit_regression()
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y_series, ["rmse"])
        y_arr = np.asarray(y_series)
        for k, (_, valid_idx) in enumerate(fit_result.splits.outer):
            y_valid = y_arr[valid_idx]
            oof_valid = fit_result.oof_pred[valid_idx]
            expected = float(np.sqrt(np.mean((y_valid - oof_valid) ** 2)))
            assert out["raw"]["oof_per_fold"][k]["rmse"] == pytest.approx(expected)

    def test_no_calibrated_key_without_calibrator(self) -> None:
        fit_result, y, _ = _cv_fit_regression()
        assert fit_result.calibrator is None
        ev = Evaluator(task="regression")
        out = ev.evaluate(fit_result, y, ["rmse"])
        assert "calibrated" not in out


class TestEvaluatorStructureBinary:
    def test_top_level_and_raw_keys(self) -> None:
        fit_result, y, _ = _cv_fit_binary()
        ev = Evaluator(task="binary")
        out = ev.evaluate(fit_result, y, ["logloss", "auc"])
        assert "raw" in out
        assert set(out["raw"].keys()) == {
            "oof",
            "oof_per_fold",
            "if_mean",
            "if_per_fold",
        }

    def test_metric_values_valid(self) -> None:
        fit_result, y, _ = _cv_fit_binary()
        ev = Evaluator(task="binary")
        out = ev.evaluate(fit_result, y, ["logloss", "auc", "f1"])
        for v in out["raw"]["oof"].values():
            assert np.isfinite(v)

    def test_auc_in_valid_range(self) -> None:
        fit_result, y, _ = _cv_fit_binary()
        ev = Evaluator(task="binary")
        out = ev.evaluate(fit_result, y, ["auc"])
        auc = out["raw"]["oof"]["auc"]
        assert 0.0 <= auc <= 1.0

    def test_if_per_fold_length_binary(self) -> None:
        n_splits = 3
        fit_result, y, _ = _cv_fit_binary(n_splits=n_splits)
        ev = Evaluator(task="binary")
        out = ev.evaluate(fit_result, y, ["logloss"])
        assert len(out["raw"]["if_per_fold"]) == n_splits


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestEvaluatorErrors:
    def test_incompatible_metric_raises(self) -> None:
        fit_result, y, _ = _cv_fit_regression()
        ev = Evaluator(task="regression")
        with pytest.raises(LizyMLError) as exc_info:
            ev.evaluate(fit_result, y, ["auc"])  # auc is binary-only
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_regression_metric_on_binary_raises(self) -> None:
        fit_result, y, _ = _cv_fit_binary()
        ev = Evaluator(task="binary")
        with pytest.raises(LizyMLError) as exc_info:
            ev.evaluate(fit_result, y, ["rmse"])  # rmse is regression-only
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------


class TestThresholding:
    def test_returns_threshold_in_range(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200).astype(int)
        y_proba = np.clip(y_true * 0.7 + rng.uniform(0, 0.3, 200), 0, 1)

        from lizyml.metrics.classification import F1

        threshold, score = optimise_threshold(
            y_true, y_proba, F1(), greater_is_better=True
        )
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= score <= 1.0

    def test_perfect_predictor_threshold_near_half(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])

        from lizyml.metrics.classification import F1

        threshold, score = optimise_threshold(
            y_true, y_proba, F1(), greater_is_better=True
        )
        # With such clear separation, threshold should be in the middle
        assert 0.2 <= threshold <= 0.8
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# T-4: Model.evaluate() contract via Facade
# ---------------------------------------------------------------------------


def _reg_config_small() -> dict:
    cfg = make_config("regression")
    cfg["evaluation"] = {"metrics": ["rmse", "mae"]}
    return cfg


def _bin_config_small() -> dict:
    cfg = make_config("binary")
    cfg["evaluation"] = {"metrics": ["logloss", "auc"]}
    return cfg


class TestModelEvaluateFacade:
    def test_evaluate_no_args_returns_precomputed(self) -> None:
        m = Model(_reg_config_small())
        m.fit(data=make_regression_df(n=150))
        result = m.evaluate()
        assert "raw" in result
        assert "oof" in result["raw"]

    def test_evaluate_with_subset_metrics_filters_output(self) -> None:
        """evaluate(metrics=["rmse"]) returns only the requested metric."""
        m = Model(_reg_config_small())
        m.fit(data=make_regression_df(n=150))
        result = m.evaluate(metrics=["rmse"])
        assert set(result["raw"]["oof"].keys()) == {"rmse"}
        assert set(result["raw"]["if_mean"].keys()) == {"rmse"}
        for fold in result["raw"]["if_per_fold"]:
            assert set(fold.keys()) == {"rmse"}
        for fold in result["raw"]["oof_per_fold"]:
            assert set(fold.keys()) == {"rmse"}

    def test_evaluate_subset_values_match_precomputed(self) -> None:
        """Filtered values must equal the pre-computed values from fit()."""
        m = Model(_reg_config_small())
        m.fit(data=make_regression_df(n=150))
        full = m.evaluate()
        filtered = m.evaluate(metrics=["rmse"])
        assert filtered["raw"]["oof"]["rmse"] == pytest.approx(
            full["raw"]["oof"]["rmse"]
        )

    def test_evaluate_incompatible_metric_raises_unsupported(self) -> None:
        """evaluate() with a task-incompatible metric must raise UNSUPPORTED_METRIC."""
        m = Model(_bin_config_small())
        m.fit(data=make_binary_df(n=150))
        with pytest.raises(LizyMLError) as exc_info:
            m.evaluate(metrics=["rmse"])  # rmse is not valid for binary
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_evaluate_incompatible_regression_metric_raises(self) -> None:
        """evaluate() with auc on regression must raise UNSUPPORTED_METRIC."""
        m = Model(_reg_config_small())
        m.fit(data=make_regression_df(n=150))
        with pytest.raises(LizyMLError) as exc_info:
            m.evaluate(metrics=["auc"])  # auc is not valid for regression
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_evaluate_before_fit_raises_model_not_fit(self) -> None:
        m = Model(_reg_config_small())
        with pytest.raises(LizyMLError) as exc_info:
            m.evaluate()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT


# ---------------------------------------------------------------------------
# H-0049: Multiclass OVA probability normalization
# ---------------------------------------------------------------------------


class _FakeSimplexMetric(BaseMetric):
    """Fake metric with needs_proba=True AND needs_simplex=True."""

    @property
    def name(self) -> str:
        return "fake_simplex"

    @property
    def needs_proba(self) -> bool:
        return True

    @property
    def needs_simplex(self) -> bool:
        return True

    @property
    def greater_is_better(self) -> bool:
        return True

    def __call__(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
        return 0.0


class _FakePerClassMetric(BaseMetric):
    """Fake metric with needs_proba=True but needs_simplex=False (default)."""

    @property
    def name(self) -> str:
        return "fake_perclass"

    @property
    def needs_proba(self) -> bool:
        return True

    @property
    def greater_is_better(self) -> bool:
        return True

    def __call__(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
        return 0.0


class _FakeMetricLabel(BaseMetric):
    """Fake metric with needs_proba=False for testing."""

    @property
    def name(self) -> str:
        return "fake_label"

    @property
    def needs_proba(self) -> bool:
        return False

    @property
    def greater_is_better(self) -> bool:
        return True

    def __call__(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
        return 0.0


class TestMulticlassOvaNormalization:
    """H-0049: _pred_for_metric normalizes only simplex-required metrics."""

    simplex_metric = _FakeSimplexMetric()
    perclass_metric = _FakePerClassMetric()
    label_metric = _FakeMetricLabel()

    # --- Simplex metrics (needs_simplex=True) are normalized ---

    def test_softmax_pred_unchanged_for_simplex(self) -> None:
        """Softmax predictions (already sum=1) are returned as-is."""
        pred = np.array([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]])
        result = _pred_for_metric(self.simplex_metric, pred, "multiclass")
        np.testing.assert_allclose(result, pred, atol=1e-12)

    def test_ova_sigmoid_normalized_for_simplex(self) -> None:
        """OVA sigmoid predictions are row-normalized for simplex metrics."""
        pred = np.array([[0.34, 0.007, 0.96], [0.5, 0.5, 0.8]])
        result = _pred_for_metric(self.simplex_metric, pred, "multiclass")
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        # Relative order preserved
        assert result[0, 2] > result[0, 0] > result[0, 1]

    def test_zero_row_handled(self) -> None:
        """All-zero row does not cause division by zero."""
        pred = np.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.4]])
        result = _pred_for_metric(self.simplex_metric, pred, "multiclass")
        assert np.all(np.isfinite(result))
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result[1].sum(), 1.0)

    def test_near_zero_row_safe(self) -> None:
        """Near-zero (subnormal) row normalizes correctly — no inf."""
        pred = np.array([[1e-310, 1e-310, 1e-310], [0.3, 0.3, 0.4]])
        result = _pred_for_metric(self.simplex_metric, pred, "multiclass")
        assert np.all(np.isfinite(result))
        # Each element <= row_sum, so result <= 1.0 always
        assert result.max() <= 1.0

    # --- Per-class metrics (needs_simplex=False) are NOT normalized ---

    def test_ova_pred_not_normalized_for_perclass(self) -> None:
        """Per-class metric receives raw predictions (no normalization)."""
        pred = np.array([[0.34, 0.007, 0.96], [0.5, 0.5, 0.8]])
        result = _pred_for_metric(self.perclass_metric, pred, "multiclass")
        np.testing.assert_array_equal(result, pred)

    def test_auc_pr_receives_raw_values(self) -> None:
        """AUCPR (needs_simplex=False) must receive raw predictions."""
        from lizyml.metrics.classification import AUCPR

        metric = AUCPR()
        assert metric.needs_proba is True
        assert metric.needs_simplex is False

        pred = np.array([[0.34, 0.007, 0.96], [0.5, 0.5, 0.8]])
        result = _pred_for_metric(metric, pred, "multiclass")
        np.testing.assert_array_equal(result, pred)

    def test_brier_receives_raw_values(self) -> None:
        """Brier (needs_simplex=False) must receive raw predictions."""
        from lizyml.metrics.classification import Brier

        metric = Brier()
        assert metric.needs_proba is True
        assert metric.needs_simplex is False

        pred = np.array([[0.34, 0.007, 0.96], [0.5, 0.5, 0.8]])
        result = _pred_for_metric(metric, pred, "multiclass")
        np.testing.assert_array_equal(result, pred)

    # --- Other task types unaffected ---

    def test_binary_not_affected(self) -> None:
        """Binary predictions are not row-normalized."""
        pred = np.array([0.2, 0.8, 0.5])
        result = _pred_for_metric(self.simplex_metric, pred, "binary")
        np.testing.assert_array_equal(result, pred)

    def test_needs_proba_false_not_affected(self) -> None:
        """Metrics with needs_proba=False skip normalization."""
        pred = np.array([[0.34, 0.007, 0.96]])
        result = _pred_for_metric(self.label_metric, pred, "multiclass")
        assert result.ndim == 1  # argmax applied, not normalized

    def test_regression_not_affected(self) -> None:
        """Regression predictions are returned as-is."""
        pred = np.array([1.5, 2.3, 3.1])
        result = _pred_for_metric(self.simplex_metric, pred, "regression")
        np.testing.assert_array_equal(result, pred)

    # --- Integration test ---

    def test_auc_end_to_end_with_multiclassova(self) -> None:
        """AUC metric end-to-end: _pred_for_metric → AUC.__call__."""
        from lizyml.metrics.classification import AUC

        metric = AUC()
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array(
            [
                [0.8, 0.1, 0.3],
                [0.2, 0.9, 0.1],
                [0.1, 0.2, 0.95],
                [0.7, 0.3, 0.2],
                [0.1, 0.85, 0.15],
            ]
        )
        assert y_pred.sum(axis=1).max() > 1.0

        pred_for_metric = _pred_for_metric(metric, y_pred, "multiclass")
        np.testing.assert_allclose(pred_for_metric.sum(axis=1), 1.0)
        score = metric(y_true, pred_for_metric)
        assert 0.0 <= score <= 1.0

    # --- needs_simplex property contract ---

    def test_auc_needs_simplex_true(self) -> None:
        """AUC metric declares needs_simplex=True."""
        from lizyml.metrics.classification import AUC

        assert AUC().needs_simplex is True

    def test_logloss_needs_simplex_true(self) -> None:
        """LogLoss metric declares needs_simplex=True."""
        from lizyml.metrics.classification import LogLoss

        assert LogLoss().needs_simplex is True

    def test_default_needs_simplex_false(self) -> None:
        """BaseMetric default needs_simplex is False."""
        assert self.perclass_metric.needs_simplex is False
