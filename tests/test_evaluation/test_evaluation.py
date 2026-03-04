"""Tests for Phase 10 — Evaluation (Evaluator, thresholding).

Covers (golden tests):
- Output structure: "raw" key with "oof", "if_mean", "if_per_fold"
- "if_per_fold" length == n_splits
- OOF metric values are finite floats
- IF-mean == mean of IF-per-fold
- Task-incompatible metrics raise UNSUPPORTED_METRIC
- "calibrated" key absent when calibrator is None
- Thresholding: optimise_threshold returns valid (threshold, score) pair
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.artifacts import RunMeta
from lizyml.data.fingerprint import compute as fp_compute
from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.evaluation.evaluator import Evaluator
from lizyml.evaluation.thresholding import optimise_threshold
from lizyml.features.pipelines_native import NativeFeaturePipeline
from lizyml.splitters.kfold import KFoldSplitter
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.inner_valid import NoInnerValid

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
        assert set(out["raw"].keys()) == {"oof", "if_mean", "if_per_fold"}

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
        assert set(out["raw"].keys()) == {"oof", "if_mean", "if_per_fold"}

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
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
        "evaluation": {"metrics": ["rmse", "mae"]},
    }


def _bin_config_small() -> dict:
    return {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
        "evaluation": {"metrics": ["logloss", "auc"]},
    }


def _reg_df_small(n: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"]
    return df


def _bin_df_small(n: int = 150, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    return df


class TestModelEvaluateFacade:
    def test_evaluate_no_args_returns_precomputed(self) -> None:
        m = Model(_reg_config_small())
        m.fit(data=_reg_df_small())
        result = m.evaluate()
        assert "raw" in result
        assert "oof" in result["raw"]

    def test_evaluate_with_subset_metrics_filters_output(self) -> None:
        """evaluate(metrics=["rmse"]) returns only the requested metric."""
        m = Model(_reg_config_small())
        m.fit(data=_reg_df_small())
        result = m.evaluate(metrics=["rmse"])
        assert set(result["raw"]["oof"].keys()) == {"rmse"}
        assert set(result["raw"]["if_mean"].keys()) == {"rmse"}
        for fold in result["raw"]["if_per_fold"]:
            assert set(fold.keys()) == {"rmse"}

    def test_evaluate_subset_values_match_precomputed(self) -> None:
        """Filtered values must equal the pre-computed values from fit()."""
        m = Model(_reg_config_small())
        m.fit(data=_reg_df_small())
        full = m.evaluate()
        filtered = m.evaluate(metrics=["rmse"])
        assert filtered["raw"]["oof"]["rmse"] == pytest.approx(
            full["raw"]["oof"]["rmse"]
        )

    def test_evaluate_incompatible_metric_raises_unsupported(self) -> None:
        """evaluate() with a task-incompatible metric must raise UNSUPPORTED_METRIC."""
        m = Model(_bin_config_small())
        m.fit(data=_bin_df_small())
        with pytest.raises(LizyMLError) as exc_info:
            m.evaluate(metrics=["rmse"])  # rmse is not valid for binary
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_evaluate_incompatible_regression_metric_raises(self) -> None:
        """evaluate() with auc on regression must raise UNSUPPORTED_METRIC."""
        m = Model(_reg_config_small())
        m.fit(data=_reg_df_small())
        with pytest.raises(LizyMLError) as exc_info:
            m.evaluate(metrics=["auc"])  # auc is not valid for regression
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_evaluate_before_fit_raises_model_not_fit(self) -> None:
        m = Model(_reg_config_small())
        with pytest.raises(LizyMLError) as exc_info:
            m.evaluate()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT
