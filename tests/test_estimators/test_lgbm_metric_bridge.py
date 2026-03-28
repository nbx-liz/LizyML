"""Tests for H-0064 — LightGBM metric bridge (mapping, validation, feval).

Phase 1: Metric name mapping (LizyML → LightGBM)
Phase 2: Whitelist validation
Phase 3: feval custom function generation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError

# These will be implemented in metric_bridge.py
from lizyml.estimators.lgbm.metric_bridge import (
    resolve_metrics,
    translate_metric,
    validate_lgbm_metrics,
)

# ============================================================================
# Phase 1: Metric name mapping
# ============================================================================


class TestTranslateMetric:
    """LizyML metric name → LightGBM metric name translation."""

    def test_logloss_binary(self) -> None:
        assert translate_metric("logloss", "binary") == "binary_logloss"

    def test_logloss_multiclass(self) -> None:
        assert translate_metric("logloss", "multiclass") == "multi_logloss"

    def test_logloss_regression_passthrough(self) -> None:
        """logloss has no regression mapping — should pass through."""
        assert translate_metric("logloss", "regression") == "logloss"

    def test_auc_pr_binary(self) -> None:
        assert translate_metric("auc_pr", "binary") == "average_precision"

    def test_auc_pr_multiclass(self) -> None:
        assert translate_metric("auc_pr", "multiclass") == "average_precision"

    def test_auc_passthrough(self) -> None:
        """auc has no mapping — same name in both systems."""
        assert translate_metric("auc", "binary") == "auc"

    def test_rmse_passthrough(self) -> None:
        assert translate_metric("rmse", "regression") == "rmse"

    def test_mae_passthrough(self) -> None:
        """mae is an alias for l1 in LightGBM — no translation needed."""
        assert translate_metric("mae", "regression") == "mae"

    def test_accuracy_no_translation(self) -> None:
        """accuracy should NOT be translated (semantic inversion with error)."""
        assert translate_metric("accuracy", "binary") == "accuracy"

    def test_unknown_metric_passthrough(self) -> None:
        """Unknown metrics pass through unchanged (validated later)."""
        assert translate_metric("some_custom_thing", "binary") == "some_custom_thing"


# ============================================================================
# Phase 2: Whitelist validation
# ============================================================================


class TestValidateLgbmMetrics:
    """Pre-validation of metric names against LightGBM native whitelist."""

    # -- Valid native metrics --

    @pytest.mark.parametrize(
        "metrics,task",
        [
            (["rmse"], "regression"),
            (["l1", "l2", "mape"], "regression"),
            (["huber", "fair"], "regression"),
            (["binary_logloss", "auc"], "binary"),
            (["average_precision"], "binary"),
            (["multi_logloss", "auc_mu"], "multiclass"),
            (["auc"], "multiclass"),
        ],
    )
    def test_valid_native_metrics_pass(self, metrics: list[str], task: str) -> None:
        validate_lgbm_metrics(metrics, task, feval_names=frozenset())

    # -- Invalid metrics --

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            validate_lgbm_metrics(["totally_bogus"], "binary", feval_names=frozenset())
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID
        assert "totally_bogus" in exc_info.value.user_message

    def test_task_mismatch_raises(self) -> None:
        """binary_logloss is not valid for regression."""
        with pytest.raises(LizyMLError) as exc_info:
            validate_lgbm_metrics(
                ["binary_logloss"], "regression", feval_names=frozenset()
            )
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_regression_metric_invalid_for_binary(self) -> None:
        with pytest.raises(LizyMLError):
            validate_lgbm_metrics(["rmse"], "binary", feval_names=frozenset())

    # -- feval bypass --

    def test_feval_metric_bypasses_validation(self) -> None:
        """Metrics in feval_names should pass even if not in native whitelist."""
        validate_lgbm_metrics(["f1"], "binary", feval_names=frozenset(["f1"]))

    def test_mixed_native_and_feval(self) -> None:
        """Mix of native and feval metrics should pass."""
        validate_lgbm_metrics(
            ["auc", "f1", "brier"],
            "binary",
            feval_names=frozenset(["f1", "brier"]),
        )

    # -- Error message quality --

    def test_error_message_lists_valid_options(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            validate_lgbm_metrics(
                ["invalid_xyz"], "regression", feval_names=frozenset()
            )
        msg = exc_info.value.user_message
        assert "rmse" in msg  # Should list valid options


# ============================================================================
# Phase 3: feval custom function generation
# ============================================================================


class TestResolveMetrics:
    """Split user metric list into native LightGBM names + feval callables."""

    def test_all_native_metrics(self) -> None:
        native, fevals, _ = resolve_metrics(["auc", "binary_logloss"], "binary")
        assert native == ["auc", "binary_logloss"]
        assert fevals == []

    def test_all_feval_metrics(self) -> None:
        native, fevals, _ = resolve_metrics(["f1"], "binary")
        assert native == []
        assert len(fevals) == 1

    def test_mixed_native_and_feval(self) -> None:
        native, fevals, _ = resolve_metrics(["auc", "brier"], "binary")
        assert native == ["auc"]
        assert len(fevals) == 1

    def test_lizyml_name_translated_before_split(self) -> None:
        """logloss should be translated to binary_logloss and classified as native."""
        native, fevals, _ = resolve_metrics(["logloss"], "binary")
        assert native == ["binary_logloss"]
        assert fevals == []

    def test_rmsle_regression(self) -> None:
        native, fevals, _ = resolve_metrics(["rmsle"], "regression")
        assert native == []
        assert len(fevals) == 1

    def test_accuracy_treated_as_feval(self) -> None:
        """accuracy has no LightGBM equivalent — should go through feval."""
        native, fevals, _ = resolve_metrics(["accuracy"], "binary")
        assert native == []
        assert len(fevals) == 1

    def test_invalid_metric_for_task_raises(self) -> None:
        """rmsle is not valid for binary task."""
        with pytest.raises(LizyMLError):
            resolve_metrics(["rmsle"], "binary")

    def test_feval_returns_correct_tuple_format(self) -> None:
        """feval callable should return (name, value, is_higher_better)."""
        _, fevals, _ = resolve_metrics(["f1"], "binary")
        feval_fn = fevals[0]
        # Create a mock dataset
        import lightgbm as lgb

        y_true = np.array([0, 1, 1, 0, 1], dtype=np.float64)
        # For binary, y_pred from LightGBM is raw logits
        # logits > 0 → prob > 0.5 → predict 1
        y_pred = np.array([-2.0, 2.0, 2.0, -2.0, 2.0])
        dataset = lgb.Dataset(np.zeros((5, 1)), label=y_true, free_raw_data=False)
        dataset.construct()

        result = feval_fn(y_pred, dataset)
        assert isinstance(result, tuple)
        assert len(result) == 3
        name, value, is_higher = result
        assert name == "f1"
        assert isinstance(value, float)
        assert is_higher is True  # F1 is greater_is_better
        assert value == pytest.approx(1.0, abs=0.01)  # Perfect predictions

    def test_feval_brier_binary(self) -> None:
        """Brier feval for binary should sigmoid raw logits."""
        _, fevals, _ = resolve_metrics(["brier"], "binary")
        feval_fn = fevals[0]
        import lightgbm as lgb

        y_true = np.array([0, 1, 1, 0], dtype=np.float64)
        # logits close to 0 → prob ≈ 0.5 → Brier ≈ 0.25
        y_pred = np.array([0.0, 0.0, 0.0, 0.0])
        dataset = lgb.Dataset(np.zeros((4, 1)), label=y_true, free_raw_data=False)
        dataset.construct()

        name, value, is_higher = feval_fn(y_pred, dataset)
        assert name == "brier"
        assert is_higher is False  # Brier: lower is better
        assert value == pytest.approx(0.25, abs=0.01)

    def test_feval_rmsle_regression(self) -> None:
        """RMSLE feval for regression (no logit transform)."""
        _, fevals, _ = resolve_metrics(["rmsle"], "regression")
        feval_fn = fevals[0]
        import lightgbm as lgb

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])  # Perfect predictions
        dataset = lgb.Dataset(np.zeros((3, 1)), label=y_true, free_raw_data=False)
        dataset.construct()

        name, value, is_higher = feval_fn(y_pred, dataset)
        assert name == "rmsle"
        assert is_higher is False
        assert value == pytest.approx(0.0, abs=1e-6)

    def test_feval_ece_binary(self) -> None:
        """ECE feval for binary."""
        _, fevals, _ = resolve_metrics(["ece"], "binary")
        assert len(fevals) == 1
        feval_fn = fevals[0]
        import lightgbm as lgb

        y_true = np.array([1, 1, 1, 0, 0], dtype=np.float64)
        # High confidence logits: sigmoid(5) ≈ 0.993, sigmoid(-5) ≈ 0.007
        y_pred = np.array([5.0, 5.0, 5.0, -5.0, -5.0])
        dataset = lgb.Dataset(np.zeros((5, 1)), label=y_true, free_raw_data=False)
        dataset.construct()

        name, value, is_higher = feval_fn(y_pred, dataset)
        assert name == "ece"
        assert is_higher is False
        assert isinstance(value, float)
        # ECE should be finite and non-negative
        assert 0.0 <= value <= 1.0

    def test_feval_multiclass_f1(self) -> None:
        """F1 feval for multiclass — y_pred is flattened (n * k)."""
        _, fevals, _ = resolve_metrics(["f1"], "multiclass", num_class=3)
        feval_fn = fevals[0]
        import lightgbm as lgb

        n = 6
        y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.float64)
        # Flattened softmax-pre logits: (n * n_classes,)
        # Each row: high logit for correct class
        logits = np.zeros(n * 3)
        for i, c in enumerate(y_true.astype(int)):
            logits[i * 3 + c] = 5.0  # Strong signal for correct class
        dataset = lgb.Dataset(np.zeros((n, 1)), label=y_true, free_raw_data=False)
        dataset.construct()

        name, value, is_higher = feval_fn(logits, dataset)
        assert name == "f1"
        assert is_higher is True
        assert value == pytest.approx(1.0, abs=0.01)

    def test_feval_precision_at_k_binary(self) -> None:
        """PrecisionAtK feval for binary."""
        _, fevals, _ = resolve_metrics(["precision_at_k"], "binary")
        assert len(fevals) == 1


# ============================================================================
# Phase 3b: Numerical correctness — feval vs direct metric computation
# ============================================================================


class TestFevalNumericalCorrectness:
    """Verify feval values match direct BaseMetric computation.

    For each metric, we:
    1. Create known raw LightGBM predictions (logits for binary)
    2. Run the feval callable
    3. Manually apply the same transform (sigmoid/softmax) and call the metric
    4. Assert the values match exactly
    """

    def _make_binary_dataset(
        self, y_true: list[float], logits: list[float]
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        import lightgbm as lgb

        yt = np.array(y_true, dtype=np.float64)
        yp = np.array(logits, dtype=np.float64)
        ds = lgb.Dataset(np.zeros((len(yt), 1)), label=yt, free_raw_data=False)
        ds.construct()
        return yt, yp, ds

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    # -- F1 (binary) --

    def test_f1_binary_numerical(self) -> None:
        """F1: logits → sigmoid → threshold 0.5 → f1_score."""
        from sklearn.metrics import f1_score

        _, fevals, _ = resolve_metrics(["f1"], "binary")
        y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        # sigmoid(1.0)=0.731, sigmoid(-1.0)=0.269, sigmoid(0.2)=0.550
        logits = [1.0, -1.0, 0.2, 2.0, -2.0, 0.5, -0.5, -1.5]
        yt, yp, ds = self._make_binary_dataset(y_true, logits)

        name, feval_val, is_higher = fevals[0](yp, ds)

        # Manual: sigmoid → threshold
        proba = self._sigmoid(yp)
        pred_labels = (proba >= 0.5).astype(int)
        expected = f1_score(yt, pred_labels, zero_division=0)

        assert name == "f1"
        assert is_higher is True
        assert feval_val == pytest.approx(expected, abs=1e-10)

    # -- Accuracy (binary) --

    def test_accuracy_binary_numerical(self) -> None:
        """Accuracy: logits → sigmoid → threshold 0.5 → accuracy_score."""
        from sklearn.metrics import accuracy_score

        _, fevals, _ = resolve_metrics(["accuracy"], "binary")
        y_true = [1, 0, 1, 1, 0, 0]
        logits = [2.0, -2.0, -0.1, 0.1, 0.5, -0.5]
        yt, yp, ds = self._make_binary_dataset(y_true, logits)

        name, feval_val, is_higher = fevals[0](yp, ds)

        proba = self._sigmoid(yp)
        pred_labels = (proba >= 0.5).astype(int)
        expected = accuracy_score(yt, pred_labels)

        assert name == "accuracy"
        assert is_higher is True
        assert feval_val == pytest.approx(expected, abs=1e-10)

    # -- Brier (binary) --

    def test_brier_binary_numerical(self) -> None:
        """Brier: logits → sigmoid → brier_score_loss."""
        from sklearn.metrics import brier_score_loss

        _, fevals, _ = resolve_metrics(["brier"], "binary")
        y_true = [1, 0, 1, 0, 1]
        logits = [0.5, -0.5, 1.5, -1.5, 0.0]
        yt, yp, ds = self._make_binary_dataset(y_true, logits)

        name, feval_val, is_higher = fevals[0](yp, ds)

        proba = self._sigmoid(yp)
        expected = brier_score_loss(yt, proba)

        assert name == "brier"
        assert is_higher is False
        assert feval_val == pytest.approx(expected, abs=1e-10)

    # -- LogLoss (binary, via feval path check — logloss maps to native) --

    # -- ECE (binary) --

    def test_ece_binary_numerical(self) -> None:
        """ECE: logits → sigmoid → equal-width bins → weighted |acc - conf|."""
        from lizyml.metrics.classification import ECE

        _, fevals, _ = resolve_metrics(["ece"], "binary")
        y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
        logits = [2.0, -2.0, 1.0, 0.5, -0.5, 3.0, -3.0, -1.0, 0.1, -0.1]
        yt, yp, ds = self._make_binary_dataset(y_true, logits)

        name, feval_val, is_higher = fevals[0](yp, ds)

        proba = self._sigmoid(yp)
        expected = ECE()(yt, proba)

        assert name == "ece"
        assert is_higher is False
        assert feval_val == pytest.approx(expected, abs=1e-10)

    # -- PrecisionAtK (binary) --

    def test_precision_at_k_binary_numerical(self) -> None:
        """PrecisionAtK: logits → sigmoid → top-K% → precision."""
        from lizyml.metrics.classification import PrecisionAtK

        _, fevals, _ = resolve_metrics(["precision_at_k"], "binary")
        y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        logits = [3.0, 2.5, 2.0, 1.5, 1.0, -1.0, -1.5, -2.0, -2.5, -3.0]
        yt, yp, ds = self._make_binary_dataset(y_true, logits)

        name, feval_val, is_higher = fevals[0](yp, ds)

        proba = self._sigmoid(yp)
        expected = PrecisionAtK(k=10)(np.array(yt), proba)

        assert name == "precision_at_k"
        assert is_higher is True
        assert feval_val == pytest.approx(expected, abs=1e-10)

    # -- RMSLE (regression) --

    def test_rmsle_regression_numerical(self) -> None:
        """RMSLE: no transform, direct log1p RMSE."""
        _, fevals, _ = resolve_metrics(["rmsle"], "regression")
        import lightgbm as lgb

        y_true = np.array([3.0, 5.0, 2.5, 8.0])
        y_pred = np.array([2.5, 5.5, 2.0, 7.0])
        ds = lgb.Dataset(np.zeros((4, 1)), label=y_true, free_raw_data=False)
        ds.construct()

        name, feval_val, is_higher = fevals[0](y_pred, ds)

        # Manual RMSLE — use label from dataset to account for
        # LightGBM's float32 internal storage precision
        ds_label = np.asarray(ds.get_label())
        expected = float(np.sqrt(np.mean((np.log1p(ds_label) - np.log1p(y_pred)) ** 2)))

        assert name == "rmsle"
        assert is_higher is False
        assert feval_val == pytest.approx(expected, abs=1e-10)

    # -- F1 (multiclass) --

    def test_f1_multiclass_numerical(self) -> None:
        """F1 multiclass: flattened logits → reshape → softmax → argmax → f1."""
        from sklearn.metrics import f1_score

        _, fevals, _ = resolve_metrics(["f1"], "multiclass", num_class=3)
        import lightgbm as lgb

        y_true = np.array([0, 1, 2, 1, 0, 2], dtype=np.float64)
        n = len(y_true)
        # Build flattened logits: (n * 3,) — row-major
        logits = np.array(
            [
                3.0,
                0.1,
                0.1,  # sample 0: class 0 highest
                0.1,
                2.0,
                0.5,  # sample 1: class 1 highest
                0.2,
                0.3,
                4.0,  # sample 2: class 2 highest
                0.1,
                1.0,
                0.5,  # sample 3: class 1 highest
                2.0,
                0.1,
                0.1,  # sample 4: class 0 highest
                0.5,
                0.5,
                0.1,  # sample 5: class 0 highest (wrong! true=2)
            ]
        )
        ds = lgb.Dataset(np.zeros((n, 1)), label=y_true, free_raw_data=False)
        ds.construct()

        name, feval_val, is_higher = fevals[0](logits, ds)

        # Manual: reshape → softmax → argmax
        reshaped = logits.reshape(-1, 3)
        e_x = np.exp(reshaped - reshaped.max(axis=1, keepdims=True))
        proba = e_x / e_x.sum(axis=1, keepdims=True)
        pred_labels = proba.argmax(axis=1)
        expected = f1_score(y_true, pred_labels, average="macro", zero_division=0)

        assert name == "f1"
        assert is_higher is True
        assert feval_val == pytest.approx(expected, abs=1e-10)

    # -- Brier (multiclass) --

    def test_brier_multiclass_numerical(self) -> None:
        """Brier multiclass: flattened logits → reshape → softmax → per-class brier."""
        from sklearn.metrics import brier_score_loss
        from sklearn.preprocessing import label_binarize

        _, fevals, _ = resolve_metrics(["brier"], "multiclass", num_class=3)
        import lightgbm as lgb

        y_true = np.array([0, 1, 2, 0], dtype=np.float64)
        n = len(y_true)
        logits = np.array(
            [
                3.0,
                0.1,
                0.1,
                0.1,
                3.0,
                0.1,
                0.1,
                0.1,
                3.0,
                0.1,
                0.1,
                3.0,  # wrong: true=0, pred=2
            ]
        )
        ds = lgb.Dataset(np.zeros((n, 1)), label=y_true, free_raw_data=False)
        ds.construct()

        name, feval_val, is_higher = fevals[0](logits, ds)

        # Manual
        reshaped = logits.reshape(-1, 3)
        e_x = np.exp(reshaped - reshaped.max(axis=1, keepdims=True))
        proba = e_x / e_x.sum(axis=1, keepdims=True)
        classes = np.arange(3)
        y_bin = label_binarize(y_true.astype(int), classes=classes)
        per_class = [brier_score_loss(y_bin[:, k], proba[:, k]) for k in range(3)]
        expected = float(np.mean(per_class))

        assert name == "brier"
        assert is_higher is False
        assert feval_val == pytest.approx(expected, abs=1e-10)

    # -- Accuracy (multiclass) --

    def test_accuracy_multiclass_numerical(self) -> None:
        """Accuracy multiclass: reshape → softmax → argmax → accuracy."""
        from sklearn.metrics import accuracy_score

        _, fevals, _ = resolve_metrics(["accuracy"], "multiclass", num_class=3)
        import lightgbm as lgb

        y_true = np.array([0, 1, 2, 1], dtype=np.float64)
        logits = np.array(
            [
                3.0,
                0.1,
                0.1,  # correct
                0.1,
                3.0,
                0.1,  # correct
                0.1,
                0.1,
                3.0,  # correct
                0.1,
                3.0,
                0.1,  # correct
            ]
        )
        ds = lgb.Dataset(np.zeros((4, 1)), label=y_true, free_raw_data=False)
        ds.construct()

        name, feval_val, is_higher = fevals[0](logits, ds)

        reshaped = logits.reshape(-1, 3)
        e_x = np.exp(reshaped - reshaped.max(axis=1, keepdims=True))
        proba = e_x / e_x.sum(axis=1, keepdims=True)
        pred_labels = proba.argmax(axis=1)
        expected = accuracy_score(y_true, pred_labels)

        assert name == "accuracy"
        assert is_higher is True
        assert feval_val == pytest.approx(expected, abs=1e-10)

    # -- Sigmoid edge cases --

    def test_sigmoid_extreme_logits(self) -> None:
        """Extreme logits should not cause overflow or NaN."""
        _, fevals, _ = resolve_metrics(["brier"], "binary")
        import lightgbm as lgb

        y_true = np.array([1, 0, 1, 0], dtype=np.float64)
        logits = np.array([1000.0, -1000.0, 500.0, -500.0])
        ds = lgb.Dataset(np.zeros((4, 1)), label=y_true, free_raw_data=False)
        ds.construct()

        name, feval_val, is_higher = fevals[0](logits, ds)

        # sigmoid(1000) ≈ 1.0, sigmoid(-1000) ≈ 0.0
        # Brier for perfect predictions ≈ 0.0
        assert not np.isnan(feval_val)
        assert not np.isinf(feval_val)
        assert feval_val == pytest.approx(0.0, abs=1e-6)


# ============================================================================
# Integration: resolve_metrics validation
# ============================================================================


class TestResolveMetricsValidation:
    """resolve_metrics should validate after mapping + splitting."""

    def test_completely_invalid_metric_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            resolve_metrics(["nonexistent_metric_xyz"], "binary")
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_task_incompatible_feval_raises(self) -> None:
        """ece is only valid for binary, not regression."""
        with pytest.raises(LizyMLError):
            resolve_metrics(["ece"], "regression")

    def test_all_tasks_with_default_metrics(self) -> None:
        """Default task metrics should always resolve without error."""
        from lizyml.estimators.lgbm.defaults import _TASK_METRIC

        for task, metrics in _TASK_METRIC.items():
            native, fevals, _ = resolve_metrics(
                metrics, task, num_class=3 if task == "multiclass" else None
            )
            assert len(native) + len(fevals) == len(metrics)

    def test_duplicate_metric_raises(self) -> None:
        """Duplicate metric names should be rejected."""
        with pytest.raises(LizyMLError) as exc_info:
            resolve_metrics(["auc", "auc"], "binary")
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID
        assert "Duplicate" in exc_info.value.user_message

    def test_duplicate_feval_metric_raises(self) -> None:
        """Duplicate feval metric names should also be rejected."""
        with pytest.raises(LizyMLError):
            resolve_metrics(["f1", "f1"], "binary")


# ============================================================================
# R2 feval migration (LightGBM 4.6.0 does not support r2 natively)
# ============================================================================


class TestR2FevalMigration:
    """Verify r2 is handled as feval, not native, for LightGBM 4.6.0.

    LightGBM's C++ binary (v4.6.0) does not implement r2 as a native metric
    despite it being documented in the master branch. Passing metric='r2' to
    lgb.train() silently produces empty eval_results, breaking early stopping.
    """

    def test_r2_not_in_native_whitelist(self) -> None:
        """r2 must NOT be in the native LightGBM metric whitelist."""
        from lizyml.estimators.lgbm.metric_bridge import _LGBM_NATIVE_METRICS

        assert "r2" not in _LGBM_NATIVE_METRICS["regression"]

    def test_r2_in_feval_metrics(self) -> None:
        """r2 must be in the feval metric set for regression."""
        from lizyml.estimators.lgbm.metric_bridge import _FEVAL_METRICS

        assert "r2" in _FEVAL_METRICS["regression"]

    def test_r2_not_valid_for_binary(self) -> None:
        """r2 is regression-only; should raise for binary."""
        with pytest.raises(LizyMLError) as exc_info:
            resolve_metrics(["r2"], "binary")
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_r2_not_valid_for_multiclass(self) -> None:
        """r2 is regression-only; should raise for multiclass."""
        with pytest.raises(LizyMLError):
            resolve_metrics(["r2"], "multiclass")

    def test_resolve_r2_returns_feval(self) -> None:
        """resolve_metrics('r2', 'regression') must return feval, not native."""
        native, fevals, _ = resolve_metrics(["r2"], "regression")
        assert native == []
        assert len(fevals) == 1

    def test_r2_feval_numerical_correctness(self) -> None:
        """R2 feval must match direct BaseMetric computation."""
        import lightgbm as lgb

        from lizyml.metrics.regression import R2

        _, fevals, _ = resolve_metrics(["r2"], "regression")
        feval_fn = fevals[0]

        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        ds = lgb.Dataset(np.zeros((4, 1)), label=y_true, free_raw_data=False)
        ds.construct()

        name, value, is_higher = feval_fn(y_pred, ds)

        # Use label from dataset to match float32 precision (cf. test_rmsle)
        ds_label = np.asarray(ds.get_label())
        expected = R2()(ds_label, y_pred)
        assert name == "r2"
        assert is_higher is True
        assert value == pytest.approx(expected, abs=1e-10)

    def test_r2_feval_perfect_predictions(self) -> None:
        """R2 feval must return 1.0 for perfect predictions."""
        import lightgbm as lgb

        _, fevals, _ = resolve_metrics(["r2"], "regression")
        feval_fn = fevals[0]

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ds = lgb.Dataset(np.zeros((5, 1)), label=y, free_raw_data=False)
        ds.construct()

        _, value, _ = feval_fn(y, ds)
        assert value == pytest.approx(1.0)

    def test_r2_feval_constant_y_true(self) -> None:
        """R2 with constant y_true (ss_tot == 0): ss_res == 0 → 1.0, else → 0.0."""
        import lightgbm as lgb

        _, fevals, _ = resolve_metrics(["r2"], "regression")
        feval_fn = fevals[0]

        # ss_res == 0 (perfect predictions on constant target) → 1.0
        y = np.array([5.0, 5.0, 5.0])
        ds = lgb.Dataset(np.zeros((3, 1)), label=y, free_raw_data=False)
        ds.construct()
        _, value, _ = feval_fn(y, ds)
        assert value == pytest.approx(1.0)

        # ss_res != 0 (imperfect predictions on constant target) → 0.0
        ds2 = lgb.Dataset(np.zeros((3, 1)), label=y, free_raw_data=False)
        ds2.construct()
        _, value2, _ = feval_fn(np.array([5.0, 5.0, 6.0]), ds2)
        assert value2 == pytest.approx(0.0)

    def test_r2_mixed_with_native_metric(self) -> None:
        """r2 + rmse: rmse should be native, r2 should be feval."""
        native, fevals, _ = resolve_metrics(["rmse", "r2"], "regression")
        assert native == ["rmse"]
        assert len(fevals) == 1

    def test_r2_validate_lgbm_metrics_rejects(self) -> None:
        """r2 without feval bypass must be rejected by validate_lgbm_metrics."""
        with pytest.raises(LizyMLError):
            validate_lgbm_metrics(["r2"], "regression", feval_names=frozenset())
