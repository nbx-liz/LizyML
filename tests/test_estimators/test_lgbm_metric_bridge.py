"""Tests for H-0064 — LightGBM metric bridge (mapping, validation, feval).

Phase 1: Metric name mapping (LizyML → LightGBM)
Phase 2: Whitelist validation
Phase 3: feval custom function generation
"""

from __future__ import annotations

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
            (["huber", "fair", "r2"], "regression"),
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
        native, fevals = resolve_metrics(["auc", "binary_logloss"], "binary")
        assert native == ["auc", "binary_logloss"]
        assert fevals == []

    def test_all_feval_metrics(self) -> None:
        native, fevals = resolve_metrics(["f1"], "binary")
        assert native == []
        assert len(fevals) == 1

    def test_mixed_native_and_feval(self) -> None:
        native, fevals = resolve_metrics(["auc", "brier"], "binary")
        assert native == ["auc"]
        assert len(fevals) == 1

    def test_lizyml_name_translated_before_split(self) -> None:
        """logloss should be translated to binary_logloss and classified as native."""
        native, fevals = resolve_metrics(["logloss"], "binary")
        assert native == ["binary_logloss"]
        assert fevals == []

    def test_rmsle_regression(self) -> None:
        native, fevals = resolve_metrics(["rmsle"], "regression")
        assert native == []
        assert len(fevals) == 1

    def test_accuracy_treated_as_feval(self) -> None:
        """accuracy has no LightGBM equivalent — should go through feval."""
        native, fevals = resolve_metrics(["accuracy"], "binary")
        assert native == []
        assert len(fevals) == 1

    def test_invalid_metric_for_task_raises(self) -> None:
        """rmsle is not valid for binary task."""
        with pytest.raises(LizyMLError):
            resolve_metrics(["rmsle"], "binary")

    def test_feval_returns_correct_tuple_format(self) -> None:
        """feval callable should return (name, value, is_higher_better)."""
        _, fevals = resolve_metrics(["f1"], "binary")
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
        _, fevals = resolve_metrics(["brier"], "binary")
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
        _, fevals = resolve_metrics(["rmsle"], "regression")
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
        _, fevals = resolve_metrics(["ece"], "binary")
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
        _, fevals = resolve_metrics(["f1"], "multiclass", num_class=3)
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
        _, fevals = resolve_metrics(["precision_at_k"], "binary")
        assert len(fevals) == 1


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
            native, fevals = resolve_metrics(
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
