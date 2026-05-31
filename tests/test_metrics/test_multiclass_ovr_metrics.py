"""Tests for multiclass OvR metrics: AUC, AUC-PR, Brier (H-0018).

Degenerate-fold robustness (a validation fold missing a class) is covered by
``TestMulticlassMissingClass`` (issue #167).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.metrics.classification import AUC, AUCPR, Brier
from lizyml.metrics.registry import get_metrics_for_task


@pytest.fixture()
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    """3-class multiclass data with reasonable predictions."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = np.array([0] * 40 + [1] * 30 + [2] * 30)
    # Create decent predictions: high prob for correct class
    y_pred = rng.dirichlet([0.3, 0.3, 0.3], size=n)
    for i in range(n):
        y_pred[i, y_true[i]] += 1.0
    # Re-normalise
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return y_true, y_pred


class TestAUCMulticlassOvR:
    def test_returns_float_in_range(
        self, multiclass_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_data
        metric = AUC()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_good_predictions_high_auc(
        self, multiclass_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_data
        metric = AUC()
        result = metric(y_true, y_pred)
        assert result > 0.8

    def test_binary_unchanged(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
        metric = AUC()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestAUCPRMulticlassOvR:
    def test_returns_float_in_range(
        self, multiclass_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_data
        metric = AUCPR()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_binary_unchanged(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
        metric = AUCPR()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestBrierMulticlassOvR:
    def test_returns_nonnegative_float(
        self, multiclass_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_data
        metric = Brier()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_good_predictions_low_brier(
        self, multiclass_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_data
        metric = Brier()
        result = metric(y_true, y_pred)
        assert result < 0.3

    def test_binary_unchanged(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
        metric = Brier()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert result >= 0.0


class TestMulticlassRegistered:
    def test_auc_registered_for_multiclass(self) -> None:
        metrics = get_metrics_for_task(["auc"], "multiclass")
        assert len(metrics) == 1
        assert metrics[0].name == "auc"

    def test_auc_pr_registered_for_multiclass(self) -> None:
        metrics = get_metrics_for_task(["auc_pr"], "multiclass")
        assert len(metrics) == 1
        assert metrics[0].name == "auc_pr"

    def test_brier_registered_for_multiclass(self) -> None:
        metrics = get_metrics_for_task(["brier"], "multiclass")
        assert len(metrics) == 1
        assert metrics[0].name == "brier"

    def test_regression_excluded(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            get_metrics_for_task(["auc"], "regression")
        assert exc_info.value.code.value == "UNSUPPORTED_METRIC"


# y_pred has 3 columns but class 2 never occurs in y_true — the shape a CV
# fold takes when one class is absent from the validation slice.
_MISSING_CLASS_Y = np.array([0, 0, 0, 1, 1, 1])
_MISSING_CLASS_P = np.array(
    [
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
        [0.3, 0.6, 0.1],
    ]
)


class TestMulticlassMissingClass:
    """A validation fold missing a class must not crash (AUC) or silently fold
    a degenerate zero-filled column into the macro mean (AUC-PR / Brier)."""

    def test_auc_macro_over_present_classes_no_crash(self) -> None:
        # Pre-fix this raised a raw sklearn ValueError.
        result = AUC()(_MISSING_CLASS_Y, _MISSING_CLASS_P)
        expected = float(
            np.mean(
                [
                    roc_auc_score(
                        (k == _MISSING_CLASS_Y).astype(int), _MISSING_CLASS_P[:, k]
                    )
                    for k in (0, 1)
                ]
            )
        )
        assert result == pytest.approx(expected)

    def test_aucpr_excludes_absent_class(self) -> None:
        result = AUCPR()(_MISSING_CLASS_Y, _MISSING_CLASS_P)
        present_only = float(
            np.mean(
                [
                    average_precision_score(
                        (k == _MISSING_CLASS_Y).astype(int), _MISSING_CLASS_P[:, k]
                    )
                    for k in (0, 1)
                ]
            )
        )
        # Old behaviour averaged class 2's zero-filled column (AP == 0.0) too,
        # dragging the mean down; the fix must exclude it.
        include_absent = present_only * 2 / 3
        assert result == pytest.approx(present_only)
        assert result != pytest.approx(include_absent)

    def test_brier_excludes_absent_class(self) -> None:
        result = Brier()(_MISSING_CLASS_Y, _MISSING_CLASS_P)
        expected = float(
            np.mean(
                [
                    brier_score_loss(
                        (k == _MISSING_CLASS_Y).astype(int), _MISSING_CLASS_P[:, k]
                    )
                    for k in (0, 1)
                ]
            )
        )
        assert result == pytest.approx(expected)

    def test_auc_single_class_fold_raises_coded(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
        with pytest.raises(LizyMLError) as exc_info:
            AUC()(y_true, y_pred)
        err = exc_info.value
        assert err.code is ErrorCode.UNSUPPORTED_METRIC
        assert err.context["metric"] == "auc"
        assert err.context["n_present_classes"] == 1
        assert err.context["n_columns"] == 3

    def test_aucpr_single_class_fold_no_crash(self) -> None:
        # min_classes=1: AUC-PR / Brier stay defined on a single-class fold.
        y_true = np.array([0, 0, 0])
        y_pred = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
        assert isinstance(AUCPR()(y_true, y_pred), float)
        assert isinstance(Brier()(y_true, y_pred), float)

    def test_label_out_of_range_raises_coded(self) -> None:
        # y_true label 5 with only 3 prediction columns — would IndexError
        # pre-fix; now a coded error.
        y_true = np.array([0, 1, 5])
        y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])
        with pytest.raises(LizyMLError) as exc_info:
            AUC()(y_true, y_pred)
        assert exc_info.value.code is ErrorCode.UNSUPPORTED_METRIC

    def test_all_classes_present_matches_sklearn_macro(
        self, multiclass_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        # Refactor must preserve the happy-path number (regression guard).
        y_true, y_pred = multiclass_data
        expected = float(
            roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")
        )
        assert AUC()(y_true, y_pred) == pytest.approx(expected)
