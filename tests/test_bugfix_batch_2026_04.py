"""Regression tests for bugs discovered 2026-04-10.

BUG-1: ECE uses binarized-prediction accuracy instead of fraction-of-positives.
BUG-2: confusion_matrix_table includes NaN-covered rows (TimeSeriesCV).
BUG-3: validate_no_target_leakage short-circuit order allows silent bypass.
BUG-4: IsotonicCalibrator uses log_evaluation(period=0) instead of period=-1.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.data.validators import validate_no_target_leakage
from lizyml.evaluation.confusion import confusion_matrix_table
from lizyml.metrics.classification import ECE

# ---------------------------------------------------------------------------
# BUG-1: ECE accuracy formula
# ---------------------------------------------------------------------------


class TestECEFormula:
    """ECE 'accuracy' in each bin must be fraction-of-positives, not
    binarized-prediction accuracy."""

    def test_low_confidence_bin_well_calibrated(self) -> None:
        """All predictions in [0.1, 0.2].  With true positive rate ~15%,
        a well-calibrated model should yield low ECE.

        Old (buggy) code: binarize at 0.5 -> all predict 0 -> acc ~ 0.85
        -> |0.85 - 0.15| = 0.70 per bin.
        Correct code: acc = mean(y_true) ~ 0.15 -> |0.15 - 0.15| ~ 0.0.
        """
        rng = np.random.default_rng(42)
        n = 2000
        y_pred = rng.uniform(0.1, 0.2, size=n)
        # Generate labels matching the predicted probabilities (well-calibrated)
        y_true = rng.binomial(1, y_pred).astype(float)
        ece = ECE(n_bins=10)(y_true, y_pred)
        # Well-calibrated predictions should have small ECE
        assert ece < 0.05, (
            f"ECE={ece:.4f} is too high for well-calibrated "
            "low-confidence predictions. This suggests the "
            "formula uses binarized accuracy instead of "
            "fraction-of-positives."
        )

    def test_high_confidence_bin_poorly_calibrated(self) -> None:
        """All predictions ~0.9 but true positive rate ~0.5.
        ECE should be ~|0.5 - 0.9| = 0.4, not ~|0.5 - 0.9|."""
        n = 1000
        y_pred = np.full(n, 0.9)
        rng = np.random.default_rng(0)
        y_true = rng.binomial(1, 0.5, size=n).astype(float)
        ece = ECE(n_bins=10)(y_true, y_pred)
        # Should reflect the miscalibration: |mean(y_true) - mean(y_pred)| ~ 0.4
        assert 0.3 < ece < 0.5, f"ECE={ece:.4f}, expected ~0.4"

    def test_deterministic_hand_calculated(self) -> None:
        """Hand-calculated example with 1 bin that spans the full range.

        y_pred = [0.2, 0.3, 0.7, 0.8], y_true = [0, 0, 1, 1]
        Correct ECE (1 bin): |mean(y_true) - mean(y_pred)| = |0.5 - 0.5| = 0.0
        Buggy ECE (1 bin): binarize=[0,0,1,1], acc=mean([0==0,0==0,1==1,1==1])=1.0
                           -> |1.0 - 0.5| = 0.5
        """
        y_pred = np.array([0.2, 0.3, 0.7, 0.8])
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        ece = ECE(n_bins=1)(y_true, y_pred)
        assert ece == pytest.approx(0.0, abs=1e-10), (
            f"ECE={ece:.4f} with 1 bin, perfectly calibrated data should be 0.0"
        )


# ---------------------------------------------------------------------------
# BUG-2: confusion_matrix_table with NaN OOF rows
# ---------------------------------------------------------------------------


class TestConfusionMatrixNaNCoverage:
    """confusion_matrix_table must exclude structurally uncovered rows."""

    @staticmethod
    def _make_fit_result_with_nan_oof(
        n_samples: int,
        nan_indices: list[int],
    ) -> MagicMock:
        """Create a minimal FitResult-like object with NaN in oof_pred."""
        oof_pred = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1], dtype=np.float64)
        for i in nan_indices:
            oof_pred[i] = np.nan

        # Outer splits: rows 0,1 are never in valid (simulating TimeSeriesCV)
        outer_splits = [
            (np.array([0, 1], dtype=np.intp), np.array([2, 3], dtype=np.intp)),
            (np.array([0, 1, 2, 3], dtype=np.intp), np.array([4, 5], dtype=np.intp)),
        ]

        if_pred_fold0 = np.array([0.6, 0.4], dtype=np.float64)
        if_pred_fold1 = np.array([0.7, 0.3, 0.8, 0.2], dtype=np.float64)

        splits = MagicMock()
        splits.outer = outer_splits

        fr = MagicMock()
        fr.oof_pred = oof_pred
        fr.splits = splits
        fr.if_pred_per_fold = [if_pred_fold0, if_pred_fold1]
        return fr

    def test_nan_rows_excluded_from_oos_confusion_matrix(self) -> None:
        """Rows with NaN OOF predictions must not appear in the OOS matrix."""
        fr = self._make_fit_result_with_nan_oof(6, nan_indices=[0, 1])
        y_true = np.array([1, 0, 1, 0, 1, 0])

        result = confusion_matrix_table(fr, y_true, threshold=0.5, task="binary")
        cm_oos = result["oos"].to_numpy()

        # Only 4 rows (indices 2-5) should be counted
        assert cm_oos.sum() == 4, (
            f"OOS confusion matrix counted {cm_oos.sum()} rows, expected 4 "
            "(rows 0,1 have NaN OOF and should be excluded)"
        )


# ---------------------------------------------------------------------------
# BUG-3: validate_no_target_leakage short-circuit order
# ---------------------------------------------------------------------------


class TestLeakageValidatorNaNOrder:
    """np.allclose must not be called before NaN-position guard."""

    def test_no_internal_exception_with_different_nan_positions(self) -> None:
        """Old code called np.allclose(dropna(), dropna()) before checking
        NaN positions, causing ValueError (mismatched lengths) that was
        silently swallowed.  After fix, isna().equals() short-circuits
        first, so np.allclose is never called on mismatched arrays.

        We verify that no ValueError is raised internally by patching
        np.allclose to detect if it receives mismatched-length arrays.
        """
        from unittest.mock import patch

        df = pd.DataFrame(
            {
                "target": [1.0, 2.0, 3.0, np.nan, 5.0],
                "tricky": [1.0, 2.0, np.nan, 4.0, 5.0],
            }
        )
        original_allclose = np.allclose

        def guarded_allclose(*args: object, **kwargs: object) -> bool:
            a, b = args[0], args[1]
            if hasattr(a, "__len__") and hasattr(b, "__len__"):
                assert len(a) == len(b), (  # type: ignore[arg-type]
                    "np.allclose called with mismatched lengths "
                    f"({len(a)} vs {len(b)}). "  # type: ignore[arg-type]
                    "isna().equals() should have short-circuited."
                )
            return original_allclose(*args, **kwargs)

        with patch("lizyml.data.validators.np.allclose", guarded_allclose):
            result = validate_no_target_leakage(df, "target", raise_on_violation=False)
        assert result == [], (
            "Columns with different NaN positions should not be flagged as leakage"
        )

    def test_real_leakage_with_matching_nan_still_detected(self) -> None:
        """Column is a true copy of target (same NaN positions).
        Must still be detected after the fix."""
        df = pd.DataFrame(
            {
                "target": [1.0, 2.0, np.nan, 4.0, 5.0],
                "leak": [1.0, 2.0, np.nan, 4.0, 5.0],
            }
        )
        with pytest.raises(LizyMLError) as exc_info:
            validate_no_target_leakage(df, "target", raise_on_violation=True)
        assert exc_info.value.code == ErrorCode.LEAKAGE_SUSPECTED


# ---------------------------------------------------------------------------
# BUG-4: IsotonicCalibrator log_evaluation period
# ---------------------------------------------------------------------------


class TestIsotonicLogEvaluationPeriod:
    """log_evaluation(period=-1) is the correct way to silence LightGBM logs."""

    def test_period_is_negative_one(self) -> None:
        """The isotonic calibrator must call log_evaluation(period=-1)."""
        from unittest.mock import patch

        from lizyml.calibration.isotonic import IsotonicCalibrator

        cal = IsotonicCalibrator()
        X = np.array([[0.1], [0.2], [0.3]])
        y = np.array([0.0, 1.0, 0.0])
        params: dict[str, object] = {
            "objective": "binary",
            "verbose": -1,
        }

        with patch("lizyml.calibration.isotonic.lgbm.log_evaluation") as mock_log:
            mock_log.return_value = lambda *a, **kw: None
            cal._prepare_training(X, y, len(y), params)
            mock_log.assert_called_once_with(period=-1)
