"""Regression test: confusion_matrix_table must exclude NaN-covered rows (BUG-2).

TimeSeriesCV leaves some rows structurally uncovered (NaN OOF); those rows must
not be counted in the OOS confusion matrix.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from lizyml.evaluation.confusion import confusion_matrix_table


class TestConfusionMatrixNaNCoverage:
    @staticmethod
    def _make_fit_result_with_nan_oof(nan_indices: list[int]) -> MagicMock:
        oof_pred = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1], dtype=np.float64)
        for i in nan_indices:
            oof_pred[i] = np.nan
        outer_splits = [
            (np.array([0, 1], dtype=np.intp), np.array([2, 3], dtype=np.intp)),
            (np.array([0, 1, 2, 3], dtype=np.intp), np.array([4, 5], dtype=np.intp)),
        ]
        splits = MagicMock()
        splits.outer = outer_splits
        fr = MagicMock()
        fr.oof_pred = oof_pred
        fr.splits = splits
        fr.if_pred_per_fold = [
            np.array([0.6, 0.4], dtype=np.float64),
            np.array([0.7, 0.3, 0.8, 0.2], dtype=np.float64),
        ]
        return fr

    def test_nan_rows_excluded_from_oos_confusion_matrix(self) -> None:
        fr = self._make_fit_result_with_nan_oof(nan_indices=[0, 1])
        y_true = np.array([1, 0, 1, 0, 1, 0])
        result = confusion_matrix_table(fr, y_true, threshold=0.5, task="binary")
        cm_oos = result["oos"].to_numpy()
        assert cm_oos.sum() == 4, (
            f"OOS confusion matrix counted {cm_oos.sum()} rows, expected 4 "
            "(rows 0,1 have NaN OOF and should be excluded)"
        )
