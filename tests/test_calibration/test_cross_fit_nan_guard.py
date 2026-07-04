"""Regression tests: cross_fit_calibrate NaN handling (#6 + code-review fixes).

- NaN OOF scores in a validation fold must use the fallback, never reach
  ``calibrator.predict``.
- Rows never covered by any split must remain NaN (NaN-initialized array).
- ``method`` reflects ``c_final.name`` without an extra calibrator instance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from lizyml.calibration.cross_fit import cross_fit_calibrate
from lizyml.calibration.platt import PlattCalibrator
from lizyml.splitters.kfold import KFoldSplitter


def _kfold_indices(
    n: int, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(
        KFoldSplitter(n_splits=n_splits, shuffle=True, random_state=seed).split(n)
    )


class StubCalibrator:
    """Calibrator that records inputs and tracks NaN."""

    name: str = "stub"

    def __init__(self) -> None:
        self.predict_received_nan: bool = False

    def fit(self, scores: npt.NDArray[np.float64], y: npt.NDArray[Any]) -> None:
        pass

    def predict(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if np.any(np.isnan(scores)):
            self.predict_received_nan = True
        return scores.copy()


class TestCrossFitNaNGuard:
    def test_nan_val_rows_use_fallback(self) -> None:
        n = 10
        oof_scores = np.array([np.nan, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1])
        fallback = np.full(n, 0.5)
        split_indices = [
            (
                np.array([2, 3, 4, 5, 6], dtype=np.intp),
                np.array([0, 1, 7, 8, 9], dtype=np.intp),
            ),
            (
                np.array([0, 1, 7, 8, 9], dtype=np.intp),
                np.array([2, 3, 4, 5, 6], dtype=np.intp),
            ),
        ]
        calibrators: list[StubCalibrator] = []

        def cal_factory() -> StubCalibrator:
            c = StubCalibrator()
            calibrators.append(c)
            return c

        result = cross_fit_calibrate(
            oof_scores,
            y,
            cal_factory,  # type: ignore[arg-type]
            split_indices=split_indices,
            oof_pred=fallback,
        )

        for i, cal in enumerate(calibrators[:-1]):  # last is c_final
            assert not cal.predict_received_nan, f"Fold {i} calibrator received NaN"
        assert result.calibrated_oof[0] == pytest.approx(0.5)
        assert result.calibrated_oof[1] == pytest.approx(0.5)
        for i in range(2, 10):
            assert not np.isnan(result.calibrated_oof[i]), (
                f"Row {i} should be calibrated"
            )


class TestCrossFitNaNInit:
    def test_unfilled_indices_are_nan(self) -> None:
        n = 100
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, n).astype(float)
        scores = np.clip(y + rng.normal(0, 0.3, n), 0.01, 0.99)
        partial_indices = [
            (train_idx, val_idx[val_idx != 0])
            for train_idx, val_idx in _kfold_indices(n, n_splits=3)
        ]
        result = cross_fit_calibrate(
            oof_scores=scores,
            y=y,
            calibrator_factory=PlattCalibrator,
            split_indices=partial_indices,
        )
        assert np.isnan(result.calibrated_oof[0]), "Unfilled index should be NaN"

    def test_method_uses_c_final_name(self) -> None:
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
