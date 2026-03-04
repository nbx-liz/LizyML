"""Cross-fit calibration — leakage-safe calibrated OOF + C_final.

Design invariants (from SKILL calibration):
- Calibrator NEVER sees X; it only receives (oof_scores, y).
- C_final is trained on ALL (oof_scores, y) and is only used for inference.
- The calibrated OOF used for evaluation is produced via cross-fit
  (each fold's calibrator is trained on the *other* folds — no same-row leak).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

from lizyml.calibration.base import BaseCalibratorAdapter


@dataclass
class CalibrationResult:
    """Outcome of cross-fit calibration.

    Attributes:
        c_final: Calibrator trained on ALL (oof_scores, y).
            Used for inference only.
        calibrated_oof: Cross-fit calibrated OOF probabilities.
            Length equals ``n_samples``; same row ordering as ``oof_scores``.
            Used for evaluation — never produced by ``c_final``.
        method: Calibration method name.
        split_indices: Per-fold ``(train_idx, valid_idx)`` used in cross-fit.
            Stored for reproducibility / audit.  Same ordering as the folds
            that produced ``calibrated_oof``.
    """

    c_final: BaseCalibratorAdapter
    calibrated_oof: np.ndarray
    method: str
    split_indices: list[tuple[np.ndarray, np.ndarray]]


def cross_fit_calibrate(
    oof_scores: np.ndarray,
    y: np.ndarray,
    calibrator_factory: Callable[[], BaseCalibratorAdapter],
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> CalibrationResult:
    """Produce leakage-safe calibrated OOF scores and a C_final calibrator.

    For each fold:
    - Calibrator is fitted on (oof_scores[train_idx], y[train_idx]).
    - Calibrated probabilities are predicted for valid_idx rows.

    C_final is trained on ALL (oof_scores, y) and is only used for inference.

    Args:
        oof_scores: 1-D base-model OOF probabilities (n_samples,).
        y: 1-D binary ground-truth labels (n_samples,).
        calibrator_factory: Callable returning a fresh calibrator instance.
        n_splits: Number of CV folds for the calibration cross-fit.
        random_state: Random seed for KFold shuffling.

    Returns:
        :class:`CalibrationResult` with ``c_final`` and ``calibrated_oof``.
    """
    calibrated_oof = np.empty_like(oof_scores, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split_indices: list[tuple[np.ndarray, np.ndarray]] = []

    for train_idx, val_idx in kf.split(oof_scores):
        split_indices.append((train_idx, val_idx))
        cal = calibrator_factory()
        cal.fit(oof_scores[train_idx], y[train_idx])
        calibrated_oof[val_idx] = cal.predict(oof_scores[val_idx])

    # C_final: trained on ALL data — for inference only, never for evaluation
    c_final = calibrator_factory()
    c_final.fit(oof_scores, y)

    return CalibrationResult(
        c_final=c_final,
        calibrated_oof=calibrated_oof,
        method=calibrator_factory().name,
        split_indices=split_indices,
    )
