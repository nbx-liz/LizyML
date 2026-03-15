"""FitResult — the contract for all CV training outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .artifacts import DataFingerprint, RunMeta, SplitIndices


@dataclass
class FitResult:
    """Complete output of a CV training run.

    All fields are required; no field may be ``None`` except ``calibrator``.

    Attributes:
        oof_pred: Out-of-fold predictions.
            Shape ``(n_samples,)`` for regression/binary;
            ``(n_samples, n_classes)`` for multiclass.
        if_pred_per_fold: In-fold predictions, one array per fold.
            Length equals ``n_splits``; each array covers the full training fold.
        metrics: Nested dict with structure::

                {
                    "raw": {
                        "oof": {metric_name: float, ...},
                        "oof_per_fold": [{metric_name: float}, ...],
                        "if_mean": {metric_name: float, ...},
                        "if_per_fold": [{metric_name: float}, ...],
                    },
                    "calibrated": { ... }  # binary + calibrator only
                }

        models: Trained model adapters, one per fold.
        history: Per-fold training history dicts.
            Each dict contains at least ``"eval_history"`` and ``"best_iteration"``.
        feature_names: Ordered list of feature column names used during training.
        dtypes: Mapping of feature name to its dtype string.
        categorical_features: Names of features encoded as categorical.
        splits: Full index record for outer/inner/calibration splits.
        data_fingerprint: Fingerprint of the training dataset.
        pipeline_state: Serializable state of the FeaturePipeline.
        calibrator: Fitted calibrator (``None`` when calibration is disabled).
        run_meta: Version and config metadata captured at fit time.
        oof_raw_scores: OOF raw scores (logits) for calibration.
            ``None`` when calibration is not enabled. Shape ``(n_samples,)``
            for binary; ``(n_samples, n_classes)`` for multiclass.
    """

    oof_pred: npt.NDArray[np.float64]
    if_pred_per_fold: list[npt.NDArray[np.float64]]
    metrics: dict[str, Any]
    models: list[Any]
    history: list[dict[str, Any]]
    feature_names: list[str]
    dtypes: dict[str, str]
    categorical_features: list[str]
    splits: SplitIndices
    data_fingerprint: DataFingerprint
    pipeline_state: Any
    calibrator: Any | None
    run_meta: RunMeta
    oof_raw_scores: npt.NDArray[np.float64] | None = None
