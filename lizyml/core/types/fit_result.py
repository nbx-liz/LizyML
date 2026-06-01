"""FitResult — the contract for all CV training outputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import numpy.typing as npt

from .artifacts import DataFingerprint, RunMeta, SplitIndices
from .target_encoder import TargetEncoder


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
                        "oof_coverage": float,  # covered-row fraction (H-0057)
                    },
                    "calibrated": { ... }  # binary + calibrator only
                }

        models: Trained model adapters, one per fold. Shared by reference
            across copies (read-only by convention — see ``__deepcopy__``).
        history: Per-fold training history dicts.
            Each dict contains at least ``"eval_history"`` and ``"best_iteration"``.
        feature_names: Ordered list of feature column names used during training.
        dtypes: Mapping of feature name to its dtype string.
        categorical_features: Names of features encoded as categorical.
        splits: Full index record for outer/inner/calibration splits.
        data_fingerprint: Fingerprint of the training dataset.
        pipeline_state: Serializable state of the FeaturePipeline. Shared by
            reference across copies (read-only by convention).
        calibrator: Fitted calibrator (``None`` when calibration is disabled).
            Shared by reference across copies (read-only by convention).
        run_meta: Version and config metadata captured at fit time.
        oof_raw_scores: OOF raw scores (logits) for calibration.
            ``None`` when calibration is not enabled. Shape ``(n_samples,)``
            for binary; ``(n_samples, n_classes)`` for multiclass.
        target_encoder: Encoder applied to y at fit time. ``no_op()`` for
            numeric targets / regression. Carries ``classes_`` so consumers
            can map predicted int codes back to original labels (H-0070).
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
    target_encoder: TargetEncoder = field(default_factory=TargetEncoder.no_op)

    #: Trained-estimator fields shared by reference on copy (see ``__deepcopy__``).
    _SHARED_ON_COPY = ("models", "calibrator", "pipeline_state")

    def __deepcopy__(self, memo: dict[int, Any]) -> FitResult:
        """Selective deep copy used by the public ``Model.fit_result`` return.

        Mutable *data* fields (``metrics``, ``history``, ``splits``, arrays,
        ...) are deep-copied so a caller mutating the returned ``FitResult``
        cannot corrupt internal state — and thereby a later ``export()`` (the
        export-contamination vector flows through ``metrics``). Trained
        estimators (``models`` / ``calibrator`` / ``pipeline_state``) are shared
        by reference: deep-copying a LightGBM ``Booster`` round-trips through its
        model string and drops ``params`` fidelity (e.g. ``objective`` becomes
        ``None``), which would degrade the metadata reachable via
        ``fit_result.models``. Those shared objects are read-only by convention
        (H-0082).
        """
        shared = set(self._SHARED_ON_COPY)
        init_kwargs: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in shared:
                # New list container for ``models`` (defensive against list-level
                # mutation) while keeping the trained adapters themselves shared.
                init_kwargs[f.name] = list(value) if isinstance(value, list) else value
            else:
                init_kwargs[f.name] = deepcopy(value, memo)
        return FitResult(**init_kwargs)
