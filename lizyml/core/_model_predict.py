"""Prediction + calibration assembly (#172).

Extracted from ``model.py`` so the facade stays assembly-only (CLAUDE.md §3).
The per-task estimator dispatch, ``CalibrationResult`` handling (with the
raw-score-vs-probability backward-compat fallback), the ``proba >= 0.5``
threshold, and SHAP computation all live here — this is the seam where
god-object pressure would otherwise reconcentrate when a second estimator is
added. ``Model.predict`` now only resolves state and delegates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from lizyml.core.types.predict_result import PredictionResult

if TYPE_CHECKING:
    import pandas as pd

    from lizyml.core.types.fit_result import FitResult
    from lizyml.core.types.task import TaskType
    from lizyml.training.refit_trainer import RefitResult


def run_predict(
    *,
    task: TaskType,
    fit_result: FitResult,
    refit_result: RefitResult,
    provider: Any,
    X: pd.DataFrame,
    return_shap: bool,
) -> PredictionResult:
    """Generate a :class:`PredictionResult` from the full-data refit model.

    Mirrors the previous inline body of :meth:`Model.predict` exactly — same
    pipeline restore, per-task branching, calibration, label inverse-mapping
    (H-0070), and optional SHAP — so the public output is unchanged.
    """
    # Restore the fitted pipeline from saved state via the provider.
    pipeline = provider.build_pipeline_factory()()
    pipeline.load_state(refit_result.pipeline_state)
    X_t, warnings = pipeline.transform_with_warnings(X)

    model = refit_result.model

    # H-0070: pred may be int (numeric target) or original-label dtype
    # (object / string / category) after inverse_transform.
    pred: npt.NDArray[Any]
    proba: npt.NDArray[np.float64] | None = None

    if task == "regression":
        pred = model.predict(X_t)
    elif task == "binary":
        proba_2d = model.predict_proba(X_t)
        proba = proba_2d[:, 1]
        # Apply C_final calibrator when available (H-0030: raw score input).
        if fit_result.calibrator is not None:
            from lizyml.calibration.cross_fit import CalibrationResult

            if isinstance(fit_result.calibrator, CalibrationResult):
                if fit_result.oof_raw_scores is not None:
                    raw_scores: npt.NDArray[np.float64] = model.predict_raw(X_t)
                    proba = fit_result.calibrator.c_final.predict(raw_scores)
                else:
                    # Backward compat: old artifact trained on probabilities.
                    proba = fit_result.calibrator.c_final.predict(proba)
        pred_codes: npt.NDArray[Any] = (proba >= 0.5).astype(int)
        # H-0070: inverse-map int codes back to original labels when the
        # target was non-numeric at fit time.
        pred = fit_result.target_encoder.inverse_transform(pred_codes)
    else:  # multiclass
        proba = model.predict_proba(X_t)
        pred_codes = proba.argmax(axis=1)
        pred = fit_result.target_encoder.inverse_transform(pred_codes)

    shap_values: npt.NDArray[np.float64] | None = None
    if return_shap:
        from lizyml.explain.shap_explainer import compute_shap_values

        shap_values = compute_shap_values(refit_result.model, X_t, task)

    return PredictionResult(
        pred=pred,
        proba=proba,
        shap_values=shap_values,
        used_features=refit_result.feature_names,
        warnings=warnings,
    )
