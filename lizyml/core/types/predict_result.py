"""PredictionResult — the contract for inference outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionResult:
    """Output of a single prediction call.

    Attributes:
        pred: Point predictions, shape ``(n_samples,)``.
        proba: Class probabilities for binary classification, shape ``(n_samples,)``.
            ``None`` for regression and multiclass.
        shap_values: SHAP explanation values, shape ``(n_samples, n_features)``.
            ``None`` when ``return_shap=False`` (default).
        used_features: Names of features that were present and used for prediction.
        warnings: Human-readable messages about column drift or other corrections
            applied during prediction.
    """

    pred: np.ndarray
    proba: np.ndarray | None
    shap_values: np.ndarray | None
    used_features: list[str]
    warnings: list[str]
