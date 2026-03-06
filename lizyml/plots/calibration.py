"""Calibration plots: reliability diagram and probability histogram.

Binary classification only. Requires calibration to be enabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult

_plotly: Any = None
try:
    import plotly.graph_objects as go

    _plotly = go
except ImportError:  # pragma: no cover
    pass


def _require_plotly() -> None:
    if _plotly is None:
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                "plotly is required for calibration plots. "
                "Install with: pip install 'lizyml[plots]'"
            ),
            context={"package": "plotly"},
        )


def _require_calibrator(fit_result: FitResult) -> Any:
    """Return the CalibrationResult or raise."""
    from lizyml.calibration.cross_fit import CalibrationResult

    if not isinstance(fit_result.calibrator, CalibrationResult):
        raise LizyMLError(
            code=ErrorCode.CALIBRATION_NOT_SUPPORTED,
            user_message=(
                "Calibration plots require calibration to be enabled. "
                "Set calibration.method in the config."
            ),
            context={},
        )
    return fit_result.calibrator


def plot_calibration_curve(
    fit_result: FitResult,
    y_true: npt.NDArray[Any],
    *,
    n_bins: int = 10,
) -> Any:
    """Plot reliability diagram (Raw vs Calibrated OOF).

    Args:
        fit_result: Completed CV training output with calibration enabled.
        y_true: Ground-truth binary target array.
        n_bins: Number of bins for the calibration curve.

    Returns:
        ``plotly.graph_objects.Figure``.

    Raises:
        LizyMLError with OPTIONAL_DEP_MISSING if plotly is not installed.
        LizyMLError with CALIBRATION_NOT_SUPPORTED if calibration is disabled.
    """
    _require_plotly()
    calibrator = _require_calibrator(fit_result)

    from sklearn.calibration import calibration_curve

    y_arr = np.asarray(y_true)
    raw_oof = fit_result.oof_pred
    cal_oof = calibrator.calibrated_oof

    prob_true_raw, prob_pred_raw = calibration_curve(y_arr, raw_oof, n_bins=n_bins)
    prob_true_cal, prob_pred_cal = calibration_curve(y_arr, cal_oof, n_bins=n_bins)

    fig = _plotly.Figure()
    fig.add_trace(
        _plotly.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect",
        )
    )
    fig.add_trace(
        _plotly.Scatter(
            x=prob_pred_raw.tolist(),
            y=prob_true_raw.tolist(),
            mode="lines+markers",
            name="Raw OOF",
            line=dict(color="steelblue"),
        )
    )
    fig.add_trace(
        _plotly.Scatter(
            x=prob_pred_cal.tolist(),
            y=prob_true_cal.tolist(),
            mode="lines+markers",
            name="Calibrated OOF",
            line=dict(color="darkorange"),
        )
    )
    fig.update_layout(
        title="Calibration Curve (Reliability Diagram)",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        height=500,
        width=600,
    )
    return fig


def plot_probability_histogram(fit_result: FitResult) -> Any:
    """Plot raw vs calibrated predicted probability histograms.

    Args:
        fit_result: Completed CV training output with calibration enabled.

    Returns:
        ``plotly.graph_objects.Figure``.

    Raises:
        LizyMLError with OPTIONAL_DEP_MISSING if plotly is not installed.
        LizyMLError with CALIBRATION_NOT_SUPPORTED if calibration is disabled.
    """
    _require_plotly()
    calibrator = _require_calibrator(fit_result)

    raw_oof = fit_result.oof_pred
    cal_oof = calibrator.calibrated_oof

    fig = _plotly.Figure()
    fig.add_trace(
        _plotly.Histogram(
            x=raw_oof.tolist(),
            nbinsx=30,
            opacity=0.6,
            name="Raw OOF",
            marker_color="steelblue",
        )
    )
    fig.add_trace(
        _plotly.Histogram(
            x=cal_oof.tolist(),
            nbinsx=30,
            opacity=0.6,
            name="Calibrated OOF",
            marker_color="darkorange",
        )
    )
    fig.update_layout(
        title="Probability Histogram (Raw vs Calibrated)",
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
        width=600,
    )
    return fig
