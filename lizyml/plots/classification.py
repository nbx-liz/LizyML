"""Classification plots: ROC Curve (binary + multiclass OvR)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score, roc_curve

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult

_plotly: Any = None
_make_subplots: Any = None
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _plotly = go
    _make_subplots = make_subplots
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OVR_COLORS = (
    "steelblue",
    "darkorange",
    "green",
    "purple",
    "crimson",
    "teal",
    "goldenrod",
    "slategray",
)


def _collect_is_data(
    fit_result: FitResult,
    y_true: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Return ``(is_pred_all, is_y_all)`` from per-fold IF data."""
    is_preds: list[npt.NDArray[Any]] = []
    is_y: list[npt.NDArray[Any]] = []
    for (train_idx, _), if_pred in zip(
        fit_result.splits.outer, fit_result.if_pred_per_fold, strict=True
    ):
        is_preds.append(if_pred)
        is_y.append(y_true[train_idx])
    return np.concatenate(is_preds, axis=0), np.concatenate(is_y)


# ---------------------------------------------------------------------------
# Binary ROC
# ---------------------------------------------------------------------------


def _plot_roc_binary(fit_result: FitResult, y_true: npt.NDArray[Any]) -> Any:
    y_arr = np.asarray(y_true)
    oof_pred = fit_result.oof_pred

    # OOS
    fpr_oos, tpr_oos, _ = roc_curve(y_arr, oof_pred)
    auc_oos = float(roc_auc_score(y_arr, oof_pred))

    # IS
    is_pred, is_y = _collect_is_data(fit_result, y_arr)
    fpr_is, tpr_is, _ = roc_curve(is_y, is_pred)
    auc_is = float(roc_auc_score(is_y, is_pred))

    fig = _plotly.Figure()
    fig.add_trace(
        _plotly.Scatter(
            x=fpr_oos.tolist(),
            y=tpr_oos.tolist(),
            mode="lines",
            name=f"OOS (AUC={auc_oos:.3f})",
            line=dict(color="steelblue"),
        )
    )
    fig.add_trace(
        _plotly.Scatter(
            x=fpr_is.tolist(),
            y=tpr_is.tolist(),
            mode="lines",
            name=f"IS (AUC={auc_is:.3f})",
            line=dict(color="darkorange", dash="dash"),
        )
    )
    fig.add_trace(
        _plotly.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Random",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="ROC Curve (IS vs OOS)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        width=600,
    )
    return fig


# ---------------------------------------------------------------------------
# Multiclass OvR ROC
# ---------------------------------------------------------------------------


def _plot_roc_multiclass(fit_result: FitResult, y_true: npt.NDArray[Any]) -> Any:
    from sklearn.preprocessing import label_binarize

    y_arr = np.asarray(y_true)
    n_classes = fit_result.oof_pred.shape[1]
    classes = np.arange(n_classes)
    y_bin = label_binarize(y_arr, classes=classes)
    oof_pred = fit_result.oof_pred

    is_pred, is_y = _collect_is_data(fit_result, y_arr)
    is_y_bin = label_binarize(is_y, classes=classes)

    fig = _make_subplots(
        rows=1, cols=2, subplot_titles=["IS ROC Curves", "OOS ROC Curves"]
    )

    for k in range(n_classes):
        color = _OVR_COLORS[k % len(_OVR_COLORS)]
        # IS
        fpr_is, tpr_is, _ = roc_curve(is_y_bin[:, k], is_pred[:, k])
        auc_is = float(roc_auc_score(is_y_bin[:, k], is_pred[:, k]))
        fig.add_trace(
            _plotly.Scatter(
                x=fpr_is.tolist(),
                y=tpr_is.tolist(),
                mode="lines",
                name=f"IS Class {k} (AUC={auc_is:.3f})",
                line=dict(color=color, dash="dash"),
            ),
            row=1,
            col=1,
        )
        # OOS
        fpr_oos, tpr_oos, _ = roc_curve(y_bin[:, k], oof_pred[:, k])
        auc_oos = float(roc_auc_score(y_bin[:, k], oof_pred[:, k]))
        fig.add_trace(
            _plotly.Scatter(
                x=fpr_oos.tolist(),
                y=tpr_oos.tolist(),
                mode="lines",
                name=f"OOS Class {k} (AUC={auc_oos:.3f})",
                line=dict(color=color),
            ),
            row=1,
            col=2,
        )

    # Reference lines
    for col in (1, 2):
        fig.add_trace(
            _plotly.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=col)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=col)

    macro_auc = float(roc_auc_score(y_bin, oof_pred, average="macro"))
    fig.update_layout(
        title=f"ROC Curves OvR (Macro AUC={macro_auc:.3f})",
        height=500,
        width=1000,
    )
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_roc_curve(
    fit_result: FitResult,
    y_true: npt.NDArray[Any],
    *,
    task: str,
) -> Any:
    """Plot ROC curve(s).

    Args:
        fit_result: Completed CV training output.
        y_true: Ground-truth target array.
        task: ``"binary"`` or ``"multiclass"``.

    Returns:
        ``plotly.graph_objects.Figure``.

    Raises:
        LizyMLError with OPTIONAL_DEP_MISSING if plotly is not installed.
        LizyMLError with UNSUPPORTED_TASK for regression.
    """
    if _plotly is None:
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                "plotly is required for ROC curve plots. "
                "Install with: pip install 'lizyml[plots]'"
            ),
            context={"package": "plotly"},
        )
    if task == "regression":
        raise LizyMLError(
            code=ErrorCode.UNSUPPORTED_TASK,
            user_message="roc_curve_plot() requires a binary or multiclass task.",
            context={"task": task},
        )
    if task == "binary":
        return _plot_roc_binary(fit_result, y_true)
    return _plot_roc_multiclass(fit_result, y_true)
