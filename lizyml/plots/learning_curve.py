"""Learning curve plot.

Visualises per-fold training history from ``FitResult.history``.
Requires early stopping to be enabled (otherwise eval history is empty).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult

_mpl: Any = None
try:
    import matplotlib.pyplot as plt

    _mpl = plt
except ImportError:  # pragma: no cover
    pass


def plot_learning_curve(fit_result: FitResult) -> Any:
    """Plot per-fold training/validation loss vs iteration.

    Args:
        fit_result: A fitted :class:`~lizyml.core.types.fit_result.FitResult`.

    Returns:
        A ``matplotlib.figure.Figure`` object.

    Raises:
        LizyMLError with ``OPTIONAL_DEP_MISSING`` when matplotlib is not installed.
        LizyMLError with ``MODEL_NOT_FIT`` when no history is available or the
            history contains no evaluation metrics (e.g. early stopping disabled).
    """
    if _mpl is None:
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message=(
                "matplotlib is required for plots. "
                "Install with: pip install 'lizyml[plots]'"
            ),
            context={"package": "matplotlib"},
        )

    if not fit_result.history:
        raise LizyMLError(
            code=ErrorCode.MODEL_NOT_FIT,
            user_message="No training history found in FitResult.",
            context={},
        )

    # Gather eval_history entries; skip folds with empty history
    fold_histories: list[dict[str, list[float]]] = []
    for fold_hist in fit_result.history:
        eval_hist = fold_hist.get("eval_history", {})
        if not eval_hist:
            continue
        # eval_hist is {dataset_name: {metric_name: [values]}}
        flat: dict[str, list[float]] = {}
        for ds_name, metrics in eval_hist.items():
            for metric_name, values in metrics.items():
                key = f"{ds_name}/{metric_name}"
                flat[key] = list(values)
        if flat:
            fold_histories.append(flat)

    if not fold_histories:
        raise LizyMLError(
            code=ErrorCode.MODEL_NOT_FIT,
            user_message=(
                "No evaluation history found. "
                "Enable early stopping to record validation metrics per fold."
            ),
            context={},
        )

    # Use first fold's keys as reference
    metric_keys = list(fold_histories[0].keys())
    n_metrics = len(metric_keys)

    fig, axes = _mpl.subplots(1, n_metrics, figsize=(6 * n_metrics, 4), squeeze=False)

    for col, key in enumerate(metric_keys):
        ax = axes[0][col]
        for fold_idx, fh in enumerate(fold_histories):
            if key in fh:
                ax.plot(fh[key], label=f"fold {fold_idx}", alpha=0.7)
        ax.set_title(key)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        if len(fold_histories) <= 10:
            ax.legend(fontsize="small")

    fig.tight_layout()
    return fig
