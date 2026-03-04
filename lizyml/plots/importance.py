"""Feature importance plot.

Uses the fold-averaged importance scores stored in ``FitResult.models``.
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


def plot_importance(
    fit_result: FitResult,
    *,
    kind: str = "split",
    top_n: int | None = 20,
) -> Any:
    """Plot fold-averaged feature importances as a horizontal bar chart.

    Args:
        fit_result: A fitted :class:`~lizyml.core.types.fit_result.FitResult`.
        kind: ``"split"`` or ``"gain"``.
        top_n: Maximum number of features to show.  ``None`` shows all.

    Returns:
        A ``matplotlib.figure.Figure`` object.

    Raises:
        LizyMLError with ``OPTIONAL_DEP_MISSING`` when matplotlib is not installed.
        LizyMLError with ``MODEL_NOT_FIT`` when no fold models are available.
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

    if not fit_result.models:
        raise LizyMLError(
            code=ErrorCode.MODEL_NOT_FIT,
            user_message="No trained models found in FitResult.",
            context={},
        )

    # Average importance across folds
    agg: dict[str, float] = {}
    n_models = len(fit_result.models)
    for model in fit_result.models:
        for feat, val in model.importance(kind=kind).items():
            agg[feat] = agg.get(feat, 0.0) + val / n_models

    # Sort descending by importance
    sorted_items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = _mpl.subplots(figsize=(8, max(4, len(features) * 0.35)))
    ax.barh(range(len(features)), values[::-1])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features[::-1])
    ax.set_xlabel(f"Importance ({kind})")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return fig
