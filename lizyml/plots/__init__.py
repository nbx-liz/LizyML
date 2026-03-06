"""LizyML plotting utilities.

All plot functions accept a :class:`~lizyml.core.types.fit_result.FitResult`
and return a ``plotly.graph_objects.Figure`` object.

Requires ``plotly>=5.0``.  Install with::

    pip install 'lizyml[plots]'
"""

from lizyml.plots.calibration import plot_calibration_curve, plot_probability_histogram
from lizyml.plots.classification import plot_roc_curve
from lizyml.plots.importance import plot_importance, plot_importance_from_dict
from lizyml.plots.learning_curve import plot_learning_curve
from lizyml.plots.oof_distribution import plot_oof_distribution
from lizyml.plots.tuning import plot_tuning_history

__all__ = [
    "plot_calibration_curve",
    "plot_importance",
    "plot_importance_from_dict",
    "plot_learning_curve",
    "plot_oof_distribution",
    "plot_probability_histogram",
    "plot_roc_curve",
    "plot_tuning_history",
]
