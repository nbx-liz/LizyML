"""LizyML plotting utilities.

All plot functions accept a :class:`~lizyml.core.types.fit_result.FitResult`
and return a ``plotly.graph_objects.Figure`` object.

Requires ``plotly>=5.0``.  Install with::

    pip install 'lizyml[plots]'
"""

from lizyml.plots.importance import plot_importance, plot_importance_from_dict
from lizyml.plots.learning_curve import plot_learning_curve
from lizyml.plots.oof_distribution import plot_oof_distribution

__all__ = [
    "plot_importance",
    "plot_importance_from_dict",
    "plot_learning_curve",
    "plot_oof_distribution",
]
