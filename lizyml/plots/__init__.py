"""LizyML plotting utilities.

All plot functions accept a :class:`~lizyml.core.types.fit_result.FitResult`
and return a ``matplotlib.figure.Figure`` object.

Requires ``matplotlib>=3.7``.  Install with::

    pip install 'lizyml[plots]'
"""

from lizyml.plots.importance import plot_importance
from lizyml.plots.learning_curve import plot_learning_curve
from lizyml.plots.oof_distribution import plot_oof_distribution

__all__ = [
    "plot_importance",
    "plot_learning_curve",
    "plot_oof_distribution",
]
