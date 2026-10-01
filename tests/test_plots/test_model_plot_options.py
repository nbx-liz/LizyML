"""The ``Model`` plot methods forward their documented options (H-0108, #268).

``Model.importance_plot(top_n=...)`` and ``Model.plot_learning_curve(metrics=...)``
are documented public options that no test bound off their defaults through
the ``Model`` method (the module-level plot functions were tested directly).
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from lizyml import Model
from tests._helpers import make_config, make_regression_df


@pytest.fixture(scope="module")
def fitted() -> Model:
    cfg = make_config("regression", n_estimators=20)
    cfg["evaluation"] = {"metrics": ["rmse", "mae"]}
    cfg["model"]["params"]["metric"] = ["rmse", "mae"]
    model = Model(cfg)
    model.fit(data=make_regression_df(n=200))
    return model


def _bar_labels(fig: go.Figure) -> list[str]:
    (bar,) = [trace for trace in fig.data if isinstance(trace, go.Bar)]
    labels = bar.y if bar.orientation == "h" else bar.x
    return list(labels)


def test_importance_plot_top_n_limits_the_features(fitted: Model) -> None:
    n_features = len(fitted.fit_result.feature_names)
    assert n_features > 1
    assert len(_bar_labels(fitted.importance_plot())) == n_features
    assert len(_bar_labels(fitted.importance_plot(top_n=1))) == 1


def test_plot_learning_curve_metrics_filters_the_subplots(fitted: Model) -> None:
    """The history records LightGBM's names (``mae`` is ``l1``); filter on one."""

    def titles(fig: go.Figure) -> list[str]:
        return [
            ann.text for ann in fig.layout.annotations if getattr(ann, "text", None)
        ]

    everything = titles(fitted.plot_learning_curve())
    assert any("rmse" in t for t in everything) and any("l1" in t for t in everything)
    only_rmse = titles(fitted.plot_learning_curve(metrics=["rmse"]))
    assert any("rmse" in t for t in only_rmse)
    assert not any("l1" in t for t in only_rmse)
