"""Probability metrics refuse values that are not probabilities (H-0106, #263).

``needs_proba`` declares that a metric needs probabilities, and LizyML's own
callers pass them. A value outside [0, 1], a non-finite value, or 1-D input for
more than two classes reaching one of these metrics means an upstream defect
(#307: ``cross_entropy_lambda``'s output is not a probability). Before this,
``auc`` / ``auc_pr`` / ``ece`` / ``precision_at_k`` computed silently on such
input, and ``logloss`` / ``brier`` failed with a raw scikit-learn error.

Hard 0/1 labels are valid probabilities and cannot be told apart; they pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.core.registries import MetricRegistry
from lizyml.metrics import get_metric
from tests._helpers import make_config

_PROBA_METRICS = sorted(
    n
    for n in MetricRegistry.keys()  # noqa: SIM118 -- a registry class, not a dict
    if get_metric(n).needs_proba
)

_Y_BIN = np.array([0, 1, 0, 1, 1, 0, 1, 0])
_P_BIN = np.array([0.1, 0.8, 0.3, 0.6, 0.9, 0.2, 0.7, 0.4])
_Y_MC = np.array([0, 1, 2, 0, 1, 2])
_P_MC = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]] * 2)

#: case -> (y_true, y_pred) that is not a probability for that y_true.
_BAD: dict[str, tuple[Any, Any]] = {
    "above_one": (_Y_BIN, _P_BIN + 0.5),
    "negative": (_Y_BIN, _P_BIN - 0.5),
    "logits": (_Y_BIN, np.log(_P_BIN / (1 - _P_BIN))),
    "nan": (_Y_BIN, np.where(_P_BIN > 0.85, np.nan, _P_BIN)),
    "inf": (_Y_BIN, np.where(_P_BIN > 0.85, np.inf, _P_BIN)),
    "not_numeric": (_Y_BIN, np.array(["a"] * len(_Y_BIN), dtype=object)),
    "1d_for_three_classes": (_Y_MC, np.array([0.1, 0.5, 0.9, 0.2, 0.6, 0.8])),
}


def test_the_population_is_every_needs_proba_metric() -> None:
    assert _PROBA_METRICS == [
        "auc",
        "auc_pr",
        "brier",
        "ece",
        "logloss",
        "precision_at_k",
    ]


@pytest.mark.parametrize("case", sorted(_BAD))
@pytest.mark.parametrize("name", _PROBA_METRICS)
def test_non_probabilities_are_refused(name: str, case: str) -> None:
    y_true, y_pred = _BAD[case]
    with pytest.raises(LizyMLError) as exc:
        get_metric(name)(y_true, y_pred)
    assert exc.value.code == ErrorCode.METRIC_REQUIRES_PROBA
    assert exc.value.context["metric"] == name
    assert exc.value.context["reason"]


#: Each metric on (_Y_BIN, _P_BIN), computed by hand, not by lizyml. The
#: classes are separated (positives 0.6-0.9, negatives 0.1-0.4), so the ranking
#: metrics are 1; every bin of the ECE holds one label, so it is the mean of
#: |y - p|; precision_at_k (k=10) takes the single top score, 0.9, a positive.
_EXPECTED = {
    "auc": 1.0,
    "auc_pr": 1.0,
    "brier": float(np.mean((_P_BIN - _Y_BIN) ** 2)),  # 0.075
    "ece": float(np.mean(np.abs(_Y_BIN - _P_BIN))),  # 0.25
    "logloss": float(
        -np.mean(_Y_BIN * np.log(_P_BIN) + (1 - _Y_BIN) * np.log(1 - _P_BIN))
    ),
    "precision_at_k": 1.0,
}


@pytest.mark.parametrize("name", _PROBA_METRICS)
def test_probabilities_and_hard_labels_pass(name: str) -> None:
    metric = get_metric(name)
    assert metric(_Y_BIN, _P_BIN) == pytest.approx(_EXPECTED[name], rel=1e-12)
    assert np.isfinite(metric(_Y_BIN, _Y_BIN.astype(float)))
    assert np.isfinite(metric(_Y_BIN, _Y_BIN.astype(bool)))
    assert metric(_Y_BIN, _P_BIN.astype(object)) == metric(_Y_BIN, _P_BIN)


_MULTICLASS = ["auc", "auc_pr", "brier", "logloss"]


@pytest.mark.parametrize("name", _MULTICLASS)
def test_multiclass_matrices(name: str) -> None:
    metric = get_metric(name)
    assert np.isfinite(metric(_Y_MC, _P_MC))
    with pytest.raises(LizyMLError) as exc:
        metric(_Y_MC, _P_MC * 2.0)
    assert exc.value.code == ErrorCode.METRIC_REQUIRES_PROBA


@pytest.mark.parametrize("name", _PROBA_METRICS)
def test_two_class_labels_other_than_zero_one_pass(name: str) -> None:
    """Two classes spelled 3 / 7 are still two classes: the 1-D rule counts
    classes, not their values. A metric may still refuse such labels through
    its own label handling (scikit-learn), which is not this check."""
    y = np.where(_Y_BIN == 1, 7, 3)
    try:
        get_metric(name)(y, _P_BIN)
    except LizyMLError as exc:
        assert exc.code != ErrorCode.METRIC_REQUIRES_PROBA
    except ValueError:
        pass  # scikit-learn's own label handling, before and after H-0106


def test_cross_entropy_lambda_fit_reports_the_metric() -> None:
    """#307's data: the objective's output exceeds 1, and a fit limited to a
    ranking metric used to succeed and report it as a probability."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = (2 * X[:, 0] + rng.normal(scale=0.5, size=1500) > 0).astype(int)
    cfg = make_config("binary", n_estimators=300, n_splits=3)
    cfg["model"]["params"].update(objective="cross_entropy_lambda", learning_rate=0.1)
    cfg["evaluation"] = {"metrics": ["auc"]}
    with pytest.raises(LizyMLError) as exc:
        Model(cfg).fit(data=df)
    assert exc.value.code == ErrorCode.METRIC_REQUIRES_PROBA
    assert exc.value.context["metric"] == "auc"
