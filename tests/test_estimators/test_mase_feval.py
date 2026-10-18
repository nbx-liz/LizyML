"""MASE as a LightGBM feval (H-0132, criterion 3).

The feval's scale comes from the rows the booster trains on, which
``LGBMAdapter.fit`` passes into metric resolution; never from the eval set's
labels (``dataset.get_label()``), which are the rows being scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.estimators.lgbm.metric_bridge import resolve_metrics
from lizyml.estimators.lgbm.param_validation import resolve_user_metric


def _scale(y: np.ndarray) -> float:
    return float(np.mean(np.abs(y - y.mean())))


def _data(seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    X_tr = pd.DataFrame({"f": rng.normal(size=200)})
    y_tr = pd.Series(X_tr["f"] + rng.normal(scale=0.3, size=200))
    X_va = pd.DataFrame({"f": rng.normal(size=60)})
    # The scored rows spread ten times wider than the training rows.
    y_va = pd.Series(10.0 * X_va["f"] + rng.normal(scale=3.0, size=60))
    return X_tr, y_tr, X_va, y_va


def test_the_feval_divides_by_the_training_rows_scale() -> None:
    X_tr, y_tr, X_va, y_va = _data()
    adapter = LGBMAdapter(
        task="regression",
        params={"metric": ["mase"], "n_estimators": 6, "verbose": -1},
        early_stopping_rounds=None,
    )
    adapter.fit(X_tr, y_tr, X_va, y_va)
    recorded = adapter.eval_results["valid_0"]["mase"][-1]
    mae = float(np.mean(np.abs(y_va.to_numpy() - adapter.predict(X_va))))
    assert recorded == pytest.approx(mae / _scale(y_tr.to_numpy()), rel=1e-6)
    assert recorded != pytest.approx(mae / _scale(y_va.to_numpy()), rel=1e-3)


def test_a_constant_training_target_is_refused_by_the_feval() -> None:
    X_tr, _, X_va, y_va = _data()
    adapter = LGBMAdapter(
        task="regression",
        params={"metric": ["mase"], "n_estimators": 3, "verbose": -1},
    )
    with pytest.raises(LizyMLError) as caught:
        adapter.fit(X_tr, pd.Series(np.full(len(X_tr), 2.0)), X_va, y_va)
    assert caught.value.code == ErrorCode.UNSUPPORTED_METRIC


def test_resolution_without_training_targets_builds_an_uncalled_feval() -> None:
    """Metadata paths (config checks, params_table, export params) resolve the
    metric without training targets; they never call the feval."""
    native, fevals, names = resolve_metrics(["mase", "l1"], "regression")
    assert names == ["mase"] and len(fevals) == 1 and native == ["l1"]
    assert resolve_user_metric(["mase"], "regression") is not None


def test_resolution_with_training_targets_binds_the_metric() -> None:
    _, fevals, _ = resolve_metrics(
        ["mase"], "regression", y_train=np.array([2.0, 6.0, 2.0, 6.0])
    )

    class _Labels:
        def get_label(self) -> np.ndarray:
            return np.array([0.0, 4.0])

    name, value, higher = fevals[0](np.array([1.0, 4.0]), _Labels())
    assert (name, higher) == ("mase", False)
    assert value == pytest.approx(0.5 / 2.0)


def test_an_unbound_feval_refuses_to_run() -> None:
    _, fevals, _ = resolve_metrics(["mase"], "regression")

    class _Labels:
        def get_label(self) -> np.ndarray:
            return np.array([0.0, 4.0])

    with pytest.raises(LizyMLError) as caught:
        fevals[0](np.array([1.0, 4.0]), _Labels())
    assert caught.value.code == ErrorCode.UNSUPPORTED_METRIC
