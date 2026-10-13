"""Positive control for firing_rate_h0125.py's wiring (run with the plugin).

Run:
  env PYTHONPATH=docs/audits/2026-10-h0125/probes:tests FIRING_RATE_H0125_OUT=<out> \
    .venv/bin/python -m pytest -q -p no:cacheprovider -p firing_rate_h0125 \
    docs/audits/2026-10-h0125/probes/firing_rate_h0125_posctl.py

The output must contain an ``obj`` row flagged ``xentlambda``, a ``metric`` row
flagged ``native_params:huber``, ``dim`` rows flagged ``bad_objective:'regression'``,
``not_categorical:IntDim`` and ``bad_metric:'rmse'``, and ``tune`` rows flagged
``trial_refusal``.
"""

from __future__ import annotations

import pytest
from _helpers import make_binary_df, make_config, make_regression_df

from lizyml import Model
from lizyml.core.exceptions import LizyMLError


def test_xentlambda_fit_is_recorded() -> None:
    cfg = make_config("binary", n_estimators=2, n_splits=2)
    cfg["model"]["params"].update(objective="cross_entropy_lambda")
    Model(cfg).fit(data=make_binary_df())


def test_native_metric_params_are_recorded() -> None:
    cfg = make_config("regression", n_estimators=2, n_splits=2)
    cfg["model"]["params"]["metric"] = [{"huber": {"delta": 9.0}}]
    Model(cfg).fit(data=make_regression_df())


def test_cross_task_objective_choice_is_recorded() -> None:
    cfg = make_config("binary", n_estimators=2, n_splits=2, tuning_n_trials=2)
    cfg["tuning"]["optuna"]["space"] = {
        "objective": {"type": "categorical", "choices": ["regression"], "category": "model"}
    }
    with pytest.raises(LizyMLError):
        Model(cfg).tune(data=make_binary_df())


def test_numeric_objective_dimension_is_recorded() -> None:
    cfg = make_config("binary", n_estimators=2, n_splits=2, tuning_n_trials=2)
    cfg["tuning"]["optuna"]["space"] = {
        "objective": {"type": "int", "low": 1, "high": 2, "category": "model"}
    }
    with pytest.raises(LizyMLError):
        Model(cfg).tune(data=make_binary_df())


def test_bad_metric_choice_is_recorded() -> None:
    cfg = make_config("binary", n_estimators=2, n_splits=2, tuning_n_trials=2)
    cfg["tuning"]["optuna"]["space"] = {
        "metric": {"type": "categorical", "choices": ["rmse"], "category": "model"}
    }
    with pytest.raises(LizyMLError):
        Model(cfg).tune(data=make_binary_df())
