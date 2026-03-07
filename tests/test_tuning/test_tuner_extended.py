"""Tests for Tuner smart/training per-trial handling (H-0024)."""

from __future__ import annotations

from lizyml import Model
from lizyml.core.types.tuning_result import TuningResult
from tests._helpers import make_regression_df


class TestSmartParamsPerTrial:
    def test_smart_params_resolved_per_trial(self) -> None:
        """Smart dims (num_leaves_ratio) should produce num_leaves in trial params."""
        config = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "target"},
            "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
            "model": {"name": "lgbm"},
            "training": {"seed": 0},
            "tuning": {
                "optuna": {
                    "params": {"n_trials": 2, "direction": "minimize"},
                    "space": {},
                }
            },
        }
        m = Model(config)
        result = m.tune(data=make_regression_df(n=100))
        assert isinstance(result, TuningResult)
        # num_leaves_ratio should be in best_params (sampled by optuna)
        assert "num_leaves_ratio" in result.best_params
        # The actual num_leaves was resolved internally — not stored in
        # best_params (it's a derived value applied inside objective)


class TestTrainingParamsPerTrial:
    def test_training_early_stopping_per_trial(self) -> None:
        """early_stopping_rounds from training category applied per trial."""
        config = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "target"},
            "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
            "model": {"name": "lgbm"},
            "training": {"seed": 0},
            "tuning": {
                "optuna": {
                    "params": {"n_trials": 2, "direction": "minimize"},
                    "space": {},
                }
            },
        }
        m = Model(config)
        result = m.tune(data=make_regression_df(n=100))
        assert isinstance(result, TuningResult)
        assert "early_stopping_rounds" in result.best_params

    def test_training_validation_ratio_per_trial(self) -> None:
        """validation_ratio from training category triggers inner_valid rebuild."""
        config = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "target"},
            "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
            "model": {"name": "lgbm"},
            "training": {"seed": 0},
            "tuning": {
                "optuna": {
                    "params": {"n_trials": 2, "direction": "minimize"},
                    "space": {},
                }
            },
        }
        m = Model(config)
        result = m.tune(data=make_regression_df(n=100))
        assert isinstance(result, TuningResult)
        assert "validation_ratio" in result.best_params


class TestBackwardCompat:
    def test_model_only_dims(self) -> None:
        """User-specified space with only model dims works as before."""
        config = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "target"},
            "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
            "model": {"name": "lgbm", "params": {"n_estimators": 10}},
            "training": {"seed": 0},
            "tuning": {
                "optuna": {
                    "params": {"n_trials": 2, "direction": "minimize"},
                    "space": {
                        "num_leaves": {"type": "int", "low": 8, "high": 32},
                    },
                }
            },
        }
        m = Model(config)
        result = m.tune(data=make_regression_df(n=100))
        assert isinstance(result, TuningResult)
        assert "num_leaves" in result.best_params
        # No smart/training params in user-specified space
        assert "num_leaves_ratio" not in result.best_params
        assert "early_stopping_rounds" not in result.best_params


class TestFixedParams:
    def test_fixed_params_applied_in_default_space(self) -> None:
        """Default space applies fixed_params (first_metric_only, metric)."""
        config = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "target"},
            "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
            "model": {"name": "lgbm"},
            "training": {"seed": 0},
            "tuning": {
                "optuna": {
                    "params": {"n_trials": 1, "direction": "minimize"},
                    "space": {},
                }
            },
        }
        m = Model(config)
        result = m.tune(data=make_regression_df(n=100))
        assert isinstance(result, TuningResult)
        # Fixed params are not in best_params (they are not search dims)
        assert "first_metric_only" not in result.best_params
        assert "auto_num_leaves" not in result.best_params
