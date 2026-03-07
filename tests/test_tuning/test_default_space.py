"""Tests for default_space(), default_fixed_params(), split_by_category() (H-0024)."""

from __future__ import annotations

import pytest

from lizyml import Model
from lizyml.core.types.tuning_result import TuningResult
from lizyml.tuning.search_space import (
    CategoricalDim,
    FloatDim,
    IntDim,
    default_fixed_params,
    default_space,
    split_by_category,
)
from tests._helpers import make_regression_df

# ---------------------------------------------------------------------------
# Unit tests — default_space()
# ---------------------------------------------------------------------------


class TestDefaultSpace:
    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_returns_10_dims(self, task: str) -> None:
        dims = default_space(task)
        assert len(dims) == 10

    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_categories_present(self, task: str) -> None:
        dims = default_space(task)
        cats = {d.category for d in dims}
        assert cats == {"model", "smart", "training"}

    def test_contains_learning_rate(self) -> None:
        dims = default_space("regression")
        names = [d.name for d in dims]
        assert "learning_rate" in names

    def test_contains_num_leaves_ratio(self) -> None:
        dims = default_space("regression")
        names = [d.name for d in dims]
        assert "num_leaves_ratio" in names

    def test_regression_objective_choices(self) -> None:
        dims = default_space("regression")
        obj_dim = next(d for d in dims if d.name == "objective")
        assert isinstance(obj_dim, CategoricalDim)
        assert obj_dim.choices == ("huber", "fair")

    def test_binary_objective_choices(self) -> None:
        dims = default_space("binary")
        obj_dim = next(d for d in dims if d.name == "objective")
        assert isinstance(obj_dim, CategoricalDim)
        assert obj_dim.choices == ("binary",)

    def test_multiclass_objective_choices(self) -> None:
        dims = default_space("multiclass")
        obj_dim = next(d for d in dims if d.name == "objective")
        assert isinstance(obj_dim, CategoricalDim)
        assert obj_dim.choices == ("multiclass", "multiclassova")

    def test_learning_rate_log_scale(self) -> None:
        dims = default_space("regression")
        lr = next(d for d in dims if d.name == "learning_rate")
        assert isinstance(lr, FloatDim)
        assert lr.log is True
        assert lr.low == 0.0001
        assert lr.high == 0.1

    def test_n_estimators_range(self) -> None:
        dims = default_space("regression")
        ne = next(d for d in dims if d.name == "n_estimators")
        assert isinstance(ne, IntDim)
        assert ne.low == 600
        assert ne.high == 2500

    def test_max_depth_range(self) -> None:
        dims = default_space("regression")
        md = next(d for d in dims if d.name == "max_depth")
        assert isinstance(md, IntDim)
        assert md.low == 3
        assert md.high == 12

    def test_smart_dims_ranges(self) -> None:
        dims = default_space("regression")
        nlr = next(d for d in dims if d.name == "num_leaves_ratio")
        assert isinstance(nlr, FloatDim)
        assert nlr.low == 0.5
        assert nlr.high == 1.0
        assert nlr.category == "smart"

        mdlr = next(d for d in dims if d.name == "min_data_in_leaf_ratio")
        assert isinstance(mdlr, FloatDim)
        assert mdlr.low == 0.01
        assert mdlr.high == 0.2
        assert mdlr.category == "smart"

    def test_training_dims_ranges(self) -> None:
        dims = default_space("regression")
        esr = next(d for d in dims if d.name == "early_stopping_rounds")
        assert isinstance(esr, IntDim)
        assert esr.low == 40
        assert esr.high == 240
        assert esr.category == "training"

        vr = next(d for d in dims if d.name == "validation_ratio")
        assert isinstance(vr, FloatDim)
        assert vr.low == 0.1
        assert vr.high == 0.3
        assert vr.category == "training"


# ---------------------------------------------------------------------------
# Unit tests — default_fixed_params()
# ---------------------------------------------------------------------------


class TestDefaultFixedParams:
    def test_regression(self) -> None:
        fp = default_fixed_params("regression")
        assert fp["auto_num_leaves"] is True
        assert fp["first_metric_only"] is True
        assert fp["metric"] == ["huber", "mae", "mape"]

    def test_binary(self) -> None:
        fp = default_fixed_params("binary")
        assert fp["metric"] == ["auc", "binary_logloss"]

    def test_multiclass(self) -> None:
        fp = default_fixed_params("multiclass")
        assert fp["metric"] == ["auc_mu", "multi_logloss"]


# ---------------------------------------------------------------------------
# Unit tests — split_by_category()
# ---------------------------------------------------------------------------


class TestSplitByCategory:
    def test_splits_correctly(self) -> None:
        dims = default_space("regression")
        params = {
            "objective": "huber",
            "learning_rate": 0.01,
            "num_leaves_ratio": 0.8,
            "early_stopping_rounds": 100,
            "validation_ratio": 0.2,
        }
        model_p, smart_p, training_p = split_by_category(params, dims)
        assert "objective" in model_p
        assert "learning_rate" in model_p
        assert "num_leaves_ratio" in smart_p
        assert "early_stopping_rounds" in training_p
        assert "validation_ratio" in training_p

    def test_unknown_param_defaults_to_model(self) -> None:
        dims = default_space("regression")
        params = {"unknown_param": 42}
        model_p, smart_p, training_p = split_by_category(params, dims)
        assert model_p == {"unknown_param": 42}
        assert smart_p == {}
        assert training_p == {}


# ---------------------------------------------------------------------------
# Integration — empty space triggers default_space
# ---------------------------------------------------------------------------


class TestDefaultSpaceE2E:
    def test_empty_space_uses_default(self) -> None:
        """When space={}, tune() should use default_space() automatically."""
        config = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "target"},
            "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
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
        # Default space includes learning_rate and smart/training dims
        assert "learning_rate" in result.best_params
        assert "num_leaves_ratio" in result.best_params
        assert "early_stopping_rounds" in result.best_params

    def test_user_space_overrides_default(self) -> None:
        """When user provides space, it should be used instead of defaults."""
        config = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "target"},
            "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
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
        # Only num_leaves should be in best_params (user-specified space)
        assert "num_leaves" in result.best_params
        assert "learning_rate" not in result.best_params
