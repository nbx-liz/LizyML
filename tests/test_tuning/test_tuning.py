"""Tests for Phase 12 — Tuning.

Covers:
- SearchSpace: parse_space converts dict to typed dims
- parse_space raises CONFIG_INVALID for unknown type and missing choices
- suggest_params samples correct param types from optuna trial
- Model.tune() with no tuning config raises CONFIG_INVALID
- Model.tune() E2E: returns best_params dict with expected keys
- Model.tune() → model.fit() uses best_params automatically
- TUNING_FAILED propagation when study raises
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.tuning.search_space import (
    CategoricalDim,
    FloatDim,
    IntDim,
    parse_space,
    suggest_params,
)

# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------


def _reg_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"]
    return df


def _reg_config_no_tuning() -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


def _reg_config_with_tuning(n_trials: int = 3) -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
        "tuning": {
            "optuna": {
                "params": {"n_trials": n_trials, "direction": "minimize"},
                "space": {
                    "num_leaves": {"type": "int", "low": 8, "high": 32},
                    "learning_rate": {
                        "type": "float",
                        "low": 0.01,
                        "high": 0.3,
                        "log": True,
                    },
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# SearchSpace unit tests
# ---------------------------------------------------------------------------


class TestParseSpace:
    def test_float_dim(self) -> None:
        dims = parse_space({"lr": {"type": "float", "low": 0.01, "high": 0.3}})
        assert len(dims) == 1
        assert isinstance(dims[0], FloatDim)
        assert dims[0].name == "lr"
        assert dims[0].low == 0.01
        assert dims[0].high == 0.3
        assert dims[0].log is False

    def test_float_dim_log(self) -> None:
        dims = parse_space(
            {"lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True}}
        )
        assert isinstance(dims[0], FloatDim)
        assert dims[0].log is True

    def test_int_dim(self) -> None:
        dims = parse_space({"num_leaves": {"type": "int", "low": 16, "high": 128}})
        assert isinstance(dims[0], IntDim)
        assert dims[0].name == "num_leaves"
        assert dims[0].low == 16
        assert dims[0].high == 128

    def test_categorical_dim(self) -> None:
        dims = parse_space(
            {"subsample": {"type": "categorical", "choices": [0.6, 0.8, 1.0]}}
        )
        assert isinstance(dims[0], CategoricalDim)
        assert dims[0].choices == (0.6, 0.8, 1.0)

    def test_multiple_dims(self) -> None:
        space = {
            "lr": {"type": "float", "low": 0.01, "high": 0.3},
            "n": {"type": "int", "low": 8, "high": 64},
            "sub": {"type": "categorical", "choices": [0.8, 1.0]},
        }
        dims = parse_space(space)
        assert len(dims) == 3

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            parse_space({"lr": {"type": "unknown", "low": 0.01, "high": 0.3}})
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_categorical_empty_choices_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            parse_space({"x": {"type": "categorical", "choices": []}})
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_empty_space_returns_empty_list(self) -> None:
        dims = parse_space({})
        assert dims == []


class TestSuggestParams:
    def test_float_suggestion(self) -> None:
        dims = [FloatDim("lr", 0.01, 0.3)]
        trial = MagicMock()
        trial.suggest_float.return_value = 0.05
        params = suggest_params(trial, dims)
        trial.suggest_float.assert_called_once_with("lr", 0.01, 0.3, log=False)
        assert params["lr"] == 0.05

    def test_int_suggestion(self) -> None:
        dims = [IntDim("leaves", 16, 128)]
        trial = MagicMock()
        trial.suggest_int.return_value = 64
        params = suggest_params(trial, dims)
        trial.suggest_int.assert_called_once_with("leaves", 16, 128, log=False)
        assert params["leaves"] == 64

    def test_categorical_suggestion(self) -> None:
        dims = [CategoricalDim("sub", (0.8, 1.0))]
        trial = MagicMock()
        trial.suggest_categorical.return_value = 0.8
        params = suggest_params(trial, dims)
        trial.suggest_categorical.assert_called_once_with("sub", (0.8, 1.0))
        assert params["sub"] == 0.8


# ---------------------------------------------------------------------------
# Model.tune() tests
# ---------------------------------------------------------------------------


class TestModelTune:
    def test_tune_without_tuning_config_raises(self) -> None:
        m = Model(_reg_config_no_tuning())
        with pytest.raises(LizyMLError) as exc_info:
            m.tune(data=_reg_df())
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_tune_returns_best_params_dict(self) -> None:
        m = Model(_reg_config_with_tuning(n_trials=3))
        best = m.tune(data=_reg_df())
        assert isinstance(best, dict)
        assert "num_leaves" in best
        assert "learning_rate" in best

    def test_tune_then_fit_uses_best_params(self) -> None:
        """After tune(), fit() should succeed and use the stored best_params."""
        df = _reg_df()
        m = Model(_reg_config_with_tuning(n_trials=2))
        best = m.tune(data=df)
        # fit() should pick up _best_params automatically
        result = m.fit(data=df)
        from lizyml.core.types.fit_result import FitResult

        assert isinstance(result, FitResult)
        # best_params are stored
        assert m._best_params == best

    def test_tune_stores_best_params_internally(self) -> None:
        m = Model(_reg_config_with_tuning(n_trials=2))
        assert m._best_params is None
        m.tune(data=_reg_df())
        assert m._best_params is not None

    def test_tune_with_no_data_raises(self) -> None:
        m = Model(_reg_config_with_tuning(n_trials=2))
        with pytest.raises(LizyMLError) as exc_info:
            m.tune()  # no data
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID

    def test_tuning_failed_propagation(self) -> None:
        """When optuna study raises, TUNING_FAILED is re-raised."""
        m = Model(_reg_config_with_tuning(n_trials=3))
        df = _reg_df()

        # patch _optuna (the module-level variable) so study.optimize raises
        with patch("lizyml.tuning.tuner._optuna") as mock_optuna:
            mock_study = MagicMock()
            mock_study.optimize.side_effect = RuntimeError("boom")
            mock_optuna.samplers.TPESampler.return_value = MagicMock()
            mock_optuna.create_study.return_value = mock_study

            with pytest.raises(LizyMLError) as exc_info:
                m.tune(data=df)
            assert exc_info.value.code == ErrorCode.TUNING_FAILED

    def test_optional_dep_missing(self) -> None:
        """OPTIONAL_DEP_MISSING when optuna is not available."""
        m = Model(_reg_config_with_tuning(n_trials=1))
        with patch("lizyml.tuning.tuner._optuna", None):
            with pytest.raises(LizyMLError) as exc_info:
                m.tune(data=_reg_df())
            assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING
