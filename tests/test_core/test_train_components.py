"""Tests for TrainComponents, _merge_params, _build_train_components (H-0050 Phase 3-4).

TDD RED phase: these tests define the target API and should fail until
the implementation is added.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lizyml import Model
from lizyml.core.types.tuning_result import TuningResult
from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.training.inner_valid import BaseInnerValidStrategy
from tests._helpers import make_config, make_regression_df

# ---------------------------------------------------------------------------
# Unit tests — TrainComponents dataclass
# ---------------------------------------------------------------------------


class TestTrainComponents:
    def test_import(self) -> None:
        from lizyml.core.train_components import TrainComponents

        assert TrainComponents is not None

    def test_fields(self) -> None:
        from lizyml.core.train_components import TrainComponents

        tc = TrainComponents(
            estimator_factory=lambda: None,  # type: ignore[arg-type]
            sample_weight=None,
            ratio_resolver=None,
            inner_valid=None,  # type: ignore[arg-type]
        )
        assert tc.estimator_factory is not None
        assert tc.sample_weight is None
        assert tc.ratio_resolver is None
        assert tc.inner_valid is None

    def test_fields_with_values(self) -> None:
        from lizyml.core.train_components import TrainComponents

        sw = np.array([1.0, 2.0])

        def resolver(n: int) -> dict[str, int]:
            return {"min_data_in_leaf": max(1, int(n * 0.01))}

        tc = TrainComponents(
            estimator_factory=lambda: None,  # type: ignore[arg-type]
            sample_weight=sw,
            ratio_resolver=resolver,
            inner_valid=None,  # type: ignore[arg-type]
        )
        assert tc.sample_weight is not None
        assert tc.ratio_resolver is not None
        assert tc.ratio_resolver(1000) == {"min_data_in_leaf": 10}


# ---------------------------------------------------------------------------
# Unit tests — _merge_params
# ---------------------------------------------------------------------------


class TestMergeParams:
    """Test Model._merge_params priority: Config < tune best < fit args."""

    def _make_model(self, **model_overrides: Any) -> Model:
        cfg = make_config("regression", n_estimators=10, **model_overrides)
        return Model(cfg)

    def test_config_defaults_only(self) -> None:
        m = self._make_model()
        model_params, smart_params = m._merge_params()
        assert model_params["n_estimators"] == 10
        # smart_params should contain the LGBMConfig smart fields
        assert "auto_num_leaves" in smart_params

    def test_tune_best_overrides_config(self) -> None:
        m = self._make_model(learning_rate=0.1)
        # Simulate tune() having been called
        m._tuning_result = TuningResult(
            best_model_params={"learning_rate": 0.05},
            best_smart_params={},
            best_training_params={},
            best_score=0.5,
            trials=[],
            metric_name="rmse",
            direction="minimize",
        )
        model_params, _ = m._merge_params()
        assert model_params["learning_rate"] == 0.05

    def test_fit_args_override_tune_best(self) -> None:
        m = self._make_model(learning_rate=0.1)
        m._tuning_result = TuningResult(
            best_model_params={"learning_rate": 0.05},
            best_smart_params={},
            best_training_params={},
            best_score=0.5,
            trials=[],
            metric_name="rmse",
            direction="minimize",
        )
        model_params, _ = m._merge_params(override={"learning_rate": 0.2})
        assert model_params["learning_rate"] == 0.2

    def test_smart_params_from_tune(self) -> None:
        m = self._make_model()
        m._tuning_result = TuningResult(
            best_model_params={},
            best_smart_params={"num_leaves_ratio": 0.5},
            best_training_params={},
            best_score=0.5,
            trials=[],
            metric_name="rmse",
            direction="minimize",
        )
        _, smart_params = m._merge_params()
        assert smart_params["num_leaves_ratio"] == 0.5

    def test_no_tuning_result_returns_config_smart(self) -> None:
        m = self._make_model()
        _, smart_params = m._merge_params()
        # Default LGBMConfig has auto_num_leaves=True, num_leaves_ratio=1.0
        assert smart_params["auto_num_leaves"] is True
        assert smart_params["num_leaves_ratio"] == 1.0


# ---------------------------------------------------------------------------
# Unit tests — _build_train_components
# ---------------------------------------------------------------------------


class TestBuildTrainComponents:
    def test_returns_train_components(self) -> None:
        from lizyml.core.train_components import TrainComponents

        m = Model(make_config("regression", n_estimators=10))
        df = make_regression_df(n=100)
        X = df[["feat_a", "feat_b"]]
        y = df["target"]

        model_params, smart_params = m._merge_params()
        tc = m._build_train_components(
            X, y, model_params=model_params, smart_params=smart_params
        )
        assert isinstance(tc, TrainComponents)

    def test_estimator_factory_produces_adapter(self) -> None:
        m = Model(make_config("regression", n_estimators=10))
        df = make_regression_df(n=100)
        X = df[["feat_a", "feat_b"]]
        y = df["target"]

        model_params, smart_params = m._merge_params()
        tc = m._build_train_components(
            X, y, model_params=model_params, smart_params=smart_params
        )
        estimator = tc.estimator_factory()
        assert isinstance(estimator, BaseEstimatorAdapter)

    def test_inner_valid_is_strategy(self) -> None:
        m = Model(make_config("regression", n_estimators=10))
        df = make_regression_df(n=100)
        X = df[["feat_a", "feat_b"]]
        y = df["target"]

        model_params, smart_params = m._merge_params()
        tc = m._build_train_components(
            X, y, model_params=model_params, smart_params=smart_params
        )
        assert isinstance(tc.inner_valid, BaseInnerValidStrategy)

    def test_ratio_resolver_when_ratio_set(self) -> None:
        cfg = make_config("regression", n_estimators=10)
        cfg["model"]["min_data_in_leaf_ratio"] = 0.01
        m = Model(cfg)
        df = make_regression_df(n=100)
        X = df[["feat_a", "feat_b"]]
        y = df["target"]

        model_params, smart_params = m._merge_params()
        tc = m._build_train_components(
            X, y, model_params=model_params, smart_params=smart_params
        )
        assert tc.ratio_resolver is not None
        resolved = tc.ratio_resolver(10000)
        assert "min_data_in_leaf" in resolved

    def test_sample_weight_for_balanced_binary(self) -> None:
        from tests._helpers import make_binary_df

        cfg = make_config("binary", n_estimators=10)
        cfg["model"]["balanced"] = True
        m = Model(cfg)
        df = make_binary_df(n=100)
        X = df[["feat_a", "feat_b"]]
        y = df["target"]

        model_params, smart_params = m._merge_params()
        # balanced=True with binary → scale_pos_weight, sample_weight=None
        tc = m._build_train_components(
            X, y, model_params=model_params, smart_params=smart_params
        )
        # For binary, balanced sets scale_pos_weight, not sample_weight
        assert tc.sample_weight is None


# ---------------------------------------------------------------------------
# Integration — fit uses TrainComponents (Phase 4)
# ---------------------------------------------------------------------------


class TestFitUsesTrainComponents:
    """Verify fit() still works correctly after TrainComponents rewrite."""

    def test_fit_regression_basic(self) -> None:
        m = Model(make_config("regression", n_estimators=5, n_splits=2))
        result = m.fit(data=make_regression_df(n=100))
        assert result.oof_pred is not None
        assert len(result.models) == 2

    def test_fit_after_tune_uses_best_params(self) -> None:
        """tune() → fit() should use tuned params (no _best_params)."""
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        cfg["tuning"] = {
            "optuna": {
                "params": {"n_trials": 2, "direction": "minimize"},
                "space": {
                    "learning_rate": {
                        "type": "float",
                        "low": 0.01,
                        "high": 0.3,
                        "log": True,
                    },
                },
            }
        }
        m = Model(cfg)
        df = make_regression_df(n=100)
        m.tune(data=df)

        # _best_params should no longer exist; _tuning_result is used
        assert not hasattr(m, "_best_params") or m._best_params is None
        assert m._tuning_result is not None

        fit_result = m.fit(data=df)
        assert fit_result.oof_pred is not None

    def test_fit_params_override(self) -> None:
        m = Model(make_config("regression", n_estimators=5, n_splits=2))
        result = m.fit(
            data=make_regression_df(n=100),
            params={"learning_rate": 0.5},
        )
        assert result.oof_pred is not None
