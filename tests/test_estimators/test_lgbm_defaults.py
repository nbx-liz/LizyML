"""Tests for H-0022 — LightGBM default parameter profile."""

from __future__ import annotations

from lizyml.estimators.lgbm import LGBMAdapter


class TestTaskDefaults:
    def test_regression_objective_huber(self) -> None:
        adapter = LGBMAdapter(task="regression")
        params = adapter._build_params()
        assert params["objective"] == "huber"

    def test_regression_metric_list(self) -> None:
        adapter = LGBMAdapter(task="regression")
        params = adapter._build_params()
        assert params["metric"] == ["huber", "mae", "mape"]

    def test_binary_objective(self) -> None:
        adapter = LGBMAdapter(task="binary")
        params = adapter._build_params()
        assert params["objective"] == "binary"

    def test_binary_metric_list(self) -> None:
        adapter = LGBMAdapter(task="binary")
        params = adapter._build_params()
        assert params["metric"] == ["auc", "binary_logloss"]

    def test_multiclass_objective(self) -> None:
        adapter = LGBMAdapter(task="multiclass", num_class=3)
        params = adapter._build_params()
        assert params["objective"] == "multiclass"

    def test_multiclass_metric_list(self) -> None:
        adapter = LGBMAdapter(task="multiclass", num_class=3)
        params = adapter._build_params()
        assert params["metric"] == ["auc_mu", "multi_logloss"]


class TestCommonDefaults:
    def test_learning_rate(self) -> None:
        params = LGBMAdapter(task="regression")._build_params()
        assert params["learning_rate"] == 0.001

    def test_max_depth(self) -> None:
        params = LGBMAdapter(task="regression")._build_params()
        assert params["max_depth"] == 5

    def test_max_bin(self) -> None:
        params = LGBMAdapter(task="regression")._build_params()
        assert params["max_bin"] == 511

    def test_n_estimators(self) -> None:
        params = LGBMAdapter(task="regression")._build_params()
        assert params["n_estimators"] == 1500

    def test_feature_fraction(self) -> None:
        params = LGBMAdapter(task="regression")._build_params()
        assert params["feature_fraction"] == 0.7

    def test_bagging_fraction(self) -> None:
        params = LGBMAdapter(task="regression")._build_params()
        assert params["bagging_fraction"] == 0.7


class TestUserOverride:
    def test_params_override_defaults(self) -> None:
        adapter = LGBMAdapter(task="regression", params={"learning_rate": 0.1})
        adapter._feature_names = ["a"]
        built = adapter._build_params()
        built.update(adapter.params)
        assert built["learning_rate"] == 0.1
        assert built["max_depth"] == 5  # non-overridden default remains
