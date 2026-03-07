"""Tests for H-0022 — LightGBM default parameter profile."""

from __future__ import annotations

from lizyml.estimators.lgbm import LGBMAdapter


class TestTaskDefaults:
    def test_regression_objective_huber(self) -> None:
        params, _ = LGBMAdapter(task="regression")._build_params()
        assert params["objective"] == "huber"

    def test_regression_metric_list(self) -> None:
        params, _ = LGBMAdapter(task="regression")._build_params()
        assert params["metric"] == ["huber", "mae", "mape"]

    def test_binary_objective(self) -> None:
        params, _ = LGBMAdapter(task="binary")._build_params()
        assert params["objective"] == "binary"

    def test_binary_metric_list(self) -> None:
        params, _ = LGBMAdapter(task="binary")._build_params()
        assert params["metric"] == ["auc", "binary_logloss"]

    def test_multiclass_objective(self) -> None:
        params, _ = LGBMAdapter(task="multiclass", num_class=3)._build_params()
        assert params["objective"] == "multiclass"

    def test_multiclass_metric_list(self) -> None:
        params, _ = LGBMAdapter(task="multiclass", num_class=3)._build_params()
        assert params["metric"] == ["auc_mu", "multi_logloss"]


class TestCommonDefaults:
    def test_learning_rate(self) -> None:
        params, _ = LGBMAdapter(task="regression")._build_params()
        assert params["learning_rate"] == 0.001

    def test_max_depth(self) -> None:
        params, _ = LGBMAdapter(task="regression")._build_params()
        assert params["max_depth"] == 5

    def test_max_bin(self) -> None:
        params, _ = LGBMAdapter(task="regression")._build_params()
        assert params["max_bin"] == 511

    def test_num_boost_round(self) -> None:
        _, num_boost_round = LGBMAdapter(task="regression")._build_params()
        assert num_boost_round == 1500

    def test_feature_fraction(self) -> None:
        params, _ = LGBMAdapter(task="regression")._build_params()
        assert params["feature_fraction"] == 0.7

    def test_bagging_fraction(self) -> None:
        params, _ = LGBMAdapter(task="regression")._build_params()
        assert params["bagging_fraction"] == 0.7

    def test_seed_from_random_state(self) -> None:
        params, _ = LGBMAdapter(task="regression", random_state=99)._build_params()
        assert params["seed"] == 99
        assert "random_state" not in params

    def test_verbosity(self) -> None:
        params, _ = LGBMAdapter(task="regression")._build_params()
        assert params["verbosity"] == -1
        assert "verbose" not in params


class TestUserOverride:
    def test_params_override_defaults(self) -> None:
        adapter = LGBMAdapter(task="regression", params={"learning_rate": 0.1})
        params, _ = adapter._build_params()
        assert params["learning_rate"] == 0.1
        assert params["max_depth"] == 5  # non-overridden default remains

    def test_n_estimators_override(self) -> None:
        adapter = LGBMAdapter(task="regression", params={"n_estimators": 500})
        params, num_boost_round = adapter._build_params()
        assert num_boost_round == 500
        assert "n_estimators" not in params

    def test_random_state_in_user_params_normalized(self) -> None:
        params, _ = LGBMAdapter(
            task="regression", params={"random_state": 77}
        )._build_params()
        assert params["seed"] == 77
        assert "random_state" not in params

    def test_verbose_in_user_params_normalized(self) -> None:
        params, _ = LGBMAdapter(
            task="regression", params={"verbose": 1}
        )._build_params()
        assert params["verbosity"] == 1
        assert "verbose" not in params

    def test_seed_takes_priority_over_random_state(self) -> None:
        params, _ = LGBMAdapter(
            task="regression", params={"random_state": 77, "seed": 88}
        )._build_params()
        assert params["seed"] == 88
        assert "random_state" not in params

    def test_no_deprecated_keys_in_output(self) -> None:
        params, _ = LGBMAdapter(
            task="regression",
            params={"n_estimators": 500, "random_state": 1, "verbose": 0},
        )._build_params()
        for key in ("n_estimators", "random_state", "verbose"):
            assert key not in params
