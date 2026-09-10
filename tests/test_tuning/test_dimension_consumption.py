"""Search dimensions reach consumers or are refused before study startup."""

from unittest.mock import patch

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.lgbm.provider import LGBMProvider
from tests._helpers import make_binary_df, make_config
from tests._train_spy import record_lightgbm_calls

SMART = {
    "auto_num_leaves": True,
    "min_data_in_leaf_ratio": 0.1,
    "min_data_in_bin_ratio": 0.1,
    "feature_weights": {"feat_a": 1.0},
    "balanced": True,
}
CLAIMED = sorted(LGBMProvider().smart_managed_param_names(SMART, "binary"))


def config_for(name, category="model"):
    config = make_config("binary", n_estimators=5, n_splits=2, tuning_n_trials=1)
    config["tuning"]["optuna"]["space"] = {
        name: {"type": "categorical", "choices": [7], "category": category}
    }
    return config


@pytest.mark.parametrize("name", CLAIMED)
def test_every_smart_owned_native_spelling_refused_before_study(name):
    config = config_for(name)
    config["model"].update(SMART)
    with patch("optuna.create_study") as study, record_lightgbm_calls() as seen:
        with pytest.raises(LizyMLError) as failure:
            Model(config, data=make_binary_df(n=90)).tune()
        assert failure.value.code == ErrorCode.CONFIG_INVALID
        assert "tuning.optuna.space" in failure.value.user_message
        assert name in failure.value.user_message
        study.assert_not_called()
        assert seen["train_params"] == []


@pytest.mark.parametrize("name", ["seed", "n_jobs", "unknown_setting"])
def test_unconsumed_training_dimension_refused(name):
    with patch("optuna.create_study") as study:
        with pytest.raises(LizyMLError) as failure:
            Model(config_for(name, "training"), data=make_binary_df(n=90)).tune()
        assert failure.value.code == ErrorCode.CONFIG_INVALID
        assert name in failure.value.context["names"]
        study.assert_not_called()


def test_native_leaf_dimension_reaches_training_with_owner_disabled():
    config = config_for("num_leaves")
    config["model"]["auto_num_leaves"] = False
    with record_lightgbm_calls() as seen:
        model = Model(config, data=make_binary_df(n=90))
        result = model.tune()
        model.fit()
    assert result.best_model_params["num_leaves"] == 7
    assert seen["train_params"]
    assert all(params["num_leaves"] == 7 for params in seen["train_params"])


def test_search_dimension_cannot_activate_conflicting_smart_owner():
    config = config_for("num_leaves")
    config["model"]["auto_num_leaves"] = False
    config["tuning"]["optuna"]["space"]["auto_num_leaves"] = {
        "type": "categorical",
        "choices": [False, True],
        "category": "smart",
    }
    with patch("optuna.create_study") as study:
        with pytest.raises(LizyMLError) as failure:
            Model(config, data=make_binary_df(n=90)).tune()
        assert failure.value.code == ErrorCode.CONFIG_INVALID
        study.assert_not_called()
