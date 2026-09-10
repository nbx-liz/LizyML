"""H-0024 partial spaces retain provider defaults by parameter identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lizyml import Model
from lizyml.config.schema import LizyMLConfig
from lizyml.core.types.search_dim import CategoricalDim
from lizyml.estimators.lgbm.provider import LGBMProvider
from tests._helpers import make_regression_df


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("name", ["learning_rate", "eta", "shrinkage_rate"])
def test_partial_space_retains_defaults(task: str, name: str) -> None:
    provider = LGBMProvider()
    model = Model(
        {
            "config_version": 1,
            "task": task,
            "data": {"target": "target"},
            "model": {"name": "lgbm"},
            "tuning": {
                "optuna": {
                    "space": {
                        name: {"type": "float", "low": 0.02, "high": 0.04},
                    }
                }
            },
        }
    )
    space, _, _ = model._resolve_search_space(resume=False, provider=provider)
    expected = {dim.name for dim in provider.default_space(task)}
    expected.remove("learning_rate")
    expected.add(name)
    assert {dim.name for dim in space} == expected
    dim = next(dim for dim in space if dim.name == name)
    assert dim.low == 0.02
    assert dim.high == 0.04


def _config(space: dict, mode: str = "merge") -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 2},
        "model": {"name": "lgbm", "params": {"n_estimators": 5}},
        "tuning": {
            "optuna": {
                "params": {"n_trials": 1},
                "space": space,
                "space_mode": mode,
            }
        },
    }


@pytest.mark.parametrize("mode", ["merge", "replace"])
def test_mode_roundtrip(mode: str) -> None:
    cfg = LizyMLConfig.model_validate(_config({}, mode))
    restored = LizyMLConfig.model_validate_json(cfg.model_dump_json())
    assert restored.tuning.optuna.space_mode == mode


def test_unknown_mode_refused() -> None:
    with pytest.raises(ValueError, match="space_mode"):
        LizyMLConfig.model_validate(_config({}, "unknown"))


@pytest.mark.parametrize("has_space", [False, True])
def test_legacy_artifact_refit_policy(
    has_space: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from lizyml.persistence import loader

    space = (
        {"learning_rate": {"type": "categorical", "choices": [0.1]}}
        if has_space
        else {}
    )
    config = _config(space)
    del config["tuning"]["optuna"]["space_mode"]
    metadata = {
        "config": config,
        "tuning": {
            "best_model_params": {},
            "best_smart_params": {},
            "best_training_params": {},
            "best_score": 1.0,
            "metric_name": "rmse",
            "direction": "minimize",
        },
    }
    monkeypatch.setattr(
        loader,
        "load",
        lambda path: (
            SimpleNamespace(metrics={}),
            None,
            metadata,
            None,
        ),
    )
    model = Model.load("trusted-artifact-fixture")
    assert model._cfg.tuning.optuna.space_mode == ("replace" if has_space else "merge")
    params, _ = model._merge_params(LGBMProvider())
    if has_space:
        assert "first_metric_only" not in params
    else:
        assert params["first_metric_only"] is True


@pytest.mark.parametrize(
    "space",
    [
        {},
        {
            "learning_rate": {
                "type": "categorical",
                "choices": [0.1],
            }
        },
    ],
)
def test_replace_has_no_inherited_dimensions_or_fixed_params(space: dict) -> None:
    model = Model(_config(space, "replace"))
    dims, automatic_expansion, fixed = model._resolve_search_space(
        resume=False,
        provider=LGBMProvider(),
    )
    assert {dim.name for dim in dims} == set(space)
    assert not automatic_expansion
    assert fixed == {}


@pytest.mark.parametrize(
    "name,category,value",
    [
        ("num_leaves_ratio", "smart", 0.6),
        ("min_data_in_leaf_ratio", "smart", 0.1),
        ("early_stopping_rounds", "training", 4),
        ("validation_ratio", "training", 0.2),
        ("lambda_l1", "model", 0.5),
    ],
)
def test_override_categories_and_extension(
    name: str, category: str, value: float
) -> None:
    model = Model(
        _config(
            {
                name: {
                    "type": "categorical",
                    "choices": [value],
                    "category": category,
                }
            }
        )
    )
    provider = LGBMProvider()
    dims, automatic_expansion, fixed = model._resolve_search_space(
        resume=False,
        provider=provider,
    )
    expected = {dim.name for dim in provider.default_space("regression")} | {name}
    assert {dim.name for dim in dims} == expected
    assert next(dim for dim in dims if dim.name == name) == CategoricalDim(
        name,
        (value,),
        category=category,
    )
    assert not automatic_expansion
    assert fixed == provider.default_fixed_params("regression")


def test_real_tune_retains_defaults_and_resume_policy() -> None:
    model = Model(
        _config(
            {
                "num_iterations": {
                    "type": "categorical",
                    "choices": [5],
                }
            }
        )
    )
    data = make_regression_df(n=60)
    result = model.tune(data=data)
    expected = {dim.name for dim in LGBMProvider().default_space("regression")}
    expected.remove("n_estimators")
    expected.add("num_iterations")
    assert set(result.best_params) == expected
    assert result.best_model_params["num_iterations"] == 5
    assert set(result.best_training_params) == {
        "early_stopping_rounds",
        "validation_ratio",
    }
    previous_dims = list(model._space)
    previous_fixed = dict(model._tuning_fixed_params)
    model._cfg.tuning.optuna.space_mode = "replace"
    model._cfg.tuning.optuna.space = {}
    resumed = model.tune(data=data, resume=True, expand_boundary=False)
    assert model._space == previous_dims
    assert model._tuning_fixed_params == previous_fixed
    assert set(resumed.best_params) == expected
    # The following fit uses the fixed defaults from the actual tuning round.
    model.fit(data=data)
    params, _ = model._merge_params(LGBMProvider())
    assert params["first_metric_only"] is True


def test_duplicate_user_aliases_still_refused_before_study(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import optuna

    def unexpected_study(**kwargs: object) -> None:
        pytest.fail("invalid duplicate dimensions created a study")

    monkeypatch.setattr(optuna, "create_study", unexpected_study)
    spec = {"type": "categorical", "choices": [0.1]}
    model = Model(_config({"learning_rate": spec, "eta": spec}))
    from lizyml.core.exceptions import LizyMLError

    with pytest.raises(LizyMLError) as caught:
        model.tune(data=make_regression_df(n=60))
    assert caught.value.code.value == "CONFIG_INVALID"


def test_fresh_replace_trials_match_following_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config({"num_iterations": {"type": "categorical", "choices": [5]}})
    config["model"]["params"]["first_metric_only"] = False
    model = Model(config)
    data = make_regression_df(n=60)
    model.tune(data=data)
    model._cfg.tuning.optuna.space_mode = "replace"
    observed = []
    original = model._build_train_components

    def capture(*args: object, **kwargs: object) -> object:
        observed.append(dict(kwargs["model_params"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_build_train_components", capture)
    model.tune(data=data, resume=False, expand_boundary=False)
    trial_params = observed[-1]
    model.fit(data=data)
    assert (
        trial_params["first_metric_only"] == observed[-1]["first_metric_only"] is False
    )


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize(
    ("name", "category", "configured", "sampled"),
    [("learning_rate", "model", 0.01, 0.04), ("num_leaves_ratio", "smart", 0.6, 0.8)],
)
def test_removed_dimensions_match_trial_and_fit(
    monkeypatch: pytest.MonkeyPatch,
    resume: bool,
    name: str,
    category: str,
    configured: float,
    sampled: float,
) -> None:
    iterations = {"type": "categorical", "choices": [5]}
    config = _config(
        {
            "num_iterations": iterations,
            name: {"type": "categorical", "choices": [sampled], "category": category},
        }
    )
    surface = config["model"]["params"] if category == "model" else config["model"]
    surface[name] = configured
    model = Model(config)
    data = make_regression_df(n=60)
    model.tune(data=data)
    model._cfg.tuning.optuna.space_mode = "replace"
    model._cfg.tuning.optuna.space = {"num_iterations": iterations}
    observed = []
    original = model._build_train_components

    def capture(*args: object, **kwargs: object) -> object:
        observed.append(dict(kwargs[category + "_params"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_build_train_components", capture)
    result = model.tune(data=data, resume=resume, expand_boundary=False)
    trial_value = observed[-1][name]
    model.fit(data=data)
    assert trial_value == observed[-1][name] == (sampled if resume else configured)
    assert (name in result.best_params) is resume


@pytest.mark.parametrize("mode", ["merge", "replace"])
def test_export_load_refit_retains_fixed_policy(mode: str, tmp_path: Path) -> None:
    config = _config({"num_iterations": {"type": "categorical", "choices": [5]}}, mode)
    config["model"]["params"]["first_metric_only"] = False
    model = Model(config)
    data = make_regression_df(n=60)
    model.tune(data=data)
    model._cfg.tuning.optuna.space_mode = "replace" if mode == "merge" else "merge"
    model.tune(data=data, resume=True, expand_boundary=False)
    model.fit(data=data)
    expected, _ = model._merge_params(LGBMProvider())
    destination = model.export(tmp_path / "export")
    loaded = Model.load(destination)
    assert loaded._cfg.tuning.optuna.space_mode == model._cfg.tuning.optuna.space_mode
    loaded.fit(data=data)
    actual, _ = loaded._merge_params(LGBMProvider())
    assert actual["first_metric_only"] == expected["first_metric_only"]
    assert actual.get("metric") == expected.get("metric")
    assert loaded._tuning_fixed_params == model._tuning_fixed_params
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["tuning"]["fixed_params"] == model._tuning_fixed_params
    # Older artifacts omit the field and retain the config-based fallback.
    metadata["tuning"].pop("fixed_params")
    metadata_path.write_text(json.dumps(metadata))
    legacy = Model.load(destination)
    assert legacy._tuning_fixed_params is None
    from lizyml.core.exceptions import ErrorCode, LizyMLError

    metadata["tuning"]["fixed_params"] = []
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(LizyMLError) as error:
        Model.load(destination)
    assert error.value.code == ErrorCode.DESERIALIZATION_FAILED


def test_failed_fresh_round_preserves_previous_fit_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        {
            "num_iterations": {"type": "categorical", "choices": [5]},
            "learning_rate": {"type": "categorical", "choices": [0.04]},
        }
    )
    config["model"]["params"]["learning_rate"] = 0.01
    model = Model(config)
    data = make_regression_df(n=60)
    model.tune(data=data)
    previous_result = model._tuning_result
    previous_fixed = dict(model._tuning_fixed_params)
    previous_space = list(model._space)
    model._cfg.tuning.optuna.space_mode = "replace"
    model._cfg.tuning.optuna.space = {}

    def fail_round(*args: object, **kwargs: object) -> None:
        raise RuntimeError("controlled study failure")

    monkeypatch.setattr(model, "_run_tune_round", fail_round)
    with pytest.raises(RuntimeError, match="controlled study failure"):
        model.tune(data=data, resume=False, expand_boundary=False)
    assert model._tuning_result is previous_result
    assert model._tuning_fixed_params == previous_fixed
    assert model._space == previous_space
    model.fit(data=data)
    params, _ = model._merge_params(LGBMProvider())
    assert params["learning_rate"] == 0.04
    assert params["first_metric_only"] is True
