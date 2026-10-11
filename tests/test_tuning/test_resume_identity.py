"""H-0122 §4 / §5.1: what a resume must keep, and what it may change.

Criteria 1 (objective elements refused, non-identity changes pass), 2 (legacy
studies), 3 (version warning), 11 (config space edits), 13 and 26 (data
identity), 27 (the mismatch names its element) and 28 (a provider ``fixed``
change is an objective change, versions or not).

Each refusal is also checked to leave the study and the Model unchanged.
"""

from __future__ import annotations

import copy
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TuneProgressInfo
from lizyml.estimators.lgbm import defaults as lgbm_defaults
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.tuning.lineage import STUDY_KEY
from tests.test_tuning._resume_helpers import (
    delete_attrs,
    fast_config,
    load,
    snapshot_model,
    snapshot_study,
    url,
)

_EXCLUDED = ["cat", "mix", "obj", "num", "extra"]


def _config(mode: str = "replace") -> dict[str, Any]:
    cfg = fast_config("binary", mode=mode)
    cfg["evaluation"] = {"metrics": ["logloss"]}
    cfg["features"] = {"exclude": list(_EXCLUDED)}
    return cfg


def _frame(n: int = 80) -> pd.DataFrame:
    """Two features and a target, plus excluded columns of every kind §4.1 names."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    df["cat"] = pd.Categorical(["a", "b"] * (n // 2), categories=["a", "b"])
    df["mix"] = pd.Categorical([1, "1"] * (n // 2), categories=[1, "1"])
    df["obj"] = pd.Series([1, 2] * (n // 2), dtype=object)
    df["num"] = np.arange(n, dtype="int64")
    df["extra"] = rng.normal(size=n)
    return df


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A stored study in state C with two rounds, built once per module."""
    root = tmp_path_factory.mktemp("identity_template")
    storage = url(root)
    Model(_config()).tune(data=_frame(), storage=storage, study_name="s")
    Model(_config()).tune(data=_frame(), resume=True, storage=storage, study_name="s")
    return root / "s.db"


@pytest.fixture
def storage(template: Path, tmp_path: Path) -> str:
    shutil.copy(template, tmp_path / "s.db")
    return url(tmp_path)


def _refused_on(
    storage: str, cfg: dict[str, Any], df: pd.DataFrame, **kwargs: Any
) -> LizyMLError:
    """Resume the stored study from a new Model; it must refuse and change nothing."""
    model = Model(cfg)
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with pytest.raises(LizyMLError) as caught:
        model.tune(data=df, resume=True, storage=storage, study_name="s", **kwargs)
    assert caught.value.code is ErrorCode.CONFIG_INVALID, caught.value
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before
    return caught.value


# ---------------------------------------------------------------------------
# Criterion 1 / 27: each objective element, one at a time
# ---------------------------------------------------------------------------


def _data_value(cfg: dict[str, Any], df: pd.DataFrame) -> None:
    df.loc[0, "feat_a"] += 1.0


def _row_order(cfg: dict[str, Any], df: pd.DataFrame) -> None:
    reversed_ = df.iloc[::-1]
    df[:] = reversed_.to_numpy()
    df.index = reversed_.index


def _n_splits(cfg: dict[str, Any], df: pd.DataFrame) -> None:
    cfg["split"]["n_splits"] = 3


def _seed(cfg: dict[str, Any], df: pd.DataFrame) -> None:
    cfg["training"]["seed"] = 1


def _model_param(cfg: dict[str, Any], df: pd.DataFrame) -> None:
    cfg["model"]["params"]["lambda_l1"] = 0.1


def _smart_value(cfg: dict[str, Any], df: pd.DataFrame) -> None:
    cfg["model"]["num_leaves_ratio"] = 0.5


def _exclude(cfg: dict[str, Any], df: pd.DataFrame) -> None:
    cfg["features"]["exclude"].append("feat_b")


def _first_metric(cfg: dict[str, Any], df: pd.DataFrame) -> None:
    cfg["evaluation"]["metrics"] = ["brier"]


ELEMENT_CHANGES: dict[str, tuple[Callable[..., None], str]] = {
    "data_value": (_data_value, "data.values"),
    "row_order": (_row_order, "data.values"),
    "split_n_splits": (_n_splits, "config.split"),
    "training_seed": (_seed, "config.training"),
    "model_params": (_model_param, "config.model"),
    "smart_value": (_smart_value, "config.model"),
    "features_exclude": (_exclude, "config.features"),
    "first_metric": (_first_metric, "metric"),
}


@pytest.mark.parametrize("change", sorted(ELEMENT_CHANGES))
def test_objective_element_change_refused(storage: str, change: str) -> None:
    edit, element = ELEMENT_CHANGES[change]
    cfg, df = _config(), _frame()
    edit(cfg, df)
    error = _refused_on(storage, cfg, df)
    assert error.context["surface"] == "tuning.study"
    assert element in error.context["mismatched"]
    assert error.context["study_name"] == "s"
    assert element in error.user_message


@pytest.mark.parametrize(
    ("change", "expected"),
    [("data", "data.values"), ("config", "config.model"), ("metric", "metric")],
)
def test_mismatch_context_names_element(
    storage: str, change: str, expected: str
) -> None:
    """Criterion 27: one change, and exactly its element is named."""
    edit = {"data": _data_value, "config": _model_param, "metric": _first_metric}
    cfg, df = _config(), _frame()
    edit[change](cfg, df)
    error = _refused_on(storage, cfg, df)
    assert error.context["mismatched"] == [expected]


def _progress(info: TuneProgressInfo) -> None:
    del info


NON_IDENTITY_CHANGES: dict[str, tuple[Callable[..., None], dict[str, Any]]] = {
    "n_trials_argument": (lambda cfg, tmp: None, {"n_trials": 1}),
    "config_timeout": (
        lambda cfg, tmp: cfg["tuning"]["optuna"]["params"].update(timeout=3600.0),
        {},
    ),
    "config_n_trials": (
        lambda cfg, tmp: cfg["tuning"]["optuna"]["params"].update(n_trials=1),
        {},
    ),
    "progress_callback": (lambda cfg, tmp: None, {"progress_callback": _progress}),
    "expand_boundary": (lambda cfg, tmp: None, {"expand_boundary": True}),
    "boundary_threshold": (lambda cfg, tmp: None, {"boundary_threshold": 0.1}),
    "second_metric": (
        lambda cfg, tmp: cfg["evaluation"].update(metrics=["logloss", "auc"]),
        {},
    ),
    "calibration": (lambda cfg, tmp: cfg.update(calibration={"method": "platt"}), {}),
    "output_dir": (lambda cfg, tmp: cfg.update(output_dir=str(tmp / "out")), {}),
}


@pytest.mark.parametrize("change", sorted(NON_IDENTITY_CHANGES))
def test_non_identity_change_passes(storage: str, tmp_path: Path, change: str) -> None:
    """§4.2: these are outside the identity, so round 3 runs."""
    edit, kwargs = NON_IDENTITY_CHANGES[change]
    cfg = _config()
    edit(cfg, tmp_path)
    model = Model(cfg)
    result = model.tune(
        data=_frame(), resume=True, storage=storage, study_name="s", **kwargs
    )
    assert [r.round for r in result.rounds] == [1, 2, 3]
    assert model._round_number == 3
    assert "lizyml:round:3:end" in load(storage, "s").user_attrs


# ---------------------------------------------------------------------------
# Criterion 2: no LizyML record, trials present (state L)
# ---------------------------------------------------------------------------


def _quadratic(trial: Any) -> float:
    return float((trial.suggest_float("x", 0.0, 1.0) - 0.3) ** 2)


@pytest.mark.parametrize("origin", ["sqlite_deleted", "raw_optuna", "in_memory"])
def test_legacy_study_refused(storage: str, tmp_path: Path, origin: str) -> None:
    if origin == "in_memory":
        model = Model(_config())
        model.tune(data=_frame())
        legacy = optuna.create_study(direction="minimize")
        legacy.optimize(_quadratic, n_trials=2)
        model._study = legacy
        before = (snapshot_study(legacy), snapshot_model(model))
        with pytest.raises(LizyMLError) as caught:
            model.tune(data=_frame(), resume=True)
        assert caught.value.code is ErrorCode.CONFIG_INVALID
        assert caught.value.context["state"] == "legacy"
        assert (snapshot_study(legacy), snapshot_model(model)) == before
        return
    if origin == "sqlite_deleted":
        assert delete_attrs(storage, "lizyml:%") > 0
    else:
        storage = url(tmp_path, "raw.db")
        raw = optuna.create_study(storage=storage, study_name="s", direction="minimize")
        raw.optimize(_quadratic, n_trials=2)
    error = _refused_on(storage, _config(), _frame())
    assert error.context["state"] == "legacy"
    assert "new study" in error.user_message


# ---------------------------------------------------------------------------
# Criterion 3: a version difference only warns
# ---------------------------------------------------------------------------


def test_version_difference_warns_and_continues(storage: str) -> None:
    study = load(storage, "s")
    record = copy.deepcopy(study.user_attrs[STUDY_KEY])
    record["versions"]["lightgbm"] = "0.0.1"
    study.set_user_attr(STUDY_KEY, record)
    model = Model(_config())
    with pytest.warns(UserWarning, match="lightgbm 0.0.1 ->"):
        result = model.tune(data=_frame(), resume=True, storage=storage, study_name="s")
    assert [r.round for r in result.rounds] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Criterion 11: the config space is the study's, and cannot be edited
# ---------------------------------------------------------------------------


def _edit_space(cfg: Any, edit: str) -> None:
    """Edit a config dict, or a live ``LizyMLConfig``."""
    if isinstance(cfg, dict):
        optuna_cfg = cfg["tuning"]["optuna"]
        if edit == "space":
            optuna_cfg["space"]["num_leaves"]["high"] = 9
        else:
            optuna_cfg["space_mode"] = "merge"
        return
    if edit == "space":
        space = copy.deepcopy(cfg.tuning.optuna.space)
        space["num_leaves"]["high"] = 9
        cfg.tuning.optuna.space = space
    else:
        cfg.tuning.optuna.space_mode = "merge"


@pytest.mark.parametrize("route", ["memory", "reattach"])
@pytest.mark.parametrize("edit", ["space", "space_mode"])
def test_config_space_change_refused(storage: str, route: str, edit: str) -> None:
    surface = {"space": "tuning.optuna.space", "space_mode": "tuning.optuna.space_mode"}
    if route == "reattach":
        cfg = _config()
        _edit_space(cfg, edit)
        error = _refused_on(storage, cfg, _frame())
    else:
        model = Model(_config())
        model.tune(data=_frame())
        _edit_space(model._cfg, edit)
        before = (snapshot_study(model._study), snapshot_model(model))
        with pytest.raises(LizyMLError) as caught:
            model.tune(data=_frame(), resume=True)
        error = caught.value
        assert error.code is ErrorCode.CONFIG_INVALID
        assert (snapshot_study(model._study), snapshot_model(model)) == before
    assert error.context["surface"] == surface[edit]
    assert "cannot be changed through the config" in error.user_message


# ---------------------------------------------------------------------------
# Criteria 13 and 26: the data identity
# ---------------------------------------------------------------------------


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"feat_b": "feat_c"})


def _reorder_categories(df: pd.DataFrame) -> pd.DataFrame:
    df["cat"] = df["cat"].cat.reorder_categories(["b", "a"])
    return df


def _swap_mixed(df: pd.DataFrame) -> pd.DataFrame:
    df["mix"] = pd.Categorical(["1", 1] * (len(df) // 2), categories=[1, "1"])
    return df


def _index_labels(df: pd.DataFrame) -> pd.DataFrame:
    df.index = df.index + 1000
    return df


def _unused_category(df: pd.DataFrame) -> pd.DataFrame:
    df["cat"] = df["cat"].cat.add_categories(["z"])
    return df


def _column_order(df: pd.DataFrame) -> pd.DataFrame:
    order = list(df.columns)
    order[0], order[1] = order[1], order[0]
    return df[order]


def _float(df: pd.DataFrame) -> pd.DataFrame:
    df["num"] = df["num"].astype("float64")
    return df


def _nullable(df: pd.DataFrame) -> pd.DataFrame:
    df["num"] = df["num"].astype("Int64")
    return df


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    df["cat"] = df["cat"].cat.as_ordered()
    return df


def _object_str(df: pd.DataFrame) -> pd.DataFrame:
    df["obj"] = df["obj"].map(str).astype(object)
    return df


def _excluded_values(df: pd.DataFrame) -> pd.DataFrame:
    df["extra"] = df["extra"] + 1.0
    return df


DATA_CHANGES: dict[str, tuple[Callable[[pd.DataFrame], pd.DataFrame], str]] = {
    # criterion 13
    "rename_column": (_rename, "data.columns"),
    "reorder_categories": (_reorder_categories, "data.categories"),
    "mixed_categorical_swap": (_swap_mixed, "data.categorical_codes"),
    # criterion 26
    "index_labels": (_index_labels, "data.values"),
    "unused_category": (_unused_category, "data.categories"),
    "column_order": (_column_order, "data.columns"),
    "int64_float64": (_float, "data.dtypes"),
    "int64_Int64": (_nullable, "data.dtypes"),
    "ordered": (_ordered, "data.categories"),
    "object_1_vs_str": (_object_str, "data.object_types"),
    "excluded_values": (_excluded_values, "data.values"),
}


@pytest.mark.parametrize("change", sorted(DATA_CHANGES))
def test_data_identity_refusals(storage: str, change: str) -> None:
    edit, element = DATA_CHANGES[change]
    error = _refused_on(storage, _config(), edit(_frame()))
    assert element in error.context["mismatched"], error.context["mismatched"]


def test_constructor_data_is_judged_by_content(storage: str) -> None:
    """Criterion 13: the constructor route is summarised like the argument."""
    changed = _excluded_values(_frame())
    model = Model(_config(), data=changed)
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, storage=storage, study_name="s")
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert caught.value.context["mismatched"] == ["data.values"]
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before
    # The same content through the constructor matches the stored identity.
    result = Model(_config(), data=_frame()).tune(
        resume=True, storage=storage, study_name="s"
    )
    assert [r.round for r in result.rounds] == [1, 2, 3]


def test_data_path_file_rewritten_in_place_refused(tmp_path: Path) -> None:
    """Criterion 13: ``data.path`` is read and its content judged (R4)."""
    csv = tmp_path / "train.csv"
    plain = _frame()[["feat_a", "feat_b", "target"]]
    plain.to_csv(csv, index=False)
    cfg = _config()
    cfg["features"] = {"exclude": []}
    cfg["data"]["path"] = str(csv)
    storage = url(tmp_path)
    Model(cfg).tune(storage=storage, study_name="s")
    # Unchanged file: the reattach goes on.
    Model(cfg).tune(resume=True, storage=storage, study_name="s")
    changed = plain.copy()
    changed.loc[0, "feat_a"] += 1.0
    changed.to_csv(csv, index=False)
    model = Model(cfg)
    before = snapshot_study(storage, "s")
    with pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, storage=storage, study_name="s")
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert caught.value.context["mismatched"] == ["data.values"]
    assert snapshot_study(storage, "s") == before


# ---------------------------------------------------------------------------
# Criterion 28: the provider's `fixed` default is part of the objective
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("versions", ["versions_same", "versions_changed"])
def test_provider_fixed_change_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, versions: str
) -> None:
    storage = url(tmp_path)
    cfg = _config(mode="merge")
    Model(cfg).tune(data=_frame(), storage=storage, study_name="s")
    if versions == "versions_changed":
        study = load(storage, "s")
        record = copy.deepcopy(study.user_attrs[STUDY_KEY])
        record["versions"]["lightgbm"] = "0.0.1"
        study.set_user_attr(STUDY_KEY, record)
    real = LGBMProvider.default_fixed_params

    def changed(self: LGBMProvider, task: Any) -> dict[str, Any]:
        return {**real(self, task), "first_metric_only": False}

    monkeypatch.setattr(LGBMProvider, "default_fixed_params", changed)
    error = _refused_on(storage, cfg, _frame())
    assert error.context["mismatched"] == ["config.fixed"]


# ---------------------------------------------------------------------------
# The stored `fixed` is checked against the record's own identity (INV-R1)
# ---------------------------------------------------------------------------


def test_edited_stored_fixed_refused_before_any_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``lizyml:study`` whose ``fixed`` no longer digests to its own
    ``config.fixed`` element is refused (state S), and no trial trains on it.

    Identity compares this call's ``config.fixed`` with the stored *element*;
    the trials use the stored *value*. Without the check an edited value
    passed, because the element still matched (DC1).
    """
    storage = url(tmp_path)
    cfg = _config(mode="merge")
    Model(cfg).tune(data=_frame(), storage=storage, study_name="s")
    study = load(storage, "s")
    record = copy.deepcopy(study.user_attrs[STUDY_KEY])
    assert record["fixed"]["first_metric_only"] is True
    record["fixed"]["first_metric_only"] = False
    study.set_user_attr(STUDY_KEY, record)

    model = Model(cfg)

    def no_trial(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("a trial trained on the edited fixed")

    monkeypatch.setattr(model, "_build_train_components", no_trial)
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with pytest.raises(LizyMLError) as caught:
        model.tune(data=_frame(), resume=True, storage=storage, study_name="s")
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert caught.value.context["state"] == "unreadable"
    assert "config.fixed" in caught.value.user_message
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before


# ---------------------------------------------------------------------------
# Spec 4.3 / Q-E: the provider's default parameters are part of the objective
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["replace", "merge"])
@pytest.mark.parametrize("versions", ["versions_same", "versions_changed"])
def test_provider_default_params_change_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, versions: str, mode: str
) -> None:
    """A default every trial trains on, outside ``fixed`` (``max_bin``), changes
    between rounds: refused naming ``config.provider_defaults``, with or
    without a changed version record -- never only a version warning.
    """
    storage = url(tmp_path)
    cfg = _config(mode=mode)
    Model(cfg).tune(data=_frame(), storage=storage, study_name="s")
    if versions == "versions_changed":
        study = load(storage, "s")
        record = copy.deepcopy(study.user_attrs[STUDY_KEY])
        record["versions"]["lightgbm"] = "0.0.1"
        study.set_user_attr(STUDY_KEY, record)
    monkeypatch.setitem(lgbm_defaults._COMMON_DEFAULTS, "max_bin", 15)
    error = _refused_on(storage, cfg, _frame())
    assert error.context["mismatched"] == ["config.provider_defaults"]
