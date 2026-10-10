"""H-0120 acceptance criterion 1: the generated train.py reproduces the refit.

Each case fits a model, exports it, runs the generated ``train.py`` on the same
data and compares the predictions of the artifacts it wrote with the LizyML
refit model's, at ``rtol=1e-7`` (H-0059's promise, restored by H-0120). The
cases are a fixed enumeration (criterion 1); a combination ``Model.fit``
refuses asserts the refusal instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests.test_codegen._retrain_harness import (
    SPLIT_METHODS,
    TASKS,
    assert_retrain_reproduces,
    make_config,
    make_frame,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

#: Outer splits that stratify on the target: a regression target has no
#: classes, and ``Model.fit`` refuses them whatever its values (amendment 2).
_FIT_REFUSES = {
    ("regression", "stratified_kfold"),
    ("regression", "stratified_group_kfold"),
}


def _fit(cfg: dict[str, Any], df: pd.DataFrame) -> Model:
    model = Model(cfg)
    model.fit(data=df)
    return model


# ---------------------------------------------------------------------------
# Task x outer split, early stopping on (the default)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", SPLIT_METHODS)
@pytest.mark.parametrize("task", TASKS)
def test_task_by_split(task: str, method: str, tmp_path: Path) -> None:
    df = make_frame(task)
    cfg = make_config(task, method)
    if (task, method) in _FIT_REFUSES:
        integer_valued = df.assign(target=np.clip(np.round(df["target"]), -2, 2))
        for frame in (df, integer_valued):
            with pytest.raises(LizyMLError) as info:
                Model(cfg).fit(data=frame)
            assert info.value.code == ErrorCode.CONFIG_INVALID
        return
    assert_retrain_reproduces(_fit(cfg, df), df, tmp_path / "gen")


def test_the_split_matrix_is_the_declared_cross_product() -> None:
    assert len(TASKS) * len(SPLIT_METHODS) == 24
    assert {(t, m) for t in TASKS for m in SPLIT_METHODS} > _FIT_REFUSES


# ---------------------------------------------------------------------------
# Early stopping off, balanced weights, explicit inner_valid, inner gap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_early_stopping_disabled(task: str, tmp_path: Path) -> None:
    df = make_frame(task)
    cfg = make_config(task, "time_series", early_stopping={"enabled": False})
    assert_retrain_reproduces(_fit(cfg, df), df, tmp_path / "gen")


@pytest.mark.parametrize("balanced", [True, None, False])
def test_multiclass_balanced(balanced: bool | None, tmp_path: Path) -> None:
    df = make_frame("multiclass")
    cfg = make_config("multiclass", "kfold", model_extra={"balanced": balanced})
    assert_retrain_reproduces(_fit(cfg, df), df, tmp_path / "gen")


_EXPLICIT_INNER_VALID: dict[str, tuple[str, str, dict[str, Any]]] = {
    "holdout-ratio": ("regression", "kfold", {"method": "holdout", "ratio": 0.25}),
    "holdout-random_state": (
        "regression",
        "kfold",
        {"method": "holdout", "random_state": 7},
    ),
    "holdout-stratify": (
        "multiclass",
        "kfold",
        {"method": "holdout", "stratify": True},
    ),
    "group_holdout": (
        "binary",
        "group_kfold",
        {"method": "group_holdout", "ratio": 0.3, "random_state": 5},
    ),
    "time_holdout": (
        "regression",
        "time_series",
        {"method": "time_holdout", "ratio": 0.2},
    ),
}


@pytest.mark.parametrize("case", sorted(_EXPLICIT_INNER_VALID))
def test_explicit_inner_valid(case: str, tmp_path: Path) -> None:
    task, method, inner_valid = _EXPLICIT_INNER_VALID[case]
    df = make_frame(task)
    cfg = make_config(task, method, early_stopping={"inner_valid": inner_valid})
    assert_retrain_reproduces(_fit(cfg, df), df, tmp_path / "gen")


@pytest.mark.parametrize("explicit_time_holdout", [False, True])
def test_inner_gap(explicit_time_holdout: bool, tmp_path: Path) -> None:
    """Auto-resolved: the outer purge_gap reaches the inner split; explicit
    ``time_holdout``: the same factory builds ``gap=0``."""
    df = make_frame("regression")
    es: dict[str, Any] = {}
    if explicit_time_holdout:
        es["inner_valid"] = {"method": "time_holdout", "ratio": 0.15}
    cfg = make_config(
        "regression",
        "purged_time_series",
        early_stopping=es,
        split_extra={"purge_gap": 4},
    )
    model = _fit(cfg, df)
    project = tmp_path / "gen"
    assert_retrain_reproduces(model, df, project)
    exported = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert exported["inner_valid"]["gap"] == (0 if explicit_time_holdout else 4)


def test_tied_time_values(tmp_path: Path) -> None:
    df = make_frame("regression", tied_time=True)
    assert df["t"].duplicated().any()
    cfg = make_config("regression", "time_series")
    assert_retrain_reproduces(_fit(cfg, df), df, tmp_path / "gen")


# ---------------------------------------------------------------------------
# Validation set and callback: the four states
# ---------------------------------------------------------------------------


def _tuned(
    cfg: dict[str, Any], space: dict[str, Any], df: pd.DataFrame, *, fit: bool = True
) -> Model:
    cfg = {
        **cfg,
        "tuning": {
            "optuna": {
                "params": {"n_trials": 1, "direction": "minimize"},
                "space": space,
                "space_mode": "replace",
            }
        },
    }
    model = Model(cfg)
    model.tune(data=df)
    if fit:
        model.fit(data=df)
    return model


def _choice(value: Any) -> dict[str, Any]:
    return {"type": "categorical", "choices": [value], "category": "training"}


def test_state_valid_set_and_callback(tmp_path: Path) -> None:
    df = make_frame("binary")
    project = tmp_path / "gen"
    assert_retrain_reproduces(_fit(make_config("binary", "kfold"), df), df, project)
    exported = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert exported["inner_valid"] is not None
    assert exported["early_stopping_rounds"] == 5


def test_state_neither(tmp_path: Path) -> None:
    df = make_frame("binary")
    cfg = make_config("binary", "kfold", early_stopping={"enabled": False})
    project = tmp_path / "gen"
    assert_retrain_reproduces(_fit(cfg, df), df, project)
    exported = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert exported["inner_valid"] is None
    assert exported["early_stopping_rounds"] is None


def test_state_valid_set_only(tmp_path: Path) -> None:
    """A tuned ``validation_ratio`` with early stopping off in the config: the
    refit holds out a validation set but attaches no callback."""
    df = make_frame("binary")
    cfg = make_config("binary", "kfold", early_stopping={"enabled": False})
    model = _tuned(cfg, {"validation_ratio": _choice(0.3)}, df)
    project = tmp_path / "gen"
    assert_retrain_reproduces(model, df, project)
    exported = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert exported["inner_valid"]["ratio"] == 0.3
    assert exported["early_stopping_rounds"] is None


def test_state_patience_only(tmp_path: Path) -> None:
    """A tuned patience with early stopping off: the adapter holds a patience,
    but no inner split exists, so no callback."""
    df = make_frame("binary")
    cfg = make_config("binary", "kfold", early_stopping={"enabled": False})
    model = _tuned(cfg, {"early_stopping_rounds": _choice(3)}, df)
    project = tmp_path / "gen"
    assert_retrain_reproduces(model, df, project)
    exported = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert exported["inner_valid"] is None
    assert exported["early_stopping_rounds"] == 3


# ---------------------------------------------------------------------------
# Lifecycles: fit -> tune -> export, tune -> fit -> export -> load -> export
# ---------------------------------------------------------------------------

_LIFECYCLE_SPACE = {
    "validation_ratio": _choice(0.35),
    "early_stopping_rounds": _choice(4),
}
_LIFECYCLE_INNER_VALID = {"method": "holdout", "ratio": 0.2, "random_state": 9}


def test_fit_then_tune_then_export(tmp_path: Path) -> None:
    """``tune()`` after ``fit`` leaves the adapters alone, so the export must
    write the values the fit used (H-0094 decision 13)."""
    df = make_frame("regression")
    cfg = make_config(
        "regression", "kfold", early_stopping={"inner_valid": _LIFECYCLE_INNER_VALID}
    )
    model = _tuned(cfg, _LIFECYCLE_SPACE, df, fit=False)
    model.fit(data=df)
    # Now a different tuning result replaces the one the fit used.
    model._cfg.tuning.optuna.space = {  # type: ignore[union-attr]
        "validation_ratio": _choice(0.45),
        "early_stopping_rounds": _choice(9),
    }
    model.tune(data=df)
    assert model._tuning_result.best_training_params["validation_ratio"] == 0.45  # type: ignore[union-attr]
    assert_retrain_reproduces(model, df, tmp_path / "gen")


def test_tune_fit_export_load_export(tmp_path: Path) -> None:
    """The loaded artifact's ``applied_training_params`` and adapter give the
    export the same values the fit used (H-0109)."""
    df = make_frame("regression")
    cfg = make_config(
        "regression", "kfold", early_stopping={"inner_valid": _LIFECYCLE_INNER_VALID}
    )
    model = _tuned(cfg, _LIFECYCLE_SPACE, df)
    artifact = tmp_path / "artifact"
    model.export(artifact)
    loaded = Model.load(artifact)
    project = tmp_path / "gen"
    loaded.export_code(project)
    assert_retrain_reproduces(model, df, project, export=False)


# ---------------------------------------------------------------------------
# feval: the nine reimplemented metrics x three tasks
# ---------------------------------------------------------------------------

FEVALS = (
    "rmsle",
    "r2",
    "f1",
    "brier",
    "ece",
    "precision_at_k",
    "accuracy",
    "smape",
    "wape",
)

#: Measured 2026-10-10: the combinations ``Model.fit`` accepts.
_FEVAL_ACCEPTED = {
    "regression": {"rmsle", "r2", "smape", "wape"},
    "binary": {"f1", "brier", "ece", "precision_at_k", "accuracy"},
    "multiclass": {"f1", "brier", "accuracy"},
}


@pytest.mark.parametrize("name", FEVALS)
@pytest.mark.parametrize("task", TASKS)
def test_feval(task: str, name: str, tmp_path: Path) -> None:
    df = make_frame(task)
    if task == "regression":
        # rmsle needs a non-negative target.
        df["target"] = df["target"].abs()
    cfg = make_config(task, "kfold")
    cfg["model"]["params"]["metric"] = [name]
    if name not in _FEVAL_ACCEPTED[task]:
        with pytest.raises(LizyMLError):
            Model(cfg).fit(data=df)
        return
    assert_retrain_reproduces(_fit(cfg, df), df, tmp_path / "gen")


def test_the_feval_matrix_is_the_declared_cross_product() -> None:
    assert len(FEVALS) * len(TASKS) == 27
    assert set().union(*_FEVAL_ACCEPTED.values()) <= set(FEVALS)


# ---------------------------------------------------------------------------
# Categories (parquet), mixed types (predict.py only), CSV
# ---------------------------------------------------------------------------


def _with_categories(kind: str, n: int = 240) -> pd.DataFrame:
    df = make_frame("binary", n=n)
    rng = np.random.default_rng(5)
    if kind == "string":
        df["c"] = rng.choice(["b", "a", "c", "10", "9"], n)
    elif kind == "integer":
        # A plain integer column, declared categorical in the config.
        df["c"] = rng.choice([3, 1, 2, 10], n)
    elif kind == "declared":
        # Declared order differs from the sorted order, and one category no
        # row holds.
        df["c"] = pd.Categorical(
            rng.choice(["z", "y", "x"], n), categories=["z", "x", "y", "w"]
        )
    elif kind == "declared-all-missing":
        df["c"] = pd.Categorical([None] * n, categories=["q", "p"])
    elif kind == "with-missing":
        values = rng.choice(["a", "b", "c"], n).astype(object)
        values[rng.random(n) < 0.2] = None
        df["c"] = values
    else:
        raise AssertionError(kind)
    # The target depends on the category, so a wrong code changes the model.
    flag = df["c"].astype(object).map(lambda v: sum(map(ord, str(v))) % 2 == 0)
    df["target"] = (flag.to_numpy() ^ (df["f0"].to_numpy() > 1.5)).astype(np.int64)
    return df


#: LightGBM needs ``min_data_per_group`` rows per category to split on one
#: (default 100); without this the codes would not reach the model at all.
_CATEGORY_SPLITS = {"min_data_per_group": 5, "cat_smooth": 1.0, "cat_l2": 1.0}


def _category_config(kind: str, method: str = "kfold") -> dict[str, Any]:
    cfg = make_config("binary", method)
    cfg["model"]["params"].update(_CATEGORY_SPLITS)
    if kind in ("integer", "mixed"):
        cfg["features"] = {"categorical": ["c"]}
    return cfg


@pytest.mark.parametrize(
    "kind", ["string", "integer", "declared", "declared-all-missing", "with-missing"]
)
def test_categories_parquet(kind: str, tmp_path: Path) -> None:
    df = _with_categories(kind)
    model = _fit(_category_config(kind), df)
    project = tmp_path / "gen"
    assert_retrain_reproduces(model, df, project)
    # LightGBM's categorical splits do not depend on how codes are numbered,
    # so equal predictions alone would miss a different numbering: the
    # retrained state must hold LizyML's categories, in order, with types.
    _assert_state_matches_encoder(model, project)


def _assert_state_matches_encoder(model: Model, project: Path) -> None:
    encoder = model._refit_result.pipeline_state["encoder"]  # type: ignore[union-attr]
    state = json.loads(
        (project / "artifacts" / "pipeline_state.json").read_text(encoding="utf-8")
    )
    for col, cats in encoder["categories"].items():
        plain = [v.item() if isinstance(v, np.generic) else v for v in cats]
        entry = state["categories"][col]
        assert entry["categories"] == plain
        assert [type(v) for v in entry["categories"]] == [type(v) for v in plain]
        mode = encoder["modes"][col]
        assert entry["mode"] == (mode.item() if isinstance(mode, np.generic) else mode)


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_auto_categorical_off_keeps_the_string_column_uncast(
    fmt: str, tmp_path: Path
) -> None:
    """With ``auto_categorical: false`` the builder leaves a string column
    alone and the encoder takes ``sorted(unique, key=str)`` (amendment 3)."""
    df = _with_categories("string")
    cfg = _category_config("string")
    cfg["features"] = {"auto_categorical": False}
    model = _fit(cfg, df)
    assert model.fit_result.dtypes["c"] != "category"
    assert "c" in model._refit_result.categorical_features  # type: ignore[union-attr]
    project = tmp_path / "gen"
    assert_retrain_reproduces(model, df, project, fmt=fmt)
    _assert_state_matches_encoder(model, project)
    config = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert config["categorical_rule"] == {"explicit": [], "auto": False}


def test_mixed_type_column_predict_py_matches(tmp_path: Path) -> None:
    """A column holding ``"1"`` and ``1`` cannot be saved with its types, so it
    is outside the retrain promise; the exported predict.py must still use the
    LizyML codes."""
    from tests.test_codegen._retrain_harness import (
        generated_prediction,
        lizyml_prediction,
    )

    df = make_frame("binary")
    df["c"] = pd.Series(["1", 1, "2", 2] * (len(df) // 4), dtype=object)
    # Only the value type tells "1" from 1, and the target depends on it.
    flag = df["c"].map(lambda v: isinstance(v, str) != (str(v) == "2"))
    df["target"] = flag.astype(np.int64)
    model = _fit(_category_config("mixed"), df)
    project = tmp_path / "gen"
    model.export_code(project)
    X = df.drop(columns=["target"])
    np.testing.assert_allclose(
        generated_prediction(project, X, "binary"),
        lizyml_prediction(model, X, "binary"),
        rtol=1e-7,
        atol=0.0,
    )


def test_csv_within_the_stated_condition(tmp_path: Path) -> None:
    """String and declared categories and a numeric time column survive CSV
    once ``declared_categories`` is applied (the CSV condition)."""
    df = make_frame("binary")
    rng = np.random.default_rng(8)
    df["s"] = rng.choice(["k", "j", "m"], len(df))
    # Declared order differs from the sorted one, so a CSV read without the
    # declared categories would code "d" differently.
    df["d"] = pd.Categorical(
        rng.choice(["v", "u"], len(df)), categories=["v", "u", "t"]
    )
    df["target"] = (
        (df["d"].astype(str) == "v") ^ (df["s"] == "k") ^ (df["f0"] > 1.5)
    ).astype(np.int64)
    cfg = _category_config("csv", "time_series")
    model = _fit(cfg, df)
    project = tmp_path / "gen"
    assert_retrain_reproduces(model, df, project, fmt="csv")
    _assert_state_matches_encoder(model, project)


def test_new_data_keeps_a_new_category_of_an_inferred_column(tmp_path: Path) -> None:
    """Only the input's own ``category`` columns are restored (amendment 4): a
    string column is re-inferred, so a category the fit never saw is learned on
    a retrain with new data instead of becoming a missing value."""
    from tests.test_codegen._retrain_harness import run_train, write_data

    df = make_frame("binary")
    rng = np.random.default_rng(9)
    df["s"] = rng.choice(["k", "j"], len(df))
    model = _fit(_category_config("csv"), df)
    project = tmp_path / "gen"
    model.export_code(project)
    new = df.copy()
    new.loc[new.index[:40], "s"] = "new"
    run_train(project, write_data(new, project, "csv"))
    state = json.loads(
        (project / "artifacts" / "pipeline_state.json").read_text(encoding="utf-8")
    )
    assert "new" in state["categories"]["s"]["categories"]


# ---------------------------------------------------------------------------
# A real subprocess per task, and the generated dependencies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_command_line_run(task: str, tmp_path: Path) -> None:
    df = make_frame(task)
    cfg = make_config(task, "time_series")
    assert_retrain_reproduces(_fit(cfg, df), df, tmp_path / "gen", subprocess_run=True)


def test_requirements_list_pyarrow(tmp_path: Path) -> None:
    df = make_frame("regression")
    project = tmp_path / "gen"
    _fit(make_config("regression", "kfold"), df).export_code(project)
    lines = (project / "requirements.txt").read_text(encoding="utf-8").split()
    assert "pyarrow" in lines
