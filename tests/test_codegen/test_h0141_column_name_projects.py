"""H-0141 acceptance 2, 3, 4 and 7 on fitted models.

- ``Model.export_code`` refuses unsupported names and alias collisions before
  anything is written.
- A model reloaded from ``Model.export`` (declared keys in JSON form) writes a
  project whose declared keys are the matching features' aliases, which passes
  the marked-state check and keeps the declared order on a retrain.
- For every admitted name type x column kind the fit accepts on this lane,
  the generated predict.py (typed frame, aliased CSV, parquet), the retrained
  project and test_equivalence.py match the runtime prediction (#382). The
  pairs the fit refuses stay refused.
- The generated train.py hands LightGBM categorical positions (H-0142).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.codegen.column_names import column_alias
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_config
from tests.test_codegen._retrain_harness import (
    assert_retrain_reproduces,
    load_module,
    run_train,
    write_data,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

PANDAS_3 = int(pd.__version__.split(".")[0]) >= 3

LABELS: dict[str, Any] = {
    "str": "c",
    "numpy.str_": np.str_("c"),
    "int": 7,
    "float": 7.5,
    "bool": True,
    "None": None,
    "numpy.float64": np.float64(7.5),
}
KINDS = ("numeric", "categorical", "declared")
DECLARED = ["a", "b", "c", "z"]  # "z" is never used: the order must survive


def _fit_refused(label: str, kind: str) -> type[Exception] | None:
    """The fit's refusals for admitted names, per lane (H-0141 measurement)."""
    if kind == "numeric":
        return None
    if label == "bool":
        return ValueError
    if label == "None" and PANDAS_3:
        return LizyMLError
    return None


def _frame(label: Any, kind: str, n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c"], size=n)
    df = pd.DataFrame({"x": rng.normal(size=n)})
    if kind == "numeric":
        df[label] = rng.normal(size=n)
    elif kind == "categorical":
        df[label] = pd.Series(cat, dtype=object).to_numpy()
    else:
        df[label] = pd.Series(cat).astype(pd.CategoricalDtype(DECLARED)).to_numpy()
        df[label] = df[label].astype(pd.CategoricalDtype(DECLARED))
    df["target"] = (pd.Series(cat) == "a").astype(float) + rng.normal(scale=0.1, size=n)
    return df


def _model() -> Model:
    return Model(make_config("regression", n_estimators=3, n_splits=2))


def _aliased(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy(deep=False)
    out.columns = [column_alias(c) for c in X.columns]
    return out


CASES = [(label, kind) for label in LABELS for kind in KINDS]


# --- acceptance 4: prediction ---------------------------------------------------------


@pytest.mark.parametrize(("label", "kind"), CASES, ids=[f"{a}-{b}" for a, b in CASES])
def test_generated_predictions_match_the_runtime(
    tmp_path: Path, label: str, kind: str
) -> None:
    df = _frame(LABELS[label], kind)
    model = _model()
    refused = _fit_refused(label, kind)
    if refused is not None:
        # The fit's range is not changed by H-0141.
        with pytest.raises(refused):
            model.fit(data=df)
        return
    model.fit(data=df)
    X = df.drop(columns=["target"])
    expected = np.asarray(model.predict(X).pred)
    project = tmp_path / "project"
    model.export_code(project)
    module = load_module(project / "predict.py", f"h0141_grid_{label}_{kind}")

    def check(got: Any) -> None:
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-7, atol=0)

    check(module.predict(X)["pred"])
    csv_path, parquet_path = tmp_path / "x.csv", tmp_path / "x.parquet"
    _aliased(X).to_csv(csv_path, index=False)
    _aliased(X).to_parquet(parquet_path)
    check(module.predict(module.read_input(csv_path))["pred"])
    check(module.predict(module.read_input(parquet_path))["pred"])

    # test_equivalence.py, as a user runs it, on the aliased CSV.
    reference = tmp_path / "reference.csv"
    pd.DataFrame({"pred": expected}).to_csv(reference, index=False)
    result = subprocess.run(
        [
            sys.executable,
            "test_equivalence.py",
            str(csv_path),
            "--reference",
            str(reference),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    # The retrained project predicts as the runtime model does.
    assert_retrain_reproduces(model, df, project, fmt="csv", export=False)


# --- acceptance 2: Model.export_code refuses before writing -------------------------


def test_export_code_refuses_an_unsupported_name(tmp_path: Path) -> None:
    df = _frame(np.int64(7), "numeric")
    model = _model()
    model.fit(data=df)
    project = tmp_path / "project"
    with pytest.raises(LizyMLError) as info:
        model.export_code(project)
    assert info.value.code is ErrorCode.SERIALIZATION_FAILED
    ctx = info.value.context
    assert (ctx["surface"], ctx["reason"], ctx["kind"]) == (
        "export_code",
        "unsupported_column_name",
        "type",
    )
    assert ctx["columns"][0]["type"] == "numpy.int64"
    assert not project.exists()


def test_export_code_refuses_a_collision_with_the_target(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({7: rng.normal(size=80), "x": rng.normal(size=80)})
    df["7"] = df[7] + rng.normal(scale=0.1, size=80)
    cfg = make_config("regression", n_estimators=3, n_splits=2)
    cfg["data"]["target"] = "7"
    model = Model(cfg)
    model.fit(data=df)
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(LizyMLError) as info:
        model.export_code(project)
    ctx = info.value.context
    assert (ctx["reason"], ctx["alias"]) == ("column_name_collision", "7")
    assert [c["where"] for c in ctx["columns"]] == ["feature_names", "data.target"]
    assert list(project.iterdir()) == []


# --- acceptance 3: a model reloaded from Model.export ---

LOADED = {
    "numpy.str_": np.str_("c"),
    "int": 7,
    "float": 7.5,
    "inf": float("inf"),
    "-inf": float("-inf"),
    "numpy.float64": np.float64(7.5),
    "None": None,
}


@pytest.mark.parametrize("label", sorted(LOADED))
def test_a_loaded_model_writes_aliased_declared_keys(
    tmp_path: Path, label: str
) -> None:
    name = LOADED[label]
    if label == "None" and PANDAS_3:
        pytest.skip("pandas 3 refuses a None-named categorical column at fit")
    df = _frame(name, "declared")
    model = _model()
    model.fit(data=df)
    saved = tmp_path / "saved"
    model.export(saved)
    loaded = Model.load(saved)
    project = tmp_path / "project"
    loaded.export_code(project)

    alias = column_alias(name)
    config = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert list(config["declared_categories"]) == [alias]
    assert config["declared_categories"][alias] == DECLARED
    X = df.drop(columns=["target"])
    module = load_module(project / "predict.py", f"h0141_loaded_{label}")
    np.testing.assert_allclose(
        np.asarray(module.predict(X)["pred"]),
        np.asarray(model.predict(X).pred),
        rtol=1e-7,
        atol=0,
    )
    # The retrain restores the declared order, the unused category included.
    run_train(project, write_data(df, project, "csv"))
    state = json.loads(
        (project / "artifacts" / "pipeline_state.json").read_text(encoding="utf-8")
    )
    assert state["column_name_policy"] == "str_alias_v1"
    assert state["categories"][alias]["categories"] == DECLARED


# --- acceptance 7: LightGBM gets positions ---


def test_the_retrain_hands_lightgbm_categorical_positions(tmp_path: Path) -> None:
    """Int names 7, 0, 1 with the categorical column (named 7) at position 0.

    The project's names are string aliases, so a name list would also select
    the right column; the generated train.py passes positions as the runtime
    does (H-0142), and the retrained booster marks exactly position 0.
    """
    rng = np.random.default_rng(0)
    n = 160
    cat = rng.choice(["a", "b", "c"], size=n)
    df = pd.DataFrame(
        {
            7: pd.Series(cat, dtype=object),
            0: rng.normal(size=n),
            1: rng.normal(size=n),
        }
    )
    df["target"] = (cat == "a").astype(float) + 0.1 * df[0]
    model = _model()
    model.fit(data=df)
    project = tmp_path / "project"
    model.export_code(project)
    assert_retrain_reproduces(model, df, project, fmt="csv", export=False)
    booster = lgb.Booster(model_file=str(project / "artifacts" / "model.txt"))
    lines = booster.model_to_string().splitlines()
    assert [line for line in lines if "categorical_feature" in line] == [
        "[categorical_feature: 0]"
    ]
    runtime = model._refit_result.model._model  # noqa: SLF001
    assert [
        line
        for line in runtime.model_to_string().splitlines()
        if "categorical_feature" in line
    ] == ["[categorical_feature: 0]"]
