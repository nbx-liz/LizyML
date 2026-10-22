"""H-0141 acceptance 1, 2, 3 (the writers), 5, 6 and 8 -- export_code column names.

- The alias rule, in the library and in both generated scripts' copies (DC3).
- The refusals before anything is written: the exact context, the order, and
  no directory or file written.
- The writers: every name field holds its alias, both JSON files carry the
  marker, and the fitted state passed in is not changed.
- The generated scripts: the marker and the marked state are checked before
  any value is read, and every input boundary (typed frames, CSV, parquet,
  the equivalence CLI) gives the decided reasons.
- A string-name project differs from the pre-H-0141 writers only by the marker.
"""

from __future__ import annotations

import ast
import copy
import decimal
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.codegen import column_names
from lizyml.codegen.artifact_writer import convert_pipeline_state
from lizyml.codegen.column_names import UnsupportedColumnName, column_alias
from lizyml.codegen.config_writer import build_config
from lizyml.codegen.generator import generate_code
from lizyml.codegen.templates import render_predict_py, render_train_py
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_config
from tests.test_codegen._retrain_harness import load_module, run_train, write_data

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# --- acceptance 1: the alias rule ---------------------------------------------------

ADMITTED: list[tuple[Any, str]] = [
    ("c", "c"),
    (np.str_("c"), "c"),
    (7, "7"),
    (-7, "-7"),
    (10**30, str(10**30)),
    (7.0, "7.0"),
    (-0.0, "-0.0"),
    (float("nan"), "nan"),
    (float("inf"), "inf"),
    (float("-inf"), "-inf"),
    (5e-324, "5e-324"),
    (1e300, "1e+300"),
    (np.float64(7.5), "7.5"),
    (np.float64(-0.0), "-0.0"),
    (True, "True"),
    (False, "False"),
    (None, "None"),
]

REFUSED: list[tuple[Any, str]] = [
    (np.int64(7), "type"),
    (np.float32(7.5), "type"),
    (np.bool_(True), "type"),
    (b"c", "type"),
    (1j, "type"),
    (decimal.Decimal("1"), "type"),
    (("a", "b"), "type"),
    ("", "empty_alias"),
    ("a\x00b", "invalid_alias_text"),
    ("\ud800", "invalid_alias_text"),
]


def _script_copy(source: str) -> SimpleNamespace:
    """The generated script's ``UnsupportedColumnName`` and ``column_alias``."""
    tree = ast.parse(source)
    wanted = {"UnsupportedColumnName", "column_alias"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name in wanted
    ]
    assert sorted(n.name for n in nodes) == sorted(wanted)
    namespace: dict[str, Any] = {"np": np}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, "<generated column names>", "exec"), namespace)  # noqa: S102
    return SimpleNamespace(**namespace)


COPIES = {
    "library": SimpleNamespace(
        column_alias=column_alias, UnsupportedColumnName=UnsupportedColumnName
    ),
    "train.py": _script_copy(render_train_py()),
    "predict.py": _script_copy(render_predict_py()),
}


@pytest.mark.parametrize("where", sorted(COPIES))
@pytest.mark.parametrize(("name", "alias"), ADMITTED, ids=repr)
def test_admitted_names_get_str_of_the_name(where: str, name: Any, alias: str) -> None:
    got = COPIES[where].column_alias(name)
    assert type(got) is str
    assert got == alias


@pytest.mark.parametrize("where", sorted(COPIES))
@pytest.mark.parametrize(("name", "kind"), REFUSED, ids=repr)
def test_refused_names_give_the_reason_and_kind(
    where: str, name: Any, kind: str
) -> None:
    copy_ = COPIES[where]
    with pytest.raises(copy_.UnsupportedColumnName) as info:
        copy_.column_alias(name)
    assert info.value.reason == "unsupported_column_name"
    assert info.value.kind == kind
    assert isinstance(info.value, ValueError)


def test_the_scripts_carry_the_library_source_once() -> None:
    """DC3: the copies are the library's source, so they cannot drift."""
    import inspect

    for source in (render_train_py(), render_predict_py()):
        for obj in (column_names.UnsupportedColumnName, column_names.column_alias):
            assert source.count(inspect.getsource(obj)) == 1


# --- acceptance 2: refusals before anything is written ------------------------------


@pytest.fixture(scope="module")
def adapter() -> Any:
    from lizyml.estimators.lgbm.adapter import LGBMAdapter

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    fitted = LGBMAdapter(task="regression", params={"n_estimators": 3, "verbose": -1})
    fitted.fit(X, pd.Series(rng.normal(size=60)))
    return fitted


def _meta(**config: Any) -> dict[str, Any]:
    normalized = {"task": "regression", "data": {"target": "target"}, **config}
    return {
        "lizyml_version": "test",
        "run_id": "h0141",
        "timestamp": "2026-10-22T00:00:00",
        "config_normalized": normalized,
    }


def _state(names: list[Any], encoder: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "feature_names": list(names),
        "categorical_cols": [],
        "encoder": encoder or {},
        "transformer": {},
    }


def _generate(
    out: Path,
    adapter: Any,
    names: list[Any],
    *,
    meta: dict[str, Any] | None = None,
    categorical: list[Any] | None = None,
    declared: dict[Any, list[Any]] | None = None,
    inner_valid: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    rule: dict[str, Any] | None = None,
) -> Path:
    return generate_code(
        output_dir=out,
        run_meta=meta or _meta(),
        feature_names=names,
        categorical_features=categorical or [],
        lgbm_params={"objective": "regression"},
        num_boost_round=3,
        early_stopping_rounds=None,
        validation_ratio=0.0,
        seed=0,
        calibration_method=None,
        calibration_n_splits=2,
        model_adapter=adapter,
        pipeline_state=state if state is not None else _state(names),
        calibrator=None,
        inner_valid=inner_valid,
        declared_categories=declared,
        categorical_rule=rule,
    )


def _column(where: str, position: int | None, name: Any, alias: str | None) -> dict:
    from lizyml.core.category_values import shown

    kind = type(name)
    return {
        "where": where,
        "position": position,
        "type": f"{kind.__module__}.{kind.__qualname__}",
        "repr": shown(name),
        "alias": alias,
    }


def _refusal(info: pytest.ExceptionInfo[LizyMLError]) -> dict[str, Any]:
    assert info.value.code is ErrorCode.SERIALIZATION_FAILED
    return dict(info.value.context)


def _assert_nothing_written(out: Path, before: list[str] | None) -> None:
    if before is None:
        assert not out.exists()
    else:
        assert sorted(p.name for p in out.iterdir()) == before


@pytest.fixture(params=["absent", "existing"])
def out(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[Path, list[str] | None]:
    path = tmp_path / "project"
    if request.param == "absent":
        return path, None
    path.mkdir()
    (path / "keep.txt").write_text("x", encoding="utf-8")
    return path, ["keep.txt"]


def _refused(
    out: tuple[Path, list[str] | None], adapter: Any, *args: Any, **kw: Any
) -> dict:
    with pytest.raises(LizyMLError) as info:
        _generate(out[0], adapter, *args, **kw)
    _assert_nothing_written(*out)
    return _refusal(info)


def test_an_unsupported_feature_name_is_named(out: Any, adapter: Any) -> None:
    ctx = _refused(out, adapter, ["a", np.int64(8)])
    assert ctx == {
        "surface": "export_code",
        "reason": "unsupported_column_name",
        "kind": "type",
        "alias": None,
        "columns": [_column("feature_names", 1, np.int64(8), None)],
    }


def test_features_are_checked_before_roles(out: Any, adapter: Any) -> None:
    ctx = _refused(out, adapter, [b"x", "b"], meta=_meta(data={"target": ""}))
    assert ctx["kind"] == "type"
    assert ctx["columns"] == [_column("feature_names", 0, b"x", None)]


@pytest.mark.parametrize(
    ("config", "where", "position", "kind"),
    [
        ({"data": {"target": ""}}, "data.target", None, "empty_alias"),
        (
            {"data": {"target": "y", "time_col": "t\x00"}},
            "data.time_col",
            None,
            "invalid_alias_text",
        ),
        ({"features": {"exclude": ["ok", ""]}}, "features.exclude", 1, "empty_alias"),
    ],
)
def test_an_unsupported_role_name_is_named(
    out: Any, adapter: Any, config: dict, where: str, position: int | None, kind: str
) -> None:
    ctx = _refused(out, adapter, ["a", "b"], meta=_meta(**config))
    assert (ctx["reason"], ctx["kind"]) == ("unsupported_column_name", kind)
    assert [c["where"] for c in ctx["columns"]] == [where]
    assert ctx["columns"][0]["position"] == position


@pytest.mark.parametrize(
    ("names", "alias"),
    [
        ([7, "7"], "7"),
        ([True, "True"], "True"),
        ([float("nan"), "nan"], "nan"),
        ([7.0, "7.0"], "7.0"),
        (["x", np.float64(7.5), 7.5], "7.5"),
    ],
    ids=["int", "bool", "nan", "float", "float64"],
)
def test_two_features_with_one_alias_collide(
    out: Any, adapter: Any, names: list[Any], alias: str
) -> None:
    ctx = _refused(out, adapter, names)
    hits = [(i, n) for i, n in enumerate(names) if column_alias(n) == alias]
    assert ctx == {
        "surface": "export_code",
        "reason": "column_name_collision",
        "kind": "duplicate_alias",
        "alias": alias,
        "columns": [_column("feature_names", i, n, alias) for i, n in hits],
    }


ROLE_CASES = [
    ({"data": {"target": "7"}}, None, "data.target", None),
    ({"data": {"target": "y", "time_col": "7"}}, None, "data.time_col", None),
    ({"data": {"target": "y", "group_col": "7"}}, None, "data.group_col", None),
    ({"features": {"exclude": ["q", "7"]}}, None, "features.exclude", 1),
    ({"features": {"categorical": ["7"]}}, None, "features.categorical", 0),
    (
        {"split": {"blocks": {"col": "7"}, "groups": {"col": "g"}}},
        None,
        "split.blocks.col",
        None,
    ),
    (
        {"split": {"blocks": {"col": "p"}, "groups": {"col": "7"}}},
        None,
        "split.groups.col",
        None,
    ),
    ({}, {"method": "group_holdout", "group_col": "7"}, "inner_valid.group_col", None),
]


@pytest.mark.parametrize(("config", "inner", "where", "position"), ROLE_CASES)
def test_a_non_string_feature_collides_with_a_role(
    out: Any, adapter: Any, config: dict, inner: Any, where: str, position: Any
) -> None:
    ctx = _refused(out, adapter, ["a", 7], meta=_meta(**config), inner_valid=inner)
    assert ctx["reason"] == "column_name_collision"
    assert ctx["alias"] == "7"
    assert ctx["columns"] == [
        _column("feature_names", 1, 7, "7"),
        {**_column(where, position, "7", "7")},
    ]


def test_the_first_collision_group_is_reported_in_order(out: Any, adapter: Any) -> None:
    names = ["z", 7, True, "True", "7"]
    ctx = _refused(out, adapter, names, meta=_meta(data={"target": "7"}))
    assert ctx["alias"] == "7"
    assert [(c["where"], c["position"]) for c in ctx["columns"]] == [
        ("feature_names", 1),
        ("feature_names", 4),
        ("data.target", None),
    ]


def test_a_string_feature_and_the_same_string_role_are_one_column(
    tmp_path: Path, adapter: Any
) -> None:
    meta = _meta(
        features={"categorical": ["c"], "exclude": ["p"]},
        split={"blocks": {"col": "b"}, "groups": {"col": "g"}},
    )
    project = _generate(tmp_path / "p", adapter, ["b", "c"], meta=meta)
    assert json.loads((project / "config.json").read_text())["feature_names"] == [
        "b",
        "c",
    ]


@pytest.mark.parametrize(
    ("names", "key", "kind"),
    [
        (["a", 7], "zz", "unknown_reference"),
        (["7", 7], "7", "ambiguous_reference"),
        ([7, np.str_("7")], "7", "ambiguous_reference"),
    ],
)
def test_an_unmatched_declared_key_is_refused(
    out: Any, adapter: Any, names: list[Any], key: Any, kind: str
) -> None:
    ctx = _refused(out, adapter, names, declared={key: ["a"]})
    assert ctx == {
        "surface": "export_code",
        "reason": "unsupported_column_name",
        "kind": kind,
        "alias": None,
        "columns": [_column("declared_categories", 0, key, None)],
    }


@pytest.mark.parametrize(
    ("name", "key"),
    [
        (7, "7"),
        (7.5, "7.5"),
        (float("inf"), "Infinity"),
        (float("-inf"), "-Infinity"),
        (float("nan"), "NaN"),
        (None, "null"),
        (True, "true"),
        (np.float64(7.5), "7.5"),
        (np.str_("c"), "c"),
    ],
    ids=repr,
)
def test_a_json_form_declared_key_is_matched_to_its_feature(
    tmp_path: Path, adapter: Any, name: Any, key: str
) -> None:
    """A model loaded from Model.export holds its declared keys in JSON form."""
    names = ["x", name]
    project = _generate(tmp_path / "p", adapter, names, declared={key: ["a", "b"]})
    config = json.loads((project / "config.json").read_text())
    assert list(config["declared_categories"]) == [column_alias(name)]


# --- acceptance 3: the writers -------------------------------------------------------


def test_every_name_field_holds_its_alias(tmp_path: Path, adapter: Any) -> None:
    encoder = {
        "unseen_policy": "mode",
        "categories": {7: ["a", "b", "c"], True: [1, 2]},
        "modes": {7: "a", True: 1},
        "stand_in_columns": [],
        "category_identity_version": 1,
        "identity_columns": [7],
    }
    names = [7, "x", True]
    state = _state(names, encoder)
    state["categorical_cols"] = [7, True]
    before = copy.deepcopy(state)
    meta = _meta(
        data={"target": "target", "group_col": "g"},
        features={"categorical": ["x"]},
    )
    project = _generate(
        tmp_path / "p",
        adapter,
        names,
        meta=meta,
        categorical=[7, True],
        declared={7: ["a", "b", "c"]},
        inner_valid={"method": "group_holdout", "ratio": 0.2, "group_col": "g"},
        state=state,
        rule={"explicit": ["x"], "auto": True},
    )
    config = json.loads((project / "config.json").read_text())
    written = json.loads((project / "artifacts" / "pipeline_state.json").read_text())
    assert config["feature_names"] == ["7", "x", "True"]
    assert config["categorical_features"] == ["7", "True"]
    assert list(config["declared_categories"]) == ["7"]
    assert config["categorical_rule"]["explicit"] == ["x"]
    assert config["_target_col"] == "target"
    assert config["inner_valid"]["group_col"] == "g"
    assert config["column_name_policy"] == "str_alias_v1"
    assert written["feature_names"] == ["7", "x", "True"]
    assert written["categorical_features"] == ["7", "True"]
    assert list(written["categories"]) == ["7", "True"]
    assert written["categories"]["7"]["categories"] == ["a", "b", "c"]
    assert written["identity_columns"] == ["7"]
    assert written["column_name_policy"] == "str_alias_v1"
    # The fitted state given to the writer is only read.
    assert state == before


# --- acceptance 8: a string-name project --------------------------------------------


def test_a_string_name_project_differs_only_by_the_marker(
    tmp_path: Path, adapter: Any
) -> None:
    names = ["a", "b"]
    encoder = {
        "unseen_policy": "mode",
        "categories": {"b": ["u", "v"]},
        "modes": {"b": "u"},
        "stand_in_columns": [],
    }
    kwargs: dict[str, Any] = {
        "categorical": ["b"],
        "declared": {"b": ["u", "v", "w"]},
        "state": {**_state(names, encoder), "categorical_cols": ["b"]},
    }
    project = _generate(tmp_path / "p", adapter, names, **kwargs)
    config = json.loads((project / "config.json").read_text())
    written = json.loads((project / "artifacts" / "pipeline_state.json").read_text())

    # The writers as they were before H-0141: the names as given, no marker.
    old_config = build_config(
        run_meta=_meta(),
        feature_names=names,
        categorical_features=["b"],
        lgbm_params={"objective": "regression"},
        num_boost_round=3,
        early_stopping_rounds=None,
        validation_ratio=0.0,
        seed=0,
        calibration_method=None,
        calibration_n_splits=2,
        declared_categories={"b": ["u", "v", "w"]},
    )
    old_state = convert_pipeline_state(kwargs["state"], old_config)
    assert set(config) - set(old_config) == {"column_name_policy"}
    assert set(written) - set(old_state) == {"column_name_policy"}
    for key, value in old_config.items():
        if key != "_versions":
            assert config[key] == value, key
    assert {k: v for k, v in written.items() if k != "column_name_policy"} == old_state
    model_text = tmp_path / "model.txt"
    adapter.save_model_text(model_text)
    assert (project / "artifacts" / "model.txt").read_bytes() == model_text.read_bytes()


# --- acceptance 5 and 6: the generated scripts --------------------------------------


def _frame(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c"], size=n)
    return pd.DataFrame(
        {
            7: pd.Series(cat, dtype=object),
            "x": rng.normal(size=n),
            "target": (cat == "a").astype(float) + rng.normal(scale=0.1, size=n),
        }
    )


@pytest.fixture(scope="module")
def marked(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """A fitted model with an int-named categorical column, and its project."""
    df = _frame()
    model = Model(make_config("regression", n_estimators=3, n_splits=2))
    model.fit(data=df)
    project = tmp_path_factory.mktemp("h0141") / "project"
    model.export_code(project)
    X = df.drop(columns=["target"])
    return SimpleNamespace(
        model=model,
        project=project,
        df=df,
        X=X,
        expected=np.asarray(model.predict(X).pred),
    )


def _copy_project(src: Path, dst: Path) -> Path:
    import shutil

    shutil.copytree(src, dst)
    return dst


def _predict_module(project: Path) -> ModuleType:
    return load_module(
        project / "predict.py", f"h0141_predict_{id(project)}_{project.name}"
    )


def _edit(project: Path, name: str, change: Any) -> None:
    path = project / name
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _reason(fn: Any) -> str:
    try:
        fn()
    except ValueError as exc:
        return str(getattr(exc, "reason", type(exc).__name__))
    return "ok"


def test_the_marked_project_predicts_from_every_input(marked: SimpleNamespace) -> None:
    module = _predict_module(marked.project)
    typed = np.asarray(module.predict(marked.X)["pred"])
    np.testing.assert_allclose(typed, marked.expected, rtol=1e-7, atol=0)
    aliased = marked.X.copy()
    aliased.columns = [column_alias(c) for c in aliased.columns]
    for suffix in ("csv", "parquet"):
        path = marked.project.parent / f"in.{suffix}"
        if suffix == "csv":
            aliased.to_csv(path, index=False)
        else:
            aliased.to_parquet(path)
        got = np.asarray(module.predict(module.read_input(path))["pred"])
        np.testing.assert_allclose(got, marked.expected, rtol=1e-7, atol=0)
    # Columns in another order are matched by name.
    reordered = marked.project.parent / "reordered.csv"
    aliased[list(reversed(aliased.columns))].to_csv(reordered, index=False)
    got = np.asarray(module.predict(module.read_input(reordered))["pred"])
    np.testing.assert_allclose(got, marked.expected, rtol=1e-7, atol=0)


def test_typed_frames_are_checked(marked: SimpleNamespace) -> None:
    module = _predict_module(marked.project)
    X = marked.X
    with_extra = X.assign(**{"7": 0.0})
    assert _reason(lambda: module.predict(with_extra)) == "column_name_collision"
    assert (
        _reason(lambda: module.predict(X.drop(columns=["x"]))) == "missing_column_alias"
    )
    odd = X.copy()
    odd[np.int64(3)] = 0.0
    assert _reason(lambda: module.predict(odd)) == "unsupported_column_name"
    multi = X.copy()
    multi.columns = pd.MultiIndex.from_tuples([("a", 7), ("a", "x")])
    assert _reason(lambda: module.predict(multi)) == "unsupported_column_name"


def _csv(project: Path, name: str, text: str) -> Path:
    path = project.parent / name
    path.write_text(text, encoding="utf-8")
    return path


def test_csv_headers_are_checked(marked: SimpleNamespace) -> None:
    module = _predict_module(marked.project)
    project = marked.project
    dup = _csv(project, "dup.csv", "7,7,x\na,b,1.0\n")
    assert _reason(lambda: module.read_input(dup)) == "column_name_collision"
    blank = _csv(project, "blank.csv", "7,,x\na,1,1.0\n")
    assert _reason(lambda: module.read_input(blank)) == "invalid_column_header"
    missing = _csv(project, "missing.csv", "x\n1.0\n")
    df = module.read_input(missing)
    assert _reason(lambda: module.predict(df)) == "missing_column_alias"
    # Genuine names that pandas would also invent are names, not inventions.
    real = _csv(project, "real.csv", "Unnamed: 0,7.1,7,x\n1,2,a,1.0\n")
    assert list(module.read_input(real).columns) == ["Unnamed: 0", "7.1", "7", "x"]
    assert _reason(lambda: module.predict(module.read_input(real))) == "ok"


def test_a_header_pandas_changes_is_refused(
    marked: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _predict_module(marked.project)
    path = _csv(marked.project, "changed.csv", "7,x\na,1.0\n")
    real = pd.read_csv

    def renaming(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return real(*args, **kwargs).rename(columns={"7": "7.0"})

    monkeypatch.setattr(module.pd, "read_csv", renaming)
    assert _reason(lambda: module.read_input(path)) == "invalid_column_header"


@pytest.mark.parametrize(
    ("columns", "reason"),
    [
        ([7, "x"], "invalid_column_header"),
        (["7", "7"], "column_name_collision"),
        (pd.MultiIndex.from_tuples([("a", "7"), ("a", "x")]), "invalid_column_header"),
    ],
    ids=["non-string", "repeated", "multiindex"],
)
def test_parquet_names_are_checked(
    marked: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, columns: Any, reason: str
) -> None:
    """pandas will not write these names to parquet; the reader is given them
    (the file itself holds valid names, so its schema passes)."""
    module = _predict_module(marked.project)
    path = marked.project.parent / "valid.parquet"
    pd.DataFrame([["a", 1.0]], columns=["7", "x"]).to_parquet(path)
    frame = pd.DataFrame([["a", 1.0]], columns=columns)
    monkeypatch.setattr(module.pd, "read_parquet", lambda *_a, **_k: frame)
    assert _reason(lambda: module.read_input(path)) == reason


def test_the_equivalence_cli_refuses_a_repeated_header(marked: SimpleNamespace) -> None:
    path = _csv(marked.project, "cli_dup.csv", "7,7,x\na,b,1.0\n")
    ref = _csv(marked.project, "ref.csv", "pred\n0.0\n")
    result = subprocess.run(
        [sys.executable, "test_equivalence.py", str(path), "--reference", str(ref)],
        cwd=marked.project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "column_name_collision" in result.stderr


POLICY_CASES = [
    ("absent", None, None, "ok"),
    ("both", "str_alias_v1", "str_alias_v1", "ok"),
    ("config only", "str_alias_v1", None, "column_name_policy_mismatch"),
    ("state only", None, "str_alias_v1", "column_name_policy_mismatch"),
    ("config null", "null", "str_alias_v1", "unsupported_column_name_policy"),
    ("state unknown", "str_alias_v1", "str_alias_v2", "unsupported_column_name_policy"),
    ("state not str", "str_alias_v1", 1, "unsupported_column_name_policy"),
    ("unknown, one side", "str_alias_v2", None, "unsupported_column_name_policy"),
]


def _set_policy(value: Any) -> Any:
    def change(doc: dict) -> None:
        doc.pop("column_name_policy", None)
        if value == "null":
            doc["column_name_policy"] = None
        elif value is not None:
            doc["column_name_policy"] = value

    return change


@pytest.mark.parametrize(("label", "in_config", "in_state", "reason"), POLICY_CASES)
def test_the_marker_decides_the_lookup(
    marked: SimpleNamespace,
    tmp_path: Path,
    label: str,
    in_config: Any,
    in_state: Any,
    reason: str,
) -> None:
    project = _copy_project(marked.project, tmp_path / "p")
    _edit(project, "config.json", _set_policy(in_config))
    _edit(project, "artifacts/pipeline_state.json", _set_policy(in_state))
    if label == "absent":
        # A project written before H-0141 keeps its own lookup: the typed
        # frame's own names, here int 7 against the JSON key "7".
        X = marked.X.copy()
        X.columns = [column_alias(c) for c in X.columns]
    else:
        X = marked.X
    module = _predict_module(project)
    assert _reason(lambda: module.predict(X)) == reason


def _poison_values(doc: dict) -> None:
    """A category the decoder refuses: a state check that ran after the
    decoding would report the decoder's error instead."""
    for entry in doc["categories"].values():
        entry["categories"] = [{"$lizyml": "no-such-tag"}]


STATE_CASES = [
    ("artifacts/pipeline_state.json", lambda d: d["feature_names"].reverse()),
    ("config.json", lambda d: d["feature_names"].append("7")),
    ("config.json", lambda d: d["feature_names"].__setitem__(0, 7)),
    ("artifacts/pipeline_state.json", lambda d: d["categorical_features"].clear()),
    ("config.json", lambda d: d["categorical_features"].append("zz")),
    ("artifacts/pipeline_state.json", lambda d: d["categories"].pop("7")),
    (
        "artifacts/pipeline_state.json",
        lambda d: d.__setitem__("identity_columns", ["x"]),
    ),
    ("config.json", lambda d: d["declared_categories"].__setitem__("x", ["1"])),
    ("config.json", lambda d: d.__setitem__("_target_col", 7)),
    # Review round 1: a missing mode, and role names that are null.
    ("artifacts/pipeline_state.json", lambda d: d["categories"]["7"].pop("mode")),
    ("config.json", lambda d: d.__setitem__("_target_col", None)),
    ("config.json", lambda d: d["categorical_rule"].__setitem__("explicit", [None])),
    # Review round 2: categories that are not an object.
    ("artifacts/pipeline_state.json", lambda d: d.__setitem__("categories", 1)),
]


@pytest.mark.parametrize(("name", "change"), STATE_CASES)
def test_a_marked_state_is_checked_before_any_value(
    marked: SimpleNamespace, tmp_path: Path, name: str, change: Any
) -> None:
    project = _copy_project(marked.project, tmp_path / "p")
    _edit(project, "artifacts/pipeline_state.json", _poison_values)
    _edit(project, name, change)
    module = _predict_module(project)
    assert _reason(lambda: module.predict(marked.X)) == "invalid_column_name_state"


RETRAIN_REFUSALS = [
    (
        "config only",
        lambda p: _edit(p, STATE, _set_policy(None)),
        "column_name_policy_mismatch",
    ),
    (
        "state only",
        lambda p: _edit(p, "config.json", _set_policy(None)),
        "column_name_policy_mismatch",
    ),
    (
        "state unknown",
        lambda p: _edit(p, STATE, _set_policy("str_alias_v2")),
        "unsupported_column_name_policy",
    ),
    (
        "state reordered",
        lambda p: _edit(p, STATE, lambda d: d["feature_names"].reverse()),
        "invalid_column_name_state",
    ),
]
STATE = "artifacts/pipeline_state.json"


@pytest.mark.parametrize(
    ("label", "change", "reason"),
    RETRAIN_REFUSALS,
    ids=[c[0] for c in RETRAIN_REFUSALS],
)
@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_a_retrain_checks_both_markers_before_writing(
    marked: SimpleNamespace,
    tmp_path: Path,
    label: str,
    change: Any,
    reason: str,
    fmt: str,
) -> None:
    """Review round 1: the retrain reads the project's state markers first."""
    project = _copy_project(marked.project, tmp_path / "p")
    change(project)
    data = write_data(marked.df, project, fmt)
    before = (project / STATE).read_bytes()
    with pytest.raises(ValueError) as info:
        run_train(project, data)
    assert getattr(info.value, "reason", None) == reason
    assert (project / STATE).read_bytes() == before


@pytest.mark.parametrize(
    ("label", "in_state", "reason"),
    [
        ("unknown", "str_alias_v2", "unsupported_column_name_policy"),
        ("absent", None, "column_name_policy_mismatch"),
    ],
)
def test_the_readers_check_both_markers_before_names(
    marked: SimpleNamespace, tmp_path: Path, label: str, in_state: Any, reason: str
) -> None:
    """Review round 1: a bad config name must not hide a marker refusal."""
    project = _copy_project(marked.project, tmp_path / "p")
    _edit(project, "config.json", lambda d: d["feature_names"].__setitem__(0, 7))
    _edit(project, STATE, _set_policy(in_state))
    path = _csv(project, "in.csv", "7,x\na,1.0\n")
    module = _predict_module(project)
    assert _reason(lambda: module.read_input(path)) == reason


def test_a_programmatic_retrain_checks_the_markers_first(
    marked: SimpleNamespace, tmp_path: Path
) -> None:
    """Review round 2: train() checks the markers before reading the names."""
    project = _copy_project(marked.project, tmp_path / "p")
    _edit(project, "config.json", lambda d: d["feature_names"].__setitem__(0, []))
    _edit(project, STATE, _set_policy("str_alias_v2"))
    module = load_module(project / "train.py", f"h0141_train_{tmp_path.name}")
    assert _reason(lambda: module.train(marked.df, calibrate=False)) == (
        "unsupported_column_name_policy"
    )


def test_the_retrain_cli_names_a_missing_target(
    marked: SimpleNamespace, tmp_path: Path
) -> None:
    """Review round 2: the CLI's own target check must not pre-empt the
    missing-alias diagnostic."""
    project = _copy_project(marked.project, tmp_path / "p")
    data = write_data(marked.df.drop(columns=["target"]), project, "csv")
    with pytest.raises(ValueError) as info:
        run_train(project, data)
    assert getattr(info.value, "reason", None) == "missing_column_alias"
    assert getattr(info.value, "names", None) == ["target"]


def test_a_real_parquet_with_repeated_names_is_refused(marked: SimpleNamespace) -> None:
    """Review round 1: pandas cannot read such a file; the reader must say why."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = marked.project.parent / "repeated.parquet"
    table = pa.Table.from_arrays(
        [pa.array(["a"]), pa.array(["b"]), pa.array([1.0])], names=["7", "7", "x"]
    )
    pq.write_table(table, path)
    module = _predict_module(marked.project)
    assert _reason(lambda: module.read_input(path)) == "column_name_collision"


def test_an_unreadable_csv_header_is_refused(marked: SimpleNamespace) -> None:
    module = _predict_module(marked.project)
    path = _csv(marked.project, "quote.csv", '"7,x\na,1.0\n')
    assert _reason(lambda: module.read_input(path)) == "invalid_column_header"


def test_an_unmarked_project_retrains_without_the_marker(
    marked: SimpleNamespace, tmp_path: Path
) -> None:
    project = _copy_project(marked.project, tmp_path / "p")
    _edit(project, "config.json", _set_policy(None))
    _edit(project, "artifacts/pipeline_state.json", _set_policy(None))
    df = marked.df.copy()
    df.columns = [column_alias(c) for c in df.columns]
    path = project / "train.csv"
    df.to_csv(path, index=False)
    run_train(project, path)
    written = json.loads((project / "artifacts" / "pipeline_state.json").read_text())
    assert "column_name_policy" not in written


def test_a_marked_project_retrains_with_the_marker(
    marked: SimpleNamespace, tmp_path: Path
) -> None:
    project = _copy_project(marked.project, tmp_path / "p")
    run_train(project, write_data(marked.df, project, "csv"))
    written = json.loads((project / "artifacts" / "pipeline_state.json").read_text())
    assert written["column_name_policy"] == "str_alias_v1"
    assert written["feature_names"] == ["7", "x"]
    assert list(written["categories"]) == ["7"]
    assert math.isfinite(
        float(np.asarray(_predict_module(project).predict(marked.X)["pred"])[0])
    )
