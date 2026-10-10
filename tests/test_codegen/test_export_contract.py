"""H-0120 acceptance criteria 2-4: refusal, versions, generated-config shapes.

Criterion 2: the accepted-type check for category values and target labels,
each branch, with nothing written on refusal. Criterion 3: ``_versions`` holds
exactly the four libraries at their export-time versions, and a mismatch in any
one of them warns by name while training completes. Criterion 4: the mandatory
keys are read without defaults, and ``declared_categories`` and
``pipeline_state.json`` have their stated shapes.
"""

from __future__ import annotations

import decimal
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.codegen.config_writer import library_versions
from lizyml.codegen.values import plain_value
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests.test_codegen._retrain_harness import (
    make_config,
    make_frame,
    run_train,
    write_data,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# ---------------------------------------------------------------------------
# Criterion 2: the accepted-type check
# ---------------------------------------------------------------------------

_ACCEPTED: dict[str, tuple[Any, type]] = {
    "str": ("a", str),
    "int": (3, int),
    "float": (0.5, float),
    "bool": (True, bool),
    "np.int64": (np.int64(3), int),
    "np.float64": (np.float64(0.5), float),
    "np.bool_": (np.bool_(True), bool),
    "np.str_": (np.str_("a"), str),
}

_REFUSED: dict[str, Any] = {
    "tuple": (1, 2),
    "bytes": b"a",
    "pd.Timestamp": pd.Timestamp("2026-01-01"),
    "decimal.Decimal": decimal.Decimal("1.5"),
    "np.bytes_": np.bytes_(b"a"),
    "np.datetime64": np.datetime64("2026-01-01"),
    "np.complex128": np.complex128(1 + 2j),
}


@pytest.mark.parametrize("name", sorted(_ACCEPTED))
def test_accepted_values_become_plain_values(name: str) -> None:
    value, expected_type = _ACCEPTED[name]
    plain = plain_value(value, where="test")
    assert type(plain) is expected_type
    assert plain == value


@pytest.mark.parametrize("name", sorted(_REFUSED))
def test_refused_values_raise(name: str) -> None:
    with pytest.raises(LizyMLError) as info:
        plain_value(_REFUSED[name], where="test")
    assert info.value.code == ErrorCode.SERIALIZATION_FAILED


def _category_frame(values: list[Any], n: int = 120) -> pd.DataFrame:
    df = make_frame("binary", n=n)
    df["c"] = pd.Series([values[i % len(values)] for i in range(n)], dtype=object)
    return df


def _target_frame(labels: list[Any], n: int = 120) -> pd.DataFrame:
    df = make_frame("binary", n=n)
    df["target"] = pd.Series([labels[int(v)] for v in df["target"]], dtype=object)
    return df


#: Refused category values that a fit accepts as a categorical column.
_REFUSED_CATEGORY_DATA: dict[str, list[Any]] = {
    "tuple": [(1, 2), (3, 4)],
    "bytes": [b"a", b"b"],
    "pd.Timestamp": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")],
    "decimal.Decimal": [decimal.Decimal("1.5"), decimal.Decimal("2.5")],
}


def _assert_refused_and_nothing_written(model: Model, out: Path) -> None:
    with pytest.raises(LizyMLError) as info:
        model.export_code(out)
    assert info.value.code == ErrorCode.SERIALIZATION_FAILED
    assert not out.exists()


@pytest.mark.parametrize("name", sorted(_REFUSED_CATEGORY_DATA))
def test_export_refuses_a_category_outside_the_set(name: str, tmp_path: Path) -> None:
    df = _category_frame(_REFUSED_CATEGORY_DATA[name])
    cfg = make_config("binary", "kfold")
    cfg["features"] = {"categorical": ["c"]}
    model = Model(cfg)
    model.fit(data=df)
    _assert_refused_and_nothing_written(model, tmp_path / "gen")


#: Refused target labels that a binary fit accepts.
_REFUSED_LABEL_DATA: dict[str, list[Any]] = {
    "tuple": [(0, 0), (1, 1)],
    "bytes": [b"no", b"yes"],
    "pd.Timestamp": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")],
    "decimal.Decimal": [decimal.Decimal("0.5"), decimal.Decimal("1.5")],
}


@pytest.mark.parametrize("name", sorted(_REFUSED_LABEL_DATA))
def test_export_refuses_a_target_label_outside_the_set(
    name: str, tmp_path: Path
) -> None:
    df = _target_frame(_REFUSED_LABEL_DATA[name])
    model = Model(make_config("binary", "kfold"))
    try:
        model.fit(data=df)
    except LizyMLError:
        pytest.skip(f"Model.fit refuses a {name} target before export is reached")
    _assert_refused_and_nothing_written(model, tmp_path / "gen")


def test_accepted_labels_and_categories_export(tmp_path: Path) -> None:
    df = _target_frame(["no", "yes"])
    df["c"] = np.where(df["f0"] > 0, "x", "y")
    model = Model(make_config("binary", "kfold"))
    model.fit(data=df)
    project = tmp_path / "gen"
    model.export_code(project)
    config = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert config["target_encoder"]["classes"] == ["no", "yes"]


# ---------------------------------------------------------------------------
# Criterion 3: versions
# ---------------------------------------------------------------------------

VERSIONED = ("lightgbm", "numpy", "pandas", "scikit-learn")


def _project(tmp_path: Path, task: str = "regression") -> tuple[Path, pd.DataFrame]:
    df = make_frame(task)
    model = Model(make_config(task, "kfold"))
    model.fit(data=df)
    project = tmp_path / "gen"
    model.export_code(project)
    return project, df


def _config(project: Path) -> dict[str, Any]:
    return json.loads((project / "config.json").read_text(encoding="utf-8"))


def _write_config(project: Path, config: dict[str, Any]) -> None:
    (project / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_versions_are_exactly_the_four_at_export(tmp_path: Path) -> None:
    import lightgbm
    import sklearn

    project, _ = _project(tmp_path)
    versions = _config(project)["_versions"]
    assert set(versions) == set(VERSIONED)
    assert versions == {
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    assert versions == library_versions()


@pytest.mark.parametrize("library", VERSIONED)
def test_a_version_mismatch_warns_by_name_and_trains(
    library: str, tmp_path: Path
) -> None:
    project, df = _project(tmp_path)
    config = _config(project)
    config["_versions"][library] = "0.0.0-not-installed"
    _write_config(project, config)
    (project / "artifacts" / "model.txt").unlink()
    data = write_data(df, project, "parquet")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_train(project, data)
    messages = [str(w.message) for w in caught if "differs from" in str(w.message)]
    assert len(messages) == 1, messages
    assert messages[0].startswith(f"{library} ")
    assert (project / "artifacts" / "model.txt").exists()


def test_matching_versions_do_not_warn(tmp_path: Path) -> None:
    project, df = _project(tmp_path)
    data = write_data(df, project, "parquet")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_train(project, data)
    assert not [w for w in caught if "differs from" in str(w.message)]


# ---------------------------------------------------------------------------
# Criterion 4: generated-config shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["inner_valid", "early_stopping_rounds"])
def test_a_missing_mandatory_key_fails_before_training(
    key: str, tmp_path: Path
) -> None:
    project, df = _project(tmp_path)
    config = _config(project)
    del config[key]
    _write_config(project, config)
    (project / "artifacts" / "model.txt").unlink()
    data = write_data(df, project, "parquet")
    with pytest.raises(KeyError, match=key):
        run_train(project, data)
    assert not (project / "artifacts" / "model.txt").exists()


def _declared_model(tmp_path: Path) -> tuple[Model, Path]:
    df = make_frame("binary")
    rng = np.random.default_rng(3)
    df["d"] = pd.Categorical(rng.choice([2, 1], len(df)), categories=[2, 1, 7])
    df["s"] = rng.choice(["q", "p"], len(df))
    model = Model(make_config("binary", "kfold"))
    model.fit(data=df)
    project = tmp_path / "gen"
    model.export_code(project)
    return model, project


def test_declared_categories_shape(tmp_path: Path) -> None:
    model, project = _declared_model(tmp_path)
    declared = _config(project)["declared_categories"]
    # Only the column that was `category` at fit, mapped to its typed list.
    assert declared == {"d": [2, 1, 7]}
    assert all(type(v) is int for v in declared["d"])
    assert model.fit_result.dtypes["d"] == "category"


def test_pipeline_state_shape(tmp_path: Path) -> None:
    model, project = _declared_model(tmp_path)
    state = json.loads(
        (project / "artifacts" / "pipeline_state.json").read_text(encoding="utf-8")
    )
    encoder = model._refit_result.pipeline_state["encoder"]  # type: ignore[union-attr]
    assert set(state["categories"]) == set(encoder["categories"])
    for col, entry in state["categories"].items():
        assert set(entry) == {"categories", "mode"}
        expected = [plain_value(v, where=col) for v in encoder["categories"][col]]
        assert entry["categories"] == expected
        assert [type(v) for v in entry["categories"]] == [type(v) for v in expected]
        mode = plain_value(encoder["modes"][col], where=col)
        assert entry["mode"] == mode and type(entry["mode"]) is type(mode)
