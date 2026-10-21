"""H-0138 acceptance 9 and 10 for the generated code.

- The identity copy in the generated scripts (``codegen/identity_template.py``,
  inside ``templates._VALUE_CODEC``) gives the library's keys (DC3 pin).
- On the issues' columns, the runtime, the generated ``train.py`` and the
  generated ``predict.py`` learn the same categories and give the same codes.
- An unmarked state (saved before H-0138) generates code that keeps the pandas
  lookup, and an unknown identity marker is refused.

The issue values (tuples, ``Decimal``, NumPy scalars) do not survive CSV or
parquet, so the generated functions run in-process on the frames themselves.
"""

from __future__ import annotations

import decimal
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.category_values import (
    category_identity_key,
    holds_temporal,
    identity_eligible,
)
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from lizyml.features.pipelines_native import NativeFeaturePipeline
from tests.test_codegen._retrain_harness import load_module
from tests.test_codegen.test_h0133_codec_copy import COPY
from tests.test_core.test_category_identity import DISTINCT, EQUAL
from tests.test_core.test_category_values import SAMPLES
from tests.test_features.test_h0138_identity import (
    ISSUE_378,
    ISSUE_380,
    _model_on,
    _refit,
    _wrap,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

_SAMPLE_VALUES = [v for vs in SAMPLES.values() for v in vs]
_OUTSIDE_V = [float("nan"), object(), np.float16(1), [1], decimal.Decimal("NaN")]


# --- the key copy (acceptance 1 and 9: COPY) ----------------------------------------


@pytest.mark.parametrize("value", [v for v in _SAMPLE_VALUES if not holds_temporal(v)])
def test_the_copy_gives_the_library_key(value: Any) -> None:
    assert COPY._identity_key(value) == category_identity_key(value)


@pytest.mark.parametrize(("a", "b"), EQUAL, ids=repr)
def test_the_copy_joins_the_equal_pairs(a: Any, b: Any) -> None:
    assert COPY._identity_key(a) == COPY._identity_key(b)


@pytest.mark.parametrize(("a", "b"), DISTINCT, ids=repr)
def test_the_copy_keeps_the_distinct_pairs_apart(a: Any, b: Any) -> None:
    assert COPY._identity_key(a) != COPY._identity_key(b)


@pytest.mark.parametrize(
    "value",
    [v for v in _SAMPLE_VALUES if holds_temporal(v)] + _OUTSIDE_V,
    ids=repr,
)
def test_the_copy_has_no_key_where_the_library_has_none(value: Any) -> None:
    with pytest.raises(TypeError):
        category_identity_key(value)
    with pytest.raises(TypeError):
        COPY._identity_key(value)


@pytest.mark.parametrize("value", _SAMPLE_VALUES + _OUTSIDE_V, ids=repr)
def test_the_copy_picks_the_same_identity_columns(value: Any) -> None:
    series = pd.Series([value, "x"], dtype=object)
    assert COPY._identity_eligible(COPY._column_values(series)) == identity_eligible(
        series
    )


# --- runtime = train.py = predict.py (acceptance 9) ---------------------------------

COLUMNS = [
    *(values for _, values in ISSUE_378),
    *(values for _, values in ISSUE_380),
    [2**70, np.True_],
    [np.True_, 2**70],
    # Columns that work today, the decision-1 pairs among them: unchanged.
    ["a", "b"],
    *([a, b] for a, b in EQUAL),
]


def _project(values: list[Any], tmp_path: Path) -> tuple[Any, pd.DataFrame, Path]:
    model, df = _model_on(values)
    project = tmp_path / "gen"
    model.export_code(project)
    return model, df, project


def _modules(project: Path) -> tuple[ModuleType, ModuleType]:
    tag = f"{id(project)}"
    return (
        load_module(project / "train.py", f"h0138_train_{tag}"),
        load_module(project / "predict.py", f"h0138_predict_{tag}"),
    )


def _runtime_codes(model: Any, batch: pd.DataFrame) -> list[int]:
    pipeline = NativeFeaturePipeline().load_state(_refit(model).pipeline_state)
    return [int(c) for c in pipeline.transform(batch)["c"].cat.codes]


def _as_ints(codes: Any) -> list[int]:
    return [-1 if np.isnan(c) else int(c) for c in np.asarray(codes, dtype=float)]


def _batches(df: pd.DataFrame) -> list[pd.DataFrame]:
    X = df.drop(columns=["target"])
    unseen = pd.DataFrame({"c": pd.Series(["zz"], dtype=object), "num": [0.0]})
    batch = pd.concat([X.iloc[:12], unseen], ignore_index=True)
    return [batch, batch.iloc[::-1].reset_index(drop=True)]


@pytest.mark.parametrize("values", COLUMNS, ids=repr)
def test_runtime_train_and_predict_give_the_same_codes(
    values: list[Any], tmp_path: Path
) -> None:
    model, df, project = _project(values, tmp_path)
    written = json.loads((project / "artifacts" / "pipeline_state.json").read_text())
    train, predict = _modules(project)
    batches = _batches(df)
    runtime = [_runtime_codes(model, b) for b in batches]
    generated = [_as_ints(predict.transform(b)["c"]) for b in batches]
    assert generated == runtime
    assert runtime[0] == runtime[1][::-1]  # batch-order independent

    # The retrain learns the state the runtime learned, written the same way.
    raw = train.cast_categoricals(df[train.CFG["feature_names"]])
    state = train.fit_pipeline(raw)
    retrained = json.loads((project / "artifacts" / "pipeline_state.json").read_text())
    assert retrained == written
    assert written["category_identity_version"] == 1
    assert written["identity_columns"] == ["c"]
    expected = runtime[0][:12]
    assert _as_ints(train.transform(raw.iloc[:12], state)["c"]) == expected


def test_the_marked_column_is_a_categorical_feature_of_the_export(
    tmp_path: Path,
) -> None:
    """A column the builder kept as object reaches the generated config."""
    model, _, project = _project([np.True_, _wrap(1, 2)], tmp_path)
    config = json.loads((project / "config.json").read_text())
    assert "c" in config["categorical_features"]
    assert config["category_identity_version"] == 1


@pytest.mark.parametrize("policy", ["mode", "nan", "error"])
@pytest.mark.parametrize(
    "values",
    [[decimal.Decimal("1"), np.int8(1)], [np.float32(1), _wrap(1, 3)]],
    ids=repr,
)
def test_every_unseen_policy_matches_the_runtime(
    values: list[Any], policy: str, tmp_path: Path
) -> None:
    train = pd.DataFrame({"c": pd.Series([*values, "x"] * 8, dtype=object)})
    encoder = CategoricalEncoder(unseen_policy=policy).fit(train, ["c"])  # type: ignore[arg-type]
    cats = encoder.get_state()["categories"]["c"]
    mode = encoder.get_state()["modes"]["c"]
    copy_cats, copy_mode = COPY._keyed_categories(train["c"])
    assert [category_identity_key(v) for v in copy_cats] == [
        category_identity_key(v) for v in cats
    ]
    assert category_identity_key(copy_mode) == category_identity_key(mode)

    batch = pd.Series([*values, "zz", None, "x"], dtype=object)
    unchanged = pd.Series(False, index=batch.index)
    if policy == "error":
        with pytest.raises(Exception, match="zz"):
            encoder.transform(pd.DataFrame({"c": batch}))
        with pytest.raises(ValueError, match="zz"):
            COPY._keyed_codes(batch, batch, unchanged, cats, mode, policy, "c")
        return
    runtime = [
        int(c) for c in encoder.transform(pd.DataFrame({"c": batch}))["c"].cat.codes
    ]
    codes, notes = COPY._keyed_codes(batch, batch, unchanged, cats, mode, policy, "c")
    assert _as_ints(codes) == runtime
    assert len(notes) == 1 and "'zz'" in notes[0]


def test_a_declared_list_pandas_cannot_hold_is_built_by_key_in_its_order() -> None:
    """``Decimal("1")`` and ``1`` (``np.int8(1)`` after the codec) are one key:
    pandas refuses them as two categories, so the fit builds the column in the
    declared order."""
    declared = [np.int8(2), decimal.Decimal("1"), 1]
    series = pd.Series([1, 2, 1, None], dtype=object)
    assert COPY._restore_declared(series, declared) is None
    cats, mode = COPY._keyed_categories(series, declared)
    assert cats == [np.int8(2), decimal.Decimal("1")]
    assert category_identity_key(mode) == category_identity_key(1)


def test_a_declared_list_is_restored_with_codes_by_key() -> None:
    series = pd.Series(["b", None, "a", "q"], dtype=object)
    restored = COPY._restore_declared(series, ["a", "b", "unused"])
    assert list(restored.cat.categories) == ["a", "b", "unused"]
    assert restored.cat.codes.tolist() == [1, -1, 0, -1]


# --- unmarked states and unknown markers (acceptance 10) ----------------------------


def _unmarked_project(
    values: list[Any], tmp_path: Path
) -> tuple[Any, pd.DataFrame, Path]:
    """The export of a state saved before H-0138 (no marker)."""
    model, df = _model_on(values)
    encoder = _refit(model).pipeline_state["encoder"]
    del encoder["category_identity_version"], encoder["identity_columns"]
    project = tmp_path / "gen"
    model.export_code(project)
    return model, df, project


@pytest.mark.parametrize(
    ("values", "today_is_right"),
    [
        (["a", "b"], True),
        ([1, "1"], True),
        ([0, -0.0, 2.5], True),
        # #380: today's lookup is wrong here, and differently in the runtime and
        # the generated copy (it depends on the hash seed); the unmarked export
        # keeps it all the same (decision 2).
        ([np.float64(1), _wrap(1, 1)], False),
    ],
    ids=repr,
)
def test_an_unmarked_state_generates_the_pandas_lookup(
    values: list[Any], today_is_right: bool, tmp_path: Path
) -> None:
    model, df, project = _unmarked_project(values, tmp_path)
    config = json.loads((project / "config.json").read_text())
    written = json.loads((project / "artifacts" / "pipeline_state.json").read_text())
    assert config["category_identity_version"] is None
    assert "category_identity_version" not in written
    assert "identity_columns" not in written

    train, predict = _modules(project)
    assert predict._load_pipeline()["identity_columns"] == []
    if today_is_right:
        legacy = CategoricalEncoder().load_state(
            _refit(model).pipeline_state["encoder"]
        )
        for batch in _batches(df):
            old = [int(c) for c in legacy.transform(batch[["c"]])["c"].cat.codes]
            assert _as_ints(predict.transform(batch)["c"]) == old

    state = train.fit_pipeline(train.cast_categoricals(df[train.CFG["feature_names"]]))
    retrained = json.loads((project / "artifacts" / "pipeline_state.json").read_text())
    assert "category_identity_version" not in retrained
    assert "identity_columns" not in state


@pytest.mark.parametrize("marker", [2, "1", None, -1, True, 1.0])
def test_generated_predict_refuses_an_unknown_marker(
    marker: Any, tmp_path: Path
) -> None:
    _, df, project = _project(["a", "b"], tmp_path)
    path = project / "artifacts" / "pipeline_state.json"
    state = json.loads(path.read_text())
    state["category_identity_version"] = marker
    path.write_text(json.dumps(state))
    _, predict = _modules(project)
    with pytest.raises(ValueError, match="category identity version"):
        predict.transform(df.drop(columns=["target"]))


@pytest.mark.parametrize("columns", [None, "c", ["missing"]], ids=repr)
def test_generated_predict_refuses_malformed_identity_columns(
    columns: Any, tmp_path: Path
) -> None:
    _, df, project = _project(["a", "b"], tmp_path)
    path = project / "artifacts" / "pipeline_state.json"
    state = json.loads(path.read_text())
    state["identity_columns"] = columns
    path.write_text(json.dumps(state))
    _, predict = _modules(project)
    with pytest.raises(ValueError, match="identity_columns"):
        predict.transform(df.drop(columns=["target"]))


def test_generated_train_refuses_an_unknown_marker_in_its_config(
    tmp_path: Path,
) -> None:
    _, df, project = _project(["a", "b"], tmp_path)
    path = project / "config.json"
    config = json.loads(path.read_text())
    config["category_identity_version"] = 2
    path.write_text(json.dumps(config))
    train, _ = _modules(project)
    with pytest.raises(ValueError, match="category identity version"):
        train.cast_categoricals(df[train.CFG["feature_names"]])
