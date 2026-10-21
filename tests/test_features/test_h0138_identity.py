"""H-0138 acceptance 3-6: the issues' columns, through the real data layer.

Each column goes through ``dataframe_builder._apply_categorical`` (the fit
entry) and ``NativeFeaturePipeline`` (fit, then transform of raw batches, as
``Model.predict`` does), and through ``Model.fit`` / ``predict``.
"""

from __future__ import annotations

import decimal
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.category_values import IDENTITY_OBJECT_COLUMNS
from lizyml.core.exceptions import LizyMLError
from lizyml.core.specs.feature_spec import FeatureSpec
from lizyml.data.dataframe_builder import _apply_categorical
from lizyml.features.pipelines_native import NativeFeaturePipeline
from tests._helpers import make_config

NP_ONE_TYPES = (np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8,
                np.uint16, np.uint32, np.uint64, np.float32, np.float64,
                np.complex64, np.complex128)  # fmt: skip
NP_INT_TYPES = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                np.uint32, np.uint64)  # fmt: skip
DEPTHS = (1, 2, 3, 8)
POLICIES = ("mode", "nan", "error")


def _wrap(value: Any, depth: int) -> Any:
    for _ in range(depth):
        value = (value,)
    return value


def _fitted(values: list[Any], policy: str) -> NativeFeaturePipeline:
    train = pd.DataFrame({"c": pd.Series([*values, *values, "x"] * 4, dtype=object)})
    X = _apply_categorical(train, FeatureSpec())
    return NativeFeaturePipeline(unseen_policy=policy).fit(X, pd.Series(range(len(X))))  # type: ignore[arg-type]


def _codes(pipeline: NativeFeaturePipeline, batch: list[Any]) -> list[int]:
    out = pipeline.transform(pd.DataFrame({"c": pd.Series(batch, dtype=object)}))
    return [int(x) for x in out["c"].cat.codes]


def _check_batch_independent(values: list[Any], policy: str) -> NativeFeaturePipeline:
    """Every value gets one code whatever the batch order; none is unseen."""
    pipeline = _fitted(values, policy)
    batch = [*values, "x"]
    forward = _codes(pipeline, batch)
    backward = _codes(pipeline, list(reversed(batch)))
    assert forward == list(reversed(backward))
    assert -1 not in forward  # a learned value is never unseen (any policy)
    return pipeline


# --- acceptance 3: #378 -------------------------------------------------------------

ISSUE_378 = [
    (f"{t.__name__}-{order}", pair if order == "a" else pair[::-1])
    for t in NP_INT_TYPES
    for pair in [[decimal.Decimal("1"), t(1)]]
    for order in ("a", "b")
]


def test_the_378_registry_has_16_columns() -> None:
    assert len(ISSUE_378) == 16


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("case_id", "values"), ISSUE_378, ids=[c for c, _ in ISSUE_378]
)
def test_378_decimal_and_numpy_int_are_one_category(
    case_id: str, values: list[Any], policy: str
) -> None:
    del case_id
    pipeline = _check_batch_independent(values, policy)
    codes = _codes(pipeline, [*values, "x"])
    assert codes[0] == codes[1]  # Decimal("1") and the NumPy 1: one category
    assert len(set(codes)) == 2


# --- acceptance 4: #380 -------------------------------------------------------------

ISSUE_380 = [
    (f"t{k}-{depth}-{order}", pair if order == "a" else pair[::-1])
    for k, t in enumerate(NP_ONE_TYPES)
    for depth in DEPTHS
    for pair in [[t(1), _wrap(1, depth)]]
    for order in ("a", "b")
]
ISSUE_380_IDS = {f"t{k}-{d}-{o}" for k in range(13) for d in DEPTHS for o in ("a", "b")}


def test_the_380_registry_is_exactly_the_104_shapes() -> None:
    assert len(ISSUE_380) == 104
    assert {c for c, _ in ISSUE_380} == ISSUE_380_IDS


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("case_id", "values"), ISSUE_380, ids=[c for c, _ in ISSUE_380]
)
def test_380_scalar_and_wrapped_one_are_apart_in_every_order(
    case_id: str, values: list[Any], policy: str
) -> None:
    del case_id
    pipeline = _check_batch_independent(values, policy)
    codes = _codes(pipeline, [*values, "x"])
    assert len(set(codes)) == 3  # the scalar, the tuple and "x" stay apart


# --- acceptance 5: 2**70 beside np.bool_ ---------------------------------------------


@pytest.mark.parametrize("flag", [np.True_, np.False_])
@pytest.mark.parametrize("policy", POLICIES)
def test_a_huge_int_beside_numpy_bool_is_apart(flag: Any, policy: str) -> None:
    values = [2**70, flag]
    pipeline = _check_batch_independent(values, policy)
    assert len(set(_codes(pipeline, [*values, "x"]))) == 3


# --- acceptance 6: the data layer ---------------------------------------------------


def test_a_column_pandas_cannot_cast_stays_object_and_is_named() -> None:
    values = [np.True_, _wrap(1, 2)]  # pandas calls these duplicates
    X = _apply_categorical(
        pd.DataFrame({"c": pd.Series(values * 3, dtype=object)}), FeatureSpec()
    )
    assert X["c"].dtype == object
    assert X.attrs[IDENTITY_OBJECT_COLUMNS] == ["c"]


def test_a_faithful_column_is_cast_as_today_and_not_named() -> None:
    X = _apply_categorical(pd.DataFrame({"c": ["a", "b", "a"]}), FeatureSpec())
    assert isinstance(X["c"].dtype, pd.CategoricalDtype)
    assert IDENTITY_OBJECT_COLUMNS not in X.attrs


@pytest.mark.parametrize(
    "op",
    [
        lambda X: X.copy(),
        lambda X: X.iloc[[2, 0, 1]],
        lambda X: X.iloc[:2],
        lambda X: X.reset_index(drop=True),
        lambda X: X[["c", "n"]],
        lambda X: X.loc[X.index[:2]],
        lambda X: X.take([1, 0]),
    ],
    ids=["copy", "iloc-reorder", "iloc-slice", "reset_index", "columns", "loc", "take"],
)
def test_the_mark_survives_the_frames_training_takes(op: Any) -> None:
    values = [np.True_, _wrap(1, 2), "x"]
    X = _apply_categorical(
        pd.DataFrame({"c": pd.Series(values, dtype=object), "n": [1.0, 2.0, 3.0]}),
        FeatureSpec(),
    )
    assert op(X).attrs.get(IDENTITY_OBJECT_COLUMNS) == ["c"]


def test_a_category_seen_only_in_validation_rows_is_not_learned() -> None:
    """The mark names columns only: a fold fit learns from its own rows."""
    values = [np.True_, _wrap(1, 2)]
    df = pd.DataFrame({"c": pd.Series([*values * 3, "only-in-valid"], dtype=object)})
    X = _apply_categorical(df, FeatureSpec())
    train = X.iloc[:-1]
    pipeline = NativeFeaturePipeline(unseen_policy="error").fit(
        train, pd.Series(range(len(train)))
    )  # type: ignore[arg-type]
    learned = pipeline.get_state()["encoder"]["categories"]["c"]
    assert "only-in-valid" not in learned
    with pytest.raises(LizyMLError, match="unseen"):
        pipeline.transform(X.iloc[-1:])


@pytest.mark.parametrize(
    "values",
    [
        [decimal.Decimal("1"), np.int8(1)],
        [np.int8(1), decimal.Decimal("1")],
        [np.True_, _wrap(1, 1)],
        [np.True_, _wrap(1, 2)],
        [2**70, np.True_],
    ],
    ids=repr,
)
def test_model_fit_and_predict_work_on_the_issue_columns(values: list[Any]) -> None:
    rng = np.random.default_rng(0)
    rows = [*values, "x"] * 20
    df = pd.DataFrame(
        {
            "c": pd.Series(rows, dtype=object),
            "num": rng.normal(size=len(rows)),
            "target": rng.normal(size=len(rows)),
        }
    )
    model = Model(make_config("regression", n_estimators=5, n_splits=2))
    model.fit(data=df)
    first = model.predict(df.drop(columns=["target"])).pred
    shuffled = df.drop(columns=["target"]).iloc[::-1]
    second = model.predict(shuffled).pred
    np.testing.assert_array_equal(first, second[::-1])


# --- acceptance 8: the LightGBM hand-off and the exported state ---------------------


def _model_on(values: list[Any]) -> tuple[Model, pd.DataFrame]:
    rng = np.random.default_rng(0)
    rows = [*values, "x"] * 20
    df = pd.DataFrame(
        {
            "c": pd.Series(rows, dtype=object),
            "num": rng.normal(size=len(rows)),
            "target": rng.normal(size=len(rows)),
        }
    )
    model = Model(make_config("regression", n_estimators=5, n_splits=2))
    model.fit(data=df)
    return model, df


def _refit(model: Model) -> Any:
    return model._refit_result  # noqa: SLF001


REBUILT = [
    [decimal.Decimal("1"), np.int8(1)],  # pandas keeps both: not faithful
    [np.True_, _wrap(1, 2)],  # the cast raises
    [2**70, np.True_],  # the cast raises
]


@pytest.mark.parametrize("values", REBUILT, ids=repr)
def test_a_rebuilt_column_reaches_lightgbm_as_positions(values: list[Any]) -> None:
    model, _ = _model_on(values)
    refit = _refit(model)
    encoder = refit.pipeline_state["encoder"]
    assert "c" in encoder["stand_in_columns"]
    assert "c" in encoder["identity_columns"]
    k = len(encoder["categories"]["c"])
    booster = refit.model.get_native_model()
    assert booster.pandas_categorical == [list(range(k))]


def test_a_faithful_column_reaches_lightgbm_with_its_categories() -> None:
    model, _ = _model_on(["a", "b"])
    refit = _refit(model)
    encoder = refit.pipeline_state["encoder"]
    assert encoder["stand_in_columns"] == []
    booster = refit.model.get_native_model()
    assert booster.pandas_categorical == [encoder["categories"]["c"]]


@pytest.mark.parametrize("values", REBUILT, ids=repr)
def test_the_booster_predicts_the_same_after_a_pickle_round_trip(
    values: list[Any], tmp_path: Any
) -> None:
    import joblib

    model, df = _model_on(values)
    before = model.predict(df.drop(columns=["target"])).pred
    path = tmp_path / "refit.pkl"
    joblib.dump(_refit(model), path)
    model._refit_result = joblib.load(path)  # noqa: SLF001
    after = model.predict(df.drop(columns=["target"])).pred
    np.testing.assert_array_equal(after, before)


def test_export_state_json_writes_the_identity_marker(tmp_path: Any) -> None:
    import json

    values = [np.True_, _wrap(1, 2)]
    pipeline = _fitted(values, "mode")
    state = pipeline.get_state()["encoder"]
    written = json.loads(pipeline.export_state_json(tmp_path / "s.json").read_text())
    assert written["encoder"]["category_identity_version"] == 1
    assert written["encoder"]["identity_columns"] == state["identity_columns"] == ["c"]


def test_declared_categories_keep_their_order() -> None:
    train = pd.DataFrame(
        {"c": pd.Categorical(["a", "b", "a"], categories=["b", "a", "unused"])}
    )
    X = _apply_categorical(train, FeatureSpec())
    pipeline = NativeFeaturePipeline().fit(X, pd.Series(range(3)))  # type: ignore[arg-type]
    assert pipeline.get_state()["encoder"]["categories"]["c"] == ["b", "a", "unused"]


def test_an_unused_unknown_declared_category_is_refused_under_error() -> None:
    train = pd.DataFrame({"c": pd.Series(["a", "b", "a"], dtype=object)})
    X = _apply_categorical(train, FeatureSpec())
    pipeline = NativeFeaturePipeline(unseen_policy="error").fit(X, pd.Series(range(3)))  # type: ignore[arg-type]
    batch = pd.DataFrame({"c": pd.Categorical(["a"], categories=["a", "zz"])})
    with pytest.raises(LizyMLError) as exc:
        pipeline.transform(batch)
    assert exc.value.code.name == "DATA_SCHEMA_INVALID"
    assert exc.value.context["unseen_categories"] == ["zz"]


# --- acceptance 7: H-0136's check on marked states ----------------------------------

EXPORTED = [*REBUILT, [np.int8(1), decimal.Decimal("1")], [np.True_, _wrap(1, 1)]]


@pytest.mark.parametrize("values", EXPORTED, ids=repr)
def test_a_marked_model_exports_and_loads_with_the_same_predictions(
    values: list[Any], tmp_path: Any
) -> None:
    model, df = _model_on(values)
    X = df.drop(columns=["target"])
    before = model.predict(X).pred
    model.export(tmp_path / "artifact")
    loaded = Model.load(tmp_path / "artifact")
    np.testing.assert_array_equal(loaded.predict(X).pred, before)
    np.testing.assert_array_equal(loaded.predict(X.iloc[::-1]).pred, before[::-1])


@pytest.mark.parametrize("values", EXPORTED, ids=repr)
def test_h0136_projections_of_a_marked_state_change_no_code(values: list[Any]) -> None:
    from lizyml.persistence._category_preflight import state_changes

    model, _ = _model_on(values)
    state = _refit(model).pipeline_state
    assert state["encoder"]["category_identity_version"] == 1
    assert state_changes(state) == []
