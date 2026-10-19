"""H-0133 acceptance (criteria 1, 2, 3, 7): saveable JSON-unsafe categories.

Criterion 1: for every admitted type, a model fitted on a category column of
it predicts the same after ``Model.export`` -> ``Model.load``, through the
generated ``predict.py`` reading ``export_code``'s artifacts, and after a
retrain with the generated ``train.py``. The population is derived from
``ADMITTED_TYPES``: a test fails when a type has no case.

Criterion 2: a model with a stand-in column gives the same fold SHAP
importance, per-row SHAP and calibrated prediction after load.

Criterion 3: ``Model.export``, ``lizyml.persistence.export`` and
``export_code`` accept and refuse the same feature-category values; target
labels keep the plain set on ``export_code``.

Criterion 7: what H-0133 does not change -- raw columns' LightGBM categories,
``unseen_policy``.
"""

from __future__ import annotations

import datetime as dt
import decimal
import warnings
import zoneinfo
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.category_values import ADMITTED_TYPES, MAX_DEPTH
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_config
from tests.test_codegen._retrain_harness import generated_prediction, load_module

NY = zoneinfo.ZoneInfo("America/New_York")


def _nested(depth: int, leaf: Any) -> Any:
    value = leaf
    for _ in range(depth):
        value = (value,)
    return value


#: Category columns, two or more distinct values each (criterion 1).
CASES: dict[str, list[Any]] = {
    "str": ["a", "b", "c"],
    "int": [1, 2, 3],
    "float": [0.5, -0.0, 2.5],
    "bool": [True, False],
    "bytes": [b"a", b"b", b"c"],
    "Decimal": [decimal.Decimal("1.50"), decimal.Decimal("2"), decimal.Decimal("-3")],
    "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
    "datetime": [dt.datetime(2024, 1, 1, 9), dt.datetime(2024, 1, 2, 9)],
    "datetime fold": [
        dt.datetime(2024, 11, 3, 1, 30, fold=0, tzinfo=NY),
        dt.datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=NY),
    ],
    "datetime fixed offset boundary": [
        dt.datetime.min.replace(tzinfo=dt.timezone(dt.timedelta(hours=23))),
        dt.datetime.max.replace(tzinfo=dt.timezone(dt.timedelta(hours=-23))),
    ],
    "timedelta": [dt.timedelta(days=1), dt.timedelta(seconds=5)],
    "complex": [complex(1, 2), complex(-1, 0)],
    "Timestamp naive s": [
        pd.Timestamp("2024-01-01").as_unit("s"),
        pd.Timestamp("2024-01-02").as_unit("s"),
    ],
    "Timestamp ms": [
        pd.Timestamp("2024-01-01").as_unit("ms"),
        pd.Timestamp("2024-01-02").as_unit("ms"),
    ],
    "Timestamp us": [
        pd.Timestamp("2024-01-01").as_unit("us"),
        pd.Timestamp("2024-01-02").as_unit("us"),
    ],
    "Timestamp tz fold": [
        pd.Timestamp(dt.datetime(2024, 11, 3, 1, 30, fold=0, tzinfo=NY)),
        pd.Timestamp(dt.datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=NY)),
    ],
    "Timedelta": [pd.Timedelta(1, "D"), pd.Timedelta(2, "h")],
    "np.bool_": [np.bool_(True), np.bool_(False)],
    "np.int8": [np.int8(1), np.int8(2)],
    "np.int16": [np.int16(1), np.int16(2)],
    "np.int32": [np.int32(1), np.int32(2)],
    "np.int64": [np.int64(1), np.int64(2)],
    "np.uint8": [np.uint8(1), np.uint8(2)],
    "np.uint16": [np.uint16(1), np.uint16(2)],
    "np.uint32": [np.uint32(1), np.uint32(2)],
    "np.uint64": [np.uint64(1), np.uint64(2)],
    "np.float32": [np.float32(0.5), np.float32(1.5)],
    "np.float64": [np.float64(0.5), np.float64(1.5)],
    "np.complex64": [np.complex64(1 + 1j), np.complex64(2)],
    "np.complex128": [np.complex128(1 + 1j), np.complex128(2)],
    "np.str_": [np.str_("a"), np.str_("b")],
    "np.bytes_": [np.bytes_(b"a"), np.bytes_(b"b")],
    "np.datetime64 D": [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
    "np.datetime64 in a mixed column": [np.datetime64(4, "s"), "x"],
    "np.timedelta64": [np.timedelta64(1, "D"), np.timedelta64(3, "15m")],
    "tuple": [(1, "a"), (2, b"b")],
    "tuple nested to the limit": [_nested(MAX_DEPTH, 1), _nested(MAX_DEPTH, 2)],
    "frozenset": [frozenset({1, 2}), frozenset({"a", b"b"})],
    "mixed": ["a", b"b", decimal.Decimal("3"), (1, 2), frozenset({4})],
}

try:  # pandas 2.x (the dependency floor) makes pytz zones from string timezones
    import pytz
except ImportError:  # pandas 3 does not depend on pytz
    pytz = None
if pytz is not None:
    _ny = pytz.timezone("America/New_York")
    CASES["Timestamp pytz fold"] = [
        pd.Timestamp(_ny.localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=d))
        for d in (True, False)
    ]


def test_the_cases_cover_every_admitted_type() -> None:
    def types(value: Any) -> set[type]:
        found = {type(value)}
        if type(value) in (tuple, frozenset):
            for item in value:
                found |= types(item)
        return found

    covered = set().union(*(types(v) for vs in CASES.values() for v in vs))
    assert set(ADMITTED_TYPES) <= covered, set(ADMITTED_TYPES) - covered


def _frame(values: list[Any], n: int = 800, seed: int = 0) -> pd.DataFrame:
    """The category decides the target (10% noise): the model splits on it,
    so a wrong category mapping changes predictions and SHAP. LightGBM needs
    ``min_data_per_group`` (100) rows per category to split on it."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), n)
    idx[: len(values)] = np.arange(len(values))  # every value occurs
    x = rng.normal(size=n)
    target = (idx % 2 == 0) ^ (rng.random(n) < 0.1)
    column = pd.Series([values[i] for i in idx], dtype=object)
    return pd.DataFrame({"c": column, "x": x, "target": target.astype(int)})


@pytest.mark.parametrize(
    "values", [["a", "b", "c", "d"], [b"a", b"b", b"c", b"d"]], ids=["str", "bytes"]
)
def test_the_frame_makes_the_model_use_the_column(values: list[Any]) -> None:
    """Power check: the tests below could not see a wrong mapping otherwise."""
    model = _fit(_frame(values))
    assert model.importance(kind="split")["c"] > 0


def _fit(df: pd.DataFrame, **cfg_overrides: Any) -> Model:
    cfg = make_config("binary", n_estimators=8, n_splits=2, **cfg_overrides)
    cfg["features"] = {"categorical": ["c"]}
    model = Model(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data=df)
    assert not [w for w in caught if "will refuse" in str(w.message)]
    return model


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_admitted_type_saves_and_predicts_the_same(
    case: str, tmp_path: Path
) -> None:
    df = _frame(CASES[case])
    X = df.drop(columns=["target"])
    model = _fit(df)
    expected = np.asarray(model.predict(X).proba)

    loaded = Model.load(model.export(tmp_path / "artifact"))
    np.testing.assert_array_equal(np.asarray(loaded.predict(X).proba), expected)

    project = tmp_path / "gen"
    model.export_code(project)
    np.testing.assert_allclose(
        generated_prediction(project, X, "binary"), expected, rtol=1e-7, atol=0.0
    )

    # The generated train.py retrains on the same frame and reproduces it.
    train = load_module(project / "train.py", f"gen_train_{case}")
    train.train(df.copy(), calibrate=False)
    np.testing.assert_allclose(
        generated_prediction(project, X, "binary"), expected, rtol=1e-7, atol=0.0
    )


# --- Criterion 2: stand-in columns on every path that reads a state -----------


@pytest.fixture(scope="module")
def stood(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Model, Model, pd.DataFrame]:
    df = _frame([b"a", b"b", b"c", b"d"])
    cfg = make_config("binary", n_estimators=8, n_splits=3, calibration="platt")
    cfg["features"] = {"categorical": ["c"]}
    model = Model(cfg)
    model.fit(data=df)
    state = model._get_fit_state().fit_result
    assert all(
        s["encoder"]["stand_in_columns"] == ["c"] for s in state.pipeline_state_per_fold
    )
    loaded = Model.load(model.export(tmp_path_factory.mktemp("stood") / "a"))
    return model, loaded, df


def test_fold_shap_importance_is_the_same_after_load(
    stood: tuple[Model, Model, pd.DataFrame],
) -> None:
    pytest.importorskip("shap")
    model, loaded, _ = stood
    assert loaded.importance(kind="shap") == pytest.approx(
        model.importance(kind="shap"), rel=1e-12
    )


def test_per_row_shap_is_the_same_after_load(
    stood: tuple[Model, Model, pd.DataFrame],
) -> None:
    pytest.importorskip("shap")
    model, loaded, df = stood
    X = df.drop(columns=["target"])
    np.testing.assert_allclose(
        loaded.predict(X, return_shap=True).shap_values,
        model.predict(X, return_shap=True).shap_values,
        rtol=1e-12,
    )


def test_fold_shap_of_a_stand_in_column_equals_that_of_the_same_codes_as_str() -> None:
    """The same codes give the same model: a fold state recorded in the wrong
    form (negative control b) would explain a positions booster with raw
    categories and differ here, even if in-memory and loaded agreed."""
    pytest.importorskip("shap")
    raw = [b"a", b"b", b"c", b"d"]
    stood_model = _fit(_frame(raw))
    as_str = _fit(_frame([v.decode() for v in raw]))
    assert stood_model.importance(kind="shap") == pytest.approx(
        as_str.importance(kind="shap"), rel=1e-12
    )


def test_calibrated_prediction_is_the_same_after_load(
    stood: tuple[Model, Model, pd.DataFrame],
) -> None:
    model, loaded, df = stood
    X = df.drop(columns=["target"])
    np.testing.assert_array_equal(
        np.asarray(loaded.predict(X).proba), np.asarray(model.predict(X).proba)
    )


# --- Criterion 3: one population over the three surfaces ----------------------


class _Opaque:
    def __init__(self, k: int) -> None:
        self.k = k

    def __hash__(self) -> int:
        return hash(self.k)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Opaque) and other.k == self.k

    def __repr__(self) -> str:
        return f"_Opaque({self.k})"


class _Str(str):
    pass


#: Feature-category columns outside V that reach the encoder as such.
REFUSED: dict[str, list[Any]] = {
    "tuple with NaN": [(1, float("nan")), (2, 3.0)],
    "tuple with Decimal NaN": [(decimal.Decimal("NaN"),), (1,)],
    "tuple with complex NaN": [(complex(float("nan"), 1),), (1,)],
    "tuple with NaT": [(np.datetime64("NaT"),), (1,)],
    "nested past the limit": [_nested(MAX_DEPTH + 1, 1), _nested(MAX_DEPTH + 1, 2)],
    "tuple of a str subclass": [(_Str("a"),), ("b",)],
    "user type": [_Opaque(1), _Opaque(2)],
    "np.longdouble": [(np.longdouble(1),), (np.longdouble(2),)],
    "tuple of a datetime with a no_cache ZoneInfo": [
        (dt.datetime(2024, 1, 1, tzinfo=zoneinfo.ZoneInfo.no_cache("Asia/Tokyo")),),
        (1,),
    ],
}

#: Values outside V that pandas itself turns into V values (or drops) when it
#: casts the column at fit, so the encoder never sees them
#: (``probes/pandas_normalizes_probe.out`` on pandas 3.0.1). Whatever the
#: pandas version does, the three surfaces must answer alike, and fit warns
#: exactly when they refuse.
NORMALIZED: dict[str, list[Any]] = {
    "complex NaN": [complex(float("nan"), 1), complex(1, 1)],
    "str subclass": [_Str("a"), "b"],
    "datetime with a no_cache ZoneInfo": [
        dt.datetime(2024, 1, 1, tzinfo=zoneinfo.ZoneInfo.no_cache("Asia/Tokyo")),
        dt.datetime(2024, 1, 2, tzinfo=zoneinfo.ZoneInfo.no_cache("Asia/Tokyo")),
    ],
}


def _answers(model: Model, tmp_path: Path) -> dict[str, bool]:
    from lizyml.persistence import export as persistence_export

    state = model._get_fit_state()

    def ok(action: Any) -> bool:
        try:
            action()
        except LizyMLError as exc:
            assert exc.code == ErrorCode.SERIALIZATION_FAILED
            return False
        return True

    return {
        "Model.export": ok(lambda: model.export(tmp_path / "a")),
        "persistence.export": ok(
            lambda: persistence_export(
                tmp_path / "b",
                state.fit_result,
                model._refit_result,
                state.cfg.model_dump(),
                state.cfg.task,
                declared_categories=state.declared_categories,
            )
        ),
        "export_code": ok(lambda: model.export_code(tmp_path / "c")),
    }


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_three_surfaces_accept_every_admitted_column(
    case: str, tmp_path: Path
) -> None:
    answers = _answers(_fit(_frame(CASES[case])), tmp_path)
    assert answers == dict.fromkeys(answers, True)


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_the_three_surfaces_refuse_every_column_outside_v(
    case: str, tmp_path: Path
) -> None:
    cfg = make_config("binary", n_estimators=8, n_splits=2)
    cfg["features"] = {"categorical": ["c"]}
    model = Model(cfg)
    with pytest.warns(UserWarning, match="Model.export and export_code will refuse"):
        model.fit(data=_frame(REFUSED[case]))
    answers = _answers(model, tmp_path)
    assert answers == dict.fromkeys(answers, False)
    assert not any((tmp_path / p).exists() for p in ("a", "b", "c"))


@pytest.mark.parametrize("case", sorted(NORMALIZED))
def test_values_pandas_normalizes_get_one_answer(case: str, tmp_path: Path) -> None:
    cfg = make_config("binary", n_estimators=8, n_splits=2)
    cfg["features"] = {"categorical": ["c"]}
    model = Model(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data=_frame(NORMALIZED[case]))
    warned = any("will refuse" in str(w.message) for w in caught)
    answers = set(_answers(model, tmp_path).values())
    assert answers == {not warned}


#: Target labels by admitted type: export_code writes only the plain set.
LABELS: dict[str, tuple[list[Any], bool]] = {
    "str": (["no", "yes"], True),
    "int": ([0, 1], True),
    "bytes": ([b"no", b"yes"], False),
    "Decimal": ([decimal.Decimal("0.5"), decimal.Decimal("1.5")], False),
    "Timestamp": ([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")], False),
    "tuple": ([(0, 0), (1, 1)], False),
    "date": ([dt.date(2024, 1, 1), dt.date(2024, 1, 2)], False),
}


@pytest.mark.parametrize("case", sorted(LABELS))
def test_target_labels_keep_the_plain_set(case: str, tmp_path: Path) -> None:
    """Judgement 3; negative control (g) swaps the label check for the
    category codec and this population catches it."""
    labels, written = LABELS[case]
    df = _frame(["a", "b"])
    df["target"] = pd.Series([labels[int(v)] for v in df["target"]], dtype=object)
    model = Model(make_config("binary", n_estimators=8, n_splits=2))
    model.fit(data=df)
    if written:
        model.export_code(tmp_path / "gen")
        return
    with pytest.raises(LizyMLError) as info:
        model.export_code(tmp_path / "gen")
    assert info.value.code == ErrorCode.SERIALIZATION_FAILED
    assert info.value.context["where"] == "target labels"


# --- Criterion 7: what does not change ---------------------------------------


def test_a_raw_column_keeps_its_raw_lightgbm_categories() -> None:
    model = _fit(_frame(["a", "b", "c"]))
    booster = model._refit_result.model.get_native_model()
    assert booster.pandas_categorical == [["a", "b", "c"]]


def test_a_stand_in_column_hands_lightgbm_positions() -> None:
    model = _fit(_frame([b"a", b"b", b"c"]))
    booster = model._refit_result.model.get_native_model()
    assert booster.pandas_categorical == [[0, 1, 2]]


@pytest.mark.parametrize("policy", ["mode", "nan", "error"])
def test_unseen_policy_is_applied_to_a_stand_in_column_as_before(
    policy: str, tmp_path: Path
) -> None:
    """The same replacements and reports as an equivalent str column."""
    new = pd.DataFrame({"c": pd.Series([b"zz", b"a"], dtype=object), "x": [0.1, -0.2]})
    results = []
    for values, unseen in (
        ([b"a", b"b", b"c"], new),
        (["a", "b", "c"], new.assign(c=["zz", "a"])),
    ):
        cfg = make_config("binary", n_estimators=8, n_splits=2)
        cfg["features"] = {"categorical": ["c"], "unseen_policy": policy}
        model = Model(cfg)
        model.fit(data=_frame(values))
        loaded = Model.load(
            model.export(tmp_path / f"{policy}-{type(values[0]).__name__}")
        )
        if policy == "error":
            with pytest.raises(LizyMLError) as info:
                loaded.predict(unseen)
            results.append(info.value.code)
            continue
        out = loaded.predict(unseen)
        results.append((np.asarray(out.proba).tolist(), len(out.warnings)))
    assert results[0] == results[1]
