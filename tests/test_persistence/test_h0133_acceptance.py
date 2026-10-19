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


#: The canonical sample inventory (one entry per admitted type).
from tests.test_core.test_category_values import SAMPLES  # noqa: E402

_DT64_UNITS = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as")


def _type_id(kind: type) -> str:
    """A case key unique per type: NumPy 2 names ``np.bool_`` ``bool`` too."""
    return f"{kind.__module__}.{kind.__qualname__}"


def test_every_admitted_type_has_its_own_case() -> None:
    # Counted against ADMITTED_TYPES itself, not against ids _type_id makes:
    # a colliding id would shrink both sides of a set comparison alike.
    assert len(set(map(_type_id, ADMITTED_TYPES))) == len(ADMITTED_TYPES)
    assert len(LABEL_CASES) == len(ADMITTED_TYPES)
    assert {_type_id(t) for t in ADMITTED_TYPES} <= set(CASES)
    assert set(LABEL_CASES) == {_type_id(t) for t in ADMITTED_TYPES}


def _equal_with_another_hash(a: Any, b: Any) -> bool:
    """Whether *a* == *b* although their hashes differ.

    A zoneinfo ``Timestamp`` with ``fold=1`` equals the pytz one at the same
    instant, but PEP 495 hashes a repeated wall time as ``fold=0``, so the two
    hash differently. pandas, which dedups through a hash table, then keeps
    both as categories; after a round trip it may or may not see them as
    duplicates, and CI failed on it intermittently (H-0134 PR #371). How the
    library should treat such a column is open in #372; until it is decided,
    no case here holds one, and the behaviour is not pinned by a test. Values
    that are equal *and* hash alike (the deliberate fold pairs) are deduped
    consistently and are not such a pair.

    A comparison numpy cannot make (``datetime64`` across far-apart units
    overflows) is between values of different units, which are distinct
    samples; it answers False, keeping both as before.
    """
    if type(a) is not type(b):
        return False
    try:
        equal = bool(a == b)
    except (OverflowError, TypeError, ValueError):
        return False
    return equal and hash(a) != hash(b)


def _distinct_by_equality(samples: Any) -> list[Any]:
    """*samples* without duplicates, including equal values that hash apart.

    ``dict.fromkeys`` drops values that are equal and hash alike; this also
    drops a value equal to an earlier one with another hash (see
    :func:`_equal_with_another_hash`). The pytz ``Timestamp`` dropped from the
    ``Timestamp`` column keeps its own case, ``Timestamp pytz fold``.
    """
    kept: list[Any] = []
    for value in dict.fromkeys(samples):
        if not any(_equal_with_another_hash(value, k) for k in kept):
            kept.append(value)
    return kept


def test_no_derived_case_holds_equal_values_that_hash_apart() -> None:
    for name, values in CASES.items():
        for i, a in enumerate(values):
            for b in values[i + 1 :]:
                assert not _equal_with_another_hash(a, b), (name, a, b)


def _derived_cases() -> dict[str, list[Any]]:
    """Category columns for criterion 1, derived from ``SAMPLES``.

    For every admitted type: a column of its distinct samples, and the same
    with a str mixed in (pandas keeps an object column's values as they are
    more often than a homogeneous one's). Then the axes criterion 1 names:
    every datetime64 / timedelta64 base unit, DST folds, ``datetime``
    boundaries, a pytz zone when installed, and a mixed-type column.
    """
    cases: dict[str, list[Any]] = {}
    for kind, samples in SAMPLES.items():
        distinct = _distinct_by_equality(samples)
        cases[_type_id(kind)] = distinct
        cases[f"{_type_id(kind)} with a str"] = [*distinct, "zz"]
    for unit in _DT64_UNITS:
        cases[f"datetime64[{unit}] with a str"] = [
            np.datetime64(4, unit),
            np.datetime64(5, unit),
            "zz",
        ]
        cases[f"timedelta64[{unit}] with a str"] = [
            np.timedelta64(4, unit),
            np.timedelta64(5, unit),
            "zz",
        ]
    cases["datetime fold"] = [
        dt.datetime(2024, 11, 3, 1, 30, fold=0, tzinfo=NY),
        dt.datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=NY),
    ]
    cases["Timestamp tz fold"] = [
        pd.Timestamp(dt.datetime(2024, 11, 3, 1, 30, fold=0, tzinfo=NY)),
        pd.Timestamp(dt.datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=NY)),
    ]
    cases["datetime fixed offset boundary"] = [
        dt.datetime.min.replace(tzinfo=dt.timezone(dt.timedelta(hours=23))),
        dt.datetime.max.replace(tzinfo=dt.timezone(dt.timedelta(hours=-23))),
    ]
    cases["tuple nested to the limit"] = [_nested(MAX_DEPTH, 1), _nested(MAX_DEPTH, 2)]
    cases["mixed"] = ["a", b"b", decimal.Decimal("3"), (1, 2), frozenset({4})]
    try:  # pandas 2.x (the dependency floor) makes pytz zones from string timezones
        import pytz
    except ImportError:  # pandas 3 does not depend on pytz
        return cases
    ny = pytz.timezone("America/New_York")
    cases["Timestamp pytz fold"] = [
        pd.Timestamp(ny.localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=d))
        for d in (True, False)
    ]
    return cases


CASES = _derived_cases()


def _pandas_cannot_build(values: list[Any]) -> str | None:
    """Why pandas cannot cast a column of *values* to ``category`` at all
    (checked before LizyML runs, so a LizyML error is never skipped)."""
    try:
        # The column the fit builds, in its row order: pandas 2.0's cast can
        # fail on one order of mixed-unit Timestamps and not on another.
        _frame(values)["c"].astype("category")
    except Exception as exc:  # noqa: BLE001 -- pandas' own refusal, reported
        return f"pandas {pd.__version__} cannot build this column: {exc!r}"
    return None


def _learned_types(values: list[Any]) -> set[type]:
    """The types the encoder learns from a column of *values* (pandas casts it
    to ``category`` first, as the data builder does)."""

    def types(value: Any) -> set[type]:
        found = {type(value)}
        if type(value) in (tuple, frozenset):
            for item in value:
                found |= types(item)
        return found

    cats = pd.Series(values * 2, dtype=object).astype("category").cat.categories
    return set().union(*(types(v) for v in cats))


def test_the_cases_reach_every_admitted_type_or_pandas_converts_it() -> None:
    """Every admitted type is learned in some case, or pandas turns a column of
    it into another type before the encoder sees it (then no model can hold
    it, and the codec tests cover it)."""
    learned: set[type] = set()
    for values in CASES.values():
        if _pandas_cannot_build(values) is None:
            learned |= _learned_types(values)
    assert learned <= set(ADMITTED_TYPES), learned - set(ADMITTED_TYPES)
    for kind in set(ADMITTED_TYPES) - learned:
        for column in (SAMPLES[kind], [*SAMPLES[kind], "zz"]):
            if _pandas_cannot_build(column) is None:
                assert kind not in _learned_types(column), kind


def _frame(values: list[Any], n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """The category decides the target (10% noise): the model splits on it,
    so a wrong category mapping changes predictions and SHAP. LightGBM needs
    ``min_data_per_group`` (100) rows per category to split on it."""
    n = n or max(800, 220 * len(values))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), n)
    idx[: len(values)] = np.arange(len(values))  # every value occurs
    x = rng.normal(size=n)
    target = (idx % 2 == 0) ^ (rng.random(n) < 0.1)
    column = pd.Series([values[i] for i in idx], dtype=object)
    return pd.DataFrame({"c": column, "x": x, "target": target.astype(int)})


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
    values = CASES[case]
    reason = _pandas_cannot_build(values)
    if reason is not None:
        pytest.skip(reason)
    df = _frame(values)
    X = df.drop(columns=["target"])
    model = _fit(df)
    # Power: the model uses the column, so a wrong mapping would show.
    assert model.importance(kind="split")["c"] > 0
    learned = model._refit_result.pipeline_state["encoder"]["categories"]["c"]
    assert {type(v) for v in learned} <= set(ADMITTED_TYPES)
    expected = np.asarray(model.predict(X).proba)

    loaded = Model.load(model.export(tmp_path / "artifact"))
    np.testing.assert_array_equal(np.asarray(loaded.predict(X).proba), expected)

    project = tmp_path / "gen"
    model.export_code(project)
    np.testing.assert_allclose(
        generated_prediction(project, X, "binary"), expected, rtol=1e-7, atol=0.0
    )

    # The generated train.py retrains on the same frame and reproduces it.
    train = load_module(project / "train.py", f"gen_train_{abs(hash(case))}")
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


def _fitted_or_skip(values: list[Any], *, expect_warning: bool = False) -> Any:
    """Fit a model on a column of *values*; skip when pandas cannot build it.

    pandas 2.0 cannot cast a tuple holding Decimal NaN, or a column of complex
    NaN, to ``category`` at all: LizyML's checks are never reached, so the
    case does not exist on that version. That is checked before the fit, so an
    error from LizyML itself is never skipped.

    Returns the model when *expect_warning* (and asserts the warning), else
    ``(model, warned)``.
    """
    reason = _pandas_cannot_build(values)
    if reason is not None:
        pytest.skip(reason)
    cfg = make_config("binary", n_estimators=8, n_splits=2)
    cfg["features"] = {"categorical": ["c"]}
    model = Model(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data=_frame(values))
    warned = any("will refuse" in str(w.message) for w in caught)
    if expect_warning:
        assert warned, "fit did not warn about a value outside V"
        return model
    return model, warned


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
    reason = _pandas_cannot_build(CASES[case])
    if reason is not None:
        pytest.skip(reason)
    answers = _answers(_fit(_frame(CASES[case])), tmp_path)
    assert answers == dict.fromkeys(answers, True)


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_the_three_surfaces_refuse_every_column_outside_v(
    case: str, tmp_path: Path
) -> None:
    model = _fitted_or_skip(REFUSED[case], expect_warning=True)
    answers = _answers(model, tmp_path)
    assert answers == dict.fromkeys(answers, False)
    assert not any((tmp_path / p).exists() for p in ("a", "b", "c"))


@pytest.mark.parametrize("case", sorted(NORMALIZED))
def test_values_pandas_normalizes_get_one_answer(case: str, tmp_path: Path) -> None:
    model, warned = _fitted_or_skip(NORMALIZED[case])
    answers = set(_answers(model, tmp_path).values())
    assert answers == {not warned}


#: The label values export_code writes (H-0120's plain set; judgement 3).
_PLAIN_LABEL_TYPES = {
    str,
    int,
    float,
    bool,
    *(
        t
        for t in ADMITTED_TYPES
        if t.__module__ == "numpy"
        and t
        not in (np.complex64, np.complex128, np.bytes_, np.datetime64, np.timedelta64)
    ),
}


def _label_cases() -> dict[str, list[Any]]:
    """Two distinct values of every admitted type, as binary target labels."""
    cases: dict[str, list[Any]] = {}
    for kind, samples in SAMPLES.items():
        # pandas holds some values in no Series (timedelta64 in Y / M units).
        usable = [
            v
            for v in _distinct_by_equality(samples)
            if _pandas_cannot_build([v]) is None
        ]
        cases[_type_id(kind)] = next(
            [a, b]
            for i, a in enumerate(usable)
            for b in usable[i + 1 :]
            if _pandas_can_index([a, b])
        )
    return cases


def _pandas_can_index(labels: list[Any]) -> bool:
    """Whether pandas can key a mapping by these labels, as
    ``TargetEncoder.transform`` does (``Series.map`` of a dict). pandas 2.0
    builds a MultiIndex from tuple keys and raises on ``()``; that is an
    existing limit of tuple labels, not what this test is about, so such
    pairs are not chosen."""
    try:
        pd.Series(["x"] * len(labels), dtype=object).map(
            {v: i for i, v in enumerate(labels)}
        )
    except ValueError:
        return False
    return True


LABEL_CASES = _label_cases()


@pytest.mark.parametrize("case", sorted(LABEL_CASES))
def test_target_labels_keep_the_plain_set(case: str, tmp_path: Path) -> None:
    """Judgement 3, over every admitted type: export_code writes the plain
    label types and refuses the rest. Negative control (g) swaps the label
    check for the category codec and this population catches it."""
    labels = LABEL_CASES[case]
    df = _frame(["a", "b"])
    df["target"] = pd.Series([labels[int(v)] for v in df["target"]], dtype=object)
    model = Model(make_config("binary", n_estimators=8, n_splits=2))
    model.fit(data=df)
    learned = model._get_fit_state().fit_result.target_encoder.classes_
    if all(type(v) in _PLAIN_LABEL_TYPES for v in learned):
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
