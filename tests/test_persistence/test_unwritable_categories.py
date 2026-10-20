"""Category values that cannot be saved are named before export and at fit
(H-0127, #338; narrowed by H-0133).

LightGBM writes each categorical column's categories into the model string as
JSON when a booster is pickled. A value JSON cannot encode used to fail
``Model.export`` half-way through the write with ``Circular reference
detected``. Since H-0133 such a column reaches LightGBM as positions, so most
of those values save; what export and the fit warning still refuse is a value
outside the closed set of ``lizyml.core.category_values`` (here, a tuple of
``np.longdouble`` or a user type). The LightGBM predicate
(``estimators/lgbm/category_json.py``) still decides which columns need
stand-ins, and the adapter check still guards a column kept raw.
"""

from __future__ import annotations

import datetime as dt
import sys
import warnings
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.estimators.base import BaseEstimatorAdapter, UnsaveableCategory
from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.estimators.lgbm.category_json import (
    is_writable,
    shown,
    type_name,
    unwritable_categories,
)
from lizyml.persistence import exporter
from tests._helpers import make_config

#: Category values by kind; LightGBM can write those in WRITABLE.
VALUES: dict[str, list[Any]] = {
    "str": ["a", "b", "c"],
    "int": [1, 2, 3],
    "float": [1.5, 2.5, 3.5],
    "bool": [True, False],
    "np.int64": [np.int64(1), np.int64(2), np.int64(3)],
    "tuple of int": [(1, 2), (3, 4), (5, 6)],
    "tuple of np.int64": [(np.int64(1),), (np.int64(2),), (np.int64(3),)],
    "bytes": [b"a", b"b", b"c"],
    "Decimal": [Decimal("1.5"), Decimal("2.5"), Decimal("3.5")],
    "Timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
    "date": [dt.date(2020, 1, 1), dt.date(2020, 1, 2)],
    "Timedelta": [pd.Timedelta(1, "D"), pd.Timedelta(2, "D")],
    "complex": [1 + 1j, 2 + 2j],
    "frozenset": [frozenset({1}), frozenset({2})],
    "str and bytes": ["a", b"b", "c"],
    "tuple of bytes": [(b"a",), (b"b",), (b"c",)],
    "tuple of Decimal": [(Decimal("1.5"),), (Decimal("2.5"),)],
    "tuple of np.longdouble": [(np.longdouble(1),), (np.longdouble(2),)],
    "nested tuple of bytes": [((b"a",),), ((b"b",),)],
    "np.float32": [np.float32(1.5), np.float32(2.5), np.float32(3.5)],
    "str, int and float": ["a", 1, 2.5],
    "tuple of str": [("a",), ("b",), ("c",)],
    "datetime": [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)],
    "np.datetime64": [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")],
}
#: A category value outside the saveable set of H-0133 (np.longdouble is not
#: in it), which LightGBM cannot write either: export still refuses it.
REFUSED: list[Any] = VALUES["tuple of np.longdouble"]

WRITABLE = {
    "str",
    "int",
    "float",
    "bool",
    "np.int64",
    "tuple of int",
    "tuple of np.int64",
    "np.float32",
    "str, int and float",
    "tuple of str",
}


def _frame(values: list[Any], n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), n)
    col = pd.Categorical([values[i] for i in idx], categories=values)
    x = rng.normal(size=n)
    return pd.DataFrame({"d": col, "x": x, "target": (x > 0).astype(int)})


def _both_answers(values: list[Any]) -> tuple[bool, bool]:
    """(predicate says writable, real writer succeeds), asked from one frame so
    a stack-depth offset cannot separate them near the recursion limit."""
    df = _frame(values)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "min_data_in_leaf": 1},
        lgb.Dataset(df[["d", "x"]], label=df["target"]),
        num_boost_round=1,
        keep_training_booster=True,
    )
    predicted = not unwritable_categories({"d": values})
    try:
        booster.model_to_string()
    except Exception:  # noqa: BLE001 - any writer failure counts
        return predicted, False
    return predicted, True


def _writer_succeeds(values: list[Any]) -> bool:
    return _both_answers(values)[1]


def _buildable(values: list[Any]) -> pd.DataFrame:
    """The frame _both_answers trains on, built at the same stack depth."""
    return _frame(values)


def _fit(df: pd.DataFrame, **features: Any) -> tuple[Model, list[str]]:
    raw = make_config("binary", n_estimators=5)
    if features:
        raw["features"] = features
    model = Model(raw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data=df)
    messages = [str(w.message) for w in caught if "Model.export" in str(w.message)]
    return model, messages


# --- Criterion 4: the predicate against LightGBM's real writer --------------


@pytest.mark.parametrize("kind", sorted(VALUES))
def test_the_predicate_agrees_with_the_real_writer(kind: str) -> None:
    values = VALUES[kind]
    predicted = not unwritable_categories({"d": values})
    assert predicted == _writer_succeeds(values)
    assert predicted == (kind in WRITABLE)


def _nested(depth: int, leaf: int) -> Any:
    value: Any = leaf
    for _ in range(depth):
        value = (value,)
    return value


def test_the_predicate_agrees_at_the_deepest_buildable_nesting() -> None:
    """Near the recursion limit: the deepest tuples pandas can still build,
    found and checked through the same helper so the stack depth is the same."""
    deepest = None
    for depth in range(sys.getrecursionlimit() - 200, sys.getrecursionlimit()):
        try:
            _buildable([_nested(depth, k) for k in range(2)])
        except RecursionError:  # pandas can no longer build the categories
            break
        deepest = depth
    assert deepest is not None
    for depth in range(deepest - 2, deepest + 1):
        predicted, succeeds = _both_answers([_nested(depth, k) for k in range(2)])
        assert predicted == succeeds, depth


def test_the_default_refuses_a_numpy_scalar_whose_item_is_not_plain() -> None:
    """np.longdouble.item() is a np.longdouble; LightGBM recurses on it. The
    default refuses it at once instead of recursing to the interpreter limit
    (the answer is the same either way; the shortcut is pinned here)."""
    from lizyml.estimators.lgbm.category_json import _numpy_default

    with pytest.raises(TypeError):
        _numpy_default(np.longdouble(1))
    assert _numpy_default(np.float32(1)) == 1.0
    assert not is_writable((np.longdouble(1),))
    assert is_writable((np.float32(1),))


# --- Criterion 4b: values whose repr (or type name) raises -------------------


class _Hostile:
    def __init__(self, k: int) -> None:
        self.k = k

    def __hash__(self) -> int:
        return hash(self.k)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Hostile) and other.k == self.k

    def __repr__(self) -> str:
        raise RuntimeError("no repr")

    __str__ = __repr__


class _NamelessMeta(type):
    def __getattribute__(cls, name: str) -> Any:
        if name == "__name__":
            raise RuntimeError("no name")
        return super().__getattribute__(name)


class _Nameless(_Hostile, metaclass=_NamelessMeta):
    pass


def test_the_formatter_never_raises() -> None:
    assert shown(b"a") == "b'a'"
    assert shown(_Hostile(1)) == "<unprintable _Hostile>"
    assert type_name(_Nameless(1)) == "_Nameless"  # read past the metaclass
    with pytest.raises(RuntimeError):
        repr(_Hostile(1))  # the witness: plain repr raises


class _ArrayWithoutList(np.ndarray):
    """A hashable 0-d array whose ``tolist()`` raises: LightGBM's conversion
    calls it, so any exception it raises is the writer's failure too."""

    def __hash__(self) -> int:
        return hash(self.item())

    def tolist(self) -> Any:  # type: ignore[override]
        raise RuntimeError("no list")


def _array_values() -> list[Any]:
    return [np.array(k).view(_ArrayWithoutList) for k in (1, 2)]


def test_an_exception_from_the_conversion_is_unwritable(tmp_path: Path) -> None:
    values = _array_values()
    predicted, succeeds = _both_answers(values)
    assert (predicted, succeeds) == (False, False)
    model, messages = _fit(_frame(values))  # fit warns and completes
    assert len(messages) == 1
    with pytest.raises(LizyMLError) as exc:
        model.export(tmp_path / "out")
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert exc.value.context["type"] == "_ArrayWithoutList"


def test_fit_warns_and_export_refuses_with_safe_text(tmp_path: Path) -> None:
    model, messages = _fit(_frame([_Hostile(0), _Hostile(1)]))
    assert len(messages) == 1 and "<unprintable _Hostile>" in messages[0]
    with pytest.raises(LizyMLError) as exc:
        model.export(tmp_path / "out")
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert exc.value.context["value"] == "<unprintable _Hostile>"
    assert exc.value.context["type"] == "_Hostile"


# --- Criterion 5: Model.export refuses before writing ------------------------


def test_export_names_the_column_and_writes_nothing(tmp_path: Path) -> None:
    model, _ = _fit(_frame(REFUSED))
    out = tmp_path / "out"
    with pytest.raises(LizyMLError) as exc:
        model.export(out)
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert exc.value.context == {
        "column": "d",
        "value": shown(REFUSED[0]),
        "type": "tuple",
        "state": "refit",
    }
    assert "Circular reference" not in str(exc.value)
    assert not out.exists()  # before H-0127 a 728-byte fit_result.pkl was left


def test_export_refuses_before_resolving_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path resolution can create a run directory under ``output_dir``; the
    check runs first, so a refused export creates no directory at all."""
    raw = make_config("binary", n_estimators=5)
    model = Model(raw, output_dir=tmp_path / "runs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data=_frame(REFUSED))
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    def _resolved(*_: Any, **__: Any) -> Path:
        raise AssertionError("the path was resolved before the check")

    monkeypatch.setattr(model, "_resolve_export_path", _resolved)
    with pytest.raises(LizyMLError) as exc:
        model.export()
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_a_failed_export_leaves_an_earlier_export_intact(tmp_path: Path) -> None:
    out = tmp_path / "out"
    good, _ = _fit(_frame(VALUES["str"]))
    good.export(out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    bad, _ = _fit(_frame(REFUSED))
    with pytest.raises(LizyMLError):
        bad.export(out)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    Model.load(out)


@pytest.mark.parametrize("kind", ["str", "tuple of int"])
def test_writable_values_still_export_and_load(kind: str, tmp_path: Path) -> None:
    df = _frame(VALUES[kind])
    model, messages = _fit(df)
    assert messages == []
    model.export(tmp_path / "out")
    loaded = Model.load(tmp_path / "out")
    X = df[["d", "x"]]
    np.testing.assert_allclose(loaded.predict(X).proba, model.predict(X).proba)


# --- Criterion 5b: every saved state is checked, one at a time ---------------

_GOOD = {"encoder": {"categories": {"c": ["a", "b"]}}}
_BAD = {"encoder": {"categories": {"c": ["a", b"x"]}}}


def _results(
    folds: list[dict[str, Any]] | None,
    last: dict[str, Any],
    refit: dict[str, Any],
    *,
    fold_models: list[Any] | None = None,
    refit_model: Any = None,
) -> tuple[Any, Any]:
    """Results whose adapters are LightGBM's unless given (H-0129: each saved
    state is checked by the adapter saved with it)."""
    if fold_models is None:
        fold_models = [LGBMAdapter() for _ in folds] if folds else [LGBMAdapter()]
    fit_result = SimpleNamespace(
        pipeline_state_per_fold=folds, pipeline_state=last, models=fold_models
    )
    model = LGBMAdapter() if refit_model is None else refit_model
    return fit_result, SimpleNamespace(pipeline_state=refit, model=model)


CASES: dict[str, tuple[tuple[Any, Any], str]] = {
    "fold 1": (_results([_BAD, _GOOD, _GOOD], _GOOD, _GOOD), "fold 1"),
    "last fold": (_results([_GOOD, _GOOD, _BAD], _BAD, _GOOD), "fold 3"),
    "refit": (_results([_GOOD, _GOOD, _GOOD], _GOOD, _BAD), "refit"),
    "no per-fold states": (_results(None, _BAD, _GOOD), "last fold"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_saved_state_is_checked(case: str, tmp_path: Path) -> None:
    (fit_result, refit_result), state = CASES[case]
    out = tmp_path / "out"
    with pytest.raises(LizyMLError) as exc:
        exporter.export(out, fit_result, refit_result, {}, "binary")
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert exc.value.context["state"] == state
    assert exc.value.context["column"] == "c"
    # Golden at 050949b (H-0129 keeps the H-0127 message word for word).
    assert exc.value.user_message == (
        f"Model.export cannot save this model: column 'c' ({state} pipeline "
        "state) holds the category value b'x' of type bytes, which LightGBM "
        "cannot write into a saved model (it stores category values as JSON). "
        "Convert the column's values to str (or int, float, bool) and fit again."
    )
    assert not out.exists()


def test_clean_states_pass_the_check() -> None:
    exporter.check_category_values(*_results([_GOOD, _GOOD], _GOOD, _GOOD))
    exporter.check_category_values(*_results(None, _GOOD, _GOOD))


# --- H-0129: persistence asks each state's own adapter ----------------------


class _Adapter(BaseEstimatorAdapter):
    """An adapter that cannot train; only ``unsaveable_categories`` matters."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> Any:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> Any:
        raise NotImplementedError

    def predict_raw(self, X: pd.DataFrame) -> Any:
        raise NotImplementedError

    def importance(self, kind: Any = "split") -> dict[str, float]:
        raise NotImplementedError

    def get_native_model(self) -> Any:
        raise NotImplementedError

    def save_model_text(self, path: str | Path) -> Path:
        raise NotImplementedError


class _Refusing(_Adapter):
    """Refuses column ``c`` in every state it is asked about."""

    def unsaveable_categories(self, pipeline_state: Any) -> list[UnsaveableCategory]:
        return [UnsaveableCategory(column="c", shown="'z'", type_name="str")]


def _silent_folds(n: int) -> list[Any]:
    return [_Adapter() for _ in range(n)]


def test_persistence_does_not_import_a_concrete_estimator() -> None:
    """#360: the check reaches LightGBM's rule only through the adapter."""
    source = Path(exporter.__file__).read_text(encoding="utf-8")
    assert "estimators.lgbm" not in source


def test_the_signature_of_persistence_export_is_unchanged() -> None:
    """#360: the public function keeps its check and its arguments (golden, 050949b).

    H-0130 changed only the text of ``refit_result``'s annotation, from
    ``RefitResult`` to the Layer 2 interface ``RefitResultView``.
    """
    import inspect

    import lizyml.persistence

    assert str(inspect.signature(lizyml.persistence.export)) == (
        "(path: 'str | Path', fit_result: 'FitResult', "
        "refit_result: 'RefitResultView', "
        "config: 'dict[str, Any]', task: 'TaskType', *, "
        "analysis_context: 'AnalysisContext | None' = None, "
        "tuning: 'TuningResult | None' = None, "
        "tuning_fixed_params: 'dict[str, Any] | None' = None, "
        "applied_training_params: 'dict[str, Any] | None' = None, "
        "applied_sample_weight: 'str | None' = None, "
        "declared_categories: 'dict[str, list[Any]] | None' = None, "
        "categories_checked: 'bool' = False) -> 'None'"
    )


def test_categories_checked_is_deprecated_and_cannot_skip_the_check(
    tmp_path: Path,
) -> None:
    """H-0133 P5: the public export always checks; the flag only warns."""
    from lizyml.persistence import export

    model, _ = _fit(_frame(REFUSED))
    state = model._get_fit_state()
    out = tmp_path / "out"
    with (
        pytest.warns(DeprecationWarning, match="Will be removed in v1.0"),
        pytest.raises(LizyMLError) as exc,
    ):
        export(
            out,
            state.fit_result,
            model._refit_result,
            state.cfg.model_dump(),
            state.cfg.task,
            categories_checked=True,
        )
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert not out.exists()


def test_a_fold_adapter_is_asked_about_its_own_state(tmp_path: Path) -> None:
    """Only fold 2's adapter refuses; asking the refit adapter alone would pass."""
    folds = _silent_folds(3)
    folds[1] = _Refusing()
    results = _results(
        [_GOOD, _GOOD, _GOOD], _GOOD, _GOOD, fold_models=folds, refit_model=_Adapter()
    )
    out = tmp_path / "out"
    with pytest.raises(LizyMLError) as exc:
        exporter.export(out, *results, {}, "binary")
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert exc.value.context["state"] == "fold 2"
    assert exc.value.context["value"] == "'z'"
    assert not out.exists()


def test_the_refit_adapter_is_asked_about_the_refit_state() -> None:
    results = _results(
        [_GOOD, _GOOD],
        _GOOD,
        _GOOD,
        fold_models=_silent_folds(2),
        refit_model=_Refusing(),
    )
    with pytest.raises(LizyMLError) as exc:
        exporter.check_category_values(*results)
    assert exc.value.context["state"] == "refit"


def test_the_last_fold_without_per_fold_states_asks_the_last_model() -> None:
    results = _results(
        None,
        _GOOD,
        _GOOD,
        fold_models=[_Adapter(), _Refusing()],
        refit_model=_Adapter(),
    )
    with pytest.raises(LizyMLError) as exc:
        exporter.check_category_values(*results)
    assert exc.value.context["state"] == "last fold"


@pytest.mark.parametrize(
    ("folds", "models"),
    [([_GOOD, _GOOD, _GOOD], 2), ([_GOOD], 2), (None, 0)],
    ids=["fewer models than fold states", "more models", "no models, no per-fold"],
)
def test_a_state_without_its_adapter_is_refused(
    folds: list[dict[str, Any]] | None, models: int, tmp_path: Path
) -> None:
    """A state nothing can be asked about is refused, never skipped (DC1)."""
    results = _results(folds, _GOOD, _GOOD, fold_models=_silent_folds(models))
    out = tmp_path / "out"
    with pytest.raises(LizyMLError) as exc:
        exporter.export(out, *results, {}, "binary")
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert exc.value.context["models"] == models
    assert exc.value.context["fold_states"] == (None if folds is None else len(folds))
    assert not out.exists()


def test_an_adapter_without_the_method_refuses_nothing() -> None:
    """The default answer is H-0127's precondition: nothing is unsaveable."""
    results = _results(
        [_BAD, _BAD], _BAD, _BAD, fold_models=_silent_folds(2), refit_model=_Adapter()
    )
    exporter.check_category_values(*results)
    assert _Adapter().unsaveable_categories(_BAD) == []


@pytest.mark.parametrize("kind", sorted(VALUES))
def test_the_lightgbm_adapter_answers_as_h0127_did(kind: str) -> None:
    values = VALUES[kind]
    state = {"encoder": {"categories": {"d": values}}}
    known = list(pd.CategoricalDtype(values).categories)
    bad = [v for v in known if not is_writable(v)]
    expected = (
        [
            UnsaveableCategory(
                column="d", shown=shown(bad[0]), type_name=type_name(bad[0])
            )
        ]
        if bad
        else []
    )
    assert LGBMAdapter().unsaveable_categories(state) == expected


# --- Criterion 6: the fit warning and the export check name the same columns -

FrameBuilder = Callable[[], tuple[pd.DataFrame, dict[str, Any]]]


def _object_column(kind: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _frame(VALUES[kind])
    df["d"] = pd.Series(list(df["d"]), dtype=object)
    return df, {}


def _fold_varying_strings() -> tuple[pd.DataFrame, dict[str, Any]]:
    """auto_categorical off: a str column with a value only a few rows hold."""
    df = _frame(["a", "b", "c"])
    df["d"] = pd.Series(list(df["d"]), dtype=object)
    df.loc[df.index[:3], "d"] = "rare"
    return df, {"auto_categorical": False}


def _strings_with_missing_rows() -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _frame(VALUES["str"])
    df.loc[df.index[:5], "d"] = np.nan
    return df, {}


BUILDERS: dict[str, FrameBuilder] = {
    **{
        f"category of {k}": (lambda k=k: (_frame(VALUES[k]), {}))
        for k in sorted(VALUES)
    },
    **{f"object of {k}": (lambda k=k: _object_column(k)) for k in sorted(VALUES)},
    "category of str with missing rows": _strings_with_missing_rows,
    "auto_categorical off, fold-varying str": _fold_varying_strings,
}


@pytest.mark.parametrize("case", sorted(BUILDERS))
def test_warned_columns_are_the_refused_columns(case: str, tmp_path: Path) -> None:
    df, features = BUILDERS[case]()
    model, messages = _fit(df, **features)
    warned = {m.split("'")[1] for m in messages}
    try:
        model.export(tmp_path / "out")
        refused: set[str] = set()
    except LizyMLError as exc:
        assert exc.code == ErrorCode.SERIALIZATION_FAILED
        refused = {exc.context["column"]}
    assert warned == refused


def test_the_warning_comes_before_training(monkeypatch: pytest.MonkeyPatch) -> None:
    import lizyml.core.model as model_module

    def _stop(*_: Any, **__: Any) -> Any:
        raise RuntimeError("training started")

    monkeypatch.setattr(model_module.CVTrainer, "fit", _stop)
    model = Model(make_config("binary", n_estimators=5))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="training started"):
            model.fit(data=_frame(REFUSED))
    assert any("Model.export" in str(w.message) for w in caught)
