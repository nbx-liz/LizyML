"""Category values LightGBM cannot write are named before export and at fit
(H-0127, #338).

LightGBM writes each categorical column's categories into the model string as
JSON when a booster is pickled. A value JSON cannot encode used to fail
``Model.export`` half-way through the write with ``Circular reference
detected``. One predicate (``estimators/lgbm/category_json.py``) decides, and
both the export check and the fit warning use it.
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
}
WRITABLE = {
    "str",
    "int",
    "float",
    "bool",
    "np.int64",
    "tuple of int",
    "tuple of np.int64",
}


def _frame(values: list[Any], n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), n)
    col = pd.Categorical([values[i] for i in idx], categories=values)
    x = rng.normal(size=n)
    return pd.DataFrame({"d": col, "x": x, "target": (x > 0).astype(int)})


def _writer_succeeds(values: list[Any]) -> bool:
    df = _frame(values)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "min_data_in_leaf": 1},
        lgb.Dataset(df[["d", "x"]], label=df["target"]),
        num_boost_round=1,
        keep_training_booster=True,
    )
    try:
        booster.model_to_string()
    except (TypeError, ValueError, RecursionError):
        return False
    return True


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
    agreements: list[bool] = []
    for depth in range(sys.getrecursionlimit() - 200, sys.getrecursionlimit()):
        values = [_nested(depth, k) for k in range(2)]
        try:
            succeeds = _writer_succeeds(values)
        except RecursionError:  # pandas can no longer build the categories
            break
        agreements.append((not unwritable_categories({"d": values})) == succeeds)
    assert len(agreements) >= 3
    assert agreements[-3:] == [True, True, True]


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
    model, _ = _fit(_frame(VALUES["bytes"]))
    out = tmp_path / "out"
    with pytest.raises(LizyMLError) as exc:
        model.export(out)
    assert exc.value.code == ErrorCode.SERIALIZATION_FAILED
    assert exc.value.context == {
        "column": "d",
        "value": "b'a'",
        "type": "bytes",
        "state": "refit",
    }
    assert "Circular reference" not in str(exc.value)
    assert not out.exists()  # before H-0127 a 728-byte fit_result.pkl was left


def test_a_failed_export_leaves_an_earlier_export_intact(tmp_path: Path) -> None:
    out = tmp_path / "out"
    good, _ = _fit(_frame(VALUES["str"]))
    good.export(out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    bad, _ = _fit(_frame(VALUES["bytes"]))
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
    folds: list[dict[str, Any]] | None, last: dict[str, Any], refit: dict[str, Any]
) -> tuple[Any, Any]:
    fit_result = SimpleNamespace(pipeline_state_per_fold=folds, pipeline_state=last)
    return fit_result, SimpleNamespace(pipeline_state=refit)


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
    assert not out.exists()


def test_clean_states_pass_the_check() -> None:
    exporter.check_category_values(*_results([_GOOD, _GOOD], _GOOD, _GOOD))
    exporter.check_category_values(*_results(None, _GOOD, _GOOD))


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


BUILDERS: dict[str, FrameBuilder] = {
    **{
        f"category of {k}": (lambda k=k: (_frame(VALUES[k]), {}))
        for k in sorted(VALUES)
    },
    **{
        f"object of {k}": (lambda k=k: _object_column(k))
        for k in ("bytes", "Decimal", "str")
    },
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
            model.fit(data=_frame(VALUES["bytes"]))
    assert any("Model.export" in str(w.message) for w in caught)
