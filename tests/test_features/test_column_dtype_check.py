"""Predict-time dtype check -> ``INCOMPATIBLE_COLUMNS`` (H-0106, #263).

A column that was numeric (or bool) at fit must arrive with a numeric dtype:
its scalar type is a numpy integer, floating or bool type, excluding
``timedelta64`` and ``longdouble``. That set is the one LightGBM accepts; on 33
arrival dtypes the rule and a real ``Model.predict`` agreed on every one
(``docs/audits/2026-09-defect-discovery/results/pr6_measurements.txt``).
Before this, every refused arrival escaped as a raw LightGBM or numpy error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.estimators.lgbm.provider import LGBMProvider
from tests._helpers import make_config
from tests.test_features.test_pipeline_conformance import MinimalPipeline

_N = 120
_RNG = np.random.default_rng(0)
_VALS = pd.Series(_RNG.integers(0, 5, _N))
_B = _RNG.normal(size=_N)
_Y = ((_VALS.to_numpy() + _B) > 2).astype(int)

#: dtype name -> column of that dtype built from the same values.
_DTYPES: dict[str, Callable[[], pd.Series]] = {
    "int8": lambda: _VALS.astype("int8"),
    "int16": lambda: _VALS.astype("int16"),
    "int32": lambda: _VALS.astype("int32"),
    "int64": lambda: _VALS.astype("int64"),
    "uint8": lambda: _VALS.astype("uint8"),
    "uint64": lambda: _VALS.astype("uint64"),
    "float16": lambda: _VALS.astype("float16"),
    "float32": lambda: _VALS.astype("float32"),
    "float64": lambda: _VALS.astype("float64"),
    "Int8": lambda: _VALS.astype("Int8"),
    "Int64": lambda: _VALS.astype("Int64"),
    "Int64_na": lambda: _VALS.astype("Int64").mask(_VALS == 0),
    "UInt32": lambda: _VALS.astype("UInt32"),
    "Float32": lambda: _VALS.astype("Float32"),
    "Float64": lambda: _VALS.astype("Float64"),
    "bool": lambda: (_VALS % 2).astype(bool),
    "boolean": lambda: (_VALS % 2).astype(bool).astype("boolean"),
    "sparse_float": lambda: _VALS.astype("float64").astype(
        pd.SparseDtype("float64", np.nan)
    ),
    "complex128": lambda: _VALS.astype("complex128"),
    "longdouble": lambda: _VALS.astype(np.longdouble),
    "timedelta": lambda: pd.to_timedelta(_VALS, unit="D"),
    "datetime": lambda: pd.to_datetime(_VALS, unit="D"),
    "datetime_tz": lambda: pd.to_datetime(_VALS, unit="D").dt.tz_localize("UTC"),
    "period": lambda: pd.Series(pd.period_range("2000-01", periods=_N, freq="M")),
    "interval": lambda: pd.Series(
        pd.arrays.IntervalArray.from_breaks(np.arange(_N + 1))
    ),
    "str": lambda: _VALS.astype(str).astype("str"),
    "string_python": lambda: _VALS.astype(str).astype("string[python]"),
    "string_pyarrow": lambda: _VALS.astype(str).astype("string[pyarrow]"),
    "object_num": lambda: pd.Series(list(_VALS), dtype=object),
    "category": lambda: _VALS.astype("category"),
    "int64_pyarrow": lambda: _VALS.astype("int64[pyarrow]"),
    "float64_pyarrow": lambda: _VALS.astype("float64[pyarrow]"),
    "bool_pyarrow": lambda: (_VALS % 2).astype(bool).astype("bool[pyarrow]"),
}

#: Arrivals the rule accepts for a numeric-at-fit column (measured: each
#: predicts successfully today; every other one fails with a raw error).
_NUMERIC = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint64",
    "float16",
    "float32",
    "float64",
    "Int8",
    "Int64",
    "Int64_na",
    "UInt32",
    "Float32",
    "Float64",
    "bool",
    "boolean",
    "sparse_float",
}

#: Dtypes LightGBM fits; the rest are refused at fit (measured).
_FITTABLE = _NUMERIC | {
    "str",
    "string_python",
    "string_pyarrow",
    "object_num",
    "category",
}


def _fit(fit_dtype: str = "float64") -> Model:
    df = pd.DataFrame({"a": _DTYPES[fit_dtype](), "b": _B, "y": _Y})
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    cfg["data"]["target"] = "y"
    model = Model(cfg)
    model.fit(data=df)
    return model


def _frame(dtype: str) -> pd.DataFrame:
    return pd.DataFrame({"a": _DTYPES[dtype](), "b": _B})


@pytest.fixture(scope="module")
def float_model() -> Model:
    return _fit("float64")


@pytest.mark.parametrize("arrival", sorted(_DTYPES))
def test_numeric_at_fit_arrival_matrix(float_model: Model, arrival: str) -> None:
    X = _frame(arrival)
    if arrival in _NUMERIC:
        float_model.predict(X)
        return
    with pytest.raises(LizyMLError) as exc:
        float_model.predict(X)
    assert exc.value.code == ErrorCode.INCOMPATIBLE_COLUMNS
    assert exc.value.context["columns"] == [
        {"column": "a", "fit_dtype": "float64", "predict_dtype": str(X["a"].dtype)}
    ]


@pytest.mark.parametrize("fit_dtype", sorted(_NUMERIC))
def test_every_numeric_fit_dtype_is_checked(fit_dtype: str) -> None:
    model = _fit(fit_dtype)
    with pytest.raises(LizyMLError) as exc:
        model.predict(_frame("str"))
    assert exc.value.code == ErrorCode.INCOMPATIBLE_COLUMNS


_CATEGORICAL_FITS = ["str", "object_num", "category"]

#: #309: the encoder fails inside pandas when integer categories meet these
#: arrivals. Strict, so the fix turns them green and must remove the marker.
_ENCODER_GAPS = {
    (fit, arrival)
    for fit in ("object_num", "category")
    for arrival in ("float16", "longdouble")
}


@pytest.fixture(scope="module")
def categorical_models() -> dict[str, Model]:
    return {fit: _fit(fit) for fit in _CATEGORICAL_FITS}


@pytest.mark.parametrize(
    ("fit_dtype", "arrival"),
    [
        pytest.param(
            fit,
            arrival,
            marks=pytest.mark.xfail(
                strict=True, reason="#309: raw pandas error in CategoricalEncoder"
            ),
        )
        if (fit, arrival) in _ENCODER_GAPS
        else (fit, arrival)
        for fit in _CATEGORICAL_FITS
        for arrival in sorted(_DTYPES)
    ],
)
def test_categorical_at_fit_is_not_dtype_checked(
    categorical_models: dict[str, Model], fit_dtype: str, arrival: str
) -> None:
    """The dtype rule covers numeric-at-fit columns only; a categorical-at-fit
    column's values go to the encoder (unseen values follow ``unseen_policy``)."""
    categorical_models[fit_dtype].predict(_frame(arrival))


def test_missing_column_is_reported_before_dtype(float_model: Model) -> None:
    X = pd.DataFrame({"a": _DTYPES["str"]()})  # "b" missing, "a" wrong dtype
    with pytest.raises(LizyMLError) as exc:
        float_model.predict(X)
    assert exc.value.code == ErrorCode.DATA_SCHEMA_INVALID


def test_every_offending_column_is_reported(float_model: Model) -> None:
    X = pd.DataFrame({"b": _B.astype(str), "a": _DTYPES["category"]()})
    with pytest.raises(LizyMLError) as exc:
        float_model.predict(X)
    assert [c["column"] for c in exc.value.context["columns"]] == ["a", "b"]


def test_check_survives_save_and_load(float_model: Model, tmp_path: Path) -> None:
    path = float_model.export(tmp_path / "m")
    loaded = Model.load(path)
    with pytest.raises(LizyMLError) as exc:
        loaded.predict(_frame("str"))
    assert exc.value.code == ErrorCode.INCOMPATIBLE_COLUMNS


class ConvertingPipeline(MinimalPipeline):
    """Converts every column to float at transform: it would predict numeric
    strings correctly, but numeric-at-fit columns must arrive numeric anyway
    (H-0106 decision 2: an input contract, checked before any pipeline)."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.astype(float)


@pytest.mark.parametrize("pipeline", [MinimalPipeline, ConvertingPipeline])
def test_custom_pipeline_gets_the_same_check(
    pipeline: type[MinimalPipeline], monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(self: LGBMProvider, *args: Any, **kwargs: Any) -> Any:
        return pipeline

    monkeypatch.setattr(LGBMProvider, "build_pipeline_factory", factory)
    model = _fit("float64")
    model.predict(_frame("float64"))
    with pytest.raises(LizyMLError) as exc:
        model.predict(_frame("str"))
    assert exc.value.code == ErrorCode.INCOMPATIBLE_COLUMNS


def test_converting_custom_pipeline_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the intended restriction: this pipeline turns the strings back into
    the same floats, so it would have predicted, and the facade refuses anyway."""

    def factory(self: LGBMProvider, *args: Any, **kwargs: Any) -> Any:
        return ConvertingPipeline

    monkeypatch.setattr(LGBMProvider, "build_pipeline_factory", factory)
    model = _fit("float64")
    numeric = _frame("float64")
    as_strings = numeric.astype({"a": str})
    pd.testing.assert_frame_equal(
        ConvertingPipeline().transform(as_strings),
        ConvertingPipeline().transform(numeric),
    )
    with pytest.raises(LizyMLError) as exc:
        model.predict(as_strings)
    assert exc.value.code == ErrorCode.INCOMPATIBLE_COLUMNS


def test_unreadable_recorded_dtype_is_exempt() -> None:
    """A recorded dtype ``pandas_dtype`` cannot read leaves its column
    unchecked: prediction proceeds as before the check existed."""
    model = _fit("float64")
    model._fit_result = replace(
        model.fit_result, dtypes={"a": "no-such-dtype", "b": "float64"}
    )
    model.predict(_frame("float64"))
    # LightGBM's own refusal, not INCOMPATIBLE_COLUMNS.
    with pytest.raises(ValueError, match="pandas dtypes must be int, float or bool"):
        model.predict(_frame("str"))


@pytest.mark.parametrize("dtype", sorted(_FITTABLE))
def test_every_fittable_dtype_records_a_parseable_string(dtype: str) -> None:
    """An unparseable recorded string leaves its column unchecked; this keeps
    that population empty, so a pandas spelling change fails here instead."""
    recorded = _fit(dtype).fit_result.dtypes["a"]
    pd.api.types.pandas_dtype(recorded)
