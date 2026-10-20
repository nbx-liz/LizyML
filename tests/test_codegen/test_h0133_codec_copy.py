"""H-0133 criterion 4 (DC3): the generated scripts' codec copy agrees with
``lizyml.core.category_values`` over the whole value population.

The copy lives in ``codegen/templates.py`` (the generated code does not
depend on LizyML). Both directions are compared: the copy writes the same JSON
the library writes, and reads back the same values; malformed objects fail in
both.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.codegen.templates import _VALUE_CODEC, render_predict_py, render_train_py
from lizyml.core import category_values as library
from lizyml.core.exceptions import LizyMLError
from tests.test_core.test_category_values import PYTHON_TYPE_RETURNS, SAMPLES


def _copy() -> SimpleNamespace:
    namespace: dict[str, Any] = {"np": np, "pd": pd, "json": json}
    exec(compile(_VALUE_CODEC, "<generated codec>", "exec"), namespace)  # noqa: S102
    return SimpleNamespace(**namespace)


COPY = _copy()


def _nested(depth: int) -> Any:
    value: Any = 1
    for _ in range(depth):
        value = (value,)
    return value


def _samples() -> list[tuple[str, Any]]:
    return [
        (f"{t.__name__}[{i}]", v) for t, vs in SAMPLES.items() for i, v in enumerate(vs)
    ]


def test_both_scripts_carry_the_codec_once() -> None:
    for source in (render_train_py(), render_predict_py()):
        assert source.count("def _decode_value(") == 1
        assert source.count("def _encode_value(") == 1
        assert "@@VALUE_CODEC@@" not in source


@pytest.mark.parametrize(("name", "value"), _samples(), ids=[n for n, _ in _samples()])
def test_the_copy_writes_what_the_library_writes(name: str, value: Any) -> None:
    assert json.dumps(COPY._encode_value(value), sort_keys=True) == json.dumps(
        library.encode(value), sort_keys=True
    )


@pytest.mark.parametrize(("name", "value"), _samples(), ids=[n for n, _ in _samples()])
def test_the_copy_reads_what_the_library_reads(name: str, value: Any) -> None:
    written = json.loads(json.dumps(library.encode(value)))
    mine, theirs = COPY._decode_value(written), library.decode(written)
    assert mine == theirs == value
    assert hash(mine) == hash(theirs)
    assert (
        type(mine) is type(theirs) is PYTHON_TYPE_RETURNS.get(type(value), type(value))
    )


@pytest.mark.parametrize(
    "obj",
    [
        {"$lizyml": "bytes"},
        {"$lizyml": "bytes", "b64": "YQ==", "extra": 1},
        {"$lizyml": "unknown", "x": 1},
        {"no_tag": 1},
        {"$lizyml": "datetime", "us": 0, "fold": 2, "tz": None},
        {"$lizyml": "timestamp", "unit": "D", "value": 0, "tz": None},
        {"$lizyml": "complex", "real": True, "imag": 0.0},
        [1, 2],
        None,
        float("nan"),
        {"$lizyml": "decimal", "str": "NaN"},
        {"$lizyml": "complex", "real": float("nan"), "imag": 0.0},
        {"$lizyml": "datetime64", "unit": "s", "count": 1, "value": -(2**63)},
        {"$lizyml": "tuple", "items": [1, float("nan")]},
        {"$lizyml": "timedelta", "days": 0, "seconds": 86400, "microseconds": 0},
        {"$lizyml": "frozenset", "items": [1, 1]},
        {"$lizyml": "frozenset", "items": [2, 1]},
    ],
)
def test_a_malformed_object_fails_in_both(obj: Any) -> None:
    with pytest.raises(ValueError):
        COPY._decode_value(obj)
    with pytest.raises(LizyMLError):
        library.decode(obj)


@pytest.mark.parametrize(
    "value",
    [
        (1, float("nan")),
        (np.datetime64("NaT"),),
        object(),
        _nested(library.MAX_DEPTH + 1),
    ],
    ids=["nested NaN", "nested NaT", "object", "nested 9"],
)
def test_the_copy_refuses_what_the_library_refuses(value: Any) -> None:
    assert library.refusal(value) is not None
    with pytest.raises(ValueError):
        COPY._encode_value(value)


def test_a_pytz_lookalike_is_refused_by_both_without_running_its_code() -> None:
    """A tzinfo whose class claims pytz's module is not one of pytz's zones."""
    import datetime as dt

    calls: list[str] = []

    class FakePytz(dt.tzinfo):
        __module__ = "pytz.tzfile"
        zone = "Asia/Tokyo"

        def utcoffset(self, d: Any) -> dt.timedelta:
            calls.append("utcoffset")
            return dt.timedelta(hours=9)

    value = pd.Timestamp("2024-01-01").tz_localize(FakePytz())
    calls.clear()
    assert library.refusal(value) is not None
    with pytest.raises(ValueError):
        COPY._encode_value(value)
    assert calls == []
